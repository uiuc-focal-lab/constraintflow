"""Schedule expensive row-separable branches of tensor selections on demand.

The predicate and both branch values already exist in the program. This pass
neither manufactures interval bounds nor knows about networks or activations.
"""
import copy
import torch
from constraintflow.compiler.ir import IrAssignment, IrVar, IrTorchWhere, IrTransRetBasic
from constraintflow.compiler.optimizations.row_regions import (
    Shapes, SelectRows, Region, Call, assign, var, names, emit_regions,
)
from constraintflow.compiler.optimizations import cse
from constraintflow.lib.globals import inductor_mode, cudagraphs


def refine_rows(stop, defaults, kernel, args, chunk_size, union=False):
    """Evaluate false branches for the union of needed rows, preserving each mask.

    A tuple of chunk sizes compiles one kernel per size, used as block_plan splits
    the rows: packed on the CPU; on the GPU (where more launches cost more than the
    padded rows: BoxPoly N16 1.20x, N33 1.24x slower packed) one covering block for the rest. union=True gives every example the refined values of a selected row
    (auto_LiRPA's sparse rule).
    """
    from constraintflow.compiler.optimizations.row_refinement import is_warming, block_plan
    indices = torch.nonzero(~stop.all(dim=0), as_tuple=False).flatten()
    n = stop.shape[1]
    if n == 0:
        return defaults
    # several block sizes, capped at the row count
    sizes = tuple(sorted({min(c, n) for c in ((chunk_size,) if isinstance(chunk_size, int) else chunk_size)}))
    if is_warming() and (indices.numel() == 0 or len(sizes) > 1):
        for size in sizes:
            kernel(torch.arange(size, device=stop.device), *args)
    outputs = list(defaults)
    offset = 0
    for size in block_plan(indices.numel(), sizes, pack=stop.device.type == 'cpu'):
        idx = indices[offset:offset + size]
        count = idx.numel()
        first, offset = offset == 0, offset + count
        if count < size:
            idx = torch.cat((idx, idx[-1:].expand(size - count)))
        values = kernel(idx, *args)
        selected = idx[:count]
        #FEAT: by default a selected row is refined for every example (--traversal-stop-per-example keeps each mask).
        mask = stop.index_select(1, selected) if stop.shape[0] > 1 and not union else None
        for i, value in enumerate(values):
            if first:
                outputs[i] = outputs[i].clone()
            outputs[i][:, selected] = (value[:, :count] if mask is None else
                torch.where(mask, defaults[i].index_select(1, selected), value[:, :count]))
    return tuple(outputs)


def refine_rows_union(stop, defaults, kernel, args, chunk_size):
    return refine_rows(stop, defaults, kernel, args, chunk_size, union=True)


def row_separable(expr, rows, probe, n, batch):
    """Conservative structural proof that operations do not mix selected rows."""
    from constraintflow.compiler import ir as I
    if not (names(expr) & rows) or isinstance(expr, IrVar):
        return True
    children = getattr(expr, 'children', ())
    if not all(row_separable(c, rows, probe, n, batch) for c in children):
        return False
    dependent = [i for i,c in enumerate(children) if names(c) & rows]
    def row_axis(child):
        shape = probe.shape(child)
        if shape and len(shape) > 1 and shape[0] in (1,batch) and shape[1] == n:
            return 1
        if shape and shape[0] == batch*n:
            return 0
        return None
    if isinstance(expr, I.IrSimpleBinary):
        op = expr.op if isinstance(expr.op, str) else getattr(expr.op, '__name__', '')
        return op in ('add', 'sub', 'mul', 'truediv', 'floordiv', 'pow', 'mod',
                      'gt', 'ge', 'lt', 'le', 'eq', 'ne', 'and_', 'or_', 'xor')
    if isinstance(expr, (I.IrTensorClamp, I.IrTorchWhere, I.IrTorchFloat)):
        return True
    if isinstance(expr, I.IrSimpleUnary):
        op = expr.op if isinstance(expr.op,str) else getattr(expr.op,'__name__','')
        return op in ('neg','abs','not_')
    if isinstance(expr, I.IrTorchMatmul):
        # A row on the RHS is a contracted/shared dimension, not an output row.
        shape = probe.shape(children[0])
        axis = row_axis(children[0])
        right = probe.shape(children[1])
        if dependent != [0] or axis is None or right is None or len(right) > len(shape):
            return False
        if axis == len(shape) - 2:
            return True
        right_axis = axis - (len(shape) - len(right))
        return axis < len(shape) - 2 and (right_axis < 0 or right[right_axis] == 1)
    if isinstance(expr, (I.IrFConv2d,I.IrFConvTranspose2d,I.IrFUnfold,I.IrFFold)):
        return dependent == [0] and row_axis(children[0]) == 0
    if isinstance(expr, I.IrTorchSum):
        shape = probe.shape(children[0]); axis = row_axis(children[0])
        dims = expr.dim if isinstance(expr.dim,(tuple,list)) else [expr.dim]
        return axis is not None and all(isinstance(d,int) and d % len(shape) > axis for d in dims)
    if isinstance(expr, (I.IrTorchReshape,I.IrTorchView,I.IrTorchExpand)):
        # Only a leading batch/row prefix may be reshaped. No interleaving rows.
        before = probe.shape(children[0]); after = probe.shape(expr)
        a,b = row_axis(children[0]),row_axis(expr)
        if a is None or b is None:return False
        import math
        return math.prod(before[a+1:]) == math.prod(after[b+1:])
    if isinstance(expr,(I.IrTorchSqueeze,I.IrTorchUnsqueeze)):
        before = probe.shape(children[0]); axis = row_axis(children[0])
        rank = len(before) + isinstance(expr,I.IrTorchUnsqueeze)
        return axis is not None and expr.index % rank > axis
    if isinstance(expr,I.IrTorchPermute):
        axis = row_axis(children[0])
        return axis is not None and list(expr.permutation[:axis+1]) == list(range(axis+1))
    if isinstance(expr,I.IrTorchTranspose):
        axis = row_axis(children[0]); rank = len(probe.shape(children[0]))
        return axis is not None and expr.dim0 % rank > axis and expr.dim1 % rank > axis
    if isinstance(expr,I.IrTorchPad):
        axis = row_axis(children[0]); rank = len(probe.shape(children[0]))
        return axis is not None and len(expr.pad)//2 < rank-axis
    if isinstance(expr,I.IrPatchesToDense):
        return row_axis(children[0]) == 1
    # Unknown operations and row-changing slices/repeats stay dense.
    return False


def plan(node, probe):
    if not getattr(node, 'flow_functional', False):
        return None
    original = node.flow_block.children
    if not original or not isinstance(original[-1], IrTransRetBasic):
        return None
    if any(not isinstance(s, IrAssignment) or not isinstance(s.children[0], IrVar) for s in original[:-1]):
        return None
    batch = node.flow_batch_size
    # Materialize rank-two selections so their branch dependency cones are explicit.
    stmts, candidates, table, shared = [], [], cse._Table(), {}
    def materialize(expr):
        if isinstance(expr, (int, float)):
            return expr
        result = copy.copy(expr)
        result.children = [materialize(c) for c in getattr(expr, 'children', ())]
        if isinstance(result, IrTorchWhere):
            shape = probe.shape(expr)
            cond_shape = probe.shape(expr.children[0])
            if shape and len(shape) == 2 and shape[0] == batch and cond_shape == shape:
                representative, _ = table.find_or_add(result)
                if id(representative) in shared:
                    return shared[id(representative)]
                base = f'_cr_value_{len(candidates)}'
                pieces = []
                for suffix, child in zip(('mask', 'coarse', 'dense'), result.children):
                    name = base + '_' + suffix
                    stmts.append(assign(name, child))
                    probe.env[name] = probe.value(child)
                    probe.shapes[name] = probe.shape(child)
                    pieces.append(var(name))
                stmts.append(assign(base, IrTorchWhere(*pieces)))
                probe.env[base] = probe.value(expr)
                probe.shapes[base] = shape
                candidates.append((base, pieces, shape))
                shared[id(representative)] = var(base)
                return var(base)
        return result
    for stmt in original:
        if isinstance(stmt, IrAssignment):
            stmts.append(assign(stmt.children[0].name, materialize(stmt.children[1])))
        else:
            stmts.append(IrTransRetBasic([materialize(x) for x in stmt.children]))
    defs = {s.children[0].name: i for i, s in enumerate(stmts[:-1])}
    uses = {n: names(stmts[i].children[1]) & defs.keys() for n, i in defs.items()}
    emitted, regions, calls = set(), [], []
    def closure(outputs, excluded=()):
        needed, todo = set(), list(outputs)
        boundary = emitted | set(excluded)
        while todo:
            n = todo.pop()
            if n in defs and n not in needed and n not in boundary:
                needed.add(n)
                todo.extend(uses[n])
        return needed
    def region(name, body, outputs, selected=False):
        assigned = {s.children[0].name for s in body}
        read = set(outputs).union(*(names(s.children[1]) for s in body))
        args = sorted((read - assigned - {'_idx', '_n'}) | {'batch_size'})
        reg = Region(name, body, list(outputs), (['_idx'] if selected else []) + args, selected)
        regions.append(reg)
        return reg
    # Lower and upper projections of the same predicate share one dispatch.
    groups, masks = [], cse._Table()
    for base, pieces, shape in candidates:
        predicate = stmts[defs[pieces[0].name]].children[1]
        representative, _ = masks.find_or_add(predicate)
        if groups and groups[-1][0] is representative:
            groups[-1][1].append((base, pieces, shape))
        else:
            groups.append((representative, [(base, pieces, shape)]))
    selected_count = 0
    for _, group in groups:
        outputs = [g[0] for g in group]
        masks_names = [g[1][0].name for g in group]
        coarse = [g[1][1].name for g in group]
        dense = [g[1][2].name for g in group]
        n = group[0][2][1]
        needed = closure(outputs)
        required = closure(masks_names + coarse)
        phase = [stmts[defs[v]] for v in sorted(needed, key=defs.get)]
        rows = set()
        for s in phase:
            name = s.children[0].name
            if name in required or name in outputs:
                continue
            shape = probe.shapes.get(name)
            if shape and len(shape) >= 3 and shape[0] in (1, batch) and shape[1] == n and shape[2] > 1:
                rows.add(name)
            elif shape and len(shape) >= 2 and shape[0] in (1, batch) and shape[1] == n and names(s.children[1]) & rows:
                rows.add(name)
        if not rows or not set(dense) <= rows:
            continue
        future = names(stmts[-1]).union(*(v for name, v in uses.items() if name not in needed and name not in emitted))
        if rows & future:
            continue
        pre = [s for s in phase if s.children[0].name not in rows and s.children[0].name not in outputs]
        if any(names(s.children[1]) & rows for s in pre):
            continue
        if any(not row_separable(s.children[1], rows, probe, n, batch)
               for s in phase if s.children[0].name in rows):
            continue
        selected = []
        for s in phase:
            if s.children[0].name not in rows:
                continue
            t = SelectRows(probe, n, rows, batch, f'_cr_hoist_{selected_count}_{len(selected)}')
            expr = s.children[1]
            expr = t.visit(expr) if t.is_row(expr) else t.operand(expr, len(probe.shapes[s.children[0].name]))
            selected.append(assign(s.children[0].name, expr))
            pre += t.hoisted
        selected_region = region(f'_cr_rows_{selected_count}', selected, dense, True)
        exported = ((needed & future) - rows - set(outputs)) | set(masks_names + coarse)
        exported |= set(selected_region.params) & {s.children[0].name for s in pre}
        pre_region = region(f'_cr_pre_{selected_count}', pre, sorted(exported))
        calls.append(Call(pre_region.outputs, pre_region.name, pre_region.params))
        # Tuples are call syntax; their members must also participate in liveness.
        defaults = '(' + ', '.join(coarse) + ',)'
        calls.append(Call(outputs, 'refine_rows', [masks_names[0], defaults],
                          (selected_region.name, selected_region.params[1:]), extra_reads=tuple(coarse)))
        emitted |= needed
        selected_count += 1
    if not selected_count:
        return None
    rest = [s for s in stmts[:-1] if s.children[0].name not in emitted]
    outputs = [f'_cr_result_{i}' for i in range(len(stmts[-1].children))]
    rest += [assign(n, e) for n, e in zip(outputs, stmts[-1].children)]
    final = region('_cr_final', rest, outputs)
    calls.append(Call([], final.name, final.params))
    print(f'[conditional-rows] {selected_count} row selections from program predicates')
    return regions, calls


def emit(cg, node):
    if (not getattr(node, 'flow_functional', False)
            or not getattr(node, 'flow_invariant_stops', False) or cudagraphs.get_flag()):
        return False
    if not any(isinstance(e, IrTorchWhere) for s in node.flow_block.children for e in walk(s)):
        return False
    probe = Shapes(cg, node)
    planned = plan(node, probe)
    if planned is None:
        return False
    from constraintflow.lib import globals as runtime_globals
    chunk_size, union = runtime_globals.traversal_stop_config
    if union:
        for call in planned[1]:
            if call.function == 'refine_rows':
                call.function = 'refine_rows_union'
    emit_regions(cg, node, *planned, chunk_size, inductor_mode.get_flag(),
        'from constraintflow.compiler.optimizations.conditional_rows import refine_rows, refine_rows_union\n'
        'from constraintflow.compiler.optimizations.row_refinement import selected_patches_to_dense',
        release_dense=True)
    return True


def walk(expr):
    yield expr
    for child in getattr(expr, 'children', ()):
        yield from walk(child)
