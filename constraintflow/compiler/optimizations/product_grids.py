"""Evaluate a grid of convolutions or broadcast mat-vecs as one operation.

When m same-shape inputs and n same-shape weights appear in all m*n
combinations, as interval propagation's W+/W- times l/u does, compute the
products once: conv(cat(inputs), cat(weights)) with inputs stacked on the batch
and weights on output channels, or cat(vectors) @ cat(matrices).T for a
batch-broadcast matrix times per-sample column vectors. Every original product
is a slice of the result with the same per-element reduction, so one larger
cuDNN/cuBLAS call replaces several small ones. Run on functional SSA before
early reductions.
"""
import copy
import operator
#FEAT: transpose_grids meta device_mode
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from constraintflow.compiler.ir import (
    #FEAT: transpose_grids
    IrAssignment, IrFConv2d, IrFConvTranspose2d, IrTorchCat, IrTorchMatmul, IrTorchSlice, IrTorchSqueeze,
    IrTorchTranspose, IrTorchUnsqueeze, IrTransRetBasic, IrVar,
)
from constraintflow.compiler.optimizations import subexp_inlining
#FEAT: per-change toggles for the DeepPoly-efficient passes
from constraintflow.gbcsr.sparse_block import RUNTIME_HELPERS

MAX_GRID_BYTES = 512 * 1024 ** 2
MAX_GRID_BATCH = 32768


def _meta_env(sized, batch_size):
    env = {'torch': torch, 'F': F, 'operator': operator, 'batch_size': batch_size, **RUNTIME_HELPERS}
    # tensors created inside probed expressions land on the meta device
    env['device_mode'] = SimpleNamespace(get_device=lambda: 'meta')
    records = list(sized.inputs.items()) + [(r['name'], r) for r in sized.values()]
    for name, r in records:
        if name and 'shape' in r:
            dtype = getattr(torch, r['dtype'].split('.')[-1])
            env[name] = torch.empty(r['shape'], dtype=dtype, device='meta')
    return env


def run(block, render, sized, batch_size):
    stmts = block.children
    definitions, occupied = {}, set()
    for i, stmt in enumerate(stmts):
        if isinstance(stmt, IrAssignment) and isinstance(stmt.children[0], IrVar):
            definitions[stmt.children[0].name] = i
        elif not isinstance(stmt, IrTransRetBasic):
            return 0
        occupied.update(v.name for v in subexp_inlining.get_vars_expr_occurrences(stmt.children))
    # conv_transpose2d is grouped too; operand shapes are evaluated on the meta device
    env = _meta_env(sized, batch_size)
    kinds = (IrFConv2d, IrFConvTranspose2d, IrTorchMatmul)
    nested = ('conv2d(', 'conv_transpose2d(', 'matmul(')

    def meta(text):
        try:
            with torch.device('meta'):
                value = eval(text, dict(env))
        except Exception:
            return None
        return value if isinstance(value, torch.Tensor) else None

    convs = []

    def walk(expr, i):
        if isinstance(expr, list):
            for e in expr:
                walk(e, i)
            return
        if not hasattr(expr, 'children'):
            return
        #FEAT: transpose_grids
        if isinstance(expr, kinds) and len(expr.children) == 2:
            conv = isinstance(expr, (IrFConv2d, IrFConvTranspose2d))
            w, x = (render(c) for c in expr.children)
            if conv:
                x, w = w, x
            #FEAT: transpose_grids
            if len(expr.children) == 2 and not any(op in x + w for op in nested):
                opts = (type(expr), render(expr.stride), render(expr.padding),
                        render(expr.output_padding) if isinstance(expr, IrFConvTranspose2d) else None) if conv else 'mm'
                convs.append((i, expr, x, w, opts))
                return
        for c in expr.children:
            walk(c, i)

    for i, stmt in enumerate(stmts):
        walk(stmt.children[1] if isinstance(stmt, IrAssignment) else stmt.children, i)

    # Union products that share an input or a weight under equal options.
    parent = list(range(len(convs)))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    seen = {}
    for k, (_, _, x, w, opts) in enumerate(convs):
        for key in (('x', x, opts), ('w', w, opts)):
            if key in seen:
                parent[find(k)] = find(seen[key])
            else:
                seen[key] = k
    groups = {}
    for k in range(len(convs)):
        groups.setdefault(find(k), []).append(k)

    #FEAT: transpose_grids (only transpose groups are rebatched)
    # A transpose kernel can recur throughout a backward traversal. Form
    # groups whose inputs already exist at the first use, rather than letting
    # a later dependent input prevent batching every earlier independent pair.
    batches = []
    for members in groups.values():
        #FEAT: transpose_grids
        if not isinstance(convs[members[0]][1], IrFConvTranspose2d):
            batches.append(members)
            continue
        remaining = list(members)
        while remaining:
            position = min(convs[k][0] for k in remaining)
            ready = []
            for k in remaining:
                deps = subexp_inlining.get_vars_expr_occurrences(convs[k][1].children)
                if all(definitions.get(v.name, -1) < position for v in deps):
                    ready.append(k)
            if not ready:
                ready = [remaining[0]]
            batches.extend(ready[i:i+2] for i in range(0, len(ready), 2))
            used = set(ready)
            remaining = [k for k in remaining if k not in used]

    replace, inserts, counter, merged = {}, {}, 0, 0
    for members in batches:
        xs = list(dict.fromkeys(convs[k][2] for k in members))
        ws = list(dict.fromkeys(convs[k][3] for k in members))
        pairs = {(convs[k][2], convs[k][3]) for k in members}
        if len(xs) * len(ws) < 2 or len(pairs) != len(xs) * len(ws):
            continue
        first = convs[members[0]][1]
        #FEAT: transpose_grids
        conv = isinstance(first, (IrFConv2d, IrFConvTranspose2d))
        transpose = isinstance(first, IrFConvTranspose2d)
        xm = [meta(x) for x in xs]
        wm = [meta(w) for w in ws]
        rank = 4 if conv else 3
        #FEAT: pack only tensors in the run dtype, so --dtype float64 keeps this pass.
        if any(t is None or t.dim() != rank or t.dtype != torch.float for t in xm + wm):
            continue
        if any(t.shape != xm[0].shape for t in xm) or any(t.shape != wm[0].shape for t in wm):
            continue
        if not conv and (xm[0].shape[-1] != 1 or wm[0].shape[0] != xm[0].shape[0]
                         or any(t.shape[0] > 1 and t.stride(0) != 0 for t in wm)):
            continue
        out = meta(render(first))
        if out is None:
            continue
        b, c = xm[0].shape[0], wm[0].shape[1 if transpose or not conv else 0]
        #FIX FOR: grid limit ignored packed inputs; N35/N36 tried a 16 GiB GEMM operand
        # Packing is real storage even when the grid result is tiny. In
        # particular, a GEMV grid can concatenate multi-GiB coefficient
        # matrices into a huge GEMM operand while producing only a few KiB.
        # Count the actual packed matrix (one broadcast batch), not its
        # logically expanded batches. Include singleton operands
        # conservatively: pointwise producers may need materialization too.
        packed_x = sum(t.numel() * t.element_size() for t in xm)
        packed_w = sum((t.numel() if conv else t[0].numel()) * t.element_size()
                       for t in wm)
        result_bytes = out.numel() * len(xs) * len(ws) * out.element_size()
        if (len(xs) * b > MAX_GRID_BATCH * (4 if transpose else 1)
                or packed_x + packed_w + result_bytes > MAX_GRID_BYTES):
            continue
        position = min(convs[k][0] for k in members)
        exprs = {}
        for k in members:
            _, node, x, w, _ = convs[k]
            exprs.setdefault(x, node.children[0 if conv else 1])
            exprs.setdefault(w, node.children[1 if conv else 0])
        deps = subexp_inlining.get_vars_expr_occurrences([exprs[t] for t in xs + ws])
        if any(definitions.get(v.name, -1) >= position for v in deps):
            continue
        while 'grid_' + str(counter) in occupied:
            counter += 1
        name = 'grid_' + str(counter)
        counter += 1
        occupied.add(name)
        #FEAT: transpose_grids
        if transpose:
            grid = IrFConvTranspose2d(IrTorchCat([copy.copy(exprs[x]) for x in xs], 0),
                                      IrTorchCat([copy.copy(exprs[w]) for w in ws], 1),
                                      first.stride, first.padding, first.output_padding)
        elif conv:
            grid = IrFConv2d(IrTorchCat([copy.copy(exprs[x]) for x in xs], 0),
                             IrTorchCat([copy.copy(exprs[w]) for w in ws], 0),
                             first.stride, first.padding)
        else:
            grid = IrTorchMatmul(
                IrTorchCat([IrTorchSqueeze(copy.copy(exprs[x]), -1) for x in xs], 0),
                IrTorchTranspose(IrTorchCat([IrTorchSlice(copy.copy(exprs[w]), ['0']) for w in ws], 0), 0, 1))
        grid.irMetadata = first.irMetadata
        inserts.setdefault(position, []).append(IrAssignment(IrVar(name, first.irMetadata), grid))
        for k in members:
            _, node, x, w, _ = convs[k]
            i, j = xs.index(x), ws.index(w)
            pick = IrTorchSlice(IrVar(name, node.irMetadata),
                                [f'{i * b}:{(i + 1) * b}', f'{j * c}:{(j + 1) * c}'])
            if not conv:
                pick.irMetadata = node.irMetadata
                pick = IrTorchUnsqueeze(pick, -1)
            pick.irMetadata = node.irMetadata
            replace[id(node)] = pick
        merged += len(members)
    if not replace:
        return 0

    def rewrite(expr):
        if isinstance(expr, list):
            return [rewrite(e) for e in expr]
        if id(expr) in replace:
            return copy.copy(replace[id(expr)])
        if not hasattr(expr, 'children') or not expr.children:
            return expr
        children = [rewrite(c) for c in expr.children]
        if all(a is b for a, b in zip(children, expr.children)):
            return expr
        out = copy.copy(expr)
        out.update_parent_child(children)
        return out

    out = []
    for i, stmt in enumerate(stmts):
        out.extend(inserts.get(i, ()))
        if isinstance(stmt, IrAssignment):
            stmt.update_parent_child([stmt.children[0], rewrite(stmt.children[1])])
        else:
            stmt.update_parent_child([rewrite(c) for c in stmt.children])
        out.append(stmt)
    block.update_parent_child(out)
    return merged
