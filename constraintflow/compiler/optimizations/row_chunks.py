#FEAT: row-chunked back-substitution for networks whose traversals do not fit in GPU memory (e.g. VGG16).
"""Run each over-budget back-substitution of an eager (no --inductor) flow() as a loop over row chunks.

A traversal bounds the N neurons of one layer. Its coefficient tensors carry those neurons on axis 1
([batch, N, columns]; [batch*N, C, P, P] inside conv_transpose2d), and every neuron's row is computed
independently and ends in a per-neuron sum. So the same statements run on rows [r0, r1) give rows
[r0, r1) of every result: the bounds are unchanged, only the peak memory drops.

The pass works on the emitted source (batch 1):
  1. fake-executes flow() for every variable's shape and every statement's allocation;
  2. from each statement over the budget, collects its traversal: statements whose 3-D+ results
     carry the same N rows on axis 1, connected through each other, plus the row-indexed statements
     that consume them (the per-neuron sums);
  3. cheap traversal tensors that use no other traversal tensor stay outside the loop and are sliced
     per chunk; the rest run in the loop with the row count in their reshapes replaced by the chunk
     size and row-aligned operands built from other layers (per-position patches tiled with .repeat,
     weights expanded over positions) taken row by row;
  4. row-indexed results used after the loop are concatenated; statements are re-ordered so each
     loop follows its inputs and precedes its consumers; `del` frees every variable after its last use.

    python -m constraintflow.compiler.optimizations.row_chunks <program dir> --network net.onnx \
        --dataset box-3x224x224-1000-1 --budget-gb 20
"""
import argparse
import ast
import copy
import heapq
import itertools
import math
import os
import sys

import torch

# Hoisted names must be unique across every loop of one rewrite; a counter keeps them deterministic.
_NAME_IDS = itertools.count()

ROWS_HELPER = '''
from constraintflow.gbcsr.sparse_block import selection_gather_map as _cf_selection_map, realized_pad as _cf_pad
from constraintflow.lib.pre_grad_cse import PRE_GRAD_CSE  # loop bodies use it; normal builds don't import it
import os as _cf_os
import time as _cf_time

_CF_PROGRESS = _cf_os.environ.get('CF_ROW_CHUNKS_PROGRESS', '0') not in ('', '0')
_cf_last = [None]


def _cf_progress(k, n_rows):
    """CF_ROW_CHUNKS_PROGRESS=1: time since the previous chunked traversal finished."""
    if _CF_PROGRESS:
        torch.cuda.synchronize()
        now = _cf_time.perf_counter()
        print(f'[row-chunks] loop {k} ({n_rows} rows) done, +{now - (_cf_last[0] or now):.1f} s', flush=True)
        _cf_last[0] = now


def _cf_chunks(n_rows, chunk):
    return range(0, n_rows, chunk)


def _cf_cat(parts, n_rows):
    return torch.cat(parts, dim=1)


def _cf_rows(x, r0, r1):
    """Rows [r0, r1) of axis 1 of a row-aligned tensor whose rows repeat with period x.shape[1]."""
    return x[:, r0:r1]


def _cf_select(x, index, scale):
    """conv_transpose2d with a channel-selection weight (disjoint taps): one gather and a scale."""
    return x.reshape(x.shape[0], -1)[:, index].view(x.shape[0], *scale.shape) * scale


def _cf_channels(x, r0, r1, positions):
    """Channels [r0 // positions, r1 // positions) of axis 1: the channel factor of rows split as channels x positions."""
    return x[:, r0 // positions:r1 // positions]


def _cf_tiled_rows(x, times, r0, r1):
    """Rows [r0, r1) of x.repeat([1, times, 1, ...]) without materializing the repeat."""
    period = x.shape[1]
    if r0 % period == 0 and (r1 - r0) % period == 0:
        return x.repeat([1, (r1 - r0) // period] + [1] * (x.dim() - 2))
    idx = torch.arange(r0, r1, device=x.device) % period
    return x.index_select(1, idx)
'''


class _Names(ast.NodeVisitor):
    def __init__(self):
        self.names = set()

    def visit_Name(self, node):
        self.names.add(node.id)


def _names(node):
    v = _Names()
    v.visit(node)
    return v.names


def _shape_of(value):
    return tuple(value.shape) if isinstance(value, torch.Tensor) else None


class _Probe:
    """Fake execution of flow(): shapes of variables and sub-expressions, bytes per statement."""

    def __init__(self, module_globals, params, inputs):
        from torch._subclasses.fake_tensor import FakeTensorMode
        from constraintflow.compiler.optimizations.flow_shapes import _Recorder
        self.mode = FakeTensorMode(allow_non_fake_inputs=True)
        self.recorder_cls = _Recorder
        self.env = dict(module_globals)
        with self.mode:
            for name, value in zip(params, inputs):
                self.env[name] = (torch.empty(value.shape, dtype=value.dtype, device=value.device)
                                  if isinstance(value, torch.Tensor) else value)
        self.shapes = {n: _shape_of(v) for n, v in self.env.items() if n in params}
        self.stmt_bytes = {}

    def run(self, stmts):
        for i, stmt in enumerate(stmts):
            if not isinstance(stmt, ast.Assign):
                continue
            rec = self.recorder_cls()
            rec.known_roots = set()
            with self.mode, rec:
                exec(compile(ast.Module([stmt], []), '<flow>', 'exec'), self.env)
            name = stmt.targets[0].id
            self.shapes[name] = _shape_of(self.env[name])
            self.stmt_bytes[i] = max(sum(rec.bytes), rec.temporary_peak)

    def shape(self, node):
        with self.mode:
            #FIX FOR: nodes built by the rewrite (e.g. _cf_select) have no line numbers, so compile() refused them.
            return _shape_of(eval(compile(ast.fix_missing_locations(ast.Expression(node)), '<expr>', 'eval'), self.env))


def _per_row(shape, n_rows):
    """[1, N] or [1, N, 1, ...]: one value per neuron."""
    return shape is not None and len(shape) >= 2 and shape[:2] == (1, n_rows) and math.prod(shape[2:]) == 1


def _row_axis_size(shape, ndim, n_rows):
    """Does an operand of `shape`, broadcast against a [1, N, ...] result of rank ndim, carry the rows?"""
    if shape is None:
        return False
    offset = ndim - len(shape)
    return 1 - offset >= 0 and shape[1 - offset] == n_rows


def _rewrite_row_literals(node, n_rows):
    """In reshape/view/expand shape lists, the leading row entry (prefix product 1) N -> _n."""
    lst = node.args[0] if len(node.args) == 1 and isinstance(node.args[0], (ast.List, ast.Tuple)) else None
    elts = lst.elts if lst is not None else node.args
    prefix = 1
    for k, e in enumerate(elts):
        if not isinstance(e, ast.Constant) or not isinstance(e.value, int):
            return False
        if prefix == 1 and e.value == n_rows:
            elts[k] = ast.Name('_n', ast.Load())
            return True
        prefix *= max(e.value, 1)
        if prefix > 1:
            return False
    return False


ELEMENTWISE = {('operator', f) for f in ('add', 'sub', 'mul', 'truediv', 'neg', 'gt', 'lt', 'ge', 'le', 'eq', 'ne')} \
    | {('torch', f) for f in ('where', 'clamp', 'maximum', 'minimum', 'abs')} | {(None, 'safe_div')}


def _elementwise(node):
    if not isinstance(node, ast.Call):
        return False
    f = node.func
    if isinstance(f, ast.Name):
        return (None, f.id) in ELEMENTWISE
    return isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name) and (f.value.id, f.attr) in ELEMENTWISE


#FEAT: row-locality checks: a loop statement may only use operations that keep each row to itself.
UNARY_METHODS = {'neg', 'abs', 'float', 'contiguous', 'clone', 'clamp'}
# Products and convolutions whose first operand may carry the rows (on an axis they do not contract).
ROW_FIRST = {'conv_transpose2d', 'training_bcast_matmul', 'bcast_matmul', 'matmul', '_cf_select'}


class _Replace(ast.NodeTransformer):
    """Replace subexpressions (by ast.dump) with names."""

    def __init__(self, names):
        self.names = names

    def visit(self, node):
        name = self.names.get(ast.dump(node)) if isinstance(node, ast.expr) else None
        return ast.Name(name, ast.Load()) if name else super().visit(node)


class _ChunkBody(ast.NodeTransformer):
    """Rewrite one loop statement so it computes rows [_r0, _r1)."""

    def __init__(self, probe, n_rows, row_vars, sliced_vars):
        self.probe, self.n_rows = probe, n_rows
        self.row_vars, self.sliced_vars = row_vars, sliced_vars
        self.hoisted = []          # (name, expression) computed once before the loop
        self.periods = []          # rows of each tiled source (positions per channel)
        self.whole_periods = []    # the chunk size must be a multiple of these
        self.failed = None
        self.mixes_rows = False    # a product contracts the rows: the statement must run unchunked
        self.channel_vars = {}     # name -> positions: [1, C, 1 or P, ...] factors sliced by channels

    def _is_row(self, node):
        return bool(_names(node) & (self.row_vars | self.channel_vars.keys()))

    def _row_operand(self, node, result_ndim):
        """Replace a column-side operand that is nevertheless row-aligned by its chunk rows."""
        shape = self.probe.shape(node)
        if not _row_axis_size(shape, result_ndim, self.n_rows):
            return node
        if len(shape) < 2 or shape[0] not in (1, getattr(self, 'batch_size', 1)) or shape[1] != self.n_rows:
            self.failed = f'unsupported batch/row layout for operand of shape {shape}'
            return node
        if _elementwise(node):
            # Broadcasting elementwise op: slice its row-aligned operands, keep the rest.
            node.args = [a if isinstance(a, (ast.Constant, ast.List, ast.Tuple)) else self._row_operand(a, result_ndim)
                         for a in node.args]
            return node
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == 'repeat'):
            reps = node.args[0].elts if len(node.args) == 1 and isinstance(node.args[0], ast.List) else node.args
            if all(isinstance(r, ast.Constant) for r in reps) and [r.value for r in reps][0] == 1 \
                    and all(r.value == 1 for r in reps[2:]):
                self.periods.append(self.probe.shape(node.func.value)[1])
                return ast.Call(ast.Name('_cf_tiled_rows', ast.Load()),
                                [node.func.value, reps[1], ast.Name('_r0', ast.Load()), ast.Name('_r1', ast.Load())], [])
        chunked = self._channel_rows(node)
        if chunked is not None:
            return chunked
        name = f'_cf_hoist_{len(self.hoisted)}_{next(_NAME_IDS)}'
        self.hoisted.append((name, node))
        return ast.Call(ast.Name('_cf_rows', ast.Load()),
                        [ast.Name(name, ast.Load()), ast.Name('_r0', ast.Load()), ast.Name('_r1', ast.Load())], [])

    #FEAT: a row-aligned operand built from other layers whose rows are channels x positions (SkipPoly's first
    # step: weight[c] times the patch at position p) runs per chunk with its [1, C, 1 or P, ...] channel factor
    # sliced, instead of in full before the loop (VGG16: up to 87 GiB). The row-locality checks still apply.
    def _channel_rows(self, node):
        leaves = []

        def find(n):
            """Innermost [1, C, 1 or P, ...] subexpressions (the small factors, not the products)."""
            found = len(leaves)
            for c in ast.iter_child_nodes(n):
                find(c)
            if len(leaves) > found or isinstance(n, (ast.Constant, ast.List, ast.Tuple)) or not isinstance(n, ast.expr):
                return
            try:
                shape = self.probe.shape(n)
            except Exception:
                shape = None
            if shape is not None and len(shape) >= 3 and shape[0] == 1 and 1 < shape[1] < self.n_rows \
                    and self.n_rows % shape[1] == 0 and shape[2] in (1, self.n_rows // shape[1]) and _names(n):
                leaves.append((n, shape))

        find(node)
        # Only when the factors are much smaller than the operand (else hoisting it costs the same memory).
        if not leaves or len({sh[1] for _, sh in leaves}) != 1 \
                or 4 * sum(math.prod(sh) for _, sh in leaves) > math.prod(self.probe.shape(node)):
            return None
        positions = self.n_rows // leaves[0][1][1]
        state = (self.failed, len(self.hoisted), len(self.periods), len(self.whole_periods), dict(self.channel_vars))
        new, names = copy.deepcopy(node), {}
        for n, _ in leaves:
            names[ast.dump(n)] = names.get(ast.dump(n)) or f'_cf_ch_{len(self.hoisted)}_{next(_NAME_IDS)}'
            name = names[ast.dump(n)]
            if name not in self.channel_vars:
                self.hoisted.append((name, n))
                self.channel_vars[name] = positions
                with self.probe.mode:
                    self.probe.env[name] = eval(compile(ast.fix_missing_locations(ast.Expression(n)), '<expr>', 'eval'),
                                                self.probe.env)
        new = _Replace({ast.dump(n): names[ast.dump(n)] for n, _ in leaves}).visit(new)
        self.failed = None
        new = self.visit(new)
        if self.failed is not None or self._row_axis(node)[0] is None:
            self.failed = state[0]
            del self.hoisted[state[1]:], self.periods[state[2]:], self.whole_periods[state[3]:]
            self.channel_vars = state[4]
            return None
        self.failed = state[0]
        self.periods.append(positions)
        self.whole_periods.append(positions)
        return new

    def _row_span(self, node):
        """(first row axis, number of row axes, rank): rows on one axis of size N, or split as channels x
        positions over two adjacent axes; the first axis is None when the first non-unit axis is neither."""
        try:
            shape = self.probe.shape(node)
        except Exception:
            return None, 0, 0
        shape = shape or ()
        d = next((k for k, z in enumerate(shape) if z != 1), None)
        if d is not None and shape[d] == self.n_rows:
            return d, 1, len(shape)
        #FEAT: rows split into channels x positions (DPE reshapes [1, N, K] to [1, C, P, K]).
        if d is not None and d + 1 < len(shape) and shape[d + 1] > 1 and shape[d] * shape[d + 1] == self.n_rows:
            return d, 2, len(shape)
        return None, 0, len(shape)

    def _row_axis(self, node):
        """(last axis carrying the rows, rank) of a row tensor; None unless its first non-unit axes hold N."""
        d, span, ndim = self._row_span(node)
        return (None if d is None else d + span - 1), ndim

    #FEAT: a reshape that splits the N rows into C channels x P positions keeps whole channels together when
    # every chunk is a multiple of P, so it becomes the same reshape with _n // P channels.
    def _split_rows(self, node):
        lst = node.args[0] if len(node.args) == 1 and isinstance(node.args[0], (ast.List, ast.Tuple)) else None
        elts = lst.elts if lst is not None else node.args
        if not all(isinstance(e, ast.Constant) and isinstance(e.value, int) for e in elts):
            return False
        sizes = [e.value for e in elts]
        k = next((i for i, z in enumerate(sizes) if z != 1), None)
        if k is None or k + 1 >= len(sizes) or sizes[k + 1] <= 1 or sizes[k] * sizes[k + 1] != self.n_rows:
            return False
        positions = sizes[k + 1]
        elts[k] = ast.BinOp(ast.Name('_n', ast.Load()), ast.FloorDiv(), ast.Constant(positions))
        self.periods.append(positions)
        self.whole_periods.append(positions)
        return True

    #FEAT: a row's patches_to_dense gather depends only on its position, so a chunk of whole channels is the same call with fewer channels.
    def _patches_rows(self, node):
        try:
            batch, channels, ox, oy = [ast.literal_eval(a) for a in node.args[1:5]]
        except ValueError:
            self.failed = self.failed or 'patches_to_dense with non-constant geometry'
            return node
        positions = ox * oy
        if batch != 1 or channels * positions != self.n_rows or self._row_axis(node.args[0])[0] is None \
                or any(self._is_row(a) for a in node.args[1:]):
            self.failed = self.failed or f'patches_to_dense that does not have the {self.n_rows} rows as channels x positions'
            return node
        node.args[0] = self.visit(node.args[0])
        node.args[2] = ast.BinOp(ast.Name('_n', ast.Load()), ast.FloorDiv(), ast.Constant(positions))
        self.periods.append(positions)
        self.whole_periods.append(positions)
        return node

    def _method_error(self, node):
        """Why a method call on a row tensor may mix rows, or None if it keeps them apart."""
        if self.n_rows == 1:   # one row: a single chunk, nothing to mix
            return None
        f = node.func
        d, ndim = self._row_axis(f.value)
        if d is None:
            return f'{f.attr} on a tensor whose first non-unit axis is not the {self.n_rows} rows'
        try:
            args = [ast.literal_eval(a) for a in node.args]
            kw = {k.arg: ast.literal_eval(k.value) for k in node.keywords}
        except ValueError:
            return f'{f.attr} with non-constant arguments on a row tensor'
        if f.attr == 'sum':
            dims = kw.get('dim', args[0] if args else None)
            dims = [dims] if isinstance(dims, int) else dims
            return None if dims and all(k % ndim > d for k in dims) else 'sum over the row axis'
        if f.attr in ('squeeze', 'unsqueeze'):
            k = kw.get('dim', args[0] if args else None)
            n = ndim + (f.attr == 'unsqueeze')
            return None if isinstance(k, int) and k % n > d else f'{f.attr} at or before the row axis'
        return None if f.attr in UNARY_METHODS else f'{f.attr} on a row tensor'

    def _contract_error(self, node):
        """Why a non-elementwise operation on row tensors may mix rows, or None."""
        if self.n_rows == 1:
            return None
        if isinstance(node, ast.BinOp):
            name, operands = ('matmul' if isinstance(node.op, ast.MatMult) else None), [node.left, node.right]
        else:
            f = node.func
            name = f.id if isinstance(f, ast.Name) else f.attr if isinstance(f, ast.Attribute) else None
            operands = node.args + [k.value for k in node.keywords]
        #FEAT: pad on dimensions after the row axis keeps rows apart (residual merges pad patches, e.g. ResNets).
        if name == 'pad' and len(operands) > 1 and self._is_row(operands[0]):
            d, ndim = self._row_axis(operands[0])
            try:
                widths = ast.literal_eval(operands[1])
            except ValueError:
                return 'pad with non-constant widths on a row tensor'
            return None if d is not None and len(widths) // 2 <= ndim - d - 1 else 'pad at or before the row axis'
        if name not in ROW_FIRST:
            return f'{name or ast.unparse(node)[:40]} on a row tensor'
        if not operands or not self._is_row(operands[0]) or any(self._is_row(o) for o in operands[1:]):
            self.mixes_rows = True
            return f'{name} with the rows outside its first operand'
        d, ndim = self._row_axis(operands[0])
        if d is None or d >= ndim - 1:
            self.mixes_rows = True
            return f'{name} contracting the row axis'
        return None

    def visit_Name(self, node):
        if node.id in self.channel_vars:
            return ast.Call(ast.Name('_cf_channels', ast.Load()),
                            [ast.Name(node.id, ast.Load()), ast.Name('_r0', ast.Load()), ast.Name('_r1', ast.Load()),
                             ast.Constant(self.channel_vars[node.id])], [])
        if node.id in self.sliced_vars:
            return ast.Call(ast.Name('_cf_rows', ast.Load()),
                            [ast.Name(node.id, ast.Load()), ast.Name('_r0', ast.Load()), ast.Name('_r1', ast.Load())], [])
        return node

    def generic_visit_row(self, node):
        """A node that touches row variables: fix its column-side operands, then recurse."""
        #FIX FOR: a matmul's column-side operand of size N (fc 4096 -> 4096) was sliced like a row operand.
        # Only a broadcasting elementwise op aligns a column-side operand with the rows.
        elementwise = isinstance(node, ast.BinOp) and not isinstance(node.op, ast.MatMult) or _elementwise(node)
        if not elementwise:
            self.failed = self.failed or self._contract_error(node)
        try:
            ndim = len(self.probe.shape(node)) if elementwise else None
        except Exception:
            ndim = None
        #FEAT: with the rows split as channels x positions, a column-side operand may vary over positions (every
        # chunk holds whole channels) but not over channels, which would need slicing per chunk.
        d, span, _ = self._row_span(node) if elementwise else (None, 0, 0)
        if span == 2:
            for item in [v for _, v in ast.iter_fields(node) for v in (v if isinstance(v, list) else [v])]:
                if isinstance(item, ast.expr) and not isinstance(item, (ast.Constant, ast.List, ast.Tuple)) \
                        and not self._is_row(item):
                    try:
                        shape = self.probe.shape(item) or ()
                    except Exception:
                        shape = None
                    c = None if shape is None else d - (ndim - len(shape))
                    if shape is None or c >= 0 and shape[c] != 1:
                        self.failed = self.failed or 'operand aligned with the channels of split rows'
                #FEAT: a sliced channel factor must sit on the channel axis of the split rows.
                elif isinstance(item, ast.expr) and _names(item) & self.channel_vars.keys():
                    try:
                        shape = self.probe.shape(item) or ()
                    except Exception:
                        shape = ()
                    c, positions = d - (ndim - len(shape)), self.channel_vars[min(_names(item) & self.channel_vars.keys())]
                    if not (0 <= c < len(shape) and shape[c] * positions == self.n_rows and all(z == 1 for z in shape[:c])
                            and (c + 1 >= len(shape) or shape[c + 1] in (1, positions))):
                        self.failed = self.failed or 'channel factor not on the channel axis of split rows'
            for field, value in ast.iter_fields(node):
                items = value if isinstance(value, list) else [value]
                new = [self.visit(i) if isinstance(i, ast.AST) and self._is_row(i) else i for i in items]
                setattr(node, field, new if isinstance(value, list) else new[0])
            return node
        for field, value in ast.iter_fields(node):
            items = value if isinstance(value, list) else [value]
            new = []
            for item in items:
                if isinstance(item, ast.expr) and not isinstance(item, (ast.Constant, ast.List, ast.Tuple)) \
                        and not self._is_row(item) and ndim is not None and not isinstance(item, ast.Attribute):
                    item = self._row_operand(item, ndim)
                elif isinstance(item, ast.AST):
                    item = self.visit(item)
                new.append(item)
            setattr(node, field, new if isinstance(value, list) else new[0])
        return node

    def visit_Call(self, node):
        if not self._is_row(node):
            return node
        func = node.func
        if isinstance(func, ast.Name) and func.id == 'patches_to_dense':
            return self._patches_rows(node)
        if isinstance(func, ast.Attribute) and func.attr in ('reshape', 'view', 'expand') and self._is_row(func.value):
            #FIX FOR: a reshape without N as its leading row entry (e.g. one merging rows) was kept as written.
            if not (_rewrite_row_literals(node, self.n_rows) or func.attr != 'expand' and self._split_rows(node)) \
                    or self.n_rows > 1 and self._row_axis(func.value)[0] is None:
                self.failed = self.failed or f'{func.attr} that does not keep the {self.n_rows} rows as its leading axis'
            node.func.value = self.visit(func.value)
            return node
        if isinstance(func, ast.Attribute) and self._is_row(func.value):
            #FIX FOR: any method on a row tensor was accepted, including ones that mix rows (transpose, sum(dim=1)).
            self.failed = self.failed or self._method_error(node)
            node.func.value = self.visit(func.value)
            node.args = [self.visit(a) for a in node.args]
            return node
        return self.generic_visit_row(node)

    def visit_BinOp(self, node):
        return self.generic_visit_row(node) if self._is_row(node) else node

    def visit_Subscript(self, node):
        if not self._is_row(node.value):
            return node
        node.value = self.visit(node.value)
        dims = node.slice.elts if isinstance(node.slice, ast.Tuple) else [node.slice]
        row = dims[1] if len(dims) > 1 else None
        if (isinstance(row, ast.Slice) and row.step is None and (row.lower is None or
                (isinstance(row.lower, ast.Constant) and row.lower.value == 0))
                and isinstance(row.upper, ast.Constant) and row.upper.value == self.n_rows):
            row.upper = ast.Name('_n', ast.Load())
        else:
            self.failed = 'subscript of a row tensor that does not span all rows'
        return node


def _components(stmts, probe, budget):
    """Traversals to chunk: (N, statement indices in the loop, cheap outside row statements)."""
    defs = {s.targets[0].id: i for i, s in enumerate(stmts) if isinstance(s, ast.Assign)}
    uses = {i: _names(s.value if isinstance(s, ast.Assign) else s) & defs.keys() for i, s in enumerate(stmts)}
    users = {i: [] for i in range(len(stmts))}
    for i, names in uses.items():
        for n in names:
            users[defs[n]].append(i)

    def rows(i):
        name = stmts[i].targets[0].id if isinstance(stmts[i], ast.Assign) else None
        shape = probe.shapes.get(name) if name else None
        return shape[1] if shape is not None and len(shape) >= 3 and shape[0] == 1 else None

    done, comps = set(), []
    for seed in sorted(probe.stmt_bytes, key=probe.stmt_bytes.get, reverse=True):
        if probe.stmt_bytes[seed] <= budget or seed in done or rows(seed) is None:
            continue
        n_rows = rows(seed)
        comp, todo = set(), [seed]
        while todo:
            i = todo.pop()
            if i in comp:
                continue
            comp.add(i)
            for j in [defs[n] for n in uses[i]] + users[i]:
                if j not in comp and rows(j) == n_rows:
                    todo.append(j)
        done |= comp
        # Statements of the traversal that use no other traversal tensor and are cheap: compute once, slice.
        names = {stmts[i].targets[0].id for i in comp}
        outside = {i for i in comp if not (uses[i] & names) and probe.stmt_bytes[i] <= budget / 8}
        loop = comp - outside
        # Row-indexed consumers of the coefficient tensors (the per-neuron sums) join the loop.
        for i in sorted(loop):
            if _per_row(probe.shapes[stmts[i].targets[0].id], n_rows):
                continue
            for j in users[i]:
                if j in loop or not isinstance(stmts[j], ast.Assign):
                    continue
                if _per_row(probe.shapes.get(stmts[j].targets[0].id), n_rows):
                    loop.add(j)
        comps.append((n_rows, loop, outside))
    return comps, defs, uses, users


#FEAT: per-neuron sums over long rows as two stages, so Inductor need not split the reduction (a split
# reduction materializes its pointwise input: a full extra write and read of the coefficients).
TWO_STAGE_MIN = int(os.environ.get('CF_TWO_STAGE_MIN', 32768))  # lower it only to test


def _sum_groups(k):
    """Groups for a two-stage sum over k elements: the largest divisor <= 1024 leaving >= 4096 per group."""
    per_group = min(4096, max(1, TWO_STAGE_MIN // 8))
    for g in range(min(1024, k // per_group), 1, -1):
        if k % g == 0:
            return g
    return None


class _TwoStageSums(ast.NodeTransformer):
    """x.sum(dim=-1) over [1, N, K] with K large -> x.reshape([1, N, G, K // G]).sum(dim=-1).sum(dim=-1)."""

    def __init__(self, probe, n_rows):
        self.probe, self.n_rows = probe, n_rows

    def visit_Call(self, node):
        f = node.func
        if (isinstance(f, ast.Attribute) and f.attr == 'sum' and not node.args and len(node.keywords) == 1
                and node.keywords[0].arg == 'dim' and isinstance(node.keywords[0].value, ast.UnaryOp)
                and ast.unparse(node.keywords[0].value) == '-1'):
            try:
                shape = self.probe.shape(f.value)
            except Exception:
                shape = None
            g = _sum_groups(shape[2]) if (shape is not None and len(shape) == 3 and shape[:2] == (1, self.n_rows)
                                         and shape[2] >= TWO_STAGE_MIN) else None
            if g is not None:
                inner = self.visit(f.value)
                dims = ast.List([ast.Constant(v) for v in (1, self.n_rows, g, shape[2] // g)], ast.Load())
                return _two_stage(inner, dims)
        return self.generic_visit(node)


def _two_stage(inner, dims):
    minus_one = [ast.keyword('dim', ast.UnaryOp(ast.USub(), ast.Constant(1)))]
    reshaped = ast.Call(ast.Attribute(inner, 'reshape', ast.Load()), [dims], [])
    first = ast.Call(ast.Attribute(reshaped, 'sum', ast.Load()), [], minus_one)
    return ast.Call(ast.Attribute(first, 'sum', ast.Load()), [], copy.deepcopy(minus_one))


def _fake_select(x, index, scale):
    """_cf_select of ROWS_HELPER, for shape probes."""
    return x.reshape(x.shape[0], -1)[:, index].view(x.shape[0], *scale.shape) * scale


#FEAT: in compiled bodies, a transposed conv whose weight selects channels (e.g. the MaxPool expansion's) is
# a gather that Inductor fuses into its consumer; flow() re-checks the weight on every call (cached).
class _SelectionConvs(ast.NodeTransformer):
    def __init__(self, probe, selections):
        self.probe, self.selections = probe, selections
        self.maps = []   # (index name, scale name, map expression) for the statements before the loop

    def visit_Call(self, node):
        self.generic_visit(node)
        if not (isinstance(node.func, ast.Name) and node.func.id == 'conv_transpose2d' and len(node.args) == 2
                and isinstance(node.args[1], ast.Name) and node.args[1].id in self.selections):
            return node
        kw = {k.arg: k.value for k in node.keywords}
        try:
            geo = [[int(v) for v in ast.literal_eval(kw[a])] for a in ('stride', 'padding', 'output_padding')]
            x_spatial = list(self.probe.shape(node.args[0])[2:])
        except Exception:
            return node
        w_shape = self.probe.shapes[node.args[1].id]
        if w_shape[2] > geo[0][0] or w_shape[3] > geo[0][1]:
            return node
        k = len(self.maps)
        idx, scale = f'_cf_selidx_{node.args[1].id}_{k}', f'_cf_selscale_{node.args[1].id}_{k}'
        self.maps.append((idx, scale, f'_cf_selection_map({node.args[1].id}, {x_spatial}, {geo[0]}, {geo[1]}, {geo[2]})'))
        #FIX FOR: the row-locality checks probe shapes of expressions holding _cf_select, which the probe did not know.
        with self.probe.mode:
            out = eval(compile(ast.fix_missing_locations(ast.Expression(node)), '<expr>', 'eval'), self.probe.env)
            self.probe.env[scale] = out.new_empty(out.shape[1:])
            self.probe.env[idx] = torch.empty(math.prod(out.shape[1:]), dtype=torch.long, device=out.device)
        self.probe.env.setdefault('_cf_select', _fake_select)
        return ast.Call(ast.Name('_cf_select', ast.Load()), [node.args[0], ast.Name(idx, ast.Load()),
                                                              ast.Name(scale, ast.Load())], [])


class _RowMixing(ValueError):
    """A loop statement whose product contracts the rows; .stmt is its index."""

    def __init__(self, stmt, message=None):
        super().__init__(message or f'statement {stmt} mixes rows')
        self.stmt = stmt


#FEAT: a statement that mixes rows (e.g. SkipPoly's next-layer weight times a same-width row matrix, VGG16
# fc 4096 -> 4096) leaves the loop: it and its loop inputs run once, unchunked, and the loop slices its result.
def _unchunk(stmts, probe, n_rows, loop, outside, users, budget, compile_bodies, selections):
    """_chunk_body, moving row-mixing statements out of the loop while they fit the budget."""
    defs = {stmts[i].targets[0].id: i for i in loop}
    moved = []
    while True:
        try:
            return _chunk_body(stmts, probe, n_rows, loop, outside, users, two_stage=compile_bodies,
                               selections=selections), moved
        except _RowMixing as e:
            group, todo = set(), [e.stmt]
            while todo:   # the statement and the loop statements it reads
                i = todo.pop()
                if i not in group:
                    group.add(i)
                    todo += [defs[n] for n in _names(stmts[i].value) & defs.keys()]
            big = [i for i in sorted(group) if probe.stmt_bytes.get(i, 0) > budget]
            if big:
                raise ValueError(f'{e}; running it unchunked needs statement {big[0]} '
                                 f'({probe.stmt_bytes[big[0]] / 2**30:.3g} GiB), over the budget') from None
            loop -= group
            outside |= group
            moved += sorted(group)
            for i in group:
                defs.pop(stmts[i].targets[0].id)


def _chunk_body(stmts, probe, n_rows, loop, outside, users, two_stage=False, selections=()):
    """Loop statements rewritten for rows [_r0, _r1), hoisted expressions, exported results, source periods."""
    row_vars = {stmts[i].targets[0].id for i in loop}
    sliced = {stmts[i].targets[0].id for i in outside}
    body, hoisted, periods, whole_periods, selection_maps = [], [], [], [], []
    for i in sorted(loop):
        stmt = copy.deepcopy(stmts[i])
        if two_stage:
            stmt.value = _TwoStageSums(probe, n_rows).visit(stmt.value)
        if selections:
            sel = _SelectionConvs(probe, selections)
            stmt.value = sel.visit(stmt.value)
            selection_maps += sel.maps
        t = _ChunkBody(probe, n_rows, row_vars | sliced, sliced)
        if t._is_row(stmt.value):
            stmt.value = t.visit(stmt.value)
        else:   # built only from other layers and weights, but row-aligned: take its rows
            stmt.value = t._row_operand(stmt.value, len(probe.shapes[stmts[i].targets[0].id]))
        if t.failed:
            msg = f'statement {i} ({stmts[i].targets[0].id}): {t.failed}'
            raise _RowMixing(i, msg) if t.mixes_rows else ValueError(msg)
        hoisted += t.hoisted
        periods += t.periods
        whole_periods += t.whole_periods
        body.append(stmt)
    exported = sorted({stmts[i].targets[0].id for i in loop
                       if any(j not in loop for j in users[i]) or isinstance(stmts[-1], ast.Return)
                       and stmts[i].targets[0].id in _names(stmts[-1])})
    for name in exported:
        shape = probe.shapes[name]
        if not _per_row(shape, n_rows):
            raise ValueError(f'{name} {shape} is used after the loop but is not a per-neuron result')
    pre = [ast.parse(f'{i}, {sc} = {e}').body[0] for i, sc, e in selection_maps]
    return pre + [ast.Assign([ast.Name(n, ast.Store())], e) for n, e in hoisted], body, exported, periods, whole_periods


def _emit_loop(n_rows, chunk, pre, body, exported, k=0):
    """Eager ast.For over row chunks plus the statements before and after it."""
    pre = list(pre)
    pre += [ast.parse(f'_cf_out_{n} = []').body[0] for n in exported]
    header = ast.parse(f'for _r0 in range(0, {n_rows}, {chunk}):\n'
                       f'    _r1 = min(_r0 + {chunk}, {n_rows})\n    _n = _r1 - _r0\n').body[0]
    header.body += body + [ast.parse(f'_cf_out_{n}.append({n})').body[0] for n in exported]
    post = [ast.parse(f'{n} = torch.cat(_cf_out_{n}, dim=1)').body[0] for n in exported]
    post.append(ast.parse(f'_cf_progress({k}, {n_rows})').body[0])
    return pre, header, post


def _largest_divisor(n, at_most):
    for d in range(min(n, max(1, at_most)), 0, -1):
        if n % d == 0:
            return d
    return 1


def _fixed_chunk(n_rows, target, periods):
    """A chunk size dividing n_rows (constant shapes, one compile) and the mode of its tiled sources."""
    periods = set(periods)
    if not periods:
        return _largest_divisor(n_rows, target), 'plain'
    if len(periods) == 1 and target >= next(iter(periods)):
        period = next(iter(periods))
        return period * _largest_divisor(n_rows // period, target // period), 'whole'
    return _largest_divisor(math.gcd(*periods), target), 'part'


class _BodyArgs(ast.NodeTransformer):
    """Turn per-chunk slicing calls into parameters of a compiled body; collect what the loop passes."""

    def __init__(self, probe, chunk, mode, k):
        self.probe, self.chunk, self.mode, self.k = probe, chunk, mode, k
        self.args = []      # (parameter, expression evaluated in the loop)
        self.positions = {} # period -> parameter holding the chunk's positions (sub-channel chunks)

    def _param(self, expr):
        name = f'_cf_a{len(self.args)}'
        self.args.append((name, expr))
        return ast.Name(name, ast.Load())

    def visit_Call(self, node):
        self.generic_visit(node)
        f = node.func
        if isinstance(f, ast.Name) and f.id in ('_cf_rows', '_cf_channels'):
            return self._param(node)
        if isinstance(f, ast.Name) and f.id == '_cf_tiled_rows':
            # The per-position source stays inside the body (Inductor fuses it into its consumer);
            # nothing of it is kept across chunks.
            src = node.args[0]
            shape = self.probe.shape(src)
            if self.mode == 'whole':
                return _repeat(src, self.chunk // shape[1], len(shape))
            period = shape[1]
            if period not in self.positions:
                self.positions[period] = self._param(ast.parse(
                    f'torch.arange(_r0 % {period}, _r0 % {period} + {self.chunk}, device=device_mode.get_device())',
                    mode='eval').body)
            return ast.Call(ast.Attribute(src, 'index_select', ast.Load()),
                            [ast.Constant(1), ast.Name(self.positions[period].id, ast.Load())], [])
        return node

    def visit_Name(self, node):
        return ast.Constant(self.chunk) if node.id == '_n' else node


def _repeat(param, times, ndim):
    return ast.Call(ast.Attribute(param, 'repeat', ast.Load()),
                    [ast.List([ast.Constant(1), ast.Constant(times)] + [ast.Constant(1)] * (ndim - 2), ast.Load())], [])


#FIX FOR: a compiled body merged F.pad with its input and produced unsound bounds (N40); keep pad out of the fusion.
class _RealizedPads(ast.NodeTransformer):
    def visit_Call(self, node):
        self.generic_visit(node)
        f = node.func
        if isinstance(f, ast.Attribute) and f.attr == 'pad' and isinstance(f.value, ast.Name) and f.value.id == 'F':
            node.func = ast.Name('_cf_pad', ast.Load())
        return node


INDUCTOR = ("dict(fullgraph=True, dynamic=False, options={'layout_optimization': False, "
            "'triton.autotune_pointwise': False, 'pre_grad_custom_pass': PRE_GRAD_CSE, "
            "'post_grad_custom_post_pass': SharedLhsMM()})")


def _emit_compiled(k, n_rows, chunk, mode, probe, pre, body, exported, module_globals):
    """A torch.compile'd body over one chunk, the eager loop calling it, and the statements around it."""
    t = _BodyArgs(probe, chunk, mode, k)
    body = [_RealizedPads().visit(t.visit(copy.deepcopy(s))) for s in body]
    defined = {tgt.id for s in body for tgt in s.targets}
    loads = set().union(*(_names(s.value) for s in body)) - defined - {a for a, _ in t.args}
    free = sorted(n for n in loads if n not in module_globals and n not in ("_cf_select", "_cf_pad"))  # helper: a module global
    params = [a for a, _ in t.args] + free
    fn = ast.FunctionDef(f'_cf_body_{k}', ast.arguments([], [ast.arg(p) for p in params], None, [], [], None, []),
                         body + [ast.Return(ast.Tuple([ast.Name(n, ast.Load()) for n in exported], ast.Load()))],
                         [], None, None, [])
    compiled = ast.parse(f'_cf_body_{k} = torch.compile(_cf_body_{k}, **{INDUCTOR})').body
    pre = list(pre)
    pre += [ast.parse(f'_cf_out_{n} = []').body[0] for n in exported]
    call = ast.Call(ast.Name(f'_cf_body_{k}', ast.Load()),
                    [e for _, e in t.args] + [ast.Name(n, ast.Load()) for n in free], [])
    loop = ast.parse(f'for _r0 in _cf_chunks({n_rows}, {chunk}):\n    _cf_res = None\n').body[0]
    loop.body = [ast.parse(f'_r1 = _r0 + {chunk}').body[0], ast.Assign([ast.Name('_cf_res', ast.Store())], call)]
    loop.body += [ast.parse(f'_cf_out_{n}.append(_cf_res[{j}])').body[0] for j, n in enumerate(exported)]
    post = [ast.parse(f'{n} = _cf_cat(_cf_out_{n}, {n_rows})').body[0] for n in exported]
    post.append(ast.parse(f'_cf_progress({k}, {n_rows})').body[0])
    return [fn] + compiled, pre, loop, post


def _add_dels(body, protected=()):
    """Insert `del` after each name's last use in a statement list (recursing into for-loops)."""
    # Only names assigned at this level; a loop body frees its own.
    defined_here = {t.id for s in body if isinstance(s, ast.Assign) for t in s.targets if isinstance(t, ast.Name)}
    last = {}
    for k, s in enumerate(body):
        for n in _names(s):
            last[n] = k
    out = []
    for k, s in enumerate(body):
        if isinstance(s, ast.For):
            s.body = _add_dels(s.body, protected=set(protected) | _loop_carried(s))
        out.append(s)
        dead = sorted(n for n, kk in last.items() if kk == k and n in defined_here and n not in protected
                      and not isinstance(s, ast.Return))
        if dead:
            out.append(ast.Delete([ast.Name(n, ast.Del()) for n in dead]))
    return out


def _loop_carried(loop):
    """Names a loop body must not delete: lists it appends to and the loop indices."""
    keep = {'_r0', '_r1', '_n'}
    for t in ast.walk(loop):
        if isinstance(t, ast.Attribute) and t.attr == 'append' and isinstance(t.value, ast.Name):
            keep.add(t.value.id)
    return keep


def rewrite(source, module_globals, inputs, budget, report=print, compile_bodies=False):
    tree = ast.parse(source)
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'flow')
    fn.decorator_list = []
    params = [a.arg for a in fn.args.args]
    stmts = fn.body
    probe = _Probe(module_globals, params, inputs)
    probe.run(stmts)
    comps, defs, uses, users = _components(stmts, probe, budget)
    #FEAT: flow() parameters holding channel-selection conv weights (compiled bodies only).
    from constraintflow.gbcsr.sparse_block import _channel_selection
    selections = {n for n, v in zip(params, inputs) if compile_bodies and isinstance(v, torch.Tensor) and v.dim() == 4
                  and v.is_floating_point() and _channel_selection(v) is not None}
    if not comps:
        report('[row-chunks] nothing over budget')
        return None
    # Each loop is one schedulable unit; everything else keeps its own statement.
    unit_of = {i: ('s', i) for i in range(len(stmts))}
    units, module_defs = {}, []
    for c, (n_rows, loop, outside) in enumerate(comps):
        (pre, body, exported, periods, whole_periods), moved = _unchunk(stmts, probe, n_rows, loop, outside, users,
                                                                        budget, compile_bodies, selections)
        if moved:
            report(f'[row-chunks] traversal of {n_rows} rows: statements {moved} mix rows, run unchunked before the loop')
        if not loop:
            continue
        per_row = max(probe.stmt_bytes.get(i, 0) for i in loop) / n_rows
        target = max(1, int(budget / (3 * per_row)))
        if compile_bodies:
            #FIX FOR: under a tight budget compiled bodies chose sub-channel chunks, which patches_to_dense refuses
            #         (SkipPoly on VGG16: 167-row target, 196-position channels); take at least whole channels.
            whole = math.lcm(*whole_periods) if whole_periods else 1
            chunk, mode = _fixed_chunk(n_rows, max(target, whole), periods)
            if any(chunk % p for p in whole_periods):
                raise ValueError(f'chunk of {chunk} rows splits a channel of patches_to_dense ({sorted(set(whole_periods))} positions)')
            defs_c, pre, header, post = _emit_compiled(c, n_rows, chunk, mode, probe, pre, body, exported,
                                                       module_globals)
            module_defs += defs_c
        else:
            whole = math.lcm(*whole_periods) if whole_periods else 1
            chunk = max(whole, target - target % whole)
            pre, header, post = _emit_loop(n_rows, chunk, pre, body, exported, c)
        units[('l', c)] = pre + [header] + post
        for i in loop:
            unit_of[i] = ('l', c)
        report(f'[row-chunks] traversal of {n_rows} rows: {len(loop)} statements in the loop, '
               f'{len(outside)} sliced, {math.ceil(n_rows / chunk)} chunks of {chunk} rows, '
               f'largest statement {max(probe.stmt_bytes.get(i, 0) for i in loop) / 2**30:.1f} GiB -> '
               f'{chunk * per_row / 2**30:.1f} GiB, results {exported}')
    for i in range(len(stmts)):
        if unit_of[i][0] == 's':
            units[unit_of[i]] = [stmts[i]]
    # Stable topological order of units by first statement index.
    first = {}
    for i in range(len(stmts)):
        first.setdefault(unit_of[i], i)
    deps = {u: set() for u in units}
    for i in range(len(stmts)):
        for n in uses[i]:
            d = unit_of[defs[n]]
            if d != unit_of[i]:
                deps[unit_of[i]].add(d)
    # Hoisted expressions inside loops may name other statements' variables too.
    for u, block in units.items():
        for s in block:
            for n in _names(s) & defs.keys():
                d = unit_of[defs[n]]
                if d != u:
                    deps[u].add(d)
    indeg = {u: len(d) for u, d in deps.items()}
    rdeps = {u: [] for u in units}
    for u, d in deps.items():
        for v in d:
            rdeps[v].append(u)
    ready = [(first[u], u) for u in units if indeg[u] == 0]
    heapq.heapify(ready)
    order = []
    while ready:
        _, u = heapq.heappop(ready)
        order.append(u)
        for v in rdeps[u]:
            indeg[v] -= 1
            if indeg[v] == 0:
                heapq.heappush(ready, (first[v], v))
    if len(order) != len(units):
        raise ValueError('row chunks: dependency cycle between loops')
    fn.body = _add_dels([s for u in order for s in units[u]], protected=set(params))
    helper = ast.parse(ROWS_HELPER).body + module_defs
    idx = tree.body.index(fn)
    tree.body[idx:idx] = helper
    return ast.unparse(ast.fix_missing_locations(tree))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('program')
    p.add_argument('--network', required=True)
    p.add_argument('--dataset', required=True)
    p.add_argument('--input-shape', help='e.g. 3x224x224; default: the ONNX input')
    p.add_argument('--budget-gb', type=float, default=20)
    p.add_argument('--compile-bodies', action='store_true',
                   help='torch.compile (Inductor) each loop body; chunk sizes divide the rows so shapes stay fixed')
    args = p.parse_args()
    from constraintflow.lib.globals import device_mode
    import onnx
    device_mode.set_mode('gpu')
    src_path = os.path.join(args.program, 'transformers.py')
    orig_path = os.path.join(args.program, 'transformers_unchunked.py')
    if not os.path.exists(orig_path):
        os.replace(src_path, orig_path)
    with open(orig_path) as f:
        source = f.read()
    with open(src_path, 'w') as f:      # load the unchunked program to build the inputs
        f.write(source)
    #FEAT: import the program as cli.py run does, so normal verification builds work (not only --training ones).
    sys.path.insert(0, os.path.abspath(args.program))
    import main as main_mod
    graph = onnx.load(args.network).graph
    init_names = {i.name for i in graph.initializer}
    data_input = next(i for i in graph.input if i.name not in init_names)
    shape = (tuple(map(int, args.input_shape.split('x'))) if args.input_shape
             else tuple(d.dim_value for d in data_input.type.tensor_type.shape.dim)[1:])
    inits = [i for i in graph.initializer if i.data_type == onnx.TensorProto.FLOAT]
    params = {i.name: torch.from_numpy(onnx.numpy_helper.to_array(i).copy()).cuda() for i in inits}
    x = torch.zeros(1, *shape, device='cuda')
    y = torch.zeros(1, dtype=torch.long, device='cuda')
    #FIX FOR: VNN-LIB datasets parse a spec into X, which must be cli.py's [batch, 2 (lo, hi), ...] placeholder box.
    from constraintflow.lib.spec import vnnlib_path
    X = torch.zeros(1, 2, *shape, device='cuda') if vnnlib_path(args.dataset) else x
    network, abs_elem = main_mod.build(args.network, 1, 0.0, X, y, args.dataset, False, False, False,
                                       initializers=params, input_box=(x, x))
    inputs = main_mod.explode_inputs(abs_elem, 1)
    transformers = sys.modules['transformers']
    out = rewrite(source, vars(transformers), inputs, args.budget_gb * 2**30, compile_bodies=args.compile_bodies)
    with open(src_path, 'w') as f:
        f.write(out if out is not None else source)


if __name__ == '__main__':
    main()
