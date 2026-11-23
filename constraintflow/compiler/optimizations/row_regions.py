"""Shared tensor row slicing, shape probing, and fixed-shape region emission.

This module contains no activation recognition or bound-generation algorithm.
"""
import copy
import io
import operator
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from constraintflow.compiler.ir import (
    IrAssignment, IrConst, IrFConv2d, IrPatchesToDense, IrRowGather, IrTorchDiagEmbed,
    IrSelectedPatchesToDense, IrSimpleBinary, IrSimpleUnary, IrTensorClamp,
    IrTorchExpand, IrTorchMatmul, IrTorchPad, IrTorchPermute, IrTorchRepeat, IrTorchReshape,
    IrTorchSlice, IrTorchSqueeze, IrTorchTranspose, IrTorchUnsqueeze, IrTorchView, IrTorchWhere,
    IrTransRetBasic, IrVar,
)
from constraintflow.compiler.optimizations.subexp_inlining import get_vars_expr_occurrences
from constraintflow.gbcsr.sparse_block import RUNTIME_HELPERS
from constraintflow.lib import globals


def names(expr):
    return {v.name for v in get_vars_expr_occurrences(expr)}


def var(name):
    return IrVar(name, [])


def assign(name, expr):
    return IrAssignment(var(name), expr)


def walk(expr):
    yield expr
    for child in getattr(expr, 'children', ()):
        yield from walk(child)


class Shapes:
    """Evaluate IR expressions with fake tensors, as in flow_shapes.probe."""
    def __init__(self, cg, node):
        from torch._subclasses.fake_tensor import FakeTensorMode
        self.cg = cg
        self.mode = FakeTensorMode(allow_non_fake_inputs=True)
        self.env = dict(torch=torch, F=F, operator=operator,
                        device_mode=globals.device_mode, **RUNTIME_HELPERS)
        self.env['batch_size'] = node.flow_batch_size
        with self.mode:
            for name, (shape, dtype) in node.flow_param_meta.items():
                self.env[name] = (torch.empty(shape, device=globals.device_mode.get_device(),
                    dtype={'Bool': torch.bool, 'Int': torch.int64}.get(dtype, torch.float))
                    if shape is not None else 0.)
            for stmt in node.flow_block.children:
                if isinstance(stmt, IrAssignment):
                    self.env[stmt.children[0].name] = self.value(stmt.children[1])
        self.shapes = {n: tuple(v.shape) if isinstance(v, torch.Tensor) else None
                       for n, v in self.env.items()}

    def value(self, expr):
        text = self.cg.visit(expr)
        with self.mode:
            for source, name in self.cg.const_pool.items():
                if name not in self.env:
                    self.env[name] = eval(source, self.env)
            return eval(text, self.env)

    def shape(self, expr):
        value = self.value(expr)
        return tuple(value.shape) if isinstance(value, torch.Tensor) else None


_ELEMENTWISE = (IrSimpleBinary, IrSimpleUnary, IrTensorClamp, IrTorchWhere)
_VIEWS = (IrTorchReshape, IrTorchView, IrTorchExpand)
# Elementwise arithmetic over views: Inductor fuses a row gather into it, so it costs only the selected rows.
#FIX FOR: identity patches (pad(diag_embed(ones))) were built for every row before the row kernel
#         (N35: 2 x 400 MiB for 65536 rows); diag_embed lowers to a pointwise select, so gather it inside.
_POINTWISE = _ELEMENTWISE + _VIEWS + (IrVar, IrConst, IrTorchSqueeze, IrTorchUnsqueeze, IrTorchPad,
                                      IrTorchSlice, IrTorchPermute, IrTorchTranspose, IrTorchRepeat,
                                      IrTorchDiagEmbed, int, float)


class SelectRows:
    def __init__(self, probe, n, rows, batch, prefix):
        self.probe, self.n, self.rows, self.batch = probe, n, rows, batch
        self.prefix, self.hoisted = prefix, []

    def is_row(self, expr):
        return bool(names(expr) & self.rows)

    def operand(self, expr, rank):
        shape = self.probe.shape(expr)
        axis = 1 - (rank - len(shape)) if shape is not None else -1
        if shape is None or axis < 0 or axis >= len(shape) or shape[axis] != self.n:
            return expr
        if len(shape) < 2 or shape[0] not in (1, self.batch) or shape[1] != self.n:
            raise ValueError(f'unsupported stable ReLU row operand: {shape}')
        if isinstance(expr, _ELEMENTWISE):
            result = copy.copy(expr)
            result.children = [self.operand(c, rank) for c in expr.children]
            return result
        # Channel weights expanded over spatial positions: gather channels first.
        if isinstance(expr, (IrTorchReshape, IrTorchView)):
            perm = expr.children[0]
            if isinstance(perm, IrTorchPermute) and list(perm.permutation) == [0, 1, 3, 2]:
                rep = perm.children[0]
                if isinstance(rep, (IrTorchRepeat, IrTorchExpand)):
                    dims = rep.repeats if isinstance(rep, IrTorchRepeat) else rep.shape
                    unsq = rep.children[0]
                    if (isinstance(dims, (list, tuple)) and len(dims) == 4
                            and list(dims[:3]) in ([1, 1, 1], [-1, -1, -1])
                            and isinstance(dims[3], int) and dims[3] > 0
                            and isinstance(unsq, IrTorchUnsqueeze) and unsq.index in (-1, 3)):
                        return IrRowGather(unsq.children[0], divisor=dims[3])
        if isinstance(expr, IrPatchesToDense):
            return IrSelectedPatchesToDense(self.operand(expr.children[0], 3), expr.geometry)
        if isinstance(expr, IrTorchRepeat):
            dims = expr.repeats
            if (isinstance(dims, (list, tuple)) and dims[0] == 1
                    and all(d == 1 for d in dims[2:])):
                return IrRowGather(expr.children[0], period=self.probe.shape(expr.children[0])[1])
        #FIX FOR: operands behind squeeze/pad views were computed for every row before the row kernel
        #         (N33's last layer: 452 MiB with 0 unstable rows); gather them inside the kernel instead.
        if all(isinstance(e, _POINTWISE) for e in walk(expr)):
            return IrRowGather(expr)
        name = f'{self.prefix}_{len(self.hoisted)}'
        self.hoisted.append(assign(name, expr))
        return IrRowGather(var(name))

    def visit(self, expr):
        if not self.is_row(expr):
            return expr
        result = copy.copy(expr)
        if isinstance(expr, _VIEWS) and self.is_row(expr.children[0]):
            dims = list(expr.shape)
            if (len(dims) >= 2 and dims[0] in (1, self.batch) and dims[1] == self.n):
                dims[1] = var('_n')
            elif isinstance(expr, (IrTorchReshape, IrTorchView)) and dims[0] == self.batch * self.n:
                dims[0] = IrSimpleBinary(IrConst(self.batch, 'Int'), var('_n'), 'mul')
            else:
                prefix = 1
                for i, dim in enumerate(dims):
                    if not isinstance(dim, int):
                        break
                    if prefix == 1 and dim == self.n:
                        dims[i] = var('_n')
                        break
                    prefix *= dim
            result.shape = dims
        if isinstance(expr, IrPatchesToDense):
            return IrSelectedPatchesToDense(self.visit(expr.children[0]), expr.geometry)
        rank = len(self.probe.shape(expr)) if isinstance(expr, _ELEMENTWISE) else None
        result.children = [self.visit(c) if self.is_row(c) else
                           self.operand(c, rank) if rank is not None else c
                           for c in expr.children]
        return result


@dataclass
class Region:
    name: str
    body: list
    outputs: list
    params: list
    selected: bool = False


@dataclass
class Call:
    outputs: list
    function: str
    args: list
    # refinement passes a region function and a tuple of its inputs
    refinement: tuple | None = None
    extra_reads: tuple = ()


def render_dense(cg, statements):
    """Render an eager diagnostic with SSA last-use cleanup."""
    real_file, old_indent = cg.file, cg.indent
    cg.file, cg.indent = io.StringIO(), 1
    local = {s.children[0].name for s in statements if isinstance(s, IrAssignment)}
    reads = [names(s.children[1]) if isinstance(s, IrAssignment) else names(s)
             for s in statements]
    last = {n: i for i, read in enumerate(reads) for n in read}
    for i, stmt in enumerate(statements):
        cg.visit(stmt)
        if not isinstance(stmt, IrTransRetBasic):
            dead = sorted(n for n in reads[i] & local if last[n] == i)
            if dead:
                cg.write('del ' + ', '.join(dead))
    result = cg.file.getvalue()
    cg.file, cg.indent = real_file, old_indent
    return result


def emit_regions(cg, node, regions, calls, chunk_size, compile_kernels, runtime_import, release_dense=False):
    """Emit fixed-shape row regions and an eager coordinator (algorithm agnostic)."""
    # Build a dense diagnostic module from the same final IR, without source matching.
    dense = ('def flow(' + ', '.join(n for n, _ in node.flow_params) + ', batch_size):\n'
             + (render_dense(cg, node.flow_block.children) if release_dense else
                cg._render_stmts(node.flow_block.children)))
    start = cg.file.tell()
    cg._emit_explode_inputs(list(node.flow_params) + [('batch_size', 'batch_size')])
    cg.indent = 0
    cg.write(runtime_import)
    for reg in regions:
        if compile_kernels:
            cg.write("@torch.compile(fullgraph=True, dynamic=False, options={'layout_optimization': False})")
        cg.write('def ' + reg.name + '(' + ', '.join(reg.params) + '):')
        cg.indent = 1
        if reg.selected:
            cg.write('_n = _idx.numel()')
        last = {n: i for i, s in enumerate(reg.body) for n in names(s.children[1])}
        local = {s.children[0].name for s in reg.body}
        for i, stmt in enumerate(reg.body):
            cg.visit(stmt)
            dead = sorted(n for n in names(stmt.children[1]) & local - set(reg.outputs) if last[n] == i)
            if dead:
                cg.write('del ' + ', '.join(dead))
        cg.write('return (' + ', '.join(reg.outputs) + (',' if reg.outputs else '') + ')')
        cg.indent = 0
        cg.write('')
    cg.write('def flow(' + ', '.join(n for n, _ in node.flow_params) + ', batch_size):')
    cg.indent = 1
    reads = [set(c.args) | set(c.extra_reads) | (set(c.refinement[1]) if c.refinement else set()) for c in calls]
    last = {n: i for i, ns in enumerate(reads) for n in ns}
    assigned = {n for c in calls for n in c.outputs}
    for i, call in enumerate(calls):
        args = list(call.args)
        if call.refinement:
            name, inputs = call.refinement
            args += [name, '(' + ', '.join(inputs) + (',' if inputs else '') + ')', repr(chunk_size)]
        expr = call.function + '(' + ', '.join(args) + ')'
        if i == len(calls) - 1:
            cg.write('return ' + expr)
        else:
            cg.write((', '.join(call.outputs) + ', = ' if call.outputs else '') + expr)
            dead = sorted(n for n in assigned & reads[i] if last[n] == i)
            if dead:
                cg.write('del ' + ', '.join(dead))
    cg.indent = 0
    cg.write('')
    end = cg.file.tell()
    # Include the matching input interface in the dense diagnostic artifact.
    real_file = cg.file
    cg.file = io.StringIO()
    cg._emit_explode_inputs(list(node.flow_params) + [('batch_size', 'batch_size')])
    dense = cg.file.getvalue() + dense
    cg.file = real_file
    cg._stable_dense_source = start, end, dense
    cg._region_count = len(regions)
