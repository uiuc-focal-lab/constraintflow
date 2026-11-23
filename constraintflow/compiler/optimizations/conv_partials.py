"""Expose partial convolution sums to bounded-region partitioning.

Inductor can fuse a long tree of additions into one kernel, retaining every
extern convolution output until that kernel runs. Materializing the existing
subtrees lets the region partitioner force earlier accumulation without
reassociating floating-point additions. Reduction producer chains stay intact.
"""
import copy

from constraintflow.compiler.ir import (
    IrAssignment, IrFConv2d, IrFConvTranspose2d, IrSimpleBinary, IrVar,
)
from constraintflow.compiler.optimizations import subexp_inlining


def run(block):
    occupied = {v.name for s in block.children for v in
                subexp_inlining.get_vars_expr_occurrences(s.children)}
    out, counter = [], 0

    def materialize(expr):
        nonlocal counter
        if isinstance(expr, IrVar):
            return expr
        while 'conv_partial_' + str(counter) in occupied:
            counter += 1
        name = 'conv_partial_' + str(counter)
        counter += 1
        occupied.add(name)
        value = IrVar(name, expr.irMetadata)
        out.append(IrAssignment(value, expr))
        return copy.copy(value)

    def extract(expr):
        if not hasattr(expr, 'children'):
            return expr, 0
        expr = copy.copy(expr)
        children, count = [], int(isinstance(expr, (IrFConv2d, IrFConvTranspose2d)))
        for child in expr.children:
            child, n = extract(child)
            children.append(child)
            count += n
        expr.update_parent_child(children)
        op = getattr(expr, 'op', None)
        op = op if isinstance(op, str) else getattr(op, '__name__', None)
        if isinstance(expr, IrSimpleBinary) and op in ('add', 'sub') and count > 2:
            expr.update_parent_child([materialize(c) for c in children])
            expr = materialize(expr)
        return expr, count

    before = len(block.children)
    for stmt in block.children:
        if isinstance(stmt, IrAssignment):
            stmt.update_parent_child([stmt.children[0], extract(stmt.children[1])[0]])
        else:
            stmt.update_parent_child([extract(c)[0] for c in stmt.children])
        out.append(stmt)
    block.update_parent_child(out)
    return len(out) - before
