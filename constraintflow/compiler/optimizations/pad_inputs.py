"""Materialize unfold-fed pad inputs (Inductor miscompile)."""
import copy

from constraintflow.compiler.ir import IrAssignment, IrFUnfold, IrTorchPad, IrVar
from constraintflow.compiler.optimizations import subexp_inlining


def run(block):
    occupied = {v.name for s in block.children for v in
                subexp_inlining.get_vars_expr_occurrences(s.children)}
    out, hoisted, counter = [], [], 0

    def has_unfold(expr):
        return isinstance(expr, IrFUnfold) or any(
            has_unfold(c) for c in getattr(expr, 'children', ()))

    def extract(expr):
        nonlocal counter
        if not hasattr(expr, 'children'):
            return expr
        expr = copy.copy(expr)
        expr.update_parent_child([extract(c) for c in expr.children])
        if isinstance(expr, IrTorchPad) and has_unfold(expr.children[0]):
            while 'pad_input_' + str(counter) in occupied:
                counter += 1
            name = 'pad_input_' + str(counter)
            occupied.add(name)
            value = IrVar(name, expr.children[0].irMetadata)
            out.append(IrAssignment(value, expr.children[0]))
            hoisted.append(name)
            expr.update_parent_child([copy.copy(value)])
        return expr

    for stmt in block.children:
        if isinstance(stmt, IrAssignment):
            stmt.update_parent_child([stmt.children[0], extract(stmt.children[1])])
        else:
            stmt.update_parent_child([extract(c) for c in stmt.children])
        out.append(stmt)
    block.update_parent_child(out)
    return hoisted
