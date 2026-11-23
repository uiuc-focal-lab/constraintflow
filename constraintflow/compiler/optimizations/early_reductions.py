"""Evaluate reductions before their large inputs outlive a traversal step.

Inlining can bury hundreds of independent sums in the final constant expression
of a backsubstitution. Their coefficient tensors then survive the entire walk.
Extract each sum or matrix-vector product immediately after its last reaching definition.
The reduction and the surrounding arithmetic tree are unchanged; only evaluation
order changes. Run after inlining, on functional SSA, and do not inline again.
"""

import copy

from constraintflow.compiler.ir import (
    IrAssignment, IrDel, IrSimpleUnary, IrTorchSum, IrTransRetBasic, IrVar,
    IrTorchMatmul, IrTorchUnsqueeze, IrTorchReshape, IrTorchView,
)
from constraintflow.compiler.optimizations import subexp_inlining


def run(block, render=None):
    """`render` (expr -> source text), when given, merges reductions with identical
    source: under SSA they compute the same value from the same definitions."""
    stmts = [s for s in block.children if not isinstance(s, IrDel)]
    definitions = {}
    for i, stmt in enumerate(stmts):
        if isinstance(stmt, IrAssignment) and isinstance(stmt.children[0], IrVar):
            name = stmt.children[0].name
            if name in definitions:
                raise ValueError('early reductions requires SSA (before name recycling)')
            definitions[name] = i
        elif not isinstance(stmt, IrTransRetBasic):
            raise ValueError('early reductions requires a functional tensor block')

    # Snapshot names before inserting anything so generated names cannot collide.
    occupied = set(definitions)
    for stmt in stmts:
        occupied.update(v.name for v in
                        subexp_inlining.get_vars_expr_occurrences(stmt.children))
    buckets, counter, seen = {}, 0, {}
    values = {s.children[0].name: s.children[1] for s in stmts
              if isinstance(s, IrAssignment)}

    def column_vector(expr):
        # Dense concretization lowers to matmul(coeff, bounds.unsqueeze(-1)).
        # Its small result must be scheduled just like an explicit sum; leaving
        # it in the final expression retains every large coefficient matrix.
        while isinstance(expr, IrVar) and expr.name in values:
            expr = values[expr.name]
        return ((isinstance(expr, IrTorchUnsqueeze) and expr.index == -1)
                or (isinstance(expr, (IrTorchReshape, IrTorchView))
                    and isinstance(expr.shape, (list, tuple))
                    and expr.shape and expr.shape[-1] == 1))

    def extract(expr, pos):
        nonlocal counter
        if not hasattr(expr, 'children'):
            return expr
        # IR expressions can be shared. Copy only the path being rewritten.
        expr = copy.copy(expr)
        expr.update_parent_child([extract(c, pos) for c in expr.children])
        if isinstance(expr, IrSimpleUnary) and isinstance(expr.op, IrVar):
            expr.op = copy.copy(expr.op)
        if not (isinstance(expr, IrTorchSum)
                or (isinstance(expr, IrTorchMatmul) and column_vector(expr.children[1]))):
            return expr
        deps = subexp_inlining.get_vars_expr_occurrences([expr])
        after = max((definitions.get(v.name, -1) for v in deps), default=-1)
        if after >= pos:
            raise ValueError('reduction reads a value before its definition')
        text = render(expr) if render is not None else None
        if text is not None and text in seen:
            return copy.copy(seen[text])
        while 'early_sum_' + str(counter) in occupied:
            counter += 1
        name = 'early_sum_' + str(counter)
        counter += 1
        occupied.add(name)
        definitions[name] = after
        value = IrVar(name, expr.irMetadata)
        buckets.setdefault(after, []).append(IrAssignment(value, expr))
        if text is not None:
            seen[text] = value
        return copy.copy(value)

    rewritten = []
    for i, stmt in enumerate(stmts):
        stmt = copy.copy(stmt)
        if isinstance(stmt, IrAssignment):
            stmt.update_parent_child([stmt.children[0], extract(stmt.children[1], i)])
        else:
            stmt.update_parent_child([extract(c, i) for c in stmt.children])
        rewritten.append(stmt)
    out = list(buckets.get(-1, ()))
    for i, stmt in enumerate(rewritten):
        # Preserve statement identity for existing network-layer cut marks.
        stmts[i].update_parent_child(stmt.children)
        out.append(stmts[i])
        out.extend(buckets.get(i, ()))
    block.update_parent_child(out)
    return sum(map(len, buckets.values()))
