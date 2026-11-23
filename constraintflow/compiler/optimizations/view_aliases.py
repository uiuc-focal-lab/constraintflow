# Optimization pass, always on in jit builds.
"""Inline storage-sharing views in functional SSA before memory planning.

Keeping a base and its views as separate segment outputs keeps both live across
segment boundaries. Recreate the cheap views at each reader instead, so only
the owner crosses them. No tensor arithmetic changes.
"""
import copy
from constraintflow.compiler.ir import (
    IrAssignment, IrVar, IrTransRetBasic, IrTorchSlice, IrTorchReshape,
    IrTorchView, IrTorchExpand, IrTorchPermute, IrTorchTranspose,
    IrTorchSqueeze, IrTorchUnsqueeze,
)

_VIEWS = (IrVar, IrTorchSlice, IrTorchReshape, IrTorchView, IrTorchExpand,
          IrTorchPermute, IrTorchTranspose, IrTorchSqueeze, IrTorchUnsqueeze)


def run(block, sized):
    if any(not isinstance(s, (IrAssignment, IrTransRetBasic)) for s in block.children):
        return 0
    names = [s.children[0].name for s in block.children if isinstance(s, IrAssignment)]
    if len(names) != len(set(names)):
        return 0
    aliases = {}
    known_roots = {v['root'] for v in sized.inputs.values()}

    def replace(expr):
        if isinstance(expr, IrVar):
            return aliases.get(expr.name, expr)
        if not getattr(expr, 'children', None):
            return expr
        children = [replace(c) for c in expr.children]
        if all(a is b for a,b in zip(children,expr.children)):
            return expr
        out = copy.copy(expr)
        out.update_parent_child(children)
        return out

    output = []
    for i, stmt in enumerate(block.children):
        if isinstance(stmt, IrAssignment):
            lhs, rhs = stmt.children
            rhs = replace(rhs)
            record = sized.get(i)
            if (record and not record['alloc'] and record['root'] in known_roots
                    and isinstance(rhs, _VIEWS)):
                aliases[lhs.name] = rhs
                continue
            stmt.update_parent_child([lhs, rhs])
            if record:
                known_roots.add(record['root'])
        else:
            stmt.update_parent_child([replace(c) for c in stmt.children])
        output.append(stmt)
    block.update_parent_child(output)
    return len(aliases)
