"""Remove redundant sign-split convolutions in functional tensor SSA.

conv(max(x, 0), w) + conv(min(x, 0), w) = conv(x, w). Unlike the
traversal-level affine substitution pass, this proves that both weights and
all convolution parameters match; it works across residual branches too.
Corresponding leaves of residual sums are matched recursively, so
(C(x+, w) + C(y+, v)) + (C(x-, w) + C(y-, v)) becomes
C(x, w) + C(y, v). Shared sum/convolution definitions are not absorbed.
The identities are over real arithmetic and can change floating-point rounding.
Run before early reductions and partitioning, then remove dead assignments.
"""
import copy
from collections import Counter

from constraintflow.compiler.ir import (
    IrAssignment, IrFConv2d, IrFConvTranspose2d, IrSimpleBinary, IrTensorClamp,
    IrTorchExpand, IrTorchPermute, IrTorchRepeat, IrTorchReshape, IrTorchSqueeze,
    IrTorchTranspose, IrTorchUnsqueeze, IrTorchView, IrTransRetBasic, IrVar,
)
from constraintflow.compiler.optimizations import subexp_inlining

_WRAPPERS = {
    IrTorchReshape: ('shape',), IrTorchView: ('shape',),
    IrTorchExpand: ('shape',), IrTorchRepeat: ('repeats',),
    IrTorchUnsqueeze: ('index',), IrTorchSqueeze: ('index',),
    IrTorchTranspose: ('dim0', 'dim1'), IrTorchPermute: ('permutation',),
}
_CONVS = {IrFConv2d: ('stride', 'padding'),
          IrFConvTranspose2d: ('stride', 'padding', 'output_padding')}


def _same(a, b):
    # SSA names denote the same tensor even if a use has different IR metadata.
    if isinstance(a, IrVar) and isinstance(b, IrVar):
        return a.name == b.name
    return a is b or (type(a) is type(b) and a == b)


def _same_options(a, b, options):
    return type(a) is type(b) and all(
        _same(getattr(a, key), getattr(b, key)) for key in options[type(a)])


def _rebuild(shells, value):
    for shell in reversed(shells):
        out = copy.copy(shell)
        out.update_parent_child([value])
        value = out
    return value


def run(block):
    definitions, occupied, counts = {}, set(), Counter()
    for stmt in block.children:
        if isinstance(stmt, IrAssignment) and isinstance(stmt.children[0], IrVar):
            name = stmt.children[0].name
            if name in occupied:
                raise ValueError('sign convolutions requires SSA')
            occupied.add(name)
            exprs = [stmt.children[1]]
        elif isinstance(stmt, IrTransRetBasic):
            exprs = stmt.children
        else:
            raise ValueError('sign convolutions requires a functional tensor block')
        counts.update(v.name for v in subexp_inlining.get_vars_expr_occurrences(exprs))

    def peel(expr, target, single_use=False):
        shells = []
        while True:
            if isinstance(expr, IrVar) and expr.name in definitions:
                # Do not duplicate a shared convolution while simplifying one use.
                if single_use and counts[expr.name] != 1:
                    return None
                expr = definitions[expr.name]
            elif type(expr) in target:
                return expr, shells
            elif type(expr) in _WRAPPERS:
                shells.append(expr)
                expr = expr.children[0]
            else:
                return None

    fused = 0

    def match(left, right):
        """Prove a whole pair of additive trees before committing any fusion.

        Only corresponding leaves are paired: no arbitrary term sorting or
        reassociation search. Single-use checks apply to every sum, convolution,
        and output wrapper followed through a definition. Input clamps may be
        shared with reductions and remain alive when still needed there.
        """
        targets = (*_CONVS, IrSimpleBinary)
        a, b = (peel(c, targets, single_use=True) for c in (left, right))
        if a is None or b is None:
            return None
        ca, sa = a
        cb, sb = b
        if len(sa) != len(sb) or not all(_same_options(x, y, _WRAPPERS) for x, y in zip(sa, sb)):
            return None
        if isinstance(ca, IrSimpleBinary) or isinstance(cb, IrSimpleBinary):
            if not all(isinstance(c, IrSimpleBinary) and (
                    c.op if isinstance(c.op, str) else getattr(c.op, '__name__', None)
                    ) == 'add' for c in (ca, cb)):
                return None
            children, count = [], 0
            for x, y in zip(ca.children, cb.children):
                result = match(x, y)
                if result is None:
                    return None
                value, n = result
                children.append(value)
                count += n
            out = copy.copy(ca)
            out.update_parent_child(children)
            return _rebuild(sa, out), count
        if not _same_options(ca, cb, _CONVS) or not _same(ca.children[1], cb.children[1]):
            return None
        a = peel(ca.children[0], (IrTensorClamp,))
        b = peel(cb.children[0], (IrTensorClamp,))
        if a is None or b is None:
            return None
        xa, wa = a
        xb, wb = b
        if (xa.const != 0 or xb.const != 0 or xa.min_true == xb.min_true
                or not _same(xa.children[0], xb.children[0])):
            return None
        if len(wa) != len(wb) or not all(_same_options(x, y, _WRAPPERS) for x, y in zip(wa, wb)):
            return None
        conv = copy.copy(ca)
        conv.update_parent_child([_rebuild(wa, xa.children[0]), ca.children[1]])
        return _rebuild(sa, conv), 1

    def rewrite(expr):
        nonlocal fused
        if not hasattr(expr, 'children') or not expr.children:
            return expr
        out = copy.copy(expr)
        out.update_parent_child([rewrite(c) for c in expr.children])
        if not isinstance(out, IrSimpleBinary):
            return out
        op = out.op if isinstance(out.op, str) else getattr(out.op, '__name__', None)
        if op != 'add':
            return out
        result = match(*out.children)
        if result is None:
            return out
        value, count = result
        fused += count
        return value

    for stmt in block.children:
        if isinstance(stmt, IrAssignment):
            value = rewrite(stmt.children[1])
            stmt.update_parent_child([stmt.children[0], value])
            definitions[stmt.children[0].name] = value
        else:
            stmt.update_parent_child([rewrite(c) for c in stmt.children])
    if fused:
        while subexp_inlining.drop_dead_assignments(block.children):
            pass
    return fused
