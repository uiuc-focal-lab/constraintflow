"""Fuse equal-vector complementary reductions in functional SSA addition trees.

sum(b * positive(A)) + sum(b * negative(A)) = sum(b * A), and likewise
for matrix-vector products. Equality is structural, never inferred from sampled
bounds. Reassociation changes floating-point rounding. Run before extraction of
early reductions; keep its scheduling and the region partitioner unchanged.
"""
import copy
from collections import Counter

from constraintflow.compiler.ir import (
    IrAssignment, IrFUnfold, IrSimpleBinary, IrTensorClamp, IrTorchMatmul,
    IrTorchSum, IrTorchUnsqueeze, IrTorchReshape, IrTorchView, IrTransRetBasic, IrVar,
)
from constraintflow.compiler.optimizations import subexp_inlining
#FEAT: per-change toggles for the DeepPoly-efficient passes
from constraintflow.compiler.optimizations.sign_convs import _WRAPPERS, _rebuild


def _op(expr, name):
    return isinstance(expr, IrSimpleBinary) and (
        expr.op if isinstance(expr.op, str) else getattr(expr.op, '__name__', None)) == name


def run(block):
    definitions, counts = {}, Counter()
    for stmt in block.children:
        if isinstance(stmt, IrAssignment) and isinstance(stmt.children[0], IrVar):
            name = stmt.children[0].name
            if name in definitions:
                raise ValueError('sign reductions requires SSA')
            definitions[name] = stmt.children[1]
            exprs = [stmt.children[1]]
        elif isinstance(stmt, IrTransRetBasic):
            exprs = stmt.children
        else:
            raise ValueError('sign reductions requires a functional tensor block')
        counts.update(v.name for v in subexp_inlining.get_vars_expr_occurrences(exprs))

    def resolve(expr, single=False):
        while isinstance(expr, IrVar) and expr.name in definitions:
            if single and counts[expr.name] != 1:
                break
            expr = definitions[expr.name]
        return expr

    # Intern structural keys to avoid expanding a large coefficient DAG into
    # exponentially large tuples. Unknown operations remain opaque identities.
    intern, memo, opaque = {}, {}, {}
    options = dict(_WRAPPERS)
    options.update({IrFUnfold: ('kernel_size', 'padding', 'stride'),
                    IrSimpleBinary: ('op',), IrTensorClamp: ('const', 'min_true')})

    def key(expr):
        if isinstance(expr, IrVar):
            if expr.name in definitions:
                if expr.name not in memo:
                    memo[expr.name] = key(definitions[expr.name])
                return memo[expr.name]
            value = ('var', expr.name)
        elif isinstance(expr, (tuple, list)):
            value = ('sequence', tuple(key(x) for x in expr))
        elif type(expr) in options:
            value = (type(expr), tuple(key(getattr(expr, a)) for a in options[type(expr)]),
                     tuple(key(c) for c in expr.children))
        elif isinstance(expr, (str, int, float, bool)) or expr is None:
            value = ('literal', type(expr), expr)
        else:
            opaque[id(expr)] = expr  # Keep identities alive for the entire pass.
            value = ('opaque', id(expr))
        return intern.setdefault(value, len(intern))

    def peel(expr, single=False):
        shells = []
        expr = resolve(expr, single)
        while type(expr) in _WRAPPERS:
            shells.append(expr)
            expr = resolve(expr.children[0], single)
        return expr, shells

    def shell_key(shells):
        return tuple((type(s), tuple(key(getattr(s, a)) for a in _WRAPPERS[type(s)]))
                     for s in shells)

    def candidate(expr):
        reduction, output = peel(expr, single=True)
        if isinstance(reduction, IrTorchSum):
            product = resolve(reduction.children[0], single=True)
            if not _op(product, 'mul'):
                return None
            kind = ('sum', key(reduction.dim))
        elif isinstance(reduction, IrTorchMatmul):
            # Restrict this pass to the column-vector lowering used by dense
            # concretization, rather than general matrix multiplication.
            rhs = resolve(reduction.children[1])
            if not ((isinstance(rhs, IrTorchUnsqueeze) and rhs.index == -1)
                    or (isinstance(rhs, (IrTorchReshape, IrTorchView))
                        and isinstance(rhs.shape, (list, tuple)) and rhs.shape
                        and rhs.shape[-1] == 1)):
                return None
            product, kind = reduction, ('matvec',)
        else:
            return None
        for index in (0, 1) if kind[0] == 'sum' else (0,):
            clamp, shells = peel(product.children[index])
            if not isinstance(clamp, IrTensorClamp) or clamp.const != 0:
                continue
            signature = (kind, index, key(clamp.children[0]), shell_key(shells),
                         key(product.children[1 - index]), shell_key(output))
            return signature, clamp.min_true, reduction, product, index, clamp, shells, output
        return None

    fused = 0
    #FIX FOR: polyzono on ResNets failed to compile: collect() pulled every single-use add-tree
    # temporary back into one statement (228 IR levels, past CPython's 200-paren limit), undoing
    # the MAX_INLINE_DEPTH cap that subexp_inlining had applied. Temporaries are now followed
    # only `limit` add levels deep, lowered per statement until it fits under the cap.
    limit = None

    def rewrite(expr):
        nonlocal fused
        root = resolve(expr, single=True)
        if _op(root, 'add'):
            leaves = []

            def collect(node, depth=0):
                if limit is not None and depth >= limit and isinstance(node, IrVar):
                    value = node
                else:
                    value = resolve(node, single=True)
                if _op(value, 'add'):
                    return (value, collect(value.children[0], depth + 1),
                            collect(value.children[1], depth + 1))
                leaves.append(node)
                return len(leaves) - 1

            tree = collect(expr)
            replacements, waiting = {}, {}
            for i, leaf in enumerate(leaves):
                c = candidate(leaf)
                if c is None:
                    continue
                signature, sign, reduction, product, index, clamp, shells, output = c
                opposite = (signature, not sign)
                if not waiting.get(opposite):
                    waiting.setdefault((signature, sign), []).append(i)
                    continue
                first = waiting[opposite].pop(0)
                # Preserve the first leaf's operands and output representation.
                _, _, reduction, product, index, clamp, shells, output = candidate(leaves[first])
                merged = copy.copy(product)
                children = list(product.children)
                children[index] = _rebuild(shells, clamp.children[0])
                merged.update_parent_child(children)
                if isinstance(reduction, IrTorchSum):
                    new_reduction = copy.copy(reduction)
                    new_reduction.update_parent_child([merged])
                    merged = new_reduction
                replacements[first] = _rebuild(output, merged)
                replacements[i] = None
                fused += 1

            def rebuild(node):
                if isinstance(node, int):
                    return replacements[node] if node in replacements else rewrite_leaf(leaves[node])
                original, left, right = node
                left, right = rebuild(left), rebuild(right)
                if left is None:
                    return right
                if right is None:
                    return left
                out = copy.copy(original)
                out.update_parent_child([left, right])
                return out

            # Do not inline aliases or reassociate a tree without a match.
            return rebuild(tree) if replacements else rewrite_leaf(expr)
        return rewrite_leaf(expr)

    def rewrite_leaf(expr):
        if not getattr(expr, 'children', None):
            return expr
        out = copy.copy(expr)
        out.update_parent_child([rewrite(c) for c in expr.children])
        return out

    def rewrite_capped(expr):
        nonlocal fused, limit
        before, limit = fused, None
        while True:
            value = rewrite(expr)
            if fused == before or subexp_inlining.expr_depth(value) <= subexp_inlining.MAX_INLINE_DEPTH:
                return value
            fused = before
            if limit == 0:
                return expr   # unreachable in practice: limit 0 inlines nothing new
            limit = subexp_inlining.MAX_INLINE_DEPTH // 2 if limit is None else limit // 2

    for stmt in block.children:
        if isinstance(stmt, IrAssignment):
            value = rewrite_capped(stmt.children[1])
            stmt.update_parent_child([stmt.children[0], value])
            definitions[stmt.children[0].name] = value
            # In straight-line SSA, preceding definitions are finalized and
            # cannot depend on this new definition, so their structural keys stay valid.
            memo.pop(stmt.children[0].name, None)
        else:
            stmt.update_parent_child([rewrite_capped(c) for c in stmt.children])
    if fused:
        while subexp_inlining.drop_dead_assignments(block.children):
            pass
    return fused
