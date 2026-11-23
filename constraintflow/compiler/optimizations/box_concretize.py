"""Concretize against the input box with center/radius instead of a sign split.

matvec(A+, x) + matvec(A-, y) = matvec(A, (x+y)/2) + matvec(A+ - A-, (x-y)/2)

where x and y are the input layer's lower/upper bound tensors (flow parameters
in_l/in_u or in_L_const/in_U_const). The sign-split form adds two large sums of
nearly equal magnitude; when the box is narrow (eps 4e-5 on CIFAR) their
cancellation loses the interval width to fp32 rounding, and the error grows
with depth. auto_LiRPA concretizes with A.(x_U+x_L)/2 -/+ |A|.(x_U-x_L)/2; the
pair is written in that order so both tools round alike. Same matvec count.
#FEAT: (box_patch_reductions)
For patch sums, fuse the center and radius terms into one reduction. The
same algebra applies to affine-constant vectors, with matching gather/view
geometry, even when they are not the input box's endpoints.
Run on functional SSA before product grids and early reductions.
"""
import copy
import re
from collections import Counter

from constraintflow.compiler.ir import (
    #FEAT: box_patch_reductions
    IrAssignment, IrConst, IrFConv2d, IrFUnfold, IrTorchSum, IrSimpleBinary, IrTensorClamp, IrTorchMatmul,
    IrTorchSlice, IrTorchUnsqueeze, IrTorchReshape, IrTorchView, IrTransRetBasic, IrVar,
)
from constraintflow.compiler.optimizations import subexp_inlining
#FEAT: per-change toggles for the DeepPoly-efficient passes
from constraintflow.compiler.optimizations.sign_convs import _WRAPPERS, _rebuild
from constraintflow.compiler.optimizations.sign_reductions import _op

# Input-layer bound parameters: (role, block) with role 'l' or 'u'.
_BOX = re.compile(r'^in_(?:(l|u)|(L|U)_const)_(\d+)$')
# unfold gathers are peeled too
_BOX_VIEWS = dict(_WRAPPERS)
_BOX_VIEWS[IrFUnfold] = ('kernel_size', 'padding', 'stride')


def _views():
    return _BOX_VIEWS


def run(block):
    definitions, counts = {}, Counter()
    for stmt in block.children:
        if isinstance(stmt, IrAssignment) and isinstance(stmt.children[0], IrVar):
            name = stmt.children[0].name
            if name in definitions:
                raise ValueError('box concretization requires SSA')
            definitions[name] = stmt.children[1]
            exprs = [stmt.children[1]]
        elif isinstance(stmt, IrTransRetBasic):
            exprs = stmt.children
        else:
            raise ValueError('box concretization requires a functional tensor block')
        counts.update(v.name for v in subexp_inlining.get_vars_expr_occurrences(exprs))

    def resolve(expr, single=False):
        while isinstance(expr, IrVar) and expr.name in definitions:
            if single and counts[expr.name] != 1:
                break
            expr = definitions[expr.name]
        return expr

    def peel(expr, single=False):
        shells = []
        expr = resolve(expr, single)
        #FEAT: box_patch_reductions
        views = _views()
        while type(expr) in views:
            shells.append(expr)
            expr = resolve(expr.children[0], single)
        return expr, shells

    def box_role(expr):
        # ('l'|'u', block) when expr is a view of an input bound parameter.
        base, _ = peel(expr)
        if isinstance(base, IrVar) and base.name not in definitions:
            m = _BOX.match(base.name)
            if m:
                return ((m.group(1) or m.group(2)).lower(), m.group(3))
        return None

    def same(a, b):
        a, b = resolve(a), resolve(b)
        if isinstance(a, IrVar) and isinstance(b, IrVar):
            return a.name == b.name
        return a is b

    def candidate(expr):
        # matmul(view(clamp(A, 0)), column view of a box tensor)
        reduction, output = peel(expr, single=True)
        # patch-sum and affine-constant reductions
        if isinstance(reduction, IrTorchSum):
            product = resolve(reduction.children[0], single=True)
            if not _op(product, 'mul'):
                return None
            for index in (0, 1):
                clamp, shells = peel(product.children[index])
                vector = product.children[1-index]
                role = box_role(vector)
                if isinstance(clamp, IrTensorClamp) and clamp.const == 0:
                    # The same identity also fuses affine-constant reductions.
                    # Equal view/gather geometry is checked by compatible().
                    role = role or ('generic_pos' if clamp.min_true else 'generic_neg', 'generic')
                    return reduction, clamp, shells, output, role, vector
            return None
        if not isinstance(reduction, IrTorchMatmul):
            return None
        rhs = resolve(reduction.children[1])
        if not ((isinstance(rhs, IrTorchUnsqueeze) and rhs.index == -1)
                or (isinstance(rhs, (IrTorchReshape, IrTorchView))
                    and isinstance(rhs.shape, (list, tuple)) and rhs.shape
                    and rhs.shape[-1] == 1)):
            return None
        role = box_role(reduction.children[1])
        clamp, shells = peel(reduction.children[0])
        if role is None or not isinstance(clamp, IrTensorClamp) or clamp.const != 0:
            return None
        #FEAT: candidates carry their vector for box_patch_reductions
        return reduction, clamp, shells, output, role, reduction.children[1]

    def compatible(a, b):
        ra, ca, sa, oa, roa, va = a
        rb, cb, sb, ob, rob, vb = b
        # vector gather geometry and reduction kinds must match
        _, av = peel(va)
        _, bv = peel(vb)
        return (type(ra) is type(rb)
                and (not isinstance(ra, IrTorchSum) or ra.dim == rb.dim)
                and ca.min_true != cb.min_true and roa[1] == rob[1] and roa[0] != rob[0]
                and same(ca.children[0], cb.children[0])
                and len(sa) == len(sb) and len(oa) == len(ob) and len(av) == len(bv)
                and all(type(x) is type(y) and all(getattr(x, k) == getattr(y, k)
                                                   for k in _BOX_VIEWS[type(x)])
                        for x, y in zip(sa + oa + av, sb + ob + bv)))

    fused = 0

    def half(lhs, rhs, op):
        return IrSimpleBinary(IrSimpleBinary(lhs, rhs, op), IrConst(0.5, 'Float'), 'mul')

    def replace_vector(vector_expr, new_base):
        # Rebuild the column view chain of `vector_expr` over `new_base`.
        _, shells = peel(vector_expr)
        return _rebuild(shells, new_base)

    def fuse(a, b):
        # a, b: candidates; returns (center term, radius term, sign of radius term)
        pos, neg = (a, b) if a[1].min_true else (b, a)
        #FEAT: box_patch_reductions
        reduction, clamp, shells, output, role, _vector = pos
        x_vec, y_vec = pos[5], neg[5]
        x_base, _ = peel(x_vec)
        y_base, _ = peel(y_vec)
        # A+.x + A-.y = A.(x+y)/2 + |A|.(x-y)/2; write it over (u, l) like auto_LiRPA.
        if role[0] == 'u':
            hi, lo, sign = x_base, y_base, 'add'
        else:
            hi, lo, sign = y_base, x_base, 'sub'
        coeff = clamp.children[0]
        abs_coeff = IrSimpleBinary(IrTensorClamp(coeff, 0, True), IrTensorClamp(coeff, 0, False), 'sub')
        #FEAT: box_patch_reductions rebuilds a fused patch-sum reduction
        if isinstance(reduction, IrTorchSum):
            if role[1] == 'generic':
                # Arbitrary constant roots can have different shapes before
                # their views. Combine the actual gathered vectors instead.
                center_vec, radius_vec = half(x_vec, y_vec, 'add'), half(x_vec, y_vec, 'sub')
                sign = 'add'
            else:
                center_vec = replace_vector(x_vec, half(hi, lo, 'add'))
                radius_vec = replace_vector(x_vec, half(hi, lo, 'sub'))
            center = IrSimpleBinary(_rebuild(shells, coeff), center_vec, 'mul')
            radius = IrSimpleBinary(_rebuild(shells, abs_coeff), radius_vec, 'mul')
            return _rebuild(output, IrTorchSum(IrSimpleBinary(center, radius, sign), reduction.dim))
        center = copy.copy(reduction)
        center.update_parent_child([_rebuild(shells, coeff), replace_vector(x_vec, half(hi, lo, 'add'))])
        radius = copy.copy(reduction)
        radius.update_parent_child([_rebuild(shells, abs_coeff), replace_vector(x_vec, half(hi, lo, 'sub'))])
        return IrSimpleBinary(_rebuild(output, center), _rebuild(output, radius), sign)

    def rewrite(expr):
        nonlocal fused
        root = resolve(expr, single=True)
        if _op(root, 'add'):
            leaves = []

            def collect(node):
                value = resolve(node, single=True)
                if _op(value, 'add'):
                    return (value, collect(value.children[0]), collect(value.children[1]))
                leaves.append(node)
                return len(leaves) - 1

            tree = collect(expr)
            cands = [candidate(leaf) for leaf in leaves]
            replacements, used = {}, set()
            for i, ci in enumerate(cands):
                if ci is None or i in used:
                    continue
                for j in range(i + 1, len(cands)):
                    cj = cands[j]
                    if cj is None or j in used or not compatible(ci, cj):
                        continue
                    replacements[i] = fuse(ci, cj)
                    replacements[j] = None
                    used.update((i, j))
                    fused += 1
                    break

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

            return rebuild(tree) if replacements else rewrite_leaf(expr)
        return rewrite_leaf(expr)

    def rewrite_leaf(expr):
        if not getattr(expr, 'children', None):
            return expr
        out = copy.copy(expr)
        out.update_parent_child([rewrite(c) for c in expr.children])
        return out

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


def fold_conv_bias(block):
    """Interval conv as auto_LiRPA computes it: conv(mid, W, b) -/+ conv(rad, |W|).

    The center/radius lowering yields (conv(mid, W) -/+ conv(rad, |W|)) + b; auto_LiRPA's
    BoundConv adds the bias inside the center convolution, i.e. (c + b) -/+ d. Equal in
    real arithmetic, rounded differently. The per-neuron bias flow parameter is constant
    per output channel, so its first column after a [C, -1] view is the conv bias.
    Returns the number of convolutions that took the bias."""
    definitions = {s.children[0].name: s.children[1] for s in block.children
                   if isinstance(s, IrAssignment) and isinstance(s.children[0], IrVar)}

    def resolve(expr):
        while isinstance(expr, IrVar) and expr.name in definitions:
            expr = definitions[expr.name]
        return expr

    def peel(expr):
        shells = []
        expr = resolve(expr)
        #FEAT: box_patch_reductions
        views = _views()
        while type(expr) in views:
            shells.append(expr)
            expr = resolve(expr.children[0])
        return expr, shells

    def param(expr):
        base, _ = peel(expr)
        return base.name if isinstance(base, IrVar) and base.name not in definitions else None

    def is_abs_of(expr, weight):
        # sub(clamp(W, min=0), clamp(W, max=0))
        e = resolve(expr)
        if not _op(e, 'sub'):
            return False
        a, b = (resolve(c) for c in e.children)
        return (isinstance(a, IrTensorClamp) and isinstance(b, IrTensorClamp)
                and a.const == 0 and b.const == 0 and a.min_true and not b.min_true
                and param(a.children[0]) == weight == param(b.children[0]))

    folded = 0

    def match(expr):
        # add(op(view(conv(mid, W)), view(conv(rad, |W|))), view(bias)) with op in {add, sub}
        nonlocal folded
        e = resolve(expr)
        if not _op(e, 'add'):
            return None
        for inner, bias in (e.children, e.children[::-1]):
            bias_name = param(bias)
            interval = resolve(inner)
            if bias_name is None or not (_op(interval, 'add') or _op(interval, 'sub')):
                continue
            center, c_shells = peel(interval.children[0])
            radius, _ = peel(interval.children[1])
            if not (isinstance(center, IrFConv2d) and isinstance(radius, IrFConv2d)
                    and len(center.children) == 2 and len(radius.children) == 2):
                continue
            weight = param(center.children[1])
            if weight is None or not is_abs_of(radius.children[1], weight):
                continue
            w_var, _ = peel(center.children[1])
            out_channels = w_var.irMetadata[-1].shape[0] if w_var.irMetadata else None
            if not isinstance(out_channels, int):
                continue
            bias_var, _ = peel(bias)
            per_channel = IrTorchSlice(IrTorchReshape(copy.copy(bias_var), [out_channels, -1]), [':', '0'])
            biased = IrFConv2d(center.children[0], center.children[1], center.stride, center.padding,
                               per_channel)
            out = copy.copy(interval)
            out.update_parent_child([_rebuild(c_shells, biased), interval.children[1]])
            folded += 1
            return out
        return None

    def rewrite(expr):
        new = match(expr)
        if new is not None:
            return new
        if not getattr(expr, 'children', None):
            return expr
        out = copy.copy(expr)
        out.update_parent_child([rewrite(c) for c in expr.children])
        return out

    for stmt in block.children:
        if isinstance(stmt, IrAssignment):
            stmt.update_parent_child([stmt.children[0], rewrite(stmt.children[1])])
            definitions[stmt.children[0].name] = stmt.children[1]
        else:
            stmt.update_parent_child([rewrite(c) for c in stmt.children])
    if folded:
        while subexp_inlining.drop_dead_assignments(block.children):
            pass
    return folded
