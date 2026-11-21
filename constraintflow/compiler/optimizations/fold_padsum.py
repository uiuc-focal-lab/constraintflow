# Optimization pass, run in jit builds for --device cpu (not yet measured on the GPU).
"""Compose kernel x patches without a scatter.

KernelBlock @ PatchesBlock is captured as
    F.fold(einsum('pckabo,mkab->pcmabo', U, W).permute(0,2,5,1,3,4).reshape(B*K*O, P, kh*kw),
           output_size=(RKX, RKY), kernel_size=(ikh, ikw), stride=(sx, sy))
F.fold places kernel tap (a, b)'s (Ci, ikh, ikw) patch at offset (a*sx, b*sy) of the (RKX, RKY)
result patch and adds overlaps: a scatter-add. The same sum is a gather: pad each tap's slice of
the einsum output to (RKX, RKY) at its offset and add the kh*kw padded slices, with the B*K*O rows
as the contiguous dimension. Same values (up to summation order), no col2im. Runs after
patch_factor, so compositions it factors into conv_transpose2d are left alone.
"""
import copy

from constraintflow.compiler.ir import (IrAssignment, IrVar, IrTorchReshape, IrTorchPermute,
                                        IrTorchEinsum, IrFFold, IrTorchSlice, IrTorchPad, IrSimpleBinary)


def run(block):
    definitions = {s.children[0].name: s.children[1] for s in block.children
                   if isinstance(s, IrAssignment) and isinstance(s.children[0], IrVar)}

    def resolve(e):
        while isinstance(e, IrVar) and e.name in definitions:
            e = definitions[e.name]
        return e

    def node(e, cls):
        e = resolve(e)
        return e if isinstance(e, cls) else None

    count = 0
    pending = []

    def match(e):
        nonlocal count
        if not isinstance(e, IrFFold):
            return None
        if e.padding not in (None, 0) and (not isinstance(e.padding, (list, tuple)) or any(e.padding)):
            return None
        hr = node(e.children[0], IrTorchReshape)
        if hr is None or len(hr.shape) != 3:
            return None
        hp = node(hr.children[0], IrTorchPermute)
        if hp is None or list(hp.permutation) != [0, 2, 5, 1, 3, 4]:
            return None
        ein = node(hp.children[0], IrTorchEinsum)
        if ein is None or ein.equation != 'pckabo,mkab->pcmabo':
            return None
        ur = node(ein.children[0], IrTorchReshape)
        if ur is None or len(ur.shape) != 6 or not all(isinstance(x, int) and x > 0 for x in ur.shape):
            return None
        B, P, Km, kh, kw, O = ur.shape
        BKO, P2, taps = hr.shape
        if P2 != P or taps != kh * kw or BKO % (B * O):
            return None
        K = BKO // (B * O)
        ikh, ikw = e.kernel_size
        sx, sy = e.stride
        RKX, RKY = e.output_size
        if P % (ikh * ikw) or RKX != (kh - 1) * sx + ikh or RKY != (kw - 1) * sy + ikw:
            return None
        Ci = P // (ikh * ikw)
        name = 'fold_padsum_h_%d' % count
        pending.append(IrAssignment(IrVar(name, []), hp.children[0]))
        total = None
        for a in range(kh):
            for b in range(kw):
                t = IrTorchSlice(IrVar(name, []), [0, 0, 0, [a, a + 1], [b, b + 1]])
                t = IrTorchReshape(t, [B, Ci, ikh, ikw, K * O])
                t = IrTorchPad(t, [0, 0, b * sy, RKY - ikw - b * sy, a * sx, RKX - ikh - a * sx])
                total = t if total is None else IrSimpleBinary(total, t, 'add')
        count += 1
        return IrTorchReshape(IrTorchPermute(total, [0, 4, 1, 2, 3]), [B * K * O, Ci, RKX, RKY])

    def rewrite(e):
        replacement = match(e)
        if replacement is not None:
            return replacement
        if not getattr(e, 'children', None):
            return e
        out = copy.copy(e)
        out.update_parent_child([rewrite(c) for c in e.children])
        return out

    new_children = []
    for stmt in block.children:
        pending.clear()
        if isinstance(stmt, IrAssignment):
            lhs, rhs = stmt.children
            stmt.update_parent_child([lhs, rewrite(rhs)])
        else:
            stmt.update_parent_child([rewrite(c) for c in stmt.children])
        new_children.extend(pending)
        new_children.append(stmt)
    if count:
        block.update_parent_child(new_children)
    return count
