# Optimization pass, always on in jit builds.
"""Factor row-scaled convolution patches before composing them.

A @ (diag(s) @ W) = (A @ diag(s)) @ W.  The captured implementation
unfolds every coefficient of diag(s) @ W and contracts/folds the result.
Recognize its exact layout and unfold only s, then use conv_transpose2d
with W. No traversal order or relaxation rule changes. Reassociation can
change floating-point rounding. Unknown layouts are left untouched.
"""
import copy
import math
from types import SimpleNamespace

from constraintflow.compiler.ir import (
    IrAssignment, IrVar, IrConst, IrTorchExpand, IrTorchReshape, IrTorchView,
    IrTorchPermute, IrTorchRepeat, IrTorchUnsqueeze, IrTorchWhere, IrTorchZeros,
    IrSimpleBinary, IrFFold, IrFUnfold, IrTorchEinsum, IrFConvTranspose2d,
    IrTorchTranspose, IrTorchMatmul, IrTorchSlice,
)
from constraintflow.compiler.optimizations.product_grids import _meta_env
from constraintflow.compiler.optimizations import subexp_inlining


def run(block, render, sized, batch_size):
    env = _meta_env(sized, batch_size)
    env['device_mode'] = SimpleNamespace(get_device=lambda: 'meta')
    definitions = {s.children[0].name: s.children[1] for s in block.children
                   if isinstance(s, IrAssignment) and isinstance(s.children[0], IrVar)}

    def resolve(e):
        while isinstance(e, IrVar) and e.name in definitions:
            e = definitions[e.name]
        return e

    shapes = {}

    def shape(e):
        if id(e) in shapes:
            return shapes[id(e)][1]
        try:
            with env['torch'].device('meta'):
                result = tuple(eval(render(e), dict(env)).shape)
                shapes[id(e)] = (e, result)
                return result
        except Exception:
            return None

    def node(e, cls):
        e = resolve(e)
        return e if isinstance(e, cls) else None

    def mul(a, b):
        return IrSimpleBinary(a, b, 'mul')

    def factor(e, B, R, inner_shape):
        """Return (fixed kernel or None for zero, [B,R,1] row scale)."""
        if shape(e) not in ((B, R, math.prod(inner_shape[1:])),
                            (1, R, math.prod(inner_shape[1:]))):
            return None
        def scalar(e):
            e = resolve(e)
            while isinstance(e, IrTorchExpand):
                e = resolve(e.children[0])
            s = shape(e)
            if s and len(s) == 3 and s[0] in (1, B) and s[1] in (1, R) and s[2] == 1:
                return IrTorchExpand(e, [B, R, 1])
            return None

        def constant(value):
            z = IrTorchZeros([B, R, 1], device='device_mode.get_device()')
            return z if value == 0 else IrSimpleBinary(z, IrConst(value, 'Float'), 'add')

        def visit(e):
            e = resolve(e)
            while isinstance(e, IrTorchExpand):
                e = resolve(e.children[0])
            if isinstance(e, IrTorchZeros):
                return None, constant(0)
            if isinstance(e, IrSimpleBinary) and e.op in ('mul',):
                for i in (0, 1):
                    s = scalar(e.children[i])
                    if s is not None:
                        f = visit(e.children[1-i])
                        if f is not None:
                            return f[0], mul(s, f[1])
            if isinstance(e, IrTorchWhere):
                c = scalar(e.children[0])
                if c is None:
                    return None
                a, b = visit(e.children[1]), visit(e.children[2])
                if a is None or b is None:
                    return None
                if a[0] is not None and b[0] is not None and render(a[0]) != render(b[0]):
                    return None
                return a[0] if a[0] is not None else b[0], IrTorchWhere(c, a[1], b[1])
            # W.view(1,K,-1).unsqueeze(-1).repeat(1,1,1,H*W)
            #  .permute(0,1,3,2).reshape(1,K*H*W,-1)
            r = node(e, (IrTorchReshape, IrTorchView))
            if r is None:
                return None
            p = node(r.children[0], IrTorchPermute)
            if p is None or list(p.permutation) != [0, 1, 3, 2]:
                return None
            rep = node(p.children[0], IrTorchRepeat)
            if rep is None or list(rep.repeats) != [1, 1, 1, R // inner_shape[0]]:
                return None
            u = node(rep.children[0], IrTorchUnsqueeze)
            if u is None or u.index not in (-1, 3):
                return None
            v = node(u.children[0], (IrTorchView, IrTorchReshape))
            if v is None or list(v.shape) != [1, inner_shape[0], -1]:
                return None
            w = v.children[0]
            if shape(w) != inner_shape or shape(r) != (1, R, math.prod(inner_shape[1:])):
                return None
            return w, constant(1)
        return visit(e)

    def channel_slice(e, axis):
        """Strip only a static contiguous slice of flattened input channels."""
        e = resolve(e)
        if not isinstance(e, IrTorchSlice):
            return e, None
        index = e.index
        if len(index) != axis + 1 or any(x != 0 for x in index[:axis]):
            return None, None
        bounds = index[axis]
        if not isinstance(bounds, list) or len(bounds) != 2 or not all(type(x) is int for x in bounds):
            return None, None
        lo, hi = bounds
        if not 0 <= lo < hi:
            return None, None
        return e.children[0], (lo, hi)

    def sliced_factor(source, B, R, Km, full, kh, kw, bounds):
        area = kh * kw
        if full % area:
            return None
        fact = factor(source, B, R, (Km, full // area, kh, kw))
        if fact is None or fact[0] is None:
            return None
        weight, scale = fact
        if bounds is not None:
            lo, hi = bounds
            if lo % area or hi % area or hi > full:
                return None
            weight = IrTorchSlice(weight, [0, [lo // area, hi // area]])
        return weight, scale

    count = 0

    def match_dense(e):
        nonlocal count
        f = node(e, IrFFold)
        if f is None:
            return None
        out = node(f.children[0], IrTorchReshape)
        if out is None or len(out.shape) != 3:
            return None
        perm = node(out.children[0], IrTorchPermute)
        if perm is None or list(perm.permutation) != [0,2,3,1]:
            return None
        tr = node(perm.children[0], IrTorchReshape)
        if tr is None or len(tr.shape) != 4:
            return None
        B,S,M,P = tr.shape
        mm = node(tr.children[0], IrTorchMatmul)
        if mm is None:
            return None
        cr,pr = (node(c, IrTorchReshape) for c in mm.children)
        if cr is None or pr is None or len(cr.shape) != 3:
            return None
        Km = cr.shape[-1]
        if list(cr.shape) != [B*S,M,Km] or list(pr.shape) != [B*S,Km,P]:
            return None
        base, bounds = channel_slice(pr.children[0], 3)
        cp,pp = node(cr.children[0],IrTorchPermute),node(base,IrTorchPermute)
        if cp is None or list(cp.permutation) != [0,3,1,2]:
            return None
        if pp is None or list(pp.permutation) != [0,2,1,3]:
            return None
        ci,pi = node(cp.children[0],IrTorchReshape),node(pp.children[0],IrTorchReshape)
        if ci is None or list(ci.shape) != [B,M,Km,S]:
            return None
        if pi is None or len(pi.shape) != 4 or list(pi.shape[:3]) != [B,Km,S]:
            return None
        kh,kw=f.kernel_size
        if P % (kh*kw) or list(out.shape) != [B*M,P,S]:
            return None
        Ci=P//(kh*kw)
        padding=f.padding or [0,0]
        ih,iw=f.output_size
        H=(ih+2*padding[0]-kh)//f.stride[0]+1
        W=(iw+2*padding[1]-kw)//f.stride[1]+1
        if H*W != S:
            return None
        full = pi.shape[-1]
        if P != (bounds[1]-bounds[0] if bounds else full):
            return None
        fact=sliced_factor(pi.children[0],B,Km*S,Km,full,kh,kw,bounds)
        if fact is None or fact[0] is None:
            return None
        weight,scale=fact
        a=IrTorchReshape(ci.children[0],[B,M,Km*S])
        d=IrTorchReshape(scale,[B,1,Km*S])
        scaled=IrTorchReshape(mul(a,d),[B*M,Km,H,W])
        output_padding=[(ih+2*padding[0]-kh)%f.stride[0],
                        (iw+2*padding[1]-kw)%f.stride[1]]
        count+=1
        return IrFConvTranspose2d(scaled,weight,f.stride,padding,output_padding)

    def match(e):
        nonlocal count
        f = node(e, IrFFold)
        if f is None or (f.padding not in (None, 0) and (not isinstance(f.padding, (list, tuple)) or any(f.padding))):
            return None
        hr = node(f.children[0], IrTorchReshape)
        if hr is None:
            return None
        hp = node(hr.children[0], IrTorchPermute)
        if hp is None or list(hp.permutation) != [0, 2, 5, 1, 3, 4]:
            return None
        contraction = node(hp.children[0], IrTorchEinsum)
        if contraction is None or contraction.equation not in ('pckabo,mkab->pcmabo', 'ncmabo,nkmabo->nckabo'):
            return None
        ur = node(contraction.children[0], IrTorchReshape)
        if ur is None or len(ur.shape) != 6:
            return None
        B, P, Km, kh, kw, O = ur.shape
        if not all(isinstance(x, int) and x > 0 for x in (B,P,Km,kh,kw,O)):
            return None
        unf = node(ur.children[0], IrFUnfold)
        if unf is None or list(unf.kernel_size) != [kh, kw]:
            return None
        mr = node(unf.children[0], IrTorchReshape)
        if mr is None or len(mr.shape) != 4 or list(mr.shape[:2]) != [B*P, Km]:
            return None
        H, W = mr.shape[2:]
        base, bounds = channel_slice(mr.children[0], 1)
        mp = node(base, IrTorchPermute)
        if mp is None or list(mp.permutation) != [0,4,1,2,3]:
            return None
        pr = node(mp.children[0], IrTorchReshape)
        if pr is None or len(pr.shape) != 5 or list(pr.shape[:4]) != [B,Km,H,W]:
            return None
        ikh, ikw = f.kernel_size
        if P % (ikh*ikw):
            return None
        Ci = P // (ikh*ikw)
        full = pr.shape[-1]
        if P != (bounds[1]-bounds[0] if bounds else full):
            return None
        fact = sliced_factor(pr.children[0], B, Km*H*W, Km, full, ikh, ikw, bounds)
        if fact is None or fact[0] is None:
            return None
        weight, scale = fact
        outkh = (kh-1)*f.stride[0]+ikh
        outkw = (kw-1)*f.stride[1]+ikw
        if list(f.output_size) != [outkh,outkw]:
            return None
        rhs = contraction.children[1]
        if contraction.equation == 'pckabo,mkab->pcmabo':
            ws = shape(rhs)
            if ws is None or len(ws) != 4 or ws[1:] != (Km,kh,kw):
                return None
            K = ws[0]
            outer = IrTorchReshape(rhs, [1,K,1,Km*kh*kw])
        else:
            wp = node(rhs, IrTorchPermute)
            if wp is None or list(wp.permutation) != [0,1,3,4,5,2]:
                return None
            wr = node(wp.children[0], IrTorchReshape)
            if wr is None or len(wr.shape) != 6:
                return None
            K = wr.shape[1]
            if list(wr.shape) != [B,K,O,Km,kh,kw]:
                return None
            outer = IrTorchReshape(wr.children[0], [B,K,O,Km*kh*kw])
        if list(hr.shape) != [B*K*O,P,kh*kw]:
            return None
        s = IrFUnfold(IrTorchReshape(scale,[B,Km,H,W]), [kh,kw], unf.padding, unf.stride)
        s = IrTorchUnsqueeze(IrTorchTranspose(s,1,2),1)
        scaled = IrTorchReshape(mul(outer,s),[B*K*O,Km,kh,kw])
        result = IrFConvTranspose2d(scaled,weight,f.stride,[0,0],[0,0])
        count += 1
        return result

    def rewrite(e):
        replacement = match(e)
        if replacement is None:
            replacement = match_dense(e)
        if replacement is not None:
            return replacement
        if not getattr(e, 'children', None):
            return e
        out = copy.copy(e)
        out.update_parent_child([rewrite(c) for c in e.children])
        return out

    for stmt in block.children:
        if isinstance(stmt, IrAssignment):
            lhs, rhs = stmt.children
            rhs = rewrite(rhs)
            stmt.update_parent_child([lhs, rhs])
            definitions[lhs.name] = rhs
        else:
            stmt.update_parent_child([rewrite(c) for c in stmt.children])
    if count:
        subexp_inlining.inline_subexp_block(block)
    return count
