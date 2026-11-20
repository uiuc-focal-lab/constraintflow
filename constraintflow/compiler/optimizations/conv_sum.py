# Optimization pass, always on in jit builds.
"""Share a fixed transpose convolution across a sum of compatible inputs.

C(x,W) + C(y,W) = C(x+y,W). Only absorb single-use convolution results,
with identical weight SSA values, options, and output view chains. This is
useful after factoring a position-dependent patch into row scales and W.
"""
import copy
from collections import Counter
from constraintflow.compiler.ir import (
    IrAssignment, IrVar, IrTransRetBasic, IrFConvTranspose2d, IrSimpleBinary, IrTorchSlice, IrTorchCat,
)
from constraintflow.compiler.optimizations import subexp_inlining
from constraintflow.compiler.optimizations.sign_convs import _WRAPPERS, _rebuild
from constraintflow.compiler.optimizations.sign_reductions import _op


def run(block):
    definitions, counts = {}, Counter()
    for s in block.children:
        if isinstance(s, IrAssignment):
            definitions[s.children[0].name] = s.children[1]
            exprs = [s.children[1]]
        else:
            exprs = s.children
        counts.update(v.name for v in subexp_inlining.get_vars_expr_occurrences(exprs))

    def resolve(e, single=False):
        while isinstance(e, IrVar) and e.name in definitions and (not single or counts[e.name] == 1):
            e = definitions[e.name]
        return e

    def candidate(e):
        shells = []
        e = resolve(e, True)
        while type(e) in _WRAPPERS:
            shells.append(e)
            e = resolve(e.children[0], True)
        if not isinstance(e, IrFConvTranspose2d):
            return None
        w = resolve(e.children[1])
        # Avoid expensive structural comparisons of arbitrary weight programs.
        def weight_key(w):
            w = resolve(w)
            if isinstance(w, IrVar):
                return ('var', w.name)
            if isinstance(w, IrTorchSlice):
                return ('slice', repr(w.index), weight_key(w.children[0]))
            return ('identity', id(w))
        key = weight_key(w)
        def option(value):
            if isinstance(value, int):
                return (value, value)
            return tuple(value or ())
        opts = tuple(option(getattr(e,k)) for k in ('stride','padding','output_padding'))
        views = tuple((type(s), tuple(str(getattr(s,k)) for k in _WRAPPERS[type(s)])) for s in shells)
        return (key,opts,views), e, shells

    count = 0

    def merge(left, right):
        nonlocal count
        a,b = candidate(left),candidate(right)
        if a is not None and b is not None and a[0] == b[0]:
            x,y = a[1],b[1]
            # Cropped strided convolutions can have equal output shapes with
            # different input shapes, so require identical input reshapes.
            ix,iy = resolve(x.children[0]),resolve(y.children[0])
            from constraintflow.compiler.ir import IrTorchReshape, IrTorchView
            if type(ix) not in (IrTorchReshape,IrTorchView) or type(iy) not in (IrTorchReshape,IrTorchView):
                return None
            if tuple(ix.shape) != tuple(iy.shape):
                return None
            out = copy.copy(x)
            out.update_parent_child([IrSimpleBinary(x.children[0],y.children[0],'add'),x.children[1]])
            count += 1
            return _rebuild(a[2],out)

        def cat(e):
            shells=[]
            e=resolve(e,True)
            while type(e) in _WRAPPERS:
                shells.append(e)
                e=resolve(e.children[0],True)
            key=tuple((type(s),tuple(str(getattr(s,k)) for k in _WRAPPERS[type(s)])) for s in shells)
            return e,shells,key
        x,xs,xkey=cat(left)
        y,ys,ykey=cat(right)
        if not isinstance(x,IrTorchCat) or not isinstance(y,IrTorchCat):
            return None
        if xkey != ykey or x.dim != y.dim or len(x.children) != len(y.children):
            return None
        # Only distribute through concatenations if every pair is proved to
        # merge; this also proves equal component shapes (no broadcasting).
        previous=count
        pieces=[]
        for l,r in zip(x.children,y.children):
            piece=merge(l,r)
            if piece is None:
                count=previous
                return None
            pieces.append(piece)
        return _rebuild(xs,IrTorchCat(pieces,x.dim))

    def rewrite(e):
        e = copy.copy(e)
        if not getattr(e,'children',None):
            return e
        e.update_parent_child([rewrite(c) for c in e.children])
        if not _op(e,'add'):
            return e
        out=merge(*e.children)
        return out if out is not None else e

    for stmt in block.children:
        if isinstance(stmt,IrAssignment):
            lhs,rhs=stmt.children
            rhs=rewrite(rhs)
            stmt.update_parent_child([lhs,rhs])
            definitions[lhs.name]=rhs
        else:
            stmt.update_parent_child([rewrite(c) for c in stmt.children])
    if count:
        subexp_inlining.inline_subexp_block(block)
    return count
