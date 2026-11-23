# Optimization pass, always on in jit builds.
"""Recombine channel chunks made unnecessary by patch composition factoring.

Concatenated output-channel slices of C(x,W) are C(x,W). Accept only the
captured batch/row reshape layout, equal inputs/options, and ordered slices
covering the complete kernel. Preserve the convolution and channel order.
"""
import copy,math
from collections import Counter
from types import SimpleNamespace
from constraintflow.compiler.ir import (IrAssignment,IrVar,IrTorchReshape,
    IrTorchView,IrTorchCat,IrTorchSlice,IrFConvTranspose2d)
from constraintflow.compiler.optimizations import subexp_inlining
from constraintflow.compiler.optimizations.product_grids import _meta_env


def run(block,render,sized,batch_size):
    env=_meta_env(sized,batch_size)
    env['device_mode']=SimpleNamespace(get_device=lambda:'meta')
    definitions={s.children[0].name:s.children[1] for s in block.children if isinstance(s,IrAssignment)}
    counts=Counter(v.name for s in block.children for v in
                   subexp_inlining.get_vars_expr_occurrences(s.children[1:] if isinstance(s,IrAssignment) else s.children))
    def resolve(e,single=False):
        while isinstance(e,IrVar) and e.name in definitions and (not single or counts[e.name]==1):
            e=definitions[e.name]
        return e
    shapes={}
    def shape(e):
        key=id(e)
        if key in shapes:return shapes[key][1]
        # The captured block has already been shape-probed. Reuse explicit
        # reshape dimensions and named-tensor metadata instead of executing
        # every convolution once again for each output-channel slice.
        if isinstance(e,(IrTorchReshape,IrTorchView)) and all(type(x) is int and x>0 for x in e.shape):
            value=tuple(e.shape);shapes[key]=(e,value);return value
        if isinstance(e,IrVar) and e.name in env and hasattr(env[e.name],'shape'):
            value=tuple(env[e.name].shape);shapes[key]=(e,value);return value
        try:
            with env['torch'].device('meta'):
                value=tuple(eval(render(e),dict(env)).shape)
            shapes[key]=(e,value)
            return value
        except Exception:return None
    def option(v):return tuple(v) if isinstance(v,(list,tuple)) else (v,v)
    removed=0
    def match(e):
        nonlocal removed
        if not isinstance(e,IrTorchCat) or e.dim not in (2,-1) or len(e.children)<2:return None
        common=None;end=0
        for child in e.children:
            r=resolve(child,True)
            if not isinstance(r,(IrTorchReshape,IrTorchView)):return None
            conv=resolve(r.children[0],True)
            if not isinstance(conv,IrFConvTranspose2d) or len(conv.children)!=2:return None
            ws=resolve(conv.children[1])
            if not isinstance(ws,IrTorchSlice) or len(ws.index)!=2 or ws.index[0]!=0:return None
            bounds=ws.index[1]
            if not isinstance(bounds,list) or len(bounds)!=2 or not all(type(v) is int for v in bounds):return None
            lo,hi=bounds
            if lo!=end or hi<=lo:return None
            weight=ws.children[0]
            wshape=shape(weight);ishape=shape(conv.children[0]);rshape=shape(r)
            if wshape is None or len(wshape)!=4 or ishape is None or len(ishape)!=4:return None
            if ishape[1]!=wshape[0]:return None
            stride=option(conv.stride) if conv.stride is not None else (1,1)
            padding=option(conv.padding) if conv.padding is not None else (0,0)
            opad=option(conv.output_padding) if conv.output_padding is not None else (0,0)
            if any(len(x)!=2 or not all(type(v) is int for v in x) for x in (stride,padding,opad)):return None
            cshape=(ishape[0],hi-lo,*((ishape[d+2]-1)*stride[d]-2*padding[d]+wshape[d+2]+opad[d] for d in range(2)))
            if min(cshape)<=0:return None
            if rshape is None or len(rshape)!=3 or rshape[0]*rshape[1]!=cshape[0]:return None
            if cshape[1]!=hi-lo or rshape[2]!=(hi-lo)*math.prod(cshape[2:]):return None
            key=(render(weight),wshape,render(conv.children[0]),
                 tuple(option(getattr(conv,k)) for k in ('stride','padding','output_padding')),
                 rshape[:2],cshape[0],cshape[2:])
            if common is None:common=(key,conv,weight,rshape)
            elif key!=common[0]:return None
            end=hi
        key,conv,weight,rshape=common
        if end!=key[1][1]:return None
        result=copy.copy(conv)
        result.update_parent_child([conv.children[0],weight])
        removed+=len(e.children)-1
        return IrTorchReshape(result,[*rshape[:2],end*math.prod(key[-1])])
    def rewrite(e):
        if not getattr(e,'children',None):return e
        out=copy.copy(e);out.update_parent_child([rewrite(c) for c in e.children])
        result=match(out)
        return result if result is not None else out
    for stmt in block.children:
        if isinstance(stmt,IrAssignment):
            lhs,rhs=stmt.children;rhs=rewrite(rhs)
            stmt.update_parent_child([lhs,rhs]);definitions[lhs.name]=rhs
        else:stmt.update_parent_child([rewrite(c) for c in stmt.children])
    if removed:subexp_inlining.inline_subexp_block(block)
    return removed
