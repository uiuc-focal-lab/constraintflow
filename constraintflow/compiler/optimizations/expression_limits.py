# Optimization pass, always on in jit builds.
"""Bound Python expression nesting by exposing ordered SSA temporaries.

No operator is changed or reassociated. The pass applies to functional tensor
blocks before rendering; subsequent algebraic rewrites may require it again.
"""
import copy
from constraintflow.compiler.ir import IrAssignment,IrVar,IrTransRetBasic
from constraintflow.compiler.optimizations import subexp_inlining


def run(block,max_depth=64,layers=None):
    if max_depth<2:raise ValueError('Expression depth limit must be at least two')
    occupied={v.name for s in block.children for v in subexp_inlining.get_vars_expr_occurrences(s.children)}
    memo={}
    def depth(e):
        if not getattr(e,'children',None):return 1
        k=id(e)
        if k not in memo:memo[k]=1+max((depth(c) for c in e.children),default=0)
        return memo[k]
    output=[];count=0
    layer=None
    def materialize(e):
        nonlocal count
        name='expr_split_'+str(count)
        while name in occupied:
            count+=1;name='expr_split_'+str(count)
        count+=1;occupied.add(name)
        var=IrVar(name,e.irMetadata)
        assignment=IrAssignment(var,e);var.defs=assignment
        output.append(assignment)
        if layers is not None and layer is not None:layers[id(assignment)]=(assignment,layer[1])
        return var
    def rewrite(e):
        d=depth(e)
        if d<=max_depth:return e,d
        out=copy.copy(e);children=[];heights=[]
        for child in e.children:
            child,h=rewrite(child)
            if h>=max_depth:child=materialize(child);h=1
            children.append(child);heights.append(h)
        out.update_parent_child(children)
        return out,1+max(heights,default=0)
    before=len(block.children)
    for stmt in block.children:
        layer=layers.get(id(stmt)) if layers is not None else None
        if isinstance(stmt,IrAssignment):
            lhs,rhs=stmt.children;rhs,_=rewrite(rhs)
            stmt.update_parent_child([lhs,rhs])
        elif isinstance(stmt,IrTransRetBasic):stmt.update_parent_child([rewrite(c)[0] for c in stmt.children])
        else:raise ValueError('Expression splitting requires a functional tensor block')
        output.append(stmt)
    block.update_parent_child(output)
    return len(output)-before
