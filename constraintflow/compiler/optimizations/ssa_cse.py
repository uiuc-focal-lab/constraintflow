#FEAT: new pass
"""Hash-bucket CSE for a functional straight-line SSA block.

Reuse the existing CSE equality checks, but emit common definitions at first use
in one walk instead of rescanning the block for every shared expression.
"""
import copy
from collections import Counter
from constraintflow.compiler.ir import IrAssignment, IrVar, IrConst, IrPhi, IrTransRetBasic
from constraintflow.compiler.optimizations import cse


def run(block):
    def leaf(e):
        return e is None or isinstance(e,(int,float,str,IrVar,IrConst,IrPhi))
    names=set()
    for stmt in block.children:
        if isinstance(stmt,IrAssignment) and isinstance(stmt.children[0],IrVar):
            name=stmt.children[0].name
            if name in names:raise ValueError('SSA CSE requires unique definitions')
            names.add(name)
        elif not isinstance(stmt,IrTransRetBasic):
            raise ValueError('SSA CSE requires a functional straight-line block')
    table=cse._Table()
    representatives={}
    counts=Counter()
    def collect(e):
        if isinstance(e,IrVar):names.add(e.name)
        if leaf(e):return
        rep,_=table.find_or_add(e)
        representatives[id(e)]=rep
        counts[id(rep)]+=1
        for child in e.children:collect(child)
    for stmt in block.children:
        for expr in (stmt.children[1:] if isinstance(stmt,IrAssignment) else stmt.children):collect(expr)
    emitted={}
    output=[]
    def rewrite(e):
        if leaf(e):return e
        rep=representatives[id(e)]
        key=id(rep)
        if key in emitted:return emitted[key]
        out=copy.copy(e)
        out.update_parent_child([rewrite(child) for child in e.children])
        if counts[key]>1:
            name=cse.get_var()
            while name in names:name=cse.get_var()
            names.add(name)
            var=IrVar(name,e.irMetadata)
            assignment=IrAssignment(var,out)
            var.defs=assignment
            output.append(assignment)
            emitted[key]=var
            return var
        return out
    for stmt in block.children:
        if isinstance(stmt,IrAssignment):
            lhs,rhs=stmt.children
            rhs=rewrite(rhs)
            stmt.update_parent_child([lhs,rhs])
        else:stmt.update_parent_child([rewrite(e) for e in stmt.children])
        output.append(stmt)
    block.update_parent_child(output)
    return len(emitted)
