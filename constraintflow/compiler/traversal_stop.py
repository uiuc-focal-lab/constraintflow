"""Hoist traversal stops independent of the visited neuron/coefficient.

For a pure map f and row-invariant predicate p,
map(traverse(e, p, subst), f) = p ? map(e, f) : map(traverse(e, false, subst), f).
This is a language-level identity; it makes no assumptions about ReLU or bounds.
"""
import copy
from constraintflow.ast_cflow import astcf as A


def transform(program):
    funcs = {}
    def collect(node):
        if isinstance(node, A.SeqNode):
            collect(node.stmt1)
            collect(node.stmt2)
        elif isinstance(node, A.FuncNode):
            funcs[node.decl.name.name] = node
    collect(program.stmt)

    def expand(node, env, stack=()):
        if isinstance(node, A.VarNode):
            return copy.deepcopy(env.get(node.name, node))
        if isinstance(node, A.FuncCallNode):
            name = node.name.name
            if name not in funcs or name in stack:
                raise ValueError('unresolved or recursive predicate')
            fn = funcs[name]
            args = [expand(x, env, stack) for x in node.arglist.exprlist]
            if len(args) != len(fn.decl.arglist.arglist):
                raise ValueError('partial call inside predicate')
            return expand(fn.expr, dict(zip((v.name for _, v in fn.decl.arglist.arglist), args)), stack + (name,))
        if isinstance(node, A.ASTNode):
            result = copy.copy(node)
            for key, value in vars(node).items():
                setattr(result, key, expand(value, env, stack))
            return result
        if isinstance(node, list):
            return [expand(x, env, stack) for x in node]
        return node

    def references(node):
        if isinstance(node, A.VarNode):
            return {node.name}
        if isinstance(node, A.ASTNode):
            return set().union(*(references(x) for x in vars(node).values()))
        if isinstance(node, list):
            return set().union(*(references(x) for x in node))
        return set()

    false_name = '_compiler_traverse_false'
    while false_name in funcs:
        false_name += '_'
    false_fn = A.FuncNode(A.FuncDeclNode(A.VarNode(false_name), A.ArgListNode([
        (A.BaseTypeNode('Neuron'), A.VarNode('_vertex')),
        (A.BaseTypeNode('Float'), A.VarNode('_coefficient'))])), A.ConstBoolNode(False))
    count = 0
    def visit(node):
        nonlocal count
        if isinstance(node, A.MapNode) and isinstance(node.expr, A.TraverseNode):
            trav = node.expr
            stop = trav.stop
            #FIX FOR: a literal stop (e.g. `false`, provesound examples) has no function name and crashed every compile.
            name = (stop.name if isinstance(stop, A.VarNode) else
                    stop.name.name if isinstance(stop, A.FuncCallNode) else None)
            fn = funcs.get(name)
            args = stop.arglist.exprlist if isinstance(stop, A.FuncCallNode) else []
            if fn is not None and len(fn.decl.arglist.arglist) == len(args) + 2:
                params = [v.name for _, v in fn.decl.arglist.arglist]
                try:
                    predicate = expand(fn.expr, {})
                    invariant = not (references(predicate) & set(params[-2:]))
                except ValueError:
                    invariant = False
                if invariant and not isinstance(predicate, A.ConstBoolNode):
                    predicate = expand(predicate, dict(zip(params, args)))
                    dense = copy.deepcopy(trav)
                    dense.stop = A.VarNode(false_name)
                    count += 1
                    result = A.TernaryNode(predicate, A.MapNode(copy.deepcopy(trav.expr), copy.deepcopy(node.func)),
                                         A.MapNode(dense, copy.deepcopy(node.func)))
                    result.invariant_stop = True
                    return result
        if isinstance(node, A.ASTNode):
            for key, value in list(vars(node).items()):
                setattr(node, key, visit(value))
        elif isinstance(node, list):
            return [visit(x) for x in node]
        return node
    visit(program)
    if count:
        program.stmt = A.SeqNode(false_fn, program.stmt)
    return count
