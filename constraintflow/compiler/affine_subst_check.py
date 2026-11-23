"""Decide from the .cf source which affine-layer shortcuts are sound for a program.

fuse: while back-substitution crosses an Affine (or program-defined Add) layer, c+ * n[L] + c- * n[U] becomes
    c * n[L]. Exact when every Affine and Neuron_add return writes the same expression for the shape's lower and
    upper PolyExp members.
skip: an affine layer whose consumers are all affine gets l = u = 0 and no concretizing traversals (Affine_skip).
    Sound when the Affine and Neuron_add rules read a neuron's Float members only through .map() over a traversal
    that never stops before the input (stop function `false`).
"""
from constraintflow.ast_cflow import astcf as AST

RULES = ('Affine', 'Neuron_add')


def _flatten(stmt):
    if isinstance(stmt, AST.SeqNode):
        return _flatten(stmt.stmt1) + _flatten(stmt.stmt2)
    return [stmt]


def _key(node):
    """Structural form of a subtree, for comparing two expressions."""
    if isinstance(node, (list, tuple)):
        return tuple(_key(x) for x in node)
    if not isinstance(node, AST.ASTNode):
        return node
    return (type(node).__name__,) + tuple((k, _key(v)) for k, v in sorted(vars(node).items())
                                          if k not in ('node_name', 'left', 'right') or isinstance(node, (AST.BinOpNode, AST.DotNode, AST.ConcatNode)))


def _leaves(ret):
    if isinstance(ret, AST.TransRetIfNode):
        return _leaves(ret.tret) + _leaves(ret.fret)
    return [ret.exprlist.exprlist]


class _Program:
    def __init__(self, ast):
        self.members = [(t.name, v.name) for t, v in ast.shape.elements.arglist]
        self.floats = {name for t, name in self.members if t == 'Float'}
        self.funcs = {}
        self.rules = []
        for stmt in _flatten(ast.stmt):
            if isinstance(stmt, AST.FuncNode):
                self.funcs[stmt.decl.name.name] = stmt.expr
            elif isinstance(stmt, AST.TransformerNode):
                self.rules += [op for op in stmt.oplist.olist if op.op.op_name in RULES]
        poly = {name for t, name in self.members if t == 'PolyExp'}
        bounds = {'<=': [], '>=': []}
        for c in ast.shape.p.exprlist:
            if (isinstance(c, AST.BinOpNode) and c.op in bounds and isinstance(c.left, AST.GetElementNode)
                    and c.left.elem.name in poly):
                bounds[c.op].append(c.left.elem.name)
        names = [name for _, name in self.members]
        self.lu = (names.index(bounds['<='][0]), names.index(bounds['>='][0])) \
            if len(bounds['<=']) == len(bounds['>=']) == 1 else None

    def _const_false(self, f):
        name = f.name.name if isinstance(f, AST.FuncCallNode) else getattr(f, 'name', None)
        body = self.funcs.get(name)
        return isinstance(body, AST.ConstBoolNode) and body.value is False

    def reads_float(self, node, seen=()):
        """Does `node` read a Float member (l, u, ...) of some neuron, other than through .map() over a traversal
        that always reaches the input?"""
        if isinstance(node, AST.MapNode) and isinstance(node.expr, AST.TraverseNode) and self._const_false(node.expr.stop):
            t = node.expr
            return (self.reads_float(t.expr, seen) or self._func_reads(t.func, seen)
                    or self._func_reads(t.priority, seen))
        if isinstance(node, AST.MapNode):
            return self.reads_float(node.expr, seen) or self._func_reads(node.func, seen)
        if isinstance(node, AST.TraverseNode):
            return True  # its result is concretized somewhere we can't see, or it may stop early
        if isinstance(node, AST.GetElementNode) and getattr(node.elem, 'name', None) in self.floats:
            return True
        if isinstance(node, AST.FuncCallNode) and self._func_reads(node.name, seen):
            return True
        if isinstance(node, (list, tuple)):
            return any(self.reads_float(x, seen) for x in node)
        if isinstance(node, AST.ASTNode):
            return any(self.reads_float(v, seen) for k, v in vars(node).items() if k not in ('left', 'right')
                       or isinstance(node, (AST.BinOpNode, AST.DotNode, AST.ConcatNode)))
        return False

    def _func_reads(self, f, seen):
        name = getattr(f, 'name', None)
        if name not in self.funcs or name in seen:
            return False
        return self.reads_float(self.funcs[name], seen + (name,))


def check(ast):
    """Return (fuse, skip, reason): reason says what blocked a shortcut ('' when both are on)."""
    prog = _Program(ast)
    leaves = [(op.op.op_name, exprs) for op in prog.rules for exprs in _leaves(op.ret)]
    if not any(name == 'Affine' for name, _ in leaves):
        return False, False, 'no Affine rule'
    if prog.lu is None:
        return False, False, 'no lower/upper PolyExp pair in the shape'
    lo, up = prog.lu
    for name, exprs in leaves:
        if len(exprs) != len(prog.members) or _key(exprs[lo]) != _key(exprs[up]):
            return False, False, f'{name} gives different lower and upper polyhedral expressions'
    for name, exprs in leaves:
        if prog.reads_float(exprs):
            return True, False, f'{name} reads the concrete bounds of its inputs'
    if len(prog.members) != 4:
        return True, False, 'Affine_skip needs an (l, u, L, U) shape'
    return True, True, ''
