import copy

from constraintflow.ast_cflow import astcf as AST

AFFINE_SKIP_OP = 'Affine_skip'


def _affine_op_stmts(stmt):
    """Yield every Affine OpStmtNode reachable from a Program's statement tree."""
    if isinstance(stmt, AST.SeqNode):
        yield from _affine_op_stmts(stmt.stmt1)
        yield from _affine_op_stmts(stmt.stmt2)
    elif isinstance(stmt, AST.TransformerNode):
        for op_stmt in stmt.oplist.olist:
            if op_stmt.op.op_name == 'Affine':
                yield stmt, op_stmt


def _make_affine_skip_return(ret):
    """Copy an Affine return while replacing every leaf's concrete bounds.

    Transformer returns may be a direct ``(l, u, L, U)`` tuple or a conditional
    tree whose leaves are such tuples (CROWN-IBP uses the latter). Affine_skip
    must preserve the conditional structure because its branches can produce
    different symbolic expressions, but neither branch needs to concretize
    those expressions at a layer whose consumers are all affine.
    """
    if isinstance(ret, AST.TransRetBasicNode):
        exprs = ret.exprlist.exprlist
        if len(exprs) != 4:
            raise ValueError(
                "inject_affine_skip: every Affine return branch must have 4 "
                f"components (l, u, L, U), got {len(exprs)}."
            )
        _, _, l_poly, u_poly = exprs
        return AST.TransRetBasicNode(AST.ExprListNode([
            AST.ConstIntNode(0), AST.ConstIntNode(0),
            copy.deepcopy(l_poly), copy.deepcopy(u_poly),
        ]))

    if isinstance(ret, AST.TransRetIfNode):
        return AST.TransRetIfNode(
            copy.deepcopy(ret.cond),
            _make_affine_skip_return(ret.tret),
            _make_affine_skip_return(ret.fret),
        )

    raise ValueError(
        "inject_affine_skip: Affine op's return must be an (l, u, L, U) "
        f"tuple or a conditional of such tuples, not {type(ret).__name__}."
    )


def inject_affine_skip(ast_program):
    """Add an Affine-shaped Affine_skip op alongside the original (untouched)
    Affine op -- flow_sparse.py's Flow.flow() dispatches a Linear/Conv2D layer to
    it whenever layer.feeds_nonlin is False (every immediate consumer is itself
    an Affine layer; computed in parse.py). Leaving the original Affine op
    untouched means the common case (a layer feeding a nonlinearity) is
    byte-identical to the unflagged generic CFG that tensor_to_block.py
    specializes -- only the rare feeds_nonlin=False layers get their own (tiny)
    CFG.

    Injected when affine_subst_check finds it sound (stop_traverse=false and
    no direct reads of the inputs' l/u), a hidden affine layer's concrete l/u are consumed
    only by a following nonlinearity's relaxation, so an affine feeding only
    further affines has zero consumers of l/u. Both concretizing traversals are
    therefore dead work. Only the L/U poly parts (needed for the next layer's
    own substitution) and any return-branching structure are kept unchanged.
    """
    for transformer_node, op_stmt in list(_affine_op_stmts(ast_program.stmt)):
        skip_ret = _make_affine_skip_return(op_stmt.ret)

        transformer_node.oplist.olist.append(
            AST.OpStmtNode(AST.OperatorNode(AFFINE_SKIP_OP), skip_ret))
