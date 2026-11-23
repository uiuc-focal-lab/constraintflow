import unittest

from constraintflow.ast_cflow import astcf as AST
from constraintflow.compiler.single_bound import inject_affine_skip


def _basic_return(start):
    return AST.TransRetBasicNode(AST.ExprListNode([
        AST.ConstIntNode(start),
        AST.ConstIntNode(start + 1),
        AST.ConstIntNode(start + 2),
        AST.ConstIntNode(start + 3),
    ]))


class InjectAffineSkipTest(unittest.TestCase):
    def test_preserves_conditional_and_symbolic_returns(self):
        original_ret = AST.TransRetIfNode(
            AST.ConstBoolNode(True),
            _basic_return(10),
            AST.TransRetIfNode(
                AST.ConstBoolNode(False),
                _basic_return(20),
                _basic_return(30),
            ),
        )
        affine = AST.OpStmtNode(AST.OperatorNode("Affine"), original_ret)
        transformer = AST.TransformerNode(
            AST.VarNode("crownibp"), AST.OpListNode([affine]))
        program = AST.ProgramNode(None, transformer)

        inject_affine_skip(program)

        self.assertEqual([op.op.op_name for op in transformer.oplist.olist],
                         ["Affine", "Affine_skip"])
        skip_ret = transformer.oplist.olist[-1].ret
        self.assertIsInstance(skip_ret, AST.TransRetIfNode)
        self.assertIsNot(skip_ret.cond, original_ret.cond)

        leaves = [skip_ret.tret, skip_ret.fret.tret, skip_ret.fret.fret]
        expected_symbolic_values = [(12, 13), (22, 23), (32, 33)]
        for leaf, expected in zip(leaves, expected_symbolic_values):
            exprs = leaf.exprlist.exprlist
            self.assertEqual([expr.value for expr in exprs], [0, 0, *expected])

        # The generic Affine operation remains untouched.
        self.assertEqual(
            [expr.value for expr in original_ret.tret.exprlist.exprlist],
            [10, 11, 12, 13],
        )


if __name__ == "__main__":
    unittest.main()
