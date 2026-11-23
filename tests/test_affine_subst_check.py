import os
import unittest

import antlr4 as antlr

from constraintflow.ast_cflow import dslLexer, dslParser, astBuilder
from constraintflow.compiler.affine_subst_check import check

EXAMPLES = os.path.join(os.path.dirname(__file__), '..', 'examples', 'compiler_examples')


def run(name):
    parser = dslParser.dslParser(antlr.CommonTokenStream(dslLexer.dslLexer(antlr.FileStream(os.path.join(EXAMPLES, name)))))
    return check(astBuilder.ASTBuilder().visit(parser.prog()))[:2]


class AffineSubstCheckTest(unittest.TestCase):
    def test_paper_programs(self):
        # (L/U fusion, Affine_skip). reuse_topo's L and U differ (fusion measured unsound on N16/N23); BoxPoly,
        # CROWN-IBP and ZID read their inputs' l/u at intermediate Affine layers.
        expected = {
            'deeppoly_priority_add.cf': (True, True),
            'deeppoly_efficient_topo.cf': (True, True),
            'deeppoly_traversal_stop.cf': (True, False),
            'crownibp.cf': (True, False),
            'zid_topo.cf': (True, False),
            'reuse_topo.cf': (False, False),
            'ibp.cf': (False, False),
            'zono.cf': (False, False),
        }
        for name, want in expected.items():
            with self.subTest(name):
                self.assertEqual(run(name), want)

    def test_direct_bound_read_in_final_branch_blocks_skip(self):
        # The last layer's upper bound maps simplify_upper over prev directly, reading an input's u.
        self.assertEqual(run('deeppoly_priority_add_lower.cf'), (True, False))


if __name__ == '__main__':
    unittest.main()
