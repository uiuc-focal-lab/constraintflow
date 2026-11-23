import unittest
from constraintflow.compiler.ir import IrVar, IrConst, IrAssignment, IrSimpleBinary
from constraintflow.compiler.optimizations.tensor_to_block import clone_traversal_body

class TraversalCloneTest(unittest.TestCase):
    def test_def_links_and_shared_nodes_are_preserved(self):
        x=IrVar('x',[])
        first=IrAssignment(x,IrConst(2,'Float'));x.defs=first
        expression=IrSimpleBinary(x,x,'add')
        second=IrAssignment(IrVar('y',[]),expression)
        first.uses=[second]
        external=IrAssignment(IrVar('outside',[]),IrConst(7,'Float'))
        second.uses=[external]
        x.parents=[expression]
        a,b=clone_traversal_body([first,second])
        lhs,rhs=b.children[1].children
        self.assertIs(lhs,rhs)
        self.assertIs(lhs.defs,a)
        self.assertIsNot(lhs,x)
        self.assertFalse(a.uses)
        self.assertFalse(b.uses)
        self.assertFalse(lhs.parents)
        a.children[1].const=9
        self.assertEqual(first.children[1].const,2)
        self.assertEqual(second.uses,[external])

if __name__=='__main__':unittest.main()
