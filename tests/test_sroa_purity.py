import unittest
from types import SimpleNamespace
from constraintflow.compiler.ir import (
    IrVar, IrBlock, IrAssignToBlock, IrAssignToView, IrTransRetBasic,
    IrAssignment, IrBlockCopy,
)
from constraintflow.compiler.optimizations.sroa import (
    _Scalarizer, TensorDesc, BlockDesc, SparseDesc,
)


class SeededScalarizer(_Scalarizer):
    def seed_entry(self):
        x = TensorDesc(IrVar('x', []))
        b = BlockDesc('DenseBlock', x, (1, 2), {})
        s = SparseDesc(2, (1, 2), ((0, 0),), (b,), 0, 'Float')
        self.env.update(x=x, y=TensorDesc(IrVar('y', [])), b=b, s=s)


class SroaPurityTest(unittest.TestCase):
    def test_payload_rebinding_is_functional_and_preserves_alias(self):
        v = lambda x: IrVar(x, [])
        ir = SimpleNamespace(flow_block=IrBlock([
            IrAssignment(v('alias'), v('b')),
            IrAssignToBlock(v('alias'), v('y')),
            IrTransRetBasic([v('s'), v('s')]),
        ]))
        result = SeededScalarizer(ir).run()
        self.assertTrue(result.functional)
        self.assertEqual(result.block_writes, 1)
        self.assertEqual(result.view_writes, 0)
        self.assertEqual(ir.flow_block.children[-1].children[0].name, 'y')

    def test_retained_view_write_is_not_functional(self):
        v = lambda x: IrVar(x, [])
        ir = SimpleNamespace(flow_block=IrBlock([
            IrAssignToView(v('x'), ['0:1','0:2'], v('y')),
            IrTransRetBasic([v('s'), v('s')]),
        ]))
        result = SeededScalarizer(ir).run()
        self.assertFalse(result.functional)
        self.assertEqual(result.view_writes, 1)
        self.assertIsInstance(ir.flow_block.children[0], IrAssignToView)


if __name__ == '__main__':
    unittest.main()
