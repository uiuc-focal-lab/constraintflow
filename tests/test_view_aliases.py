import operator
import tempfile
import unittest
from unittest.mock import patch
import torch
from constraintflow.compiler.ir import *
from constraintflow.compiler.codeGen import CodeGen
from constraintflow.compiler.optimizations import view_aliases,flow_shapes
from constraintflow.lib.globals import device_mode

class ViewAliasesTest(unittest.TestCase):
    def test_views_inline_but_allocating_reshape_stays(self):
        v=lambda n:IrVar(n,[])
        block=IrBlock([
            IrAssignment(v('base'),IrSimpleBinary(v('x'),v('x'),'add')),
            IrAssignment(v('view'),IrTorchTranspose(v('base'),0,1)),
            IrAssignment(v('copy'),IrTorchReshape(v('view'),[6])),
            IrAssignment(v('other'),IrTorchUnsqueeze(v('view'),0)),
            IrTransRetBasic([v('base'),v('view'),v('copy'),v('other'),v('view')])])
        with tempfile.TemporaryDirectory() as d:
            cg=CodeGen(d);cg.file.close()
            sources=[cg._render_stmts([s]).strip() for s in block.children]
            with patch.object(device_mode,'get_device',return_value='cpu'):
                sized=flow_shapes.probe(block.children,sources,{'x':([2,3],'Float')},1,{})
            self.assertIsNotNone(sized)
            self.assertFalse(sized[1]['alloc']);self.assertTrue(sized[2]['alloc'])
            def evaluate():
                env={'x':torch.arange(6.).reshape(2,3),'torch':torch,'operator':operator}
                for stmt in block.children[:-1]:exec(cg._render_stmts([stmt]).strip(),env)
                return [eval(cg.visit(c),env) for c in block.children[-1].children]
            before=evaluate()
            self.assertEqual(view_aliases.run(block,sized),2)
            after=evaluate()
            self.assertEqual([s.children[0].name for s in block.children[:-1]],['base','copy'])
            for a,b in zip(before,after):torch.testing.assert_close(a,b,rtol=0,atol=0)
            self.assertEqual(after[0].untyped_storage()._cdata,after[1].untyped_storage()._cdata)
            self.assertNotEqual(after[0].untyped_storage()._cdata,after[2].untyped_storage()._cdata)

if __name__=='__main__':unittest.main()
