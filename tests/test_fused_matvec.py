import operator
import tempfile
import unittest
import torch
from constraintflow.compiler.ir import *
from constraintflow.compiler.codeGen import CodeGen
from constraintflow.compiler.optimizations import fused_matvec


class Shapes(dict):
    pass


class FusedMatvecTest(unittest.TestCase):
    def check_case(self, batch_a, batch_x, dtype, transpose=False, positive=True):
        torch.manual_seed(7)
        a = torch.randn(batch_a, 13, 17, dtype=dtype)
        if transpose:
            a = a.transpose(-1, -2)
        x = torch.randn(batch_x, a.shape[-1], 1, dtype=dtype)
        v = lambda n: IrVar(n, [])
        node = IrTorchMatmul(IrTensorClamp(v('a'), 0, positive), v('x'))
        block = IrBlock([IrAssignment(v('y'), node), IrTransRetBasic([v('y')])])
        sized = Shapes()
        sized.inputs = {n: {'shape': list(t.shape), 'dtype': str(t.dtype)}
                        for n, t in [('a', a), ('x', x)]}
        with tempfile.TemporaryDirectory() as d:
            cg = CodeGen(d); cg.file.close()
            env = dict(a=a, x=x, torch=torch, operator=operator)
            expected = eval(cg.visit(node), env)
            self.assertEqual(fused_matvec.run(block, cg.visit, sized, batch_x, min_bytes=0), 1)
            actual = eval(cg.visit(block.children[0].children[1]), env)
            torch.testing.assert_close(actual, expected, rtol=2e-6, atol=2e-6)
            self.assertEqual(actual.shape, expected.shape)

    def test_broadcast_and_noncontiguous(self):
        for dtype in (torch.float32, torch.float64):
            for ba, bx in ((1,1), (1,3), (3,1), (3,3)):
                for transpose in (False,True):
                    for positive in (False,True):
                        with self.subTest(dtype=dtype, ba=ba, bx=bx, transpose=transpose, positive=positive):
                            self.check_case(ba,bx,dtype,transpose,positive)

    def test_reject_matrix_matrix_and_small(self):
        v=lambda n:IrVar(n,[])
        for cols, min_bytes in ((1,1024**2),(4,0)):
            node=IrTorchMatmul(IrTensorClamp(v('a'),0,True),v('x'))
            block=IrBlock([IrAssignment(v('y'),node)])
            sized=Shapes();sized.inputs={n:{'shape':s,'dtype':'torch.float32'}
                for n,s in [('a',[1,13,17]),('x',[1,17,cols])]}
            with tempfile.TemporaryDirectory() as d:
                cg=CodeGen(d);cg.file.close()
                before=cg.visit(node)
                self.assertEqual(fused_matvec.run(block,cg.visit,sized,1,min_bytes=min_bytes),0)
                self.assertEqual(cg.visit(block.children[0].children[1]),before)

if __name__=='__main__':unittest.main()
