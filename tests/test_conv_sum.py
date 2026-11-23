import operator
import tempfile
import unittest
import torch
import torch.nn.functional as F
from constraintflow.compiler.ir import *
from constraintflow.compiler.codeGen import CodeGen
from constraintflow.compiler.optimizations.conv_sum import run


class ConvSumTest(unittest.TestCase):
    def test_factor_and_reject_different_weights(self):
        for same,chunk in ((True,False),(False,False),(True,True),(False,True)):
            v=lambda n:IrVar(n,[])
            conv=lambda n,w:IrTorchReshape(IrFConvTranspose2d(
                IrTorchReshape(v(n),[2,3,4,5]),
                IrTorchSlice(v('w'),[0,[0,1] if w=='w' else [1,2]]) if chunk else v(w),[1,1],[0,0],[0,0]),[2,-1])
            expr=IrSimpleBinary(conv('x','w'),conv('y','w' if same else 'other'),'add')
            block=IrBlock([IrAssignment(v('z'),expr),IrTransRetBasic([v('z'),v('z')])])
            with tempfile.TemporaryDirectory() as d:
                cg=CodeGen(d);cg.file.close()
                before=cg.visit(expr)
                count=run(block)
                self.assertEqual(count,int(same))
                torch.manual_seed(12)
                env=dict(torch=torch,F=F,operator=operator,conv_transpose2d=F.conv_transpose2d,
                         x=torch.randn(2,3,4,5,dtype=torch.float64),y=torch.randn(2,3,4,5,dtype=torch.float64),
                         w=torch.randn(3,2,3,3,dtype=torch.float64),other=torch.randn(3,2,3,3,dtype=torch.float64))
                expected=eval(before,env)
                for s in block.children:
                    if isinstance(s,IrAssignment):env[s.children[0].name]=eval(cg.visit(s.children[1]),env)
                    else:actual=eval(cg.visit(s.children[0]),env)
                torch.testing.assert_close(actual,expected,rtol=1e-12,atol=1e-12)

    def test_concatenated_channel_chunks(self):
        for same in (True,False):
            v=lambda n:IrVar(n,[])
            def branch(n, different=False):
                pieces=[]
                for lo,hi in ((0,2),(2,3)):
                    w=v('other') if different and lo==2 else v('w')
                    conv=IrFConvTranspose2d(IrTorchReshape(v(n),[2,3,4,5]),
                        IrTorchSlice(w,[0,[lo,hi]]),[1,1],[0,0],[0,0])
                    pieces.append(IrTorchReshape(conv,[2,1,-1]))
                return IrTorchReshape(IrTorchCat(pieces,2),[2,-1])
            expr=IrSimpleBinary(branch('x'),branch('y',not same),'add')
            block=IrBlock([IrAssignment(v('z'),expr),IrTransRetBasic([v('z'),v('z')])])
            with tempfile.TemporaryDirectory() as d:
                cg=CodeGen(d);cg.file.close();before=cg.visit(expr)
                self.assertEqual(run(block),2 if same else 0)
                torch.manual_seed(14)
                env=dict(torch=torch,F=F,operator=operator,conv_transpose2d=F.conv_transpose2d,
                    x=torch.randn(2,3,4,5,dtype=torch.float64),y=torch.randn(2,3,4,5,dtype=torch.float64),
                    w=torch.randn(3,3,3,2,dtype=torch.float64),other=torch.randn(3,3,3,2,dtype=torch.float64))
                expected=eval(before,env)
                for stmt in block.children:
                    if isinstance(stmt,IrAssignment):env[stmt.children[0].name]=eval(cg.visit(stmt.children[1]),env)
                    else:actual=eval(cg.visit(stmt.children[0]),env)
                torch.testing.assert_close(actual,expected,rtol=1e-12,atol=1e-12)


if __name__=='__main__':unittest.main()
