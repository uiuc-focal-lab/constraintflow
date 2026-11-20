import operator
import tempfile
import unittest
import torch
import torch.nn.functional as F
from constraintflow.compiler.ir import *
from constraintflow.compiler.codeGen import CodeGen
from constraintflow.compiler.optimizations.box_concretize import run

class BoxPatchReductionTest(unittest.TestCase):
    def test_lower_and_upper_patch_reductions(self):
        for upper,generic in ((False,False),(True,False),(False,True),(True,True)):
            v=lambda n:IrVar(n,[])
            def gather(name):
                return IrTorchRepeat(IrTorchTranspose(IrFUnfold(IrTorchView(v(name),[2,3,5,6]),[3,2],[1,0],[1,2]),1,2),[1,4,1])
            lo,hi=('constant_a','constant_b') if generic else ('in_l_0','in_u_0')
            x,y=(hi,lo) if upper else (lo,hi)
            def term(name,pos):return IrTorchSum(IrSimpleBinary(gather(name),IrTensorClamp(v('a'),0,pos),'mul'),-1)
            expr=IrSimpleBinary(term(x,True),term(y,False),'add')
            block=IrBlock([IrAssignment(v('r'),expr),IrTransRetBasic([v('r'),v('r')])])
            torch.manual_seed(9)
            env=dict(torch=torch,F=F,operator=operator,in_l_0=torch.randn(2,90,dtype=torch.float64))
            env['in_u_0']=env['in_l_0']+torch.rand(2,90,dtype=torch.float64)
            env[lo]=env['in_l_0'].reshape(2,3,5,6) if generic else env['in_l_0']
            env[hi]=env['in_u_0']
            env['a']=torch.randn(2,60,18,dtype=torch.float64)
            with tempfile.TemporaryDirectory() as d:
                cg=CodeGen(d);cg.file.close();before=cg.visit(expr)
                self.assertEqual(run(block),1)
                expected=eval(before,env)
                for s in block.children:
                    if isinstance(s,IrAssignment):env[s.children[0].name]=eval(cg.visit(s.children[1]),env)
                    else:actual=eval(cg.visit(s.children[0]),env)
                torch.testing.assert_close(actual,expected,rtol=1e-11,atol=1e-11)

class BoxReductionSafetyTest(unittest.TestCase):
    def test_distinct_gather_geometry_is_not_combined(self):
        v=lambda n:IrVar(n,[])
        def term(name,pos,stride):
            x=IrTorchRepeat(IrTorchTranspose(IrFUnfold(IrTorchView(v(name),[2,3,5,6]),[3,2],[1,0],stride),1,2),[1,4,1])
            return IrTorchSum(IrSimpleBinary(x,IrTensorClamp(v('a'),0,pos),'mul'),-1)
        expr=IrSimpleBinary(term('in_l_0',True,[1,2]),term('in_u_0',False,[2,1]),'add')
        block=IrBlock([IrAssignment(v('r'),expr),IrTransRetBasic([v('r'),v('r')])])
        self.assertEqual(run(block),0)

if __name__=='__main__':unittest.main()
