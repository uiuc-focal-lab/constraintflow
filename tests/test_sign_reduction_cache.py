import copy,operator,tempfile,unittest
import torch
from constraintflow.compiler.ir import *
from constraintflow.compiler.codeGen import CodeGen
from constraintflow.compiler.optimizations.sign_reductions import run
class SignReductionCacheTest(unittest.TestCase):
    def test_finalized_definitions_remain_valid_across_later_fusions(self):
        for different in (False,True):
            pool={};v=lambda n:pool.setdefault(n,IrVar(n,[]))
            def reduction(name,positive,vector='b'):
                return IrTorchSum(IrSimpleBinary(IrTensorClamp(v(name),0,positive),v(vector),'mul'),-1)
            statements=[IrAssignment(v('t'),IrSimpleBinary(v('a'),v('scale'),'mul')),
                IrAssignment(v('p'),reduction('t',True)),IrAssignment(v('m'),reduction('t',False)),
                IrAssignment(v('r'),IrSimpleBinary(v('p'),v('m'),'add')),
                IrAssignment(v('updated'),IrSimpleBinary(v('t'),IrTorchUnsqueeze(v('r'),-1),'add')),
                IrAssignment(v('r2'),IrSimpleBinary(reduction('updated',True),reduction('updated',False,'c' if different else 'b'),'add')),
                IrTransRetBasic([v('r'),v('r2')])]
            block=IrBlock(statements)
            torch.manual_seed(31)
            inputs=dict(a=torch.randn(2,3,dtype=torch.float64),scale=torch.randn(2,3,dtype=torch.float64),b=torch.randn(3,dtype=torch.float64),c=torch.randn(3,dtype=torch.float64))
            with tempfile.TemporaryDirectory() as d:
                cg=CodeGen(d);cg.file.close()
                def execute(b):
                    env=dict(inputs,torch=torch,operator=operator)
                    for stmt in b.children:
                        if isinstance(stmt,IrAssignment):env[stmt.children[0].name]=eval(cg.visit(stmt.children[1]),env)
                        else:return [eval(cg.visit(e),env) for e in stmt.children]
                expected=execute(copy.deepcopy(block))
                self.assertEqual(run(block),1 if different else 2)
                for actual,reference in zip(execute(block),expected):
                    torch.testing.assert_close(actual,reference,rtol=1e-12,atol=1e-12)
if __name__=='__main__':unittest.main()
