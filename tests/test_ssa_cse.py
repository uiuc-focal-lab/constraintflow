import copy,operator,tempfile,unittest
import torch
from constraintflow.compiler.ir import *
from constraintflow.compiler.codeGen import CodeGen
from constraintflow.compiler.optimizations import ssa_cse,block_passes

class SSACSETest(unittest.TestCase):
    def execute(self,block,inputs):
        env=dict(inputs,operator=operator,torch=torch)
        with tempfile.TemporaryDirectory() as d:
            cg=CodeGen(d);cg.file.close()
            for stmt in block.children:
                if isinstance(stmt,IrAssignment):env[stmt.children[0].name]=eval(cg.visit(stmt.children[1]),env)
                else:return [eval(cg.visit(e),env) for e in stmt.children]

    def test_nested_shared_expressions_and_distinct_options(self):
        pool={}
        v=lambda name:pool.setdefault(name,IrVar(name,[]))
        def expr(dim):
            repeated=IrSimpleBinary(v('x'),v('y'),'mul')
            return IrTorchSum(IrSimpleBinary(repeated,copy.deepcopy(repeated),'add'),dim)
        block=IrBlock([IrAssignment(v('a'),expr(1)),IrAssignment(v('b'),expr(0)),
                       IrAssignment(v('c'),expr(1)),IrTransRetBasic([v('a'),v('b'),v('c')])])
        inputs=dict(x=torch.randn(3,4),y=torch.randn(3,4))
        expected=self.execute(copy.deepcopy(block),inputs)
        legacy=copy.deepcopy(block);block_passes.cse(legacy)
        self.assertGreater(ssa_cse.run(block),0)
        block_passes.copy_propagate(block)
        block_passes.dce(block)
        from constraintflow.compiler.optimizations import subexp_inlining
        subexp_inlining.inline_subexp_block(block)
        for actual,reference,old in zip(self.execute(block,inputs),expected,self.execute(legacy,inputs)):
            torch.testing.assert_close(actual,reference,rtol=0,atol=0)
            torch.testing.assert_close(actual,old,rtol=0,atol=0)

    def test_input_name_collisions_and_first_use(self):
        from constraintflow.compiler.optimizations import cse
        v=lambda n:IrVar(n,[])
        def expr():return IrSimpleBinary(v('cse_var_1'),v('x'),'add')
        block=IrBlock([IrAssignment(v('a'),expr()),IrAssignment(v('b'),expr()),
                       IrTransRetBasic([v('a'),v('b')])])
        inputs=dict(cse_var_1=torch.randn(3),x=torch.randn(3))
        expected=self.execute(copy.deepcopy(block),inputs)
        cse.counter=0
        ssa_cse.run(block)
        self.assertNotEqual(block.children[0].children[0].name,'cse_var_1')
        for actual,reference in zip(self.execute(block,inputs),expected):
            torch.testing.assert_close(actual,reference,rtol=0,atol=0)

    def test_non_ssa_is_rejected(self):
        v=lambda n:IrVar(n,[])
        block=IrBlock([IrAssignment(v('a'),v('x')),IrAssignment(v('a'),v('y'))])
        with self.assertRaises(ValueError):ssa_cse.run(block)

if __name__=='__main__':unittest.main()
