import ast,operator,tempfile,unittest
import torch
from constraintflow.compiler.ir import *
from constraintflow.compiler.codeGen import CodeGen
from constraintflow.compiler.optimizations.expression_limits import run
class ExpressionLimitsTest(unittest.TestCase):
    def test_deep_arithmetic_keeps_order_and_parses(self):
        v=lambda n:IrVar(n,[])
        expression=v('x');reference=torch.tensor([1e16],dtype=torch.float64)
        for i in range(360):
            value=(1.0,-1e16,1e16)[i%3]
            expression=IrSimpleBinary(expression,IrConst(value,'Float'),'add')
            reference=reference+value
        block=IrBlock([IrAssignment(v('result'),expression),IrTransRetBasic([v('result'),v('result')])])
        with tempfile.TemporaryDirectory() as d:
            cg=CodeGen(d);cg.file.close()
            with self.assertRaises(SyntaxError):ast.parse(cg.visit(expression),mode='eval')
            self.assertGreater(run(block),0)
            env=dict(x=torch.tensor([1e16],dtype=torch.float64),torch=torch,operator=operator)
            for stmt in block.children:
                if isinstance(stmt,IrAssignment):
                    source=cg.visit(stmt.children[1]);ast.parse(source,mode='eval')
                    env[stmt.children[0].name]=eval(source,env)
                else:actual=eval(cg.visit(stmt.children[0]),env)
            torch.testing.assert_close(actual,reference,rtol=0,atol=0)
    def test_existing_names_and_repeated_application(self):
        v=lambda n:IrVar(n,[])
        expr=v('expr_split_0')
        for _ in range(12):expr=IrSimpleBinary(expr,IrConst(1,'Float'),'add')
        block=IrBlock([IrAssignment(v('r'),expr),IrTransRetBasic([v('r')])])
        self.assertGreater(run(block,max_depth=4),0)
        names=[s.children[0].name for s in block.children if isinstance(s,IrAssignment)]
        self.assertNotIn('expr_split_0',names)
        self.assertEqual(run(block,max_depth=4),0)
if __name__=='__main__':unittest.main()
