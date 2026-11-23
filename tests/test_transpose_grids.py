import operator
import tempfile
import unittest
from types import SimpleNamespace
import torch
import torch.nn.functional as F
from constraintflow.compiler.ir import *
from constraintflow.compiler.codeGen import CodeGen
from constraintflow.compiler.optimizations.product_grids import run

class Shapes(dict):
    inputs={}

class TransposeGridsTest(unittest.TestCase):
    def test_full_product_grid(self):
        torch.manual_seed(9)
        v=lambda n:IrVar(n,[])
        tensors={n:torch.randn(2,3,4,5) for n in ('x','y')}
        tensors.update({n:torch.randn(3,2,3,2) for n in ('w','z')})
        nodes=[IrFConvTranspose2d(v(x),v(w),[2,1],[1,0],[1,0]) for x in ('x','y') for w in ('w','z')]
        block=IrBlock([IrAssignment(v('r'+str(i)),e) for i,e in enumerate(nodes)]+[IrTransRetBasic([v('r'+str(i)) for i in range(4)])])
        sized=Shapes();sized.inputs={n:{'shape':list(x.shape),'dtype':str(x.dtype)} for n,x in tensors.items()}
        with tempfile.TemporaryDirectory() as d:
            cg=CodeGen(d);cg.file.close()
            before=[cg.visit(e) for e in nodes]
            self.assertEqual(run(block,cg.visit,sized,2),4)
            env=dict(tensors,torch=torch,F=F,operator=operator,conv_transpose2d=F.conv_transpose2d)
            expected=[eval(e,env) for e in before]
            for s in block.children:
                if isinstance(s,IrAssignment):env[s.children[0].name]=eval(cg.visit(s.children[1]),env)
                else:actual=[eval(cg.visit(c),env) for c in s.children]
            for x,y in zip(actual,expected):torch.testing.assert_close(x,y)

    def test_dependent_convolution_is_not_hoisted(self):
        v=lambda n:IrVar(n,[])
        a=IrFConvTranspose2d(v('x'),v('w'),[1,1],[1,1],[0,0])
        b=IrFConvTranspose2d(v('a'),v('w'),[1,1],[1,1],[0,0])
        block=IrBlock([IrAssignment(v('a'),a),IrAssignment(v('b'),b),IrTransRetBasic([v('b'),v('b')])])
        sized=Shapes();sized.inputs={n:{'shape':s,'dtype':'torch.float32'} for n,s in [('x',[1,2,4,4]),('w',[2,2,3,3])]}
        sized[0]={'name':'a','shape':[1,2,4,4],'dtype':'torch.float32'}
        with tempfile.TemporaryDirectory() as d:
            cg=CodeGen(d);cg.file.close()
            self.assertEqual(run(block,cg.visit,sized,1),0)

class MatvecGridMemoryTest(unittest.TestCase):
    def build(self, rows, cols):
        v=lambda n:IrVar(n,[])
        nodes=[IrTorchMatmul(v(w),v('x')) for w in ('w','z')]
        block=IrBlock([IrAssignment(v('r'+str(i)),e) for i,e in enumerate(nodes)]
                      +[IrTransRetBasic([v('r0'),v('r1')])])
        sized=Shapes()
        sized.inputs={n:{'shape':shape,'dtype':'torch.float32'} for n,shape in
                      [('w',[1,rows,cols]),('z',[1,rows,cols]),('x',[1,cols,1])]}
        return block,sized,nodes

    def test_large_input_small_output_is_not_packed(self):
        # Exact N35 failure family: tiny result, multi-GiB input matrices.
        # Shape-only analysis must reject it without allocating real tensors.
        block,sized,nodes=self.build(16384,65536)
        with tempfile.TemporaryDirectory() as d:
            cg=CodeGen(d);cg.file.close()
            before=[cg.visit(n) for n in nodes]
            self.assertEqual(run(block,cg.visit,sized,1),0)
            self.assertEqual([cg.visit(s.children[1]) for s in block.children[:-1]],before)

    def test_small_matvec_grid_still_matches(self):
        block,sized,nodes=self.build(7,13)
        torch.manual_seed(3)
        env=dict(torch=torch,operator=operator,bcast_matmul=torch.matmul,
                 w=torch.randn(1,7,13),z=torch.randn(1,7,13),x=torch.randn(1,13,1))
        with tempfile.TemporaryDirectory() as d:
            cg=CodeGen(d);cg.file.close()
            expected=[eval(cg.visit(n),env) for n in nodes]
            self.assertEqual(run(block,cg.visit,sized,1),2)
            for stmt in block.children[:-1]:
                env[stmt.children[0].name]=eval(cg.visit(stmt.children[1]),env)
            actual=[eval(cg.visit(c),env) for c in block.children[-1].children]
            for x,y in zip(actual,expected):torch.testing.assert_close(x,y)

if __name__=='__main__':unittest.main()
