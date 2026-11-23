import operator,tempfile,unittest
import torch
import torch.nn.functional as F
from constraintflow.compiler.ir import *
from constraintflow.compiler.codeGen import CodeGen
from constraintflow.compiler.optimizations.conv_chunks import run
class Shapes(dict):inputs={}
class ConvChunksTest(unittest.TestCase):
    def case(self,bad=None):
        v=lambda n:IrVar(n,[])
        pieces=[]
        for i,(lo,hi) in enumerate(((0,2),(2,5))):
            if bad=='order' and i:lo,hi=1,4
            w=IrTorchSlice(v('w'),[0,[lo,hi]])
            x=v('other' if bad=='input' and i else 'x')
            c=IrFConvTranspose2d(x,w,[2,1],[1,0],[1,0])
            pieces.append(IrTorchReshape(c,[1,2,(hi-lo)*8*6]))
        expr=IrTorchCat(pieces,1 if bad=='axis' else 2)
        block=IrBlock([IrAssignment(v('r'),expr),IrTransRetBasic([v('r'),v('r')])])
        torch.manual_seed(7)
        tensors=dict(w=torch.randn(3,5,3,2,dtype=torch.float64),x=torch.randn(2,3,4,5,dtype=torch.float64),other=torch.randn(2,3,4,5,dtype=torch.float64))
        sized=Shapes();sized.inputs={k:{'shape':list(t.shape),'dtype':str(t.dtype)} for k,t in tensors.items()}
        with tempfile.TemporaryDirectory() as d:
            cg=CodeGen(d);cg.file.close();original=cg.visit(expr)
            count=run(block,cg.visit,sized,1)
            self.assertEqual(count,0 if bad else 1)
            if bad:return
            ns=dict(tensors,torch=torch,F=F,operator=operator,conv_transpose2d=F.conv_transpose2d)
            expected=eval(original,ns)
            for s in block.children:
                if isinstance(s,IrAssignment):ns[s.children[0].name]=eval(cg.visit(s.children[1]),ns)
                else:actual=eval(cg.visit(s.children[0]),ns)
            torch.testing.assert_close(actual,expected,rtol=1e-12,atol=1e-12)
    def test_recombine(self):self.case()
    def test_reject_different_inputs(self):self.case('input')
    def test_reject_overlapping_channels(self):self.case('order')
    def test_reject_wrong_axis(self):self.case('axis')
if __name__=='__main__':unittest.main()
