import operator
import tempfile
import unittest
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from constraintflow.compiler.ir import *
from constraintflow.compiler.codeGen import CodeGen
from constraintflow.compiler.optimizations.patch_factor import run


class Shapes(dict):
    inputs = {}


class PatchFactorTest(unittest.TestCase):
    def case(self, patches=False, where=False, inner_stride=(1,1), outer_stride=(1,1), bad=False, chunk=False, unaligned=False):
        torch.manual_seed(12)
        B, Km, Ci, K, H, W = 2, 3, 2, 4, 5, 6
        ih, iw, kh, kw = 2, 3, 3, 2
        pad = [1,0]
        O = ((H+2*pad[0]-kh)//outer_stride[0]+1)*((W+2*pad[1]-kw)//outer_stride[1]+1)
        P, R = Ci*ih*iw, Km*H*W
        v = lambda n: IrVar(n, [])
        weight = IrTorchReshape(IrTorchPermute(IrTorchRepeat(IrTorchUnsqueeze(
            IrTorchView(v('weight'),[1,Km,-1]),-1),[1,1,1,H*W]),[0,1,3,2]),[1,R,-1])
        weight = IrTorchExpand(weight,[B,R,P])
        row = IrTorchExpand(v('scale'),[B,R,P])
        inner = IrSimpleBinary(row,weight,'mul')
        if where:
            inner = IrTorchWhere(IrTorchExpand(v('cond'),[B,R,P]),weight,
                         IrTorchWhere(IrTorchExpand(v('cond2'),[B,R,P]),IrTorchZeros([B,R,P]),inner))
        source = IrTorchPermute(IrTorchReshape(inner,[B,Km,H,W,P]),[0,4,1,2,3])
        if chunk:
            source = IrTorchSlice(source,[0,[1,1+ih*iw] if unaligned else [ih*iw,2*ih*iw]])
            P = ih*iw
        source = IrTorchReshape(source,[B*P,Km,H,W])
        u = IrTorchReshape(IrFUnfold(source,[kh,kw],pad,list(outer_stride)),[B,P,Km,kh,kw,O])
        outer = v('outer')
        eq = 'pckabo,mkab->pcmabo'
        if patches:
            outer = IrTorchPermute(IrTorchReshape(outer,[B,K,O,Km,kh,kw]),[0,1,3,4,5,2])
            eq = 'ncmabo,nkmabo->nckabo'
        contraction = IrTorchEinsum(eq,[u,outer])
        perm = [0,2,5,1,3,4] if not bad else [0,2,5,1,4,3]
        h = IrTorchReshape(IrTorchPermute(contraction,perm),[B*K*O,P,kh*kw])
        outsize = [(kh-1)*inner_stride[0]+ih,(kw-1)*inner_stride[1]+iw]
        expr = IrFFold(h,outsize,[ih,iw],list(inner_stride))
        tensors = {'weight': torch.randn(Km,Ci,ih,iw,dtype=torch.float64),
                   'scale':torch.randn(B,R,1,dtype=torch.float64),
                   'cond':torch.rand(B,R,1)>0.6,'cond2':torch.rand(B,R,1)>0.5,
                   'outer':torch.randn((B,K*O,Km*kh*kw) if patches else (K,Km,kh,kw),dtype=torch.float64)}
        sized = Shapes()
        sized.inputs = {k:{'shape':list(t.shape),'dtype':str(t.dtype)} for k,t in tensors.items()}
        block = IrBlock([IrAssignment(v('result'),expr),IrTransRetBasic([v('result'),v('result')])])
        with tempfile.TemporaryDirectory() as d:
            cg = CodeGen(d)
            cg.file.close()
            original = cg.visit(expr)
            count = run(block,cg.visit,sized,B)
            if bad or unaligned:
                self.assertEqual(count,0)
                return
            self.assertEqual(count,1)
            # Inlining may move the expression to the return.
            ns = dict(tensors,device_mode=SimpleNamespace(get_device=lambda: 'cpu'),torch=torch,F=F,operator=operator,conv_transpose2d=F.conv_transpose2d)
            expected = eval(original,ns)
            for stmt in block.children:
                if isinstance(stmt,IrAssignment):
                    ns[stmt.children[0].name]=eval(cg.visit(stmt.children[1]),ns)
                elif isinstance(stmt,IrTransRetBasic):
                    actual=eval(cg.visit(stmt.children[0]),ns)
            torch.testing.assert_close(actual,expected,rtol=1e-11,atol=1e-11)

    def test_fixed_outer(self):
        self.case()

    def test_position_dependent_outer(self):
        self.case(patches=True)

    def test_relu_piecewise_scale(self):
        self.case(where=True)

    def test_strides(self):
        self.case(patches=True,where=True,inner_stride=(2,1),outer_stride=(2,1))

    def test_channel_chunk(self):
        self.case(chunk=True,where=True,inner_stride=(2,1),outer_stride=(2,1))

    def test_channel_chunk_position_dependent(self):
        self.case(chunk=True,patches=True,where=True)

    def test_reject_partial_input_channel(self):
        self.case(chunk=True,unaligned=True)

    def test_reject_other_layout(self):
        self.case(bad=True)

class DensePatchFactorTest(unittest.TestCase):
    def case(self, chunk=False):
        torch.manual_seed(18)
        B,M,K,C,H,W,kh,kw=2,4,3,2,3,4,3,2
        S,P=H*W,C*kh*kw
        v=lambda n:IrVar(n,[])
        weight=IrTorchReshape(IrTorchPermute(IrTorchRepeat(IrTorchUnsqueeze(
            IrTorchView(v('weight'),[1,K,-1]),-1),[1,1,1,S]),[0,1,3,2]),[1,K*S,-1])
        scaled=IrSimpleBinary(IrTorchExpand(v('scale'),[B,K*S,P]),IrTorchExpand(weight,[B,K*S,P]),'mul')
        a=IrTorchReshape(IrTorchPermute(IrTorchReshape(v('a'),[B,M,K,S]),[0,3,1,2]),[B*S,M,K])
        b=IrTorchPermute(IrTorchReshape(scaled,[B,K,S,P]),[0,2,1,3])
        if chunk:
            b=IrTorchSlice(b,[0,0,0,[kh*kw,2*kh*kw]])
            P=kh*kw
        b=IrTorchReshape(b,[B*S,K,P])
        prod=IrTorchReshape(IrTorchPermute(IrTorchReshape(IrTorchMatmul(a,b),[B,S,M,P]),[0,2,3,1]),[B*M,P,S])
        expr=IrFFold(prod,[6,5],[kh,kw],[2,1],[1,0])
        tensors=dict(a=torch.randn(B,M,K*S,dtype=torch.float64),scale=torch.randn(B,K*S,1,dtype=torch.float64),weight=torch.randn(K,C,kh,kw,dtype=torch.float64))
        sized=Shapes();sized.inputs={k:{'shape':list(t.shape),'dtype':str(t.dtype)} for k,t in tensors.items()}
        block=IrBlock([IrAssignment(v('result'),expr),IrTransRetBasic([v('result'),v('result')])])
        with tempfile.TemporaryDirectory() as d:
            cg=CodeGen(d);cg.file.close();before=cg.visit(expr)
            self.assertEqual(run(block,cg.visit,sized,B),1)
            from constraintflow.gbcsr.sparse_block import bcast_matmul
            ns=dict(tensors,device_mode=SimpleNamespace(get_device=lambda: 'cpu'),torch=torch,F=F,operator=operator,bcast_matmul=bcast_matmul,conv_transpose2d=F.conv_transpose2d)
            expected=eval(before,ns)
            for s in block.children:
                if isinstance(s,IrAssignment):ns[s.children[0].name]=eval(cg.visit(s.children[1]),ns)
                else:actual=eval(cg.visit(s.children[0]),ns)
            torch.testing.assert_close(actual,expected,rtol=1e-11,atol=1e-11)

    def test_dense_patch_with_padding_and_stride(self):
        self.case()

    def test_dense_channel_chunk(self):
        self.case(chunk=True)


if __name__ == '__main__':
    unittest.main()
