import unittest
import torch
import torch.nn.functional as F
from constraintflow.gbcsr.sparse_block import conv_transpose2d
# Certified training differentiates bounds through cf::conv_transpose2d; its backward must match aten's.
CASES=[dict(stride=1,padding=0,output_padding=0),dict(stride=2,padding=1,output_padding=1),dict(stride=(2,1),padding=(1,0),output_padding=(1,0))]
class ConvTransposeAutogradTest(unittest.TestCase):
    def test_gradcheck(self):
        torch.manual_seed(0)
        for kw in CASES:
            with self.subTest(**kw):
                x=torch.randn(2,3,4,5,dtype=torch.float64,requires_grad=True)
                w=torch.randn(3,4,3,2,dtype=torch.float64,requires_grad=True)
                self.assertTrue(torch.autograd.gradcheck(lambda x,w:conv_transpose2d(x,w,**kw),(x,w)))
    def test_matches_aten(self):
        torch.manual_seed(1)
        for kw in CASES:
            with self.subTest(**kw):
                x=torch.randn(2,3,4,5,dtype=torch.float64,requires_grad=True)
                w=torch.randn(3,4,3,2,dtype=torch.float64,requires_grad=True)
                g=torch.randn_like(F.conv_transpose2d(x,w,**kw))
                ours=torch.autograd.grad(conv_transpose2d(x,w,**kw),(x,w),g)
                ref=torch.autograd.grad(F.conv_transpose2d(x,w,**kw),(x,w),g)
                for a,b in zip(ours,ref):torch.testing.assert_close(a,b,rtol=1e-12,atol=1e-12)
    def test_input_only(self):
        x=torch.randn(1,3,4,4,dtype=torch.float64,requires_grad=True)
        w=torch.randn(3,2,3,3,dtype=torch.float64)
        conv_transpose2d(x,w,stride=2).sum().backward()
        self.assertIsNotNone(x.grad)
if __name__=='__main__':unittest.main()
