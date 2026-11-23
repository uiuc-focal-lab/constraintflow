import unittest
import torch
from constraintflow.gbcsr.sparse_block import safe_div
# Training-build divisions: same values and gradients as `/`, but no NaN from arms torch.where discards.
class SafeDivTest(unittest.TestCase):
    def test_gradcheck(self):
        torch.manual_seed(0)
        num=torch.randn(3,4,dtype=torch.float64,requires_grad=True)
        den=(torch.rand(3,4,dtype=torch.float64)+0.5).requires_grad_(True)
        self.assertTrue(torch.autograd.gradcheck(safe_div,(num,den)))
        row=(torch.rand(1,4,dtype=torch.float64)+0.5).requires_grad_(True)
        self.assertTrue(torch.autograd.gradcheck(safe_div,(num,row)))  # broadcast denominator
    def test_matches_division(self):
        torch.manual_seed(1)
        num=torch.randn(5,dtype=torch.float64,requires_grad=True); den=(torch.rand(5,dtype=torch.float64)+0.5).requires_grad_(True)
        g=torch.randn(5,dtype=torch.float64)
        ours=torch.autograd.grad(safe_div(num,den),(num,den),g); ref=torch.autograd.grad(num/den,(num,den),g)
        for a,b in zip(ours,ref):torch.testing.assert_close(a,b,rtol=1e-12,atol=1e-12)
    def test_unselected_arm(self):
        # u/(u-l) of a stable neuron with a tiny width: a finite forward, but den**2 underflows in float32.
        u=torch.tensor([1e-3,0.5],requires_grad=True); l=torch.tensor([1e-3-1e-20,-0.5],requires_grad=True)
        y=torch.where(torch.tensor([False,True]),safe_div(u,u-l),torch.zeros(2))
        y.sum().backward()
        self.assertTrue(torch.isfinite(u.grad).all() and torch.isfinite(l.grad).all())
        self.assertEqual(u.grad[0].item(),0.0)
    def test_zero_denominator(self):
        x=torch.tensor([0.0,2.0],requires_grad=True)
        y=torch.where(torch.tensor([False,True]),safe_div(x,x),torch.zeros(2))
        y.sum().backward()
        self.assertTrue(torch.isfinite(x.grad).all())
    def test_scalar_operands(self):
        d=torch.tensor([2.0,4.0],requires_grad=True)
        torch.testing.assert_close(safe_div(1.0,d),torch.tensor([0.5,0.25]))
        self.assertEqual(safe_div(torch.tensor([3.0]),2.0).item(),1.5)
    def test_compiles(self):
        f=torch.compile(lambda a,b:torch.where(b>0,safe_div(a,b),a),fullgraph=True,backend='aot_eager')
        a=torch.randn(4,requires_grad=True);b=torch.tensor([1.0,-1.0,2.0,-1e-30],requires_grad=True)  # tiny den only in the unselected arm
        f(a,b).sum().backward()
        self.assertTrue(torch.isfinite(a.grad).all() and torch.isfinite(b.grad).all())
if __name__=='__main__':unittest.main()
