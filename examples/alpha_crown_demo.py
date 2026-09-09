"""alpha-CROWN from a ConstraintFlow spec, alpha as a first-class optimizable value."""
import sys, os, torch
OUT = sys.argv[1]; sys.path.insert(0, OUT)
from constraintflow.lib.symexp import alpha_parameters
from constraintflow.cli import get_dataset
import main

BS  = int(os.environ.get('CF_BS', 100))
EPS = float(os.environ.get('CF_EPS', 0.03))
NET = os.environ.get('CF_NET', '/home/aval/constraintflow/nets/mnist/mnist_relu_3_50.onnx')
X, y = get_dataset(BS, 'mnist', train=False)
torch.manual_seed(0)

def run():
    return main.run(NET, BS, EPS, X, y, dataset='mnist', train=False,
                    print_intermediate_results=False, no_sparsity=False)

lb, ub = run()                                   # first run materialises the alphas
P = alpha_parameters()
print(f'alpha tensors: {[tuple(p.shape) for p in P]}')
print(f'random init : verified {int((lb>=0).all(dim=1).sum())}/{BS}  sum(lb)={float(lb.sum()):.2f}')

opt = torch.optim.Adam(P, lr=0.5)
sched = torch.optim.lr_scheduler.ExponentialLR(opt, 0.98)
best = lb.detach().clone()
ITERS = int(os.environ.get('CF_ITERS','20'))
for it in range(ITERS):
    opt.zero_grad()
    lb, ub = run()
    best = torch.maximum(best, lb.detach())
    (-lb.sum()).backward()
    opt.step(); sched.step()
    if it % 20 == 0 or it == ITERS-1:
        print(f'  it{it:2d} verified={int((lb>=0).all(dim=1).sum()):3d}/{BS} '
              f'sum(lb)={float(lb.sum()):10.2f}  best={int((best>=0).all(dim=1).sum())}')
print(f'FINAL spec alpha-CROWN: verified {int((best>=0).all(dim=1).sum())}/{BS}  sum(lb)={float(best.sum()):.2f}')
torch.save({'lb_ac': best}, os.path.join(OUT, 'spec_alpha_out.pt'))
