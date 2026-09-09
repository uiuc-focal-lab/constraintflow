from constraintflow.gbcsr.sparse_tensor import *
from constraintflow.gbcsr.sparse_block import *

def get_num_eps(mat):
    if mat==None:
        return 0
    num = 0
    for i in range(mat.num_blocks):
        num += mat.blocks[i].total_shape[-1]
    return num

def get_new_eps(network, initial_shape):
    num = initial_shape[-1].item()
    const = SparseTensor([], [], len(initial_shape), initial_shape)
    start_index = torch.concat([torch.zeros(len(initial_shape), dtype=int), torch.tensor([SymExpSparse.count])])
    mat_tensor = torch.ones(num, dtype=int) 
    for i in range(len(initial_shape)-1):
        mat_tensor = mat_tensor.unsqueeze(0)
    mat_tensor = mat_tensor.repeat(*(list(initial_shape[:-1]) + [1]))
    mat = DiagonalBlock(mat_tensor, torch.tensor(list(initial_shape) + [num]), diag_index=len(initial_shape))
    mat = SparseTensor([start_index], [mat], len(initial_shape)+1, torch.tensor(list(initial_shape) + [num+SymExpSparse.count]))
    if network.no_sparsity:
        dense_mat = mat.blocks[0].get_dense()
        mat.blocks[0] = DenseBlock(dense_mat)
        mat.end_indices[0] = start_index + torch.tensor(dense_mat.shape)
    SymExpSparse.count += num
    return SymExpSparse(network, mat, const)

class SymExpSparse:
    count = 0
    def __init__(self, network, mat = None, const = 0.0):
        if SymExpSparse.count < get_num_eps(mat) :
            SymExpSparse.count = get_num_eps(mat)
        self.mat = mat
        self.const = const
        self.network = network
        if mat==None:
            if isinstance(const, SparseTensor):
                self.mat = SparseTensor([], [], const.dims+1, torch.tensor(list(const.total_size) + [SymExpSparse.count]))
            

    def expand_mat(self):
        assert(self.mat.dense_const==0)
        self.mat.total_size[-1] = SymExpSparse.count

    def get_mat(self, sym_size):
        if self.mat == None:
            return SparseTensor([], [], 0, torch.tensor([]))
        self.expand_mat()
        return self.mat

    def get_const(self):
        return self.const

# ---------------------------------------------------------------- alpha params
# An `alpha` in a ConstraintFlow spec is a concrete, optimizable Float tensor.
# Shape comes from shape analysis ([batch_size, curr_size]), so it is per
# (input, neuron) -- not shared across the batch the way neuron metadata is.
_ALPHA_STORE = {}     # (site, occurrence) -> leaf tensor, requires_grad=True
_ALPHA_CALLS = {}     # site -> occurrences seen in the current flow


def reset_alpha():
    """Called at the start of each flow so a site maps to the same tensor every run."""
    _ALPHA_CALLS.clear()


def clamp_alpha():
    """Project every alpha back onto [0,1]. Call after each optimizer step.

    get_alpha() also clamps in the forward, which keeps every run sound, but a
    leaf that drifts outside [0,1] then sees zero gradient through the clamp and
    is stuck there. Projecting the parameter itself keeps it on the boundary
    where gradient still flows. (auto_LiRPA does the same in clip_alpha().)
    """
    with torch.no_grad():
        for t in _ALPHA_STORE.values():
            t.clamp_(0.0, 1.0)


def alpha_parameters():
    """The leaves, for torch.optim."""
    return [_ALPHA_STORE[k] for k in sorted(_ALPHA_STORE)]


def get_alpha(site, shape, init=None):
    n = _ALPHA_CALLS.get(site, 0)
    _ALPHA_CALLS[site] = n + 1
    key = (site, n)
    if key not in _ALPHA_STORE:
        dims = [int(x) for x in shape.tolist()]
        if init is None:
            v = torch.rand(*dims)                                   # random init, in the sound range
        else:
            v = init.get_dense() if hasattr(init, 'get_dense') else torch.as_tensor(init)
            v = v.detach().float().expand(*dims).clone()
        _ALPHA_STORE[key] = v.requires_grad_(True)
    val = _ALPHA_STORE[key].clamp(0.0, 1.0)                         # sound for every value in [0,1]
    return SparseTensor([torch.zeros(len(val.shape), dtype=int)],
                        [DenseBlock(val)], len(val.shape),
                        torch.tensor(list(val.shape)))
