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