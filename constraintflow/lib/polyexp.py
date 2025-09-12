import torch 
from constraintflow.gbcsr.sparse_tensor import *
 

class PolyExpSparse:
    def __init__(self, network, mat, const):
        self.network = network
        self.mat = mat 
        self.const = const
        if not isinstance(self.const, SparseTensor):
            if isinstance(self.const, torch.Tensor):
                self.const = SparseTensor([torch.tensor([0]*self.const.dim())], [SparseBlock(self.const)], self.const.dim(), torch.tensor(self.const.shape))

    def copy(self):
        if isinstance(self.mat, SparseTensor):
            new_mat = self.mat.copy()
        elif isinstance(self.mat, torch.Tensor):
            new_mat = self.mat.clone()
        else:
            new_mat = self.mat

        if isinstance(self.const, SparseTensor):
            new_const = self.const.copy()
        elif isinstance(self.const, torch.Tensor):
            new_const = self.const.clone()
        else:
            new_const = self.const

        return PolyExpSparse(self.network, new_mat, new_const)

    def get_mat(self, abs_elem, dense=False):
        
        if isinstance(self.mat, float):
            return self.mat
        if dense:
            block = self.mat.get_dense()
            sp_mat = SparseTensor([torch.tensor([0]*block.dim())], [SparseBlock(block)], block.dim(), torch.tensor(block.shape))
        else:
            sp_mat = self.mat
        start, end = torch.nonzero(abs_elem.d['llist']).flatten().tolist()[0], torch.nonzero(abs_elem.d['llist']).flatten().tolist()[-1]
        start, end = self.network[start].start, self.network[end].end
        start_index = torch.zeros(sp_mat.dims, dtype=torch.int64)
        end_index = sp_mat.total_size
        start_index[-1] = start
        end_index[-1] = end
        return sp_mat.get_sparse_custom_range(start_index, end_index)
        
    def get_const(self):
        return self.const
    
    def get_dense_layers(self):
        layer = 0
        dense_layers = set()
        for j, i in enumerate(self.mat.start_indices):
            while(True):
                if self.network[layer].start<=i[-1]:
                    break
                layer+=1
            
            while(True):
                dense_layers.add(layer)
                if self.network[layer].start<=self.mat.end_indices[j][-1]:
                    break
                layer+=1
        return list(dense_layers)
    
    def create_similar(self, network=None, mat=None, const=None):
        if network == None:
            network = self.network
        if mat == None:
            mat = self.mat
        if const == None:
            const = self.const
        return PolyExpSparse(network, mat, const)
