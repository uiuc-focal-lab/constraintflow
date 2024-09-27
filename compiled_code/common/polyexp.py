import torch 
import copy
import time 
from common.sparse_tensor import *


class Network_graph:
    def __init__(self, shapes, layers):
        self.old_network = layers
        self.layers = dict()
        self.layers_size = dict()
        self.layers_start = dict()
        self.layers_end = dict()
        self.size = 0
        self.layers[0] = None
        self.layers_start[0] = 0
        self.layers_end[0] = compute_size(shapes[0])
        self.layers_size[0] = compute_size(shapes[0])
        self.size += self.layers_end[0]
        for i in range(1, len(shapes)):
            # print(layers[i-1].type)
            # print(layers[i-1].weight)
            self.layers[i] = layers[i-1]
            self.layers_size[i] = compute_size(shapes[i])
            self.layers_start[i] = self.size
            self.size += self.layers_size[i]
            self.layers_end[i] = self.size
        self.input_size = self.layers_size[0]
        # kjsd
    


# NEEDED BECAUSE THE L, U, l, u in main_old ARE POLYEXPNEW
class PolyExpNew:
    def __init__(self, size, mat, const):
        if mat==None:
            mat = 0.0
        self.size = size 
        self.mat = mat 
            
        self.const = const
        self.debug_flag = False 
        if not isinstance(const, torch.Tensor) and isinstance(mat, torch.Tensor):
            if self.mat.dim()>1:
                self.const = torch.ones((self.mat.size(0))) * const 

    def copy(self):
        return copy.deepcopy(self)
    
    def get_mat(self, abs_elem):
        live_neurons = torch.nonzero(abs_elem.d['t']).flatten()
        if not isinstance(self.mat, torch.Tensor):
            new_mat = torch.zeros(self.const.shape[0], live_neurons.shape[0])
            return new_mat
        return self.mat[:, live_neurons]
    
    def get_const(self):
        return self.const
    
    def convert_to_polyexp(self):
        return PolyExp(self.const.shape[0], self.size, self.mat, self.const)
    
    def convert_to_polyexp_sparse(self, network):
        return PolyExpSparse(network, SparseTensorBlock([], [], 3, torch.tensor([1, 1004, 1004])), self.const)
    



class PolyExp:
    def __init__(self, rows, cols, mat, const):
        self.rows = rows
        self.cols = cols
        self.mat = mat 
        self.const = const
        self.debug_flag = False 

    def copy(self):
        return copy.deepcopy(self)

    def get_mat(self, poly_size):
        assert(poly_size <= self.cols)
        if self.mat == None and  isinstance(self.const, torch.Tensor):
            return torch.zeros(self.rows, poly_size)
        elif not isinstance(self.mat, torch.Tensor) and  isinstance(self.const, torch.Tensor):
            return torch.ones(self.rows, poly_size)*self.mat
        elif self.mat == None:
            return 0.0
        return self.mat[:, :poly_size]
    
    def get_const(self):
        if self.const == None and  isinstance(self.mat, torch.Tensor):
            return torch.zeros(self.rows)
        elif not isinstance(self.const, torch.Tensor) and  isinstance(self.mat, torch.Tensor):
            return torch.ones(self.rows)*self.const
        elif self.const == None:
            return 0.0
        return self.const
    




class SymExp:
    count = 0
    def __init__(self, rows, cols, mat, const, start, end):
        if SymExp.count < mat.shape[1] :
            SymExp.count = mat.shape[1]
        self.rows = rows
        self.cols = cols
        self.mat = mat 
        self.const = const
        self.start = start 
        self.end = end
        self.debug_flag = False 

    def add_eps(self, num):
        SymExp.count += num
        self.cols += num 
        self.mat = torch.concat([self.mat, torch.eye(num)], dim=1)
        self.end = SymExp.count
        return self
    
    def copy(self):
        return copy.deepcopy(self)

    def get_mat(self, sym_size):
        assert(sym_size <= SymExp.count)
        if self.mat == None and  isinstance(self.const, torch.Tensor):
            return torch.zeros(self.rows, sym_size)
        elif not isinstance(self.mat, torch.Tensor) and  isinstance(self.const, torch.Tensor):
            return torch.ones(self.rows, sym_size)*self.mat
        elif self.mat == None:
            return 0.0
        elif self.mat.shape[1]<sym_size:
            temp = torch.zeros(self.rows, sym_size-self.mat.shape[1])
            return torch.concat([self.mat, temp], dim=1)
        return self.mat[:, :sym_size]
    
    def get_const(self):
        if self.const == None and  isinstance(self.mat, torch.Tensor):
            return torch.zeros(self.rows)
        elif not isinstance(self.const, torch.Tensor) and  isinstance(self.mat, torch.Tensor):
            return torch.ones(self.rows)*self.const
        elif self.const == None:
            return 0.0
        return self.const
    



class PolyExpSparse:
    def __init__(self, network, mat, const):
        self.network = network
        self.mat = mat 
        self.const = const
        if not isinstance(self.const, SparseTensorBlock):
            if isinstance(self.const, torch.Tensor):
                self.const = SparseTensorBlock([torch.tensor([0]*self.const.dim())], [self.const], self.const.dim(), torch.tensor(self.const.shape))
            # else:
            #     self.const = SparseTensorBlock([torch.tensor(0)], [torch.tensor(self.const)], 0, torch.tensor(1))

    def copy(self):
        return PolyExpSparse(self.network, copy.deepcopy(self.mat), copy.deepcopy(self.const))

    def get_mat(self, abs_elem, dense=True):
        
        if isinstance(self.mat, float):
            return self.mat
        if dense:
            block = self.mat.get_dense()
            sp_mat = SparseTensorBlock([torch.tensor([0]*block.dim())], [block], block.dim(), torch.tensor(block.shape))
        else:
            sp_mat = self.mat
        start, end = torch.nonzero(abs_elem.d['llist']).flatten().tolist()[0], torch.nonzero(abs_elem.d['llist']).flatten().tolist()[-1]
        start, end = self.network.layers_start[start], self.network.layers_end[end]
        start_index = torch.zeros(sp_mat.dims)
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
                if self.network.layers_start[layer]<=i[-1]:
                    break
                layer+=1
            
            while(True):
                dense_layers.add(layer)
                if self.network.layers_start[layer]<=self.mat.end_indices[j][-1]:
                    break
                layer+=1
        return list(dense_layers)
