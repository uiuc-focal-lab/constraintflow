import torch 
import math

from constraintflow.lib.polyexp import *
from constraintflow.lib.network import LayerType
from constraintflow.gbcsr.sparse_block import DenseBlock, KernelBlock, ConstBlock, DiagonalBlock
from constraintflow.gbcsr.sparse_tensor import SparseTensor

class Llist:
    def __init__(self, network, initial_shape, start=None, end=None, llist=None):
        self.network = network
        self.initial_shape = initial_shape
        self.start = start
        self.end = end
        self.llist = llist
        self.llist_flag = True
        if llist==None:
            self.llist_flag = False

    def get_metadata(self, elem, batch_size):
        self.coalesce()
        if not self.llist_flag:
            ret = []
            start_indices = []
            temp = 0
            for k in range(self.start, self.end):
                if elem == 'weight' or elem == 'w':
                    if self.network[k].type == LayerType.Linear:
                        block = DenseBlock(self.network[k].weight)
                        if not self.network[k].last_layer:
                            for i in range(len(self.initial_shape)):
                                block = block.unsqueeze(0)
                            repeat_dims = [batch_size]
                            for i in range(len(block.total_shape)-1):
                                repeat_dims.append(1)
                            repeat_dims = torch.tensor(repeat_dims)
                            block = block.repeat(repeat_dims)
                        ret.append(block)
                        start_index = torch.tensor([0]*len(self.initial_shape) + [temp, 0])
                        start_indices.append(start_index)
                        temp += self.network[k].weight.shape[0]
                    elif self.network[k].type == LayerType.Conv2D:
                        ix, iy = self.network[self.network[k].parents[0]].shape[-2:]
                        ox, oy = self.network[k].shape[-2:]
                        sx, sy = self.network[k].stride
                        px, py = self.network[k].padding
                        block = KernelBlock(self.network[k].weight, torch.tensor([self.network[k].size, self.network[self.network[k].parents[0]].size]), ix, iy, ox, oy, sx, sy, px, py)
                        if self.network.no_sparsity:
                            block = DenseBlock(block.get_dense().squeeze(0))
                        for i in range(len(self.initial_shape)):
                            block = block.unsqueeze(0)
                        repeat_dims = [batch_size]
                        for i in range(len(block.total_shape)-1):
                            repeat_dims.append(1)
                        repeat_dims = torch.tensor(repeat_dims)
                        block = block.repeat(repeat_dims)
                        ret.append(block)
                        start_index = torch.tensor([0]*len(self.initial_shape) + [temp, 0])
                        start_indices.append(start_index)
                        temp += self.network[k].size
                    else:
                        raise NotImplementedError
                elif elem == 'bias' or elem == 'b':
                    block = DenseBlock(self.network[k].bias)
                    for i in range(len(self.initial_shape)):
                        block = block.unsqueeze(0)
                    ret.append(block)
                    start_index = torch.tensor([0]*len(self.initial_shape) + [temp])
                    start_indices.append(start_index)
                    temp += self.network[k].size
                elif elem == 'layer':
                    # block = DenseBlock(torch.ones(self.network[k].size, dtype=int)*k)
                    block = ConstBlock(k, torch.tensor([self.network[k].size]))
                    for i in range(len(self.initial_shape)):
                        block = block.unsqueeze(0)
                    ret.append(block)
                    start_index = torch.tensor([0]*len(self.initial_shape) + [temp])
                    start_indices.append(start_index)
                    temp += self.network[k].size
                elif elem == 'last_layer':
                    # block = DenseBlock(torch.ones(self.network[k].size, dtype=int)*k)
                    mat = (k == len(self.network)-1)
                    block = ConstBlock(int(mat), torch.tensor([self.network[k].size]))
                    for i in range(len(self.initial_shape)):
                        block = block.unsqueeze(0)
                    ret.append(block)
                    start_index = torch.tensor([0]*len(self.initial_shape) + [temp])
                    start_indices.append(start_index)
                    temp += self.network[k].size
                else:
                    raise NotImplementedError
            total_shape = start_indices[-1] + ret[-1].total_shape
            dim = len(total_shape)
            return SparseTensor(start_indices, ret, dim, total_shape)
        else:
            raise NotImplementedError
        
    def coalesce(self):
        if not self.llist_flag:
            return True
        for i in range(len(self.llist)-1):
            if self.llist[i]!=self.llist[i+1]-1:
                return False
        self.start = self.llist[0]
        self.end = self.llist[-1]+1
        self.llist_flag = False
        return True
    
    def decoalesce(self):
        if self.llist_flag:
            return True
        self.llist = []
        for i in range(self.start, self.end):
            self.llist.append(i)
        self.llist_flag = True
        return True
                
    def dot(self, mats, total_size):
        if not isinstance(mats, list):
            mats = [mats]
        else:
            assert(False)
        initial_shape = self.initial_shape
        polyexp_const = SparseTensor([], [], 1, torch.tensor([1]))
        polyexp_const = 0.0
        if self.llist_flag:
            start_indices = [torch.tensor([0]*len(self.initial_shape) + [self.network[i].start]) for i in self.llist]
        else:
            start_indices = [torch.tensor([0]*len(self.initial_shape) + [self.network[i].start]) for i in range(self.start, self.end)]
        cols = 0
        for j, mat in enumerate(self.llist):
            if self.llist_flag:
                cols += self.network[self.llist[j]].size
            else:
                cols += self.network[self.start+j].size
        assert(mats[0].total_size[-1] == cols)
        
        initial_shape = []
        for j in range(len(self.initial_shape)):
            initial_shape.append(math.lcm(self.initial_shape[j], mats[0].total_size[j].item()))
        
        new_total_size = torch.tensor(initial_shape+[total_size])
        res_blocks = [i.copy() for i in mats[0].blocks]
        return PolyExpSparse(self.network, SparseTensor(start_indices, res_blocks, len(self.initial_shape)+1, new_total_size), polyexp_const)
    
    def convert_to_poly(self, abs_elem):
        mats = []
        start_indices = []
        index = 0
        if self.llist:
            for i in self.llist:
                mat = torch.ones(self.network[i].size).reshape(*self.initial_shape, self.network[i].size)
                mats.append(DiagonalBlock(mat, total_shape=torch.tensor([*self.initial_shape, self.network[i].size, self.network[i].size]), diag_index=len(self.initial_shape) + 1))
                start_indices.append(torch.tensor([0]*len(self.initial_shape) + [index, self.network[i].start]))
                index += self.network[i].size
        else:
            raise NotImplementedError
    
        polyexp_const = SparseTensor([], [], len(self.initial_shape)+1, torch.tensor(self.initial_shape+[index]))
        return PolyExpSparse(self.network, SparseTensor(start_indices, mats, len(self.initial_shape)+2, torch.tensor(self.initial_shape+[index, abs_elem.get_poly_size()])), polyexp_const)
        