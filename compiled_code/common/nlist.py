from common.polyexp import *
from specs.network import Network, LayerType

import torch 
import math

class Nlist:
    network = None 
    def __init__(self, size=0, start=None, end=None, nlist=None, network = None):
        self.size = size
        self.start = start
        self.end = end
        self.nlist = nlist
        self.nlist_flag = True
        network = network 
        if Nlist.network==None:
            Nlist.network = network 
        if nlist==None:
            self.nlist_flag = False

    def get_metadata(self, elem):
        if not self.nlist_flag:
            counter = Nlist.network.input_size
            for k, layer in enumerate(Nlist.network):
                if counter == self.start:
                    if elem == 'weight' or elem == 'w':
                        return SparseBlock(layer.weight)
                    elif elem == 'bias' or elem == 'b':
                        return SparseBlock(layer.bias)
                    elif elem == 'layer':
                        return torch.ones(layer.bias.shape)*k
                counter += layer.size
        else:
            raise Exception('NOT IMPLEMENTED')
                
    def dot(self, w):
        if isinstance(w, torch.Tensor):
            n = w.size(0)
        elif isinstance(w, list):
            n = len(w)
        
        N = self.size 
        polyexp_const = torch.zeros((n))
        polyexp_coeff = torch.zeros(N)
        if isinstance(w, list) or w.dim()==2:
            polyexp_coeff = polyexp_coeff.unsqueeze(0).expand(n, -1).clone()
            if self.nlist_flag:
                if not(isinstance(self.nlist, list)) and self.nlist.dim() == 1:
                    self.nlist = self.nlist.unsqueeze(0).expand(n, -1)
        if self.nlist_flag:
            if isinstance(w, list):
                for i in range(n):
                    polyexp_coeff[i, self.nlist[i]] = w[i]
            else:
                polyexp_coeff[torch.arange(n).unsqueeze(1), self.nlist] = w 
        else:
            polyexp_coeff[:, self.start:self.end+1] = w 
        res = PolyExp(n, N, polyexp_coeff, polyexp_const)
        # res = PolyExpNew(N, polyexp_coeff, polyexp_const)
        return res 
    
    def sum(self):
        N = self.size 
        polyexp_coeff = torch.zeros(N)
        if self.nlist_flag:
            if self.nlist.dim()>1:
                n = self.nlist.size(0)
                polyexp_coeff = polyexp_coeff.unsqueeze(0).expand(n, -1).clone()
                polyexp_coeff[torch.arange(n).unsqueeze(1), self.nlist] = 1
                polyexp_const = torch.zeros(n)
            else:
                polyexp_coeff[self.nlist] = 1
                polyexp_const = 0
        else:
            if isinstance(self.start, torch.Tensor):
                if self.start.shape[0]>1:
                    n = self.start.size(0)
                    polyexp_coeff = polyexp_coeff.unsqueeze(0).expand(n, -1).clone()
                    polyexp_coeff[:, self.start:self.end+1] = 1
                    polyexp_const = torch.zeros(n)
                else:
                    polyexp_coeff[self.start:self.end+1] = 1
                    polyexp_const = 0
            else:
                polyexp_coeff[self.start:self.end+1] = 1
                polyexp_const = 0
        res = PolyExp(n, N, polyexp_coeff, polyexp_const)
        # res = PolyExpNew(N, polyexp_coeff, polyexp_const)
        return res 
    
    def avg(self):
        res = self.sum()
        if self.nlist_flag:
            if self.nlist.dim()>1:
                num_elems = self.nlist.shape[1]
            else:
                num_elems = self.nlist.shape[0]
        else:
            num_elems = self.end + 1 - self.start
        res.coeff = res.coeff / num_elems
        res.const = res.const / num_elems
        return res 

    def convert_to_poly(self):
        if self.nlist_flag:
            mat = torch.zeros(len(self.nlist), self.size)
            const = torch.zeros(len(self.nlist))
            for i in range(len(self.nlist)):
                mat[i, self.nlist[i]] = 1
            return PolyExp(len(self.nlist), self.size, mat, const)
            # return PolyExpNew(self.size, mat, const)
        else:
            mat = torch.zeros(self.end-self.start+1, self.size)
            const = torch.zeros(self.end-self.start+1)
            for i in range(self.end-self.start+1):
                mat[i, self.start+i] = 1
            return PolyExp(self.end-self.start+1, mat, const)
            # return PolyExpNew(self.size, mat, const)


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

    def get_metadata(self, elem):
        self.coalesce()
        if not self.llist_flag:
            ret = []
            start_indices = []
            for k in range(self.start, self.end):
                if elem == 'weight' or elem == 'w':
                    if self.network[k].type == LayerType.Linear:
                        ret.append(SparseBlock(self.network[k].weight.unsqueeze(0)))
                        total_shape = torch.tensor(ret[0].block.shape)
                        dim = 3
                    if self.network[k].type == LayerType.Conv2D:
                        # if k-1==0:
                        #     total_shape = torch.tensor([1, self.network[k].size, self.network.input_size])
                        #     ix, iy = self.network.input_shape[-2:]
                        # else:
                        total_shape = torch.tensor([1, self.network[k].size, self.network[k-1].size])
                        ix, iy = self.network[k-1].shape[-2:]
                        ox, oy = self.network[k].shape[-2:]
                        sx, sy = self.network[k].stride
                        px, py = self.network[k].padding
                        ret.append(SparseBlock(self.network[k].weight, 'Kernel', total_shape, ix, iy, ox, oy, sx, sy, px, py))
                        dim = 3
                    start_indices.append(torch.tensor([0,0,0]))
                elif elem == 'bias' or elem == 'b':
                    ret.append(SparseBlock(self.network[k].bias.unsqueeze(0)))
                    start_indices.append(torch.tensor([0,0]))
                    dim = 2
                    total_shape = torch.tensor(ret[0].block.shape)
            assert(len(ret) == 1)
            return SparseTensorBlock(start_indices, ret, dim, total_shape)
        else:
            raise Exception('NOT NEEDED')
            ret = []
            for k in self.llist:
                if elem == 'weight' or elem == 'w':
                    ret.append(self.network[k].weight.unsqueeze(0))
                elif elem == 'bias' or elem == 'b':
                    ret.append(self.network[k].bias.unsqueeze(0))
            return ret
        
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
        polyexp_const = SparseTensorBlock([], [], 1, torch.tensor([1]))
        polyexp_const = 0.0
        if self.llist_flag:
            start_indices = [torch.tensor([0]*len(self.initial_shape) + [self.network[i].start]) for i in self.llist]
        else:
            start_indices = [torch.tensor([0]*len(self.initial_shape) + [self.network[i].start]) for i in range(self.start, self.end)]
        for j, mat in enumerate(mats):
            if self.llist_flag:
                cols = self.network[self.llist[j]].size
            else:
                cols = self.network[self.start+j].size
            # assert(list(mat.total_size[:-1]) == initial_shape)
            assert(mat.total_size[-1] == cols)
        
        initial_shape = self.initial_shape
        for j in range(len(initial_shape)):
            initial_shape[j] = math.lcm(initial_shape[j], mats[0].total_size[j].item())
        
        return PolyExpSparse(self.network, SparseTensorBlock(start_indices, copy.deepcopy(mats[0].blocks), len(self.initial_shape)+1, torch.tensor(initial_shape+[total_size])), polyexp_const)
        return PolyExpSparse(self.network, initial_shape, SparseTensor(start_indices, copy.deepcopy(mats), self.network.size), polyexp_const)

    def sum(self):
        polyexp_const = SparseTensorBlock([], [], 1, torch.tensor([1]))
        polyexp_const = 0.0
        mats = []
        if self.llist_flag:
            for i in self.llist:
                mats.append(torch.ones(self.network[i].size))
            start_indices = [torch.tensor([0]*len(self.initial_shape) + [self.network[i].start]) for i in self.llist]
        else:
            for i in range(self.start, self.end):
                mats.append(torch.ones(self.network[i].size))
            start_indices = [torch.tensor([0]*len(self.initial_shape) + [self.network[i].start]) for i in range(self.start, self.end)]
        return PolyExpSparse(self.network, SparseTensorBlock(start_indices, copy.deepcopy(mats), len(self.initial_shape)+1, torch.tensor(self.initial_shape+[self.network.size])), polyexp_const)
        return PolyExpSparse(self.network, initial_shape, SparseTensor(start_indices, copy.deepcopy(mats), self.network.size), polyexp_const)

    
    def avg(self):
        res = self.sum()
        size = 0
        if self.llist_flag:
            for i in self.llist:
                size += self.network[i].size
        else:
            for i in range(self.start, self.end):
                size += self.network[i].size
        new_mat = res.mat.binary(size, '/')
        new_const = res.const.binary(size, '/')
        # polyexp_const = SparseTensorBlock([0], [new_const], 1, torch.tensor([1]))
        return PolyExpSparse(self.network, new_mat, new_const)
        return PolyExpSparse(res.network, res.initial_shape, new_mat, new_const)
    
    def convert_to_poly(self, abs_elem):
        
        mats = []
        start_indices = []
        index = 0
        if self.llist:
            for i in self.llist:
                # assert(self.initial_shape == self.network[i].size)
                mat = torch.eye(self.network[i].size).reshape(*self.initial_shape, self.network[i].size, self.network[i].size)
                mats.append(SparseBlock(mat))
                start_indices.append(torch.tensor([0]*len(self.initial_shape) + [index, self.network[i].start]))
                index += self.network[i].size
            # start_indices = [torch.tensor([0]*len(self.initial_shape) + [0+, self.network[i].start]) for i in self.llist]
        else:
            raise Exception('NOT NEEDED')
            for i in range(self.start, self.end):
                # assert(self.initial_shape == self.network[i].size)
                mat = torch.eye(self.network[i].size).reshape(*self.initial_shape, self.network[i].size, self.network[i].size)
                mats.append(mat)
                # mats.append(torch.eye(self.initial_shape))
            start_indices = [torch.tensor([0]*len(self.initial_shape) + [self.network[i].start, self.network[i].start]) for i in range(self.start, self.end)]
        # const = 0.0
    
        polyexp_const = SparseTensorBlock([], [], len(self.initial_shape)+1, torch.tensor(self.initial_shape+[index]))
        # polyexp_const = 0.0
        return PolyExpSparse(self.network, SparseTensorBlock(start_indices, copy.deepcopy(mats), len(self.initial_shape)+2, torch.tensor(self.initial_shape+[index, abs_elem.get_poly_size()])), polyexp_const)
        