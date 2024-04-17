import torch 
import copy
import time 

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
        # if not self.nlist_flag:
        counter = Nlist.network.input_size
        for k, layer in enumerate(Nlist.network):
            if counter == self.start:
                if elem == 'weight' or elem == 'w':
                    return layer.weight
                elif elem == 'bias' or elem == 'b':
                    return layer.bias
                elif elem == 'layer':
                    return torch.ones(layer.bias.shape)*k
            counter += layer.size
        # else:
        #     raise Exception('NOT IMPLEMENTED')
                
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
        res = PolyExpNew(N, polyexp_coeff, polyexp_const)
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
        res = PolyExpNew(N, polyexp_coeff, polyexp_const)
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
            return PolyExpNew(self.size, mat, const)
        else:
            mat = torch.zeros(self.end-self.start+1, self.size)
            const = torch.zeros(self.end-self.start+1)
            for i in range(self.end-self.start+1):
                mat[i, self.start+i] = 1
            return PolyExpNew(self.size, mat, const)

class PolyExpNew:
    def __init__(self, size, mat, const):
        if mat==None:
            mat = 0.0
        self.size = size 
        self.mat = mat 
            
        self.const = const
        # assert(self.const.dim()==1)
        self.debug_flag = False 
        if not isinstance(const, torch.Tensor) and isinstance(mat, torch.Tensor):
            if self.mat.dim()>1:
                self.const = torch.ones((self.mat.size(0))) * const 
        # if mat==None:
        #     self.mat = torch.zeros((self.const.shape[0], size))
        #     print(self.mat.size)
        

    def copy(self):
        return copy.deepcopy(self)
    
    def get_mat(self, abs_elem):
        if not isinstance(self.mat, torch.Tensor):
            return self.mat
        live_neurons = torch.nonzero(abs_elem.d['t']).flatten()
        # print(self.mat)
        return self.mat[:, live_neurons]
    
    def get_const(self):
        # assert(self.const.dim()==1)
        return self.const