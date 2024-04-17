# from common.abs_elem import Abs_elem
# from common.nlist import Nlist
import torch 
import itertools
import copy
import time 
# from common.transformer import replace_lower_new

class Nlist:
    def __init__(self, size=0, start=None, end=None, nlist=None):
        self.size = size
        self.start = start
        self.end = end
        self.nlist = nlist
        self.nlist_flag = True 
        if nlist==None:
            self.nlist_flag = False

    def dot(self, w):
        if isinstance(w, torch.Tensor):
            n = w.size(0)
        elif isinstance(w, list):
            n = len(w)
        N = self.size 
        polyexp_const = torch.zeros((n,1))
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
            # polyexp_coeff[self.nlist] = w
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

    
class PolyExpNew:
    def __init__(self, size, mat, const):
        self.size = size 
        self.mat = mat 
        self.const = const
        self.debug_flag = False 
        if not isinstance(const, torch.Tensor):
            if self.mat.dim()>1:
                self.const = torch.ones((self.mat.size(0))) * const 
        

    def copy(self):
        return copy.deepcopy(self)
    
    def mult(self, c):
        res = self.copy()
        res.const = res.const * c 
        res.mat = res.mat * c 
        return res 
    
    def map(self, f, abs_elem, neighbours):
        indices = Nlist(size=self.size, nlist=torch.arange(self.size))
        n = self.mat.size(0)
        # indices = indices.unsqueeze(0).repeat(n, 1)

        res = f(indices, self.mat, abs_elem, neighbours)
        if isinstance(res, PolyExpNew):
            res.const = res.const + self.const 
        else:
            res = res + self.const.reshape(res.shape)
        return res
    
    def map_compiler(self, f, abs_elem, neighbours):
        indices = Nlist(size=self.size, nlist=torch.arange(self.size))
        n = self.mat.size(0)
        # indices = indices.unsqueeze(0).repeat(n, 1)

        res = f(indices, self.mat, abs_elem, neighbours)
        if isinstance(res, PolyExpNew):
            # print(res.mat)
            # print(res.const)
            # fj
            res.mat = res.mat.sum(dim=-2)
            res.const = res.const.sum(dim=-1)
            res.const = res.const + self.const.reshape(res.const.shape)
        else:
            res = res.sum(dim=-1)
            res = res + self.const.reshape(res.shape)
        return res

    def traverse(self, stop, priority, f, abs_elem, neighbours, f2=None):
        if not callable(stop):
            if stop:
                return self.copy()
        indices = torch.arange(self.size)
        res = self.copy()
        while True:
            s_time = time.time()
            a = res.mat != 0
            if callable(stop):
                b = stop(indices, res.mat, abs_elem, neighbours)
                b = b!=True
            else:
                b = torch.ones(self.size, dtype=torch.bool)
            vertices = a & b
            if not(vertices.any()):
                break  
            all_priorities = priority(indices, res.mat, abs_elem, neighbours)
            masked_priorities = all_priorities[vertices]
            min_priority = masked_priorities.min()
            priority_vertices = all_priorities == min_priority
            priority_vertices[~vertices] = False
            temp = PolyExpNew(res.size, res.mat.clone() * priority_vertices, 0)
            res.mat = res.mat - temp.mat
            print('extra time inside traverse after map', time.time()-s_time)
            s_time = time.time()
            temp = temp.map_compiler(f, abs_elem, neighbours)
            # print(res.mat.shape)
            # print(res.const.shape)
            # print(temp.mat)
            # print(temp.const)
            # lkdh
            print('inside traverse after map', time.time()-s_time)
            s_time = time.time()
            
            res = res.add(temp)
            
            # vertices_temp = vertices ^ priority_vertices
            # for i in range(priority_vertices.shape[1]):
            #     # print(i)
            #     n = neighbours[i]
            #     for j in range(priority_vertices.shape[0]):
            #         if priority_vertices[j][i]:
            #             vertices[j][n] = True 
            #             vertices[j][i] = False 
            # # test = (vertices == vertices_temp).all()
            # # assert(test)
            # print('inside traverse after updating vertices', time.time()-s_time)
            # s_time = time.time()
        return res 

    
    def add(self, p):
        N1 = self.size 
        n1 = self.const.size(0)
        if isinstance(p, PolyExpNew):
            N2 = p.size 
            n2 = p.const.size(0)
            if N1==N2 and n1==n2:
                res = self.copy()
                # print(res.mat.shape)
                # print(p.mat.shape)
                # klj
                res.mat = res.mat + p.mat 
                res.const = res.const + p.const.reshape(self.const.shape)
                return res 
            else:
                print(N1, n1)
                print(N2, n2)
                raise Exception('Size Mismatch in PolyExpNew Add - 1')
        else:
            if isinstance(p, torch.Tensor):
                n2 = p.size(0)
                if n1==n2:
                    res = self.copy()
                    res.const = res.const + p.reshape(res.const.shape)
                    return res 
                else:
                    print(N1)
                    print(n1, n2)
                    raise Exception('Size Mismatch in PolyExpNew Add - 2')
            else:
                res = self.copy()
                res.const = res.const + p.reshape(self.const.shape)
                return res