# from common.abs_elem import Abs_elem
# from common.nlist import Nlist
import torch 
import itertools
import copy
import time 

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
        polyexp_const = torch.zeros(n)
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

class PolyExp:
    def __init__(self, shapes, mat = None, const = 0.0):
        self.shapes = shapes 
        self.flag = [False]*len(self.shapes)
        self.mat = mat 
        if self.mat == None or self.mat == []:
            self.mat = []
            for i in range(len(self.shapes)):
                self.mat.append(None)
        else:
            for i in range(len(self.shapes)):
                if self.mat[i] != None:
                    self.flag[i] = True 
        if isinstance(const, torch.Tensor):
            self.const = const.item()
        else:
            self.const = const 

    def copy(self):
        res = PolyExp(self.shapes, const=copy.deepcopy(self.const))
        for i in range(len(self.mat)):
            if self.flag[i]:
                res.mat[i] = self.mat[i].clone()
        res.flag = copy.deepcopy(self.flag)
        return res 

    def get_const(self):
        return self.const

    def populate(self, const, prev, w = None):
        if isinstance(const, torch.Tensor):
            self.const = const.item()
        else:
            self.const = const  
        if w == None:
            w = torch.ones(len(prev))
        if(isinstance(prev, tuple)):
            prev = [prev]
        for i in range(len(prev)):
            if self.flag[prev[i][0]] == False:
                self.flag[prev[i][0]] = True 
                self.mat[prev[i][0]] = torch.zeros(self.shapes[prev[i][0]])
            self.mat[prev[i][0]][prev[i][1]] = w[i].item()
        return self

    def add(self, p):
        if isinstance(p, PolyExp):
            self.const = self.const + p.const
            for i in range(len(self.mat)):
                if self.flag[i] and p.flag[i]:
                    self.mat[i] = self.mat[i] + p.mat[i]
                elif p.flag[i]:
                    self.flag[i] = True
                    self.mat[i] = p.mat[i].clone()
        else:
            if isinstance(p, torch.Tensor):
                self.const += p.item()
            else:
                self.const += p 
        return self

    def minus(self, p):
        if isinstance(p, PolyExp):
            self.const = self.const + p.const
            for i in range(len(self.mat)):
                if self.flag[i] and p.flag[i]:
                    self.mat[i] = self.mat[i] - p.mat[i]
                elif p.flag[i]:
                    self.flag[i] = True
                    self.mat[i] = -1 * p.mat[i].clone()
        else:
            if isinstance(p, torch.Tensor):
                self.const -= p.item()
            else:
                self.const -= p 
        return self

    def mult(self, c):
        if isinstance(c, torch.Tensor):
            c = c.item()
        self.const = self.const*c 
        for i in range(len(self.mat)):
            if self.flag[i]:
                self.mat[i] = self.mat[i]*c
        return self

    def map(self, abs_elem, neighbours, f):
        res = PolyExp(self.shapes, const = self.const)
        for i in range(len(self.shapes)):
            if self.flag[i] == False:
                continue 
            indices = itertools.product(*[range(dim) for dim in self.shapes[i]])
            for j in indices:
                if self.mat[i][j] != 0:
                    tmp = f((i, j), self.mat[i][j], abs_elem, neighbours)
                    res.add(tmp)
        return res  
    
    def traverse(self, abs_elem, neighbours, stop, priority, f):
        res = self.copy()
        vertices = set()
        for i in range(len(self.mat)):
            if self.flag[i] == False:
                continue 
            indices = itertools.product(*[range(dim) for dim in self.shapes[i]])
            for j in indices:
                if self.mat[i][j]!=0:
                    if(callable(stop)):
                        if not stop((i, j), self.mat[i][j], abs_elem):
                            vertices.add((i, j))
                    else:
                        if not stop:
                            vertices.add((i, j))

        while len(vertices)>0:
            priority_vertices = [(priority(i, res.mat[i[0]][i[1]], abs_elem), i) for i in vertices]
            priority_vertices.sort()
            min = priority_vertices[0][0]
            temp = PolyExp(res.shapes)
            n = []
            for i in priority_vertices:
                if i[0] != min:
                    break 
                if temp.flag[i[1][0]] == False:
                    temp.flag[i[1][0]] = True 
                    temp.mat[i[1][0]] = torch.zeros(temp.shapes[i[1][0]])
                temp.mat[i[1][0]][i[1][1]] = res.mat[i[1][0]][i[1][1]]
                res.mat[i[1][0]][i[1][1]] = 0
                vertices.remove(i[1])
                n += neighbours[i[1]]
            temp = temp.map(abs_elem, neighbours, f)
            res.add(temp)
            for i in n:
                if(callable(stop)):
                    if res.mat[i[0]][i[1]] == 0 or stop(i, res.mat[i[0]][i[1]], abs_elem):
                        n.remove(i)
                else:
                    if res.mat[i[0]][i[1]] == 0 or stop:
                        n.remove(i)
            vertices = vertices.union(n) 
        return res 
    
class PolyExpNew:
    def __init__(self, size, mat, const):
        self.size = size 
        self.mat = mat 
        self.const = const 
        self.debug_flag = False 
        if not isinstance(const, torch.Tensor):
            if self.mat.dim()>1:
                self.const = torch.ones((self.mat.size(0), 1)) * const 

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
        res_mat = []
        res_const = []
        res = []
        time_taken = []
        for i in range(n):
            # print(i, n, 'profiling')
            c = self.mat[i]
            checkpoint = 0
            t_time = time.time()
            if len(time_taken) <= checkpoint:
                time_taken.append(0)
            time_taken[checkpoint] += time.time()-t_time
            # print('check point', checkpoint, time.time()-t_time)
            checkpoint += 1
            t_time = time.time()
            local_res = f(indices, c, abs_elem, neighbours)
            if len(time_taken) <= checkpoint:
                time_taken.append(0)
            time_taken[checkpoint] += time.time()-t_time
            # print('check point', checkpoint, time.time()-t_time)
            checkpoint += 1
            t_time = time.time()
            if isinstance(local_res, torch.Tensor):
                c = c.reshape(local_res.shape)
                local_res = torch.where(c!=0, local_res, 0)
                local_res = torch.sum(local_res, dim=0)
                res.append(local_res)
            else:
                c1 = c.unsqueeze(0).repeat(local_res.mat.size(0), 1).T
                c2 = c.reshape(local_res.const.shape)
                local_res.mat = torch.where(c1!=0, local_res.mat, 0)
                local_res.const = torch.where(c2!=0, local_res.const, 0)
                local_res.mat = torch.sum(local_res.mat, dim=0)
                local_res.const = torch.sum(local_res.const, dim=0)
                res_mat.append(local_res.mat)
                res_const.append(local_res.const)
            if len(time_taken) <= checkpoint:
                time_taken.append(0)
            time_taken[checkpoint] += time.time()-t_time
            # print('check point', checkpoint, time.time()-t_time)
            checkpoint += 1
            t_time = time.time()
            # print()
        print(f, n, time_taken)
        if len(res)>0:
            res = torch.stack(res)
            self.const = self.const.reshape(-1,1)

            res = res + self.const
        else:
            res_mat = torch.stack(res_mat)    
            res_const = torch.stack(res_const)    
            res_const = res_const + self.const
            res = PolyExpNew(size=self.size, mat=res_mat, const=res_const)
        return res
    
    def traverse(self, stop, priority, f, abs_elem, neighbours):
        if not callable(stop):
            if stop:
                return self.copy()
        indices = torch.arange(self.size)
        res = self.copy()
        a = res.mat != 0
        if callable(stop):
            b = stop(indices, res.mat, abs_elem, neighbours)
            b = b!=True
        else:
            b = torch.ones(self.size, dtype=torch.bool)
        vertices = a & b 
        while(vertices.any()):
            all_priorities = priority(indices, res.mat, abs_elem, neighbours)
            masked_priorities = all_priorities[vertices]
            min_priority = masked_priorities.min()
            priority_vertices = all_priorities == min_priority
            priority_vertices[~vertices] = False
            temp = res.copy()
            temp.mat = temp.mat * priority_vertices
            res.mat = res.mat - temp.mat
            temp.const = 0
            temp = temp.map(f, abs_elem, neighbours)
            
            res = res.add(temp)
            for i in range(priority_vertices.shape[1]):
                # print(i)
                n = neighbours[i]
                for j in range(priority_vertices.shape[0]):
                    if priority_vertices[j][i]:
                        vertices[j][n] = True 
                        vertices[j][i] = False 
        return res 

    def traverse_old(self, f, stop, priority, abs_elem, neighbours):
        res = self.copy()
        vertices = set()
        for i in range(len(self.mat)):
            if self.flag[i] == False:
                continue 
            indices = itertools.product(*[range(dim) for dim in self.shapes[i]])
            for j in indices:
                if self.mat[i][j]!=0:
                    if(callable(stop)):
                        if not stop((i, j), self.mat[i][j], abs_elem):
                            vertices.add((i, j))
                    else:
                        if not stop:
                            vertices.add((i, j))

        while len(vertices)>0:
            priority_vertices = [(priority(i, res.mat[i[0]][i[1]], abs_elem), i) for i in vertices]
            priority_vertices.sort()
            min = priority_vertices[0][0]
            temp = PolyExp(res.shapes)
            n = []
            for i in priority_vertices:
                if i[0] != min:
                    break 
                if temp.flag[i[1][0]] == False:
                    temp.flag[i[1][0]] = True 
                    temp.mat[i[1][0]] = torch.zeros(temp.shapes[i[1][0]])
                temp.mat[i[1][0]][i[1][1]] = res.mat[i[1][0]][i[1][1]]
                res.mat[i[1][0]][i[1][1]] = 0
                vertices.remove(i[1])
                n += neighbours[i[1]]
            temp = temp.map(abs_elem, neighbours, f)
            res.add(temp)
            for i in n:
                if(callable(stop)):
                    if res.mat[i[0]][i[1]] == 0 or stop(i, res.mat[i[0]][i[1]], abs_elem):
                        n.remove(i)
                else:
                    if res.mat[i[0]][i[1]] == 0 or stop:
                        n.remove(i)
            vertices = vertices.union(n) 
        return res 
    
    def add(self, p):
        N1 = self.size 
        n1 = self.const.size(0)
        if isinstance(p, PolyExpNew):
            N2 = p.size 
            n2 = p.const.size(0)
            if N1==N2 and n1==n2:
                res = self.copy()
                res.mat = res.mat + p.mat 
                res.const = res.const + p.const 
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
                    res.const = res.const + p 
                    return res 
                else:
                    print(N1)
                    print(n1, n2)
                    raise Exception('Size Mismatch in PolyExpNew Add - 2')
            else:
                res = self.copy()
                res.const = res.const + p 
                return res