from common.abs_elem import Abs_elem
import torch 
import itertools
import copy

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
            print('here')
            kdsjxg
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

    # def traverse(self, abs_elem, neighbours, stop, priority, f):
    #     res = self.copy()
    #     vertices = set()
    #     for i in range(len(self.mat)):
    #         if self.flag[i] == False:
    #             continue 
    #         indices = itertools.product(*[range(dim) for dim in self.shapes[i]])
    #         for j in indices:
    #             if self.mat[i][j]!=0:
    #                 if(callable(stop)):
    #                     if not stop((i, j), self.mat[i][j], abs_elem):
    #                         vertices.add((i, j))
    #                 else:
    #                     if not stop:
    #                         vertices.add((i, j))

    #     while len(vertices)>0:
    #         priority_vertices = [(priority(i, self.mat[v[0]][v[1]], abs_elem), v) for v in vertices]
    #         priority_vertices.sort()
    #         min = priority_vertices[0][0]
    #         temp = PolyExp(res.shapes)
    #         n = []
    #         for v in priority_vertices:
    #             if v[0] != min:
    #                 break 
    #             temp.mat[v[1][0]][v[1][1]] = res.mat[v[1][0]][v[1][1]]
    #             res.mat[v[1][0]][v[1][1]] = 0
    #             vertices.remove(v[1])
    #             n += neighbours[v[1]]
    #         temp = temp.map(abs_elem, neighbours, f)
    #         res.add(temp)
    #         for i in n:
    #             if(callable(stop)):
    #                 if res.mat[i[0]][i[1]] == 0 or stop(i, res.mat[i[0]][i[1]], abs_elem):
    #                     n.remove(i)
    #             else:
    #                 if res.mat[i[0]][i[1]] == 0 or stop:
    #                     n.remove(i)
    #         vertices = vertices.union(n) 
    #     return res 