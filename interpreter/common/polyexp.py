from common.abs_elem import Abs_elem
import torch 
import itertools
import copy

class PolyExp:
    def __init__(self, shapes, mat = None, const = 0.0):
        self.shapes = shapes 
        self.mat = mat 
        if self.mat == None:
            self.mat = []
            for i in self.shapes:
                self.mat.append(torch.zeros(i))
        self.const = const 

    def copy(self):
        res = PolyExp(self.shapes, [], const=self.const)
        for i in range(len(self.mat)):
            res.mat.append(self.mat[i].clone())
        return res 

    def get_const(self):
        return self.const

    def populate(self, const, prev, w = None):
        self.const = const 
        if w == None:
            #w = [1]*len(prev)
            w = torch.ones(len(prev))
        if(isinstance(prev, tuple)):
            prev = [prev]
        for i in range(len(prev)):
            self.mat[prev[i][0]][prev[i][1]] = w[i].item()
        return self

    def add(self, p):
        if isinstance(p, PolyExp):
            self.const = self.const + p.const
            for i in range(len(self.mat)):
                self.mat[i] = self.mat[i] + p.mat[i]
        else:
            # print((p.shape))
            # print(self.const)
            # print(p)
            self.const += p 
        return self

    def minus(self, p):
        if isinstance(p, PolyExp):
            self.const = self.const - p.const 
            for i in range(len(self.mat)):
                self.mat[i] = self.mat[i] - p.mat[i]
        else:
            self.const = self.const - p
        return self

    def mult(self, c):
        self.const = self.const*c 
        for i in range(len(self.mat)):
            self.mat[i] = self.mat[i]*c
        return self

    def map(self, abs_elem, neighbours, f):
        res = PolyExp(self.shapes, const = self.const)
        for i in range(len(self.shapes)):
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
            indices = itertools.product(*[range(dim) for dim in self.shapes[i]])
            for j in indices:
                if self.mat[i][j]!=0:
                    if(callable(stop)):
                        if not stop((i, j), self.mat[i][j], abs_elem):
                            vertices.add((i, j))
                    else:
                        if not stop:
                            vertices.add((i, j))

        debug_ctr = 0
        while len(vertices)>0:
            debug_ctr += 1
            priority_vertices = [(priority(i, self.mat[i[0]][i[1]], abs_elem), i) for i in vertices]
            priority_vertices.sort()
            min = priority_vertices[0][0]
            temp = PolyExp(res.shapes)
            n = []
            for i in priority_vertices:
                if i[0] != min:
                    break 
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