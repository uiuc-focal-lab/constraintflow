from common.abs_elem import Abs_elem
import torch 
import itertools
import copy

class PolyExp:
    def __init__(self, shapes, mat = None, const = 0.0):
        # self.abs_elem = abs_elem
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
            w = [1]*len(prev)
        for i in range(len(prev)):
            self.mat[prev[i][0]][prev[i][1]] = w[i].item()

    def add(self, p):
        self.const = self.const + p.const
        for i in range(len(self.mat)):
            self.mat[i] = self.mat[i] + p.mat[i]

    def minus(self, p):
        self.const = self.const - p.const 
        for i in range(len(self.mat)):
            self.mat[i] = self.mat[i] - p.mat[i]

    def mult(self, c):
        self.const = self.const*c 
        for i in range(len(self.mat)):
            self.mat[i] = self.mat[i]*c

    def map(self, abs_elem, f):
        res = PolyExp(self.shapes, const = self.const)
        for i in range(len(self.shapes)):
            indices = itertools.product(*[range(dim) for dim in self.shapes[i]])
            for j in indices:
                if self.mat[i][j] != 0:
                    tmp = f((i, j), self.mat[i][j], abs_elem)
                    res.add(tmp)
        return res  
    
    def traverse(self, abs_elem, neighbours, stop, priority, f):
        # print('here')
        res = self.copy()
        vertices = set()
        for i in range(len(self.mat)):
            for j in range(len(self.mat[i])):
                if self.mat[i][j]!=0:
                    if not stop((i, j), self.mat[i][j], abs_elem):
                        vertices.add((i, j))

        debug_ctr = 0
        while len(vertices)>0:
            debug_ctr += 1
            priority_vertices = [(priority(i, self.mat[i[0]][i[1]], abs_elem), i) for i in vertices]
            priority_vertices.sort()
            min = priority_vertices[0][0]
            temp = PolyExp(res.shapes)
            n = []
            # print(priority_vertices)
            for i in priority_vertices:
                if i[0] != min:
                    break 
                temp.mat[i[1][0]][i[1][1]] = res.mat[i[1][0]][i[1][1]]
                res.mat[i[1][0]][i[1][1]] = 0
                vertices.remove(i[1])
                n += neighbours[i[1]]
            # if debug_ctr==2:
            #     print(temp.mat)
            #     print(res.mat)
            temp = temp.map(abs_elem, f)
            # if debug_ctr==2:
            #     print('debug_ctr !!!!!!!!!!!!!!!!!!', debug_ctr)
            #     print(temp.mat)
            res.add(temp)
            # print(res.mat)
            # print(res.const)
            for i in n:
                if res.mat[i[0]][i[1]] == 0 or stop(i, res.mat[i[0]][i[1]], abs_elem):
                    n.remove(i)
            vertices = vertices.union(n) 
        return res 