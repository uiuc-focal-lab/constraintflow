from common.abs_elem import Abs_elem
import torch 
import itertools
import copy

class SymExp:
    def __init__(self, count = 0, mat = [], const = 0.0):
        if count==0 and len(mat)>0 :
            raise ValueError('Incorrect symbolic expression')
        self.count = count 
        self.mat = mat 
        self.const = const 

    def copy(self):
        res = SymExp(self.count, [], const=self.const)
        res.mat = copy.deepcopy(self.mat)
        return res 

    def get_const(self):
        return self.const
    
    def new_symbol(self):
        self.count += 1
        self.mat.append(0)

    def populate(self, const=0, n=-1, coeff=0):
        self.const = const 
        self.mat[n] = coeff
    
    def add(self, p):
        if isinstance(p, SymExp):
            self.const = self.const + p.const
            if self.count < p.count:
                tmp = [0]*(p.const - self.const)
                self.mat += tmp 
                self.count = p.count
            for i in range(self.count):
                self.mat[i] = self.mat[i] + p.mat[i]
        else:
            self.count += p

    def minus(self, p):
        if isinstance(p, SymExp):
            self.const = self.const - p.const 
            if self.count < p.count:
                tmp = [0]*(p.const - self.const)
                self.mat += tmp 
                self.count = p.count
            for i in range(self.count):
                self.mat[i] = self.mat[i] - p.mat[i]
        else:
            self.const = self.const - p

    def mult(self, c):
        self.const = self.const*c 
        for i in range(len(self.mat)):
            self.mat[i] = self.mat[i]*c
    
    def map(self, abs_elem, f):
        res = SymExp(self.count, const = self.const)
        for i in range(len(self.count)):
            if self.mat[i] != 0:
                tmp = f(i, self.mat[i])
                res.add(tmp)
        return res 