import numpy as np
import torch

class Abs_elem:
    def __init__(self, l, u):
        self.l = l
        self.u = u

class PolyExp:
    def __init__(self, shape, mat = None, const = 0):
        # self.abs_elem = abs_elem
        self.shape = shape 
        self.mat = mat 
        if not self.mat:
            self.mat = torch.zeros(self.shape)
        self.const = const 

    def populate(self, const, prev, w = None):
        self.const = const 
        if not w:
            w = [1]*len(prev)
        for i in range(len(prev)):
            self.mat[prev[i]] = w[i]

    def add(self, p):
        self.const = self.const + p.const
        self.mat = self.mat + p.mat 

    def minus(self, p):
        self.const = self.const - p.const 
        self.mat = self.mat - p.mat 

    def map(self, abs_elem, f):
        res = Abs_elem(self.abs_elem)
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                res.add(f((i, j), self.mat[(i, j)], abs_elem))
        return res  

def simplify_lower(neuron, coeff, abs_elem):
    res = coeff * abs_elem.l[neuron] if coeff >= 0 else coeff * abs_elem.u[neuron]
    return Abs_elem(abs_elem.shape, const = res)
    
def simplify_upper(neuron, coeff, abs_elem):
    res = coeff * abs_elem.u[neuron] if coeff >= 0 else coeff * abs_elem.l[neuron]
    return Abs_elem(abs_elem.shape, const = res)

class Domain:
    def __init__(self, abs_elem):
        self.abs_elem = abs_elem 

    def fc(self, prev, curr):
        pass 

    def relu(self, prev, curr):
        pass 

class MyInterval(Domain):
    def relu(self, prev, curr):
        l = self.abs_elem.l[prev]
        u = self.abs_elem.u[prev]
        return max(l, 0), max(u, 0)
    
    def fc(self, prev, curr, w, b):
        l = [self.abs_elem.l[i] for i in prev]
        u = [self.abs_elem.u[i] for i in prev]
        l_new = b
        for i in range(len(prev)):
            if w[i]>0:
                l_new += l[i]*w[i]
            else:
                l_new += u[i]*w[i]
        u_new = b
        for i in range(len(prev)):
            if w[i]>0:
                u_new += u[i]*w[i]
            else:
                u_new += l[i]*w[i]
        return l_new, u_new
 
class CflowInterval(Domain): 
    def relu(self, prev):
        l = self.abs_elem.l[prev]
        u = self.abs_elem.u[prev]
        l_new = None 
        u_new = None
        if l>=0:
            l_new = l 
            u_new = u 
        elif u<=0:
            l_new = 0
            u_new = 0
        else:
            l_new = 0 
            u_new = u 
        return l_new, u_new 

    def fc(self, prev, w, b):
        temp = PolyExp()
        temp.populate(b, prev, w)
        l_new = temp.map(self.abs_elem, simplify_lower)
        u_new = temp.map(self.abs_elem, simplify_upper)
        return l_new, u_new 
        
