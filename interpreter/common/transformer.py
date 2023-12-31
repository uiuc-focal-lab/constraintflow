from common.abs_elem import Abs_elem
from common.polyexp import PolyExp
from common.symexp import SymExp
import torch 



def simplify_lower(n, c, abs_elem, neighbours):
    return c * abs_elem.get_elem('l', n) if c>=0 else c * abs_elem.get_elem('u', n)
    
def simplify_upper(n, c, abs_elem, neighbours):
    return c * abs_elem.get_elem('u', n) if c>=0 else c * abs_elem.get_elem('l', n)

def deepz_lower(n, c):
    return 1.0 if c>=0 else -1.0
    
def deepz_upper(n, c):
    return -1.0 if c>=0 else 1.0

def replace_lower(n, c, abs_elem, neighbours):
    res = abs_elem.get_elem('L', n) if c>=0 else abs_elem.get_elem('U', n)
    res.mult(c)
    return res 

def replace_upper(n, c, abs_elem, neighbours):
    res = abs_elem.get_elem('U', n) if c>=0 else abs_elem.get_elem('L', n)
    res.mult(c)
    return res 

def stop(n, c, abs_elem):
    return False 

def priority(n, c, abs_elem):
    return 1

def backsubs_lower(p, n, abs_elem, neighbours):
    res = p.traverse(abs_elem, neighbours, stop, priority, replace_lower)
    return res.map(abs_elem, neighbours, simplify_lower)

def backsubs_upper(p, n, abs_elem, neighbours):
    res = p.traverse(abs_elem, neighbours, stop, priority, replace_upper)
    return res.map(abs_elem, neighbours, simplify_upper)

class Transformer:
    def relu(self, abs_elem, neighbours, prev, curr):
        pass 

    def tanh(self, abs_elem, neighbours, prev, curr):
        pass 

    def fc(self, abs_elem, neighbours, prev, curr, w, b):
        pass 

class MyInterval(Transformer):
    def relu(self, abs_elem, neighbours, prev, curr):
        l = abs_elem.d['l'][prev[0]][prev[1]]
        u = abs_elem.d['u'][prev[0]][prev[1]]
        return max(l, 0), max(u, 0)
    
    def tanh(self, abs_elem, neighbours, prev, curr):
        l = abs_elem.d['l'][prev[0]][prev[1]]
        u = abs_elem.d['u'][prev[0]][prev[1]]
        l_new = torch.tanh(l)
        u_new = torch.tanh(u)
        return l_new, u_new 
    
    def fc(self, abs_elem, neighbours, prev, curr, w, b):
        l = [abs_elem.d['l'][i[0]][i[1]] for i in prev]
        u = [abs_elem.d['u'][i[0]][i[1]] for i in prev]
        l_new = b
        for i in range(len(prev)):
            if w[i]>0:
                l_new = l_new + l[i]*w[i]
            else:
                l_new = l_new + u[i]*w[i]
        u_new = b
        for i in range(len(prev)):
            if w[i]>0:
                u_new = u_new + u[i]*w[i]
            else:
                u_new = u_new + l[i]*w[i]
        return l_new, u_new
 
class CflowInterval(Transformer): 
    def relu(self, abs_elem, neighbours, prev, curr):
        l = abs_elem.d['l'][prev[0]][prev[1]]
        u = abs_elem.d['u'][prev[0]][prev[1]]
        l_new = None 
        u_new = None
        if l>=0:
            l_new = l 
            u_new = u 
        elif u<=0:
            l_new = 0.0
            u_new = 0.0
        else:
            l_new = 0.0 
            u_new = u 
        return l_new, u_new 
    
    def tanh(self, abs_elem, neighbours, prev, curr):
        l = abs_elem.d['l'][prev[0]][prev[1]]
        u = abs_elem.d['u'][prev[0]][prev[1]]
        l_new = torch.tanh(l)
        u_new = torch.tanh(u)
        return l_new, u_new 

    def fc(self, abs_elem, neighbours, prev, curr, w, b):
        temp = PolyExp(abs_elem.shapes)
        temp.populate(b, prev, w)
        l_new = temp.copy().map(abs_elem, neighbours, simplify_lower)
        u_new = temp.copy().map(abs_elem, neighbours, simplify_upper)
        return l_new.get_const(), u_new.get_const() 
    
class CflowDeepPoly(Transformer): 
    def relu(self, abs_elem, neighbours, prev, curr):
        l = abs_elem.get_elem('l', prev)
        u = abs_elem.get_elem('u', prev)
        L = abs_elem.get_elem('L', prev)
        U = abs_elem.get_elem('U', prev)
        l_new = None 
        u_new = None
        L_new = None
        U_new = None
        if l>=0:
            l_new = l 
            u_new = u
            L_new = PolyExp(L.shapes)
            U_new = PolyExp(U.shapes)
            L_new.mat[prev[0]][prev[1]] = 1.0 
            U_new.mat[prev[0]][prev[1]] = 1.0 
        elif u<=0:
            l_new = 0.0
            u_new = 0.0
            L_new = PolyExp(L.shapes)
            U_new = PolyExp(U.shapes)
        else:
            l_new = 0.0 
            u_new = u 
            L_new = PolyExp(L.shapes) 
            U_new = PolyExp(U.shapes)
            slope = u / (u-l)
            intercept = -u*l / (u-l)
            U_new.const = intercept
            U_new.mat[prev[0]][prev[1]] = slope
        return l_new, u_new, L_new, U_new 

    def fc(self, abs_elem, neighbours, prev, curr, w, b):
        temp = PolyExp(abs_elem.shapes)
        temp.populate(b, prev, w)
        L_new = temp.copy()
        U_new = temp.copy()
        l_new = (backsubs_lower(temp, curr, abs_elem, neighbours)).get_const() 
        u_new = (backsubs_upper(temp, curr, abs_elem, neighbours)).get_const()
        return l_new, u_new, L_new, U_new
    
class CflowDeepZ(Transformer): 
    def relu(self, abs_elem, neighbours, prev, curr):
        l = abs_elem.get_elem('l', prev)
        u = abs_elem.get_elem('u', prev)
        Z = abs_elem.get_elem('Z', prev)
        l_new = None 
        u_new = None
        Z_new = None
        if l>=0:
            l_new = l 
            u_new = u
            Z_new = Z.copy()
        elif u<=0:
            l_new = 0.0
            u_new = 0.0
            Z_new = SymExp()
        else:
            l_new = 0.0 
            u_new = u 
            Z_new = SymExp()
            Z_new.new_symbol()
            Z_new.populate(const=u/2, coeff=u/2)
        return l_new, u_new, Z_new  

    def fc(self, abs_elem, neighbours, prev, curr, w, b):
        Z_new = SymExp(const = b)
        for i, p in enumerate(prev):
            tmp = abs_elem.get_elem('Z', p)
            tmp.mult(w[i])
            Z_new.add(tmp)
        l_new = Z_new.map(deepz_lower).const
        u_new = Z_new.map(deepz_upper).const
        return l_new, u_new, Z_new 