from common.abs_elem import Abs_elem
from common.polyexp import PolyExp
import torch 



def simplify_lower(n, c, abs_elem):
    # print(n)
    # print(abs_elem.d['u'][n[0]].shape)
    res = c * abs_elem.d['l'][n[0]][n[1]] if c >= 0 else c * abs_elem.d['u'][n[0]][n[1]]
    return PolyExp(abs_elem.shapes, const = res)
    
def simplify_upper(n, c, abs_elem):
    res = c * abs_elem.d['u'][n[0]][n[1]] if c >= 0 else c * abs_elem.d['l'][n[0]][n[1]]
    return PolyExp(abs_elem.shapes, const = res) 

def replace_lower(n, c, abs_elem):
    res = abs_elem.d['L'][n[0]][n[1]].copy() if c >= 0 else abs_elem.d['U'][n[0]][n[1]].copy()
    # print(res.)
    res.mult(c)
    return res 

def replace_upper(n, c, abs_elem):
    res = abs_elem.d['U'][n[0]][n[1]].copy() if c >= 0 else abs_elem.d['L'][n[0]][n[1]].copy()
    res.mult(c)
    return res 

def stop(n, c, abs_elem):
    return False 

def priority(n, c, abs_elem):
    return 1

def backsubs_lower(p, n, abs_elem, neighbours):
    res = p.traverse(abs_elem, neighbours, stop, priority, replace_lower)
    return res.map(abs_elem, simplify_lower)

def backsubs_upper(p, n, abs_elem, neighbours):
    res = p.traverse(abs_elem, neighbours, stop, priority, replace_upper)
    return res.map(abs_elem, simplify_upper)

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
        l_new = temp.map(abs_elem, simplify_lower)
        u_new = temp.map(abs_elem, simplify_upper)
        return l_new.get_const(), u_new.get_const() 
    
class CflowDeepPoly(Transformer): 
    def relu(self, abs_elem, neighbours, prev, curr):
        l = abs_elem.d['l'][prev[0]][prev[1]]
        u = abs_elem.d['u'][prev[0]][prev[1]]
        L = abs_elem.d['L'][prev[0]][prev[1]]
        U = abs_elem.d['U'][prev[0]][prev[1]]
        l_new = None 
        u_new = None
        L_new = None
        U_new = None
        if l>=0:
            l_new = l 
            u_new = u
            L_new = PolyExp(L.widths)
            U_new = PolyExp(U.widths)
            L_new.mat[prev[0]][prev[1]] = 1.0 
            U_new.mat[prev[0]][prev[1]] = 1.0 
        elif u<=0:
            l_new = 0.0
            u_new = 0.0
            L_new = PolyExp(L.widths)
            U_new = PolyExp(U.widths)
        else:
            l_new = 0.0 
            u_new = u 
            L_new = PolyExp(L.widths) 
            U_new = PolyExp(U.widths)
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