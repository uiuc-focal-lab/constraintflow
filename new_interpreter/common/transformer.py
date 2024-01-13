from common.abs_elem import Abs_elem
from common.polyexp import PolyExp, PolyExpNew, Nlist
# from common.nlist import Nlist
from common.symexp import SymExp
import torch 
import time 



# def simplify_lower(n, c, abs_elem, neighbours):
#     return c * abs_elem.get_elem('l', n) if c>=0 else c * abs_elem.get_elem('u', n)
    
# def simplify_upper(n, c, abs_elem, neighbours):
#     return c * abs_elem.get_elem('u', n) if c>=0 else c * abs_elem.get_elem('l', n)

def deepz_lower(n, c):
    return 1.0 if c>=0 else -1.0
    
def deepz_upper(n, c):
    return -1.0 if c>=0 else 1.0

# def replace_lower(n, c, abs_elem, neighbours):
#     res = abs_elem.get_elem('L', n) if c>=0 else abs_elem.get_elem('U', n)
#     res.mult(c)
#     return res 

# def replace_upper(n, c, abs_elem, neighbours):
#     res = abs_elem.get_elem('U', n) if c>=0 else abs_elem.get_elem('L', n)
#     res.mult(c)
#     return res 

# def stop(n, c, abs_elem):
#     return False 

# def priority(n, c, abs_elem):
#     return 1

# def backsubs_lower(p, n, abs_elem, neighbours):
#     res = p.traverse(abs_elem, neighbours, stop, priority, replace_lower)
#     return res.map(abs_elem, neighbours, simplify_lower)

# def backsubs_upper(p, n, abs_elem, neighbours):
#     res = p.traverse(abs_elem, neighbours, stop, priority, replace_upper)
#     return res.map(abs_elem, neighbours, simplify_upper)

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
    
class CflowNewDeepPoly(Transformer): 
    def relu(self, abs_elem, neighbours, prev, curr):
        N = abs_elem.d['l'].size(0)
        l = abs_elem.get_elem_new('l', prev)
        u = abs_elem.get_elem_new('u', prev)

        l_new = torch.where(l>=0, l, torch.where(u<=0, 0, 0))
        u_new = torch.where(l>=0, u, torch.where(u<=0, 0, u))
        
        prev_polyexp_mat = prev.sum().mat 
        prev_polyexp_const = prev.sum().const.reshape(-1,1)

        
        zero_mat = torch.zeros(N)
        zero_const = 0
        
        lbda = (u / (u - l))
        mu = (-1 * u * l) / (u - l)
        U_temp_mat = prev_polyexp_mat * lbda
        U_temp_const = prev_polyexp_const * lbda + mu
        # U_temp_const = (prev_polyexp_const * (u / (u - l))) + ( (-1 * u * l) / (u - l))

        # print(U_temp_mat.shape)
        # print(U_temp_const.shape)
        # print(lbda)
        # print(mu)
        # print(prev_polyexp_const)
        # sjkd

        L_new_mat = torch.where(l>=0, prev_polyexp_mat, torch.where(u<=0, zero_mat, zero_mat))
        L_new_const = torch.where(l>=0, prev_polyexp_const, torch.where(u<=0, zero_const, zero_const))

        U_new_mat = torch.where(l>=0, prev_polyexp_mat, torch.where(u<=0, zero_mat, U_temp_mat))
        U_new_const = torch.where(l>=0, prev_polyexp_const, torch.where(u<=0, zero_const, U_temp_const))
        
        L_new = PolyExpNew(N, L_new_mat, L_new_const)
        U_new = PolyExpNew(N, U_new_mat, U_new_const)

        # print(L_new_mat[1:, 785:834])
        # print(U_new_mat[:, 784:834])
        # print(torch.sum(U_new_mat[1])/1004)
        # print(L_new_const)
        # print(l_new)
        # print(u_new)
        # print(U_new.const)
        # sjd
        
        return l_new, u_new, L_new, U_new 

    def fc(self, abs_elem, neighbours, prev, curr, W, B, debug_flag):
        polyexp = prev.dot(W).add(B)
        polyexp.debug_flag = debug_flag
        l_new = (backsubs_lower(polyexp, curr, abs_elem, neighbours))
        u_new = (backsubs_upper(polyexp, curr, abs_elem, neighbours))
        L_new = polyexp.copy()
        U_new = polyexp.copy()
        return l_new, u_new, L_new, U_new 
    
class CflowNewInterval(Transformer): 
    def relu(self, abs_elem, neighbours, prev, curr):
        l = abs_elem.get_elem_new('l', prev)
        u = abs_elem.get_elem_new('u', prev)
        l_new = torch.where(l>=0, l, torch.where(u<=0, 0, 0))
        u_new = torch.where(l>=0, u, torch.where(u<=0, 0, u))
        return l_new, u_new 
    
    def fc(self, abs_elem, neighbours, prev, curr, W, B, debug_flag):
        polyexp = prev.dot(W).add(B)
        l_new = polyexp.map(simplify_lower, abs_elem, neighbours)
        u_new = polyexp.map(simplify_upper, abs_elem, neighbours)
        return l_new, u_new 

    

def simplify_lower(n, c, abs_elem, neighbours):
    l = abs_elem.get_elem_new('l', n)
    u = abs_elem.get_elem_new('u', n)
    c= c.reshape(l.shape)
    res = torch.where(c >= 0, c * l, c * u)
    return res
    
def simplify_upper(n, c, abs_elem, neighbours):
    l = abs_elem.get_elem_new('l', n)
    u = abs_elem.get_elem_new('u', n)
    c = c.reshape(l.shape)
    res = torch.where(c >= 0, c * u, c * l)
    return res

def simplify_upper_old(n, c, abs_elem, neighbours):
    res = torch.where(c >= 0, c * abs_elem.get_elem_new('u', n), c * abs_elem.get_elem_new('l', n))
    return res

def replace_lower(n, c, abs_elem, neighbours):
    # print('profiling inside replace_lower')
    checkpoint = 0
    t_time = time.time()
    # print('checkpoint', checkpoint, time.time()-t_time)
    checkpoint += 1
    t_time = time.time()
    L = abs_elem.get_elem_new('L', n)
    size = L.size 
    # print(size)
    L_mat = L.mat
    L_const = L.const

    # print('checkpoint', checkpoint, time.time()-t_time)
    checkpoint += 1
    t_time = time.time()  
    U = abs_elem.get_elem_new('U', n)

    U_mat = U.mat
    U_const = U.const

    # print('checkpoint', checkpoint, time.time()-t_time)
    checkpoint += 1
    t_time = time.time()

    c1 = c.unsqueeze(1).repeat(1, L_mat.size(0)) 
    c2 = c.reshape(L_const.shape)

    # print('checkpoint', checkpoint, time.time()-t_time)
    checkpoint += 1
    t_time = time.time()    
    
    c1_1 = c1 * L_mat
    # print(L_mat.shape)

    # print('checkpoint', checkpoint, time.time()-t_time)
    checkpoint += 1
    t_time = time.time()    

    c1_2 = c1 * U_mat

    # print('checkpoint', checkpoint, time.time()-t_time)
    checkpoint += 1
    t_time = time.time() 

    res_mat = torch.where(c1 >= 0, c1_1, c1_2)

    # print('checkpoint', checkpoint, time.time()-t_time)
    checkpoint += 1
    t_time = time.time()


    res_const = torch.where(c2 >= 0, c2 * L_const, c2 * U_const)

    # print('checkpoint', checkpoint, time.time()-t_time)
    checkpoint += 1
    t_time = time.time()


    res = PolyExpNew(size=size, mat=res_mat, const=res_const)

    # print('exiting replace_lower', checkpoint, time.time()-t_time)
    checkpoint += 1
    t_time = time.time()
    # print()
    return res 

def replace_upper(n, c, abs_elem, neighbours):
    size = abs_elem.get_elem_new('L', n).size 
    L_mat = abs_elem.get_elem_new('L', n).mat
    L_const = abs_elem.get_elem_new('L', n).const
    U_mat = abs_elem.get_elem_new('U', n).mat
    U_const = abs_elem.get_elem_new('U', n).const
    c1 = c.unsqueeze(1).repeat(1, L_mat.size(0))
    c2 = c.reshape(L_const.shape)
    
    res_mat = torch.where(c1 < 0, c1 * L_mat, c1 * U_mat)
    res_const = torch.where(c2 < 0, c2 * L_const, c2 * U_const)
    res = PolyExpNew(size=size, mat=res_mat, const=res_const)
    return res 

def replace_lower_old(n, c, abs_elem, neighbours):
    res_mat = torch.where(c >= 0, c * abs_elem.get_elem_new('L', n).mat, c * abs_elem.get_elem_new('U', n).mat)
    res_const = torch.where(c >= 0, c * abs_elem.get_elem_new('L', n).const, c * abs_elem.get_elem_new('U', n).const)
    size = abs_elem.get_elem_new('L', n).size 
    res = PolyExpNew(size=size, mat=res_mat, const=res_const)
    return res 

def replace_upper_old(n, c, abs_elem, neighbours):
    # res = torch.where(c >= 0, abs_elem.d['U'][n], abs_elem.d['L'][n])
    res = torch.where(c >= 0, c * abs_elem.get_elem_new('U', n), c * abs_elem.get_elem_new('L', n))
    return res.mult(c)

def stop(n, c, abs_elem, neighbours):
    return torch.ones(c.shape, dtype=torch.bool)*0
    return False 

def priority(n, c, abs_elem, neighbours):
    return torch.ones(c.shape)
    return 1

def backsubs_lower(p, n, abs_elem, neighbours):
    res = p.traverse(stop, priority, replace_lower, abs_elem, neighbours)
    return res.map(simplify_lower, abs_elem, neighbours)

def backsubs_upper(p, n, abs_elem, neighbours):
    res = p.traverse(stop, priority, replace_upper, abs_elem, neighbours)
    return res.map(simplify_upper, abs_elem, neighbours)