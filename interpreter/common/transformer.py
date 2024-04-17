from common.polyexp import PolyExpNew, Nlist
# from common.nlist import Nlist
from common.symexp import SymExp
import torch 
import time 



class Transformer:
    def relu(self, abs_elem, neighbours, prev, curr):
        pass 

    def tanh(self, abs_elem, neighbours, prev, curr):
        pass 

    def fc(self, abs_elem, neighbours, prev, curr, w, b):
        pass 


class CflowNewDeepPoly(Transformer): 
    def relu(self, abs_elem, neighbours, prev, curr, debug_flag=False):
        N = abs_elem.d['l'].size(0)
        l = abs_elem.get_elem_new('l', prev)
        u = abs_elem.get_elem_new('u', prev)
        
        # if debug_flag:
        #     l_new = torch.where(l>=0, l, torch.where(u<=0, 0, 0))
        #     u_new = torch.where(l>=0, u, torch.where(u<=0, 0, u))
        # else:
        #     l_new = l 
        #     u_new = u

        l_new = torch.where(l>=0, l, torch.where(u<=0, 0, 0))
        u_new = torch.where(l>=0, u, torch.where(u<=0, 0, u))

        prev_polyexp_mat = prev.sum().mat 
        prev_polyexp_const = prev.sum().const.reshape(-1,1)

        
        zero_mat = torch.zeros(prev_polyexp_mat.shape)
        zero_const = torch.zeros(prev_polyexp_const.shape)
        
        lbda = (u / (u - l))
        mu = (-1 * u * l) / (u - l)
        U_temp_mat = prev_polyexp_mat * lbda
        U_temp_const = prev_polyexp_const * lbda + mu

        cond1 = l.repeat(1, prev_polyexp_const.size(1)) >= 0
        cond2 = u.repeat(1, prev_polyexp_const.size(1)) <= 0

        L_new_mat = torch.where(cond1, prev_polyexp_mat, torch.where(cond2, zero_mat, zero_mat))
        L_new_const = torch.where(cond1, prev_polyexp_const, torch.where(cond2, zero_const, zero_const))

        U_new_mat = torch.where(l>=0, prev_polyexp_mat, torch.where(u<=0, zero_mat, U_temp_mat))
        U_new_const = torch.where(l>=0, prev_polyexp_const, torch.where(u<=0, zero_const, U_temp_const))
        
        L_new = PolyExpNew(N, L_new_mat, L_new_const)
        U_new = PolyExpNew(N, U_new_mat, U_new_const)
        
        temp = (u_new - l_new) >= 0
        assert(temp.all())

        # if debug_flag:
        #     return l_new, u_new, L_new, U_new 
        return l_new, u_new, L_new, U_new 
        return l, u, L_new, U_new 

    def fc(self, abs_elem, neighbours, prev, curr, W, B, debug_flag):
        polyexp = prev.dot(W).add(B)
        polyexp.debug_flag = debug_flag
        s_time = time.time()
        l_new = (backsubs_lower(polyexp, curr, abs_elem, neighbours))
        print(time.time()-s_time)
        s_time = time.time()
        u_new = (backsubs_upper(polyexp, curr, abs_elem, neighbours))
        print(time.time()-s_time)
        s_time = time.time()
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
    n = abs_elem.get_live_nlist(n)
    if c.dim()==1:
        c = c[n.nlist]
    else:
        c = c[:, n.nlist]
    l = abs_elem.get_elem_new('l', n)
    u = abs_elem.get_elem_new('u', n)
    
    indices = c >= 0
    c_pos = torch.where(indices, c, 0)
    c_neg = torch.where(~indices, c, 0)

    
    
    res = c_pos @ l + c_neg @ u
    

    return res 
    

def simplify_upper(n, c, abs_elem, neighbours):
    n = abs_elem.get_live_nlist(n)
    if c.dim()==1:
        c = c[n.nlist]
    else:
        c = c[:, n.nlist]
    l = abs_elem.get_elem_new('l', n)
    u = abs_elem.get_elem_new('u', n)
    
    indices = c >= 0
    c_pos = torch.where(indices, c, 0)
    c_neg = torch.where(~indices, c, 0)

    
    
    res = c_pos @ u + c_neg @ l

    return res 

def replace_lower(n, c, abs_elem, neighbours):
    n = abs_elem.get_live_nlist(n)
    if c.dim()==1:
        c = c[n.nlist]
    else:
        c = c[:, n.nlist]
    indices = c >= 0
    c_pos = torch.where(indices, c, 0)
    c_neg = torch.where(~indices, c, 0)
    

    L = abs_elem.get_elem_new('L', n)
    size = L.size 
    L_mat = L.mat
    L_const = L.const

    U = abs_elem.get_elem_new('U', n)
    U_mat = U.mat
    U_const = U.const
    
    res_mat = c_pos @ L_mat + c_neg @ U_mat
    res_const = (c_pos @ L_const + c_neg @ U_const)

    res = PolyExpNew(size=size, mat=res_mat, const=res_const)

    return res 

def replace_upper(n, c, abs_elem, neighbours):
    n = abs_elem.get_live_nlist(n)
    if c.dim()==1:
        c = c[n.nlist]
    else:
        c = c[:, n.nlist]
    indices = c >= 0
    c_pos = torch.where(indices, c, 0)
    c_neg = torch.where(~indices, c, 0)
    

    L = abs_elem.get_elem_new('L', n)
    size = L.size 
    L_mat = L.mat
    L_const = L.const

    U = abs_elem.get_elem_new('U', n)
    U_mat = U.mat
    U_const = U.const
    
    res_mat = c_pos @ U_mat + c_neg @ L_mat
    res_const = (c_pos @ U_const + c_neg @ L_const)

    res = PolyExpNew(size=size, mat=res_mat, const=res_const)

    return res 


def stop(n, c, abs_elem, neighbours):
    return n<784
    return torch.zeros(c.shape, dtype=torch.bool)
    return False 

def priority(n, c, abs_elem, neighbours):
    return torch.ones(c.shape)
    return 1

def backsubs_lower(p, n, abs_elem, neighbours):
    res = p.traverse(stop, priority, replace_lower, abs_elem, neighbours, replace_lower)
    return res.map(simplify_lower, abs_elem, neighbours)
    return res.const 

def backsubs_upper(p, n, abs_elem, neighbours):
    res = p.traverse(stop, priority, replace_upper, abs_elem, neighbours)
    return res.map(simplify_upper, abs_elem, neighbours)
    return res.const 


class CflowNewDeepPolyCompiler(Transformer): 
    def relu(self, abs_elem, neighbours, prev, curr, debug_flag=False):
        N = abs_elem.d['l'].size(0)
        l = abs_elem.get_elem_new('l', prev).reshape(-1,1)
        u = abs_elem.get_elem_new('u', prev).reshape(-1,1)
        
        # if debug_flag:
        #     l_new = torch.where(l>=0, l, torch.where(u<=0, 0, 0))
        #     u_new = torch.where(l>=0, u, torch.where(u<=0, 0, u))
        # else:
        #     l_new = l 
        #     u_new = u

        l_new = torch.where(l>=0, l, torch.where(u<=0, 0, 0))
        u_new = torch.where(l>=0, u, torch.where(u<=0, 0, u))

        prev_polyexp_mat = prev.sum().mat 
        prev_polyexp_const = prev.sum().const.reshape(-1,1)

        
        zero_mat = torch.zeros(prev_polyexp_mat.shape)
        zero_const = torch.zeros(prev_polyexp_const.shape)
        
        lbda = (u / (u - l))
        mu = (-1 * u * l) / (u - l)
        U_temp_mat = prev_polyexp_mat * lbda
        U_temp_const = prev_polyexp_const * lbda + mu

        cond1 = l.repeat(1, prev_polyexp_const.size(1)) >= 0
        cond2 = u.repeat(1, prev_polyexp_const.size(1)) <= 0

        L_new_mat = torch.where(cond1, prev_polyexp_mat, torch.where(cond2, zero_mat, zero_mat))
        L_new_const = torch.where(cond1, prev_polyexp_const, torch.where(cond2, zero_const, zero_const))

        U_new_mat = torch.where(l>=0, prev_polyexp_mat, torch.where(u<=0, zero_mat, U_temp_mat))
        U_new_const = torch.where(l>=0, prev_polyexp_const, torch.where(u<=0, zero_const, U_temp_const))
        
        L_new = PolyExpNew(N, L_new_mat, L_new_const)
        U_new = PolyExpNew(N, U_new_mat, U_new_const)
        
        temp = (u_new - l_new) >= 0
        assert(temp.all())

        # if debug_flag:
        #     return l_new, u_new, L_new, U_new 
        return l_new, u_new, L_new, U_new 
        return l, u, L_new, U_new 

    def fc(self, abs_elem, neighbours, prev, curr, W, B, debug_flag):
        polyexp = prev.dot(W).add(B)
        polyexp.debug_flag = debug_flag
        s_time = time.time()
        l_new = (backsubs_lower_compiler(polyexp, curr, abs_elem, neighbours))
        print(time.time()-s_time)
        s_time = time.time()
        u_new = (backsubs_upper_compiler(polyexp, curr, abs_elem, neighbours))
        print(time.time()-s_time)
        s_time = time.time()
        L_new = polyexp.copy()
        U_new = polyexp.copy()
        print(l_new )
        return l_new, u_new, L_new, U_new 
    
def replace_lower_compiler(n, c, abs_elem, neighbours):
    n = abs_elem.get_live_nlist(n)
    if c.dim()==1:
        c = c[n.nlist]
    else:
        c = c[:, n.nlist]

    
    num_dims = c.dim()
    zero = torch.tensor(0)
    zero = zero.resize(*[1]*num_dims)
    zero = zero.repeat(c.shape)

    cond = c>=0

    n_L = abs_elem.get_elem_new('L', n)
    n_U = abs_elem.get_elem_new('U', n)

    # n_L.mat = n_L.mat[:, n.nlist]
    # n_U.mat = n_U.mat[:, n.nlist]

    # print(n_L.mat.shape)
    # print(n_L.mat.shape)
    # fdhj

    n_L.mat = n_L.mat.unsqueeze(0)
    n_L.const = n_L.const.unsqueeze(0)
    n_L.mat = n_L.mat.repeat(c.size(0), 1, 1)
    n_L.const = n_L.const.repeat(c.size(0), 1)

    n_U.mat = n_U.mat.unsqueeze(0)
    n_U.const = n_U.const.unsqueeze(0)
    n_U.mat = n_U.mat.repeat(c.size(0), 1, 1)
    n_U.const = n_U.const.repeat(c.size(0), 1)

    c_temp = c.unsqueeze(-1)
    c_temp = c_temp.repeat(1, 1, n_L.mat.size(-1))
    c_n_L_mat = c_temp * n_L.mat
    c_n_L_const = c * n_L.const

    c_temp2 = c.unsqueeze(-1)
    c_temp2 = c_temp2.repeat(1, 1, n_L.mat.size(-1))
    c_n_U_mat = c_temp2 * n_U.mat
    c_n_U_const = c * n_U.const

    
    res_const = torch.where(cond, c_n_L_const, c_n_U_const)
    cond_temp = cond.unsqueeze(-1)
    cond_temp = cond_temp.repeat(1, 1, n_L.mat.size(-1))
    res_mat = torch.where(cond_temp, c_n_L_mat, c_n_U_mat)

    return PolyExpNew(size=n_L.mat.size(-1), mat=res_mat, const=res_const)

def replace_upper_compiler(n, c, abs_elem, neighbours):
    n = abs_elem.get_live_nlist(n)
    if c.dim()==1:
        c = c[n.nlist]
    else:
        c = c[:, n.nlist]
    num_dims = c.dim()
    zero = torch.tensor(0)
    zero = zero.resize(*[1]*num_dims)
    zero = zero.repeat(c.shape)

    cond = c<=0

    n_L = abs_elem.get_elem_new('L', n)
    n_U = abs_elem.get_elem_new('U', n)

    n_L.mat = n_L.mat.unsqueeze(0)
    n_L.const = n_L.const.unsqueeze(0)
    n_L.mat = n_L.mat.repeat(c.size(0), 1, 1)
    n_L.const = n_L.const.repeat(c.size(0), 1)

    n_U.mat = n_U.mat.unsqueeze(0)
    n_U.const = n_U.const.unsqueeze(0)
    n_U.mat = n_U.mat.repeat(c.size(0), 1, 1)
    n_U.const = n_U.const.repeat(c.size(0), 1)

    c_temp = c.unsqueeze(-1)
    c_temp = c_temp.repeat(1, 1, n_L.mat.size(-1))
    c_n_L_mat = c_temp * n_L.mat
    c_n_L_const = c * n_L.const

    c_temp2 = c.unsqueeze(-1)
    c_temp2 = c_temp2.repeat(1, 1, n_L.mat.size(-1))
    c_n_U_mat = c_temp2 * n_U.mat
    c_n_U_const = c * n_U.const

    res_const = torch.where(cond, c_n_L_const, c_n_U_const)
    cond_temp = cond.unsqueeze(-1)
    cond_temp = cond_temp.repeat(1, 1, n_L.mat.size(-1))
    res_mat = torch.where(cond_temp, c_n_L_mat, c_n_U_mat)

    return PolyExpNew(size=n_L.mat.size(-1), mat=res_mat, const=res_const)

def simplify_lower_compiler(n, c, abs_elem, neighbours):
    n = abs_elem.get_live_nlist(n)
    if c.dim()==1:
        c = c[n.nlist]
    else:
        c = c[:, n.nlist]
    num_dims = c.dim()
    zero = torch.tensor(0)
    zero = zero.resize(*[1]*num_dims)
    zero = zero.repeat(c.shape)

    cond = c>=0

    n_l = abs_elem.get_elem_new('l', n)
    n_u = abs_elem.get_elem_new('u', n)


    n_l = n_l.unsqueeze(0)
    n_l = n_l.repeat(c.size(0), 1)

    n_u = n_u.unsqueeze(0)
    n_u = n_u.repeat(c.size(0), 1)

    c_n_l = c * n_l

    c_n_u = c * n_u

    res = torch.where(cond, c_n_l, c_n_u)
    return res

def simplify_upper_compiler(n, c, abs_elem, neighbours):
    n = abs_elem.get_live_nlist(n)
    if c.dim()==1:
        c = c[n.nlist]
    else:
        c = c[:, n.nlist]
    num_dims = c.dim()
    zero = torch.tensor(0)
    zero = zero.resize(*[1]*num_dims)
    zero = zero.repeat(c.shape)

    cond = c<=0

    n_l = abs_elem.get_elem_new('l', n)
    n_u = abs_elem.get_elem_new('u', n)

    n_l = n_l.unsqueeze(0)
    n_l = n_l.repeat(c.size(0), 1)

    n_u = n_u.unsqueeze(0)
    n_u = n_u.repeat(c.size(0), 1)

    c_n_l = c * n_l

    c_n_u = c * n_u

    res = torch.where(cond, c_n_l, c_n_u)
    return res

def stop_compiler(n, c, abs_elem, neighbours):
    # return n<784
    return torch.zeros(c.shape, dtype=torch.bool)
    return False 

def priority_compiler(n, c, abs_elem, neighbours):
    return torch.ones(c.shape)
    return 1

def backsubs_lower_compiler(p, n, abs_elem, neighbours):
    res = p.traverse(stop_compiler, priority_compiler, replace_lower_compiler, abs_elem, neighbours)
    # print(res.mat)
    # print(res.const)
    # sdjg
    return res.map_compiler(simplify_lower_compiler, abs_elem, neighbours)
    return res.const 

def backsubs_upper_compiler(p, n, abs_elem, neighbours):
    res = p.traverse(stop_compiler, priority_compiler, replace_upper_compiler, abs_elem, neighbours)
    return res.map_compiler(simplify_upper_compiler, abs_elem, neighbours)
    return res.const 