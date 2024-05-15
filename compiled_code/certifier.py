from common.abs_elem import Abs_elem
from common.polyexp import Nlist, PolyExp
from utils import *
# from common.transformer import Transformer
from specs.network import Network, LayerType

import torch
import time

def compute_size(shape):
    s = 1
    while len(shape)>0:
        s *= shape[0]
        shape = shape[1:]
    return s

class Certifier:
    def __init__(self, abs_elem: Abs_elem, transformer, model: Network, neighbours):
        self.abs_elem = abs_elem 
        self.transformer = transformer 
        self.model = model
        self.neighbours = neighbours
        self.size = self.abs_elem.d['l'].shape[0]
        self.input_size = model.input_size

    def flow(self):
        begin_time = time.time()
        t_time = time.time()
        curr_s = compute_size(self.model.input_shape)
        for tmp, layer in enumerate(self.model):
            print(tmp+1, layer.type, layer.shape)
            shape = tuple(layer.shape) 
            size = compute_size(shape)
            curr_e = curr_s + size - 1 
            
            if layer.type == LayerType.ReLU:
                prev_s = curr_s - size
                prev_e = curr_e - size

                prev_mat = torch.tensor([i for i in range(prev_s, prev_e+1)])
                curr_mat = torch.tensor([i for i in range(curr_s, curr_e+1)])

                prev = Nlist(size = self.size, nlist = prev_mat, network=self.model)
                curr = Nlist(size = self.size, nlist = curr_mat)

                poly_size = torch.nonzero(self.abs_elem.d['t']).flatten().shape[0]
                curr_size = curr_e+1-curr_s
                prev_size = prev_e+1-prev_s

                debug_flag = False
                if tmp==3:
                    debug_flag = True

                if tmp>1:
                    abs_shape = self.transformer.Relu(self.abs_elem, prev, curr, poly_size, curr_size, prev_size, self.input_size, debug_flag)
                else:
                    abs_shape = self.transformer.Relu2(self.abs_elem, prev, curr, poly_size, curr_size, prev_size, self.input_size)
                
            elif layer.type == LayerType.Linear:
                W = layer.weight
                B = layer.bias
                
                prev_size = W.shape[1]
                prev_s = curr_s - prev_size
                prev_e = curr_s - 1

                debug_flag = False
                if tmp==4:
                    debug_flag = True
                
                prev = Nlist(size = self.size, start=prev_s, end = prev_e, network=self.model)
                curr = Nlist(size = self.size, start=curr_s, end = curr_e)
                
                poly_size = torch.nonzero(self.abs_elem.d['t']).flatten().shape[0]
                curr_size = curr_e+1-curr_s
                prev_size = prev_e+1-prev_s

                abs_shape = self.transformer.Affine(self.abs_elem, prev, curr, poly_size, curr_size, prev_size, self.input_size, debug_flag)
            
            self.abs_elem.update(curr, abs_shape)
            curr_s = curr_e + 1
            print(time.time()-t_time)
            t_time = time.time()
            
            temp = (abs_shape[1] - abs_shape[0] >= 0).any()
            if not temp:
                raise Exception('Something is not right')
        # matrix = self.abs_elem.d['U'].mat[-10:, -10:]
        # torch.set_printoptions(edgeitems=matrix.numel(), threshold=matrix.numel())
            print(abs_shape[0])
            print(abs_shape[1])
        # print(abs_shape[2].mat.shape)
        # print(abs_shape[2].mat[:, -10:], abs_shape[2].const)
        # print(abs_shape[3].mat[:, -10:], abs_shape[3].const)
        print()
        # print((self.abs_elem.d['U'].mat[-20:, :]).sum())
        print('time taken', time.time() - begin_time)


        curr_size = 10
        poly_size = 984
        prev_size = 50
        var_1 = PolyExp(curr_size, poly_size, self.abs_elem.d['L'].mat[984:994, :poly_size], self.abs_elem.d['L'].const[984:994])
        trav_size_0_0 = get_shape_1(var_1.get_mat(poly_size))
        phi_trav_exp1_1_1 = var_1
        phi_trav_exp1_2_3 = var_1
        if debug_flag:
            print('@@@@@@@@@@@')
            print(var_1.mat[0,:][-prev_size:])
        while(True):
            if debug_flag:
                print('@@@@@@@@@@@')
                print(phi_trav_exp1_1_1.mat[0, :][-2*prev_size:-prev_size])
                # print(phi_trav_exp1_1_1.mat[0, :].sum())
            trav_size_1_2 = get_shape_1(phi_trav_exp1_1_1.get_mat(poly_size))
            vertices_stop1 = False
            vertices1_1_1 = ne(phi_trav_exp1_1_1.get_mat(poly_size), torch.tensor(0).unsqueeze(0).unsqueeze(1).repeat(curr_size, poly_size))
            vertices_stop_default1 = get_default_stop2([curr_size, poly_size])
            vertices_stop_temp1 = disj(torch.tensor(vertices_stop1).unsqueeze(0).unsqueeze(1).repeat(curr_size, poly_size), vertices_stop_default1)
            vertices1_1_2 = conj(vertices1_1_1, boolNeg(vertices_stop_temp1))
            phi_trav_exp1_2_3 = phi_trav_exp1_1_1
            if(boolNeg(any(vertices1_1_2))):
                phi_trav_exp1_2_3 = phi_trav_exp1_1_1
                break
            var_2 = self.abs_elem.get_elem_new('L', self.abs_elem.get_live_nlist(Nlist(poly_size, 0, poly_size-1, None)))
            var_3 = self.abs_elem.get_elem_new('U', self.abs_elem.get_live_nlist(Nlist(poly_size, 0, poly_size-1, None)))
            rewrite_2 = convert_to_float(ge(phi_trav_exp1_1_1.get_mat(poly_size), torch.tensor(0).unsqueeze(0).unsqueeze(1).repeat(curr_size, poly_size)))
            trav_exp1_4_2 = PolyExp(curr_size, poly_size, plus(inner_prod(mult(convert_to_float(ge(phi_trav_exp1_1_1.get_mat(poly_size), torch.tensor(0).unsqueeze(0).unsqueeze(1).repeat(curr_size, poly_size))), phi_trav_exp1_1_1.get_mat(poly_size)).unsqueeze(2).squeeze(2), var_2.get_mat(poly_size).squeeze(0)), inner_prod(mult(minus(torch.tensor(1.0).unsqueeze(0).unsqueeze(1).repeat(curr_size, poly_size), convert_to_float(ge(phi_trav_exp1_1_1.get_mat(poly_size), torch.tensor(0).unsqueeze(0).unsqueeze(1).repeat(curr_size, poly_size)))), phi_trav_exp1_1_1.get_mat(poly_size)).unsqueeze(2).squeeze(2), var_3.get_mat(poly_size).squeeze(0))), plus(phi_trav_exp1_1_1.get_const(), plus(inner_prod(mult(convert_to_float(ge(phi_trav_exp1_1_1.get_mat(poly_size), torch.tensor(0).unsqueeze(0).unsqueeze(1).repeat(curr_size, poly_size))), phi_trav_exp1_1_1.get_mat(poly_size)), var_2.get_const().squeeze(0)), inner_prod(mult(minus(torch.tensor(1.0).unsqueeze(0).unsqueeze(1).repeat(curr_size, poly_size), rewrite_2), phi_trav_exp1_1_1.get_mat(poly_size)), var_3.get_const().squeeze(0)))))
            phi_trav_exp1_1_1 = trav_exp1_4_2
        rewrite_0 = convert_to_float(ge(phi_trav_exp1_2_3.get_mat(poly_size), torch.tensor(0).unsqueeze(0).unsqueeze(1).repeat(curr_size, poly_size)))
        L_new = phi_trav_exp1_1_1
        
        var_6 = PolyExp(curr_size, poly_size, self.abs_elem.d['L'].mat[984:994, :poly_size], self.abs_elem.d['L'].const[984:994])
        trav_size_2_4 = get_shape_1(var_6.get_mat(poly_size))
        phi_trav_exp2_5_1 = var_6
        phi_trav_exp2_6_3 = var_6
        while(True):
            trav_size_5_6 = get_shape_1(phi_trav_exp2_5_1.get_mat(poly_size))
            vertices_stop2 = False
            vertices2_5_1 = ne(phi_trav_exp2_5_1.get_mat(poly_size), torch.tensor(0).unsqueeze(0).unsqueeze(1).repeat(curr_size, poly_size))
            vertices_stop_default2 = get_default_stop2([curr_size, poly_size])
            vertices_stop_temp2 = disj(torch.tensor(vertices_stop2).unsqueeze(0).unsqueeze(1).repeat(curr_size, poly_size), vertices_stop_default2)
            vertices2_5_2 = conj(vertices2_5_1, boolNeg(vertices_stop_temp2))
            phi_trav_exp2_6_3 = phi_trav_exp2_5_1
            if(boolNeg(any(vertices2_5_2))):
                phi_trav_exp2_6_3 = phi_trav_exp2_5_1
                break
            var_7 = self.abs_elem.get_elem_new('U', self.abs_elem.get_live_nlist(Nlist(poly_size, 0, poly_size-1, None)))
            var_8 = self.abs_elem.get_elem_new('L', self.abs_elem.get_live_nlist(Nlist(poly_size, 0, poly_size-1, None)))
            rewrite_5 = convert_to_float(ge(phi_trav_exp2_5_1.get_mat(poly_size), torch.tensor(0).unsqueeze(0).unsqueeze(1).repeat(curr_size, poly_size)))
            trav_exp2_8_2 = PolyExp(curr_size, poly_size, plus(inner_prod(mult(convert_to_float(ge(phi_trav_exp2_5_1.get_mat(poly_size), torch.tensor(0).unsqueeze(0).unsqueeze(1).repeat(curr_size, poly_size))), phi_trav_exp2_5_1.get_mat(poly_size)).unsqueeze(2).squeeze(2), var_7.get_mat(poly_size).squeeze(0)), inner_prod(mult(minus(torch.tensor(1.0).unsqueeze(0).unsqueeze(1).repeat(curr_size, poly_size), convert_to_float(ge(phi_trav_exp2_5_1.get_mat(poly_size), torch.tensor(0).unsqueeze(0).unsqueeze(1).repeat(curr_size, poly_size)))), phi_trav_exp2_5_1.get_mat(poly_size)).unsqueeze(2).squeeze(2), var_8.get_mat(poly_size).squeeze(0))), plus(phi_trav_exp2_5_1.get_const(), plus(inner_prod(mult(convert_to_float(ge(phi_trav_exp2_5_1.get_mat(poly_size), torch.tensor(0).unsqueeze(0).unsqueeze(1).repeat(curr_size, poly_size))), phi_trav_exp2_5_1.get_mat(poly_size)), var_7.get_const().squeeze(0)), inner_prod(mult(minus(torch.tensor(1.0).unsqueeze(0).unsqueeze(1).repeat(curr_size, poly_size), rewrite_5), phi_trav_exp2_5_1.get_mat(poly_size)), var_8.get_const().squeeze(0)))))
            phi_trav_exp2_5_1 = trav_exp2_8_2
        U_new = phi_trav_exp2_5_1
        print(L_new.mat[:, 784:834].sum())
        print(U_new.mat[:, 784:834].sum())
        