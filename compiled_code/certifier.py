from common.abs_elem import Abs_elem
from common.polyexp import Nlist, PolyExp
from utils import *
# from common.transformer import Transformer
from specs.network import Network, LayerType

import torch
import time

# torch.set_printoptions(edgeitems=100000000, threshold=100000000)

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
            # print('Shape of Z is ', self.abs_elem.d['Z'].mat.shape)
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
                    abs_shape = self.transformer.Relu(self.abs_elem, prev, curr, poly_size, curr_size, prev_size, self.input_size)
                else:
                    abs_shape = self.transformer.Relu(self.abs_elem, prev, curr, poly_size, curr_size, prev_size, self.input_size)

                mat = (abs_shape[2].mat[:, -size:])
                const = (abs_shape[2].const)
                

                torch.set_printoptions(edgeitems=mat.numel(), threshold=mat.numel())
                # print(mat.sum())

                # print('Debug stats')
                # print(mat.shape)
                # print(const.shape)

                print(torch.diagonal(mat))
                # print(const)


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

                abs_shape = self.transformer.Affine(self.abs_elem, prev, curr, poly_size, curr_size, prev_size, self.input_size)

                mat = (abs_shape[2].mat[:, -prev_size:])
                const = (abs_shape[2].const)


                # print('Debug stats')
                # print(mat.shape)
                # print(const.shape)

                print(abs_shape[0])
                print(abs_shape[1])

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

