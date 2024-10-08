from common.abs_elem import Abs_elem_sparse
from common.polyexp import *
from common.nlist import *
from utils import *
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
    def __init__(self, abs_elem: Abs_elem_sparse, transformer, model: Network, neighbours):
        self.abs_elem = abs_elem 
        self.transformer = transformer 
        self.model = model
        self.neighbours = neighbours
        self.input_size = model.input_size

    def flow(self):
        begin_time = time.time()
        t_time = time.time()
        curr_s = compute_size(self.model.old_network.input_shape)
        for tmp, layer in enumerate(self.model.old_network):
            print(tmp+1, layer.type, layer.shape)
            shape = tuple(layer.shape) 
            size = compute_size(shape)
            curr_e = curr_s + size - 1 
            
            if layer.type == LayerType.ReLU:
                prev_s = curr_s - size
                prev_e = curr_e - size

                prev = Llist(self.model, [1], None, None, [tmp])
                curr = Llist(self.model, [1], None, None, [tmp+1])

                poly_size = self.model.layers_end[list(torch.nonzero(self.abs_elem.d['llist']))[-1].item()]
                curr_size = curr_e+1-curr_s
                prev_size = prev_e+1-prev_s

                abs_shape = self.transformer.Relu(self.abs_elem, prev, curr, poly_size, curr_size, prev_size, self.input_size, 1)

            elif layer.type == LayerType.Linear:
                W = layer.weight
                B = layer.bias
                
                prev_size = W.shape[1]
                prev_s = curr_s - prev_size
                prev_e = curr_s - 1
                
                poly_size = self.model.layers_end[list(torch.nonzero(self.abs_elem.d['llist']))[-1].item()]
                curr_size = curr_e+1-curr_s
                prev_size = prev_e+1-prev_s
                prev = Llist(self.model, [1, 1], None, None, [tmp])
                curr = Llist(self.model, [1], None, None, [tmp+1])
                
                abs_shape = self.transformer.Affine(self.abs_elem, prev, curr, poly_size, curr_size, prev_size, self.input_size, 1)
            self.abs_elem.update(curr, abs_shape)
            curr_s = curr_e + 1
            print(time.time()-t_time)
            t_time = time.time()
        print(abs_shape[0].get_dense())
        print(abs_shape[1].get_dense())
        print()
        print('time taken', time.time() - begin_time)

