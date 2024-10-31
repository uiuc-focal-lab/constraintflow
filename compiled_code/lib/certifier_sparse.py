from compiled_code.lib.abs_elem import Abs_elem_sparse
from compiled_code.lib.polyexp import *
from compiled_code.lib.nlist import *
from compiled_code.lib.utils import *
from compiled_code.lib.network import Network, LayerType

import torch
import time


class Certifier:
    def __init__(self, abs_elem: Abs_elem_sparse, transformer, model: Network, neighbours):
        self.abs_elem = abs_elem 
        self.transformer = transformer 
        self.model = model
        self.neighbours = neighbours
        self.input_size = model.input_size
        self.batch_size = abs_elem.batch_size

    def flow(self):
        begin_time = time.time()
        t_time = time.time()
        prev_size = self.model.input_size
        size = self.model.input_size

        for tmp, layer in enumerate(self.model):
            print(tmp+1, layer.type, layer.shape)
            poly_size = self.model[list(torch.nonzero(self.abs_elem.d['llist']))[-1].item()].end
            curr_size = self.model[tmp].end-size
            if layer.type == LayerType.ReLU:
                prev = Llist(self.model, [1], None, None, [tmp-1])
                curr = Llist(self.model, [1], None, None, [tmp])
                abs_shape = self.transformer.Relu(self.abs_elem, prev, curr, poly_size, curr_size, prev_size, self.input_size, self.batch_size)

            elif layer.type == LayerType.Linear:
                prev = Llist(self.model, [1, 1], None, None, [tmp-1])
                curr = Llist(self.model, [1], None, None, [tmp])
                abs_shape = self.transformer.Affine(self.abs_elem, prev, curr, poly_size, curr_size, prev_size, self.input_size, self.batch_size)

            elif layer.type == LayerType.Conv2D:
                prev = Llist(self.model, [1, 1], None, None, [tmp-1])
                curr = Llist(self.model, [1], None, None, [tmp])
                abs_shape = self.transformer.Affine(self.abs_elem, prev, curr, poly_size, curr_size, prev_size, self.input_size, self.batch_size)

            elif layer.type == LayerType.Input:
                continue

            else:
                print(layer.type)
                assert(False)
            size += curr_size
            prev_size = self.model[tmp].size
            self.abs_elem.update(curr, abs_shape)
            print(time.time()-t_time)
            t_time = time.time()
        print(abs_shape[0].get_dense())
        print(abs_shape[1].get_dense())
        # ldfyu
        print()
        print('time taken', time.time() - begin_time)

