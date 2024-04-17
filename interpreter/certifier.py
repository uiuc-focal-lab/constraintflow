from common.abs_elem import Abs_elem
from common.polyexp import Nlist
from common.transformer import Transformer
from specs.network import Network, LayerType

import torch
import itertools 
import time

# SIZE = 1004

def compute_size(shape):
    s = 1
    while len(shape)>0:
        s *= shape[0]
        shape = shape[1:]
    return s

class Certifier:
    def __init__(self, abs_elem: Abs_elem, transformer:Transformer, model: Network, neighbours):
        self.abs_elem = abs_elem 
        self.transformer = transformer 
        self.model = model
        self.neighbours = neighbours
        self.size = self.abs_elem.d['l'].shape[0]

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

                prev_mat = torch.tensor([i for i in range(prev_s, prev_e+1)]).reshape(-1,1)
                curr_mat = torch.tensor([i for i in range(curr_s, curr_e+1)]).reshape(-1,1)

                prev = Nlist(size = self.size, nlist = prev_mat)
                curr = Nlist(size = self.size, nlist = curr_mat)

                # abs_shape = self.transformer.relu(self.abs_elem, self.neighbours, prev_s, prev_e, curr_s, curr_e)
                debug_flag = False 
                if tmp==5:
                    debug_flag = True 
                abs_shape = self.transformer.relu(self.abs_elem, self.neighbours, prev, curr)
                # print(abs_shape[2].const.shape)
                # dfsjh
                self.abs_elem.d['t'][curr_s:curr_e+1] = True
                self.abs_elem.d['l'][curr_s:curr_e+1] = abs_shape[0].reshape(self.abs_elem.d['l'][curr_s:curr_e+1].shape)
                self.abs_elem.d['u'][curr_s:curr_e+1] = abs_shape[1].reshape(self.abs_elem.d['u'][curr_s:curr_e+1].shape)
                if len(abs_shape)>2:
                    self.abs_elem.d['L'].mat[curr_s:curr_e+1] = abs_shape[2].mat 
                    self.abs_elem.d['L'].const[curr_s:curr_e+1] = abs_shape[2].const
                    self.abs_elem.d['U'].mat[curr_s:curr_e+1] = abs_shape[3].mat 
                    self.abs_elem.d['U'].const[curr_s:curr_e+1] = abs_shape[3].const
                curr_s = curr_e + 1
                
            elif layer.type == LayerType.Linear:
                W = layer.weight
                B = layer.bias
                
                prev_size = W.shape[1]
                prev_s = curr_s - prev_size
                prev_e = curr_s - 1
                
                prev_mat = torch.tensor([i for i in range(prev_s, prev_e+1)]).unsqueeze(0).repeat(size, 1)
                curr_mat = torch.tensor([i for i in range(curr_s, curr_e+1)]).reshape(-1,1)
                prev = Nlist(size = self.size, nlist = prev_mat)
                curr = Nlist(size = self.size, nlist = curr_mat)
                debug_flag = False 
                if tmp==2:
                    debug_flag = True 
                abs_shape = self.transformer.fc(self.abs_elem, self.neighbours, prev, curr, W, B, debug_flag)
                # abs_shape = self.transformer.fc(self.abs_elem, self.neighbours, prev_s, prev_e, curr_s, curr_e, W, B)
                # print(abs_shape[2].const.shape)
                # print(abs_shape[3].const.shape)
                # jsdh
                self.abs_elem.d['t'][curr_s:curr_e+1] = True
                self.abs_elem.d['l'][curr_s:curr_e+1] = abs_shape[0].reshape(-1,1)
                self.abs_elem.d['u'][curr_s:curr_e+1] = abs_shape[1].reshape(-1,1)
                if len(abs_shape)>2:
                    self.abs_elem.d['L'].mat[curr_s:curr_e+1] = abs_shape[2].mat 
                    self.abs_elem.d['L'].const[curr_s:curr_e+1] = abs_shape[2].const.reshape(-1,1)
                    self.abs_elem.d['U'].mat[curr_s:curr_e+1] = abs_shape[3].mat 
                    self.abs_elem.d['U'].const[curr_s:curr_e+1] = abs_shape[3].const.reshape(-1,1)
                curr_s = curr_e + 1

            elif layer.type == LayerType.Conv2D:
                prev = []
                curr = []
                W = []
                B = layer.bias
                for i in range(curr_s, curr_e+1):
                    curr.append(i)
                    prev.append(torch.tensor(layer.prev[i]))
                    # print(len(prev[-1]))
                    W.append(torch.tensor(layer.prev_weight[i]))
                    
                # prev_mat = torch.tensor([i for i in range(prev_s, prev_e+1)]).unsqueeze(0).repeat(size, 1)
                # curr_mat = torch.tensor([i for i in range(curr_s, curr_e+1)]).reshape(-1,1)
                # print(prev)
                prev = Nlist(size = self.size, nlist = (prev))
                curr = Nlist(size = self.size, nlist = (curr))
                debug_flag = False 
                # if tmp==2:
                #     debug_flag = True 
                abs_shape = self.transformer.fc(self.abs_elem, self.neighbours, prev, curr, W, B, debug_flag)
                # abs_shape = self.transformer.fc(self.abs_elem, self.neighbours, prev_s, prev_e, curr_s, curr_e, W, B)
                # print(abs_shape[2].const.shape)
                # print(abs_shape[3].const.shape)
                # jsdh
                self.abs_elem.d['t'][curr_s:curr_e+1] = True
                self.abs_elem.d['l'][curr_s:curr_e+1] = abs_shape[0]
                self.abs_elem.d['u'][curr_s:curr_e+1] = abs_shape[1]
                if len(abs_shape)>2:
                    self.abs_elem.d['L'].mat[curr_s:curr_e+1] = abs_shape[2].mat 
                    self.abs_elem.d['L'].const[curr_s:curr_e+1] = abs_shape[2].const 
                    self.abs_elem.d['U'].mat[curr_s:curr_e+1] = abs_shape[3].mat 
                    self.abs_elem.d['U'].const[curr_s:curr_e+1] = abs_shape[3].const 
                curr_s = curr_e + 1
                
            # elif layer.type == LayerType.Conv2D:
            #     # W = layer.weight
            #     # B = layer.bias
            #     for neuron_num, index in enumerate(indices):
            #         print(neuron_num, index)
            #         curr = (layer_num, index)
            #         # prev = self.neighbours[curr]
            #         prev = layer.prev[curr]
            #         # print(prev)
            #         w = layer.prev_weight[curr]
            #         b = layer.bias[index[1]]
            #         abs_shape = self.transformer.fc(self.abs_elem, self.neighbours, prev, curr, w, b)
            #         self.abs_elem.update_elem(curr, abs_shape)
            print(time.time()-t_time)
            t_time = time.time()
            
            temp = (abs_shape[1] - abs_shape[0] >= 0).any()
            if not temp:
                print('what the hell in layer', tmp)
                kjf
        print(abs_shape[0]) 
        print(abs_shape[1])
        print('time taken', time.time() - begin_time)
        