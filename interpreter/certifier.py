from common.abs_elem import Abs_elem
from common.transformer import Transformer
from specs.network import Network, Layer, LayerType

import itertools 

class Certifier:
    def __init__(self, abs_elem: Abs_elem, transformer:Transformer, model: Network, neighbours):
        self.abs_elem = abs_elem 
        self.transformer = transformer 
        self.model = model
        self.neighbours = neighbours

    def flow(self):
        for tmp, layer in enumerate(self.model):
            layer_num = tmp+1
            # if layer_num == len(self.model.widths):
            #     break
            print(layer_num, layer.type, layer.shape)
            shape = tuple(layer.shape) 
            indices = itertools.product(*[range(dim) for dim in shape])
            if layer.type == LayerType.ReLU:
                for index in indices:
                    curr = (layer_num, index)
                    prev = (layer_num-1, index)
                    abs_shape = self.transformer.relu(self.abs_elem, self.neighbours, prev, curr)
                    self.abs_elem.update(curr, abs_shape)
                # for neuron_num in range(self.model.widths[layer_num]):
                #     curr = (layer_num, neuron_num)
                #     prev = (layer_num-1, neuron_num)
                #     abs_shape = self.transformer.relu(self.abs_elem, self.neighbours, prev, curr)
                #     self.abs_elem.update(curr, abs_shape)
            elif layer.type == LayerType.Linear:
                W = layer.weight
                B = layer.bias
                prev_shape = tuple(self.model[tmp-1].shape)
                if tmp==0:
                    prev_shape = tuple(self.model.input_shape)
                prev_indices = itertools.product(*[range(dim) for dim in prev_shape])
                prev = [(layer_num-1, i) for i in prev_indices]
                for neuron_num, index in enumerate(indices):
                    curr = (layer_num, index)
                    # print(prev)
                    # print(curr)
                    # print(W.shape)
                    w = W[neuron_num]
                    b = B[neuron_num]
                    abs_shape = self.transformer.fc(self.abs_elem, self.neighbours, prev, curr, w, b)
                    self.abs_elem.update(curr, abs_shape)