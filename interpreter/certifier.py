from common.abs_elem import Abs_elem
from common.transformer import Transformer
from specs.network import Network, LayerType

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
            print(layer_num, layer.type, layer.shape)
            shape = tuple(layer.shape) 
            indices = itertools.product(*[range(dim) for dim in shape])
            if layer.type == LayerType.ReLU:
                for index in indices:
                    curr = (layer_num, index)
                    prev = (layer_num-1, index)
                    abs_shape = self.transformer.relu(self.abs_elem, self.neighbours, prev, curr)
                    self.abs_elem.update_elem(curr, abs_shape)
            elif layer.type == LayerType.Linear:
                W = layer.weight
                B = layer.bias
                for neuron_num, index in enumerate(indices):
                    # print(neuron_num, index)
                    curr = (layer_num, index)
                    # prev = self.neighbours[curr]
                    prev = layer.prev[curr]
                    w = W[neuron_num]
                    b = B[neuron_num]
                    abs_shape = self.transformer.fc(self.abs_elem, self.neighbours, prev, curr, w, b)
                    self.abs_elem.update_elem(curr, abs_shape)
            elif layer.type == LayerType.Conv2D:
                # W = layer.weight
                # B = layer.bias
                for neuron_num, index in enumerate(indices):
                    # print(neuron_num)
                    curr = (layer_num, index)
                    # prev = self.neighbours[curr]
                    prev = layer.prev[curr]
                    # print(prev)
                    w = layer.prev_weight[curr]
                    b = layer.bias[index[1]]
                    abs_shape = self.transformer.fc(self.abs_elem, self.neighbours, prev, curr, w, b)
                    self.abs_elem.update_elem(curr, abs_shape)
