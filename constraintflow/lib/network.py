from enum import Enum


class Network(list):
    def __init__(self, net_name=None, input_name=None, input_shape=None, input_size = 0, input_start=0, input_end=0, num_layers=0, torch_net=None, net_format='torch', no_sparsity=False):  
        super().__init__()
        self.net_name = net_name
        self.input_name = input_name
        self.input_shape = input_shape
        self.input_size = input_size
        self.size = 0
        self.input_start = input_start
        self.input_end = input_end
        self.num_layers = num_layers
        self.torch_net = torch_net
        self.net_format = net_format 
        self.no_sparsity = no_sparsity

    def similar(self):
        res = Network()
        res.net_name = self.net_name
        res.input_name = self.input_name
        res.input_shape = self.input_shape
        res.input_size = self.input_size
        res.size = self.size
        res.input_start = self.input_start
        res.input_end = self.input_end
        res.num_layers = self.num_layers
        res.torch_net = self.torch_net
        res.net_format = self.net_format
        res.no_sparsity = self.no_sparsity
        return res


class Layer:
    def __init__(self, weight=None, bias=None, type=None, shape=None, start=None, end=None, size = 0, prev = {}, prev_weight = {}, mean = 0, sigma = 1, identifier = 0, parents = []):
        self.weight = weight
        self.bias = bias
        self.type = type
        self.shape = shape 
        self.size = size 
        self.start = start
        self.end = end
        self.prev = prev 
        self.prev_weight = prev_weight 
        self.index_hash = dict()
        self.mean = mean
        self.sigma = sigma
        self.identifier = identifier
        self.parents = parents
        self.children = list()
        self.last_layer = False

class LayerType(Enum):
    Conv2D = 1
    Linear = 2
    ReLU = 3
    Flatten = 4
    MaxPool1D = 5
    Normalization = 6
    NoOp = 7
    Tanh = 8
    Input = 9
    Concat = 10
    Sigmoid = 11
    Add = 12