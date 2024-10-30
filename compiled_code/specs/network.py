from enum import Enum


class Network(list):
    def __init__(self, net_name=None, input_name=None, input_shape=None, input_size = 0, input_start=0, input_end=0, num_layers=0, torch_net=None, net_format='torch'):
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


class Layer:
    def __init__(self, weight=None, bias=None, type=None, shape=None, start=None, end=None, size = 0, prev = {}, prev_weight = {}):
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