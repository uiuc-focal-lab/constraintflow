from enum import Enum


class Network(list):
    def __init__(self, net_name=None, input_name=None, input_shape=None, torch_net=None, net_format='torch'):
        super().__init__()
        self.net_name = net_name
        self.input_name = input_name
        self.input_shape = input_shape
        self.torch_net = torch_net
        self.net_format = net_format


class Layer:
    def __init__(self, weight=None, bias=None, type=None, shape=None):
        self.weight = weight
        self.bias = bias
        self.type = type
        self.shape = shape 

class LayerType(Enum):
    Conv2D = 1
    Linear = 2
    ReLU = 3
    Flatten = 4
    MaxPool1D = 5
    Normalization = 6
    NoOp = 7
    Tanh = 8