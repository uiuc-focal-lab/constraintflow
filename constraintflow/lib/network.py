from enum import Enum
import inspect

import torch

tp = type

class Network(list):
    # element type of this list: lib.network.Layer
    def __init__(self, net_name=None, input_name=None, input_shape=None, input_size = 0, input_start=0, input_end=0, num_layers=0, torch_net=None, net_format='torch', no_sparsity=False):  
        super().__init__()
        self.net_name = net_name
        self.input_name = input_name
        self.input_shape = input_shape
        # input_size == the product of all elements in input_shape
        self.input_size = input_size
        self.size = 0
        self.input_start = input_start
        self.input_end = input_end
        # Why num_layers != length of this list?
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
        # print(f'Layer construction'
        #       f'shape: {shape}, start: {start}, end: {end}, size: {size}')
        # caller_info = inspect.getframeinfo(inspect.currentframe().f_back)
        # print(f'Called from {caller_info.filename}:{caller_info.lineno} in function {caller_info.function}')
        self.weight = weight
        # type of self.bias: torch.Tensor
        self.bias = bias
        # type of self.type: LayerType
        self.type= type
        # type of self.shape: list[int]
        self.shape = shape 
        # type of self.size: int
        self.size = size 
        # start, end: All the *neurons* in the DNN are numbered using a unique
        #   integer identifier.
        # Each layer is a list of consecutive neurons.
        # So start and end just encode that consecutive list.
        # start, end type: int
        self.start = start
        self.end = end
        self.prev = prev 
        self.prev_weight = prev_weight 
        self.index_hash = dict()
        self.mean = mean
        self.sigma = sigma
        self.identifier = identifier
        # parents type: list[int]
        self.parents = parents
        self.children = list()
        self.last_layer = False
        # True unless every immediate consumer is Linear/Conv2D (set precisely in
        # parse.py's post_process, once children are final). Conservative default
        # so a layer nothing recomputes this for is treated as needing traversal.
        self.feeds_nonlin = True

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
    #FEAT: elementwise product of two network values (Neuron_mult), e.g. the quadratic Lyapunov form in lsnc_relu.
    Mul = 13


# Single source of truth for which generated transformer method a given
# LayerType dispatches to in Flow.flow() (constraintflow/lib/flow_sparse.py),
# and for which key of jit_layers/layers.json records its per-layer index
# during simulacrum. tensor_to_block.tensor_to_block() reads that same key
# to decide which layer indices get specialized during reuse -- keeping both
# derived from this one table is what prevents an op (e.g. Sigmoid,
# previously) from being profiled but never specialized.
JIT_OP_FOR_LAYER = {
    LayerType.ReLU: 'Relu',
    LayerType.Sigmoid: 'Sigmoid',
    LayerType.Tanh: 'Tanh',  #FEAT: ERAN tanh networks
    LayerType.Linear: 'Affine',
    LayerType.Conv2D: 'Affine',
    LayerType.Add: 'Add',
    LayerType.Concat: 'Concat',
    LayerType.Mul: 'Mult',
}
JIT_OPS = sorted(set(JIT_OP_FOR_LAYER.values()))