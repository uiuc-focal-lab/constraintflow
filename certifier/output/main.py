import torch
import functools
import sys
import pickle


from certifier.lib.spec import *
from certifier.output.flow_sparse import Flow
from certifier.lib.abs_elem import Abs_elem_sparse
from certifier.lib.utils import *
from certifier.lib.network import LayerType
from certifier.output.transformers import *


batch_size = int(sys.argv[2])
network = get_net(sys.argv[1])
llist = torch.tensor([True] + [False]*network.num_layers)
shapes = [layer.shape for layer in network]
l, u = get_input_spec(shapes=shapes, n=0, transformer='ibp', eps=0.01)
L = PolyExpSparse(network, SparseTensorBlock([], [], 3, torch.tensor([batch_size, network.size, network.size])), 0)
U = PolyExpSparse(network, SparseTensorBlock([], [], 3, torch.tensor([batch_size, network.size, network.size])), 0)
U = PolyExpSparse(network, SparseTensorBlock([], [], 3, torch.tensor([batch_size, network.size, network.size])), 0)
l = convert_to_sparse(l, float('-inf'), network.size, batch_size)
u = convert_to_sparse(u, float('inf'), network.size, batch_size)
L.const = copy.deepcopy(l)
U.const = copy.deepcopy(u)
abs_elem = Abs_elem_sparse({'llist' : llist, 'l' : l, 'u' : u, 'L' : L, 'U' : U}, {'l': 'Float', 'u': 'Float', 'L': 'PolyExp', 'U': 'PolyExp', 'llist': 'bool'}, network, batch_size=batch_size)
flow = Flow(abs_elem, deeppoly(), network, None)
flow.flow()