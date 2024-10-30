import torch
import functools
import sys
import pickle


from compiled_code.specs.spec import *
from compiled_code.certifier_sparse import Certifier
from compiled_code.common.abs_elem import Abs_elem_sparse
from compiled_code.specs.network import LayerType
from compiled_code.transformers_compiled2 import *


batch_size = int(sys.argv[2])
network = get_net(sys.argv[1])
# input_filename = sys.argv[2]
# print(input_filename)
# with open(input_filename, 'rb') as file:
# 	input_spec = pickle.load(file)
llist = torch.tensor([True] + [False]*network.num_layers)
shapes = [layer.shape for layer in network]
l, u = get_input_spec(shapes=shapes, n=0, transformer='ibp', eps=0.01)
# l = input_spec[1]
# u = input_spec[2]
L = PolyExpSparse(network, SparseTensorBlock([], [], 3, torch.tensor([batch_size, network.size, network.size])), 0)
U = PolyExpSparse(network, SparseTensorBlock([], [], 3, torch.tensor([batch_size, network.size, network.size])), 0)
# L = L.convert_to_polyexp_sparse(network, batch_size)
# U = U.convert_to_polyexp_sparse(network, batch_size)
l = convert_to_sparse(l, float('-inf'), network.size, batch_size)
u = convert_to_sparse(u, float('inf'), network.size, batch_size)
L.const = copy.deepcopy(l)
U.const = copy.deepcopy(u)
abs_elem = Abs_elem_sparse({'llist' : llist, 'l' : l, 'u' : u, 'L' : L, 'U' : U}, {'l': 'Float', 'u': 'Float', 'L': 'PolyExp', 'U': 'PolyExp', 'llist': 'bool'}, network, batch_size=batch_size)
certifier = Certifier(abs_elem, deeppoly(), network, None)
certifier.flow()
