import torch
import functools
import sys
import pickle


from specs.spec import *
from certifier_sparse import Certifier
from common.abs_elem import Abs_elem_sparse
from common.polyexp import Network_graph
from specs.network import LayerType
from transformers_compiled2 import *


network_file = sys.argv[1]
network = get_net(network_file)
input_filename = sys.argv[2]
with open(input_filename, 'rb') as file:
	input_spec = pickle.load(file)
shapes = [network.input_shape]
for layer in network:
	shapes.append(layer.shape)
network = Network_graph(shapes, network)
llist = torch.tensor([True, False, False, False, False, False, False])
l = input_spec[1]
u = input_spec[2]
L = input_spec[3].convert_to_polyexp_sparse(network)
U = input_spec[4].convert_to_polyexp_sparse(network)
l = convert_to_sparse(l, float('-inf'))
u = convert_to_sparse(u, float('inf'))
L.const = copy.deepcopy(l)
U.const = copy.deepcopy(u)
abs_elem = Abs_elem_sparse({'llist' : llist, 'l' : l, 'u' : u, 'L' : L, 'U' : U}, {'l': 'Float', 'u': 'Float', 'L': 'PolyExp', 'U': 'PolyExp', 'llist': 'bool'}, network)
certifier = Certifier(abs_elem, deeppoly(), network, None)
certifier.flow()
