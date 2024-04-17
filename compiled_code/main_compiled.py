import torch
import functools
import sys
import pickle


from specs.spec import *
from certifier import Certifier
from common.abs_elem import Abs_elem
from common.transformer import *
from specs.network import LayerType
from transformers_compiled import *


network_file = sys.argv[1]
network = get_net(network_file)
input_filename = sys.argv[2]
# input_spec = torch.load(input_filename)
with open(input_filename, 'rb') as file:
    input_spec = pickle.load(file)
shapes = [network.input_shape]
for layer in network:
	shapes.append(layer.shape)
l = input_spec[0]
u = input_spec[1]
abs_elem = Abs_elem({'l' : l, 'u' : u}, {'l': 'Float', 'u': 'Float'}, shapes)
certifier = Certifier(abs_elem, Ibp(), network, None)
certifier.flow()
