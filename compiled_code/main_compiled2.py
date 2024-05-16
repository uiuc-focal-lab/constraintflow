import torch
import functools
import sys
import pickle


from specs.spec import *
from certifier import Certifier
from common.abs_elem import Abs_elem
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
t = input_spec[0]
l = input_spec[1]
u = input_spec[2]
Z = input_spec[3]
abs_elem = Abs_elem({'t' : t, 'l' : l, 'u' : u, 'Z' : Z}, {'l': 'Float', 'u': 'Float', 'Z': 'ZonoExp', 't': 'bool'}, shapes)
