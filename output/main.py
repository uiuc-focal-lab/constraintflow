import torch
import functools
import sys


from interpreter.specs.spec import *
from interpreter.certifier import Certifier
from interpreter.common.abs_elem import Abs_elem
from interpreter.common.transformer import *
from interpreter.specs.network import LayerType
from transformer import *


network_file = sys.argv[1]
network = get_net(network_file)
input_filename = sys.argv[2]
input_spec = torch.load(input_filename)
shapes = [network.input_shape]
for layer in network:
	shapes.append(layer.shape)
l = input_spec[0]
u = input_spec[1]
abs_elem = Abs_elem({'l' : l, 'u' : u}, {'l': 'Float', 'u': 'Float'}, shapes)
certifier = Certifier(abs_elem, ibp, network, None)
certifier.flow()
