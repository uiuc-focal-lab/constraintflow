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
import torch.nn as nn


def convert_onnx_to_pytorch(onnx_network):
    layers = []
    for onnx_layer in onnx_network:
        if onnx_layer.type == LayerType.ReLU:
            layer = nn.ReLU()
        elif onnx_layer.type == LayerType.Linear:
            layer = nn.Linear(onnx_layer.weight.shape[1], onnx_layer.weight.shape[0])
            layer.weight = nn.Parameter(onnx_layer.weight)
            layer.bias = nn.Parameter(onnx_layer.bias)
            print(layer.weight.dtype)
            print(layer.bias.dtype)
        layers.append(layer)
    pytorch_model = nn.Sequential(*layers)
    return pytorch_model



    #   def mnist_model():
    #     model = nn.Sequential(
    #         nn.Conv2d(1, 16, 4, stride=2, padding=1),
    #         nn.ReLU(),
    #         nn.Conv2d(16, 32, 4, stride=2, padding=1),
    #         nn.ReLU(),
    #         Flatten(),
    #         nn.Linear(32*7*7,100),
    #         nn.ReLU(),
    #         nn.Linear(100, 10)
    #     )
    #     return model

network_file = sys.argv[1]
network = get_net(network_file)

pytorch_network = convert_onnx_to_pytorch(network)

file_path = 'pytorch_mnist_relu.pkl'
df
with open(file_path, 'wb') as file:
    pickle.dump(pytorch_network, file)
fd
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
