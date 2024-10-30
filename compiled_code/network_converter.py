from enum import Enum

# import the necessary packages
import torch.nn
from torch.nn import ModuleList
from torch.nn import Module
from torch.nn import Conv2d
from torch.nn import Linear
from torch.nn import MaxPool2d
from torch.nn import ReLU
from torch.nn import Flatten
from torch.nn import LogSoftmax
from torch.nn import Sequential
from torch import flatten
from onnx import numpy_helper
import onnx2pytorch
from enum import Enum

class Network(list):
    def __init__(self, net_name=None, input_name=None, input_shape=None, torch_net=None):
        super().__init__()
        self.net_name = net_name
        self.input_name = input_name
        self.input_shape = input_shape
        self.torch_net = torch_net


class Layer:
    def __init__(self, weight=None, bias=None, type=None):
        self.weight = weight
        self.bias = bias
        self.type = type


class LayerType(Enum):
    Conv2D = 1
    Linear = 2
    ReLU = 3
    Flatten = 4
    MaxPool1D = 5
    Normalization = 6
    NoOp = 7
    
class TransformedNet(Module):
    def __init__(self, layers, ignore_last_layer=False, all_linear=False):
        super().__init__()
        constructed_layers = []
        self.conv_layer_count = 0
        self.linear_layer_count = 0
        # Initially the there will be only one channel.
        self.last_conv_layer_channels = 1
        if all_linear:
            constructed_layers.append(Flatten(start_dim=1))
        for layer in layers:
            if layer.type == LayerType.Linear:
                if self.linear_layer_count == 0 and self.conv_layer_count > 0:
                    constructed_layers.append(Flatten(start_dim=1))
                self.linear_layer_count += 1
                shape = layer.weight.shape
                if shape is None:
                    raise ValueError("Shape of the linear layer should be not null")
                input_len = shape[1]
                output_len = shape[0]
                constructed_layers.append(Linear(input_len, output_len))
                constructed_layers[-1].weight = torch.nn.Parameter(layer.weight)
                constructed_layers[-1].bias = torch.nn.Parameter(layer.bias)
            elif layer.type == LayerType.Conv2D:
                self.conv_layer_count += 1
                kernel_size = layer.kernel_size
                padding = layer.padding
                stride = layer.stride
                dilation = layer.dilation
                in_channels = self.last_conv_layer_channels
                out_channels = layer.weight.shape[0]
                self.last_conv_layer_channels = out_channels
                constructed_layers.append(Conv2d(in_channels=in_channels, out_channels=out_channels,
                                                kernel_size=kernel_size, stride=stride,
                                                padding=padding, dilation=dilation))
                constructed_layers[-1].weight = torch.nn.Parameter(layer.weight)
                constructed_layers[-1].bias = torch.nn.Parameter(layer.bias)
            elif layer.type == LayerType.ReLU:
                constructed_layers.append(ReLU())
            else:
                raise NotImplementedError("Layer conversion of type {} is not supported".format(layer.type))
        if ignore_last_layer:
            _ = constructed_layers.pop()
        self.model = Sequential(*constructed_layers)

    def forward(self, x):
        return self.model(x)


def convert_model(parsed_net, remove_last_layer=True, all_linear=False):
    return TransformedNet(parsed_net, remove_last_layer, all_linear)


def is_linear(net_name):
    if net_name == 'mnist-net_256x2.onnx':
        return True
    else:
        return False

def get_pytorch_net(model, remove_last_layer, all_linear):
    converted_model = convert_model(parsed_net=model, remove_last_layer=remove_last_layer, all_linear=all_linear)
    return converted_model

def parse_onnx_layers(net):
    input_shape = tuple([dim.dim_value for dim in net.graph.input[0].type.tensor_type.shape.dim])
    # Create the new Network object
    layers = Network(input_name=net.graph.input[0].name, input_shape=input_shape)
    num_layers = len(net.graph.node) 
    model_name_to_val_dict = {init_vals.name: torch.tensor(numpy_helper.to_array(init_vals)) for init_vals in
                              net.graph.initializer}

    final_relu = False
    for cur_layer in range(num_layers):
        final_relu = False
        node = net.graph.node[cur_layer]
        operation = node.op_type
        nd_inps = node.input
        if operation == 'MatMul':
            # Assuming that the add node is followed by the MatMul node
            add_node = net.graph.node[cur_layer + 1]
            bias = model_name_to_val_dict[add_node.input[1]]

            # Making some weird assumption that the weight is always 0th index
            layer = Layer(weight=model_name_to_val_dict[nd_inps[0]], bias=bias, type=LayerType.Linear)
            layers.append(layer)

        elif operation == 'Conv':
            layer = Layer(weight=model_name_to_val_dict[nd_inps[1]], bias=(model_name_to_val_dict[nd_inps[2]]),
                          type=LayerType.Conv2D)
            layer.kernel_size = (node.attribute[2].ints[0], node.attribute[2].ints[1])
            layer.padding = (node.attribute[3].ints[0], node.attribute[3].ints[1])
            layer.stride = (node.attribute[4].ints[0], node.attribute[4].ints[1])
            layer.dilation = (1, 1)
            layers.append(layer)

        elif operation == 'Gemm':
            # Making some weird assumption that the weight is always 1th index
            layer = Layer(weight=model_name_to_val_dict[nd_inps[1]], bias=(model_name_to_val_dict[nd_inps[2]]),
                          type=LayerType.Linear)
            layers.append(layer)

        elif operation == 'Relu':
            final_relu = True
            layers.append(Layer(type=LayerType.ReLU))
    
    # The final most layer is relu and no linear layer after that
    # remove the linear layer (Hack find better solutions).
    if final_relu is True:
        layers.pop()
    return layers

def onnx2torch(onnx_model):
    # find the input shape from onnx_model generally
    # https://github.com/onnx/onnx/issues/2657
    input_all = [node.name for node in onnx_model.graph.input]
    input_initializer = [node.name for node in onnx_model.graph.initializer]
    net_feed_input = list(set(input_all) - set(input_initializer))
    net_feed_input = [node for node in onnx_model.graph.input if node.name in net_feed_input]

    if len(net_feed_input) != 1:
        # in some rare case, we use the following way to find input shape but this is not always true (collins-rul-cnn)
        net_feed_input = [onnx_model.graph.input[0]]

    onnx_input_dims = net_feed_input[0].type.tensor_type.shape.dim
    onnx_shape = tuple(d.dim_value for d in onnx_input_dims[1:])

    pytorch_model = onnx2pytorch.ConvertModel(onnx_model, experimental=False, debug=True)
    pytorch_model.eval()
    pytorch_model.to(dtype=torch.get_default_dtype())

    return pytorch_model, onnx_shape