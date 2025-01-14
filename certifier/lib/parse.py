import torch
import math
import onnx

from onnx import numpy_helper
from certifier.lib.network import Layer, LayerType, Network
from certifier.lib.utils import compute_size


def get_net_format(net_name):
    net_format = None
    if 'pt' in net_name:
        net_format = 'torch'
    if 'onnx' in net_name:
        net_format = 'onnx'
    return net_format

def get_net(net_name, dataset=None):
    net_format = get_net_format(net_name)
    if net_format == 'onnx':
        net_onnx = onnx.load(net_name)
        net = parse_onnx_layers(net_onnx)
    else:
        raise ValueError("Unsupported net format!")

    net.net_name = net_name
    return net

def forward_layers(net, relu_mask, transformers):
    for layer in net:
        if layer.type == LayerType.ReLU:
            transformers.handle_relu(layer, optimize=True, relu_mask=relu_mask)
        elif layer.type == LayerType.Linear:
            if layer == net[-1]:
                transformers.handle_linear(layer, last_layer=True)
            else:
                transformers.handle_linear(layer)
        elif layer.type == LayerType.Conv2D:
            transformers.handle_conv2d(layer)
        elif layer.type == LayerType.Normalization:
            transformers.handle_normalization(layer)
    return transformers


def parse_onnx_layers(net):
    input_shape = [dim.dim_value for dim in net.graph.input[0].type.tensor_type.shape.dim]
    if len(input_shape)==3:
        input_shape = [1] + input_shape
    input_size = compute_size(input_shape)
    # Create the new Network object
    layers = Network(input_name=net.graph.input[0].name, input_shape=input_shape, input_size=input_size, input_start=0, input_end=input_size, net_format='onnx')
    num_layers = len(net.graph.node)
    layers.num_layers = num_layers
    model_name_to_val_dict = {init_vals.name: torch.tensor(numpy_helper.to_array(init_vals)) for init_vals in net.graph.initializer}

    layers.size = input_size
    shape = input_shape
    
    layer = Layer(type=LayerType.Input, shape=input_shape, size=input_size, start=0, end=input_size)
    layers.append(layer)
    for cur_layer in range(num_layers):
        node = net.graph.node[cur_layer]
        operation = node.op_type
        nd_inps = node.input

        if operation == 'Conv':
            layer = Layer(weight=model_name_to_val_dict[nd_inps[1]], bias=(model_name_to_val_dict[nd_inps[2]]), type=LayerType.Conv2D)
            layers.append(layer)

            layer.kernel_size = (node.attribute[2].ints[0], node.attribute[2].ints[1])
            layer.padding = (node.attribute[3].ints[0], node.attribute[3].ints[1])
            layer.stride = (node.attribute[4].ints[0], node.attribute[4].ints[1])
            layer.dilation = (1, 1)
            
            [i_1, i_2, i_3, i_4] = shape
            [k_1, k_2, k_3, k_4] = layer.weight.shape 
            (p_1, p_2) = layer.padding
            (s_1, s_2) = layer.stride 
            
            o_1 = i_1 
            o_2 = k_1 
            o_3 = math.floor((i_3 + 2*p_1 - k_3) / s_1) + 1
            o_4 = math.floor((i_4 + 2*p_2 - k_4) / s_2) + 1
            layer.shape = [o_1, o_2, o_3, o_4]
            layer.bias = (layer.bias.unsqueeze(1).repeat(1, o_3*o_4)).flatten()

        elif operation == 'Gemm':
            print('Linear')
            # Making some weird assumption that the weight is always 1th index
            layer = Layer(weight=model_name_to_val_dict[nd_inps[1]], bias=(model_name_to_val_dict[nd_inps[2]]), type=LayerType.Linear)
            layers.append(layer)
            [i_1, i_2, i_3, i_4] = shape 
            [w_1, w_2] = layer.weight.shape 
            o_1 = i_1 
            o_2 = w_1 
            o_3 = 1 
            o_4 = 1 
            layer.shape = [o_1, o_2, o_3, o_4]


        elif operation == 'Relu':
            print('Relu')
            layer = Layer(type=LayerType.ReLU)
            layers.append(layer)
            layer.shape = layers[-2].shape 

        else:
            assert(f"{operation} not supported")
            continue

        layer.size = compute_size(layer.shape)
        layer.start = layers.size 
        layers.size += layer.size 
        layer.end = layers.size
        shape = layer.shape

        print(cur_layer, layers[-1].type, layers[-1].shape, layers[-1].size)

    return layers

