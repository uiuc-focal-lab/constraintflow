import torch
import math
import itertools

from torch.nn import ReLU, Linear, Conv2d
from onnx import numpy_helper
from specs.network import Layer, LayerType, Network

from test_torch import convert_to_fully_connected

def compute_size(shape):
    s = 1
    while len(shape)>0:
        s *= shape[0]
        shape = shape[1:]
    return s

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
    layers = Network(input_name=net.graph.input[0].name, input_shape=input_shape, input_size=input_size, net_format='onnx')
    num_layers = len(net.graph.node)
    model_name_to_val_dict = {init_vals.name: torch.tensor(numpy_helper.to_array(init_vals)) for init_vals in
                              net.graph.initializer}
    
    curr_s = input_size
    for cur_layer in range(num_layers):
        node = net.graph.node[cur_layer]
        operation = node.op_type
        nd_inps = node.input
        # print(operation)
        if operation == 'MatMul':
            print('Linear')
            # Assuming that the add node is followed by the MatMul node
            add_node = net.graph.node[cur_layer + 1]
            bias = model_name_to_val_dict[add_node.input[1]]
            weight = model_name_to_val_dict[nd_inps[0]]
            # print(weight.shape)
            # print(bias.shape)
            # Making some weird assumption that the weight is always 0th index
            # layer = Layer(weight=model_name_to_val_dict[nd_inps[0]], bias=bias, type=LayerType.Linear)
            # layers.append(layer)

            layer = Layer(weight=weight, bias=bias, type=LayerType.Linear)
            layers.append(layer)
            if len(layers)==1:
                # [i_1, i_2, i_3, i_4] = layers.input_shape
                [i_1] = layers.input_shape
                prev_size = layers.input_size
            else:
                # [i_1, i_2, i_3, i_4] = layers[-2].shape
                i_1 = layers[-2].shape
                prev_size = layers[-2].size
            [w_1, w_2] = layer.weight.shape 
            o_1 = w_1 
            # o_2 = w_1 
            # o_3 = 1 
            # o_4 = 1 
            # layer.shape = [o_1, o_2, o_3, o_4]
            layer.shape = [o_1]
            layer.size = compute_size(layer.shape)
            layer.prev = dict()
            layer.prev_weight = dict()
            curr_e = curr_s + layer.size - 1
            prev_s = curr_s - prev_size
            prev_e = curr_s - 1
            for i in range(curr_s, curr_e + 1):
                layer.prev[i] = [j for j in range(prev_s, prev_e + 1)]
                layer.prev_weight[i] = layer.weight[i - curr_s]
            curr_s = curr_e + 1

        elif operation == 'Conv':
            layer = Layer(weight=model_name_to_val_dict[nd_inps[1]], bias=(model_name_to_val_dict[nd_inps[2]]),
                          type=LayerType.Conv2D)
            layer.kernel_size = (node.attribute[2].ints[0], node.attribute[2].ints[1])
            layer.padding = (node.attribute[3].ints[0], node.attribute[3].ints[1])
            layer.stride = (node.attribute[4].ints[0], node.attribute[4].ints[1])
            layer.dilation = (1, 1)
            
            layers.append(layer)
            if len(layers)==1:
                [i_1, i_2, i_3, i_4] = layers.input_shape
                prev_size = layers.input_size
            else:
                [i_1, i_2, i_3, i_4] = layers[-2].shape 
                prev_size = layers[-2].size
            [k_1, k_2, k_3, k_4] = layer.weight.shape 
            (p_1, p_2) = layer.padding
            (s_1, s_2) = layer.stride 
            
            o_1 = i_1 
            o_2 = k_1 
            o_3 = math.floor((i_3 + 2*p_1 - k_3) / s_1) + 1
            o_4 = math.floor((i_4 + 2*p_2 - k_4) / s_2) + 1
            layer.shape = [o_1, o_2, o_3, o_4]
            # layer.shape = [o_2*o_3*o_4]
            layer.size = compute_size(layer.shape)
            layer.prev = dict()
            layer.prev_weight = dict()
            layer.dict_hash = dict()
            curr_e = curr_s + layer.size - 1 
            prev_s = curr_s - prev_size
            layer.bias = (layer.bias.unsqueeze(1).repeat(1, o_3*o_4)).flatten()
            # layer.weight = convert_to_fully_connected(k_1, k_2, layer.weight, i_3, i_4, o_3, o_4, k_3, k_4, s_1, s_2, p_1, p_2)

            # layer_num = len(layers)
            # print(curr_s, curr_e)
            # for j, idx in enumerate(itertools.product(*[range(dim) for dim in layer.shape])):
            #     (b, c_out, h_out, w_out) = idx 
            #     # layer.prev[(layer_num, idx)] = []
            #     # layer.prev_weight[(layer_num, idx)] = []
            #     layer.prev[j + curr_s] = []
            #     layer.prev_weight[j + curr_s] = []
            #     # print(i_2, k_3, k_4)
            #     for k in range(i_2):
            #         for h in range(k_3):
            #             for w in range(k_4):
            #                 h_in = layer.stride[0] * h_out - layer.padding[0] + h
            #                 w_in = layer.stride[1] * w_out - layer.padding[1] + w 
            #                 c_in = k 
            #                 weight = layer.weight[(c_out, c_in, h, w)]
            #                 if b>=0 and c_in>=0 and h_in>=0 and w_in>=0:
            #                     if h_in<i_3 and w_in<i_4:
            #                         prev_linear_index = prev_s + b*i_2*i_3*i_4 + c_in*i_3*i_4 + h_in*i_4 + w_in
            #                         layer.prev[j + curr_s].append(prev_linear_index)
            #                         layer.prev_weight[j + curr_s].append(weight)
            #                         layer.index_hash[j + curr_s] = (layer_num, idx)
            #                         # layer.prev[(layer_num, idx)].append((layer_num-1, (b, c_in, h_in, w_in)))
            #                         # layer.prev_weight[(layer_num, idx)].append(weight)
            curr_s = curr_e + 1
            

        elif operation == 'Gemm':
            print('Linear')
            # Making some weird assumption that the weight is always 1th index
            layer = Layer(weight=model_name_to_val_dict[nd_inps[1]], bias=(model_name_to_val_dict[nd_inps[2]]),
                          type=LayerType.Linear)
            layers.append(layer)
            if len(layers)==1:
                [i_1, i_2, i_3, i_4] = layers.input_shape
                prev_size = layers.input_size
            else:
                [i_1, i_2, i_3, i_4] = layers[-2].shape
                prev_size = layers[-2].size
            [w_1, w_2] = layer.weight.shape 
            o_1 = i_1 
            o_2 = w_1 
            o_3 = 1 
            o_4 = 1 
            layer.shape = [o_1, o_2, o_3, o_4]
            layer.size = compute_size(layer.shape)
            layer.prev = dict()
            layer.prev_weight = dict()
            curr_e = curr_s + layer.size - 1
            prev_s = curr_s - prev_size
            prev_e = curr_s - 1
            for i in range(curr_s, curr_e + 1):
                layer.prev[i] = [j for j in range(prev_s, prev_e + 1)]
                layer.prev_weight[i] = layer.weight[i - curr_s]
            curr_s = curr_e + 1
            # layer_num = len(layers)
            # for idx in itertools.product(*[range(dim) for dim in layers[-1].shape]):
            #     if len(layers)>1:
            #         prev_indices = itertools.product(*[range(dim) for dim in layers[-2].shape])
            #     else:
            #         prev_indices = itertools.product(*[range(dim) for dim in layers.input_shape])
            #     layers[-1].prev[(layer_num, idx)] = [(layer_num-1, j) for j in prev_indices]


        elif operation == 'Relu':
            print('Relu')
            layers.append(Layer(type=LayerType.ReLU))
            layers[-1].shape = layers[-2].shape 
            layers[-1].size = layers[-2].size 
            layers[-1].prev = dict()
            layers[-1].prev_weight = dict()
            layers[-1].dict_hash = dict()
            layer_num = len(layers)
            curr_e = curr_s + layers[-1].size - 1
            for i in range(curr_s, curr_e + 1):
                layers[-1].prev[i] = [i - layers[-1].size]
            curr_s = curr_e + 1
            # for idx in itertools.product(*[range(dim) for dim in layers[-1].shape]):
            #     layers[-1].prev[(layer_num, idx)] = [(layer_num-1, idx)]

        else:
            continue

        if len(layers)>0:
            print(cur_layer, layers[-1].type, layers[-1].shape, layers[-1].size)

    return layers


def parse_torch_layers(net):
    layers = Network(torch_net=net, net_format='torch')

    for torch_layer in net:
        if isinstance(torch_layer, ReLU):
            layers.append(Layer(type=LayerType.ReLU))
        elif isinstance(torch_layer, Linear):
            layer = Layer(weight=torch_layer.weight, bias=torch_layer.bias, type=LayerType.Linear)
            layers.append(layer)
        elif isinstance(torch_layer, Conv2d):
            layer = Layer(weight=torch_layer.weight, bias=torch_layer.bias,
                          type=LayerType.Conv2D)
            layer.kernel_size = torch_layer.kernel_size
            layer.padding = torch_layer.padding
            layer.stride = torch_layer.stride
            layer.dilation = (1, 1)
            layers.append(layer)

    return layers