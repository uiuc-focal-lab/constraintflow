import torch
import math
import onnx
import numpy as np
import torch.nn as nn
import copy

from onnx import numpy_helper
from constraintflow.lib.network import Layer, LayerType, Network

from collections import deque

def compute_size(shape):
    s = 1
    while len(shape)>0:
        s *= shape[0]
        shape = shape[1:]
    return s


def get_net_format(net_name):
    net_format = None
    if 'pt' in net_name:
        net_format = 'torch'
    if 'onnx' in net_name:
        net_format = 'onnx'
    return net_format

def get_net(net_name, spec_weight, spec_bias, no_sparsity):
    net_format = get_net_format(net_name)
    if net_format == 'onnx':
        net_onnx = onnx.load(net_name)
        net = parse_onnx_layers(net_onnx, spec_weight, spec_bias, no_sparsity)
    else:
        raise ValueError("Unsupported net format!")

    net.net_name = net_name

    print('Network loaded')
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


def parse_onnx_layers(net, spec_weight, spec_bias, no_sparsity):
    input_shape = [dim.dim_value for dim in net.graph.input[0].type.tensor_type.shape.dim]
    input_shape = [1 if i == 0 else i for i in input_shape]
    if len(input_shape)==3:
        input_shape = [1] + input_shape
    input_size = compute_size(input_shape)
    # Create the new Network object
    layers = Network(input_name=net.graph.input[0].name, input_shape=input_shape, input_size=input_size, input_start=0, input_end=input_size, net_format='onnx', no_sparsity=no_sparsity)
    num_layers = len(net.graph.node)
    layers.num_layers = num_layers
    model_name_to_val_dict = {init_vals.name: torch.tensor(numpy_helper.to_array(init_vals)) for init_vals in net.graph.initializer}

    layers.size = input_size
    shape = input_shape

    names_hash = dict()
    index = 0
    names_hash[net.graph.input[0].name] = index
    parents = dict()
    
    layer = Layer(type=LayerType.Input, shape=input_shape, size=input_size, start=0, end=input_size)
    layers.append(layer)
    for cur_layer in range(num_layers):
        node = net.graph.node[cur_layer]
        operation = node.op_type
        nd_inps = node.input
        index+=1


        if operation == 'Conv':
            names_hash[str(net.graph.node[cur_layer].output[0])] = index
            parents[index] = [names_hash[str(net.graph.node[cur_layer].input[0])]]

            if isinstance(nd_inps[0], str):
                w_key = None
                b_key = None 
                for i in range(len(nd_inps)):
                    if 'weight' in nd_inps[i]:
                        w_key = nd_inps[i]
                    if 'bias' in nd_inps[i]:
                        b_key = nd_inps[i]
                if b_key == None:
                    layer = Layer(weight=model_name_to_val_dict[w_key], bias=torch.zeros(model_name_to_val_dict[w_key].shape[0]), type=LayerType.Conv2D, identifier=index, parents=parents[index])
                else:
                    layer = Layer(weight=model_name_to_val_dict[w_key], bias=model_name_to_val_dict[b_key], type=LayerType.Conv2D, identifier=index, parents=parents[index])

            else:
                layer = Layer(weight=model_name_to_val_dict[nd_inps[1]], bias=(model_name_to_val_dict[nd_inps[2]]), type=LayerType.Conv2D, identifier=index, parents=parents[index])
            layers.append(layer)

            layer.kernel_size = (node.attribute[2].ints[0], node.attribute[2].ints[1])
            layer.padding = (node.attribute[3].ints[0], node.attribute[3].ints[1])
            layer.stride = (node.attribute[4].ints[0], node.attribute[4].ints[1])
            layer.dilation = (1, 1)
            
            shape = layers[parents[index][0]].shape
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
            names_hash[str(net.graph.node[cur_layer].output[0])] = index
            parents[index] = [names_hash[str(net.graph.node[cur_layer].input[0])]]
            
            # Making some weird assumption that the weight is always 1th index
            layer = Layer(weight=model_name_to_val_dict[nd_inps[1]], bias=(model_name_to_val_dict[nd_inps[2]]), type=LayerType.Linear, identifier=index, parents=parents[index])
            layers.append(layer)
            [i_1, i_2, i_3, i_4] = shape 
            [w_1, w_2] = layer.weight.shape 
            o_1 = i_1 
            o_2 = w_1 
            o_3 = 1 
            o_4 = 1 
            layer.shape = [o_1, o_2, o_3, o_4]

            
        elif operation == 'Relu':
            names_hash[str(net.graph.node[cur_layer].output[0])] = index
            parents[index] = [names_hash[str(net.graph.node[cur_layer].input[0])]]
            
            layer = Layer(type=LayerType.ReLU, identifier=index, parents=parents[index])
            layers.append(layer)
            layer.shape = layers[parents[index][0]].shape

        elif operation == 'Sigmoid':
            names_hash[str(net.graph.node[cur_layer].output[0])] = index
            parents[index] = [names_hash[str(net.graph.node[cur_layer].input[0])]]
            
            layer = Layer(type=LayerType.Sigmoid, identifier=index, parents=parents[index])
            layers.append(layer)
            layer.shape = layers[parents[index][0]].shape

            


        elif operation == 'Concat':
            names_hash[str(net.graph.node[cur_layer].output[0])] = index
            if str(net.graph.node[cur_layer].input[0]) in names_hash and str(net.graph.node[cur_layer].input[1]) in names_hash:
                parents[index] = [names_hash[str(net.graph.node[cur_layer].input[0])], names_hash[str(net.graph.node[cur_layer].input[1])]]
            else:
                index-=1
                continue

            layer = Layer(type=LayerType.Concat, identifier=index, parents=parents[index])
            layers.append(layer)
            layer.shape = [1, layers[parents[index][0]].shape[1] + layers[parents[index][1]].shape[1], 1, 1]

            

        elif operation == 'MatMul':
            names_hash[str(net.graph.node[cur_layer].output[0])] = index
            parents[index] = [names_hash[str(net.graph.node[cur_layer].input[1])]]
            
            # Making some weird assumption that the weight is always 1th index
            layer = Layer(weight=model_name_to_val_dict[nd_inps[0]], type=LayerType.Linear, identifier=index, parents=parents[index])
            layers.append(layer)
            [w_1, w_2] = layer.weight.shape 
            o_1 = 1 
            o_2 = w_1 
            o_3 = 1 
            o_4 = 1 
            layer.shape = [o_1, o_2, o_3, o_4]

        
        elif operation == 'Add':
            if nd_inps[1] not in model_name_to_val_dict:
                names_hash[str(net.graph.node[cur_layer].output[0])] = index
                parent1 = names_hash[nd_inps[0]]
                parent2 = names_hash[nd_inps[1]]
                parents[index] = [parent1, parent2]
                layer = Layer(weight=None, type=LayerType.Add, identifier=index, parents=parents[index])
                layers.append(layer)
                layer.shape = copy.deepcopy(layers[parents[index][0]].shape)
            else:
                index -= 1
                names_hash[str(net.graph.node[cur_layer].output[0])] = index
                layer = layers[-1]
                layer.bias = model_name_to_val_dict[nd_inps[1]]


            


        else:
            if len(net.graph.node[cur_layer].input)>0:
                if str(net.graph.node[cur_layer].input[0]) in names_hash:
                    names_hash[str(net.graph.node[cur_layer].output[0])] = names_hash[str(net.graph.node[cur_layer].input[0])]

            index-=1
            # assert(f"{operation} not supported")
            continue

        layer.size = compute_size(layer.shape)
        layer.start = layers.size 
        layers.size += layer.size 
        layer.end = layers.size
        shape = layer.shape

    if layers[-1].type != LayerType.Linear:
        layer = Layer(weight=spec_weight, bias=spec_bias, type=LayerType.Linear, identifier=index+1, parents=[layers[-1].identifier])
        layer.last_layer = True
        layer.shape = [1, spec_weight.shape[-2], 1, 1]
        layer.size = compute_size(layer.shape)
        layer.start = layers.size 
        layers.size += layer.size 
        layer.end = layers.size
        layers.append(layer)
    else:
        layers[-1].weight = (spec_weight@layers[-1].weight)[0] 
        layers[-1].bias = (spec_weight@layers[-1].bias + spec_bias)[0]
        layers[-1].shape[1] = spec_weight.shape[-2]
        layers.size = layers.size - layers[-1].size
        layers[-1].size = compute_size(layers[-1].shape)
        layers.size += layers[-1].size
        layers[-1].end = layers.size
    layers = post_process(layers)
    return layers




def post_process(layers):
    identifier_to_index = dict()
    for i, layer in enumerate(layers):
        identifier_to_index[layer.identifier] = i
    for i, layer in enumerate(layers):
        for j, parent in enumerate(layer.parents):
            layers[identifier_to_index[parent]].children.append(layer.identifier)
    queue = deque([layers[identifier_to_index[0]]])
    visited = set()
    layer_num = 0
    new_layers = layers.similar()
    while queue:
        layer = queue.popleft()
        if layer in visited:
            continue
        layer.new_identifier = layer_num
        new_layers.append(layer)
        layer_num += 1
        visited.add(layer)
        for child in layer.children:
            if child not in visited:
                parent_visited = True 
                for parent in layers[identifier_to_index[child]].parents:
                    if layers[identifier_to_index[parent]] not in visited:
                        parent_visited = False
                        break
                if parent_visited:
                    queue.append(layers[identifier_to_index[child]])
    
    start = 0
    for i, layer in enumerate(new_layers):
        layer.start = start 
        layer.end = start + layer.size
        parents = []
        for parent in layer.parents:
            parents.append(layers[parent].new_identifier)
        layer.new_parents = parents
        children = []
        for child in layer.children:
            children.append(layers[child].new_identifier)
        layer.new_children = children
        start += layer.size
    for i,layer in enumerate(new_layers):
        layer.identifier = layer.new_identifier
        layer.parents = layer.new_parents
        layer.children = layer.new_children
    
    
    return new_layers


def parse_torch_layers(net, input_shape):
    if len(input_shape)==3:
        input_shape = [1] + input_shape
    input_size = compute_size(input_shape)

    # Create the new Network object
    layers = Network(input_name='input.torch', input_shape=input_shape, input_size=input_size, input_start=0, input_end=input_size, net_format='torch')
    layers.num_layers = 1

    layers.size = input_size
    shape = input_shape

    names_hash = dict()
    index = 0
    names_hash[net.graph.input[0].name] = index
    parents = dict()
    
    layer = Layer(type=LayerType.Input, shape=input_shape, size=input_size, start=0, end=input_size)
    layers.append(layer)

    for cur_layer, torch_layer in enumerate(net.blocks):
        index+=1

        if isinstance(torch_layer, torch.nn.Conv2d):
            names_hash[cur_layer] = index
            parents[index] = [names_hash[cur_layer-1]]

            layer = Layer(weight=torch_layer.weight, bias=torch_layer.bias, type=LayerType.Conv2D, identifier=index, parents=parents[index])
            layers.append(layer)

            layer.kernel_size = torch_layer.kernel_size
            layer.padding = (torch_layer.padding, torch_layer.padding)
            layer.stride = (torch_layer.stride, torch_layer.stride)
            layer.dilation = (torch_layer.dilation, torch_layer.dilation)
            
            shape = layers[parents[index][0]].shape
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


        elif isinstance(torch_layer, torch.nn.ReLU):
            names_hash[cur_layer] = index
            parents[index] = [names_hash[cur_layer-1]]
            
            layer = Layer(type=LayerType.ReLU, identifier=index, parents=parents[index])
            layers.append(layer)
            layer.shape = layers[parents[index][0]].shape

        elif isinstance(torch_layer, torch.nn.Linear):
            names_hash[cur_layer] = index
            parents[index] = [names_hash[cur_layer-1]]
            
            layer = Layer(weight=torch_layer.weight, bias=torch_layer.bias, type=LayerType.Linear, identifier=index, parents=parents[index])
            layers.append(layer)
            [i_1, i_2, i_3, i_4] = shape 
            [w_1, w_2] = layer.weight.shape 
            o_1 = i_1 
            o_2 = w_1 
            o_3 = 1 
            o_4 = 1 
            layer.shape = [o_1, o_2, o_3, o_4]

            
            
        else:
            print(type(torch_layer))
            assert(False)
    return layers


def load_pytorch_network(dataset, n_class=10, input_size=32, input_channel=3, conv_widths=None,
                 kernel_sizes=None, linear_sizes=None, depth_conv=None, paddings=None, strides=None,
                 dilations=None, pool=False, net_dim=None, bn=False, bn2=False, max=False, scale_width=True, mean=0, sigma=1):
    if kernel_sizes is None:
        kernel_sizes = [3]
    if conv_widths is None:
        conv_widths = [2]
    if linear_sizes is None:
        linear_sizes = [200]
    if paddings is None:
        paddings = [1]
    if strides is None:
        strides = [2]
    if dilations is None:
        dilations = [1]
    if net_dim is None:
        net_dim = input_size

    if len(conv_widths) != len(kernel_sizes):
        kernel_sizes = len(conv_widths) * [kernel_sizes[0]]
    if len(conv_widths) != len(paddings):
        paddings = len(conv_widths) * [paddings[0]]
    if len(conv_widths) != len(strides):
        strides = len(conv_widths) * [strides[0]]
    if len(conv_widths) != len(dilations):
        dilations = len(conv_widths) * [dilations[0]]

    if dataset == "fashionmnist":
        mean = 0.1307
        sigma = 0.3081
    elif dataset == "cifar10":
        mean = [0.4914, 0.4822, 0.4465]
        sigma = [0.2023, 0.1994, 0.2010]
    elif dataset == "tinyimagenet":
        mean = [0.4802, 0.4481, 0.3975]
        sigma = [0.2302, 0.2265, 0.2262]

    layers = []
    # layers += [Normalization((input_channel,input_size,input_size),mean, sigma)]

    N = net_dim
    n_channels = input_channel
    dims = [(n_channels,N,N)]

    for width, kernel_size, padding, stride, dilation in zip(conv_widths, kernel_sizes, paddings, strides, dilations):
        if scale_width:
            width *= 16
        N = int(np.floor((N + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1))
        layers += [nn.Conv2d(n_channels, int(width), kernel_size, stride=stride, padding=padding, dilation=dilation)]
        if bn:
            layers += [nn.BatchNorm2d(int(width))]
        if max:
            layers += [nn.MaxPool2d(int(width))]
        layers += [nn.ReLU((int(width), N, N))]
        n_channels = int(width)
        dims += 2*[(n_channels,N,N)]

    if depth_conv is not None:
        layers += [nn.Conv2d(n_channels, depth_conv, 1, stride=1, padding=0),
                    nn.ReLU((n_channels, N, N))]
        n_channels = depth_conv
        dims += 2*[(n_channels,N,N)]

    if pool:
        layers += [nn.GlobalAvgPool2d()]
        dims += 2 * [(n_channels, 1, 1)]
        N=1

    layers += [nn.Flatten()]
    N = n_channels * N ** 2
    dims += [(N,)]

    for width in linear_sizes:
        if width == 0:
            continue
        layers += [nn.Linear(int(N), int(width))]
        if bn2:
            layers += [nn.BatchNorm1d(int(width))]
        layers += [nn.ReLU(width)]
        N = width
        dims+=2*[(N,)]

    layers += [nn.Linear(N, n_class)]
    dims+=[(n_class,)]

    blocks = nn.Sequential(*layers)

    return blocks
