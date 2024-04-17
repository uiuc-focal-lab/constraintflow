from specs.spec import *
from certifier import Certifier
from common.abs_elem import Abs_elem
from common.transformer import *
from specs.network import LayerType
from newtransformer import *

import matplotlib.pyplot as plt
import itertools

def compute_size(shape):
    s = 1
    while len(shape)>0:
        s *= shape[0]
        shape = shape[1:]
    return s

def get_all_indices(nested_list, current_index=None):
    if current_index is None:
        current_index = []

    if not isinstance(nested_list, list):
        return [current_index]
    
    indices = []
    for i, sublist in enumerate(nested_list):
        new_index = current_index + [i]
        indices.extend(get_all_indices(sublist, new_index))
    
    return indices

# net = get_net(net_name='nets/mnist_relu_3_50.onnx')
net = get_net(net_name='nets/mnist-net_256x2.onnx')
# net = get_net(net_name='nets/mnist_0.1.onnx')
# djkhf

neighbours = dict()
shapes = [net.input_shape]
for layer in net:
    shapes.append(layer.shape)

# size = 0
# for i in range(len(shapes)):
#     size += compute_size(shapes[i])
# print(size)
# kdzjg

l, u, L, U = get_input_spec(shapes=shapes, n=0, transformer='deeppoly', eps=0.0)
# l, u, Z = get_input_spec(shapes=shapes, n=0, transformer='deepz', eps=0.0)

# abs_elem = Abs_elem({'l': l, 'u': u}, {'l': 'float', 'u': 'float'}, shapes)
abs_elem = Abs_elem({'l': l, 'u': u, 'L': L, 'U': U}, {'l': 'float', 'u': 'float', 'L': 'PolyExp', 'U': 'PolyExp'}, shapes)
# abs_elem = Abs_elem({'l': l, 'u': u, 'Z': Z}, {'l': 'float', 'u': 'float', 'Z': 'SymExp'}, shapes)


ctr = 0
for i in range(compute_size(shapes[0])):
    neighbours[ctr] = []
    ctr += 1

for layer in range(1, len(shapes)):
    size = compute_size(shapes[layer])
    prev_size = compute_size(shapes[layer-1])
    p_start = ctr - prev_size
    p_end = ctr - 1 
    c_start = ctr 
    c_end = ctr + size - 1 

    if layer%2==0:
        for j in range(size):
            neighbours[ctr] = [ctr - size] 
            ctr += 1
    else:
        prev_layer = [j for j in range(p_start, p_end+1)]
        for j in range(size):
            neighbours[ctr] = prev_layer
            ctr += 1

print(shapes[0])
for i in range(1, len(shapes)):
    layer = net[i-1]
    if layer.weight == None:
        print(layer.weight, shapes[i])
    else:
        print(layer.weight.shape, shapes[i])

# print(net[0].prev[784])
# print(net[0].index_hash[784])
# print(net[0].prev[784 + 196])
# print(net[0].index_hash[784 + 196])
# print(net[0].prev[784 + 196*2])
# print(net[0].index_hash[784 + 196*2])
# print(net[0].prev[784 + 196*3])
# print(net[0].index_hash[784 + 196*3])
# print(net[0].prev[784 + 196*4])
# print(net[0].index_hash[784 + 196*4])
# print(net[0].prev[784 + 196*5])
# print(net[0].index_hash[784 + 196*5])
# print(net[0].prev[784 + 196*6])
# print(net[0].index_hash[784 + 196*6])
# for layer in net:
#     print(id(layer.prev))
# kjd
# for layer in range(len(shapes) - 1):
#     d = list(net[layer].prev.keys())
#     print(d[0], d[-1])
#     # for j in range(5):
#     #     k = d[j]
#     #     print(net[layer].prev[k])
#     #     print(neighbours[k])
#     #     print(net[layer].prev[k] == neighbours[k])
# print(net[layer].prev)
# jhdg


# for idx in itertools.product(*[range(dim) for dim in shapes[0]]):
#     neighbours[(0, idx)] = []
# for i in range(1, len(shapes)):

#     for idx in itertools.product(*[range(dim) for dim in shapes[i]]):
#         # if net[i].type==LayerType.ReLU:
#         if i%2==0:
#             neighbours[(i, idx)] = [(i-1, idx)]
#         # elif net[i].type==LayerType.Linear:
#         else:
#             prev_indices = itertools.product(*[range(dim) for dim in shapes[i-1]])
#             neighbours[(i, idx)] = [(i-1, j) for j in prev_indices]
#         # elif net[i].type==LayerType.Conv2D:
            
# # for layer in net:
# #     for idx in itertools.product(*[range(dim) for dim in layer.shape[i]]):
# #     print(len(layer.prev))


transformer = CflowNewDeepPoly()
# transformer = CflowNewInterval()
# transformer = CflowDeepZ()
# transformer = Cflowdeeppoly()
# transformer = Cflowibp()
certifier = Certifier(abs_elem, transformer, net, neighbours)
certifier.flow()
# print(certifier.abs_elem.d['l'][-1])
# print(certifier.abs_elem.d['u'][-1])