from specs.spec import *
from certifier import Certifier
from common.abs_elem import Abs_elem
from common.transformer import *

import matplotlib.pyplot as plt
import itertools

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

# # Example usage:
# nested_list = [[[1, 2, 3], [4, 5, 6]]]
# print(len(nested_list))
# print(len(nested_list[0]))
# print(len(nested_list[0][0]))
# # kjehf
# all_indices = get_all_indices(nested_list)

# print(all_indices)

# sjdfh


net = get_net(net_name='nets/mnist_relu_3_50.onnx')
print("input layer", net.input_shape)
for layer in net:
    print(layer.type, layer.shape)


neighbours = dict()
shapes = [net.input_shape]
for layer in net:
    shapes.append(layer.shape)

l, u, L, U = get_input_spec(shapes=shapes, n=0, transformer='deeppoly', eps=0.0)

# print(len(L))
# print(type(L[0]))
# print(len(L[0]))
# print(type(L[0][0]))
# print(len(L[0][0]))
# print(type(L[0][0][0]))
# print(len(L[0][0][0]))
# print(type(L[0][0][0][0]))
# print(len(L[0][0][0][0]))
# print(type(L[0][0][0][0][0]))

abs_elem1 = Abs_elem({'l': l, 'u': u, 'L': L, 'U': U}, {'l': 'float', 'u': 'float', 'L': 'PolyExp', 'U': 'PolyExp'}, shapes)
# x = abs_elem1.get_elem('L', (1, (0,0,0,0)))
# print(type(x))
# kasjg
print(shapes)
for idx in itertools.product(*[range(dim) for dim in shapes[0]]):
    neighbours[(0, idx)] = []
for i in range(1, len(shapes)):
    for idx in itertools.product(*[range(dim) for dim in shapes[i]]):
        if i%2==0:
            neighbours[(i, idx)] = [(i-1, idx)]
        else:
            prev_indices = itertools.product(*[range(dim) for dim in shapes[i-1]])
            neighbours[(i, idx)] = [(i-1, j) for j in prev_indices]
# print(neighbours)
            # for k in range(shapes[i-1]):
            #     neighbours[(i, j)].append((i-1, k))
# abs_elem1 = Abs_elem({'l': l, 'u': u, 'L': L, 'U': U}, net.shapes)
# abs_elem1 = Abs_elem({'l': l, 'u': u}, {'l': 'float', 'u': 'float'}, shapes)
# print(abs_elem1.d['l'][0].shape)
transformer1 = CflowDeepPoly()
certifier1 = Certifier(abs_elem1, transformer1, net, neighbours)
certifier1.flow()
print(certifier1.abs_elem.d['l'][-1])
print(certifier1.abs_elem.d['u'][-1])