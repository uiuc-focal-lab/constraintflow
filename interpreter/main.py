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

net = get_net(net_name='nets/mnist_relu_3_50.onnx')
print("input layer", net.input_shape)
for layer in net:
    print(layer.type, layer.shape)


neighbours = dict()
shapes = [net.input_shape]
for layer in net:
    shapes.append(layer.shape)

l, u, L, U = get_input_spec(shapes=shapes, n=0, transformer='deeppoly', eps=0.02)
abs_elem = Abs_elem({'l': l, 'u': u, 'L': L, 'U': U}, {'l': 'float', 'u': 'float', 'L': 'PolyExp', 'U': 'PolyExp'}, shapes)

for idx in itertools.product(*[range(dim) for dim in shapes[0]]):
    neighbours[(0, idx)] = []
for i in range(1, len(shapes)):
    for idx in itertools.product(*[range(dim) for dim in shapes[i]]):
        if i%2==0:
            neighbours[(i, idx)] = [(i-1, idx)]
        else:
            prev_indices = itertools.product(*[range(dim) for dim in shapes[i-1]])
            neighbours[(i, idx)] = [(i-1, j) for j in prev_indices]
            
transformer = CflowDeepPoly()
certifier = Certifier(abs_elem, transformer, net, neighbours)
certifier.flow()
print(certifier.abs_elem.d['l'][-1])
print(certifier.abs_elem.d['u'][-1])