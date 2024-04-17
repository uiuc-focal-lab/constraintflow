from specs.spec import *
from certifier import Certifier
from common.abs_elem import Abs_elem
from common.transformer import *
from specs.network import LayerType
from newtransformer import *

import matplotlib.pyplot as plt
import itertools
import pickle

def compute_size(shape):
    s = 1
    while len(shape)>0:
        s *= shape[0]
        shape = shape[1:]
    return s

# net = get_net(net_name='nets/mnist_relu_3_50.onnx')
net = get_net(net_name='nets/mnist_relu_3_50.onnx')
# net = get_net(net_name='nets/mnist_0.1.onnx')

neighbours = dict()
shapes = [net.input_shape]
for layer in net:
    shapes.append(layer.shape)


t, l, u, L, U = get_input_spec(shapes=shapes, n=0, transformer='deeppoly', eps=0.01)
temp_save = [t, l, u, L, U]
# file_path = 'specs_eps001.pkl'
# with open(file_path, 'wb') as file:
#     pickle.dump(temp_save, file)

# jdfg
# t, l, u, Z = get_input_spec(shapes=shapes, n=0, transformer='deepz', eps=0.0)

# abs_elem = Abs_elem({'t': t, 'l': l, 'u': u}, {'t': 'bool', 'l': 'float', 'u': 'float'}, shapes)
abs_elem = Abs_elem({'t': t, 'l': l, 'u': u, 'L': L, 'U': U}, {'t': 'bool', 'l': 'float', 'u': 'float', 'L': 'PolyExp', 'U': 'PolyExp'}, shapes)
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


transformer = CflowNewInterval()
# transformer = CflowNewDeepPolyCompiler()

certifier = Certifier(abs_elem, transformer, net, neighbours)
certifier.flow()