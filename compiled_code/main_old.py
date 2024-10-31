from specs.spec import *
from certifier import Certifier
from lib.abs_elem import Abs_elem
# from lib.transformer import *
from specs.network import LayerType
from newtransformer import *
import copy
from compiled_code.lib.utils import *

import matplotlib.pyplot as plt
import itertools
import pickle

net = get_net(net_name='nets/ACASXU_run2a_1_1_batch_2000.onnx')
# net = get_net(net_name='nets/mnist_relu_3_50.onnx')
# net = get_net(net_name='nets/mnist_0.1.onnx')

neighbours = dict()
shapes = [net.input_shape]
for layer in net:
    shapes.append(layer.shape)

size = 0
for i in range(len(shapes)):
    size += compute_size(shapes[i])

t = torch.zeros(size)
u = torch.full((size,), float('inf'))
l = torch.full((size,), float('-inf'))
t[0:5] = 1
l[0:5] = torch.tensor([0.6000, -0.5000, -0.5000,  0.4500, -0.5000])
u[0:5] = torch.tensor([ 0.6799,  0.5000,  0.5000,  0.5000, -0.4500])

coeff = torch.zeros((size, size))
L = PolyExp(size, size, copy.deepcopy(coeff), copy.deepcopy(l))
U = PolyExp(size, size, copy.deepcopy(coeff), copy.deepcopy(u))

const = copy.deepcopy((l + u) / 2)
coeff_5 = copy.deepcopy(torch.eye(l[0:5].shape[0]) * (u[0:5] - l[0:5])/2)
coeff = torch.zeros(610, 5)
coeff[:5, :] = coeff_5
Z = SymExp(l.shape[0], 5, coeff, const, 0, 4)
print(Z.mat.shape)
# with open('acas_specs/acasxu_prop_1_input_prenormalized.txt', 'r') as f:
#     lines = f.readlines()
#     for i, line in enumerate(lines):
#         if i==5:
#             break
#         l[i] = float(line[1:-2].split(',')[0])
#         u[i] = float(line[1:-2].split(',')[1])


# t, l, u = get_input_spec(shapes=shapes, n=0, transformer='ibp', eps=0.01)
# t, l, u, L, U = get_input_spec(shapes=shapes, n=0, transformer='deeppoly', eps=0.01)
# t_, l_, u_, Z = get_input_spec(shapes=shapes, n=0, transformer='deepz', eps=0.01)
# print(l.shape)
# print(l[:784].sum())
# print(u[:784].sum())
# kjsd
temp_save = [t, l, u, L, U, Z]
file_path = 'specs_acas1_polyzono.pkl'
with open(file_path, 'wb') as file:
    pickle.dump(temp_save, file)

jdfg
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


# transformer = CflowNewInterval()
transformer = CflowNewDeepPolyCompiler()

certifier = Certifier(abs_elem, transformer, net, neighbours)
certifier.flow()