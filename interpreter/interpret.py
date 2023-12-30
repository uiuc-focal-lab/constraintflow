from common.astinterpreter import *

import sys

import antlr4 as antlr

import astcf as AST
import dslLexer
import dslParser
import astBuilder
import astTC

from specs.spec import *
from certifier import Certifier
from common.abs_elem import Abs_elem
from common.transformer import *
from specs.network import LayerType

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

shapes = [net.input_shape]
for layer in net:
    shapes.append(layer.shape)

neighbours = dict()
for idx in itertools.product(*[range(dim) for dim in shapes[0]]):
    neighbours[(0, idx)] = []
for i in range(1, len(shapes)):
    for idx in itertools.product(*[range(dim) for dim in shapes[i]]):
        if i%2==0:
            neighbours[(i, idx)] = [(i-1, idx)]
        else:
            prev_indices = itertools.product(*[range(dim) for dim in shapes[i-1]])
            neighbours[(i, idx)] = [(i-1, j) for j in prev_indices]

l, u, L, U = get_input_spec(shapes=shapes, n=0, transformer='deeppoly', eps=0.02)
abs_elem = Abs_elem({'l': l, 'u': u, 'L': L, 'U': U}, {'l': 'float', 'u': 'float', 'L': 'PolyExp', 'U': 'PolyExp'}, shapes)

# l, u = get_input_spec(shapes=shapes, n=0, transformer='ibp', eps=2.0)
# abs_elem = Abs_elem({'l': l, 'u': u}, {'l': 'float', 'u': 'float'}, shapes)

# l, u, Z= get_input_spec(shapes=shapes, n=1, transformer='deepz', eps=2.0)
# abs_elem = Abs_elem({'l': l, 'u': u, 'Z': Z}, {'l': 'float', 'u': 'float', 'Z': 'SymExp'}, shapes)


def genAST(inputfile):
    lexer = dslLexer.dslLexer(antlr.FileStream(inputfile))
    tokens = antlr.CommonTokenStream(lexer)
    parser = dslParser.dslParser(tokens)
    tree = parser.prog()
    ast = astBuilder.ASTBuilder().visit(tree)

    x = astTC.ASTTC().visit(ast)

    newtrans = AstInterpret("newtransformer.py")
    newtrans.visit(ast)


genAST(sys.argv[1])

from newtransformer import *

transformer = Cflowdeeppoly()
certifier = Certifier(abs_elem, transformer, net, neighbours)
certifier.flow()
print(certifier.abs_elem.d['l'][-1])
print(certifier.abs_elem.d['u'][-1])