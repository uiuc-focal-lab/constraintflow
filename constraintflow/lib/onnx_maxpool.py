#FEAT: run networks with 2x2/stride-2 MaxPool (e.g. VNN-COMP vggnet16) using only Conv, Relu and Add.
"""Rewrite each MaxPool(2x2, stride 2, no padding) as max(a, b) = b + relu(a - b), and Dropout as Identity.

For the window [[a, b], [c, d]]:
    [m1, m2] = relu([a - b, c - d]) + [b, d]     (two 2x2/stride-2 Convs, Relu, Add)
    max      = relu(m1 - m2) + m2                (two 1x1 Convs, Relu, Add)
Exact in real arithmetic; abstract domains bound the two Relus instead of applying a max-pool rule.

    python -m constraintflow.lib.onnx_maxpool in.onnx out.onnx
"""
import sys

import numpy as np
import onnx
from onnx import helper, numpy_helper, shape_inference


def _conv(name, inp, out, weight, stride):
    inits = [numpy_helper.from_array(weight, name + '.weight'),
             numpy_helper.from_array(np.zeros(weight.shape[0], np.float32), name + '.bias')]
    # parse.py reads Conv attributes by position: dilations, group, kernel_shape, pads, strides.
    node = helper.make_node('Conv', [inp, name + '.weight', name + '.bias'], [out], name=name)
    node.attribute.extend([helper.make_attribute('dilations', [1, 1]), helper.make_attribute('group', 1),
                           helper.make_attribute('kernel_shape', list(weight.shape[2:])),
                           helper.make_attribute('pads', [0, 0, 0, 0]), helper.make_attribute('strides', [stride, stride])])
    return node, inits


def rewrite(model):
    """New ModelProto without MaxPool or Dropout nodes."""
    model = shape_inference.infer_shapes(model)
    channels = {v.name: v.type.tensor_type.shape.dim[1].dim_value
                for v in list(model.graph.value_info) + list(model.graph.input) if len(v.type.tensor_type.shape.dim) == 4}
    nodes, inits = [], []
    for k, n in enumerate(model.graph.node):
        if n.op_type == 'Dropout':
            nodes.append(helper.make_node('Identity', [n.input[0]], [n.output[0]], name=n.name))
            continue
        if n.op_type != 'MaxPool':
            nodes.append(n)
            continue
        attrs = {a.name: helper.get_attribute_value(a) for a in n.attribute}
        if (list(attrs.get('kernel_shape', [])) != [2, 2] or list(attrs.get('strides', [1, 1])) != [2, 2]
                or any(attrs.get('pads', [0] * 4)) or len(n.output) != 1):
            raise ValueError(f'MaxPool {n.name!r}: only 2x2, stride 2, no padding is supported')
        C, x, p = channels[n.input[0]], n.input[0], f'__maxpool{k}'
        diff, keep = np.zeros((2 * C, C, 2, 2), np.float32), np.zeros((2 * C, C, 2, 2), np.float32)
        sub, second = np.zeros((C, 2 * C, 1, 1), np.float32), np.zeros((C, 2 * C, 1, 1), np.float32)
        for c in range(C):
            diff[c, c, 0, 0], diff[c, c, 0, 1] = 1, -1           # a - b
            diff[C + c, c, 1, 0], diff[C + c, c, 1, 1] = 1, -1   # c - d
            keep[c, c, 0, 1], keep[C + c, c, 1, 1] = 1, 1        # b, d
            sub[c, c], sub[c, C + c] = 1, -1                     # m1 - m2
            second[c, C + c] = 1                                 # m2
        for node_inits in (_conv(p + '_diff', x, p + '_diff', diff, 2), _conv(p + '_keep', x, p + '_keep', keep, 2)):
            nodes.append(node_inits[0])
            inits += node_inits[1]
        nodes += [helper.make_node('Relu', [p + '_diff'], [p + '_relu1'], name=p + '_relu1'),
                  helper.make_node('Add', [p + '_relu1', p + '_keep'], [p + '_pairs'], name=p + '_pairs')]
        for node_inits in (_conv(p + '_sub', p + '_pairs', p + '_sub', sub, 1),
                           _conv(p + '_second', p + '_pairs', p + '_second', second, 1)):
            nodes.append(node_inits[0])
            inits += node_inits[1]
        nodes += [helper.make_node('Relu', [p + '_sub'], [p + '_relu2'], name=p + '_relu2'),
                  helper.make_node('Add', [p + '_relu2', p + '_second'], [n.output[0]], name=p + '_max')]
    graph = helper.make_graph(nodes, model.graph.name, model.graph.input, model.graph.output,
                              list(model.graph.initializer) + inits)
    out = helper.make_model(graph, opset_imports=model.opset_import, producer_name='onnx_maxpool')
    onnx.checker.check_model(out)
    return out


if __name__ == '__main__':
    onnx.save(rewrite(onnx.load(sys.argv[1])), sys.argv[2])
