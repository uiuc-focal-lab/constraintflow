#FEAT: fold inference BatchNormalization into its preceding Conv at load time (TinyImageNet/CIFAR100 ResNets).
# Shared verbatim by constraintflow-gpu, original constraintflow and auto_LiRPA's glue parser.
"""Inference Conv+BN folding: W' = s W, b' = (b - mean) s + beta, s = gamma / sqrt(var + eps)."""
from collections import Counter
import math
import torch
from onnx import helper


class _GraphView:
    """The graph fields the parsers read (node, input, output, initializer), with a new node list."""
    def __init__(self, graph, nodes):
        self.node, self.input, self.output = nodes, graph.input, graph.output
        self.initializer, self.name = graph.initializer, graph.name


class _ModelView:
    def __init__(self, net, nodes):
        self.graph = _GraphView(net.graph, nodes)


def fold_batch_norm(net, parameters):
    """(net, parameters) with every BatchNormalization folded into the Conv feeding it.
    Returns the inputs unchanged when the graph has no BN; never mutates them."""
    graph = net.graph
    if not any(n.op_type == 'BatchNormalization' for n in graph.node):
        return net, parameters
    parameters = dict(parameters)
    producers = {o: n for n in graph.node for o in n.output}
    consumers = Counter(i for n in graph.node for i in n.input)
    outputs = {o.name for o in graph.output}
    folded = {}  # id(original Conv) -> folded copy
    removed = set()
    for node in graph.node:
        if node.op_type != 'BatchNormalization':
            continue
        def fail(reason):
            raise ValueError(f"Cannot fold BatchNormalization {node.name!r}: {reason}")
        attrs = {a.name: helper.get_attribute_value(a) for a in node.attribute}
        if attrs.get('training_mode', 0) or len(node.output) != 1:
            fail('only inference mode with one output is supported')
        if len(node.input) != 5 or any(i not in parameters for i in node.input[1:]):
            fail('requires constant scale, bias, mean and variance')
        conv = producers.get(node.input[0])
        if conv is None or conv.op_type != 'Conv' or id(conv) in folded:
            fail('input must be an unfolded Conv output')
        if consumers[node.input[0]] != 1 or node.input[0] in outputs:
            fail('Conv output is shared or is a graph output')
        if any(i not in parameters for i in conv.input[1:]):
            fail('requires constant Conv weight and bias')
        weight = parameters[conv.input[1]]
        bias = parameters[conv.input[2]] if len(conv.input) == 3 else weight.new_zeros(weight.shape[0])
        gamma, beta, mean, var = (parameters[i] for i in node.input[1:])
        if any(p.shape != (weight.shape[0],) for p in (bias, gamma, beta, mean, var)):
            fail('parameter shapes do not match the Conv output channels')
        eps = attrs.get('epsilon', 1e-5)
        if not math.isfinite(eps) or eps < 0:
            fail('invalid epsilon')
        #FIX FOR: bool() on the variance check read GPU weights back to the host (one D2H per BN layer per batch).
        torch._assert_async(torch.all(var + eps > 0), f"Cannot fold BatchNormalization {node.name!r}: invalid variance")
        scale = gamma / torch.sqrt(var + eps)
        # Names keep 'weight'/'bias' in them: auto_LiRPA's glue parser finds Conv operands by substring.
        prefix = f'__folded_bn_{len(folded)}'
        parameters[prefix + '.weight'] = weight * scale.reshape(-1, *([1] * (weight.ndim - 1)))
        parameters[prefix + '.bias'] = (bias - mean) * scale + beta
        new_conv = type(conv)()
        new_conv.CopyFrom(conv)
        del new_conv.input[1:]
        new_conv.input.extend([prefix + '.weight', prefix + '.bias'])
        new_conv.output[0] = node.output[0]
        folded[id(conv)] = new_conv
        removed.add(id(node))
    nodes = [folded.get(id(n), n) for n in graph.node if id(n) not in removed]
    return _ModelView(net, nodes), parameters


#FEAT: 2x2/stride-2 MaxPool (e.g. VNN-COMP vggnet16) as Conv/Relu/Add, which every parser here supports.
# max(a, b) = b + relu(a - b); for the window [[a, b], [c, d]]:
#   [m1, m2] = relu([a - b, c - d]) + [b, d]   (2x2/stride-2 Convs "diff" and "keep", Relu, Add)
#   max      = relu(m1 - m2) + m2             (1x1 Convs "sub" and "second", Relu, Add)
# Exact; abstract domains bound the two Relus instead of applying a max-pool rule.
_SAME_CHANNELS = {'Relu', 'Identity', 'Dropout', 'BatchNormalization', 'Sigmoid', 'Tanh', 'LeakyRelu', 'Clip',
                  'Add', 'Sub', 'Mul', 'Div', 'MaxPool'}


def _pool_channels(name, producers, parameters, graph):
    """Channels of a 4-D activation: from the Conv that made it, or from the graph input."""
    node = producers.get(name)
    while node is not None and node.op_type in _SAME_CHANNELS:
        node = producers.get(node.input[0])
        name = node.output[0] if node is not None else name
    if node is not None and node.op_type == 'Conv' and node.input[1] in parameters:
        weight = parameters[node.input[1]]
        return weight.shape[0], weight
    for i in graph.input:
        dims = i.type.tensor_type.shape.dim
        if i.name not in parameters and len(dims) == 4 and dims[1].dim_value > 0:
            return dims[1].dim_value, None
    raise ValueError(f'Cannot expand MaxPool: channel count of {name!r} not found')


def _pool_conv(prefix, inp, out, weight, stride):
    """Conv node with attributes in the positional order the parsers read (dilations, group, kernel, pads, strides)."""
    node = helper.make_node('Conv', [inp, prefix + '.weight', prefix + '.bias'], [out], name=prefix)
    node.attribute.extend([helper.make_attribute('dilations', [1, 1]), helper.make_attribute('group', 1),
                           helper.make_attribute('kernel_shape', list(weight.shape[2:])),
                           helper.make_attribute('pads', [0, 0, 0, 0]), helper.make_attribute('strides', [stride, stride])])
    return node


def expand_maxpool(net, parameters):
    """(net, parameters) with every 2x2/stride-2 MaxPool as Conv/Relu/Add and every Dropout as Identity.
    Returns the inputs unchanged when the graph has neither; never mutates them."""
    graph = net.graph
    if not any(n.op_type in ('MaxPool', 'Dropout') for n in graph.node):
        return net, parameters
    parameters = dict(parameters)
    producers = {o: n for n in graph.node for o in n.output}
    nodes = []
    for k, node in enumerate(graph.node):
        if node.op_type == 'Dropout':
            nodes.append(helper.make_node('Identity', [node.input[0]], [node.output[0]], name=node.name))
            continue
        if node.op_type != 'MaxPool':
            nodes.append(node)
            continue
        attrs = {a.name: helper.get_attribute_value(a) for a in node.attribute}
        if (list(attrs.get('kernel_shape', [])) != [2, 2] or list(attrs.get('strides', [1, 1])) != [2, 2]
                or any(attrs.get('pads', [0] * 4)) or len(node.output) != 1):
            raise ValueError(f'MaxPool {node.name!r}: only 2x2, stride 2, no padding is supported')
        C, like = _pool_channels(node.input[0], producers, parameters, graph)
        kw = dict(dtype=like.dtype, device=like.device) if like is not None else {}
        diff, keep = torch.zeros(2 * C, C, 2, 2, **kw), torch.zeros(2 * C, C, 2, 2, **kw)
        sub, second = torch.zeros(C, 2 * C, 1, 1, **kw), torch.zeros(C, 2 * C, 1, 1, **kw)
        c = torch.arange(C)
        diff[c, c, 0, 0], diff[c, c, 0, 1] = 1, -1             # a - b
        diff[C + c, c, 1, 0], diff[C + c, c, 1, 1] = 1, -1     # c - d
        keep[c, c, 0, 1], keep[C + c, c, 1, 1] = 1, 1          # b, d
        sub[c, c], sub[c, C + c] = 1, -1                       # m1 - m2
        second[c, C + c] = 1                                   # m2
        # Names keep 'weight'/'bias' in them: the parsers find Conv operands by substring.
        p, x = f'__maxpool_{k}', node.input[0]
        for name, w, inp, stride in ((p + '_diff', diff, x, 2), (p + '_keep', keep, x, 2),
                                     (p + '_sub', sub, p + '_pairs', 1), (p + '_second', second, p + '_pairs', 1)):
            parameters[name + '.weight'], parameters[name + '.bias'] = w, torch.zeros(w.shape[0], **kw)
        nodes += [_pool_conv(p + '_diff', x, p + '_diff', diff, 2), _pool_conv(p + '_keep', x, p + '_keep', keep, 2),
                  helper.make_node('Relu', [p + '_diff'], [p + '_relu1'], name=p + '_relu1'),
                  helper.make_node('Add', [p + '_relu1', p + '_keep'], [p + '_pairs'], name=p + '_pairs'),
                  _pool_conv(p + '_sub', p + '_pairs', p + '_sub', sub, 1),
                  _pool_conv(p + '_second', p + '_pairs', p + '_second', second, 1),
                  helper.make_node('Relu', [p + '_sub'], [p + '_relu2'], name=p + '_relu2'),
                  helper.make_node('Add', [p + '_relu2', p + '_second'], [node.output[0]], name=p + '_max')]
    return _ModelView(net, nodes), parameters
