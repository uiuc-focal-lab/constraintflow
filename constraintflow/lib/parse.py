import torch
import math
import onnx
import numpy as np
import torch.nn as nn
import copy
import os

from onnx import numpy_helper
from constraintflow.lib.globals import device_mode
from constraintflow.lib.network import Layer, LayerType, Network
#FEAT: inference BatchNormalization is folded into its Conv at load time.
from constraintflow.lib.onnx_preprocess import expand_maxpool, fold_batch_norm

from collections import deque

# This would be cheating hence I have removed this. 
# _cuda_initializer_cache_key = None
# _cuda_initializer_cache = None
# def _initializer_tensors(net, net_name=None):
#     """Materialize ONNX initializers, caching one network on the active CUDA device."""
#     global _cuda_initializer_cache_key, _cuda_initializer_cache

#     if device_mode.get_device() != 'cuda' or net_name is None:
#         return {
#             init_vals.name: torch.tensor(numpy_helper.to_array(init_vals))
#             for init_vals in net.graph.initializer
#         }

#     stat = os.stat(net_name)
#     cache_key = (
#         os.path.realpath(net_name),
#         stat.st_mtime_ns,
#         stat.st_size,
#         torch.cuda.current_device(),
#     )
#     if cache_key != _cuda_initializer_cache_key:
#         _cuda_initializer_cache = {
#             init_vals.name: torch.tensor(
#                 numpy_helper.to_array(init_vals),
#                 device=device_mode.get_device(),
#             )
#             for init_vals in net.graph.initializer
#         }
#         _cuda_initializer_cache_key = cache_key
#     return _cuda_initializer_cache

#FEAT: weight cast for --dtype float64.
def _default_float(t):
    """Float weights in the run's dtype (fp32 weights, or exact fp64 copies under --dtype float64)."""
    return t.to(torch.get_default_dtype()) if t.is_floating_point() else t


def _initializer_tensors(net, net_name=None):
    target_device = (
        device_mode.get_device()
        if device_mode.get_device() == "cuda" and net_name is not None
        else "cpu"
    )
    return {
        #FEAT: float weights follow the run dtype.
        init_vals.name: _default_float(torch.tensor(
            numpy_helper.to_array(init_vals),
            device=target_device,
        ))
        for init_vals in net.graph.initializer
    }




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

def get_net(net_name, spec_weight, spec_bias, no_sparsity, initializers=None, fold_spec=False):
    net_format = get_net_format(net_name)
    if net_format == 'onnx':
        net_onnx = onnx.load(net_name)
        model_name_to_val_dict = _initializer_tensors(net_onnx, net_name)
        values = None
        # net type: constraintflow.lib.network.Network (inherits list)
        # net element type: constraintflow.lib.network.Layer
        if initializers is not None:
            unknown = set(initializers) - set(model_name_to_val_dict)
            if unknown:
                raise ValueError(f"Unknown ONNX initializer names: {sorted(unknown)}")
            for name, tensor in initializers.items():
                if not isinstance(tensor, torch.Tensor):
                    raise TypeError(f"Initializer {name!r} must be a torch.Tensor")
                if tensor.shape != model_name_to_val_dict[name].shape:
                    raise ValueError(f"Initializer {name!r} has incorrect shape: {tensor.shape}")
                if tensor.dtype != model_name_to_val_dict[name].dtype:
                    raise ValueError(f"Initializer {name!r} must retain its ONNX dtype")
                if tensor.device != spec_weight.device:
                    raise ValueError(f"Initializer {name!r} and inputs must share a device")
                # Preserve the live Parameter (or differentiable view) itself.
                model_name_to_val_dict[name] = tensor
        net = parse_onnx_layers(
            net_onnx,
            spec_weight,
            spec_bias,
            no_sparsity,
            model_name_to_val_dict=model_name_to_val_dict,
            values=values,
            fold_spec=fold_spec,
        )
        #FEAT: training fast path, see record_layer_sources.
        if initializers is not None:
            record_layer_sources(net, model_name_to_val_dict)
    else:
        raise ValueError("Unsupported net format!")

    net.net_name = net_name

    print('Network loaded')
    return net

#FEAT: training fast path (constraintflow/lib/training.py): a parsed network is reused across
# steps, and refresh_layers re-derives each layer's tensors from the current parameters and spec.
def record_layer_sources(layers, values):
    """Name the ONNX initializer each layer's weight/bias came from (by identity, before any
    transform parse applied). A layer whose gradient-carrying tensor has no source is not refreshable."""
    name_of = {id(t): name for name, t in values.items()}
    for layer in layers:
        layer.sources = {}
        layer.refreshable = True
        for attr in ('weight', 'bias'):
            raw = getattr(layer, attr + '_raw', getattr(layer, attr, None))
            if not isinstance(raw, torch.Tensor) or getattr(layer, 'is_spec_layer', False):
                continue
            name = name_of.get(id(raw))
            if name is not None:
                layer.sources[attr] = name
            elif raw.requires_grad:
                layer.refreshable = False


def refresh_layers(layers, values, spec_weight, spec_bias):
    """Recompute every layer tensor parse derived from `values` (the ONNX initializers) and the spec."""
    for layer in layers:
        if getattr(layer, 'is_spec_layer', False):
            layer.weight, layer.bias = spec_weight, spec_bias
            continue
        src = layer.sources
        weight = values[src['weight']] if 'weight' in src else None
        bias = values[src['bias']] if 'bias' in src else None
        if getattr(layer, 'folded_spec', False):
            #FIX FOR: a MatMul weight is a transposed copy with no ONNX source; it is constant, so fold the parsed one.
            weight = layer.weight_raw if weight is None else weight
            bias = layer.bias_raw if bias is None else bias
            layer.spec_offset = (spec_weight @ bias.unsqueeze(-1)).squeeze(-1) + spec_bias
            layer.weight = spec_weight @ weight
            #FEAT: spec_bias may be per sample [batch, rows] (one unsafe condition per input box); the compiled bias stays [rows].
            layer.bias = spec_bias.new_zeros(spec_bias.shape[-1:])
            continue
        if weight is not None:
            layer.weight = weight
        if bias is not None:
            if layer.type == LayerType.Conv2D:
                bias = bias.unsqueeze(1).repeat(1, layer.shape[2] * layer.shape[3]).flatten()
            layer.bias = bias


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


class _LazyArrays(dict):
    """Initializer arrays for shape folding, converted on first lookup; weights are never read."""

    def __init__(self, initializers):
        super().__init__()
        self._protos = {init.name: init for init in initializers}

    def __contains__(self, name):
        return dict.__contains__(self, name) or name in self._protos

    def __missing__(self, name):
        value = numpy_helper.to_array(self._protos[name])
        self[name] = value
        return value

    def get(self, name, default=None):
        return self[name] if name in self else default


def _shape_value(node, values, shapes):
    op = node.op_type
    attrs = {a.name: onnx.helper.get_attribute_value(a) for a in node.attribute}
    if op == 'Constant':
        if 'value' in attrs:
            return numpy_helper.to_array(attrs['value'])
        for key in ('value_int', 'value_ints', 'value_float', 'value_floats'):
            if key in attrs:
                return np.asarray(attrs[key])
        raise ValueError(f"Unsupported Constant {node.name!r}")
    if op == 'Shape':
        shape = shapes.get(node.input[0])
        if shape is None:
            raise ValueError(f"Shape {node.name!r} has an unresolved input")
        return np.asarray(shape[attrs.get('start', 0):attrs.get('end')], dtype=np.int64)
    #FEAT: constant subgraphs fold too (lsnc_relu: P = eps*I + R^T R, and the controller run on the fixed
    # equilibrium x*, which needs MatMul, Gemm and Relu of constants).
    if op not in ('Gather', 'Unsqueeze', 'Squeeze', 'Concat', 'Mul', 'Add', 'Sub', 'Div', 'Cast', 'MatMul', 'Gemm', 'Relu'):
        return None
    if not all(name in values for name in node.input):
        #FEAT: Gather/Cast of activations go on to _lower_vector_op (which raises for anything but [1, n]).
        if op in ('Unsqueeze', 'Squeeze'):
            raise ValueError(f"Unsupported activation operation {op} {node.name!r}")
        return None
    args = [values[name] for name in node.input]
    if op == 'Gather':
        return np.take(args[0], args[1], axis=attrs.get('axis', 0))
    if op in ('Unsqueeze', 'Squeeze'):
        axes = args[1].tolist() if len(args) > 1 else attrs.get('axes')
        axes = tuple(axes) if axes is not None else None
        return np.expand_dims(args[0], axes) if op == 'Unsqueeze' else np.squeeze(args[0], axes)
    if op == 'Concat':
        return np.concatenate(args, axis=attrs['axis'])
    if op == 'Cast':
        return args[0].astype(onnx.helper.tensor_dtype_to_np_dtype(attrs['to']))
    if op == 'Div':
        result = np.divide(*args)
        return np.trunc(result).astype(args[0].dtype) if np.issubdtype(args[0].dtype, np.integer) else result
    if op == 'Relu':
        return np.maximum(args[0], 0)
    if op == 'Gemm':
        a = args[0].T if attrs.get('transA', 0) else args[0]
        b = args[1].T if attrs.get('transB', 0) else args[1]
        res = attrs.get('alpha', 1.0) * (a @ b)
        return res + attrs.get('beta', 1.0) * args[2] if len(args) > 2 else res
    return {'Mul': np.multiply, 'Add': np.add, 'Sub': np.subtract, 'MatMul': np.matmul}[op](*args)


#FEAT: vector graphs (every activation [1, n]; e.g. lsnc_relu = controller + learned dynamics + quadratic Lyapunov
# form). A linear op on one activation becomes a Linear layer with a constant weight (no ONNX source, so
# refresh_layers keeps it); a product of two activations becomes a Mul layer (the program's Neuron_mult rule).
# Only patterns the existing cases rejected or silently skipped are taken; returns the new layer index, else None.
def _lower_vector_op(node, names_hash, tensor_shapes, values, initializers, layers, index, consumers):
    op, ins = node.op_type, list(node.input)
    attrs = {a.name: onnx.helper.get_attribute_value(a) for a in node.attribute}
    tensors = [t for t in initializers.values() if isinstance(t, torch.Tensor)]
    device = tensors[0].device if tensors else 'cpu'
    act = lambda name: name in names_hash
    const = lambda name: None if act(name) or not name or name not in values else np.asarray(values[name])
    #FIX FOR: ACAS Xu's Sub(input [1, 1, 1, 5], zero offset) was taken here and raised; only vector graphs
    # are lowered, anything else is left to the existing cases unchanged.
    if not all(len(tensor_shapes[name]) == 2 for name in ins if act(name)):
        return None

    def width(name):
        shape = tensor_shapes[name]
        if len(shape) != 2 or shape[0] != 1:
            raise ValueError(f"{op} {node.name!r}: vector ops need [1, n] activations, got {shape}")
        return shape[1]

    def emit(layer_type, parents, size, weight=None, bias=None):
        nonlocal index
        index += 1
        layer = Layer(weight=weight, bias=bias, type=layer_type, identifier=index, parents=parents)
        layer.shape, layer.size = [1, size, 1, 1], size
        layer.start = layers.size
        layers.size += size
        layer.end = layers.size
        layers.append(layer)
        return index

    def linear(src, weight, bias=None):  # weight [out, in]
        weight = torch.tensor(np.asarray(weight, dtype=np.float64), dtype=torch.get_default_dtype(), device=device)
        out = weight.shape[0]
        bias = np.zeros(out) if bias is None else np.broadcast_to(np.asarray(bias, dtype=np.float64).reshape(-1), (out,))
        bias = torch.tensor(np.array(bias), dtype=torch.get_default_dtype(), device=device)
        return emit(LayerType.Linear, [names_hash[src]], out, weight, bias)

    def done(layer_index, size):
        names_hash[node.output[0]] = layer_index
        tensor_shapes[node.output[0]] = [1, size]
        return index

    def diagonal(src, scale):
        n = width(src)
        scale = np.asarray(scale, dtype=np.float64).reshape(-1)
        if scale.size not in (1, n):
            raise ValueError(f"{op} {node.name!r}: constant of size {scale.size} for {n} activations")
        return done(linear(src, np.diag(np.broadcast_to(scale, (n,)))), n)

    if op == 'Cast' and act(ins[0]):
        if attrs.get('to') != onnx.TensorProto.FLOAT:
            raise ValueError(f"Cast {node.name!r} to {attrs.get('to')} is not supported")
        names_hash[node.output[0]] = names_hash[ins[0]]
        tensor_shapes[node.output[0]] = tensor_shapes[ins[0]]
        return index
    if op == 'Neg' and act(ins[0]):
        return diagonal(ins[0], -1.0)
    if op == 'MatMul' and act(ins[0]) and const(ins[1]) is not None and ins[1] not in initializers:
        weight = const(ins[1])  # computed in the graph; the existing case takes initializer weights
        if weight.ndim != 2:
            raise ValueError(f"MatMul {node.name!r} needs a 2-D constant, got {weight.shape}")
        return done(linear(ins[0], weight.T), weight.shape[1])
    if op in ('Add', 'Sub'):
        a, b = ins
        if act(a) and act(b):
            if op == 'Add':
                return None  # the existing residual Add
            n = width(a)
            return done(emit(LayerType.Add, [names_hash[a], linear(b, -np.eye(n))], n), n)
        if act(a) and const(b) is not None:
            if op == 'Add' and b in initializers:
                return None  # the existing case: a bias folded into the producing layer
            n = width(a)
            return done(linear(a, np.eye(n), const(b) if op == 'Add' else -const(b)), n)
        if const(a) is not None and act(b):
            n = width(b)
            return done(linear(b, np.eye(n) if op == 'Add' else -np.eye(n), const(a)), n)
        return None
    if op in ('Mul', 'Div'):
        a, b = ins
        if act(a) and act(b):
            if op == 'Div':
                raise ValueError(f"Div {node.name!r} of two activations is not supported")
            n = width(a)
            if width(b) != n:
                raise ValueError(f"Mul {node.name!r}: broadcasting {tensor_shapes[a]} x {tensor_shapes[b]} is not supported")
            return done(emit(LayerType.Mul, [names_hash[a], names_hash[b]], n), n)
        if act(a) and const(b) is not None:
            return diagonal(a, const(b) if op == 'Mul' else 1.0 / const(b))
        if op == 'Mul' and const(a) is not None and act(b):
            return diagonal(b, const(a))
        if op == 'Div' and const(a) is not None and act(b):
            raise ValueError(f"Div {node.name!r} of a constant by an activation is not supported")
        return None
    if op == 'Slice' and act(ins[0]):
        n = width(ins[0])
        starts, ends = const(ins[1]).reshape(-1), const(ins[2]).reshape(-1)
        axes = const(ins[3]).reshape(-1) if len(ins) > 3 and ins[3] else np.arange(len(starts))
        steps = const(ins[4]).reshape(-1) if len(ins) > 4 and ins[4] else np.ones(len(starts), dtype=np.int64)
        if len(starts) != 1 or int(axes[0]) not in (1, -1) or int(steps[0]) != 1:
            raise ValueError(f"Slice {node.name!r}: only unit-step slices along the last axis are supported")
        start, end = (int(v) + n if int(v) < 0 else min(int(v), n) for v in (starts[0], ends[0]))
        return done(linear(ins[0], np.eye(n)[start:end]), end - start)
    if op == 'Gather' and act(ins[0]):
        n = width(ins[0])
        idx = const(ins[1])
        if idx is None or idx.ndim != 1 or attrs.get('axis', 0) not in (1, -1):
            raise ValueError(f"Gather {node.name!r}: only a 1-D index list along the last axis is supported")
        return done(linear(ins[0], np.eye(n)[[int(i) % n for i in idx]]), len(idx))
    if op == 'ReduceSum' and act(ins[0]):
        n = width(ins[0])
        axes = attrs['axes'] if 'axes' in attrs else const(ins[1]).reshape(-1).tolist()
        if [int(a) for a in axes] not in ([1], [-1]) or not attrs.get('keepdims', 1):
            raise ValueError(f"ReduceSum {node.name!r}: only a sum over the last axis with keepdims is supported")
        return done(linear(ins[0], np.ones((1, n))), 1)
    if (op == 'Concat' and all(act(name) for name in ins)
            and all(len(tensor_shapes[name]) == 2 for name in ins)):
        # The Concat layer serves conv channels and fails on vectors (a Concat feeding a Relu or an Add);
        # concat(a, b, ...) = E_a a + E_b b + ..., with E_i placing input i in its slice, is exact and uses
        # only Linear and Add.
        if attrs['axis'] not in (1, -1):
            raise ValueError(f"Concat {node.name!r}: only the last axis is supported")
        widths = [width(name) for name in ins]
        total, offset, layer_index = sum(widths), 0, None
        for name, w in zip(ins, widths):
            place = np.zeros((total, w))
            place[offset:offset + w] = np.eye(w)
            offset += w
            placed = linear(name, place)
            layer_index = placed if layer_index is None else emit(LayerType.Add, [layer_index, placed], total)
        return done(layer_index, total)
    return None


#FEAT: collapse affine regions of vector graphs (lsnc_relu: 41 Linear + 12 Add, mostly lowered cheap ops).
def _collapse_affine(layers):
    """Rewrite every Linear/Add value whose bounds a nonlinearity reads (or the output) as sum_s A_s s + b over its
    nonlinear sources s, one Linear per source plus Adds; the affine layers in between go. Back-substitution
    through affine layers is exact, so only rounding changes. Returns `layers` unchanged when not applicable."""
    affine = (LayerType.Linear, LayerType.Add)
    if (any(l.type not in affine + (LayerType.Input, LayerType.ReLU, LayerType.Sigmoid, LayerType.Tanh, LayerType.Mul) for l in layers)
            or any(l.type == LayerType.Linear and (l.weight.dim() != 2 or l.weight.requires_grad
                                                   or (l.bias is not None and l.bias.requires_grad)) for l in layers)):
        return layers
    children = {l.identifier: [] for l in layers}
    for l in layers:
        for p in l.parents:
            children[p].append(l)
    needed = lambda l: not children[l.identifier] or any(c.type not in affine for c in children[l.identifier])
    if sum(l.type in affine for l in layers) == sum(l.type in affine and needed(l) for l in layers):
        return layers
    weight = next(l.weight for l in layers if l.type == LayerType.Linear)
    dtype, device = weight.dtype, weight.device
    rep = {}  # identifier -> ({source identifier: A [size, source size]}, b [size]), in float64
    out = layers.similar()
    new_id = {}

    def emit(layer_type, parents, size, weight=None, bias=None):
        layer = Layer(weight=weight, bias=bias, type=layer_type, identifier=len(out), parents=parents)
        layer.shape, layer.size = [1, size, 1, 1], size
        out.append(layer)
        return layer.identifier

    for l in layers:
        old = l.identifier
        if l.type == LayerType.Linear:
            terms, b = rep[l.parents[0]]
            w = l.weight.to(torch.float64)
            bias = l.bias.to(torch.float64) if l.bias is not None else 0
            rep[old] = ({s: w @ a for s, a in terms.items()}, w @ b + bias)
        elif l.type == LayerType.Add:
            (t0, b0), (t1, b1) = rep[l.parents[0]], rep[l.parents[1]]
            terms = dict(t0)
            for s, a in t1.items():
                terms[s] = terms[s] + a if s in terms else a
            rep[old] = (terms, b0 + b1)
        else:  # a source: kept as is
            l.identifier, l.parents = len(out), [new_id[p] for p in l.parents]
            out.append(l)
            new_id[old] = l.identifier
            rep[old] = ({old: torch.eye(l.size, dtype=torch.float64, device=device)},
                        torch.zeros(l.size, dtype=torch.float64, device=device))
            continue
        if not needed(l):
            continue
        terms, b = rep[old]
        last = None
        for s in sorted(terms):
            placed = emit(LayerType.Linear, [new_id[s]], b.shape[0], terms[s].to(dtype),
                          (b if last is None else torch.zeros_like(b)).to(dtype))
            last = placed if last is None else emit(LayerType.Add, [last, placed], b.shape[0])
        new_id[old] = last
    out.num_layers = len(out) - 1
    out.size = sum(l.size for l in out)
    return out


def _reshape_shape(node, source, values):
    target = values.get(node.input[1])
    if target is None or target.ndim != 1 or not np.issubdtype(target.dtype, np.integer):
        raise ValueError(f"Reshape {node.name!r} requires a resolved integer shape")
    attrs = {a.name: onnx.helper.get_attribute_value(a) for a in node.attribute}
    shape = target.tolist()
    if not attrs.get('allowzero', 0):
        for i, dim in enumerate(shape):
            if dim == 0:
                if i >= len(source):
                    raise ValueError(f"Reshape {node.name!r} has an invalid zero dimension")
                shape[i] = source[i]
    size = math.prod(source)
    if shape.count(-1) > 1 or any(dim < -1 or dim == 0 for dim in shape):
        raise ValueError(f"Reshape {node.name!r} has invalid dimensions {shape}")
    if -1 in shape:
        known = -math.prod(shape)
        if size % known:
            raise ValueError(f"Reshape {node.name!r} changes the neuron count")
        shape[shape.index(-1)] = size // known
    if math.prod(shape) != size or len(shape) not in (2, 4) or shape[0] != 1:
        raise ValueError(f"Unsupported Reshape {node.name!r}: {source} -> {shape}")
    return shape


def parse_onnx_layers(
    net,
    spec_weight,
    spec_bias,
    no_sparsity,
    model_name_to_val_dict=None,
    values=None,
    fold_spec=False,
):
    #FIX FOR: older exporters (VNN-COMP ACAS Xu) list initializers as graph inputs, so input[0] was a weight, not the data input.
    init_names = {i.name for i in net.graph.initializer}
    data_input = next(i for i in net.graph.input if i.name not in init_names)
    input_shape = [dim.dim_value for dim in data_input.type.tensor_type.shape.dim]
    input_shape = [1 if i == 0 else i for i in input_shape]
    if len(input_shape)==3:
        input_shape = [1] + input_shape
    input_size = compute_size(input_shape)
    # Create the new Network object
    layers = Network(input_name=data_input.name, input_shape=input_shape, input_size=input_size, input_start=0, input_end=input_size, net_format='onnx', no_sparsity=no_sparsity)
    if model_name_to_val_dict is None:
        model_name_to_val_dict = _initializer_tensors(net)
    #FEAT: fold BN before the layer walk (the identity fallback below used to drop it silently).
    net, model_name_to_val_dict = fold_batch_norm(net, model_name_to_val_dict)
    #FEAT: 2x2/stride-2 MaxPool as Conv/Relu/Add, Dropout as Identity (VNN-COMP vggnet16).
    net, model_name_to_val_dict = expand_maxpool(net, model_name_to_val_dict)
    num_layers = len(net.graph.node)
    layers.num_layers = num_layers

    if values is None:
        values = _LazyArrays(net.graph.initializer)
    else:
        values = dict(values)
    tensor_shapes = {data_input.name: input_shape}
    layers.size = input_size
    shape = input_shape

    names_hash = dict()
    index = 0
    names_hash[data_input.name] = index
    parents = dict()
    
    layer = Layer(type=LayerType.Input, shape=input_shape, size=input_size, start=0, end=input_size)
    layers.append(layer)
    consumers = {}
    for n in net.graph.node:
        for name in n.input:
            consumers[name] = consumers.get(name, 0) + 1
    for cur_layer in range(num_layers):
        node = net.graph.node[cur_layer]
        operation = node.op_type
        nd_inps = node.input
        value = _shape_value(node, values, tensor_shapes)
        if value is not None:
            values[node.output[0]] = value
            tensor_shapes[node.output[0]] = list(value.shape)
            continue
        if operation in ('Reshape', 'Flatten', 'Identity'):
            source = tensor_shapes[nd_inps[0]]
            result_shape = source
            if operation == 'Reshape':
                result_shape = _reshape_shape(node, source, values)
            elif operation == 'Flatten':
                attrs = {a.name: onnx.helper.get_attribute_value(a) for a in node.attribute}
                axis = attrs.get('axis', 1)
                if axis < 0:
                    axis += len(source)
                if axis != 1:
                    raise ValueError(f"Unsupported Flatten axis {axis}")
                result_shape = [source[0], math.prod(source[1:])]
            names_hash[node.output[0]] = names_hash[nd_inps[0]]
            tensor_shapes[node.output[0]] = result_shape
            continue
        #FEAT: vector graphs (activations [1, n], e.g. lsnc_relu): see _lower_vector_op.
        lowered = _lower_vector_op(node, names_hash, tensor_shapes, values, model_name_to_val_dict, layers, index, consumers)
        if lowered is not None:
            index = lowered
            continue
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
                    weight = model_name_to_val_dict[w_key]
                    bias = torch.zeros(
                        weight.shape[0],
                        device=weight.device,
                        dtype=weight.dtype,
                    )
                    layer = Layer(weight=weight, bias=bias, type=LayerType.Conv2D, identifier=index, parents=parents[index])
                else:
                    layer = Layer(weight=model_name_to_val_dict[w_key], bias=model_name_to_val_dict[b_key], type=LayerType.Conv2D, identifier=index, parents=parents[index])

            else:
                layer = Layer(weight=model_name_to_val_dict[nd_inps[1]], bias=(model_name_to_val_dict[nd_inps[2]]), type=LayerType.Conv2D, identifier=index, parents=parents[index])
            layers.append(layer)

            layer.kernel_size = (node.attribute[2].ints[0], node.attribute[2].ints[1])
            layer.padding = (node.attribute[3].ints[0], node.attribute[3].ints[1])
            layer.stride = (node.attribute[4].ints[0], node.attribute[4].ints[1])
            layer.dilation = (1, 1)
            
            shape = tensor_shapes[nd_inps[0]]
            if shape != layers[parents[index][0]].shape:
                raise ValueError(f"Conv {node.name!r} cannot consume a spatially reshaped alias")
            [i_1, i_2, i_3, i_4] = shape
            [k_1, k_2, k_3, k_4] = layer.weight.shape 
            (p_1, p_2) = layer.padding
            (s_1, s_2) = layer.stride 
            
            o_1 = i_1 
            o_2 = k_1 
            o_3 = math.floor((i_3 + 2*p_1 - k_3) / s_1) + 1
            o_4 = math.floor((i_4 + 2*p_2 - k_4) / s_2) + 1
            layer.shape = [o_1, o_2, o_3, o_4]
            #FEAT: keep the per-channel bias for refresh_layers (training fast path).
            layer.bias_raw = layer.bias
            layer.bias = (layer.bias.unsqueeze(1).repeat(1, o_3*o_4)).flatten()

            

        elif operation == 'Gemm':
            names_hash[str(net.graph.node[cur_layer].output[0])] = index
            parents[index] = [names_hash[str(net.graph.node[cur_layer].input[0])]]
            
            # Making some weird assumption that the weight is always 1th index
            #FIX FOR: Gemm attributes were ignored, so a transB=0 weight ([in, out], the ONNX default; e.g. Cersyve) was read as
            # [out, in]. Like MatMul, the transposed copy has no ONNX source, so refresh_layers keeps it as a constant.
            gemm_attrs = {a.name: onnx.helper.get_attribute_value(a) for a in net.graph.node[cur_layer].attribute}
            if gemm_attrs.get('transA', 0) or gemm_attrs.get('alpha', 1.0) != 1.0 or gemm_attrs.get('beta', 1.0) != 1.0:
                raise ValueError(f"Gemm {net.graph.node[cur_layer].name!r}: only transA=0, alpha=beta=1 are supported")
            gemm_weight = model_name_to_val_dict[nd_inps[1]]
            if not gemm_attrs.get('transB', 0):
                gemm_weight = gemm_weight.T.contiguous()
            layer = Layer(weight=gemm_weight, bias=(model_name_to_val_dict[nd_inps[2]]), type=LayerType.Linear, identifier=index, parents=parents[index])
            layers.append(layer)
            #FIX FOR: a Gemm reading a 2-D network input ([1, n], e.g. Cersyve) failed a 4-D unpack; only the batch dim is used.
            i_1 = shape[0]
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
            source_shape = tensor_shapes[nd_inps[0]]
            layer.shape = source_shape + [1] * (4 - len(source_shape))

        #FEAT: Tanh, as Sigmoid (ERAN tanh networks).
        elif operation in ('Sigmoid', 'Tanh'):
            names_hash[str(net.graph.node[cur_layer].output[0])] = index
            parents[index] = [names_hash[str(net.graph.node[cur_layer].input[0])]]
            
            layer = Layer(type=LayerType.Sigmoid if operation == 'Sigmoid' else LayerType.Tanh,
                          identifier=index, parents=parents[index])
            layers.append(layer)
            source_shape = tensor_shapes[nd_inps[0]]
            layer.shape = source_shape + [1] * (4 - len(source_shape))

            


        elif operation == 'Concat':
            if len(nd_inps) != 2 or not all(name in names_hash for name in nd_inps):
                raise ValueError(f"Concat {node.name!r} requires two activation inputs")
            left, right = (tensor_shapes[name] for name in nd_inps)
            attrs = {a.name: onnx.helper.get_attribute_value(a) for a in node.attribute}
            axis = attrs['axis']
            if axis < 0:
                axis += len(left)
            if (len(left) != len(right) or not 0 < axis < len(left)
                    or math.prod(left[:axis]) != 1
                    or any(a != b for i, (a, b) in enumerate(zip(left, right)) if i != axis)):
                raise ValueError(f"Unsupported Concat {node.name!r}: axis {axis}, {left}, {right}")
            result_shape = list(left)
            result_shape[axis] += right[axis]
            names_hash[node.output[0]] = index
            parents[index] = [names_hash[name] for name in nd_inps]
            layer = Layer(type=LayerType.Concat, identifier=index, parents=parents[index])
            layers.append(layer)
            layer.shape = result_shape + [1] * (4 - len(result_shape))

        elif operation == 'MatMul':
            names_hash[str(net.graph.node[cur_layer].output[0])] = index
            constant_inputs = [name in model_name_to_val_dict for name in nd_inps]
            if sum(constant_inputs) != 1:
                raise ValueError(
                    f"MatMul {node.name!r} requires exactly one initializer operand"
                )
            weight_index = constant_inputs.index(True)
            weight = model_name_to_val_dict[nd_inps[weight_index]]
            if weight.ndim != 2:
                raise ValueError(
                    f"MatMul {node.name!r} requires a 2D weight, got {tuple(weight.shape)}"
                )
            # Linear layers store [out_features, in_features]. For X @ W,
            # ONNX stores W as [in_features, out_features].
            if weight_index == 1:
                weight = weight.T.contiguous()
            parents[index] = [names_hash[nd_inps[1 - weight_index]]]
            layer = Layer(weight=weight, type=LayerType.Linear, identifier=index, parents=parents[index])
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
                tensor_shapes[node.output[0]] = tensor_shapes[nd_inps[0]]
                continue


            


        else:
            # Some layers are skipped and regarded as identity?
            # How can soundness still hold then?
            if len(net.graph.node[cur_layer].input)>0:
                if str(net.graph.node[cur_layer].input[0]) in names_hash:
                    names_hash[str(net.graph.node[cur_layer].output[0])] = names_hash[str(net.graph.node[cur_layer].input[0])]
                    tensor_shapes[node.output[0]] = tensor_shapes[nd_inps[0]]

            index-=1
            # assert(f"{operation} not supported")
            continue

        if operation in ('MatMul', 'Gemm'):
            tensor_shapes[node.output[0]] = [1, layer.shape[1]]
        elif operation in ('Relu', 'Sigmoid', 'Tanh', 'Add'):
            tensor_shapes[node.output[0]] = tensor_shapes[nd_inps[0]]
        elif operation == 'Concat':
            tensor_shapes[node.output[0]] = result_shape
        else:
            tensor_shapes[node.output[0]] = list(layer.shape)
        layer.size = compute_size(layer.shape)
        layer.start = layers.size 
        layers.size += layer.size 
        layer.end = layers.size
        shape = layer.shape

    layers = _collapse_affine(layers)
    index = len(layers) - 1
    #FEAT: fold the margin spec C into the last Linear (W' = C W), as auto_LiRPA's IBP does. The per-sample
    # bias C b is added to the output by spec.apply_spec_offset, since DSL biases are shared across the batch.
    if fold_spec and layers[-1].type == LayerType.Linear:
        last = layers[-1]
        last.weight_raw, last.bias_raw, last.folded_spec = last.weight, last.bias, True
        last.spec_offset = (spec_weight @ last.bias.unsqueeze(-1)).squeeze(-1) + spec_bias
        last.weight = spec_weight @ last.weight
        last.bias = torch.zeros_like(spec_bias)
        last.last_layer = True
        last.shape = [1, spec_weight.shape[-2], 1, 1]
        layers.size = layers.size - last.size
        last.size = compute_size(last.shape)
        last.start = layers.size
        layers.size += last.size
        last.end = layers.size
    elif True or layers[-1].type != LayerType.Linear:
        layer = Layer(weight=spec_weight, bias=spec_bias, type=LayerType.Linear, identifier=index+1, parents=[layers[-1].identifier])
        layer.last_layer = True
        layer.is_spec_layer = True  #FEAT: refresh_layers swaps in each batch's spec.
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
    #FIX FOR: num_layers counts ONNX nodes, but llist needs one slot per layer identifier: an appended spec layer
    # (a network ending in a residual Add, e.g. Cersyve) or one node lowered to several layers (vector Concat/Sub,
    # e.g. lsnc_relu) overflowed it.
    layers.num_layers = max(layers.num_layers, max(layer.identifier for layer in layers))
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

    # feeds_nonlin: False iff every immediate consumer is itself Affine
    # (Linear/Conv2D), i.e. this layer's l/u are never read by a relaxation.
    # A program-defined Add rule also never reads its parents' l/u (the builtin one does).
    # A layer with no children (only the synthesized spec layer) is conservatively
    # left at the True default set in Layer.__init__.
    from constraintflow.lib.globals import program_add_rule
    linear_consumers = (LayerType.Linear, LayerType.Conv2D) + ((LayerType.Add,) if program_add_rule else ())
    for layer in new_layers:
        if layer.children:
            layer.feeds_nonlin = not all(
                new_layers[child].type in linear_consumers
                for child in layer.children
            )

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
            # print(type(torch_layer))
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
