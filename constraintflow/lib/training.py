#FEAT: certified training with compiled ConstraintFlow programs.
"""Differentiable margin bounds for a torch model, used like auto_LiRPA's BoundedModule.

Each (method, batch size) is compiled once with `cli jit --training`, cached on disk, loaded as a
package, and built once. A step only swaps in the box, the labels' margin spec and the parameters.
"""
import hashlib
import importlib
import importlib.util
import os
import shutil
import subprocess
import sys

import onnx
import torch

import constraintflow
from constraintflow.lib.globals import device_mode
from constraintflow.lib.parse import refresh_layers
from constraintflow.lib.spec import ImageDataset, apply_spec_offset

CF_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(constraintflow.__file__)))
#FIX FOR: training back-substituted the unused spec upper bound (auto_LiRPA trains with bound_upper=False), doubling the backward.
PROGRAMS = {'IBP': 'ibp.cf', 'CROWN-IBP': 'crownibp_lower.cf',
            'CROWN': 'deeppoly_priority_add_lower.cf',  # DeepPoly with the residual Add rule = auto_LiRPA's CROWN
            'zid': 'zid_lower.cf',  #FIX FOR: zid back-substituted the unused spec upper bound too
            'CROWN-lower': 'deeppoly_priority_add_lower.cf'}  #FEAT: CROWN with only the last layer's lower bound back-substituted
#FEAT: DeepZ and DeepPoly with back-substitution stopped KK layers back, spec upper bound one-step interval.
PROGRAMS.update({'zono': 'zono_lower.cf', **{f'DP-stop{k:02d}': f'custom_stops/deeppoly_custom_stop{k:02d}_lower.cf' for k in range(19)}})
#FEAT: the same stops with CROWN-0's (c0), CROWN-0-C's (c0c) or Fast-Lin's (fl) ReLU lower bound.
PROGRAMS.update({f'DP-stop{k:02d}-{v}': f'custom_stops/deeppoly_custom_stop{k:02d}_{v}_lower.cf' for k in (1, 2, 3, 18) for v in ('c0', 'c0c', 'fl')})


def _source_hash():
    h = hashlib.sha256()
    package = os.path.dirname(os.path.abspath(constraintflow.__file__))
    for root, dirs, files in os.walk(package):
        dirs[:] = sorted(d for d in dirs if d != '__pycache__')
        for name in sorted(f for f in files if f.endswith('.py')):
            with open(os.path.join(root, name), 'rb') as f:
                h.update(f.read())
    return h.hexdigest()


# Computed at import, so editing cf-gpu invalidates the compiled-program cache.
SOURCE_HASH = _source_hash()


def export_model(model, dummy_input, cache_dir):
    """Write the model as ONNX; return its path and the parameters, keyed by ONNX initializer name."""
    os.makedirs(cache_dir, exist_ok=True)
    #FIX FOR: runs on different nets sharing a cache overwrote one model.onnx, so one compiled the other's net.
    path = os.path.join(cache_dir, f'model{os.getpid()}.onnx')
    was_training = model.training
    model.eval()
    torch.onnx.export(model, dummy_input[:1].to(next(model.parameters()).device), path,
                      dynamo=False, do_constant_folding=False, opset_version=13)
    model.train(was_training)
    params = dict(model.named_parameters())
    #FIX FOR: the exporter stores equal parameters once and re-creates the rest with Identity nodes, so those live
    #         parameters were never bound (parser KeyError; aliasing them would train the wrong parameter).
    graph = onnx.load(path)
    stored = {i.name: i for i in graph.graph.initializer}
    copies = [n for n in graph.graph.node if n.op_type == 'Identity' and n.input[0] in stored and n.output[0] in params]
    for n in copies:
        init = onnx.TensorProto()
        init.CopyFrom(stored[n.input[0]])
        init.name = n.output[0]
        graph.graph.initializer.append(init)
        graph.graph.node.remove(n)
    if copies:
        onnx.save(graph, path)
    inits = [i for i in onnx.load(path).graph.initializer if i.data_type == onnx.TensorProto.FLOAT]
    unknown = [i.name for i in inits if i.name not in params or tuple(params[i.name].shape) != tuple(i.dims)]
    if unknown:
        raise ValueError(f'ONNX initializers that are not model parameters: {unknown}')
    return path, {i.name: params[i.name] for i in inits}


def program_dir(cache_dir, onnx_path, method, batch_size, dataset, cudagraphs, jit_flags=()):
    """Cache folder named by everything the generated code depends on (not the weights)."""
    graph = onnx.load(onnx_path).graph
    structure = repr([(n.op_type, list(n.input), list(n.output), [str(a) for a in n.attribute]) for n in graph.node]
                     + [(i.name, list(i.dims), i.data_type) for i in graph.initializer])
    with open(os.path.join(CF_ROOT, 'examples', 'compiler_examples', PROGRAMS[method]), 'rb') as f:
        program = f.read()
    key = hashlib.sha256(b'|'.join([SOURCE_HASH.encode(), structure.encode(), program,
                                    f'{batch_size}|{dataset}|{cudagraphs}|{" ".join(jit_flags)}'.encode()])).hexdigest()[:16]
    return os.path.join(cache_dir, f'{method}_b{batch_size}_{key}')


#FEAT: jit_flags = extra `cli jit` options (e.g. --fuse-affine-subst), part of the cache key.
def compile_program(onnx_path, method, batch_size, dataset, cache_dir, cudagraphs=True, data_root='./data', jit_flags=()):
    """Compile into the cache unless already there; return the program folder."""
    folder = program_dir(cache_dir, onnx_path, method, batch_size, dataset, cudagraphs, jit_flags)
    if os.path.exists(os.path.join(folder, '__init__.py')):
        return folder
    #FIX FOR: concurrent runs sharing a cache compiled the same program into one .tmp folder and deleted each other's.
    tmp = f'{folder}.tmp{os.getpid()}'
    shutil.rmtree(tmp, ignore_errors=True)
    cmd = [sys.executable, '-m', 'constraintflow.cli', 'jit', os.path.join(CF_ROOT, 'examples', 'compiler_examples', PROGRAMS[method]),
           '--network', os.path.abspath(onnx_path), '--dataset', dataset, '--batch-size', str(batch_size), '--eps', '0.01',
           '--device', 'gpu', '--output-path', os.path.abspath(tmp) + '/', '--inductor', '--training'] + (['--cudagraphs'] if cudagraphs else []) + list(jit_flags)
    #FEAT: DeepPoly programs get --fuse-affine-subst, as in the verification harness.
    #FIX FOR: custom_stops/deeppoly_* programs missed the flag (the test was on the path, not the file name).
    if os.path.basename(PROGRAMS[method]).startswith('deeppoly'):
        cmd.append('--fuse-affine-subst')
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([CF_ROOT, os.environ.get('PYTHONPATH', '')]))
    # cli jit traces one batch read from torchvision under its working directory.
    res = subprocess.run(cmd, cwd=data_root, env=env, capture_output=True, text=True)
    with open(f'{folder}.compile{os.getpid()}.log', 'w') as f:
        f.write(res.stdout + res.stderr)
    if res.returncode != 0:
        raise RuntimeError(f'cf-gpu compile failed, see {folder}.compile{os.getpid()}.log')
    try:
        os.rename(tmp, folder)
    except OSError:  # another run finished the same program first
        shutil.rmtree(tmp, ignore_errors=True)
    return folder


def compile_programs(model, dummy_input, dataset, methods, batch_sizes, cache_dir, cudagraphs=True, data_root='./data'):
    onnx_path, _ = export_model(model, dummy_input, cache_dir)
    return {(m, b): compile_program(onnx_path, m, b, dataset, cache_dir, cudagraphs, data_root)
            for m in methods for b in batch_sizes}


def load_program(folder):
    """Import a compiled folder as its own package, so several programs can be loaded at once."""
    name = 'cf_program_' + ''.join(c if c.isalnum() else '_' for c in os.path.basename(folder))
    spec = importlib.util.spec_from_file_location(name, os.path.join(folder, '__init__.py'),
                                                  submodule_search_locations=[folder])
    sys.modules[name] = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sys.modules[name])
    return importlib.import_module(name + '.main')


class TrainingBounds:
    """Lower/upper bounds on the margins f_y - f_j (j != y) under an Linf box, differentiable in the
    model's parameters. One compiled program per (method, batch size)."""

    def __init__(self, model, dummy_input, dataset, methods, batch_sizes, cache_dir, cudagraphs=True, data_root='./data'):
        device_mode.set_mode('gpu')
        self.dataset, self.cudagraphs = dataset, cudagraphs
        self.onnx_path, self.params = export_model(model, dummy_input, cache_dir)
        self.programs = {}
        for method in methods:
            for batch_size in batch_sizes:
                folder = compile_program(self.onnx_path, method, batch_size, dataset, cache_dir, cudagraphs, data_root)
                self.programs[method, batch_size] = self._build(load_program(folder), dummy_input, batch_size)

    def _build(self, main, dummy_input, batch_size):
        x = torch.zeros(batch_size, *dummy_input.shape[1:], device='cuda')
        y = torch.zeros(batch_size, dtype=torch.long, device='cuda')
        network, abs_elem = main.build(self.onnx_path, batch_size, 0.0, x, y, self.dataset, False, False, False,
                                       initializers=self.params, input_box=(x, x))
        if not set(abs_elem.d) <= {'llist', 'l', 'u', 'L', 'U', 'Z'} or not all(layer.refreshable for layer in network):
            raise ValueError('this program or network is not supported for training')
        return network, abs_elem, main.flow, main.explode_inputs

    def begin_step(self):
        """Call once per batch: outputs of the previous step's CUDA graph replays may then be reused."""
        if self.cudagraphs:
            torch.compiler.cudagraph_mark_step_begin()

    def compute_bounds(self, method, x_L, x_U, labels):
        spec_weight, spec_bias = ImageDataset.get_output_spec_weight_and_bias(x_L, labels, self.dataset)
        return self.compute_spec_bounds(method, x_L, x_U, spec_weight, spec_bias)

    #FEAT: bounds on spec_weight @ f(x) + spec_bias for any [batch, rows, outputs] spec (input splitting).
    def compute_spec_bounds(self, method, x_L, x_U, spec_weight, spec_bias):
        batch_size = x_L.size(0)
        network, abs_elem, flow, explode_inputs = self.programs[method, batch_size]
        refresh_layers(network, self.params, spec_weight, spec_bias)
        l, u = x_L.reshape(batch_size, -1), x_U.reshape(batch_size, -1)
        abs_elem.d['l'].blocks[0].block, abs_elem.d['u'].blocks[0].block = l, u
        # build() gives L and U their own copies of the box.
        if 'L' in abs_elem.d:
            abs_elem.d['L'].const.blocks[0].block = l.clone()
        if 'U' in abs_elem.d:
            abs_elem.d['U'].const.blocks[0].block = u.clone()
        #FEAT: zonotope input (zid): center and per-input radius of the box, as spec.create_Z builds them.
        if 'Z' in abs_elem.d:
            abs_elem.d['Z'].mat.blocks[0].block = (u - l) / 2
            abs_elem.d['Z'].const.blocks[0].block = (l + u) / 2
        return apply_spec_offset(network, flow(*explode_inputs(abs_elem, batch_size)))


#FEAT: verification of a fixed ONNX network over batches of input boxes (input splitting, e.g. ACAS Xu).
class SpecBounds(TrainingBounds):
    """compute_spec_bounds on a fixed ONNX file: its own initializers stand in for model parameters."""

    def __init__(self, onnx_path, input_shape, dataset, methods, batch_sizes, cache_dir, cudagraphs=True, data_root='./data', jit_flags=()):
        device_mode.set_mode('gpu')
        self.dataset, self.cudagraphs, self.onnx_path = dataset, cudagraphs, onnx_path
        inits = [i for i in onnx.load(onnx_path).graph.initializer if i.data_type == onnx.TensorProto.FLOAT]
        self.params = {i.name: torch.from_numpy(onnx.numpy_helper.to_array(i).copy()).cuda() for i in inits}
        dummy_input = torch.zeros(1, *input_shape)
        os.makedirs(cache_dir, exist_ok=True)
        self.programs = {}
        for method in methods:
            for batch_size in batch_sizes:
                folder = compile_program(onnx_path, method, batch_size, dataset, cache_dir, cudagraphs, data_root, jit_flags)
                self.programs[method, batch_size] = self._build(load_program(folder), dummy_input, batch_size)
