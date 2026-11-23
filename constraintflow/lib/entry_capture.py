"""Structure of abs_elem at flow entry, captured for SROA to seed from."""

import torch

from constraintflow.gbcsr.sparse_block import identifySparseBlockType
from constraintflow.gbcsr.sparse_tensor import SparseTensor
from constraintflow.lib.globals import save_capture

CAPTURE_PATH = "jit_input/abs_elem.json"

# mirrors codeGen.BLOCK_FIELDS
BLOCK_GEOMETRY = {
    'DenseBlock': ('batch_size',),
    'ConstBlock': (),
    'RepeatBlock': ('repeat_dims', 'only_one_repeat'),
    'DiagonalBlock': ('diag_index', 'batch_size'),
    'KernelBlock': ('ix', 'iy', 'ox', 'oy', 'sx', 'sy', 'px', 'py',
                    'kx', 'ky', 'num_channels', 'num_kernels'),
    'PatchesBlock': ('ix', 'iy', 'ox', 'oy', 'sx', 'sy', 'px', 'py',
                     'kx', 'ky', 'num_channels', 'num_kernels'),
}


def _plain(value):
    if isinstance(value, torch.Tensor):
        return value.tolist()
    return value


def _dtype(payload):
    if isinstance(payload, bool):
        return 'py_bool'
    if not isinstance(payload, torch.Tensor):
        return 'py_float'
    return 'bool' if payload.dtype == torch.bool else 'f32'


def _delete_indices(tensor):
    # mirrors SparseTensor.__init__
    out = []
    for i, block in enumerate(tensor.blocks):
        if block.block_type != 'C':
            continue
        payload = block.block
        if isinstance(payload, torch.Tensor) and not (
                payload.numel() == 1 and not payload.is_meta):
            continue
        if payload == tensor.dense_const and tensor.dense_const in (0.0, False):
            out.append(i)
    out.reverse()
    return out


def _describe_block(block):
    kind = type(block).__name__
    if kind not in BLOCK_GEOMETRY:
        raise RuntimeError(f"entry_capture: unsupported block {kind}")
    return {
        "kind": kind,
        "block_type": block.block_type,
        "tier": identifySparseBlockType(block.block),
        "total_shape": _plain(block.total_shape),
        "shape": (list(block.block.shape)
                  if isinstance(block.block, torch.Tensor) else None),
        "dtype": _dtype(block.block),
        "fields": {name: _plain(getattr(block, name))
                   for name in BLOCK_GEOMETRY[kind]},
    }


def _describe_sparse(tensor):
    return {
        "node": "sparse",
        "dims": tensor.dims,
        "total_size": _plain(tensor.total_size),
        "start_indices": [_plain(x) for x in tensor.start_indices],
        "end_indices": [_plain(x) for x in tensor.end_indices],
        "type": tensor.type.__name__,
        "dense_const": tensor.dense_const,
        "delete_indices": _delete_indices(tensor),
        "blocks": [_describe_block(b) for b in tensor.blocks],
    }


def describe(value):
    if isinstance(value, SparseTensor):
        return _describe_sparse(value)
    if hasattr(value, 'mat') and hasattr(value, 'const'):
        return {"node": "poly",
                "mat": describe(value.mat),
                "const": describe(value.const)}
    if isinstance(value, (bool, int, float)):
        return {"node": "scalar", "value": value}
    return {"node": "opaque", "kind": type(value).__name__}


def _describe_network(network):
    out = {}
    for i, layer in enumerate(network):
        params = {name: list(getattr(layer, name).shape)
                  for name in ('weight', 'bias')
                  if isinstance(getattr(layer, name, None), torch.Tensor)}
        if params:
            out[str(i)] = params
    return out


def save_entry_capture(abs_elem):
    save_capture(CAPTURE_PATH, {
        "entry": {key: describe(value) for key, value in abs_elem.d.items()},
        "network": _describe_network(abs_elem.network),
        "batch_size": abs_elem.batch_size,
    })
