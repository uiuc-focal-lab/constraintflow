"""Runtime helpers for row-selected refinement (traversal-stop builds).

Generated code imports selected_patches_to_dense; `run` wraps every warmup in
warmup_row_kernels so each compiled row-block size is built before timing.
"""
from contextvars import ContextVar

import torch

_warming = ContextVar('warming_row_kernels', default=False)


def is_warming():
    return _warming.get()


def warmup_row_kernels(call, *args, **kwargs):
    """Run a warmup that exercises even currently unused refinement kernels.

    Dummy results are discarded, so warmup returns the same bounds as a normal
    call. Use the same grad mode, dtype and device as subsequent measurements.
    No warmup state persists after the call, including if compilation raises.
    """
    token = _warming.set(True)
    try:
        return call(*args, **kwargs)
    finally:
        _warming.reset(token)


def selected_patches_to_dense(pieces, indices, batch, oc, ox, oy, ic, kx, ky, ix, iy, px, py, sx, sy):
    """Patch conversion retaining each selected neuron's original spatial origin."""
    pos = indices % (ox * oy)
    x = torch.arange(ix, device=pieces.device)[None, :] + px - (pos // oy)[:, None] * sx
    y = torch.arange(iy, device=pieces.device)[None, :] + py - (pos % oy)[:, None] * sy
    valid = ((x >= 0) & (x < kx))[:, :, None] & ((y >= 0) & (y < ky))[:, None, :]
    index = (x.clamp(0, kx - 1)[:, :, None] * ky + y.clamp(0, ky - 1)[:, None, :])
    count = indices.numel()
    source = pieces.reshape(batch, count, ic, kx * ky)
    index = index.reshape(1, count, 1, ix * iy).expand(batch, count, ic, ix * iy)
    values = source.gather(-1, index)
    return torch.where(valid.reshape(1, count, 1, ix * iy), values, 0).reshape(batch, count, ic * ix * iy)
