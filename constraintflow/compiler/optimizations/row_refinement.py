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


#FIX FOR: the rest went to the smallest size covering it, so 260 rows ran as a 500-row block (240 padded
#         rows of real work; N8 _cr_rows_3). Rows are refined independently, so the split changes no value.
def block_plan(rows, sizes, pack=True):
    """Block sizes covering `rows`: full blocks of the largest size, then the smallest size covering the
    rest if that pads fewer rows than the smallest size, else the largest size that fits, repeated.
    pack=False covers the rest with one block of the smallest size that fits it (fewer launches)."""
    plan = []
    while rows > 0:
        if rows >= sizes[-1]:
            size = sizes[-1]
        elif not pack:
            size = next(c for c in sizes if c >= rows)
        else:
            cover = next(c for c in sizes if c >= rows)
            fits = [c for c in sizes if c <= rows]
            size = cover if not fits or cover - rows < sizes[0] else fits[-1]
        plan.append(size)
        rows -= min(size, rows)
    return plan


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
