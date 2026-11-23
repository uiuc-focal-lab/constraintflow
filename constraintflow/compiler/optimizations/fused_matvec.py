#FEAT: new pass
"""Lower large pointwise-produced matvecs to fusible row reductions.

BLAS requires a materialized matrix. Inductor can instead fuse the matrix's
pointwise producer, multiplication and sum, avoiding a full write/read of the
clamped coefficients. Keep individual dot products and surrounding additions
separate; this changes only the floating-point reduction implementation.
"""
import copy
import torch
from constraintflow.compiler.ir import (
    IrAssignment, IrTransRetBasic, IrTorchMatmul, IrTorchTranspose,
    IrTorchUnsqueeze, IrTorchSum, IrSimpleBinary,
)
from constraintflow.compiler.optimizations.product_grids import _meta_env


def run(block, render, sized, batch_size, min_bytes=1024**2):
    if any(not isinstance(s, (IrAssignment, IrTransRetBasic)) for s in block.children):
        return 0
    env = _meta_env(sized, batch_size)
    cache = {}
    count = 0

    def shape(expr):
        text = render(expr)
        if text not in cache:
            try:
                with torch.device('meta'):
                    value = eval(text, dict(env))
                cache[text] = value if isinstance(value, torch.Tensor) else None
            except Exception:
                cache[text] = None
        return cache[text]

    def rewrite(expr):
        nonlocal count
        if not getattr(expr, 'children', None):
            return expr
        children = [rewrite(c) for c in expr.children]
        out = copy.copy(expr)
        out.update_parent_child(children)
        if not isinstance(out, IrTorchMatmul):
            return out
        a, x = children
        # A cheap producer inside the same expression can be fused with sum.
        # Avoid convolution/matmul producers whose full result is required.
        source = render(a)
        if not any(op in source for op in ('torch.clamp(', 'torch.where(')):
            return out
        if any(op in source for op in ('conv2d(', 'conv_transpose2d(', 'matmul(')):
            return out
        am, xm = shape(a), shape(x)
        if (am is None or xm is None or am.ndim < 2 or xm.ndim != am.ndim
                or xm.shape[-1] != 1 or am.shape[-1] != xm.shape[-2]
                or am.dtype != xm.dtype or am.dtype not in (torch.float32, torch.float64)
                or am.numel() * am.element_size() < min_bytes):
            return out
        try:
            torch.broadcast_shapes(am.shape[:-2], xm.shape[:-2])
        except RuntimeError:
            return out
        count += 1
        product = IrSimpleBinary(a, IrTorchTranspose(x, -1, -2), 'mul')
        return IrTorchUnsqueeze(IrTorchSum(product, -1), -1)

    for stmt in block.children:
        if isinstance(stmt, IrAssignment):
            stmt.update_parent_child([stmt.children[0], rewrite(stmt.children[1])])
        else:
            stmt.update_parent_child([rewrite(c) for c in stmt.children])
    return count
