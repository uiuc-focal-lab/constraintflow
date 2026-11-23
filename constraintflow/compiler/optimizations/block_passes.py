"""
Block-level optimization round over the scalarized flow block.

The passes in `optimizations_rewrite` are keyed to `ir.tstore[...].cfg`, so none
of them reach `ir.flow_block`. Wrapping the block in a one-node CFG lets the
existing cfg-level passes run on it directly -- valid only while the block is
pure SSA, which it is exactly when sroa reports no view writes, since those are
the only mutations it ever emits.
"""

from constraintflow.compiler import representations
from constraintflow.compiler.ir import IrBlock
from constraintflow.compiler.optimizations import copyPropagation
from constraintflow.compiler.optimizations import uses
from constraintflow.compiler.optimizations import dce as dce_pass


def _as_cfg(block):
    """One-node CFG so the cfg-level passes apply to a straight-line block."""
    cfg = representations.Graph()
    cfg.ir[cfg.add()] = block
    return cfg


def cse(block):
    from constraintflow.compiler.optimizations import ssa_cse
    return ssa_cse.run(block)  # one-walk hash-bucket CSE on functional SSA


def copy_propagate(block):
    copyPropagation.cp_cfg(_as_cfg(block))


def dce(block):
    cfg = _as_cfg(block)
    uses.populate_uses_defs_cfg(cfg)
    dce_pass.dce_cfg(cfg)


block_optimizations = [cse, copy_propagate, dce]

# Only cse may run on a layer slice: it adds definitions and never removes any.
# copy_propagate deletes `x = y` after rewriting, and dce needs whole-block
# liveness -- per slice both would drop values a later layer still reads. cse is
# also the only quadratic one, so slicing it is where the time goes anyway.
sliceable = (cse,)

# Slicing costs cross-layer cse, worth ~16% of the split win on convBig, so only
# take it when the whole-block pass would be too slow (it is quadratic).
SLICE_ABOVE = 5000


def _apply(piece, optimizations, counts):
    for opt in optimizations:
        name = getattr(opt, 'pass_name', None) or getattr(opt, '__name__', repr(opt))
        before = len(piece.children)
        opt(piece)
        counts[name] = counts.get(name, 0) + (before - len(piece.children))


def run(block, bounds=None, optimizations=None):
    """
    public
    Requires: `block` is straight-line SSA (sroa reported no view writes).

    With `bounds` (layer end offsets from splice_flow) each layer is optimized on
    its own. These passes are quadratic in block size, so L slices of n/L cost
    n^2/L; the trade is that cse no longer sees across layers.
    """
    optimizations = block_optimizations if optimizations is None else optimizations
    before = len(block.children)
    counts, layers = {}, None
    slicing = bounds and before > SLICE_ABOVE
    per_slice = [o for o in optimizations if o in sliceable] if slicing else []
    if per_slice:
        children = list(block.children)
        edges = [0] + list(bounds[:-1]) + [len(children)]
        out, layers = [], {}
        for i in range(len(edges) - 1):
            piece = IrBlock(children[edges[i]:edges[i + 1]])
            _apply(piece, per_slice, counts)
            for stmt in piece.children:
                layers[id(stmt)] = (stmt, i)   # keep a ref; a freed id gets reused
            out.extend(piece.children)
        block.update_parent_child(out)
    _apply(block, [o for o in optimizations if o not in per_slice], counts)
    return {'statements_before': before, 'statements_after': len(block.children),
            'counts': counts, 'layers': layers}
