from constraintflow.compiler.ir import (
    IrAssignment, IrVar, IrBinaryOp, IrMult, IrInnerProduct, IrClamp,
    IrRemoveDimension, IrAddDimension, IrRepeat,
)

_topology_cache = {}
_fusion_total = 0  # DEBUG: running count of fusions across the whole compile


def _load_topology(network_path):
    """Layer type + parents for every layer in network_path, computed once and
    cached. Only topology is used here -- the spec passed to parse_onnx_layers
    is a placeholder, since the synthesized final layer's type/parent don't
    depend on its actual values, only on the network's own structure.
    """
    if network_path in _topology_cache:
        return _topology_cache[network_path]
    import onnx
    import torch
    from constraintflow.lib.parse import parse_onnx_layers
    net = onnx.load(network_path)
    layers = parse_onnx_layers(net, torch.zeros(1, 1), torch.zeros(1), no_sparsity=True)
    types = {layer.identifier: layer.type for layer in layers}
    parents = {layer.identifier: layer.parents for layer in layers}
    _topology_cache[network_path] = (types, parents)
    return types, parents


def crossed_layer_at(layer_index, iteration, parents):
    """Layer crossed at unrolled iteration `iteration` (0-indexed) of a full
    backward traversal starting at layer_index. Correct for a simple-chain
    network under priority = n[layer] / stop_traverse = false (every
    deeppoly*/crown spec in this repo): traversal visits strictly decreasing
    layer indices, i.e. layer_index's parent, then its parent, etc. Returns
    None (don't know) rather than guessing when a layer has other than exactly
    one parent (Add/Concat) -- the caller must then skip the merge.
    """
    current = layer_index
    for _ in range(iteration + 1):
        ps = parents.get(current)
        if ps is None or len(ps) != 1:
            return None
        current = ps[0]
    return current


def selected_layers_at(layer_index, while_number, iteration):
    """Actual priority selection; never infer a residual traversal from depth."""
    from constraintflow.lib.globals import capture_exists, load_capture
    path = f"jit_selection/selection_{layer_index}_{while_number}_{iteration}.json"
    if not capture_exists(path):
        return []
    selection = load_capture(path)
    if not selection.get('substitutions'):
        raise ValueError(f'Unvalidated priority selection {path}; rebuild captures.')
    return selection['layers']


def _deref(expr):
    if isinstance(expr, IrVar) and expr.defs is not None:
        return expr.defs.children[1]
    return expr


_WRAPPERS = (IrRemoveDimension, IrAddDimension, IrRepeat)


def _unwrap_clamp(expr):
    """Peel at most one IrRemoveDimension/IrAddDimension/IrRepeat wrapper to find
    a bare IrClamp underneath. Returns (clamp, wrapper) where `wrapper` is the
    node whose own first child must be reassigned in place (preserving its
    ttb_counter/identity) to bypass the clamp, or None meaning the caller's own
    node has no wrapper and points at the clamp directly. Returns (None, None)
    if there's no IrClamp at the core.
    """
    e = _deref(expr)
    if isinstance(e, IrClamp):
        return e, None
    if isinstance(e, _WRAPPERS):
        inner = _deref(e.children[0])
        if isinstance(inner, IrClamp):
            return inner, e
    return None, None


def _clamp_reduce(expr):
    """expr is expected to be a (possibly IrVar-indirected) IrInnerProduct, or
    IrMult('*'), whose first or second operand unwraps to an IrClamp. Returns
    (node, clamp, clamp_is_first_child, wrapper) or None -- see _unwrap_clamp
    for `wrapper`.
    """
    e = _deref(expr)
    if not (isinstance(e, IrInnerProduct) or (isinstance(e, IrMult) and e.op == '*')):
        return None
    lhs, rhs = e.children
    clamp, wrapper = _unwrap_clamp(lhs)
    if clamp is not None:
        return e, clamp, True, wrapper
    clamp, wrapper = _unwrap_clamp(rhs)
    if clamp is not None:
        return e, clamp, False, wrapper
    return None


def fuse_iteration(statements, is_affine):
    """Mutates `statements` -- one unrolled traverse() iteration's raw IR,
    already deep-copied for this iteration, before tensor_to_block_block
    converts it into captured-replay form -- in place. When is_affine,
    collapses every clamp+(c).X + clamp-(c).Y pair into c.X directly: exact
    when X == Y, which affine_subst_check proves for every Affine op's
    L/U (both write the same expression). Only rewires references (every node
    that survives keeps its own pre-assigned ttb_counter, so its capture stays
    valid); the now-orphaned clamp-/mult- branch is left in place for
    subexp_inlining.drop_dead_assignments (which runs after tensor_to_block,
    recomputing liveness from the instruction list) to remove.
    """
    if not is_affine:
        return

    global _fusion_total
    fused_here = 0  # DEBUG

    for stmt in statements:
        if not isinstance(stmt, IrAssignment):
            continue
        rhs = stmt.children[1]
        if not (isinstance(rhs, IrBinaryOp) and rhs.op == '+'):
            continue
        a_operand, b_operand = rhs.children
        a = _clamp_reduce(a_operand)
        b = _clamp_reduce(b_operand)
        if a is None or b is None:
            continue
        a_node, a_clamp, a_clamp_first, a_wrapper = a
        _, b_clamp, _, _ = b
        if a_clamp.min_true == b_clamp.min_true:
            continue
        if _deref(a_clamp.children[0]) != _deref(b_clamp.children[0]):
            continue

        # a_node's own result, once unclamped, equals what the sum would compute
        # (a_other == b_other under the Affine L==U assertion) -- so keep a_node,
        # drop its clamp, and point this statement straight at it.
        unclamped = a_clamp.children[0]
        if a_wrapper is not None:
            # The clamp sits under a wrapper (squeeze/unsqueeze/repeat) that
            # a_node's own child already points at -- mutate the wrapper's own
            # child in place so the wrapper keeps its identity/ttb_counter.
            a_wrapper.update_parent_child([unclamped] + list(a_wrapper.children[1:]))
        elif a_clamp_first:
            a_node.update_parent_child([unclamped, a_node.children[1]])
        else:
            a_node.update_parent_child([a_node.children[0], unclamped])

        stmt.update_parent_child([stmt.children[0], a_operand])
        fused_here += 1  # DEBUG

    if fused_here:  # DEBUG
        _fusion_total += fused_here
        print(f"[fuse_affine_subst] fused {fused_here} pair(s) this call (total so far: {_fusion_total})")
