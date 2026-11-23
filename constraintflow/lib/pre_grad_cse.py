#FEAT: CSE on the Dynamo graph before AOTAutograd, so a duplicated op gets one backward, not one per copy.
"""Generated flows repeat subexpressions (e.g. an interval conv appears in both the l and u
expressions). AOTAutograd's own CSE merges the forward copies only after tracing a backward for
each, so training paid every duplicate conv/matmul backward twice."""
import operator

import torch
from torch._inductor.custom_graph_pass import CustomGraphPass, get_hash_for_files

_NONDETERMINISTIC = ('rand', 'dropout', 'normal', 'bernoulli', 'multinomial', 'empty')


class _Unhashable(Exception):
    pass


def _key(a):
    if isinstance(a, torch.fx.Node):
        return a
    if isinstance(a, (list, tuple)):
        return (type(a).__name__,) + tuple(_key(x) for x in a)
    if isinstance(a, dict):
        return ('dict',) + tuple(sorted((k, _key(v)) for k, v in a.items()))
    if isinstance(a, slice):
        return ('slice', _key(a.start), _key(a.stop), _key(a.step))
    if isinstance(a, float):
        return ('float', repr(a))  # keeps 0.0 and -0.0 apart
    if a is None or isinstance(a, (bool, int, str, torch.dtype, torch.device, torch.layout, torch.memory_format)):
        return (type(a).__name__, a)
    raise _Unhashable


def _mutates(node):
    name = getattr(node.target, '__name__', str(node.target))
    return (node.target is operator.setitem or 'out' in node.kwargs
            or (node.op in ('call_function', 'call_method') and name.endswith('_') and not name.endswith('__')))


_VIEWS = {'squeeze', 'unsqueeze', 'reshape', 'view', 'expand', 'transpose', 'permute', 'contiguous', 'flatten', '__getitem__'}
_BILINEAR = {torch.matmul, torch.bmm, torch.mm, operator.matmul, operator.mul, torch.mul}
_ZERO = (('int', 0), ('float', '0.0'))


def _sign_split_path(a, b):
    """[(node, arg position), ...] from a down to clamp(x, min=0) when b is the same chain over clamp(x, max=0)
    (or the reverse) and every step is linear in that argument; None otherwise."""
    if not (isinstance(a, torch.fx.Node) and isinstance(b, torch.fx.Node)) or a is b:
        return None
    if a.target is torch.clamp and b.target is torch.clamp and len(a.args) == len(b.args) == 1 and a.args[0] is b.args[0]:
        bounds = {k: _key(v) for k, v in {**a.kwargs, **{'_' + k: v for k, v in b.kwargs.items()}}.items()}
        if set(bounds) in ({'min', '_max'}, {'max', '_min'}) and all(v in _ZERO for v in bounds.values()):
            return []
        return None
    if a.op != b.op or a.target != b.target or len(a.args) != len(b.args):
        return None
    if not ((a.op == 'call_method' and a.target in _VIEWS) or (a.op == 'call_function' and a.target in _BILINEAR)):
        return None
    try:
        if _key(a.kwargs) != _key(b.kwargs):
            return None
        diff = [i for i, (x, y) in enumerate(zip(a.args, b.args)) if _key(x) != _key(y)]
    except _Unhashable:
        return None
    if len(diff) != 1 or (a.op == 'call_method' and diff[0] != 0):
        return None
    rest = _sign_split_path(a.args[diff[0]], b.args[diff[0]])
    return None if rest is None else [(a, diff[0])] + rest


def _merge_sign_splits(graph):
    """f(clamp(x, min=0)) + f(clamp(x, max=0)) -> f(x) for linear f: one matmul (and its backward) instead of two."""
    changed = False
    for node in list(graph.nodes):
        if node.target not in (operator.add, torch.add) or len(node.args) != 2 or node.kwargs:
            continue
        path = _sign_split_path(*node.args)
        if not path:
            continue
        with graph.inserting_before(node):
            cur = path[-1][0].args[path[-1][1]].args[0]  # x
            for chain_node, pos in reversed(path):
                args = list(chain_node.args)
                args[pos] = cur
                new = graph.create_node(chain_node.op, chain_node.target, tuple(args), dict(chain_node.kwargs))
                new.meta = dict(chain_node.meta)
                cur = new
        node.replace_all_uses_with(cur)
        graph.erase_node(node)
        changed = True
    if changed:
        graph.eliminate_dead_code()


class PreGradCSE(CustomGraphPass):
    def __call__(self, graph):
        nodes = list(graph.nodes)
        if any(_mutates(n) for n in nodes):
            return  # aliasing makes value numbering unsafe; leave such graphs alone
        seen = {}
        for node in nodes:
            name = getattr(node.target, '__name__', str(node.target))
            if (node.op not in ('call_function', 'call_method') or node.is_impure()
                    or any(s in name for s in _NONDETERMINISTIC)):
                continue
            try:
                key = (node.op, node.target, _key(node.args), _key(node.kwargs))
                hash(key)
            except (_Unhashable, TypeError):
                continue
            if key in seen:
                node.replace_all_uses_with(seen[key])
                graph.erase_node(node)
            else:
                seen[key] = node
        #FEAT: merge sign-split products (A+ W + A- W) left by back-substitution into one product.
        _merge_sign_splits(graph)

    def uuid(self):
        return get_hash_for_files((__file__,))


PRE_GRAD_CSE = PreGradCSE()
