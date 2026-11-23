#FEAT: Inductor post-grad pass for the generated flow(): mm(x, w1), mm(x, w2), ... sharing a left
# operand become one mm(x, cat(w1, w2, ...)) and column slices. In back-substitution a relaxed
# coefficient matrix meets a layer's weight and its bias (x @ W, x @ b), so x is read once, not twice.
import torch
from torch._inductor.custom_graph_pass import CustomGraphPass, get_hash_for_files

aten = torch.ops.aten
_VIEWS = (aten.view.default, aten.reshape.default, aten._unsafe_view.default, aten.squeeze.dim, aten.unsqueeze.default)


def _base(node):
    """The tensor a chain of reshapes starts from, so x.view(...) and x.reshape(...) group together."""
    while node.op == 'call_function' and node.target in _VIEWS:
        node = node.args[0]
    return node


def _pure(node):
    schema = getattr(node.target, '_schema', None)
    return schema is not None and not schema.is_mutable


class SharedLhsMM(CustomGraphPass):
    def __call__(self, graph):
        groups = {}
        for n in graph.nodes:
            if n.op == 'call_function' and n.target == aten.mm.default:
                x = n.args[0]
                groups.setdefault((_base(x), tuple(x.meta['val'].shape)), []).append(n)
        for mms in groups.values():
            if len(mms) < 2:
                continue
            #FIX FOR: positions taken once went stale after an earlier group moved nodes up; a later group then
            # "moved" such a node down past its users (N18 polyzono/deeppoly_add: used before defined).
            order = {n: i for i, n in enumerate(graph.nodes)}
            first = min(mms, key=order.get)
            # Weights built after the first mm move before it, if they only depend on earlier nodes.
            late, ok = [], True

            def visit(n):
                nonlocal ok
                if n.op != 'call_function' or order[n] < order[first] or n in late:
                    return
                if n in mms or not _pure(n):
                    ok = False
                    return
                for a in n.all_input_nodes:
                    visit(a)
                late.append(n)
            for m in mms:
                visit(m.args[1])
            if not ok:
                continue
            for n in sorted(late, key=order.get):
                first.prepend(n)
            with graph.inserting_before(first):
                cat = graph.call_function(aten.cat.default, ([m.args[1] for m in mms], 1))
                big = graph.call_function(aten.mm.default, (first.args[0], cat))
                col = 0
                for m in mms:
                    width = m.args[1].meta['val'].shape[1]
                    m.replace_all_uses_with(graph.call_function(aten.slice.Tensor, (big, 1, col, col + width)))
                    col += width
            for m in mms:
                graph.erase_node(m)
        graph.lint()

    def uuid(self):
        return get_hash_for_files((__file__,))
