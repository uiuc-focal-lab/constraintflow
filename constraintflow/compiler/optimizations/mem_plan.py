"""
Reorder and rematerialize the functional flow block to fit a memory budget.

Zonotope-style flows keep one large coefficient tensor per noise-symbol group
and layer. Emitted in layer order, every group of a layer is live before the
first group of the next is consumed, and a residual block holds its input,
the first convolution and the second convolution at the same time. On
ResNet18/34 that exceeds an 80 GB GPU although each group is independent.

Rewrites, all value-preserving (no arithmetic is reassociated):

  * fuse_reduction_inputs: a pointwise value read only by early reductions is
    inlined into each of them, so it is never a named, materialized tensor.
  * inline_constants: tensors of ones/zeros hoisted by CSE are recreated at
    each reader instead of staying live between them.
  * schedule: depth-first topological order from the flow's outputs; after
    each statement, any ready statement that frees at least as much as it
    allocates runs immediately. This processes one group's chain at a time
    and releases inputs as soon as their last reader runs.
  * remat: a large value read by reductions now and by later consumers is
    recomputed right before them, when every input of its producer chain is
    still alive there anyway (a residual block's input feeding its first
    convolution and its skip connection).

Remat trades compute for memory, so it runs only while the estimated peak is
over budget, a batch of the largest values live at over-budget points per
round, keeping the best schedule seen. What still does not fit runs over
budget.

Requires functional SSA (after early reductions); sizes come from a
flow_shapes probe of the same block.
"""

import copy

from constraintflow.compiler.ir import (
    IrAssignment, IrBlockClamp, IrConst, IrSimpleBinary, IrSimpleUnary,
    IrTensorClamp, IrTensorOnes, IrTorchExpand, IrTorchSqueeze, IrTorchUnsqueeze,
    IrTorchWhere, IrTorchZeros, IrTransRetBasic, IrVar,
)
from constraintflow.compiler.optimizations import flow_split

_REDUCTION_PREFIX = 'early_sum_'
# Below this a value is not worth recomputing.
REMAT_MIN_BYTES = 64 * 1024 ** 2
# A statement that allocates at most this much more than it frees runs eagerly.
EAGER_BYTES = 16 * 1024 ** 2
MAX_REMAT_ROUNDS = 1000
REMAT_BATCH = 8

_POINTWISE = (IrTensorClamp, IrBlockClamp, IrSimpleBinary, IrSimpleUnary,
              IrTorchWhere, IrTorchExpand, IrTorchUnsqueeze, IrTorchSqueeze)


def _gb(n):
    return '{:.1f} GB'.format(n / 1024 ** 3)


def _is_reduction(name):
    return name is not None and name.startswith(_REDUCTION_PREFIX)


def _clone(expr, rename):
    """Structural copy of an expression tree, renaming variables in `rename`."""
    if isinstance(expr, list):
        return [_clone(e, rename) for e in expr]
    if isinstance(expr, IrVar):
        var = copy.copy(expr)
        var.name = rename.get(expr.name, expr.name)
        return var
    if not hasattr(expr, 'children'):
        return expr
    out = copy.copy(expr)
    out.update_parent_child([_clone(c, rename) for c in expr.children])
    if isinstance(out, IrSimpleUnary) and isinstance(out.op, IrVar):
        out.op = _clone(out.op, rename)
    return out


def _pointwise(expr):
    if isinstance(expr, (IrVar, IrConst, int, float)) or expr is None:
        return True
    if not isinstance(expr, _POINTWISE):
        return False
    if isinstance(expr, IrSimpleUnary) and isinstance(expr.op, IrVar):
        return False
    return all(_pointwise(c) for c in expr.children)


def fuse_reduction_inputs(block):
    """Inline pointwise values whose only readers are early reductions."""
    stmts = block.children

    def inline(name, expr, readers):
        return (not _is_reduction(name) and _pointwise(expr) and readers
                and all(_is_reduction(flow_split._def_name(stmts[k])) for k in readers))
    return _inline_where(block, inline)


def inline_constants(block):
    """Inline tensors of ones/zeros (hoisted by CSE) into every reader.

    A hoisted constant is a full-size tensor that stays live from its first to
    its last reader; recreating it per reader costs nothing once fused.
    """
    return _inline_where(block, lambda name, expr, readers:
                         readers and isinstance(expr, (IrTensorOnes, IrTorchZeros)))


def _inline_where(block, predicate):
    stmts = block.children
    users = {}
    for i, stmt in enumerate(stmts):
        for name in set(flow_split._reads(stmt)):
            users.setdefault(name, []).append(i)
    drop, inlined = set(), {}
    for i, stmt in enumerate(stmts):
        name = flow_split._def_name(stmt)
        if name is None or not isinstance(stmt, IrAssignment):
            continue
        if predicate(name, stmt.children[1], users.get(name, [])):
            inlined[name] = stmt.children[1]
            drop.add(i)
    if not inlined:
        return 0
    out = []
    for i, stmt in enumerate(stmts):
        if i in drop:
            continue
        reads = set(flow_split._reads(stmt))
        if reads & inlined.keys():
            stmt = copy.copy(stmt)
            if isinstance(stmt, IrAssignment):
                stmt.update_parent_child([stmt.children[0],
                                          _substitute(stmt.children[1], inlined)])
            else:
                stmt.update_parent_child(_substitute(list(stmt.children), inlined))
        out.append(stmt)
    block.update_parent_child(out)
    return len(drop)


def _substitute(expr, values):
    """Replace variables by (copies of) expressions, recursively."""
    if isinstance(expr, list):
        return [_substitute(e, values) for e in expr]
    if isinstance(expr, IrVar):
        if expr.name in values:
            return _substitute(_clone(values[expr.name], {}), values)
        return expr
    if not hasattr(expr, 'children'):
        return expr
    out = copy.copy(expr)
    out.update_parent_child([_substitute(c, values) for c in expr.children])
    return out


class _Model:
    """Byte sizes by storage root, keyed by value name so statements may move."""

    def __init__(self, stmts, sized):
        self.root_of, self.size, self.temp = {}, {}, {}
        self.inputs = set()
        for name, v in getattr(sized, 'inputs', {}).items():
            self.root_of[name] = v['root']
            self.size[v['root']] = v['storage_bytes']
            self.inputs.add(v['root'])
        self.alloc = {}                         # defined name -> root it allocates
        for i, stmt in enumerate(stmts):
            v = sized.get(i)
            name = flow_split._def_name(stmt)
            if v is None or name is None:
                continue
            self.root_of[name] = v['root']
            self.size[v['root']] = v.get('storage_bytes', v['nbytes'])
            self.temp[name] = v.get('temporary_peak', 0)
            if v['alloc']:
                self.alloc[name] = v['root']

    def add_clone(self, name, original):
        if original in self.alloc:
            root = original + '@' + name
            self.alloc[name] = root
            self.root_of[name] = root
            self.size[root] = self.size[self.alloc[original]]
        elif original in self.root_of:
            self.root_of[name] = self.root_of[original]
        self.temp[name] = self.temp.get(original, 0)


class _Graph:
    def __init__(self, stmts, model):
        self.stmts = stmts
        self.n = len(stmts)
        self.names = [flow_split._def_name(s) for s in stmts]
        self.index = {name: i for i, name in enumerate(self.names) if name is not None}
        self.reads = [sorted(set(flow_split._reads(s))) for s in stmts]
        self.deps = [sorted({self.index[x] for x in r if x in self.index} - {i})
                     for i, r in enumerate(self.reads)]
        self.succ = [[] for _ in range(self.n)]
        for i, ds in enumerate(self.deps):
            for j in ds:
                self.succ[j].append(i)
        self.roots = [sorted({model.root_of[x] for x in r if x in model.root_of})
                      for r in self.reads]
        self.alloc = [model.alloc.get(name) for name in self.names]
        self.temp = [model.temp.get(name, 0) for name in self.names]


def simulate(graph, model, order, budget=None):
    """(peak bytes incl. statement temporaries, position of the peak, live roots).

    With `budget`, the roots are those live anywhere the estimate exceeds it.
    """
    uses = {}
    for i in order:
        for r in graph.roots[i]:
            uses[r] = uses.get(r, 0) + 1
    alive = {r for r in model.inputs if uses.get(r, 0)}
    live = sum(model.size.get(r, 0) for r in alive)
    best = (-1, 0, ())
    over = set()
    for k, i in enumerate(order):
        # as flow_shapes.peak_live_bytes: live entering i plus i's own temporaries
        if live + graph.temp[i] > best[0]:
            best = (live + graph.temp[i], k, tuple(alive))
        if budget is not None and live + graph.temp[i] > budget:
            over |= alive
        a = graph.alloc[i]
        if a is not None and a not in alive and uses.get(a, 0):
            alive.add(a)
            live += model.size.get(a, 0)
            if live > best[0]:
                best = (live, k, tuple(alive))
        for r in graph.roots[i]:
            uses[r] -= 1
            if uses[r] == 0 and r in alive:
                alive.discard(r)
                live -= model.size.get(r, 0)
    if budget is not None:
        return best[0], best[1], tuple(over)
    return best


def schedule(graph, model, position):
    """Depth-first order from the outputs with eager release; see module doc."""
    n = graph.n
    uses = {}
    readers = {}
    for i in range(n):
        for r in graph.roots[i]:
            uses[r] = uses.get(r, 0) + 1
            readers.setdefault(r, []).append(i)
    done = [False] * n
    pending = [len(d) for d in graph.deps]
    alive = set(model.inputs)
    out = []

    def gain(i):
        a = graph.alloc[i]
        grow = model.size.get(a, 0) if a is not None and a not in alive else 0
        return grow - sum(model.size.get(r, 0) for r in graph.roots[i]
                          if uses.get(r, 0) == 1 and r in alive)

    def emit(first):
        work = [first]
        while work:
            i = work.pop()
            if done[i] or pending[i]:
                continue
            done[i] = True
            out.append(i)
            if graph.alloc[i] is not None:
                alive.add(graph.alloc[i])
            ready = []
            for r in graph.roots[i]:
                uses[r] -= 1
                if uses[r] == 0:
                    alive.discard(r)
                elif uses[r] == 1:
                    ready.extend(j for j in readers[r] if not done[j])
            for j in graph.succ[i]:
                pending[j] -= 1
                if pending[j] == 0:
                    ready.append(j)
            # pop order: earliest original position first
            for j in sorted(set(ready), key=lambda j: -position[j]):
                if not done[j] and not pending[j] and gain(j) <= EAGER_BYTES:
                    work.append(j)

    def deps_of(i):
        return iter(sorted(graph.deps[i], key=lambda j: position[j]))

    returns = [i for i in range(n) if isinstance(graph.stmts[i], IrTransRetBasic)]
    sinks = sorted((i for i in range(n) if not graph.succ[i] and i not in returns),
                   key=lambda i: position[i]) + returns
    for sink in sinks:
        stack = [(sink, deps_of(sink))]
        while stack:
            node, it = stack[-1]
            if done[node]:
                stack.pop()
                continue
            nxt = next((j for j in it if not done[j]), None)
            if nxt is None:
                stack.pop()
                emit(node)
            else:
                stack.append((nxt, deps_of(nxt)))
    if len(out) != n:
        raise RuntimeError('mem-plan: schedule dropped statements')
    return out


def _remat_candidates(graph, model, order, live_roots):
    """Values live at the peak whose producer chain can be recomputed for free."""
    pos = {i: k for k, i in enumerate(order)}
    live = set(live_roots)
    found = []
    for j in range(graph.n):
        name, a = graph.names[j], graph.alloc[j]
        if a is None or a not in live or _is_reduction(name) or '_remat' in name:
            continue
        if model.size.get(a, 0) < REMAT_MIN_BYTES or not isinstance(graph.stmts[j], IrAssignment):
            continue
        late = [k for k in graph.succ[j] if not _is_reduction(graph.names[k])]
        if not late:
            continue
        first_late = min(pos[k] for k in late)
        # The producer chain: j plus single-reader intermediates feeding only it.
        chain, frontier = {j}, list(graph.deps[j])
        while frontier:
            p = frontier.pop()
            if p in chain or graph.alloc[p] is None:
                continue
            if all(s in chain for s in graph.succ[p]):
                chain.add(p)
                frontier.extend(graph.deps[p])
        inputs = {d for c in chain for d in graph.deps[c]} - chain
        if any(not any(pos[s] > first_late for s in graph.succ[d] if s not in chain)
               for d in inputs):
            continue
        if any(pos[c] > first_late for c in chain):
            continue
        found.append((model.size[a], j, sorted(chain), late))
    found.sort(key=lambda t: -t[0])
    return found


def _apply_remat(stmts, graph, model, chain, late, counter):
    """Clone `chain` and point the late readers at the clone of its result."""
    rename = {}
    for c in chain:
        name = graph.names[c]
        new = name + '_remat' + str(counter)
        while new in graph.index or new in rename.values():
            counter += 1
            new = name + '_remat' + str(counter)
        rename[name] = new
    clones = []
    for c in sorted(chain):
        stmt = stmts[c]
        clone = copy.copy(stmt)
        clone.update_parent_child([_clone(stmt.children[0], rename),
                                   _clone(stmt.children[1], rename)])
        clones.append(clone)
        model.add_clone(rename[graph.names[c]], graph.names[c])
    out = list(stmts)
    for k in late:
        stmt = copy.copy(out[k])
        if isinstance(stmt, IrAssignment):
            stmt.update_parent_child([stmt.children[0], _clone(stmt.children[1], rename)])
        else:
            stmt.update_parent_child(_clone(list(stmt.children), rename))
        out[k] = stmt
    # Insert the clones just before the first late reader (scheduling reorders anyway).
    first = min(late)
    return out[:first] + clones + out[first:], counter + 1


def run(block, sized, budget):
    """Returns a log line, or None when the block already fits `budget` bytes."""
    stmts = list(block.children)
    if not all(isinstance(s, (IrAssignment, IrTransRetBasic)) for s in stmts):
        return 'skipped: the flow block is not functional SSA'
    model = _Model(stmts, sized)
    graph = _Graph(stmts, model)
    before = simulate(graph, model, list(range(graph.n)))[0]
    if before <= budget:
        return None
    fused = fuse_reduction_inputs(block)
    constants = inline_constants(block)
    stmts = list(block.children)
    graph = _Graph(stmts, model)
    position = list(range(graph.n))
    order = schedule(graph, model, position)
    peak, at, live = simulate(graph, model, order, budget)
    best = (peak, stmts, order, 0)
    rounds, counter, rematted = 0, 0, 0
    while peak > budget and rounds < MAX_REMAT_ROUNDS:
        rounds += 1
        candidates = _remat_candidates(graph, model, order, live)
        if not candidates:
            break
        # Statements in scheduled order; candidates are disjoint chains, so
        # apply a batch of the largest before rescheduling.
        ordered = [stmts[i] for i in order]
        pos = {i: k for k, i in enumerate(order)}
        original = list(ordered)
        for _, j, chain, late in candidates[:REMAT_BATCH]:
            where = {id(st): k for k, st in enumerate(ordered)}
            chain_s = [original[pos[c]] for c in chain]
            late_s = [original[pos[k]] for k in late]
            if any(id(st) not in where for st in chain_s + late_s):
                continue        # rewritten by an earlier remat in this batch
            ordered, counter = _apply_remat(ordered, _Graph(ordered, model), model,
                                            sorted(where[id(st)] for st in chain_s),
                                            [where[id(st)] for st in late_s], counter)
            rematted += 1
        stmts = ordered
        graph = _Graph(stmts, model)
        order = schedule(graph, model, list(range(graph.n)))
        peak, at, live = simulate(graph, model, order, budget)
        if peak < best[0]:
            best = (peak, stmts, order, rematted)
    peak, stmts, order, rematted = best
    block.update_parent_child([stmts[i] for i in order])
    return ('{} -> {} estimated peak ({} budget), {} reduction inputs fused, '
            '{} constants inlined, {} values rematerialized'.format(
                _gb(before), _gb(peak), _gb(budget), fused, constants, rematted))
