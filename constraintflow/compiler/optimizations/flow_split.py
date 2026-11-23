"""
Cut the scalarized flow block into separately-emitted functions.

Requires sroa: ir.flow_block is one straight-line SSA block without mutations, so
any cut point is valid and the values crossing it are exactly those defined
before it and still read after it.
"""

from constraintflow.compiler.ir import *
from constraintflow.compiler.optimizations import subexp_inlining

def _mb(n):
    return '{:.1f} MB'.format(n / 1024 ** 2)


class Segment:
    def __init__(self, index, stmts, live_in, live_out):
        self.index = index
        self.stmts = stmts
        self.live_in = live_in
        self.live_out = live_out

    @property
    def name(self):
        return 'flow_seg_' + str(self.index)


def _reads(stmt):
    """Names a statement reads; an assignment's lhs is a def, not a read."""
    if isinstance(stmt, IrDel):
        return []
    src = [stmt.children[1]] if isinstance(stmt, IrAssignment) else list(stmt.children)
    return [v.name for v in subexp_inlining.get_vars_expr_occurrences(src)]


_INPLACE = (IrAssignToView, IrAssignToBlock, IrSetBlockTotalShapeLastDim)


def _mutates(stmt):
    """Target of an in-place write; both read and written, so it must cross out."""
    if isinstance(stmt, _INPLACE) and isinstance(stmt.children[0], IrVar):
        return stmt.children[0].name
    return None


def _def_name(stmt):
    if isinstance(stmt, IrAssignment) and isinstance(stmt.children[0], IrVar):
        return stmt.children[0].name
    return None


def liveness(stmts):
    """First def index and last use index per name."""
    def_at, last_use = {}, {}
    for i, stmt in enumerate(stmts):
        for name in _reads(stmt):
            last_use[name] = i
        name = _def_name(stmt)
        if name is not None and name not in def_at:
            def_at[name] = i
    return def_at, last_use


def root_liveness(stmts, sized):
    """Bytes, first def and last use per storage root, from a flow_shapes probe."""
    inputs = getattr(sized, 'inputs', {})
    root_of = {name: v['root'] for name, v in inputs.items()}
    size_of = {v['root']: v['storage_bytes'] for v in inputs.values()}
    def_at, last_use = {root: -1 for root in size_of}, {}
    for i, stmt in enumerate(stmts):
        for name in _reads(stmt):
            last_use[root_of.get(name, name)] = i
        v = sized.get(i)
        if v is not None:
            root_of[v['name']] = v['root']
            size_of[v['root']] = v.get('storage_bytes', v['nbytes'])
            def_at.setdefault(v['root'], -1 if v.get('input') else i)
    return size_of, def_at, last_use


def crossing(def_at, last_use, cut):
    """Names defined before `cut` and still read at or after it."""
    return [n for n, d in def_at.items()
            if d < cut and last_use.get(n, -1) >= cut]


def live_curve(stmts, def_at, last_use, size_of=None):
    """Bytes live entering each point, or value count when `size_of` is None.

    out[i] is exactly what crossing(def_at, last_use, i) selects: defined before i
    and still read at or after it.
    """
    born, dies = {}, {}
    for name, d in def_at.items():
        born.setdefault(d, []).append(name)
        dies.setdefault(max(last_use.get(name, d), d), []).append(name)
    out = [0] * (len(stmts) + 1)
    held = sum(size_of.get(name, 0) if size_of is not None else 1
               for name, d in def_at.items() if d < 0 and last_use.get(name, -1) >= 0)
    for i in range(len(stmts) + 1):
        if i > 0:
            for name in born.get(i - 1, ()):
                held += size_of.get(name, 0) if size_of is not None else 1
            for name in dies.get(i - 1, ()):
                held -= size_of.get(name, 0) if size_of is not None else 1
        out[i] = held
    return out


def live_bytes(stmts, size_of, def_at, last_use):
    """Bytes held entering each program point."""
    return live_curve(stmts, def_at, last_use, size_of)


def allocation_cuts(stmts, sized, budget):
    """Bound new named storage per region, not total live memory or workspace.

    Small regions keep Inductor from fusing early reductions all the way back
    into the final traversal expression. A single statement remains indivisible.
    """
    cuts, seen, held, start = [], set(), 0, 0
    for i, stmt in enumerate(stmts):
        value = sized.get(i)
        cost = 0
        if value is not None and value['root'] not in seen:
            seen.add(value['root'])
            cost = 0 if value.get('input') else value.get('storage_bytes', value['nbytes'])
        if i > start and held + cost > budget:
            cuts.append(i)
            held, start = 0, i
        held += cost
    return cuts


def split(block, sized=None, max_region_bytes=0, barriers=()):
    """Partition functional SSA by newly allocated storage, with no statement cap."""
    if max_region_bytes <= 0:
        return None
    stmts = block.children
    names = [_def_name(s) for s in stmts if _def_name(s) is not None]
    if len(names) != len(set(names)):
        raise ValueError('flow splitting requires SSA; partition before recycling names')
    if not sized:
        raise ValueError('--flow-segment-mb requires a successful shape probe')
    cuts = set(allocation_cuts(stmts, sized, max_region_bytes))
    cuts |= {i + 1 for i, s in enumerate(stmts[:-1]) if _def_name(s) in barriers}
    cuts = sorted(cuts)
    if not cuts:
        return None
    def_at, last_use = liveness(stmts)
    bounds = [0] + cuts + [len(stmts)]
    segments = []
    for i in range(len(bounds) - 1):
        start, end = bounds[i], bounds[i + 1]
        body = stmts[start:end]
        live_in, seen = [], set()
        for stmt in body:
            for name in _reads(stmt):
                if name in seen:
                    continue
                seen.add(name)
                # defined in an earlier segment, or never here -- a flow param
                defined = def_at.get(name)
                if defined is None or defined < start:
                    live_in.append(name)
        out = {n for n, d in def_at.items()
               if start <= d < end and last_use.get(n, -1) >= end}
        for stmt in body:                      # mutations must leave the segment
            name = _mutates(stmt)
            if name is not None and last_use.get(name, -1) >= end:
                out.add(name)
        live_out = sorted(out)
        segments.append(Segment(i, body, live_in, live_out))
    return segments


def describe(segments):
    """One-line summary per segment, for the compile log."""
    return ', '.join('{}({} stmts, {} in, {} out)'.format(
        s.index, len(s.stmts), len(s.live_in), len(s.live_out)) for s in segments)
