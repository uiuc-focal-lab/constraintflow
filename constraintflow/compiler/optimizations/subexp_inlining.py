"""
Def-use based subexpression inlining on non-SSA IR.

If an intermediate variable is used exactly once, inline its definition at the
use site and delete the variable. Python releases locals only at function end,
so folding the value frees the intermediate's memory earlier.
"""

from __future__ import annotations
import bisect
import heapq
from constraintflow.compiler.ir import *
from constraintflow.compiler.representations import Graph


#FIX FOR: reuse.cf on 9-layer nets (N5/N7/N21) failed to compile: inlining built one statement
# nested 258 parentheses deep, past CPython's parser limit of 200.
# Inlining stops where the use site would nest deeper than this many IR levels; the
# value then stays a named temporary (same ops, same order). CPython's parser rejects
# more than 200 nested parentheses, and reuse.cf on 9-layer nets otherwise sums 2**9
# back-substitution terms into one 258-deep statement. Wrapper nodes render no
# parentheses (N4 reuse: IR depth > 64 renders 35 deep), so the cap sits well above
# what normal programs reach and leaves their code unchanged.
MAX_INLINE_DEPTH = 150
depth_capped = [0]   # uses left un-inlined by the cap, reported by the caller


def expr_depth(expr) -> int:
    """Nesting depth of an expression tree (a leaf is 1)."""
    if expr is None or isinstance(expr, (int, float, str)):
        return 0
    depth, stack = 0, [(expr, 1)]
    while stack:
        node, d = stack.pop()
        depth = max(depth, d)
        for child in get_generalized_children(node):
            if child is not None and not isinstance(child, (int, float, str)):
                stack.append((child, d + 1))
    return depth


def _stmt_depth(stmt) -> int:
    operands = [stmt.children[1]] if isinstance(stmt, IrAssignment) else stmt.children
    return max((expr_depth(o) for o in operands), default=0)


def get_generalized_children(expr) -> list[IrExpression]:
    ret: list[IrExpression] = []
    if isinstance(expr, IrSimpleUnary) and isinstance(expr.op, IrVar):
        ret.append(expr.op)
        ret.extend(expr.children)
    else:
        ret.extend(expr.children)
    return ret


def replace_all_occurrences_expr(expr, var_map: dict[str, IrExpression]):
    if isinstance(expr, IrVar) and expr.name in var_map.keys(): 
        return var_map[expr.name]
    if isinstance(expr, (int, float)):
        return expr
    if expr is None:
        return expr
    if isinstance(expr, list):
        return [replace_all_occurrences_expr(x, var_map) for x in expr]
    has_generalized_op_child = (
        isinstance(expr, IrSimpleUnary) and isinstance(expr.op, IrVar))
    generalized_children = get_generalized_children(expr)
    for i in range(len(generalized_children)):
        new_child = replace_all_occurrences_expr(generalized_children[i], var_map)
        if has_generalized_op_child and i == 0:
            expr.op = new_child
        elif has_generalized_op_child and i != 0:
            expr.children[i - 1] = new_child
        else:
            expr.children[i] = new_child
    return expr


def replace_var_with_expr(             
        instructions: list[IrStatement], use_instr_index: int,
        replace_expr: IrExpression, var: IrVar) -> None:
    var_name = var.name
    if isinstance(instructions[use_instr_index], IrAssignment):
        new_expr = replace_all_occurrences_expr(
            instructions[use_instr_index].children[1],
            {var_name: replace_expr})
        new_children = [instructions[use_instr_index].children[0], new_expr]
        instructions[use_instr_index].update_parent_child(new_children)
    elif isinstance(
        instructions[use_instr_index], IrTransRetBasic):
        new_children = []
        for j in range(len(instructions[use_instr_index].children)):
            new_expr = replace_all_occurrences_expr(
            instructions[use_instr_index].children[j],
            {var_name: replace_expr})
            new_children.append(new_expr)
        instructions[use_instr_index].update_parent_child(new_children)
    elif isinstance(instructions[use_instr_index], IrAssignToView):
        new_children = []
        for j in range(len(instructions[use_instr_index].children)):
            new_expr = replace_all_occurrences_expr(
            instructions[use_instr_index].children[j],
            {var_name: replace_expr})
            new_children.append(new_expr)
        instructions[use_instr_index].update_parent_child(new_children)
    elif isinstance(instructions[use_instr_index], IrAssignToBlock):
        new_children = []
        for j in range(len(instructions[use_instr_index].children)):
            new_expr = replace_all_occurrences_expr(
            instructions[use_instr_index].children[j],
            {var_name: replace_expr})
            new_children.append(new_expr)
        instructions[use_instr_index].update_parent_child(new_children)
    elif isinstance(instructions[use_instr_index], IrSetBlockTotalShapeLastDim):
        new_children = []
        for j in range(len(instructions[use_instr_index].children)):
            new_expr = replace_all_occurrences_expr(
            instructions[use_instr_index].children[j],
            {var_name: replace_expr})
            new_children.append(new_expr)
        instructions[use_instr_index].update_parent_child(new_children)
    else:
        assert False, f'Unexpected instruction type: {type(instructions[use_instr_index])}'


def compute_def_indices(
        instructions: list[IrStatement]) -> dict[str, list[int]]:
    """Each variable to the ascending indices that assign it. On a straight-line
    block the reaching definition at any point is the nearest preceding one."""
    def_indices: dict[str, list[int]] = {}
    for idx, instr in enumerate(instructions):
        if isinstance(instr, IrAssignment):
            lhs = instr.children[0]
            if isinstance(lhs, IrVar):
                def_indices.setdefault(lhs.name, []).append(idx)
    return def_indices

_FRESH_NODE_TYPES = (
    IrTorchZeros, IrTorchEye, IrTensorOnes,
    IrTorchMatmul, IrBlockInnerProduct,
    IrSimpleBinary, IrBlockBinaryOp,
    IrTorchSum, IrTorchRepeat, IrTensorRepeat,
    IrTorchDiagEmbed, IrTorchWhere, IrBlockWhereBlock, IrTensorScatter, IrPatchesToDense,
    IrTensorClamp, IrBlockClamp, IrConvertBoolToFloat,
    IrFConv2d, IrFConvTranspose2d, IrFUnfold,
    IrTorchStride, IrBlockGetDims, IrTorchPad, IrTorchCat,
    IrBlockAll, IrBlockAny, IrBlockCopy,
    IrEmptyList,
)


def _resolve_storage(expr, reads_of: dict[str, frozenset]) -> frozenset:
    """Storage roots `expr`'s result reads. A FRESH node reads none (its result is
    newly allocated); a bare var is its own root; everything else passes through
    to its operands' roots."""
    if isinstance(expr, IrVar):
        return reads_of.get(expr.name, frozenset((expr.name,)))
    if expr is None or isinstance(expr, (int, float, IrConst)):
        return frozenset()
    if isinstance(expr, list):
        out: set = set()
        for x in expr:
            out |= _resolve_storage(x, reads_of)
        return frozenset(out)
    if isinstance(expr, _FRESH_NODE_TYPES):
        return frozenset()
    out = set()
    for child in get_generalized_children(expr):
        out |= _resolve_storage(child, reads_of)
    return frozenset(out)


def _expr_eval_roots(expr, reads_of: dict[str, frozenset]) -> frozenset:
    """Storage roots that *evaluating* `expr` depends on. Unlike `_resolve_storage`,
    this recurses into FRESH nodes too: allocating new storage doesn't mean reading
    the operands was free, so mutating one before a relocated evaluation would still
    change the result."""
    if isinstance(expr, IrVar):
        return reads_of.get(expr.name, frozenset((expr.name,)))
    if expr is None or isinstance(expr, (int, float, IrConst)):
        return frozenset()
    if isinstance(expr, list):
        out: set = set()
        for x in expr:
            out |= _expr_eval_roots(x, reads_of)
        return frozenset(out)
    out = set()
    for child in get_generalized_children(expr):
        out |= _expr_eval_roots(child, reads_of)
    return frozenset(out)


def compute_storage_reads_and_defs(instructions: list[IrStatement]):
    """What each value reads, and where storage is mutated in place."""
    reads_of: dict[str, frozenset] = {}
    for instr in instructions:
        if isinstance(instr, IrAssignment):
            lhs = instr.children[0]
            if isinstance(lhs, IrVar):
                roots = _resolve_storage(instr.children[1], reads_of)
                if not roots:                    # fresh allocation: own root
                    roots = frozenset((lhs.name,))
                prev = reads_of.get(lhs.name)
                reads_of[lhs.name] = roots if prev is None else (prev | roots)
    storage_defs: list = []
    for idx, instr in enumerate(instructions):
        if isinstance(instr, (IrAssignToView, IrAssignToBlock, IrSetBlockTotalShapeLastDim)):
            # an in-place write defs the storage its target touches
            roots = _resolve_storage(instr.children[0], reads_of)
            if roots:
                storage_defs.append((idx, roots))
    storage_def_keys = [idx for idx, _ in storage_defs]
    return reads_of, storage_defs, storage_def_keys


def storage_redefd_between(
        storage_defs: list, storage_def_keys: list, read_roots: set,
        lo: int, hi: int) -> bool:
    """Storage-space analog of `redefined_between`: True iff some in-place write in
    (lo, hi) defs a root in `read_roots`, i.e. the value goes stale before its use."""
    if not read_roots:
        return False
    k = bisect.bisect_right(storage_def_keys, lo)
    n = len(storage_defs)
    while k < n:
        idx, roots = storage_defs[k]
        if idx >= hi:
            return False
        if roots & read_roots:
            return True
        k += 1
    return False


def reaching_def_index(
        def_indices: dict[str, list[int]], name: str, point: int):
    """Index of the definition of `name` strictly before `point`, or None if
    `name` is never defined in the block (e.g. a transformer parameter)."""
    idxs = def_indices.get(name)
    if not idxs:
        return None
    j = bisect.bisect_left(idxs, point) - 1
    return idxs[j] if j >= 0 else None


def redefined_between(
        def_indices: dict[str, list[int]], name: str, lo: int, hi: int) -> bool:
    """True iff `name` is assigned at some index in the open interval (lo, hi)."""
    idxs = def_indices.get(name)
    if not idxs:
        return False
    k = bisect.bisect_right(idxs, lo)
    return k < len(idxs) and idxs[k] < hi


def resolve_value(
        instructions: list[IrStatement], def_indices: dict[str, list[int]],
        var: IrVar, point: int):
    """(expr, def_index) for `var`'s value at `point`; def_index is -1 for a
    block-external variable.

    Copy chains (`t = a; a = expr`) are followed through reaching definitions. Each
    hop moves strictly earlier, so this terminates and binds the right definition
    even for reassigned variables.
    """
    d = reaching_def_index(def_indices, var.name, point)
    if d is None:
        return var, -1
    rhs = instructions[d].children[1]
    if isinstance(rhs, IrVar):
        return resolve_value(instructions, def_indices, rhs, d)
    return rhs, d


def get_vars_expr_occurrences(expr) -> list[IrVar]:
    if isinstance(expr, (int, float)):
        return []
    if isinstance(expr, IrVar):
        return [expr]
    if isinstance(expr, list):
        vars: list[IrVar] = []
        for x in expr:
            vars.extend(get_vars_expr_occurrences(x))
        return vars
    vars: list[IrVar] = []
    if expr is None:
        return vars
    for child in get_generalized_children(expr):
        vars.extend(get_vars_expr_occurrences(child))
    return vars


def is_safe_to_inline(
        def_indices: dict[str, list[int]], reads_of: dict[str, frozenset],
        storage_defs: list, storage_def_keys: list, var: IrVar, expr,
        def_index: int, use_index: int) -> bool:
    """
    Moving `var`'s value `expr` from `def_index` to `use_index` preserves it iff
    nothing it depends on is def'd in that open interval -- the same test in both
    spaces: no variable `expr` reads is reassigned (`redefined_between`), and no
    in-place write mutates a storage root it reads (`storage_redefd_between`).

    Read roots are those of the resolved `expr` plus `reads_of[var]`, the storage the
    value *is*. Keying the latter on `var` keeps the alias case covered after
    `resolve_value` has collapsed the chain to a var-free `torch.zeros`.

    The storage check is skipped entirely when `expr` is a bare IrVar: substituting
    `var`'s use-site with a different *name for the same object*, at that same fixed
    program point, moves no evaluation and so cannot observe a value the original
    name would not also have observed -- regardless of any storage mutation between
    def_index and use_index. Only `redefined_between` (name rebinding, not in-place
    mutation) can matter for a bare-var substitution.
    """
    if not isinstance(expr, IrVar):
        read_roots: set = set(_expr_eval_roots(expr, reads_of))
        read_roots |= reads_of.get(var.name, frozenset((var.name,)))
        if storage_redefd_between(storage_defs, storage_def_keys, read_roots,
                                  def_index, use_index):
            return False
    for v in get_vars_expr_occurrences(expr):
        if redefined_between(def_indices, v.name, def_index, use_index):
            return False
    return True


_FIELD_READ_TYPES = (
    IrGetPolyExpSparseMat, IrGetPolyExpSparseConst,
    IrGetSymExpSparseMat, IrGetSymExpSparseConst,
    IrPolyExpMat, IrExtractPolyConst, IrExtractSymConst,
)


def is_trivial(expr) -> bool:
    """A leaf that is free to recompute. Only these may be folded into more than one
    use site; copying anything heavier would duplicate real work. In reuse mode the
    .mat/.const reads are __slots__ loads on the emitted data-only classes, so they
    are as cheap as a bare variable."""
    if isinstance(expr, (IrVar, IrConst)):
        return True
    return (isinstance(expr, _FIELD_READ_TYPES)
            and len(expr.children) == 1
            and is_trivial(expr.children[0]))


def copy_leaf(expr):
    """Fresh copy of a trivial leaf so several use sites never share one node object.
    Non-leaves (only ever inlined once) are returned unchanged."""
    if isinstance(expr, IrVar):
        return IrVar(expr.name, expr.irMetadata)
    if isinstance(expr, IrConst):
        return IrConst(expr.const, expr.irMetadata[-1].type)
    if isinstance(expr, _FIELD_READ_TYPES):
        clone = copy.copy(expr)
        clone.update_parent_child([copy_leaf(c) for c in expr.children])
        return clone
    return expr


def try_inline_definition(
        instructions: list[IrStatement], def_indices: dict[str, list[int]],
        reads_of: dict[str, frozenset], storage_defs: list,
        storage_def_keys: list, var: IrVar, def_stmt_index: int,
        use_indices: list[int], remat: frozenset = frozenset()) -> bool:
    """
    Fold `var`'s definition (at `def_stmt_index`, used at `use_indices`) into its
    use sites, reporting whether anything was rewritten:

      - 0 uses  -> nothing to fold; `drop_dead_assignments` collects it.
      - 1 use   -> inline if the move is value-preserving.
      - >1 uses -> only if the resolved value is a trivial leaf (free to
                   duplicate). When it is not but this definition is itself a
                   bare copy, propagate the copied variable instead:
                   `resolve_value` looks through it to the root value, and that
                   variable is a trivial leaf even when the root is not.
    """
    if not use_indices:
        return False
    inline_expr, value_def_index = resolve_value(
        instructions, def_indices, var, use_indices[0])
    if len(use_indices) > 1 and not is_trivial(inline_expr) and var.name not in remat:
        rhs = instructions[def_stmt_index].children[1]
        if not isinstance(rhs, IrVar):
            return False
        inline_expr, value_def_index = rhs, def_stmt_index
    substituted = False
    #FIX FOR: statements nested too deep for CPython's parser; skip inlining past MAX_INLINE_DEPTH.
    inline_depth = 0 if is_trivial(inline_expr) else expr_depth(inline_expr)
    for use_index in use_indices:
        if (inline_depth > 1
                and _stmt_depth(instructions[use_index]) + inline_depth - 1 > MAX_INLINE_DEPTH):
            depth_capped[0] += 1
            continue
        if is_safe_to_inline(def_indices, reads_of, storage_defs, storage_def_keys,
                             var, inline_expr, value_def_index, use_index):
            replace_var_with_expr(
                instructions, use_index, copy_leaf(inline_expr), var)
            substituted = True
    return substituted


def substitute_definitions(instructions: list[IrStatement],
                           remat: frozenset = frozenset()) -> bool:
    def_indices: dict[str, list[int]] = compute_def_indices(instructions)
    reads_of, storage_defs, storage_def_keys = compute_storage_reads_and_defs(
        instructions)
    current_vars_def_index: dict[str, int] = {}
    uses_instr_count: dict[str, list[int]] = {}
    substituted = False
    name_to_var: dict[str, IrVar] = {}
    for i in range(len(instructions)):
        if isinstance(instructions[i], IrDel):
            continue
        used_vars: list[str]
        temp: list[IrVar]
        if isinstance(instructions[i], IrAssignment):
            temp = get_vars_expr_occurrences(instructions[i].children[1])
        elif isinstance(instructions[i], IrTransRetBasic):
            temp = []
            for j in range(len(instructions[i].children)):
                temp.extend(
                    get_vars_expr_occurrences(instructions[i].children[j]))
        elif isinstance(instructions[i], IrAssignToView):
            temp = []
            for j in range(len(instructions[i].children)):
                temp.extend(
                    get_vars_expr_occurrences(instructions[i].children[j]))
        elif isinstance(instructions[i], IrAssignToBlock):
            temp = []
            for j in range(len(instructions[i].children)):
                temp.extend(
                    get_vars_expr_occurrences(instructions[i].children[j]))
        elif isinstance(instructions[i], IrSetBlockTotalShapeLastDim):
            temp = []
            for j in range(len(instructions[i].children)):
                temp.extend(
                    get_vars_expr_occurrences(instructions[i].children[j]))
        else:
            assert False, f'Unexpected instruction type: {type(instructions[i])}'
        used_vars = [var.name for var in temp]
        for var in used_vars:
            if var not in current_vars_def_index.keys():
                continue
            if var not in uses_instr_count.keys():
                uses_instr_count[var] = [i]
            else:
                uses_instr_count[var].append(i)
        if isinstance(instructions[i], IrAssignment):
            defined_var = instructions[i].children[0]
            name_to_var[defined_var.name] = defined_var
            assert isinstance(defined_var, IrVar)
            if defined_var.name in current_vars_def_index.keys():
                substituted |= try_inline_definition(
                    instructions, def_indices, reads_of, storage_defs,
                    storage_def_keys, defined_var,
                    current_vars_def_index[defined_var.name],
                    uses_instr_count[defined_var.name], remat)
            current_vars_def_index[defined_var.name] = i
            uses_instr_count[defined_var.name] = []

    for var in current_vars_def_index.keys():
        substituted |= try_inline_definition(
            instructions, def_indices, reads_of, storage_defs, storage_def_keys,
            name_to_var[var], current_vars_def_index[var],
            uses_instr_count.get(var, []), remat)

    return substituted


def drop_dead_assignments(instructions: list[IrStatement]) -> bool:
    """Delete every assignment whose target is never read, against a use map built
    fresh from `instructions`. A name assigned more than once is kept if any of its
    definitions is read, since one live use keeps the whole name live."""
    used: set[str] = set()
    for instr in instructions:
        if isinstance(instr, IrDel):
            continue
        if isinstance(instr, IrAssignment):
            operands = [instr.children[1]]
        else:
            operands = instr.children
        for operand in operands:
            for v in get_vars_expr_occurrences(operand):
                used.add(v.name)
    kept: list[IrStatement] = []
    for instr in instructions:
        if (isinstance(instr, IrAssignment)
                and isinstance(instr.children[0], IrVar)
                and instr.children[0].name not in used):
            continue
        kept.append(instr)
    dropped = len(kept) != len(instructions)
    instructions[:] = kept
    return dropped


def inline_fixpoint(instructions: list[IrStatement],
                    remat: frozenset = frozenset()) -> None:
    """Iterate to a fixpoint, in place so a caller holding the list sees the result.

    Substitution and deletion are separate phases on purpose: substituting creates
    references the walk's use map does not know about, so deciding deletions from
    that same stale map drops definitions the round has just made live again.
    `drop_dead_assignments` rebuilds the map from the mutated list, so it is correct
    however the preceding phase rewrote things.
    """
    while True:
        substituted = substitute_definitions(instructions, remat)
        dropped = drop_dead_assignments(instructions)
        if not substituted and not dropped:
            break


def inline_subexp_block(block: IrBlock) -> None:
    """private"""
    #FIX FOR: reuse N5/N7/N21 compile failure (too-deep nesting); log uses kept by the depth cap.
    depth_capped[0] = 0
    inline_fixpoint(block.children)
    if depth_capped[0]:
        print('[inline] {} uses kept as temporaries: inlining would nest deeper than {} '
              'IR levels'.format(depth_capped[0], MAX_INLINE_DEPTH))


def rematerialize_block(block: IrBlock, names) -> None:
    """public
    Re-inline the named multi-use temporaries into their uses, trading recompute
    for the memory their long live ranges hold. Every substitution still goes
    through is_safe_to_inline."""
    if names:
        inline_fixpoint(block.children, frozenset(names))


def inline_subexp_cfg(cfg: Graph) -> None:
    """
    private
    Inline single-use temporaries within one transformer CFG.

    `tensor_to_block` collapses reuse-mode per-layer CFGs to a single straight-line
    block, so the common case is inlined directly. A CFG that still has control flow
    (e.g. an op that was not specialized) has no linearizable live path and is left
    unchanged.
    """
    if len(cfg.nodes) == 1:
        inline_subexp_block(cfg.ir[cfg.nodes[0]])


def inline_subexp(ir: IrProgram) -> None:
    """
    public
    Requires:
    - `ir` is already optimized to block level.
    - `tensor_to_block` has run: per-layer CFGs are collapsed to a single
      straight-line block, with control flow resolved along the profiled live
      path.
    """
    for transformer in ir.tstore.keys():
        for i in range(len(ir.tstore[transformer])):
            transformer_ir = ir.tstore[transformer][i]
            if transformer_ir.layerwise_cfgs is not None:
                for cfg in transformer_ir.layerwise_cfgs.values():
                    inline_subexp_cfg(cfg)
            else:
                inline_subexp_cfg(transformer_ir.cfg)


# Liveness-based name recycling (linear-scan register allocation).
def _temp_liveness(instructions, prefix):
    """
    Per `prefix` temporary: its def index, last-use index, and every IrVar node naming
    it, so the caller can rename in place. Names assigned more than once go in
    `multi_def` (no single live interval) and are left alone.
    """
    def_pos: dict[str, int] = {}
    last_use: dict[str, int] = {}
    nodes: dict[str, list] = {}
    multi_def: set[str] = set()
    for i, instr in enumerate(instructions):
        # Uses before the def: the RHS is evaluated before the LHS binds.
        if isinstance(instr, IrAssignment):
            occ = get_vars_expr_occurrences(instr.children[1])
        elif isinstance(instr, IrTransRetBasic):
            occ = []
            for child in instr.children:
                occ.extend(get_vars_expr_occurrences(child))
        elif isinstance(instr, (IrAssignToView, IrSetBlockTotalShapeLastDim, IrAssignToBlock)):
            # In-place mutations (`view[idx] = value`, `block.total_shape[-1] = value`,
            # `block.block = value`): unlike IrAssignment, children[0] here is itself a
            # *use* -- e.g. IrAssignToView's target is the buffer being strided over/
            # mutated, not a name being bound -- so both children must be scanned, or
            # the liveness scan sees neither, undercounting last_use for whatever they
            # reference. That let the pool allocator below recycle a still-needed name
            # mid-block (reproduced with a Conv2D "Patches" block scatter sequence,
            # several IrAssignToView statements deep, in the TinyImageNet resnet).
            occ = []
            for child in instr.children:
                occ.extend(get_vars_expr_occurrences(child))
        else:
            occ = []
        for v in occ:
            if v.name.startswith(prefix):
                nodes.setdefault(v.name, []).append(v)
                last_use[v.name] = i
        if isinstance(instr, IrAssignment):
            lhs = instr.children[0]
            if isinstance(lhs, IrVar) and lhs.name.startswith(prefix):
                if lhs.name in def_pos:
                    multi_def.add(lhs.name)
                else:
                    def_pos[lhs.name] = i
                nodes.setdefault(lhs.name, []).append(lhs)
    return def_pos, last_use, nodes, multi_def


def recycle_temp_names_block(
        instructions, prefix="ttb_var_", pool_prefix="ttb_r_") -> dict[str, str]:
    """
    Linear-scan register allocation over a straight-line block: rename single-def
    `prefix` temporaries onto a minimal pool (`pool_prefix`0, 1, ...), reusing a name
    only after its occupant's last use. Returns the applied name map.

    Aliasing is safe: reuse only drops that binding, so a block still referenced
    elsewhere (e.g. `.blocks[k]` held by a survivor) stays alive through it.
    """
    # Drop inert `del` markers (codegen emits nothing for them) so they can't pin a
    # recycled name.
    instructions[:] = [ins for ins in instructions if not isinstance(ins, IrDel)]

    def_pos, last_use, nodes, multi_def = _temp_liveness(instructions, prefix)
    names = sorted(
        (n for n in def_pos if n not in multi_def), key=lambda n: def_pos[n])

    free: list[int] = []                # reusable pool indices (min-heap)
    active: list[tuple[int, int]] = []  # (last_use_index, pool_index) still live
    next_idx = 0
    rename: dict[str, str] = {}
    for name in names:
        p = def_pos[name]
        survivors = []
        for end, idx in active:
            if end < p:                 # occupant dead before this def
                heapq.heappush(free, idx)
            else:
                survivors.append((end, idx))
        active = survivors
        idx = heapq.heappop(free) if free else next_idx
        if idx == next_idx:
            next_idx += 1
        rename[name] = pool_prefix + str(idx)
        active.append((last_use.get(name, p), idx))

    for name, new_name in rename.items():
        for node in nodes[name]:
            node.name = new_name
    return rename


def recycle_temp_names_cfg(cfg: Graph) -> None:
    """private"""
    if len(cfg.nodes) == 1:
        recycle_temp_names_block(cfg.ir[cfg.nodes[0]].children)


def recycle_temp_names(ir: IrProgram) -> None:
    """
    public
    Run after `inline_subexp` on the collapsed per-layer blocks; recycles the
    temporaries it left behind so their tensors free on rebind.
    """
    for transformer in ir.tstore.keys():
        for i in range(len(ir.tstore[transformer])):
            transformer_ir = ir.tstore[transformer][i]
            if transformer_ir.layerwise_cfgs is not None:
                for cfg in transformer_ir.layerwise_cfgs.values():
                    recycle_temp_names_cfg(cfg)
            else:
                recycle_temp_names_cfg(transformer_ir.cfg)
