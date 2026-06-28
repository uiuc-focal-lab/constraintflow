"""
Detect two sequential, independent while-loops in an Affine transformer
(the L-bound loop and U-bound loop in DeepPoly) and wrap them in an
IrParallelBlock so the code generator can emit Python threading code.

The pass runs AFTER remove_phi and BEFORE codeGen.
"""

from constraintflow.compiler.ir import *


# ── helpers ──────────────────────────────────────────────────────────────────

def _collect_reads(expr, out):
    if isinstance(expr, int):
        return
    if isinstance(expr, IrVar):
        out.add(expr.name)
    else:
        for child in expr.children:
            _collect_reads(child, out)

def get_reads(stmts):
    reads = set()
    for stmt in stmts:
        if isinstance(stmt, IrAssignment):
            _collect_reads(stmt.children[1], reads)
        elif isinstance(stmt, IrTransRetBasic):
            for child in stmt.children:
                _collect_reads(child, reads)
    return reads

def get_defs(stmts):
    defs = set()
    for stmt in stmts:
        if isinstance(stmt, IrAssignment):
            defs.add(stmt.children[0].name)
    return defs

def get_reads_in_while(while_block):
    """All var names read anywhere inside a WhileBlock (header + body)."""
    reads = set()
    for block in while_block.loopBody:
        for stmt in block.children:
            if isinstance(stmt, IrAssignment):
                _collect_reads(stmt.children[1], reads)
        if block.inner_jump:
            _collect_reads(block.inner_jump[0], reads)
        if block.jump:
            _collect_reads(block.jump[0], reads)
    return reads

def get_defs_in_while(while_block):
    defs = set()
    for block in while_block.loopBody:
        for stmt in block.children:
            if isinstance(stmt, IrAssignment):
                defs.add(stmt.children[0].name)
    return defs


# ── pattern detection ─────────────────────────────────────────────────────────

def _find_pattern(cfg):
    """
    Look for:  Block_A → W1 (IrWhileBlock) → Block_B → W2 (IrWhileBlock) → Block_C
    Returns (block_a, w1, block_b, w2, block_c) or None.
    """
    for node in cfg.nodes:
        block_a = cfg.ir[node]
        if isinstance(block_a, IrWhileBlock):
            continue
        if not isinstance(block_a, IrBlock):
            continue
        if block_a.inner_jump is None or block_a.jump is None:
            continue
        if not isinstance(block_a.inner_jump[1], IrWhileBlock):
            continue
        w1 = block_a.inner_jump[1]
        block_b = block_a.jump[1]
        if isinstance(block_b, IrWhileBlock):
            continue
        if not isinstance(block_b, IrBlock):
            continue
        if block_b.inner_jump is None or block_b.jump is None:
            continue
        if not isinstance(block_b.inner_jump[1], IrWhileBlock):
            continue
        w2 = block_b.inner_jump[1]
        block_c = block_b.jump[1]
        return block_a, w1, block_b, w2, block_c
    return None


def _check_independent(w1, w2):
    """
    W1 and W2 are independent if neither writes anything the other reads.
    """
    w1_defs  = get_defs_in_while(w1)
    w2_defs  = get_defs_in_while(w2)
    w1_reads = get_reads_in_while(w1)
    w2_reads = get_reads_in_while(w2)
    return len(w1_defs & w2_reads) == 0 and len(w2_defs & w1_reads) == 0


# ── zone splitting ────────────────────────────────────────────────────────────

def _split(block_a, w1, block_b, w2, block_c):
    """
    Returns:
        shared_preamble  – Block A stmts both threads need (emitted before IrParallelBlock)
        shared_block_b   – Block B stmts needed by BOTH threads (emitted before IrParallelBlock)
        zone1_init       – Block A stmts only thread1 needs
        zone1_post       – Block B stmts that use W1's output (l_new computation)
        zone2_init       – Block B stmts that seed W2 (vertices_stop2, phi_trav_exp2)
        zone2_post       – Block C stmts that use W2's output (u_new computation)
        escape1          – var names thread1 produces that are read after the join
        escape2          – var names thread2 produces that are read after the join
        return_stmts     – IrTransRetBasic nodes from Block C (kept in Block C)
    None if parallelization is unsafe.
    """

    # ── Step 1: split Block C into zone2_post and return ──────────────────
    block_c_stmts = list(block_c.children)
    return_stmts = [s for s in block_c_stmts if isinstance(s, IrTransRetBasic)]
    zone2_post   = [s for s in block_c_stmts if not isinstance(s, IrTransRetBasic)]

    # ── Step 2: split Block B into zone1_post and zone2_init ──────────────
    # zone2_init = stmts whose LHS is directly read by W2
    w2_reads = get_reads_in_while(w2)
    block_b_stmts = list(block_b.children)

    zone2_init_names = {s.children[0].name
                        for s in block_b_stmts
                        if isinstance(s, IrAssignment)
                        and s.children[0].name in w2_reads}

    # Backward-propagate through Block B: if a stmt feeds a zone2_init var, it's also zone2_init
    changed = True
    while changed:
        changed = False
        for stmt in reversed(block_b_stmts):
            if isinstance(stmt, IrAssignment):
                if stmt.children[0].name in zone2_init_names:
                    rhs_reads = get_reads([stmt])
                    for s2 in block_b_stmts:
                        if isinstance(s2, IrAssignment):
                            n = s2.children[0].name
                            if n in rhs_reads and n not in zone2_init_names:
                                zone2_init_names.add(n)
                                changed = True

    zone1_post = [s for s in block_b_stmts
                  if not (isinstance(s, IrAssignment) and s.children[0].name in zone2_init_names)]
    zone2_init = [s for s in block_b_stmts
                  if isinstance(s, IrAssignment) and s.children[0].name in zone2_init_names]

    # ── Step 2b: detect cross-thread vars and move to shared_block_b ──────
    # Some Block B vars end up in zone1_post but are also needed by zone2_post
    # (thread 2). These must be computed before threads start so both can read
    # them as closed-over variables. If any of them depend on W1's output,
    # parallelisation is unsafe and we bail out.
    w1_defs = get_defs_in_while(w1)
    zone2_post_reads = get_reads(zone2_post)
    zone1_post_defs = get_defs(zone1_post)

    cross_thread_names = zone1_post_defs & zone2_post_reads
    if cross_thread_names:
        # Propagate: pull in transitive Block B dependencies of cross_thread_names
        changed = True
        while changed:
            changed = False
            for stmt in reversed(zone1_post):
                if isinstance(stmt, IrAssignment):
                    if stmt.children[0].name in cross_thread_names:
                        for s2 in zone1_post:
                            if isinstance(s2, IrAssignment):
                                n = s2.children[0].name
                                if n in get_reads([stmt]) and n not in cross_thread_names:
                                    cross_thread_names.add(n)
                                    changed = True

        # Safety: if any cross-thread var reads a W1-produced variable, bail
        cross_thread_reads = set()
        for stmt in zone1_post:
            if isinstance(stmt, IrAssignment) and stmt.children[0].name in cross_thread_names:
                cross_thread_reads.update(get_reads([stmt]))
        if cross_thread_reads & w1_defs:
            return None

        shared_block_b = [s for s in zone1_post
                          if isinstance(s, IrAssignment)
                          and s.children[0].name in cross_thread_names]
        zone1_post = [s for s in zone1_post
                      if not (isinstance(s, IrAssignment)
                              and s.children[0].name in cross_thread_names)]
    else:
        shared_block_b = []

    # ── Step 3: split Block A into shared_preamble and zone1_init ─────────
    # Build zone2_seed = everything needed by W2 / zone2_init / zone2_post
    zone2_seed = set(w2_reads)
    zone2_seed.update(get_reads(zone2_init))
    zone2_seed.update(get_reads(zone2_post))

    # Backward-propagate through Block A to pull in transitive Block A deps of zone2_seed
    block_a_stmts = list(block_a.children)
    changed = True
    while changed:
        changed = False
        for stmt in reversed(block_a_stmts):
            if isinstance(stmt, IrAssignment):
                lhs = stmt.children[0].name
                if lhs in zone2_seed:
                    new_reads = get_reads([stmt]) - zone2_seed
                    if new_reads:
                        zone2_seed.update(new_reads)
                        changed = True

    zone1_init = [s for s in block_a_stmts
                  if isinstance(s, IrAssignment) and s.children[0].name not in zone2_seed]
    shared_preamble = [s for s in block_a_stmts
                       if not (isinstance(s, IrAssignment) and s.children[0].name not in zone2_seed)]

    # ── Step 3b: promote zone1_init defs read by shared_preamble ─────────
    # If shared_preamble reads a var that is only defined in zone1_init, that
    # var must execute before threads start. We promote it (and its transitive
    # zone1_init dependencies) into shared_preamble by adding to zone2_seed
    # and rebuilding.  This prevents the Python scoping bug where an escape-var
    # init ("symexp_1 = None") appears after a shared-preamble read of it.
    zone1_init_defs = get_defs(zone1_init)
    extra_to_promote = zone1_init_defs & get_reads(shared_preamble)
    if extra_to_promote:
        changed = True
        while changed:
            changed = False
            for stmt in zone1_init:
                if isinstance(stmt, IrAssignment):
                    lhs = stmt.children[0].name
                    if lhs in extra_to_promote:
                        for r in get_reads([stmt]):
                            if r in zone1_init_defs and r not in extra_to_promote:
                                extra_to_promote.add(r)
                                changed = True
        zone2_seed.update(extra_to_promote)
        zone1_init = [s for s in block_a_stmts
                      if isinstance(s, IrAssignment) and s.children[0].name not in zone2_seed]
        shared_preamble = [s for s in block_a_stmts
                           if not (isinstance(s, IrAssignment) and s.children[0].name not in zone2_seed)]

    # ── Step 4: find escape vars ──────────────────────────────────────────
    w1_loop_stmts = []
    for blk in w1.loopBody:
        w1_loop_stmts.extend(blk.children)
    w2_loop_stmts = []
    for blk in w2.loopBody:
        w2_loop_stmts.extend(blk.children)

    thread1_defs = get_defs(zone1_init) | get_defs(w1_loop_stmts) | get_defs(zone1_post)
    thread2_defs = get_defs(zone2_init) | get_defs(w2_loop_stmts) | get_defs(zone2_post)

    after_parallel_reads = get_reads(return_stmts)

    escape1 = sorted(thread1_defs & after_parallel_reads)
    escape2 = sorted(thread2_defs & after_parallel_reads)

    return (shared_preamble, shared_block_b, zone1_init, zone1_post,
            zone2_init, zone2_post,
            escape1, escape2, return_stmts)


# ── CFG transformation ────────────────────────────────────────────────────────

def _parallelize_cfg(cfg):
    result = _find_pattern(cfg)
    if result is None:
        return
    block_a, w1, block_b, w2, block_c = result

    if not _check_independent(w1, w2):
        return

    split_result = _split(block_a, w1, block_b, w2, block_c)
    if split_result is None:
        return  # cross-thread vars depend on W1's output — unsafe to parallelize

    (shared_preamble, shared_block_b, zone1_init, zone1_post,
     zone2_init, zone2_post,
     escape1, escape2, return_stmts) = split_result

    # Guard: both loops must have at least one exclusive init var
    if not zone1_init or not zone2_init:
        return

    par = IrParallelBlock(zone1_init, w1, zone1_post,
                          zone2_init, w2, zone2_post,
                          escape1, escape2)

    # Modify Block A: shared preamble + shared Block B vars + IrParallelBlock
    # shared_block_b vars are needed by both threads; emitting them before the
    # parallel block lets both thread closures read them without data races.
    block_a.children = shared_preamble + shared_block_b + [par]
    block_a.inner_jump = None
    block_a.jump = [block_a.jump[0], block_c]

    # Block C: only the return statement
    block_c.children = return_stmts


# ── public entry point ────────────────────────────────────────────────────────

def parallelize_loops(ir):
    for transformer in ir.tstore.keys():
        for i in range(len(ir.tstore[transformer])):
            cfg = ir.tstore[transformer][i].cfg
            _parallelize_cfg(cfg)
    return ir
