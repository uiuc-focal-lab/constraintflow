"""Synthesizes IrOpStmt CFGs for layer kinds that have no DSL representation:
Add (2-parent elementwise sum) and Concat (2-parent stitch along the neuron
axis). Both are handled inline in constraintflow/lib/flow_sparse.py's Flow.flow()
today, with no tracing, so they never get a per-layer specialized kernel and
inherit whatever the *previous* layer's abs_elem components are -- including,
after a specialized Affine/Relu, blocks made of the generated Jit* classes that
don't implement the gbcsr block methods (SparseTensor.binary calls block.binary
and sparse_block.create_similar) Add's inline SparseTensor.binary call needs.
Concat survives today only because it moves blocks without calling block
methods.

Injecting these as ordinary IrOpStmts lets the existing tensor_to_block ->
subexp_inlining -> codeGen pipeline specialize them exactly like Affine/Relu,
so the reuse-mode `Add_<idx>`/`Concat_<idx>` kernels end up pure torch/operator
expressions over the inlined Jit* classes -- no gbcsr call survives, so there is
nothing left to crash.

Call this once, right after `ConvertToIr().visit(ast)` returns and before
`representations.ssa(ir)` runs, so the synthesized CFGs go through ssa/rewrite/
tensor_to_block/codegen exactly like DSL-declared ops. It must run for every
compile (normal, simulacrum, and reuse alike) -- never gated on dummy_mode/
reuse_mode or on whether the network being compiled actually has an Add/Concat
layer -- because tensor_to_block.assign_ttb_counter numbers nodes by walking
ir.tstore in list order with one counter shared across the whole compile.
Appending Add/Concat at the end of that list (never prepending, never
conditionally) is what keeps every existing Affine_*/Relu_* counter -- and
therefore every already-built kernel_cache/ entry -- unperturbed.
"""

from constraintflow.compiler.ir import *
from constraintflow.compiler import representations

# Same params both ops need: two single-layer operands instead of DSL ops'
# single `prev`. An Llist spanning both parents would coalesce them into one
# range (Abs_elem_sparse.get_elem takes min/max over llist.llist), not give a
# per-parent slice, so the signature grows a second `prev` instead.
BINARY_OP_PARAMS = ['abs_elem', 'prev1', 'prev2', 'curr', 'poly_size',
                    'curr_size', 'prev_size', 'input_size', 'batch_size']


def _relu_shaped_operand(name):
    # Mirrors convertToIr.py's Relu/Sigmoid operand metadata (visitOpStmt):
    # a single Neuron-typed value over [1, curr_size] broadcasting over batch.
    return IrVar(name, [IrMetadataElement([1, IrAst.curr_size], 'Neuron', [IrAst.batch_size, 1], False)])


def _stitched_2d_metadata(prev1, key_type):
    # Mirrors IrAccess(prev1, key, key_type, isMetadata=False)'s metadata
    # transform exactly (ir.py:1055-1090), without needing a real IrAccess
    # (IrConcatStitch's tape refers to no operand -- see tensor_to_block.py).
    m = copy_metadata(prev1.irMetadata)
    m[-1].type = key_type
    m[0].shape[0] = IrAst.batch_size
    m[0].broadcast[0] = 1
    return m


def _stitched_mat_metadata(prev1):
    # Mirrors IrExtractPolyCoeff's metadata transform (ir.py:1000-1008) applied
    # to a PolyExp-typed 2-D value: appends the trailing poly_size dimension.
    m = _stitched_2d_metadata(prev1, 'Float')
    m[-1].shape.append(IrAst.poly_size)
    m[-1].broadcast.append(1)
    return m


def _stitched_sym_mat_metadata(prev1):
    # Mirrors IrExtractSymCoeff's metadata transform (ir.py:1056-1065) applied
    # to a SymExp-typed 2-D value: appends the trailing sym_size dimension.
    m = _stitched_2d_metadata(prev1, 'Float')
    m[-1].shape.append(IrAst.sym_size)
    m[-1].broadcast.append(1)
    return m


def _build_concat_cfg(converter, ir_shape):
    prev1 = _relu_shaped_operand('prev1')
    prev2 = _relu_shaped_operand('prev2')

    exprIrs = []
    for key in ir_shape.keys():
        key_type = ir_shape[key]
        if key_type in ('Float', 'Int', 'Bool'):
            exprIrs.append(IrConcatStitch(prev1, prev2, key, 'direct', _stitched_2d_metadata(prev1, key_type)))
        elif key_type == 'PolyExp':
            const_ir = IrConcatStitch(prev1, prev2, key, 'const', _stitched_2d_metadata(prev1, 'Float'))
            mat_ir = IrConcatStitchMat(prev1, prev2, key, _stitched_mat_metadata(prev1))
            exprIrs.append(IrCombineToPoly(mat_ir, const_ir))
        elif key_type == 'SymExp':
            const_ir = IrConcatStitch(prev1, prev2, key, 'const', _stitched_2d_metadata(prev1, 'Float'))
            mat_ir = IrConcatStitchMat(prev1, prev2, key, _stitched_sym_mat_metadata(prev1))
            exprIrs.append(IrCombineToSym(mat_ir, const_ir))
        else:
            raise NotImplementedError(
                f"Concat: shape key {key!r} has type {key_type!r}; only "
                "Float/Int/Bool/PolyExp/SymExp are supported")

    tail_seqIr, retlist = converter.build_trans_ret(exprIrs)
    tail_seqIr.append(IrTransRetBasic(retlist))

    cfg = representations.create_cfg(tail_seqIr)
    return IrOpStmt('Concat', cfg, params=BINARY_OP_PARAMS)


def _build_add_cfg(converter, ir_shape):
    prev1 = _relu_shaped_operand('prev1')
    prev2 = _relu_shaped_operand('prev2')

    seqIr = []
    exprIrs = []
    for key in ir_shape.keys():
        key_type = ir_shape[key]
        a = IrAccess(prev1, key, key_type, isMetadata=False)
        b = IrAccess(prev2, key, key_type, isMetadata=False)
        # visitBinOp is a pure IR combinator when lhsIr/rhsIr are supplied: it
        # ignores ast_node entirely (convertToIr.py:52-56) and produces exactly
        # the Float/PolyExp/SymExp '+' semantics flow_sparse.py's inline Add
        # hand-wrote (IrBinaryOp / IrCombineToPoly / IrCombineToSym), hoisting
        # PolyExp/SymExp operands into named vars itself.
        resultIr, resultSeqIr = converter.visitBinOp(None, lhsIr=a, rhsIr=b, ast_node_type=key_type, op='+')
        seqIr += resultSeqIr
        exprIrs.append(resultIr)

    tail_seqIr, retlist = converter.build_trans_ret(exprIrs)
    seqIr += tail_seqIr
    seqIr.append(IrTransRetBasic(retlist))

    cfg = representations.create_cfg(seqIr)
    return IrOpStmt('Add', cfg, params=BINARY_OP_PARAMS)


def inject_builtin_ops(ir, converter):
    """Append synthesized Add/Concat IrOpStmts to every transformer in `ir.tstore`.

    `ir` is the IrProgram returned by ConvertToIr().visit(ast); `converter` is
    the same ConvertToIr instance that produced it (compile.py must keep it
    alive across the call instead of discarding it), so get_var()/visitBinOp
    continue the same variable/counter state DSL ops already used.
    """
    from constraintflow.lib import globals as G
    G.program_add_rule.reset_flag()
    for transformer_name in ir.tstore.keys():
        # A program's own `Neuron_add` rule (lowered to an 'Add' op) takes precedence.
        if any(op.op == 'Add' for op in ir.tstore[transformer_name]):
            G.program_add_rule.set_flag()
        else:
            ir.tstore[transformer_name].append(_build_add_cfg(converter, ir.shape))
        ir.tstore[transformer_name].append(_build_concat_cfg(converter, ir.shape))
