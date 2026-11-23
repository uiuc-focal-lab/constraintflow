"""
This pass removes the `SparseTensor` layer remaining in the IR after the
    `tensor_to_block` pass.
After this pass, there should be no `SparseTensor` in IR and generated code.
After `tensor_to_block`, the `SparseTensor` is essentially just a list of
    blocks, and all operations are directly applied to the blocks inside.
Must apply `tensor_to_block` first!!!
"""


from __future__ import annotations
from constraintflow.compiler.ir import *
from constraintflow.compiler.representations import Graph
from constraintflow.compiler.optimizations.tensor_to_block import \
    replace_all_occurrences_expr
from constraintflow.compiler.optimizations.loopInvariantCodeMotion import \
    get_vars_expr


def remove_sparse_tensor_block(block: IrBlock) -> None:
    """private"""
    pass


def remove_sparse_tensor_cfg(cfg: Graph) -> None:
    """private"""
    pass


def remove_sparse_tensor(ir: IrProgram) -> None:
    """
    public
    Requires:
    - Must be applied after `tensor_to_block.tensor_to_block(ir)`.
    Ensures:
    - No warpping `SparseTensor` in IR.
    """
    for transformer in ir.tstore.keys():
        for i in range(len(ir.tstore[transformer])):
            transformer_ir = ir.tstore[transformer][i]
            if transformer_ir.layerwise_cfgs is not None:
                for cfg in transformer_ir.layerwise_cfgs.values():
                    remove_sparse_tensor_cfg(cfg)
            else:
                cfg: Graph = transformer_ir.cfg
                remove_sparse_tensor_cfg(cfg)
