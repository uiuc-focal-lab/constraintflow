from . import uses
from .ir_ast_stack2 import *

def dce_block(block):
    ir_list = block.children
    for i in range(len(ir_list)-1,-1,-1):
        if isinstance(ir_list[i], IrAssignment):
            if len(ir_list[i].uses) == 0:
                del ir_list[i]

def dce_cfg(cfg):
    for node in cfg.nodes:
        block = cfg.ir[node]
        dce_block(block)

def dce(ir):
    uses.populate_uses_defs(ir)
    for transformer in ir.tstore.keys():
        for i in range(len(ir.tstore[transformer])):
            cfg = ir.tstore[transformer][i].cfg
            dce_cfg(cfg)