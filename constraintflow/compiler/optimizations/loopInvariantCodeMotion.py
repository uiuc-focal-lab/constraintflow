from constraintflow.compiler.ir import *
from constraintflow.compiler import representations
from constraintflow.compiler.optimizations import uses

def get_vars_expr(expr):
    vars = set()
    if isinstance(expr, int):
        pass
    elif isinstance(expr, IrVar):
        vars.add(expr)
    else:
        for child in expr.children:
            vars = vars.union(get_vars_expr(child))
    return vars

def licm_while_block(block, combined_ir_list, predecessors, cfg):
    ir_list = block.children
    to_be_removed = []
    for i in range(len(ir_list)):
        if isinstance(ir_list[i], IrAssignment):
            vars = get_vars_expr(ir_list[i].children[1])
            can_be_removed = True
            for var in vars:
                d = var.defs 
                if d in combined_ir_list:
                    can_be_removed = False
                    break
            if can_be_removed:
                to_be_removed.append(i)
    for predecessor in predecessors:
        for i in to_be_removed:
            cfg.ir[predecessor].children.append(ir_list[i])
    for i in range(len(to_be_removed)-1, -1, -1):
        del ir_list[to_be_removed[i]]
    return block.children, len(to_be_removed)>0

def licm_cfg(cfg):
    uses.populate_uses_defs_cfg(cfg)
    for node in cfg.nodes:
        block = cfg.ir[node]
        if isinstance(block, IrWhileBlock):
            predecessors = cfg.predecessors[node]
            for inner_block in block.loopBody:
                inner_block_id = cfg.get_block_id(inner_block)
                if inner_block_id in predecessors:
                    predecessors.remove(inner_block_id)
            flag = True
            while flag:
                flag = False
                combined_ir_list = []
                for inner_block in block.loopBody:
                    combined_ir_list = combined_ir_list + inner_block.children
                for inner_block in block.loopBody:
                    debug_ir_list, flag_ = licm_while_block(inner_block, combined_ir_list, predecessors, cfg)
                    flag = flag or flag_
                    
            

def licm(ir):
    for transformer in ir.tstore.keys():
        for i in range(len(ir.tstore[transformer])):
            cfg = ir.tstore[transformer][i].cfg
            licm_cfg(cfg)
    return ir

def print_list(ir_list):
    for i in ir_list:
        if isinstance(i, IrAssignment):
            print(i.children[0].name)