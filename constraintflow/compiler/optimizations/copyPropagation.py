from constraintflow.compiler.ir import *
from constraintflow.compiler import representations



def replace_all_occurrences_metadata(irMetadata, old_var, new_var):
    for j, irMetadataElement in enumerate(irMetadata):
        for i in range(len(irMetadataElement.shape)):
            if isinstance(irMetadataElement.shape[i], IrAst):
                replace_all_occurrences_expr(irMetadataElement.shape[i], old_var, new_var)
        for i in range(len(irMetadataElement.broadcast)):
            if isinstance(irMetadataElement.broadcast[i], IrAst):
                replace_all_occurrences_expr(irMetadataElement.broadcast[i], old_var, new_var)
                

def replace_all_occurrences_expr(expr, old_var, new_var):
    if not isinstance(expr, int):
        replace_all_occurrences_metadata(expr.irMetadata, old_var, new_var)
        if expr == old_var:
            return new_var
        for i in range(len(expr.children)):
            new_child = replace_all_occurrences_expr(expr.children[i], old_var, new_var)
            expr.children[i] = new_child
    return expr
    

def replace_all_occurrences(old_var, new_var, cfg):
    for node in cfg.nodes:
        block = cfg.ir[node]
        ir_list = block.children
        for i in range(len(ir_list)):
            new_children = []
            for child in ir_list[i].children:
                new_children.append(replace_all_occurrences_expr(child, old_var, new_var))
            ir_list[i].update_parent_child(new_children)
        if block.inner_jump != None:
            block.inner_jump[0] = replace_all_occurrences_expr(block.inner_jump[0], old_var, new_var)
        if block.jump != None:
            block.jump[0] = replace_all_occurrences_expr(block.jump[0], old_var, new_var)

def cp_block(block, cfg):
    ir_list = block.children
    to_be_removed = []
    for i in range(len(ir_list)):
        if isinstance(ir_list[i], IrAssignment):
            if isinstance(ir_list[i].children[1], IrVar):
                replace_all_occurrences(ir_list[i].children[0], ir_list[i].children[1], cfg)
                to_be_removed.append(i)
    for i in range(len(to_be_removed)-1, -1, -1):
        del ir_list[to_be_removed[i]]
    return ir_list

def cp_cfg(cfg):
    for node in cfg.nodes:
        block = cfg.ir[node]
        cp_block(block, cfg)

def copy_proagate(ir):
    for transformer in ir.tstore.keys():
        for i in range(len(ir.tstore[transformer])):
            cfg = ir.tstore[transformer][i].cfg
            cp_cfg(cfg)
    return ir