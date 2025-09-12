from constraintflow.compiler.ir import *
from constraintflow.compiler.optimizations import uses


def replace(expr):
    if isinstance(expr, int):
        return expr
    new_children = []
    for child in expr.children:
        new_children.append(replace(child))
    expr.update_parent_child(new_children)
    if isinstance(expr, IrExtractPolyCoeff):
        inputIr = expr.children[0]
        if isinstance(inputIr, IrCombineToPoly):
            expr = inputIr.children[0]
        elif isinstance(inputIr, IrVar):
            def_assignment = inputIr.defs
            inputIr = def_assignment.children[1]
            if isinstance(inputIr, IrCombineToPoly):
                expr = inputIr.children[0]
    elif isinstance(expr, IrExtractPolyConst):
        inputIr = expr.children[0]
        if isinstance(inputIr, IrCombineToPoly):
            expr = inputIr.children[1]
        elif isinstance(inputIr, IrVar):
            def_assignment = inputIr.defs
            inputIr = def_assignment.children[1]
            if isinstance(inputIr, IrCombineToPoly):
                expr = inputIr.children[1]
    return expr

def poly_opt_block(block):
    ir_list = block.children
    for i in range(len(ir_list)):
        new_children = []
        for child in ir_list[i].children:
            new_children.append(replace(child))
        ir_list[i].update_parent_child(new_children)
    if block.inner_jump != None:
        block.inner_jump[0] = replace(block.inner_jump[0])
    if block.jump != None:
        block.jump[0] = replace(block.jump[0])

def poly_opt_cfg(cfg):
    for node in cfg.nodes:
        block = cfg.ir[node]
        poly_opt_block(block)


def poly_opt(ir):
    uses.populate_uses_defs(ir)
    for transformer in ir.tstore.keys():
        for i in range(len(ir.tstore[transformer])):
            cfg = ir.tstore[transformer][i].cfg
            poly_opt_cfg(cfg)