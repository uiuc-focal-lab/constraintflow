from constraintflow.compiler.ir import *

def populate_uses_metadata(irMetadata, ir_node, vars):
    for irMetadataElement in irMetadata:
        for i in range(len(irMetadataElement.shape)):
            if isinstance(irMetadataElement.shape[i], IrAst):
                populate_uses_expr(irMetadataElement.shape[i], ir_node, vars)
        for i in range(len(irMetadataElement.broadcast)):
            if isinstance(irMetadataElement.broadcast[i], IrAst):
                populate_uses_expr(irMetadataElement.broadcast[i], ir_node, vars)

def populate_uses_expr(expr, ir_node, vars):
    if isinstance(expr, int):
        return 
    populate_uses_metadata(expr.irMetadata, ir_node, vars)
    if isinstance(expr, IrVar):
        if expr.formal_argument:
            expr.uses.append(ir_node)
        else:
            if ir_node not in expr.defs.uses: 
                expr.defs.uses.append(ir_node)
    else:
        for child in expr.children:
            populate_uses_expr(child, ir_node, vars)

def populate_uses_block(block, vars):
    ir_list = block.children
    for i in range(len(ir_list)):
        if isinstance(ir_list[i], IrAssignment):
            populate_uses_expr(ir_list[i].children[1], ir_list[i], vars)
        elif isinstance(ir_list[i], IrTransRetBasic):
            for child in ir_list[i].children:
                populate_uses_expr(child, ir_list[i], vars)
    if block.inner_jump != None:
        populate_uses_expr(block.inner_jump[0], block.inner_jump, vars)
    if block.jump != None:
        populate_uses_expr(block.jump[0], block.jump, vars)

def populate_defs_block(block, vars):
    ir_list = block.children
    for i in range(len(ir_list)):
        if isinstance(ir_list[i], IrAssignment):
            if ir_list[i].children[0].defs != None:
                print(ir_list[i].children[0])
            assert(ir_list[i].children[0].defs == None)
            ir_list[i].children[0].defs = ir_list[i]
            vars[ir_list[i].children[0].name] = ir_list[i]
            ir_list[i].children[0].formal_argument = False

def clear_defs_expr(expr):
    if isinstance(expr, int):
        return 
    if isinstance(expr, IrVar):
        if expr.formal_argument:
            expr.uses = []
        expr.defs = None
    else:
        for child in expr.children:
            clear_defs_expr(child)

def clear_uses_defs(cfg):
    for node in cfg.nodes:
        block = cfg.ir[node]
        ir_list = block.children
        for i in range(len(ir_list)):
            if isinstance(ir_list[i], IrAssignment):
                ir_list[i].uses = []
            for child in ir_list[i].children:
                clear_defs_expr(child)


def populate_uses_defs_cfg(cfg):
    vars = dict()
    clear_uses_defs(cfg)
    for node in cfg.nodes:
        block = cfg.ir[node]
        populate_defs_block(block, vars)
    for node in cfg.nodes:
        block = cfg.ir[node]
        populate_uses_block(block, vars)

def populate_uses_defs(ir):
    for transformer in ir.tstore.keys():
        for i in range(len(ir.tstore[transformer])):
            cfg = ir.tstore[transformer][i].cfg
            populate_uses_defs_cfg(cfg)