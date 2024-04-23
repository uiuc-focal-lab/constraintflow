from ir_ast_stack2 import *

def populate_uses_expr(expr, ir_node):
    if isinstance(expr, IrVar) or isinstance(expr, IrSymbolic):
        expr.uses.append(ir_node)
    else:
        for child in expr.children:
            populate_uses_expr(child, ir_node)

def populate_uses_transformer(ir_list):
    for i in range(len(ir_list)):
        if isinstance(ir_list[i], IrAssignment):
            populate_uses_expr(ir_list[i].children[1], ir_list[i])
        elif isinstance(ir_list[i], IrWhile):
            populate_uses_expr(ir_list[i].children[0], ir_list[i])
            populate_uses_transformer(ir_list[i].children[1:])
        elif isinstance(ir_list[i], IrTransRetBasic):
            for child in ir_list[i].children:
                populate_uses_expr(child, ir_list[i])

def populate_uses(ir):
    for transformer in ir.tstore.keys():
        for i in range(len(ir.tstore[transformer])):
            populate_uses_transformer(ir.tstore[transformer][i].children)
    return ir