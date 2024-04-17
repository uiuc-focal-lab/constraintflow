from ir_ast_stack2 import *

def replace(expr, old_var, new_var):
    if isinstance(expr, IrSymbolic):
        if expr.name == old_var.name:
            return new_var
    if expr == old_var:
        return new_var
    for i in range(len(expr.children)):
        new_child = replace(expr.children[i], old_var, new_var)
        expr.children[i] = new_child
    return expr

def replace_all_occurrences(ir_list, old_var, new_var):
    for i in range(len(ir_list)):
        if isinstance(ir_list[i], IrAssignment):
            new_expr = replace(ir_list[i].children[0], old_var, new_var)
            ir_list[i].children[0] = new_expr
            new_expr = replace(ir_list[i].children[1], old_var, new_var)
            ir_list[i].children[1] = new_expr
        elif isinstance(ir_list[i], IrWhile):
            new_expr = replace(ir_list[i].children[0], old_var, new_var)
            ir_list[i].children[0] = new_expr
            replace_all_occurrences(ir_list[i].children[1:], old_var, new_var)
        elif isinstance(ir_list[i], IrTransRetBasic):
            for j in range(len(ir_list[i].children)):
                new_expr = replace(ir_list[i].children[j], old_var, new_var)
                ir_list[i].children[j] = new_expr

def cp_list(ir):
    to_be_removed = []
    for i in range(len(ir)):
        if isinstance(ir[i], IrAssignment):
            if isinstance(ir[i].children[1], IrVar) and 'trav' not in ir[i].children[0].name:
                replace_all_occurrences(ir[i+1:], ir[i].children[0], ir[i].children[1])
                to_be_removed.append(i)
        elif isinstance(ir[i], IrWhile):
            ir[i].children[1:] = cp_list(ir[i].children[1:])
    for i in range(len(to_be_removed)-1, -1, -1):
        ir = ir[:to_be_removed[i]] + ir[to_be_removed[i]+1:]
    return ir

def copy_proagate(ir):
    for transformer in ir.tstore.keys():
        for i in range(len(ir.tstore[transformer])):
            ir.tstore[transformer][i].children = cp_list(ir.tstore[transformer][i].children)
    return ir