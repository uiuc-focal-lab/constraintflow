from ir_ast_stack2 import *
from representations import *
counter = -1

def get_var():
    global counter 
    counter += 1
    return 'rewrite_' + str(counter)

def rewrite_expr(expr):
    new_children = []
    new_assignments = []
    for child in expr.children:
        new_child, new_assignment = rewrite_expr(child)
        new_children.append(new_child)
        new_assignments += new_assignment
    expr.update_parent_child(new_children)
    if isinstance(expr, IrTernary):
        new_name = get_var()
        new_var = IrVar(new_name, expr.children[0].irMetadata)
        new_assignment = IrAssignment(new_var, IrConvertBoolToFloat(expr.children[0]))
        new_lhs = IrMult(new_var, expr.children[1], '*')
        new_rhs = IrMult(IrBinaryOp(IrConst(1.0, 'Float'), new_var, '-'), expr.children[2], '*')
        new_expr = IrBinaryOp(new_lhs, new_rhs, '+')
        new_assignments.append(new_assignment)
        expr = new_expr
    
    return expr, new_assignments

def rewrite_ir_list(ir_list):
    length = len(ir_list)
    index = 0
    for i in range(length):
        l = ir_list[index]
        if isinstance(l, IrAssignment):
            new_expr, new_assignments = rewrite_expr(l.children[1])
            new_children = [l.children[0], new_expr]
            l.update_parent_child(new_children)
            for j in range(len(new_assignments)):
                ir_list.insert(index, new_assignments[j])
                index += 1
        elif isinstance(l, IrTransRetBasic):
            new_children = []
            new_assignments = []
            for child in l.children:
                new_expr, new_assignments_inner = rewrite_expr(child)
                new_children.append(new_expr)
                new_assignments += new_assignments_inner
            l.update_parent_child(new_children)
            for j in range(len(new_assignments)):
                ir_list.insert(index, new_assignments[j])
                index += 1
        elif isinstance(l, IrWhile):
            new_ir_list = rewrite_ir_list(l.children[1:])
            new_children = [l.children[0]] + new_ir_list
            l.update_parent_child(new_children)
        index += 1
    return ir_list


def rewrite(ir):
    for transformer in ir.tstore.keys():
        for i in range(len(ir.tstore[transformer])):
            ir_transformer = ir.tstore[transformer][i].children
            new_ir = rewrite_ir_list(ir_transformer)
            ir.tstore[transformer][i].children = new_ir
    return ir