from constraintflow.compiler.ir import *
from constraintflow.compiler.optimizations import uses

counter = -1
def get_var():
    global counter 
    counter += 1
    return 'symexp_' + str(counter)

def get_assignment(expr):
    if isinstance(expr, int):
        return expr, []
    new_children = []
    new_assignments = []
    for child in expr.children:
        new_child, new_assignment = get_assignment(child)
        new_children.append(new_child)
        new_assignments += new_assignment
    expr.update_parent_child(new_children)
    if isinstance(expr, IrExtractSymCoeff) or isinstance(expr, IrCombineToSym):
        new_name = get_var()
        new_var = IrVar(new_name, expr.irMetadata)
        new_assignment = IrAssignment(new_var, expr)
        new_assignments.append(new_assignment)
        expr = new_var
    return expr, new_assignments

def new_assignment_for_symexp_block(block):
    ir_list = block.children
    length = len(ir_list)
    index = 0
    for i in range(length):
        l = ir_list[index]
        if isinstance(l, IrAssignment):
            new_expr, new_assignments = get_assignment(l.children[1])
            new_children = [l.children[0], new_expr]
            l.update_parent_child(new_children)
            for j in range(len(new_assignments)):
                ir_list.insert(index, new_assignments[j])
                index += 1
        elif isinstance(l, IrTransRetBasic):
            new_children = []
            new_assignments = []
            for child in l.children:
                new_expr, new_assignments_inner = get_assignment(child)
                new_children.append(new_expr)
                new_assignments += new_assignments_inner
            l.update_parent_child(new_children)
            for j in range(len(new_assignments)):
                ir_list.insert(index, new_assignments[j])
                index += 1
        index += 1
    return ir_list

def new_assignment_for_symexp_cfg(cfg):
    for node in cfg.nodes:
        block = cfg.ir[node]
        new_assignment_for_symexp_block(block)



def replace_expr_with_expanded_mat(expr, var):
    if isinstance(expr, int) or isinstance(expr, IrExpandSymExp):
        return expr
    new_children = []
    for child in expr.children:
        new_child = replace_expr_with_expanded_mat(child, var)
        new_children.append(new_child)
    expr.update_parent_child(new_children)
    if expr == var:
        new_expr = IrExpandSymExp(expr)
        expr = new_expr
    return expr

def compute_tainted_recursive(var):
    for use in var.defs.uses:
        if isinstance(use, IrAssignment):
            rhs = replace_expr_with_expanded_mat(use.children[1], var)
            use.update_parent_child([use.children[0], rhs])
            if not isinstance(use.children[1], IrCombineToSym):
                compute_tainted_recursive(use.children[0])
        if isinstance(use, IrTransRetBasic):
            new_children = []
            for child in use.children:
                new_expr = replace_expr_with_expanded_mat(child, var)
                new_children.append(new_expr)
            use.update_parent_child(new_children)

def compute_tainted_block(block):
    ir_list = block.children
    for i in range(len(ir_list)):
        if isinstance(ir_list[i], IrAssignment):
            if isinstance(ir_list[i].children[1], IrExtractSymCoeff):
                var = ir_list[i].children[0]
                compute_tainted_recursive(ir_list[i].children[0])

def compute_tainted_cfg(cfg):
    for node in cfg.nodes:
        block = cfg.ir[node]
        compute_tainted_block(block)

def correct_symexp_size(ir):
    for transformer in ir.tstore.keys():
        for i in range(len(ir.tstore[transformer])):
            cfg = ir.tstore[transformer][i].cfg
            new_assignment_for_symexp_cfg(cfg)
            uses.populate_uses_defs_cfg(cfg)
            compute_tainted_cfg(cfg)
    return ir