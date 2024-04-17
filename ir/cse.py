from ir_ast_stack2 import *

counter = 0

def get_var():
    global counter 
    counter += 1
    return 'cse_var_' + str(counter)

def compare(x):
    return x[0]

def check_expr_visit(expr, visited_expr, visited_order):
    if isinstance(expr, IrConst) or isinstance(expr, IrVar) or isinstance(expr, IrSymbolic):
        return
    if expr not in visited_expr:
        visited_expr[expr] = 1
        for i in range(len(expr.children)):
            check_expr_visit(expr.children[i], visited_expr, visited_order)        
    elif visited_expr[expr] == 1:
        visited_expr[expr] += 1
        visited_order.append(expr)

def cse_list(ir_list, visited_expr, visited_order):
    for ir in ir_list:
        if isinstance(ir, IrAssignment):
            check_expr_visit(ir.children[1], visited_expr, visited_order)
        elif isinstance(ir, IrWhile):
            check_expr_visit(ir.children[0], visited_expr, visited_order)
            cse_list(ir.children[1:], visited_expr, visited_order)
        elif isinstance(ir, IrTransRetBasic):
            for j in range(len(ir.children)):
                check_expr_visit(ir.children[j], visited_expr, visited_order)

def replace(expr, sub_expr, var):
    if expr == sub_expr:
        return var, True 
    flag = False
    for i in range(len(expr.children)):
        new_child, replaced = replace(expr.children[i], sub_expr, var)
        if replaced:
            expr.children[i] = new_child
            flag = replaced
    return expr, flag

def replace_all_occurrences(ir_list, new_assignment, flag):
    index = []
    for i in range(len(ir_list)):
        if isinstance(ir_list[i], IrAssignment):
            new_expr, replaced = replace(ir_list[i].children[1], new_assignment.children[1], new_assignment.children[0])
            if replaced:
                ir_list[i].children[1] = new_expr
                if not flag:
                    flag = True
                    index = [i] 
        elif isinstance(ir_list[i], IrWhile):
            new_expr, replaced = replace(ir_list[i].children[0], new_assignment.children[1], new_assignment.children[0])
            if replaced:
                ir_list[i].children[0] = new_expr
                if not flag:
                    flag = True
                    index = [i] 
            inner_index = replace_all_occurrences(ir_list[i].children[1:], new_assignment, flag)
            if len(inner_index)>0:
                index = [i] + inner_index
        elif isinstance(ir_list[i], IrTransRetBasic):
            for j in range(len(ir_list[i].children)):
                new_expr, replaced = replace(ir_list[i].children[j], new_assignment.children[1], new_assignment.children[0])
                if replaced:
                    ir_list[i].children[j] = new_expr
                    if not flag:
                        flag = True
                        index = [i] 
    return index



def create_new_assignments(ir, visited_order, new_assignments, new_assignments_index):
    for i in range(len(visited_order)-1, -1, -1):
        original_expr = visited_order[i]
        new_var = IrVar(get_var(), original_expr.irMetadata)
        new_assignment = IrAssignment(new_var, original_expr)
        new_assignments.append(new_assignment)
        index = replace_all_occurrences(ir, new_assignment, False)
        ir = add_assignment(index, ir, new_assignment)
        new_assignments_index.append(index)
    return ir

def add_assignment(index, ir_list, assignment):
    if len(index)==1:
        index = index[0]
        ir_list = ir_list[:index] + [assignment] + ir_list[index:]
        return ir_list
    else:
        ir_list[index[0]].children[1:] = add_assignment(index[1:], ir_list[index[0]].children[1:], assignment)
        return ir_list
    
def add_all_assignments(ir_list, new_assignments, new_assignments_index):
    for i in range(len(new_assignments)-1, -1, -1):
        ir_list = add_assignment(new_assignments_index[i], ir_list, new_assignments[i])
    return ir_list

def cse_transformer(ir):
    visited_expr = {}
    visited_order = []
    cse_list(ir, visited_expr, visited_order)
    new_assignments = []
    new_assignments_index = []
    ir = create_new_assignments(ir, visited_order, new_assignments, new_assignments_index)
    return ir 

def cse(ir):
    for transformer in ir.tstore.keys():
        for i in range(len(ir.tstore[transformer])):
            ir.tstore[transformer][i].children = cse_transformer(ir.tstore[transformer][i].children)
    return ir