from ir_ast_stack2 import *
import representations 

counter = 0

def get_var():
    global counter 
    counter += 1
    return 'cse_var_' + str(counter)

def compare(x):
    return x[0]

def check_expr_visit(expr, node, visited_expr, visited_order):
    if isinstance(expr, IrConst) or isinstance(expr, IrVar) or isinstance(expr, IrSymbolic) or isinstance(expr, IrPhi) or isinstance(expr, int):
        return
    if expr not in visited_expr:
        visited_expr[expr] = {node}
        for i in range(len(expr.children)):
            check_expr_visit(expr.children[i], node, visited_expr, visited_order)        
    else:
        if expr not in visited_order:
        # if len(visited_expr[expr]) == 1:
            visited_order.append(expr)
        visited_expr[expr].add(node)

def cse_block(block, node, visited_expr, visited_order):
    ir_list = block.children
    for ir in ir_list:
        if isinstance(ir, IrAssignment):
            check_expr_visit(ir.children[1], node, visited_expr, visited_order)
        elif isinstance(ir, IrTransRetBasic):
            for j in range(len(ir.children)):
                check_expr_visit(ir.children[j], node, visited_expr, visited_order)

def replace_all_occurrences_expr(expr, sub_expr, var):
    if expr == sub_expr:
        return var
    # flag = False
    if isinstance(expr, int):
        return expr
    for i in range(len(expr.children)):
        new_child = replace_all_occurrences_expr(expr.children[i], sub_expr, var)
        # if replaced:
        expr.children[i] = new_child
        # flag = replaced
    return expr

def replace_all_occurrences_block(block, new_assignment):
    ir_list = block.children
    # index = []
    for i in range(len(ir_list)):
        if isinstance(ir_list[i], IrAssignment):
            new_expr = replace_all_occurrences_expr(ir_list[i].children[1], new_assignment.children[1], new_assignment.children[0])
            new_children = [ir_list[i].children[0], new_expr]
            ir_list[i].update_parent_child(new_children)
        elif isinstance(ir_list[i], IrTransRetBasic):
            new_children = []
            for j in range(len(ir_list[i].children)):
                new_expr = replace_all_occurrences_expr(ir_list[i].children[j], new_assignment.children[1], new_assignment.children[0])
                new_children.append(new_expr)
            ir_list[i].update_parent_child(new_children)
    if block.inner_jump != None:
        block.inner_jump[0] = replace_all_occurrences_expr(block.inner_jump[0], new_assignment.children[1], new_assignment.children[0])
    if block.jump != None:
        block.jump[0] = replace_all_occurrences_expr(block.jump[0], new_assignment.children[1], new_assignment.children[0])


def check_ancestor(dtree, ancestor, nodes):
    if ancestor in nodes:
        nodes.remove(ancestor)
    for child in dtree.successors[ancestor]:
        check_ancestor(dtree, child, nodes)
    if len(nodes)>0:
        return False 
    return True

def compute_ancestor(occurrences, dtree, node):
    for child in dtree.successors[node]:
        temp = copy.deepcopy(occurrences)
        if check_ancestor(dtree, child, temp):
            return compute_ancestor(occurrences, dtree, child)
    return node
    

def check_occurrence(ir, var):
    if ir == var:
        return True
    if isinstance(ir, int):
        return False
    occurrs = False
    for i in range(len(ir.children)):
        occurrs = occurrs or check_occurrence(ir.children[i], var)
    return occurrs

def add_assignment(assignment, occurrences, cfg, dtree):
    node = compute_ancestor(occurrences, dtree, cfg.entry_node)
    ir_list = cfg.ir[node].children
    # parent_ir = ir_list[0].parents[0]
    index = -1
    if node in occurrences:
        for l in ir_list:
            if check_occurrence(l, assignment.children[0]):
                index = ir_list.index(l)
                break
    else:
        index = len(ir_list)
        # index = ir_list.index(cfg.ir[node][-1])+1
    # new_children = ir_list
    ir_list.insert(index, assignment)
    # parent_ir.update_parent_child(new_children)
            
def create_new_assignments(visited_order, visited_expr, cfg, dtree):
    for i in range(len(visited_order)-1, -1, -1):
        original_expr = visited_order[i]
        new_var = IrVar(get_var(), original_expr.irMetadata)
        new_assignment = IrAssignment(new_var, original_expr)
        new_var.defs = new_assignment
        for node in cfg.nodes:
            block = cfg.ir[node]
            replace_all_occurrences_block(block, new_assignment)
        add_assignment(new_assignment, visited_expr[original_expr], cfg, dtree)

def cse_cfg(cfg, dtree):
    visited_expr = {}
    visited_order = []
    for node in cfg.nodes:
        cse_block(cfg.ir[node], node, visited_expr, visited_order)
    print(len(visited_order))
    create_new_assignments(visited_order, visited_expr, cfg, dtree)

def cse(ir):
    for transformer in ir.tstore.keys():
        for i in range(len(ir.tstore[transformer])):
            cfg = ir.tstore[transformer][i].cfg
            dtree = representations.construct_dominator_tree(cfg)
            cse_cfg(cfg, dtree)