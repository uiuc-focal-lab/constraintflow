from ir_ast_stack2 import *
from representations import *
counter = -1

def get_var():
    global counter 
    counter += 1
    return 'rewrite_' + str(counter)

def rewrite_expr_1(expr):
    if isinstance(expr, int):
        return expr, []
    new_children = []
    new_assignments = []
    for child in expr.children:
        new_child, new_assignment = rewrite_expr_1(child)
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

def rewrite_expr_2(expr):
    if isinstance(expr, int):
        return expr, []
    new_children = []
    new_assignments = []
    for child in expr.children:
        new_child, new_assignment = rewrite_expr_2(child)
        new_children.append(new_child)
        new_assignments += new_assignment
    expr.update_parent_child(new_children)
    if isinstance(expr, IrBinaryOp):
        if expr.op == '+':
            lhs = expr.children[0]
            rhs = expr.children[1]
            def collect_multiplicands(expr):
                if not (isinstance(expr, IrMult) and (expr.op == '*')):
                    return [expr]
                multiplicands = []
                for child in expr.children:
                    temp = collect_multiplicands(child)
                    multiplicands.extend(temp)
                return multiplicands
            lhs_multiplicands = collect_multiplicands(lhs)
            rhs_multiplicands = collect_multiplicands(rhs)
            if len(lhs_multiplicands) > 1 and len(rhs_multiplicands)>1:
                common = list(set(lhs_multiplicands).intersection(set(rhs_multiplicands)))
                if len(common)==0:
                    return expr, []
                common = list(set(lhs_multiplicands).intersection(set(rhs_multiplicands)))[0]
                new_lhs = common 
                lhs_multiplicands.remove(common)
                rhs_multiplicands.remove(common)
                new_rhs_lhs = lhs_multiplicands[0]
                new_rhs_rhs = rhs_multiplicands[0]
                for i in range(1, len(lhs_multiplicands)):
                    new_rhs_lhs = IrMult(new_rhs_lhs, lhs_multiplicands[i], '*')
                for i in range(1, len(rhs_multiplicands)):
                    new_rhs_rhs = IrMult(new_rhs_rhs, rhs_multiplicands[i], '*')
                new_rhs = IrBinaryOp(new_rhs_lhs, new_rhs_rhs, expr.op)
                new_expr = IrMult(new_lhs, new_rhs, '*')
                return new_expr, []
    return expr, new_assignments

def rewrite_block(block, rewrite_func):
    ir_list = block.children
    length = len(ir_list)
    index = 0
    for i in range(length):
        l = ir_list[index]
        if isinstance(l, IrAssignment):
            new_expr, new_assignments = rewrite_func(l.children[1])
            new_children = [l.children[0], new_expr]
            l.update_parent_child(new_children)
            for j in range(len(new_assignments)):
                ir_list.insert(index, new_assignments[j])
                index += 1
        elif isinstance(l, IrTransRetBasic):
            new_children = []
            new_assignments = []
            for child in l.children:
                new_expr, new_assignments_inner = rewrite_func(child)
                new_children.append(new_expr)
                new_assignments += new_assignments_inner
            l.update_parent_child(new_children)
            for j in range(len(new_assignments)):
                ir_list.insert(index, new_assignments[j])
                index += 1
        index += 1
    # return ir_list


def rewrite_cfg(cfg):
    for node in cfg.nodes:
        block = cfg.ir[node]
        rewrite_block(block, rewrite_expr_1)
        rewrite_block(block, rewrite_expr_2)

def rewrite(ir):
    for transformer in ir.tstore.keys():
        for i in range(len(ir.tstore[transformer])):
            cfg = ir.tstore[transformer][i].cfg
            rewrite_cfg(cfg)
    return ir