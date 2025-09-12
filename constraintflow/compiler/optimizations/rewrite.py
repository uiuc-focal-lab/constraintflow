from constraintflow.compiler.ir import *
from constraintflow.compiler.optimizations import uses


counter = -1

def get_var(flag=False):
    global counter 
    counter += 1
    if flag:
        return 'rewrite_new_' + str(counter)
    return 'rewrite_' + str(counter)

def is_div(expr):
    if isinstance(expr, IrPhi):
        return False
    if isinstance(expr, int):
        return False
    if isinstance(expr, IrMult):
        if expr.op == '/':
            return True
    if isinstance(expr, IrVar) and expr.defs is not None:
        return is_div(expr.defs.children[1])
    for child in expr.children:
        if is_div(child):
            return True
    return False

def is_ternary(expr):
    if isinstance(expr, int):
        return False
    if isinstance(expr, IrTernary):
        return True
    for child in expr.children:
        if is_ternary(child):
            return True
    return False

# torch.where(c, e1, e2) = c * e1 + (1-c) * e2
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
        if not is_div(expr) :
            new_name = get_var(True)
            new_var = IrVar(new_name, expr.children[0].irMetadata)
            new_assignment = IrAssignment(new_var, IrConvertBoolToFloat(expr.children[0]))
            new_lhs = IrMult(new_var, expr.children[1], '*')
            new_rhs = IrMult(IrBinaryOp(IrConst(1.0, 'Float'), new_var, '-'), expr.children[2], '*')
            new_expr = IrBinaryOp(new_lhs, new_rhs, '+')
            new_assignments.append(new_assignment)
            expr = new_expr
    
    return expr, new_assignments

def percolate_repeat_and_add_dimension(expr):
    if isinstance(expr, int):
        return expr
    if isinstance(expr, IrVar):
        return percolate_repeat_and_add_dimension(expr.defs.children[1])
    if isinstance(expr, IrRepeat):
        new_input = percolate_repeat_and_add_dimension(expr.children[0])
        if isinstance(new_input, IrBinaryOp):
            new_lhs = IrRepeat(new_input.children[0], repeat_dims = expr.children[1:])
            new_rhs = IrRepeat(new_input.children[1], repeat_dims = expr.children[1:])
            expr = IrBinaryOp(new_lhs, new_rhs, new_input.op)
            return expr
        else:
            return expr
        
    elif isinstance(expr, IrAddDimension):
        new_input = percolate_repeat_and_add_dimension(expr.children[0])
        new_type = new_input.children[0].irMetadata[-1].type
        irMetadataElement = expr.irMetadata[-1].copy()
        irMetadataElement.type = new_type
        if isinstance(new_input, IrBinaryOp):
            new_lhs = IrAddDimension(new_input.children[0], irMetadataElement)
            new_rhs = IrAddDimension(new_input.children[1], irMetadataElement)
            expr = IrBinaryOp(new_lhs, new_rhs, new_input.op)
            return expr
        else:
            return expr
    else:
        return expr
    
def depercolate_repeat_and_add_dimension(expr):
    if isinstance(expr, int):
        return expr
    if isinstance(expr, IrVar):
        return depercolate_repeat_and_add_dimension(expr.defs.children[1])
    if isinstance(expr, IrClamp):
        if isinstance(expr.children[0], IrRepeat):
            new_input = depercolate_repeat_and_add_dimension(expr.children[0].children[0])
            expr = IrRepeat(IrClamp(new_input, expr.const, expr.min_true), repeat_dims = expr.children[0].children[1:])
        elif isinstance(expr.children[0], IrAddDimension):
            new_input = depercolate_repeat_and_add_dimension(expr.children[0].children[0])
            expr = IrAddDimension(IrClamp(new_input, expr.const, expr.min_true), expr.children[0].irMetadata[-1])
        return expr
    else:
        return expr

def rewrite_expr_ternary_to_clamp(expr):
    if isinstance(expr, int):
        return expr, []
    new_children = []
    new_assignments = []
    for child in expr.children:
        new_child, new_assignment = rewrite_expr_ternary_to_clamp(child)
        new_children.append(new_child)
        new_assignments += new_assignment
    expr.update_parent_child(new_children)
    if isinstance(expr, IrTernary):
        if not is_div(expr) :
            cond, lhs, rhs = expr.children
            if isinstance(cond, IrVar) and cond.defs is not None:
                cond = cond.defs.children[1]
            if isinstance(lhs, IrVar) and lhs.defs is not None:
                lhs = lhs.defs.children[1]
            if isinstance(rhs, IrVar) and rhs.defs is not None:
                rhs = rhs.defs.children[1]
            new_cond = percolate_repeat_and_add_dimension(cond) 
            cond = new_cond
            if isinstance(cond, IrBinaryOp):
                if cond.op in ['>=', '>']:
                    lhs_min = True
                elif cond.op in ['<=', '<']:
                    lhs_min = False
                else:
                    pass
                cond_lhs, cond_rhs = cond.children
                if isinstance(cond_lhs, IrVar) and cond_lhs.defs is not None:
                    cond_lhs = cond_lhs.defs.children[1]
                if get_const(cond_rhs) == 0: 
                    if isinstance(lhs, IrMult) and lhs.op == '*' and isinstance(rhs, IrMult) and rhs.op == '*':
                        in_lhs = False
                        in_lhs_index = -1
                        in_rhs = False
                        in_rhs_index = -1
                        for i, child in enumerate(lhs.children):                                
                            if isinstance(child, IrVar) and child.defs is not None:
                                child = child.defs.children[1]
                            if child == cond_lhs:
                                in_lhs = True
                                in_lhs_index = i
                                break
                        for i, child in enumerate(rhs.children):
                            if isinstance(child, IrVar) and child.defs is not None:
                                child = child.defs.children[1]
                            if child == cond_lhs:
                                in_rhs_index = i
                                in_rhs = True
                                break
                        if in_lhs and in_rhs:
                            other_lhs = lhs.children[1-in_lhs_index]
                            other_rhs = rhs.children[1-in_rhs_index]
                            new_lhs = IrMult(depercolate_repeat_and_add_dimension(IrClamp(cond_lhs, 0, min_true=lhs_min)), other_lhs, '*')
                            new_rhs = IrMult(depercolate_repeat_and_add_dimension(IrClamp(cond_lhs, 0, min_true=not(lhs_min))), other_rhs, '*')
                            expr = IrBinaryOp(new_lhs, new_rhs, '+')
    return expr, new_assignments


# e1 * c + e2 * c = (e1 + e2) * c
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

def rewrite_expr_6(expr):
    if isinstance(expr, int):
        return expr, []
    new_children = []
    new_assignments = []
    for child in expr.children:
        new_child, new_assignment = rewrite_expr_6(child)
        new_children.append(new_child)
        new_assignments += new_assignment
    expr.update_parent_child(new_children)
    if isinstance(expr, IrReduce):
        child = expr.children[0]
        if isinstance(child, IrBinaryOp) and (child.op == '+' or child.op == '-'):
            lhs = child.children[0]
            rhs = child.children[1]
            new_lhs, new_lhs_assignments = rewrite_expr_6(IrReduce(lhs))
            new_rhs, new_rhs_assignments = rewrite_expr_6(IrReduce(rhs))
            new_expr = IrBinaryOp(new_lhs, new_rhs, child.op)
            return new_expr, [] + new_lhs_assignments + new_rhs_assignments
    return expr, new_assignments

# Bring repeat out of add dimensions
def rewrite_expr_3(expr):
    if isinstance(expr, int):
        return expr, []
    new_children = []
    new_assignments = []
    for child in expr.children:
        new_child, new_assignment = rewrite_expr_3(child)
        new_children.append(new_child)
        new_assignments += new_assignment
    expr.update_parent_child(new_children)
    if isinstance(expr, IrBinaryOp) or isinstance(expr, IrMult):
        [lhs, rhs] = expr.children
        if isinstance(lhs, IrVar):
            lhs = lhs.defs.children[1]
        if isinstance(rhs, IrVar):
            rhs = rhs.defs.children[1]
        if (isinstance(lhs, IrRepeat) and isinstance(rhs, IrRepeat)) or (isinstance(lhs, IrAddDimensionConst) and isinstance(rhs, IrRepeat)) or (isinstance(lhs, IrRepeat) and isinstance(rhs, IrAddDimensionConst)) or (isinstance(lhs, IrAddDimensionConst) and isinstance(rhs, IrAddDimensionConst)):
            repeat_dims_lhs, repeat_dims_rhs = lhs.children[1:], rhs.children[1:]
            assert(len(repeat_dims_lhs) == len(repeat_dims_rhs))
            repeat_dims_hcf = []
            repeat_dims_new_lhs = []
            repeat_dims_new_rhs = []
            for i in range(len(repeat_dims_lhs)):
                if repeat_dims_lhs[i] == repeat_dims_rhs[i]:
                    repeat_dims_hcf.append(repeat_dims_lhs[i])
                    repeat_dims_new_lhs.append(1)
                    repeat_dims_new_rhs.append(1)
                elif (repeat_dims_lhs[i] == 1) or (repeat_dims_rhs[i] == 1):
                    repeat_dims_hcf.append(1)
                    repeat_dims_new_lhs.append(repeat_dims_lhs[i])
                    repeat_dims_new_rhs.append(repeat_dims_rhs[i])
                else:
                    lhs_multiplicands = collect_multiplicands(lhs)
                    rhs_multiplicands = collect_multiplicands(rhs)
                    common = list((set(lhs_multiplicands)).intersection(set(rhs_multiplicands)))
                    if common != []:
                        lhs_diff = lhs_multiplicands
                        for k in range(len(common)):
                            for l in range(len(lhs_diff)-1, -1, -1):
                                if common[k] == lhs_diff[l]:
                                    del lhs_diff[l]
                                    break
                        rhs_diff = rhs_multiplicands
                        for k in range(len(common)):
                            for l in range(len(rhs_diff)-1, -1, -1):
                                if common[k] == rhs_diff[l]:
                                    del rhs_diff[l]
                                    break
                        hcf_elem = common[0]
                        for j in range(1, len(common)):
                            hcf_elem = IrMult(hcf_elem, common[j], '*')
                        lhs_multiplicand_elem = 1
                        for j in range(len(lhs_diff)):
                            lhs_multiplicand_elem = mult_metadata(lhs_multiplicand_elem, lhs_diff[j])
                        rhs_multiplicand_elem = 1
                        for j in range(len(rhs_diff)):
                            rhs_multiplicand_elem = mult_metadata(rhs_multiplicand_elem, rhs_diff[j])
                        repeat_dims_hcf.append(hcf_elem)
                        repeat_dims_new_lhs.append(lhs_multiplicand_elem)
                        repeat_dims_new_rhs.append(rhs_multiplicand_elem)
                    else:
                        repeat_dims_hcf.append(1)
                        repeat_dims_new_lhs.append(repeat_dims_lhs[i])
                        repeat_dims_new_rhs.append(repeat_dims_rhs[i])
            
            flag = False 
            for i in range(len(repeat_dims_hcf)):
                if repeat_dims_hcf[i]!=1:
                    flag = True 
                    break
            if flag:
                if isinstance(lhs, IrAddDimensionConst):
                    lhs_flag = True
                else:
                    lhs_flag = False 
                    for i in range(len(repeat_dims_new_lhs)):
                        if repeat_dims_new_lhs[i]!=1:
                            lhs_flag = True 
                            break
                if isinstance(rhs, IrAddDimensionConst):
                    rhs_flag = True
                else:
                    rhs_flag = False 
                    for i in range(len(repeat_dims_new_rhs)):
                        if repeat_dims_new_rhs[i]!=1:
                            rhs_flag = True 
                            break
                new_children = []
                if lhs_flag:
                    if isinstance(lhs, IrRepeat):
                        lhs = IrRepeat(lhs.children[0], repeat_dims=repeat_dims_new_lhs)
                    if isinstance(lhs, IrAddDimensionConst):
                        lhs = lhs.children[0]
                        assert(isinstance(lhs, IrConst))
                else:
                    lhs = lhs.children[0]
                if rhs_flag:
                    if isinstance(rhs, IrRepeat):
                        rhs = IrRepeat(rhs.children[0], repeat_dims=repeat_dims_new_rhs)
                    if isinstance(rhs, IrAddDimensionConst):
                        rhs = rhs.children[0]
                else:
                    rhs = rhs.children[0]
                new_children = [lhs, rhs]
                if isinstance(expr, IrBinaryOp):
                    expr = IrBinaryOp(lhs, rhs, expr.op)
                elif isinstance(expr, IrMult): 
                    expr = IrMult(lhs, rhs, expr.op)
                expr = IrRepeat(expr, None, repeat_dims_hcf)

    elif isinstance(expr, IrUnaryOp) and expr.op != 'get_dims':
        child = expr.children[0]
        if isinstance(child, IrVar):
            child = child.defs.children[1]
        if isinstance(child, IrRepeat):
            last_irMetadata = child.irMetadata
            new_unary_op = IrUnaryOp(child.children[0], expr.op)
            expr = IrRepeat(new_unary_op, last_irMetadata)
    elif isinstance(expr, IrConvertBoolToFloat):
        child = expr.children[0]
        if isinstance(child, IrVar):
            child = child.defs.children[1]
        if isinstance(child, IrRepeat):
            last_irMetadata = child.irMetadata
            last_irMetadata[-1].type = 'Float'
            new_unary_op = IrConvertBoolToFloat(child.children[0])
            expr = IrRepeat(new_unary_op, last_irMetadata)
    
    return expr, []


def compute_dim_difference(expr):
    if isinstance(expr, IrAddDimension):
        return len(expr.irMetadata[-1].shape) - len(expr.children[0].irMetadata[-1].shape)
    elif isinstance(expr, IrAddDimensionConst):
        s = 0
        for i in range(len(expr.irMetadata)):
            s += len(expr.irMetadata[i].shape)
        return s


# Bring add dimension outside
def rewrite_expr_4(expr):
    if isinstance(expr, int):
        return expr, []
    new_children = []
    new_assignments = []
    for child in expr.children:
        new_child, new_assignment = rewrite_expr_4(child)
        new_children.append(new_child)
        new_assignments += new_assignment
    expr.update_parent_child(new_children)
    if isinstance(expr, IrBinaryOp) or isinstance(expr, IrMult):
        [lhs, rhs] = expr.children
        if isinstance(lhs, IrVar):
            lhs = lhs.defs.children[1]
        if isinstance(rhs, IrVar):
            rhs = rhs.defs.children[1]
        if (isinstance(lhs, IrAddDimension) and isinstance(rhs, IrAddDimension)) or (isinstance(lhs, IrAddDimensionConst) and isinstance(rhs, IrAddDimension)) or (isinstance(lhs, IrAddDimension) and isinstance(rhs, IrAddDimensionConst)) or (isinstance(lhs, IrAddDimensionConst) and isinstance(rhs, IrAddDimensionConst)):
            num_elements_to_remove = min(compute_dim_difference(lhs), compute_dim_difference(rhs))
            if num_elements_to_remove > 0:
                if(isinstance(lhs, IrAddDimensionConst) and isinstance(rhs, IrAddDimensionConst)):
                    if isinstance(expr, IrBinaryOp):
                        expr = IrAddDimensionConst(IrBinaryOp(lhs.children[0], rhs.children[0], expr.op), copy_metadata(lhs.irMetadata))
                    elif isinstance(expr, IrMult):
                        expr = IrAddDimensionConst(IrMult(lhs.children[0], rhs.children[0], expr.op), copy_metadata(lhs.irMetadata))
                elif(isinstance(lhs, IrAddDimension) and isinstance(rhs, IrAddDimensionConst)):
                    new_irMetadataElement = lhs.irMetadata[-1]
                    lhs = lhs.children[0]
                    if isinstance(expr, IrBinaryOp):
                        expr = IrAddDimension(IrBinaryOp(lhs, rhs.children[0], expr.op), new_irMetadataElement)
                    elif isinstance(expr, IrMult):
                        expr = IrAddDimension(IrMult(lhs, rhs.children[0], expr.op), new_irMetadataElement)
                elif(isinstance(rhs, IrAddDimension) and isinstance(lhs, IrAddDimensionConst)):
                    new_irMetadataElement = rhs.irMetadata[-1]
                    rhs = rhs.children[0]
                    if isinstance(expr, IrBinaryOp):
                        expr = IrAddDimension(IrBinaryOp(lhs.children[0], rhs, expr.op), new_irMetadataElement)
                    elif isinstance(expr, IrMult):
                        expr = IrAddDimension(IrMult(lhs.children[0], rhs, expr.op), new_irMetadataElement)
                elif(isinstance(lhs, IrAddDimension) and isinstance(rhs, IrAddDimension)):
                    if compute_dim_difference(lhs) == num_elements_to_remove and compute_dim_difference(rhs) == num_elements_to_remove:
                        new_irMetadataElement = lhs.irMetadata[-1]
                        lhs = lhs.children[0]
                        rhs = rhs.children[0]
                    elif compute_dim_difference(lhs) == num_elements_to_remove:
                        new_irMetadataElement = lhs.irMetadata[-1]
                        lhs = lhs.children[0]
                        rhs.irMetadata.shape = rhs.irMetadata.shape[:-num_elements_to_remove]
                        rhs.irMetadata.broadcast = rhs.irMetadata.broadcast[:-num_elements_to_remove]
                    elif compute_dim_difference(rhs) == num_elements_to_remove:
                        new_irMetadataElement = rhs.irMetadata[-1]
                        rhs = rhs.children[0]
                        lhs.irMetadata.shape = lhs.irMetadata.shape[:-num_elements_to_remove]
                        lhs.irMetadata.broadcast = lhs.irMetadata.broadcast[:-num_elements_to_remove]
                    if isinstance(expr, IrBinaryOp):
                        expr = IrAddDimension(IrBinaryOp(lhs, rhs, expr.op), new_irMetadataElement)
                    elif isinstance(expr, IrMult):
                        expr = IrAddDimension(IrMult(lhs, rhs, expr.op), new_irMetadataElement)
            
    elif isinstance(expr, IrUnaryOp):
        child = expr.children[0]
        if isinstance(child, IrVar):
            child = child.defs.children[1]
        if isinstance(child, IrAddDimension):
            last_irMetadata = child.irMetadata[-1]
            new_unary_op = IrUnaryOp(child.children[0], expr.op)
            expr = IrAddDimension(new_unary_op, last_irMetadata)
    elif isinstance(expr, IrConvertBoolToFloat):
        child = expr.children[0]
        if isinstance(child, IrVar):
            child = child.defs.children[1]
        if isinstance(child, IrAddDimension):
            last_irMetadata = child.irMetadata[-1]
            last_irMetadata.type = 'Float'
            new_unary_op = IrConvertBoolToFloat(child.children[0])
            expr = IrAddDimension(new_unary_op, last_irMetadata)
    
    return expr, []

def get_const(expr):
    if isinstance(expr, int):
        return expr
    if isinstance(expr, IrConst):
        return expr.const
    if isinstance(expr, IrAddDimensionConst):
        return get_const(expr.children[0])
    if isinstance(expr, IrAddDimension):
        return get_const(expr.children[0])
    if isinstance(expr, IrRepeat):
        return get_const(expr.children[0])
    if isinstance(expr, IrVar):
        return get_const(expr.defs.children[1])
    return None
    
def rewrite_expr_clamp(expr):
    if isinstance(expr, int):
        return expr, []
    new_children = []
    new_assignments = []
    for child in expr.children:
        new_child, new_assignment = rewrite_expr_1(child)
        new_children.append(new_child)
        new_assignments += new_assignment
    expr.update_parent_child(new_children)
    if isinstance(expr, IrMult) and expr.op == '*':
        if not is_div(expr):
            lhs = expr.children[0]
            rhs = expr.children[1]
            if isinstance(lhs, IrVar):
                lhs = lhs.defs.children[1]
            if isinstance(lhs, IrConvertBoolToFloat):
                lhs = lhs.children[0]
                if isinstance(lhs, IrVar):
                    lhs = lhs.defs.children[1]
                if isinstance(lhs, IrBinaryOp) and lhs.op == '>=':
                    lhs_lhs, lhs_rhs = lhs.children
                    if get_const(lhs_rhs) == 0: 
                        if lhs_lhs == rhs:
                            expr = IrClamp(rhs, 0)
    
    return expr, new_assignments





def get_size(irMetadata):
    s = 0
    for i in range(len(irMetadata)):
        s += len(irMetadata[i].shape)
    return s

def get_shape(irMetadata):
    shape = []
    for i in range(len(irMetadata)):
        shape += irMetadata[i].shape
    return shape


def rewrite_percolate_reduce_inside_phi(expr):
    if isinstance(expr, int):
        return expr, []
    new_children = []
    new_assignments = []
    for child in expr.children:
        new_child, new_assignment = rewrite_percolate_reduce_inside_phi(child)
        new_children.append(new_child)
        new_assignments += new_assignment
    expr.update_parent_child(new_children)
    if isinstance(expr, IrReduce):
        child = expr.children[0]
        if isinstance(child, IrVar):
            child = child.defs.children[1]
        if isinstance(child, IrPhi):
            can_replace = True
            for phi_child in child.children:
                definition = phi_child.defs 
                if len(definition.uses) != 1:
                    can_replace = False
                    return expr, []
            if can_replace:
                for phi_child in child.children:
                    definition = phi_child.defs 
                    new_child = IrReduce(definition.children[1])
                    definition.update_parent_child([definition.children[0], new_child])
            return child, []
    return expr, []


# Inner prod from elementwise prod
def rewrite_expr_5(expr):
    if isinstance(expr, int):
        return expr, []
    new_children = []
    new_assignments = []
    for child in expr.children:
        new_child, new_assignment = rewrite_expr_5(child)
        new_children.append(new_child)
        new_assignments += new_assignment
    expr.update_parent_child(new_children)
    def get_repeat_dims(expr, indices=[-1,0]):
        repeat_dims = expr.children[1:]
        shape = get_shape(expr.children[0].irMetadata)
        ret = []
        for i in indices:
            if shape[i] == 1 and repeat_dims[i] != 1:
                ret.append(i)
        return ret
    if isinstance(expr, IrReduce) and len(expr.children[0].irMetadata[-1].shape)==2:
        child = expr.children[0]
        if isinstance(child, IrVar):
            child = child.defs.children[1]
        if isinstance(child, IrMult) and child.op == '*':
            [lhs, rhs] = child.children
            if isinstance(lhs, IrVar):
                lhs = lhs.defs.children[1]
            if isinstance(rhs, IrVar):
                rhs = rhs.defs.children[1]
            if isinstance(lhs, IrRepeat) and isinstance(rhs, IrRepeat):
                lhs_end_points = get_repeat_dims(lhs, [-1, -3])
                rhs_end_points = get_repeat_dims(rhs, [-1, -3])

                lhs_size = get_size(lhs.irMetadata)
                rhs_size = get_size(rhs.irMetadata)
                
                if (-1 in lhs_end_points) and (-3 in rhs_end_points):
                    new_lhs = IrRemoveDimension(lhs.children[0], lhs_size-1)
                    new_rhs = IrRemoveDimension(rhs.children[0], rhs_size-3)
                    expr = IrInnerProduct(new_lhs, new_rhs)
                elif (-3 in lhs_end_points) and (-1 in rhs_end_points):
                    new_lhs = IrRemoveDimension(rhs.children[0], lhs_size-3)
                    new_rhs = IrRemoveDimension(lhs.children[0], rhs_size-1)
                    expr = IrInnerProduct(new_lhs, new_rhs)

    if isinstance(expr, IrReduce) and len(expr.children[0].irMetadata[-1].shape)==1:
        child = expr.children[0]
        if isinstance(child, IrVar):
            child = child.defs.children[1]
        if isinstance(child, IrMult) and child.op == '*':
            [lhs, rhs] = child.children
            if isinstance(lhs, IrVar):
                lhs = lhs.defs.children[1]
            if isinstance(rhs, IrVar):
                rhs = rhs.defs.children[1]
            if isinstance(lhs, IrRepeat):
                lhs_end_points = get_repeat_dims(lhs, [-1, -2])
                lhs_size = get_size(lhs.irMetadata)
                rhs_size = get_size(rhs.irMetadata)
                
                if (-1 in lhs_end_points):
                    new_lhs = IrRemoveDimension(lhs.children[0], lhs_size-1)
                    new_rhs = rhs
                    expr = IrInnerProduct(new_lhs, new_rhs)
                elif (-2 in lhs_end_points):
                    new_lhs = rhs
                    new_rhs = IrRemoveDimension(lhs.children[0], rhs_size-2)
                    expr = IrInnerProduct(new_lhs, new_rhs)
            
            if isinstance(rhs, IrRepeat):
                rhs_end_points = get_repeat_dims(rhs, [-1, -2])
                rhs_size = get_size(rhs.irMetadata)

                if (-1 in rhs_end_points):
                    new_lhs = IrRemoveDimension(rhs.children[0], rhs_size-1)
                    new_rhs = lhs
                    expr = IrInnerProduct(new_lhs, new_rhs)
                elif (-2 in rhs_end_points):
                    new_lhs = lhs
                    new_rhs = IrRemoveDimension(rhs.children[0], rhs_size-2)
                    expr = IrInnerProduct(new_lhs, new_rhs)
    return expr, []

def rewrite_percolate_repeat_inside_binary(expr):
    if isinstance(expr, int):
        return expr, []
    new_children = []
    new_assignments = []
    for child in expr.children:
        new_child, new_assignment = rewrite_percolate_repeat_inside_binary(child)
        new_children.append(new_child)
        new_assignments += new_assignment
    expr.update_parent_child(new_children)
    if isinstance(expr, IrReduce):
        child = expr.children[0]
        if isinstance(expr.children[0], IrVar):
            child = child.defs.children[1]
        if isinstance(child, IrBinaryOp) and (child.op == '+' or child.op == '-'):
            expr = IrBinaryOp(IrReduce(child.children[0]), IrReduce(child.children[1]), child.op)
    return expr, []
    

def rewrite_extract_coeff(expr):
    if isinstance(expr, int):
        return expr, []
    new_children = []
    new_assignments = []
    for child in expr.children:
        new_child, new_assignment = rewrite_extract_coeff(child)
        new_children.append(new_child)
        new_assignments += new_assignment
    expr.update_parent_child(new_children)
    if isinstance(expr, IrExtractPolyCoeff) or isinstance(expr, IrExtractSymCoeff):
        parent_expr = expr.children[0] 
        if isinstance(parent_expr, IrVar):
            parent_expr = parent_expr.defs.children[1]
        if(isinstance(parent_expr, IrCombineToPoly) or isinstance(parent_expr, IrCombineToSym)):
            expr = parent_expr.children[0]
    if isinstance(expr, IrExtractPolyConst) or isinstance(expr, IrExtractSymConst):
        parent_expr = expr.children[0] 
        if isinstance(parent_expr, IrVar):
            parent_expr = parent_expr.defs.children[1]
        if(isinstance(parent_expr, IrCombineToPoly) or isinstance(parent_expr, IrCombineToSym)):
            expr = parent_expr.children[1]
    return expr, []



def rewrite_associativity(expr):
    if isinstance(expr, int):
        return expr, []
    new_children = []
    new_assignments = []
    for child in expr.children:
        new_child, new_assignment = rewrite_associativity(child)
        new_children.append(new_child)
        new_assignments += new_assignment
    expr.update_parent_child(new_children)
    
    if isinstance(expr, IrMult) and expr.op == '*':
        def get_multiplicands(expr):
            if isinstance(expr, IrRepeat) or isinstance(expr, IrAddDimension) or isinstance(expr, IrAddDimensionConst):
                return get_multiplicands(expr.children[0])
            elif isinstance(expr, IrVar):
                return get_multiplicands(expr.defs.children[1])
            elif isinstance(expr, IrMult) and expr.op == '*':
                return get_multiplicands(expr.children[0]) + get_multiplicands(expr.children[1])
            else:
                return [expr]
        multiplicands = get_multiplicands(expr)
        if len(multiplicands) == 3:
            if checkEqualMetadata(multiplicands[0].irMetadata, multiplicands[1].irMetadata):
                expr = IrMult(IrMult(multiplicands[0], multiplicands[1], '*'), multiplicands[2], '*')
            elif checkEqualMetadata(multiplicands[0].irMetadata, multiplicands[2].irMetadata):
                expr = IrMult(IrMult(multiplicands[0], multiplicands[2], '*'), multiplicands[1], '*')
            elif checkEqualMetadata(multiplicands[1].irMetadata, multiplicands[2].irMetadata):
                expr = IrMult(multiplicands[0], IrMult(multiplicands[2], multiplicands[1], '*'), '*')
    return expr, []


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
    uses.populate_uses_defs_cfg(cfg)
    for node in cfg.nodes:
        block = cfg.ir[node]
        rewrite_block(block, rewrite_expr_ternary_to_clamp)
    uses.populate_uses_defs_cfg(cfg)
    for node in cfg.nodes:
        block = cfg.ir[node]
        rewrite_block(block, rewrite_extract_coeff)
    uses.populate_uses_defs_cfg(cfg)
    for node in cfg.nodes:
        block = cfg.ir[node]
        rewrite_block(block, rewrite_expr_1)
    uses.populate_uses_defs_cfg(cfg)
    for node in cfg.nodes:
        block = cfg.ir[node]
        rewrite_block(block, rewrite_expr_6)
    uses.populate_uses_defs_cfg(cfg)
    for node in cfg.nodes:
        block = cfg.ir[node]
        rewrite_block(block, rewrite_expr_3)
    for node in cfg.nodes:
        block = cfg.ir[node]
        rewrite_block(block, rewrite_expr_4)
    # uses.populate_uses_defs_cfg(cfg)
    for node in cfg.nodes:
        block = cfg.ir[node]
        rewrite_block(block, rewrite_associativity)
    # uses.populate_uses_defs_cfg(cfg)
    uses.populate_uses_defs_cfg(cfg)
    for node in cfg.nodes:
        block = cfg.ir[node]
        rewrite_block(block, rewrite_percolate_reduce_inside_phi)
    for node in cfg.nodes:
        block = cfg.ir[node]
        rewrite_block(block, rewrite_percolate_repeat_inside_binary)
    for node in cfg.nodes:
        block = cfg.ir[node]
        rewrite_block(block, rewrite_expr_5)
    uses.populate_uses_defs_cfg(cfg)
    for node in cfg.nodes:
        block = cfg.ir[node]
        rewrite_block(block, rewrite_expr_clamp)
    for node in cfg.nodes:
        block = cfg.ir[node]
        rewrite_block(block, rewrite_expr_2)
    
def rewrite(ir):
    for transformer in ir.tstore.keys():
        for i in range(len(ir.tstore[transformer])):
            cfg = ir.tstore[transformer][i].cfg
            rewrite_cfg(cfg)
    return ir