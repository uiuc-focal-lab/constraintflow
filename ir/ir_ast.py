import copy
POLYEXP_SIZE = 1004

def create_minimum_shape(shape, broadcast):
    l1 = []
    l2 = []
    for i in range(len(shape)):
        l1.append(1)
        l2.append(shape[i]*broadcast[i])
    return l1, l2

def create_lcm_shape(shape1, broadcast1, shape2, broadcast2):
    assert(is_convertible(shape1, broadcast1, shape2, broadcast2))
    l1 = []
    l2 = []
    for i in range(len(shape1)):
        if(shape1[i] > shape2[i]):
            assert(shape1[i] % shape2[i] == 0)
            l1.append(shape1[i])
            l2.append(broadcast1[i])
        else:
            assert(shape1[i] % shape2[i] == 0)
            l1.append(shape2[i])
            l2.append(broadcast2[i])
    return l1, l2

class Shape:
    def __init__(self, shape1, broadcast1, split_index, type, shape2=None,  broadcast2=None):
        self.shape1 = shape1  
        self.broadcast1 = broadcast1 
        self.shape2 = shape2  
        self.broadcast2 = broadcast2  
        self.split_index = split_index
        self.type = type 
    
    def copy(self):
        return copy.deepcopy(self)
    
    def __eq__(self,obj):
        return self.shape1 == obj.shape1 and self.broadcast1 == self.broadcast2 and self.split_index == obj.split_index and self.type == obj.type and self.shape1 == obj.shape1 and self.broadcast1 == self.broadcast2


def swap_args_ir(lhs_ir, rhs_ir):
    if(lhs_ir.shape.type == "Float"):
        if(rhs_ir.shape.type == "Int"):
            return rhs_ir, lhs_ir
    elif(lhs_ir.shape.type == "PolyExp"):
        return rhs_ir, lhs_ir
    return lhs_ir, rhs_ir


def convert_const(self, ir_node, new_shape):
    temp_shape, temp_broadcast = create_minimum_shape(new_shape.shape1, new_shape.broadcast1)
    add_dim_ir_node = Ir_add_dimension(ir_node, temp_shape, temp_broadcast, new_shape.split_index)
    broadcast_ir_node = Ir_broadcast(add_dim_ir_node, new_shape)
    return broadcast_ir_node

def adjust_shape(self, lhs_ir, rhs_ir):
    lhs_shape = lhs_ir.shape
    rhs_shape = rhs_ir.shape
    if(lhs_shape.shape1 == [1] and rhs_shape.shape1 != [1]):
        return convert_const(lhs_ir, rhs_shape), rhs_ir
    elif(rhs_shape.shape1 == [1] and lhs_shape.shape1 != [1]):
        return lhs_ir, convert_const(rhs_ir, lhs_shape)
    else:
        curr_lhs = lhs_ir
        curr_rhs = rhs_ir
        if(len(lhs_shape.shape1) < len(rhs_shape.shape1)):
            new_shape1, new_broadcast1 = create_minimum_shape(rhs_shape.shape1[len(lhs_shape.shape1):], rhs_shape.broadcast1[len(lhs_shape.broadcast1):])
            new_shape1 = lhs_shape.shape1 + new_shape1
            new_broadcast1 = lhs_shape.broadcast1 + new_broadcast1
            add_dim_ir_node = Ir_add_dimension(lhs_ir, new_shape1, new_broadcast1, lhs_shape.split_index)
            curr_lhs = add_dim_ir_node
        elif(len(rhs_shape.shape1) < len(lhs_shape.shape1)):
            new_shape1, new_broadcast1 = create_minimum_shape(lhs_shape.shape1[len(rhs_shape.shape1):], lhs_shape.broadcast1[len(rhs_shape.broadcast1):])
            new_shape1 = rhs_shape.shape1 + new_shape1
            new_broadcast1 = rhs_shape.broadcast1 + new_broadcast1
            add_dim_ir_node = Ir_add_dimension(rhs_ir, new_shape1, new_broadcast1, rhs_shape.split_index)
            curr_rhs = add_dim_ir_node

        lcm_shape, lcm_broadcast = create_lcm_shape(curr_lhs.shape.shape1, curr_lhs.shape.broadcast1, curr_rhs.shape.shape1, curr_rhs.shape.broadcast1)
        if(curr_lhs.shape.shape1 != lcm_shape):
            curr_lhs = Ir_broadcast(curr_lhs, Shape(lcm_shape, lcm_broadcast, curr_lhs.shape.split_index, curr_lhs.shape.type))
        if(curr_rhs.shape.shape1 != lcm_shape):
            curr_rhs = Ir_broadcast(curr_rhs, Shape(lcm_shape, lcm_broadcast, curr_rhs.shape.split_index, curr_rhs.shape.type))
        return curr_lhs, curr_rhs



class Ir_ast:
    def __init__(self):
        self.node_name = "IR_ASTNode"
        

def is_convertible(shape1, shape2, broadcast1, broadcast2):
    if(shape1 == None and shape2 == None):
        return True

    if(len(shape1) != len(shape2)):
        return False
    
    for i in range(len(shape1)):
        if(shape1[i]*broadcast1[i] != shape2[i] * broadcast2[i]):
            return False
    return True

class Ir_const(Ir_ast):
    def __init__(self, const, type):
        self.const = const 
        self.shape = Shape([1],[1],0,type)

class Ir_var(Ir_ast):
    def __init__(self, name, shape):
        self.name = name 
        self.shape = shape
        
class Ir_broadcast(Ir_ast):
    def __init__(self, input_ir, shape):
        assert(is_convertible(input_ir.shape.shape1, shape.shape1, input_ir.shape.broadcast1, shape.broadcast1))
        assert(is_convertible(input_ir.shape.shape2, shape.shape2, input_ir.shape.broadcast2, shape.broadcast2))
        assert(input_ir.shape.split_index == shape.split_index)
        self.shape = shape
        self.input_ir = input_ir

class Ir_add_dimension(Ir_ast):
    def __init__(self, input_ir, shape, broadcast, split_index):
        self.shape = Shape(shape, broadcast, split_index, input_ir.type)
        self.input_ir = input_ir

class Ir_convert_neuron_to_poly(Ir_ast):
    def __init__(self, input_ir):
        super().__init__()
        self.input_ir = input_ir 
        input_shape = input_ir.shape
        new_shape = input_shape.shape1.copy() + [POLYEXP_SIZE]
        new_broadcast = input_shape.broadcast1.copy() + [1]
        self.shape = Shape(new_shape, new_broadcast, input_shape.split_index, 'PolyExp', input_shape.shape1.copy(), input_shape.broadcast1.copy())

class Ir_convert_const_to_poly(Ir_ast):
    def __init__(self, input_ir):
        super().__init__()
        self.input_ir = input_ir 
        input_shape = input_ir.shape
        new_shape = input_shape.shape1.copy() + [POLYEXP_SIZE]
        new_broadcast = input_shape.broadcast1.copy() + [1]
        self.shape = Shape(new_shape, new_broadcast, input_shape.split_index, 'PolyExp', input_shape.shape1.copy(), input_shape.broadcast1.copy())

class Ir_extract_poly_coeff(Ir_ast):
    def __init__(self, input_ir):
        super().__init__()
        self.input_ir = input_ir 
        input_shape = input_ir.shape
        assert(input_shape.type == 'PolyExp')
        self.shape = Shape(input_shape.shape1, input_shape.broadcast1, input_shape.split_index, 'Float', None, None)

class Ir_extract_poly_const(Ir_ast):
    def __init__(self, input_ir):
        super().__init__()
        self.input_ir = input_ir 
        input_shape = input_ir.shape
        assert(input_shape.type == 'PolyExp')
        self.shape = Shape(input_shape.shape2, input_shape.broadcast2, input_shape.split_index, 'Float', None, None)



class Ir_access(Ir_ast):
    def __init__(self, lhs_ir, rhs_ir, is_metadata=False):
        super().__init__()
        self.lhs_ir = lhs_ir
        self.rhs_ir = rhs_ir
        self.is_metadata = is_metadata

        lhs_shape = lhs_ir.shape
        rhs_shape = rhs_ir.shape

        if rhs_shape.type == 'Int' or rhs_shape.type == 'Float':
            self.shape = lhs_shape.copy()
            self.shape.type = rhs_shape.type
        else:
            new_shape = lhs_shape.shape1 + [POLYEXP_SIZE]
            new_broadcast = lhs_shape.broadcast1 + [1]
            self.shape = Shape(new_shape, new_broadcast, lhs_shape.split_index, 'PolyExp', lhs_shape.shape1.copy(), lhs_shape.broadcast1.copy())


class Ir_add_simple(Ir_ast):
    def __init__(self, lhs_ir, rhs_ir, op):
        super().__init__()
        self.lhs_ir = lhs_ir
        self.rhs_ir = rhs_ir
        self.op = op
        
        lhs_shape = self.lhs_ir.shape
        rhs_shape = self.rhs_ir.shape

        self.lhs_ir, self.rhs_ir = adjust_shape(self.lhs_ir, self.rhs_ir)
        self.shape = self.lhs_ir.shape.copy()
        new_type = 'Float' if lhs_shape.type!=rhs_shape.type else lhs_shape.type
        if self.op in ['>', '>=', '==', '<', '<=']:
            new_type = 'Bool'
        self.shape.type = new_type

class Ir_add_poly(Ir_ast):
    def __init__(self, lhs_ir, rhs_ir, op):
        super().__init__()
        self.lhs_ir = lhs_ir
        self.rhs_ir = rhs_ir
        self.op = op
        
        if(lhs_ir.shape.type == "Neuron"):
            self.lhs_ir = Ir_convert_neuron_to_poly(lhs_ir)
        if(rhs_ir.shape.type == "Neuron"):
            self.rhs_ir = Ir_convert_neuron_to_poly(rhs_ir)

        self.lhs_ir,self.rhs_ir = swap_args_ir(self.lhs_ir, self.rhs_ir)
        lhs_shape = self.lhs_ir.shape
        rhs_shape = self.rhs_ir.shape

        if(lhs_shape.type == "Float" or lhs_shape.type == "Int"):
            new_lhs_ir = Ir_extract_poly_coeff(self.rhs_ir)
            new_rhs_ir = Ir_add_simple(self.lhs_ir, Ir_extract_poly_const(self.rhs_ir), self.op)

            self.lhs_ir = new_lhs_ir #this is the polyexp coeff matrix
            self.rhs_ir = new_rhs_ir #this is the polyexp const matrix

            self.shape = Shape(lhs_shape.shape1, lhs_shape.broadcast1, lhs_shape.split_index, 'PolyExp', rhs_shape.shape1, rhs_shape.broadcast1)
        else:
            new_lhs_ir = Ir_add_simple(Ir_extract_poly_coeff(self.lhs_ir), Ir_extract_poly_coeff(self.rhs_ir), self.op)
            new_rhs_ir = Ir_add_simple(Ir_extract_poly_const(self.lhs_ir), Ir_extract_poly_const(self.rhs_ir), self.op)

            self.lhs_ir = new_lhs_ir #this is the polyexp coeff matrix
            self.rhs_ir = new_rhs_ir #this is the polyexp const matrix

            self.shape = Shape(lhs_shape.shape1, lhs_shape.broadcast1, lhs_shape.split_index, 'PolyExp', rhs_shape.shape1, rhs_shape.broadcast1)



class Ir_mult_simple(Ir_ast):
    def __init__(self, lhs_ir, rhs_ir, op):
        super().__init__()
        self.lhs_ir = lhs_ir
        self.rhs_ir = rhs_ir
        self.op = op

        lhs_shape = self.lhs_ir.shape
        rhs_shape = self.rhs_ir.shape

        self.lhs_ir, self.rhs_ir = adjust_shape(self.lhs_ir, self.rhs_ir)
        self.shape = self.lhs_ir.shape.copy()
        new_type = 'Float' if lhs_shape.type!=rhs_shape.type else lhs_shape.type
        self.shape.type = new_type
                    

class Ir_mult_poly(Ir_ast):
    def __init__(self, lhs_ir, rhs_ir, op):
        super().__init__()
        self.lhs_ir = lhs_ir
        self.rhs_ir = rhs_ir
        self.op = op
        
        if(lhs_ir.shape.type == "Neuron"):
            self.lhs_ir = Ir_convert_neuron_to_poly(lhs_ir)
        if(rhs_ir.shape.type == "Neuron"):
            self.rhs_ir = Ir_convert_neuron_to_poly(rhs_ir)

        self.lhs_ir,self.rhs_ir = swap_args_ir(self.lhs_ir, self.rhs_ir)

        new_lhs_ir = Ir_mult_simple(self.lhs_ir, Ir_extract_poly_coeff(self.rhs_ir))
        new_rhs_ir = Ir_mult_simple(self.lhs_ir, Ir_extract_poly_const(self.rhs_ir))

        self.lhs_ir = new_lhs_ir #this is the polyexp coeff matrix
        self.rhs_ir = new_rhs_ir #this is the polyexp const matrix

        lhs_shape = self.lhs_ir.shape
        rhs_shape = self.rhs_ir.shape

        self.shape = Shape(lhs_shape.shape1, lhs_shape.broadcast1, lhs_shape.split_index, 'PolyExp', rhs_shape.shape1, rhs_shape.broadcast1)



class Ir_ternary_simple(Ir_ast):
    def __init__(self, cond_ir, lhs_ir, rhs_ir):
        super().__init__()
        self.cond_ir = cond_ir
        self.lhs_ir = lhs_ir
        self.rhs_ir = rhs_ir

        self.cond_ir, self.lhs_ir = adjust_shape(self.cond_ir, self.lhs_ir)
        self.cond_ir, self.rhs_ir = adjust_shape(self.cond_ir, self.rhs_ir)
        self.lhs_ir, self.rhs_ir = adjust_shape(self.lhs_ir, self.rhs_ir)

        self.shape = self.cond_ir.shape.copy()
        

class Ir_ternary_poly(Ir_ast):
    def __init__(self, cond_ir, lhs_ir, rhs_ir):
        super().__init__()
        self.cond_ir = cond_ir
        self.lhs_ir = lhs_ir
        self.rhs_ir = rhs_ir

        if(lhs_ir.shape.type == "Neuron"):
            self.lhs_ir = Ir_convert_neuron_to_poly(lhs_ir)
        elif lhs_ir.type == 'Float' or lhs_ir.type == 'Int':
            self.lhs_ir = Ir_convert_const_to_poly(lhs_ir)
        if(rhs_ir.shape.type == "Neuron"):
            self.rhs_ir = Ir_convert_neuron_to_poly(rhs_ir)
        elif rhs_ir.type == 'Float' or rhs_ir.type == 'Int':
            self.rhs_ir = Ir_convert_const_to_poly(rhs_ir)

        lhs_coeff = Ir_extract_poly_coeff(self.lhs_ir)
        lhs_const = Ir_extract_poly_const(self.lhs_ir)

        rhs_coeff = Ir_extract_poly_coeff(self.rhs_ir)
        rhs_const = Ir_extract_poly_const(self.rhs_ir)

        new_lhs_ir = Ir_ternary_simple(self.cond_ir, lhs_coeff, rhs_coeff)
        new_rhs_ir = Ir_ternary_simple(self.cond_ir, lhs_const, rhs_const)

        self.lhs_ir = new_lhs_ir #this is the polyexp coeff matrix
        self.rhs_ir = new_rhs_ir #this is the polyexp const matrix

        lhs_shape = self.lhs_ir.shape
        rhs_shape = self.rhs_ir.shape

        self.shape = Shape(lhs_shape.shape1, lhs_shape.broadcast1, lhs_shape.split_index, 'PolyExp', rhs_shape.shape1, rhs_shape.broadcast1)

class Ir_reduce(Ir_ast):
    def __init__(self, input_ir):
        super().__init__()
        self.input_ir = input_ir 
        input_shape = input_ir.shape 
        new_shape1 = input_shape.shape1.copy() 
        new_broadcast1 = input_shape.broadcast1.copy() 
        if input_shape.split_index == len(input_shape.shape1)-1: 
            new_shape1[-1] = 1
        else:
            new_shape1 = new_shape1[0:input_shape.split_index] + new_shape1[input_shape.split_index+1:-1]
            new_broadcast1 = new_broadcast1[0:input_shape.split_index] + new_broadcast1[input_shape.split_index+1:-1]
        new_shape2 = None
        new_broadcast2 = None 
        if input_shape.shape2 != None:
            new_shape2 = input_shape.shape2.copy() 
            new_broadcast2 = input_shape.broadcast2.copy() 
            if input_shape.split_index == len(input_shape.shape1)-1: 
                new_shape2[-1] = 1
            else:
                new_shape2 = new_shape2[0:input_shape.split_index] + new_shape2[input_shape.split_index+1:-1]
                new_broadcast2 = new_broadcast2[0:input_shape.split_index] + new_broadcast1[input_shape.split_index+1:-1]
        self.shape = Shape(new_shape1, new_broadcast1, input_shape.split_index, input_shape.type, new_shape2, new_broadcast2)

