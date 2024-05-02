import copy
from z3 import *

POLYEXP_SIZE = Int('poly_size')
CURR_SIZE = Int('curr_size')
PREV_SIZE = Int('prev_size')

def check_eq(x, y):
    if isinstance(x, list) and isinstance(y, list):
        if len(x) != len(y):
            return False 
        for i in range(len(x)):
            if not check_eq(x[i], y[i]):
                return False 
        return True
    if type(x) != type(y):
        return False
    if isinstance(x, int):
        return x==y 
    if isinstance(x, IrVar):
        return x==y 
    if isinstance(x, IrMult):
        def collect_multiplicands(expr):
            if not (isinstance(expr, IrMult) and (expr.op == '*')):
                return [expr]
            multiplicands = []
            for child in expr.children:
                multiplicands.extend(collect_multiplicands(child))
            return multiplicands
        x_multiplicands = collect_multiplicands(x)
        y_multiplicands = collect_multiplicands(y)
        if len(x_multiplicands) != len(y_multiplicands):
            return False 
        for i in range(len(x_multiplicands)):
            for j in range(len(y_multiplicands)-1, -1, -1):
                if check_eq(x_multiplicands[i], y_multiplicands[j]):
                    del y_multiplicands[j]
                    break
        if len(y_multiplicands)>0:
            return False
        return True
    # if isinstance(x==y, bool):
    #     return x==y
    # return bool(simplify(x==y))

def mult_metadata(m1, m2):
    if isinstance(m1, int):
        assert (m1==1)
        return m2 
    if isinstance(m2, int):
        assert(m2==1)
        return m1 
    return IrMult(m1, m2, '*')

class IrMetadataElement:
    def __init__(self, shape, type, broadcast, isConst = False):
        self.shape = shape  
        self.broadcast = broadcast
        self.isConst = isConst
        self.type = type 
        assert(len(shape) == len(broadcast))
    
    def copy(self):
        return copy.deepcopy(self)
    
    def copy(self):
        obj = IrMetadataElement([], self.type, [], self.isConst)
        for i in range(len(self.shape)):
            obj.shape.append(self.shape[i])
        for i in range(len(self.broadcast)):
            obj.broadcast.append(self.broadcast[i])
        return obj
    
    def __eq__(self,obj):
        return self.shape == obj.shape and self.type == obj.type and self.isConst == obj.isConst and self.broadcast == obj.broadcast

    def __str__(self):
        print(self.shape)
        print(self.broadcast)
        return ' '

def copy_metadata(irMetadata):
    new_irMetadata = []
    for i in range(len(irMetadata)):
        new_irMetadata.append(irMetadata[i].copy())
    return new_irMetadata
# def checkLargerShape(shape1, shape2):
#     for i in range(len(shape1)):
#         if shape1[i]>shape2[i]:
#             return -1
#         elif shape1[i]<shape2[i]:
#             return 1
#     return 0 

def checkEqualMetadata(irMetadata1, irMetadata2):
    if len(irMetadata1) != len(irMetadata2):
        return False
    for i in range(len(irMetadata1)):
        if len(irMetadata1[i].shape) != len(irMetadata2[i].shape):
            return False 
        for j in range(len(irMetadata1[i].shape)):
            if not check_eq(irMetadata1[i].shape[j], irMetadata2[i].shape[j]):
                return False
            if not check_eq(irMetadata1[i].broadcast[j], irMetadata2[i].broadcast[j]):
                return False
    return True

# def checkSmallerMetadata(irMetadata1, irMetadata2):
#     small_flag = False
#     if len(irMetadata1) != len(irMetadata2):
#         return False
#     for i in range(len(irMetadata1)):
#         if len(irMetadata1[i].shape) != len(irMetadata2[i].shape):
#             return False 
#         for j in range(len(irMetadata1[i].shape)):
#             if check_eq(irMetadata1[i].shape[j], 1):
#                 if not check_eq(irMetadata2[i].shape[j], 1):
#                     small_flag = True
#             else:
#                 if not check_eq(irMetadata1[i].shape[j], irMetadata2[i].shape[j]):
#                     return False
#     return small_flag

def createLcm(irMetadata1, irMetadata2):
    flag1 = False 
    flag2 = False
    # newIrMetadata1 = copy.deepcopy(irMetadata1)
    newIrMetadata1 = copy_metadata(irMetadata1)
    # newIrMetadata2 = copy.deepcopy(irMetadata2)
    newIrMetadata2 = copy_metadata(irMetadata2)
    assert len(irMetadata1) == len(irMetadata2)
    for i in range(len(irMetadata1)):
        assert len(irMetadata1[i].shape) == len(irMetadata2[i].shape)
        for j in range(len(irMetadata1[i].shape)):
            if check_eq(irMetadata1[i].shape[j], 1):
                if not check_eq(irMetadata2[i].shape[j], 1):
                    assert(check_eq(irMetadata1[i].broadcast[j], irMetadata2[i].shape[j]))
                    flag1 = True
                    newIrMetadata1[i].shape[j] = irMetadata2[i].shape[j]
                    newIrMetadata1[i].broadcast[j] = 1
            else:
                if check_eq(irMetadata2[i].shape[j], 1):
                    assert(check_eq(irMetadata2[i].broadcast[j], irMetadata1[i].shape[j]))
                    flag2 = True
                    newIrMetadata2[i].shape[j] = irMetadata1[i].shape[j]
                    newIrMetadata2[i].broadcast[j] = 1
                else:
                    assert check_eq(irMetadata1[i].shape[j], irMetadata2[i].shape[j])
    return flag1, flag2, newIrMetadata1, newIrMetadata2




def canRepeat(irMetadata1, irMetadata2):
    if len(irMetadata1) != len(irMetadata2):
        return False 
    for i in range(len(irMetadata1)):
        assert(irMetadata1[i].type == irMetadata2[i].type)
        assert(irMetadata1[i].isConst == irMetadata2[i].isConst)
        assert(len(irMetadata1[i].shape) == len(irMetadata2[i].shape))
        if not irMetadata1[i].isConst:
            for j in range(len(irMetadata1[i].shape)):
                if not check_eq(mult_metadata(irMetadata1[i].shape[j], irMetadata1[i].broadcast[j]), mult_metadata(irMetadata2[i].shape[j], irMetadata2[i].broadcast[j])):
                # if not check_eq(irMetadata1[i].shape[j] * irMetadata1[i].broadcast[j], irMetadata2[i].shape[j] * irMetadata2[i].broadcast[j]):
                    return False
    return True

def canAddDimension(irMetadataElement1, irMetadataElement2):
    if len(irMetadataElement1.shape) >= len(irMetadataElement2.shape):
        return False 
    for i in range(len(irMetadataElement1.shape)):
        if not check_eq(irMetadataElement1.shape[i], irMetadataElement2.shape[i]):
            return False
        if not check_eq(irMetadataElement1.broadcast[i], irMetadataElement2.broadcast[i]):
            return False
    for i in range(len(irMetadataElement1.shape), len(irMetadataElement2.shape)):
        if irMetadataElement2.shape[i]!=1:
            return False
    return True

def matchDims(lhsIr, rhsIr):
    lhsIrMetadata = lhsIr.irMetadata
    rhsIrMetadata = rhsIr.irMetadata
    if checkEqualMetadata(lhsIr.irMetadata, rhsIr.irMetadata):
        return lhsIr, rhsIr
    if lhsIrMetadata[-1].isConst:
        updated_irMetadata = copy_metadata(rhsIrMetadata)
        # updated_irMetadata = copy.deepcopy(rhsIrMetadata)
        for i in range(len(updated_irMetadata)):
            updated_irMetadata[i].isConst = True 
            updated_irMetadata[i].type = lhsIrMetadata[-1].type
        lhsIr = IrAddDimensionConst(lhsIr, updated_irMetadata)
        return lhsIr, rhsIr

    if rhsIrMetadata[-1].isConst:
        # if len(lhsIrMetadata)>1:
        #     print(dir(lhsIrMetadata[0].shape[0].irMetadata[0]))
        #     print(dir(lhsIrMetadata[1].shape[0].irMetadata[0]))
        # updated_irMetadata = []
        # print('Length of the loop is', len(lhsIrMetadata))
        # for i in range(len(lhsIrMetadata)):
        #     print(i)
        #     new_shape = []
        #     print('lengths are ', len(lhsIrMetadata[i].shape), len(lhsIrMetadata[i].broadcast))
        #     for j in range(len(lhsIrMetadata[i].shape)):
        #         print(i, j)
        #         new_shape.append(copy.deepcopy(lhsIrMetadata[i].shape[j]))
        #     new_broadcast = []
        #     print('done shape')
        #     for j in range(len(lhsIrMetadata[i].broadcast)):
        #         print(i, j)
        #         print(type(lhsIrMetadata[i].broadcast[j]), lhsIrMetadata[i].broadcast[j])
        #         new_broadcast.append(copy.deepcopy(lhsIrMetadata[i].broadcast[j])) 
        #     print('done broadcast')
        #     new_element = IrMetadataElement(new_shape, lhsIrMetadata[-1].type, new_broadcast, lhsIrMetadata[-1].isConst)
        #     updated_irMetadata.append(new_element)
        #     print('Done i=', i)
        # updated_irMetadata = copy.deepcopy(lhsIrMetadata)
        updated_irMetadata = copy_metadata(lhsIrMetadata)
        for i in range(len(updated_irMetadata)):
            updated_irMetadata[i].isConst = True 
            updated_irMetadata[i].type = rhsIrMetadata[-1].type
        rhsIr = IrAddDimensionConst(rhsIr, updated_irMetadata) 
        return lhsIr, rhsIr

    assert(len(lhsIrMetadata) == len(rhsIrMetadata))

    if len(lhsIrMetadata[-1].shape) < len(rhsIrMetadata[-1].shape):
        updated_irMetadataElement = rhsIrMetadata[-1].copy()
        for i in range(len(lhsIrMetadata[-1].shape), len(rhsIrMetadata[-1].shape)):
            updated_irMetadataElement.broadcast[i] = mult_metadata(updated_irMetadataElement.broadcast[i], updated_irMetadataElement.shape[i])
            # updated_irMetadataElement.broadcast[i] *= updated_irMetadataElement.shape[i]
            updated_irMetadataElement.shape[i] = 1
        lhsIr = IrAddDimension(lhsIr, updated_irMetadataElement)

    elif len(rhsIrMetadata[-1].shape) < len(lhsIrMetadata[-1].shape):
        updated_irMetadataElement = lhsIrMetadata[-1].copy()
        for i in range(len(rhsIrMetadata[-1].shape), len(lhsIrMetadata[-1].shape)):
            updated_irMetadataElement.broadcast[i] = mult_metadata(updated_irMetadataElement.broadcast[i], updated_irMetadataElement.shape[i])
            # updated_irMetadataElement.broadcast[i] *= updated_irMetadataElement.shape[i]
            updated_irMetadataElement.shape[i] = 1
        rhsIr = IrAddDimension(rhsIr, updated_irMetadataElement)

    lhsIrMetadata = lhsIr.irMetadata
    rhsIrMetadata = rhsIr.irMetadata

    if checkEqualMetadata(lhsIrMetadata, rhsIrMetadata):
        return lhsIr, rhsIr
    

    flag1, flag2, newIrMetadata1, newIrMetadata2 = createLcm(lhsIr.irMetadata, rhsIr.irMetadata)
    # if lhsIr.irMetadata[-1].isConst:
    if flag1:
        lhsIr = IrRepeat(lhsIr, newIrMetadata1)
    if flag2:
        rhsIr = IrRepeat(rhsIr, newIrMetadata2)
    return lhsIr, rhsIr

class IrAst:
    counter = 0
    poly_size = None
    curr_size = None
    prev_size = None
    trav_size = None
    
    def __init__(self):
        IrAst.counter += 1
        self.node_name = "IR_ASTNode"
        self.identifier = IrAst.counter
        self.parents = []
        self.children = []
        self.irMetadata = []
        
    
    def update_parent_child(self, children):
        self.children = children 
        for child in children:
            if child ==None:
                continue
            if isinstance(child, int):
                continue
            new_parents = set(child.parents)
            new_parents.add(self)
            child.parents = list(new_parents)

    def hash(self):
        self.hash_str = str(type(self))
        for i in self.children:
            self.hash_str += i.hash()
        return self.hash_str
    
    def __str__(self):
        print(type(self))
        for child in self.children:
            print(child)
        return ''  

    def __hash__(self):
        return 0   
    
    def __eq__(self, obj):
        if type(self)==type(obj):
            if len(self.children) == len(obj.children):
                for i in range(len(self.children)):
                    if self.children[i] != obj.children[i]:
                        return False 
                if checkEqualMetadata(self.irMetadata, obj.irMetadata):
                    return True 
                else:
                    return False
                # return True
        return False
    
        
class IrExpression(IrAst):
    def __init__(self):
        super().__init__()
        

class IrStatement(IrAst):
    def __init__(self):
        super().__init__()



class IrConst(IrExpression):
    def __init__(self, const, type):
        super().__init__()
        self.const = const 
        self.irMetadata = [IrMetadataElement([1], type, [1], True)]

    def __str__(self):
        print(type(self), self.const)
        return ''
    
    def __eq__(self, obj):
        if isinstance(obj, IrConst):
            if self.const == obj.const:
                return True 
        return False
    
    def hash(self):
        self.hash_str = str(type(self))
        self.hash_str += str(self.const) 
        return self.hash_str

class IrVar(IrExpression):
    
    def __init__(self, name, irMetadata):
        super().__init__()
        self.name = name 
        self.irMetadata = irMetadata
        self.defs = None
        self.formal_argument = True
        self.uses = []

        if IrAst.poly_size == None:
            IrAst.poly_size = 1
            IrAst.poly_size = IrVar('poly_size', [IrMetadataElement([1], 'Int', [1], True)])
            IrAst.curr_size = IrVar('curr_size', [IrMetadataElement([1], 'Int', [1], True)])
            IrAst.prev_size = IrVar('prev_size', [IrMetadataElement([1], 'Int', [1], True)])
            IrAst.trav_size = IrVar('trav_size', [IrMetadataElement([1], 'Int', [1], True)])
        
    def __str__(self):
        print(type(self), self.name)
        return ''
    
    def __eq__(self, obj):
        if isinstance(obj, IrVar):
            if self.name == obj.name:
                return True 
        return False
    
    def __hash__(self):
        return 0
    
    def hash(self):
        self.hash_str = str(type(self))
        self.hash_str += self.name 
        return self.hash_str
    
class IrPhi(IrExpression):
    def __init__(self, original_name, vars, irMetadata):
        super().__init__()
        self.original_name = original_name
        self.irMetadata = irMetadata
        self.update_parent_child(vars)
        self.parent_nodes = []


    def __eq__(self, obj):
        if type(obj) == type(self):
            return self.identifier == obj.identifier
        else:
            return False
    
    def __hash__(self):
        return 0

class IrRepeat(IrExpression):
    def __init__(self, inputIr, irMetadata):
        super().__init__()
        assert(canRepeat(inputIr.irMetadata, irMetadata))
        
        self.irMetadata = irMetadata
        repeat_dims = []
        for i in range(len(inputIr.irMetadata)):
            for j in range(len(inputIr.irMetadata[i].broadcast)):
                # print(inputIr.irMetadata[i].broadcast[j], convert_z3_to_ir(inputIr.irMetadata[i].broadcast[j]))
                # repeat_dims.append(convert_z3_to_ir(inputIr.irMetadata[i].broadcast[j]))
                repeat_dims.append(inputIr.irMetadata[i].broadcast[j])
        children = [inputIr] + repeat_dims
        # print(repeat_dims)
        # jf
        self.update_parent_child(children)

class IrConvertBoolToFloat(IrExpression):
    def __init__(self, inputIr):
        super().__init__()
        # self.irMetadata = copy.deepcopy(inputIr.irMetadata)
        self.irMetadata = copy_metadata(inputIr.irMetadata)
        self.irMetadata[-1].type = 'Float'
        self.update_parent_child([inputIr])

class IrConvertToTensor(IrExpression):
    def __init__(self, inputIr, irMetadata):
        super().__init__()
        self.irMetadata = irMetadata

        repeat_dims = []
        for i in range(len(inputIr.irMetadata)):
            for j in range(len(inputIr.irMetadata[i].shape)):
                # print(inputIr.irMetadata[i].shape[j], convert_z3_to_ir(inputIr.irMetadata[i].shape[j]))
                # repeat_dims.append(convert_z3_to_ir(inputIr.irMetadata[i].shape[j]))
                repeat_dims.append(inputIr.irMetadata[i].shape[j])

        new_children = [inputIr] + repeat_dims
        self.update_parent_child(new_children)


class IrGetDefaultStop(IrExpression):
    def __init__(self, inputIr):
        super().__init__()
        # self.irMetadata = copy.deepcopy(inputIr.irMetadata)
        self.irMetadata = copy_metadata(inputIr.irMetadata)
        self.irMetadata[-1].shape.append(IrAst.poly_size)
        # self.irMetadata[-1].shape.append(POLYEXP_SIZE)
        self.irMetadata[-1].broadcast.append(1)
        self.irMetadata[-1].type = 'Bool'
        self.irMetadata[-1].isConst = False

        repeat_dims = []
        for i in range(len(self.irMetadata)):
            for j in range(len(self.irMetadata[i].shape)):
                # print(self.irMetadata[i].shape[j], convert_z3_to_ir(self.irMetadata[i].shape[j]))
                # repeat_dims.append(convert_z3_to_ir(self.irMetadata[i].shape[j]))
                repeat_dims.append(self.irMetadata[i].shape[j])
        # print(repeat_dims)
        # dfg

        new_children = [inputIr] + repeat_dims
        self.update_parent_child(new_children)


class IrAddDimension(IrExpression):
    def __init__(self, inputIr, irMetadataElement):
        super().__init__()
        assert(canAddDimension(inputIr.irMetadata[-1], irMetadataElement))

        self.irMetadata = inputIr.irMetadata[:-1].copy()
        self.irMetadata.append(irMetadataElement) 
        self.update_parent_child([inputIr])
        


class IrAddDimensionConst(IrExpression):
    def __init__(self, inputIr, irMetadata):
        if isinstance(inputIr, IrAddDimensionConst):
            inputIr = inputIr.children[0]
        super().__init__()
        assert(inputIr.irMetadata[-1].isConst)
        for i in range(len(irMetadata)):
            assert(irMetadata[i].type == inputIr.irMetadata[-1].type)

        self.irMetadata = irMetadata

        repeat_dims = []
        for i in range(len(self.irMetadata)):
            for j in range(len(self.irMetadata[i].broadcast)):
                # repeat_dims.append(convert_z3_to_ir(self.irMetadata[i].broadcast[j] * self.irMetadata[i].shape[j]))
                repeat_dims.append(mult_metadata(self.irMetadata[i].broadcast[j], self.irMetadata[i].shape[j]))
        new_children = [inputIr] + repeat_dims

        self.update_parent_child(new_children)
        self.irMetadata[-1].isConst = False

class IrConvertNeuronToPoly(IrExpression):
    def __init__(self, inputIr):
        super().__init__()
        assert(inputIr.irMetadata[-1].type == 'Neuron')

        # self.inputIr = inputIr 
        # self.irMetadata = copy.deepcopy(inputIr.irMetadata)
        self.irMetadata = copy_metadata(inputIr.irMetadata)
        self.irMetadata[-1].type = 'PolyExp'
        self.update_parent_child([inputIr])
    
    

class IrConvertConstToPoly(IrExpression):
    def __init__(self, inputIr):
        super().__init__()
        assert((inputIr.irMetadata[-1].type == 'Float') or (inputIr.irMetadata[-1].type == 'Int'))
        
        # self.inputIr = inputIr 
        # self.irMetadata = copy.deepcopy(inputIr.irMetadata)
        self.irMetadata = copy_metadata(inputIr.irMetadata)
        self.irMetadata[-1].type = 'PolyExp'
        new_children = [inputIr, inputIr.irMetadata[-1].shape[0]]
        # new_children = [inputIr, convert_z3_to_ir(inputIr.irMetadata[-1].shape[0])]
        self.update_parent_child(new_children)
        # self.update_parent_child([inputIr])


class IrExtractPolyCoeff(IrExpression):
    def __init__(self, inputIr):
        super().__init__()
        if(inputIr.irMetadata[-1].type != 'PolyExp'):
            print(inputIr.irMetadata[-1].type)
        assert(inputIr.irMetadata[-1].type == 'PolyExp')

        # self.inputIr = inputIr 
        # self.irMetadata = copy.deepcopy(inputIr.irMetadata)
        self.irMetadata = copy_metadata(inputIr.irMetadata)
        self.irMetadata[-1].type = 'Float'
        self.irMetadata[-1].shape.append(IrAst.poly_size)
        self.irMetadata[-1].broadcast.append(1)
        self.update_parent_child([inputIr])


class IrExtractPolyConst(IrExpression):
    def __init__(self, inputIr):
        super().__init__()
        assert(inputIr.irMetadata[-1].type == 'PolyExp')

        self.inputIr = inputIr 
        # self.irMetadata = copy.deepcopy(inputIr.irMetadata)
        self.irMetadata = copy_metadata(inputIr.irMetadata)
        self.irMetadata[-1].type = 'Float'
        self.update_parent_child([inputIr])


class IrAccess(IrExpression):
    def __init__(self, lhsIr, elem, type=None, isMetadata=False):
        super().__init__()
        self.isMetadata = isMetadata
        self.elem = elem

        lhsIrMetadata = lhsIr.irMetadata

        # self.irMetadata = copy.deepcopy(lhsIrMetadata)
        self.irMetadata = copy_metadata(lhsIrMetadata)
        self.update_parent_child([lhsIr])

        if isMetadata:
            assert(lhsIrMetadata[-1].type == 'Neuron')
            if elem == 'weight':
                self.irMetadata[-1].shape.append(IrAst.prev_size)
                self.irMetadata[-1].broadcast.append(1)
                self.irMetadata[-1].type = 'Float'

            elif elem == 'bias':
                self.irMetadata[-1].type = 'Float'

            elif elem == 'layer':
                self.irMetadata[-1].type = 'Int'
        
        else:
            self.irMetadata[-1].type = type

    def __eq__(self, obj):
        if isinstance(obj, IrAccess):
            if self.elem == obj.elem and self.isMetadata == obj.isMetadata:
                return self.children[0] == obj.children[0]
        return False
    
    def __hash__(self):
        return 0


class IrBinaryOp(IrExpression):
    def __init__(self, lhsIr, rhsIr, op):
        # [lhsIr, rhsIr] = inputIrs
        super().__init__()
        # self.lhsIr = lhsIr
        # self.rhsIr = rhsIr
        self.op = op
        
        lhsIr, rhsIr = matchDims(lhsIr, rhsIr)
        lhsIrMetadata = lhsIr.irMetadata
        rhsIrMetadata = rhsIr.irMetadata

        # self.irMetadata = copy.deepcopy(lhsIr.irMetadata)
        self.irMetadata = copy_metadata(lhsIr.irMetadata)
        new_type = 'Float' if lhsIrMetadata[-1].type!=rhsIrMetadata[-1].type else lhsIrMetadata[-1].type
        if self.op in ['>', '>=', '==', '<', '<=']:
            new_type = 'Bool'
        self.irMetadata[-1].type = new_type
        self.irMetadata[-1].isConst = lhsIrMetadata[-1].isConst and rhsIrMetadata[-1].isConst    
        self.update_parent_child([lhsIr, rhsIr])

class IrUnaryOp(IrExpression):
    def __init__(self, inputIr, op):
        super().__init__()
        self.op = op
        
        # self.irMetadata = copy.deepcopy(inputIr.irMetadata)
        self.irMetadata = copy_metadata(inputIr.irMetadata)
        self.update_parent_child([inputIr])



class IrMult(IrExpression):
    def __init__(self, lhsIr, rhsIr, op):
        super().__init__()
        self.op = op 

        lhsIr, rhsIr = matchDims(lhsIr, rhsIr)
        lhsIrMetadata = lhsIr.irMetadata
        rhsIrMetadata = rhsIr.irMetadata

        # self.irMetadata = copy.deepcopy(lhsIr.irMetadata)
        self.irMetadata = copy_metadata(lhsIr.irMetadata)
        new_type = 'Float' if lhsIrMetadata[-1].type!=rhsIrMetadata[-1].type else lhsIrMetadata[-1].type
        self.irMetadata[-1].type = new_type
        self.update_parent_child([lhsIr, rhsIr])


class IrTernary(IrExpression):
    def __init__(self, condIr, lhsIr, rhsIr):
        super().__init__()

        condIr, lhsIr = matchDims(condIr, lhsIr)
        condIr, rhsIr = matchDims(condIr, rhsIr)
        lhsIr, rhsIr = matchDims(lhsIr, rhsIr)

        self.irMetadata = lhsIr.irMetadata.copy()
        if not rhsIr.irMetadata[-1].isConst:
            self.irMetadata[-1].isConst = False
        if not condIr.irMetadata[-1].isConst:
            self.irMetadata[-1].isConst = False
        self.update_parent_child([condIr, lhsIr, rhsIr])


class IrDot(IrExpression):
    def __init__(self, lhsIr, rhsIr):
        super().__init__()
        assert(lhsIr.irMetadata[-1].type == 'Neuron')
        assert(rhsIr.irMetadata[-1].type == 'Float')
        assert(len(lhsIr.irMetadata[-1].shape) == 2)
        # assert(check_eq(lhsIr.irMetadata[-1].shape[0]*lhsIr.irMetadata[-1].broadcast[0], rhsIr.irMetadata[-1].shape[0]*rhsIr.irMetadata[-1].broadcast[0]))
        # assert(check_eq(lhsIr.irMetadata[-1].shape[1]*lhsIr.irMetadata[-1].broadcast[1], rhsIr.irMetadata[-1].shape[1]*rhsIr.irMetadata[-1].broadcast[1]))
        assert(check_eq(mult_metadata(lhsIr.irMetadata[-1].shape[0], lhsIr.irMetadata[-1].broadcast[0]), mult_metadata(rhsIr.irMetadata[-1].shape[0], rhsIr.irMetadata[-1].broadcast[0])))
        assert(check_eq(mult_metadata(lhsIr.irMetadata[-1].shape[1], lhsIr.irMetadata[-1].broadcast[1]), mult_metadata(rhsIr.irMetadata[-1].shape[1], rhsIr.irMetadata[-1].broadcast[1])))

        # self.irMetadata = copy.deepcopy(lhsIr.irMetadata)
        self.irMetadata = copy_metadata(lhsIr.irMetadata)
        self.irMetadata[-1].shape = [mult_metadata(lhsIr.irMetadata[-1].shape[0], lhsIr.irMetadata[-1].broadcast[0])]
        self.irMetadata[-1].broadcast = [1]
        self.irMetadata[-1].type = 'PolyExp'
        self.update_parent_child([lhsIr, rhsIr])


class IrCombineToPoly(IrExpression):
    def __init__(self, coeffIr, constIr):
        super().__init__()
        if not check_eq(coeffIr.irMetadata[-1].shape[:-1], constIr.irMetadata[-1].shape):
            print(coeffIr.irMetadata[-1].shape[:-1], constIr.irMetadata[-1].shape)
        assert(check_eq(coeffIr.irMetadata[-1].shape[:-1], constIr.irMetadata[-1].shape))
        assert(check_eq(coeffIr.irMetadata[-1].broadcast[:-1], constIr.irMetadata[-1].broadcast))
        assert(coeffIr.irMetadata[-1].isConst == constIr.irMetadata[-1].isConst)

        # self.irMetadata = copy.deepcopy(constIr.irMetadata)
        self.irMetadata = copy_metadata(constIr.irMetadata)
        self.irMetadata[-1].type = 'PolyExp'
        new_children = [coeffIr, constIr, constIr.irMetadata[-1].shape[0]]
        # new_children = [coeffIr, constIr, convert_z3_to_ir(constIr.irMetadata[-1].shape[0])]
        self.update_parent_child(new_children)


class IrReduce(IrExpression):
    def __init__(self, inputIr):
        super().__init__()
        irMetadata = inputIr.irMetadata 
        assert(len(irMetadata) > 1)
        assert(len(irMetadata[-1].shape) <= 2)

        # self.irMetadata = copy.deepcopy(irMetadata[:-1])
        self.irMetadata = copy_metadata(irMetadata[:-1])
        if len(irMetadata[-1].shape)==1:
            # irMetadataElement = copy.deepcopy(inputIr.irMetadata[-1])
            irMetadataElement = inputIr.irMetadata[-1].copy()
            irMetadataElement.shape.append(1)
            irMetadataElement.broadcast.append(1)
            self.irMetadata[-1].type = 'Float'
        else:
            self.irMetadata[-1].type = 'PolyExp'
            self.irMetadata[-1].shape += irMetadata[-1].shape[1:]
            self.irMetadata[-1].broadcast += irMetadata[-1].broadcast[1:]
            print('here')
            print(self.irMetadata[-1].shape)
            print(irMetadata[-1].shape)
        self.update_parent_child([inputIr])

class IrMapCoeff(IrExpression):
    def __init__(self, inputIr):
        super().__init__()
        assert(inputIr.irMetadata[-1].type == 'PolyExp')
        
        # self.irMetadata = copy.deepcopy(inputIr.irMetadata)
        self.irMetadata = copy_metadata(inputIr.irMetadata)
        irMetadataElement = IrMetadataElement([IrAst.poly_size], 'Float', [1], False)
        self.irMetadata.append(irMetadataElement)
        self.update_parent_child([inputIr])

class IrMapNeuron(IrExpression):
    def __init__(self, inputIr):
        super().__init__()
        assert(inputIr.irMetadata[-1].type == 'PolyExp')
        
        # self.irMetadata = copy.deepcopy(inputIr.irMetadata)
        self.irMetadata = copy_metadata(inputIr.irMetadata)
        
        for i in range(len(self.irMetadata)):
            for j in range(len(self.irMetadata[i].shape)):
                self.irMetadata[i].broadcast[j] = mult_metadata(self.irMetadata[i].broadcast[j], self.irMetadata[i].shape[j])
                # self.irMetadata[i].broadcast[j] *= self.irMetadata[i].shape[j]
                self.irMetadata[i].shape[j] = 1

        irMetadataElement = IrMetadataElement([IrAst.poly_size], 'Neuron', [1], False)
        self.irMetadata.append(irMetadataElement)
        self.update_parent_child([inputIr])

class IrSymbolic(IrExpression):
    def __init__(self, name, irMetadata):
        super().__init__()
        self.name = name
        self.irMetadata = irMetadata
        self.uses = []

    def __eq__(self, obj):
        if type(obj)==IrSymbolic:
            if self.name == obj.name:
                return True 
        return False 
    
    def __hash__(self):
        return 0

class IrAssignment(IrStatement):
    def __init__(self, varIr, inputIr):
        super().__init__()
        # if not checkEqualMetadata(varIr.irMetadata, inputIr.irMetadata):
        #     print(varIr.irMetadata[0].shape, inputIr.irMetadata[0].shape)
        # assert(checkEqualMetadata(varIr.irMetadata, inputIr.irMetadata))
        self.irMetadata = inputIr.irMetadata
        self.update_parent_child([varIr, inputIr])
        self.uses = []

class IrTransRetBasic(IrStatement):
    def __init__(self, exprIrs):
        super().__init__()
        self.exprIrs = exprIrs
        self.update_parent_child(exprIrs)



class IrBreak(IrStatement):
    def __init__(self):
        super().__init__()

# class IrCustomCodeGen(IrStatement):
#     def __init__(self, documentation, var):
#         super().__init__()
#         self.documentation = documentation
#         self.update_parent_child([var])

#     def __eq__(self, obj):
#         if type(obj)==IrCustomCodeGen:
#             if self.documentation == obj.documentation and self.children == obj.children:
#                 return True 
#         return False 
    
#     def __hash__(self):
#         return 0
    
class IrWhile(IrStatement):
    def __init__(self, condIr, inputIrs):
        super().__init__()
        self.update_parent_child([condIr] + inputIrs)

class IrIte(IrAst):
    def __init__(self, condIr, lhsIrs, rhsIrs):
        super().__init__()
        self.condIr = condIr
        self.lhsIrs = lhsIrs
        self.rhsIrs = rhsIrs
        self.children.append(condIr)
        self.children.append(lhsIrs)
        self.children.append(rhsIrs)

class IrBreak(IrStatement):
    def __init__(self):
        super().__init__()

class IrBlock(IrAst):
    def __init__(self, ir_list = [], jump = None, inner_jump = None, loopBack = None):
        super().__init__()
        self.inner_jump = inner_jump
        self.jump = jump
        self.update_parent_child(ir_list)

    def __eq__(self, obj):
        if isinstance(obj, IrBlock):
            return self.identifier == obj.identifier 
        return False 
    
    def __hash__(self):
        return 0

class IrWhileBlock(IrBlock):
    def __init__(self, condIr, ir_list = [], loopBody = None, jump = None, inner_jump = None, loopBack = None):
        super().__init__(ir_list, jump, inner_jump, loopBack)
        self.condIr = condIr
        self.loopBody = loopBody
        if self.loopBody == None:
            self.loopBody = [self]


class IrOpStmt(IrAst):
    def __init__(self, op, cfg):
        super().__init__()
        self.op = op
        self.cfg = cfg
        # self.update_parent_child(inputIrs)

class IrFlow(IrAst):
    def __init__(self, stop, priority, transformer, direction):
        super().__init__()
        self.stop = stop
        self.priority = priority
        self.transformer = transformer
        self.direction = direction

class IrProgram(IrAst):
    def __init__(self, shape, tstore, fstore, irNodes):
        super().__init__()
        self.shape = shape
        self.tstore = tstore
        self.fstore = fstore
        self.irNodes = irNodes
        


# def convert_z3_to_ir(z3_expr):
#     if IrAst.poly_size == None:
#         IrAst.poly_size = IrVar('poly_size', [IrMetadataElement([1], 'Int', [1], True)])
#         IrAst.curr_size = IrVar('curr_size', [IrMetadataElement([1], 'Int', [1], True)])
#         IrAst.prev_size = IrVar('prev_size', [IrMetadataElement([1], 'Int', [1], True)])
#         IrAst.trav_size = IrVar('trav_size', [IrMetadataElement([1], 'Int', [1], True)])
#     if isinstance(z3_expr, int) or isinstance(z3_expr, z3.z3.IntNumRef):
#         ret = IrConst(int(str(z3_expr)), 'Int')
#         return ret
#     elif str(z3_expr.decl()) == '+':
#         lhs = convert_z3_to_ir(z3_expr.children()[0])
#         rhs = convert_z3_to_ir(z3_expr.children()[1])
#         return IrBinaryOp(lhs, rhs, '+')
#     elif str(z3_expr.decl()) == '*':
#         lhs = convert_z3_to_ir(z3_expr.children()[0])
#         rhs = convert_z3_to_ir(z3_expr.children()[1])
#         ret = IrMult(lhs, rhs, '*')
#         print(ret.irMetadata[0].shape, ret.irMetadata[0].broadcast)
#         # kdfjh
#         return ret 
#     elif isinstance(z3_expr, z3.z3.ArithRef):
#         if str(z3_expr) == 'poly_size':
#             return IrAst.poly_size
#         elif  str(z3_expr) == 'curr_size':
#             return IrAst.curr_size
#         elif str(z3_expr) == 'prev_size':
#             return IrAst.prev_size
#         elif str(z3_expr) == 'trav_size':
#             return IrAst.trav_size
#     else:
#         print(z3_expr, type(z3_expr))
#         raise Exception('CHECK THIS')