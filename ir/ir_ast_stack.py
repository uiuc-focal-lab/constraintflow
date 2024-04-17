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
    if isinstance(x==y, bool):
        return x==y
    return bool(simplify(x==y))

class IrMetadataElement:
    def __init__(self, shape, type, broadcast, isConst = False):
        self.shape = shape  
        self.broadcast = broadcast
        self.isConst = isConst
        self.type = type 
        assert(len(shape) == len(broadcast))
    
    def copy(self):
        return copy.deepcopy(self)
    
    def __eq__(self,obj):
        return self.shape == obj.shape and self.type == obj.type and self.isConst == obj.isConst and self.isExpanded == obj.isExpanded

    def __str__(self):
        print(self.shape)
        print(self.broadcast)
        return ' '

def checkLargerShape(shape1, shape2):
    for i in range(len(shape1)):
        if shape1[i]>shape2[i]:
            return -1
        elif shape1[i]<shape2[i]:
            return 1
    return 0 

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

def checkSmallerMetadata(irMetadata1, irMetadata2):
    small_flag = False
    if len(irMetadata1) != len(irMetadata2):
        return False
    for i in range(len(irMetadata1)):
        if len(irMetadata1[i].shape) != len(irMetadata2[i].shape):
            return False 
        for j in range(len(irMetadata1[i].shape)):
            if check_eq(irMetadata1[i].shape[j], 1):
                if not check_eq(irMetadata2[i].shape[j], 1):
                    small_flag = True
            else:
                if not check_eq(irMetadata1[i].shape[j], irMetadata2[i].shape[j]):
                    return False
    return small_flag

def createLcm(irMetadata1, irMetadata2):
    flag1 = False 
    flag2 = False
    newIrMetadata1 = copy.deepcopy(irMetadata1)
    newIrMetadata2 = copy.deepcopy(irMetadata2)
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
                if not check_eq(irMetadata1[i].shape[j] * irMetadata1[i].broadcast[j], irMetadata2[i].shape[j] * irMetadata2[i].broadcast[j]):
                    return False
    return True

def canAddDimension(irMetadataElement1, irMetadataElement2):
    if len(irMetadataElement1.shape) >= len(irMetadataElement2.shape):
        return False 
    for i in range(len(irMetadataElement1.shape)):
        if not check_eq(irMetadataElement1.shape[i], irMetadataElement1.shape[i]):
            return False
        if not check_eq(irMetadataElement1.broadcast[i], irMetadataElement1.broadcast[i]):
            return False
    for i in range(len(irMetadataElement1.shape), len(irMetadataElement2.shape)):
        if irMetadataElement2.shape[i]!=1:
            return False
    return True

def matchDims(lhsIr, rhsIr):
    lhsIrMetadata = lhsIr.irMetadata
    rhsIrMetadata = rhsIr.irMetadata

    if lhsIrMetadata[-1].isConst:
        updated_irMetadata = copy.deepcopy(rhsIrMetadata)
        for i in range(len(updated_irMetadata)):
            updated_irMetadata[i].isConst = True 
            updated_irMetadata[i].type = lhsIrMetadata[-1].type
        lhsIr = IrAddDimensionConst(lhsIr, updated_irMetadata)
        return lhsIr, rhsIr

    if rhsIrMetadata[-1].isConst:
        updated_irMetadata = copy.deepcopy(lhsIrMetadata)
        for i in range(len(updated_irMetadata)):
            updated_irMetadata[i].isConst = True 
            updated_irMetadata[i].type = rhsIrMetadata[-1].type
        rhsIr = IrAddDimensionConst(rhsIr, updated_irMetadata) 
        return lhsIr, rhsIr

    assert(len(lhsIrMetadata) == len(rhsIrMetadata))

    if len(lhsIrMetadata[-1].shape) < len(rhsIrMetadata[-1].shape):
        updated_irMetadataElement = rhsIrMetadata[-1].copy()
        for i in range(len(lhsIrMetadata[-1].shape), len(rhsIrMetadata[-1].shape)):
            updated_irMetadataElement.broadcast[i] *= updated_irMetadataElement.shape[i]
            updated_irMetadataElement.shape[i] = 1
        lhsIr = IrAddDimension(lhsIr, updated_irMetadataElement)

    elif len(rhsIrMetadata[-1].shape) < len(lhsIrMetadata[-1].shape):
        updated_irMetadataElement = lhsIrMetadata[-1].copy()
        for i in range(len(rhsIrMetadata[-1].shape), len(lhsIrMetadata[-1].shape)):
            updated_irMetadataElement.broadcast[i] *= updated_irMetadataElement.shape[i]
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
    def __init__(self):
        self.node_name = "IR_ASTNode"
        self.parents = []
        self.children = []
        self.identifier = IrAst.counter
        IrAst.counter += 1
        # self.hash_str = str(type(self))

    def __str__(self):
        print(type(self))
        return ''
    
    def hash(self):
        self.hash_str = str(type(self))
        for i in self.children:
            self.hash_str += i.hash()
        return self.hash_str
        
        


class IrConst(IrAst):
    def __init__(self, const, type):
        super().__init__()
        self.const = const 
        self.irMetadata = [IrMetadataElement([1], type, [1], True)]

    def __str__(self):
        print(type(self), self.const)
        return ''
    
    def hash(self):
        self.hash_str = str(type(self))
        self.hash_str += str(self.const) 
        return self.hash_str

class IrVar(IrAst):
    def __init__(self, name, irMetadata):
        super().__init__()
        self.name = name 
        self.irMetadata = irMetadata
        
    def __str__(self):
        print(type(self), self.name)
        return ''
    
    def hash(self):
        self.hash_str = str(type(self))
        # print(self.hash_str)
        self.hash_str += self.name 
        # print(self.hash_str)
        return self.hash_str

class IrRepeat(IrAst):
    def __init__(self, inputIr, irMetadata):
        super().__init__()
        assert(canRepeat(inputIr.irMetadata, irMetadata))
        
        self.irMetadata = irMetadata
        self.inputIr = inputIr
        self.children.append(inputIr)

    def __str__(self):
        print(type(self))
        print(self.inputIr)
        return ''



class IrAddDimension(IrAst):
    def __init__(self, inputIr, irMetadataElement):
        super().__init__()
        assert(canAddDimension(inputIr.irMetadata[-1], irMetadataElement))

        self.irMetadata = inputIr.irMetadata[:-1].copy()
        self.irMetadata.append(irMetadataElement) 
        self.inputIr = inputIr
        self.children.append(inputIr)

    def __str__(self):
        print(type(self))
        print(self.inputIr)
        return ''
    
    def update_child(self, original_child, new_child):
        if self.inputIr == original_child:
            self.children.remove(self.inputIr)
            self.inputIr = new_child 
            self.children.append(new_child)
            new_child.parents.append(self)

class IrAddDimensionConst(IrAst):
    def __init__(self, inputIr, irMetadata):
        if isinstance(inputIr, IrAddDimensionConst):
            inputIr = inputIr.inputIr
        super().__init__()
        assert(inputIr.irMetadata[-1].isConst)
        for i in range(len(irMetadata)):
            assert(irMetadata[i].type == inputIr.irMetadata[-1].type)

        self.irMetadata = irMetadata
        self.inputIr = inputIr
        self.children.append(inputIr)

    def __str__(self):
        print(type(self))
        print(self.inputIr)
        return ''
    
    def update_child(self, original_child, new_child):
        if self.inputIr == original_child:
            self.children.remove(self.inputIr)
            self.inputIr = new_child 
            self.children.append(new_child)
            new_child.parents.append(self)

class IrConvertNeuronToPoly(IrAst):
    def __init__(self, inputIr):
        super().__init__()
        assert(inputIr.irMetadata[-1].type == 'Neuron')

        self.inputIr = inputIr 
        self.irMetadata = copy.deepcopy(inputIr.irMetadata)
        self.irMetadata[-1].type = 'PolyExp'
        self.children.append(inputIr)

    def __str__(self):
        print(type(self))
        print(self.inputIr)
        return ''
    
    def update_child(self, original_child, new_child):
        if self.inputIr == original_child:
            self.children.remove(self.inputIr)
            self.inputIr = new_child 
            self.children.append(new_child)
            new_child.parents.append(self)
    
    

class IrConvertConstToPoly(IrAst):
    def __init__(self, inputIr):
        super().__init__()
        assert((inputIr.irMetadata[-1].type == 'Float') or (inputIr.irMetadata[-1].type == 'Int'))
        
        self.inputIr = inputIr 
        self.irMetadata = copy.deepcopy(inputIr.irMetadata)
        self.irMetadata[-1].type = 'PolyExp'
        self.children.append(inputIr)

    def __str__(self):
        print(type(self))
        print(self.inputIr)
        return ''
    
    def update_child(self, original_child, new_child):
        if self.inputIr == original_child:
            self.children.remove(self.inputIr)
            self.inputIr = new_child 
            self.children.append(new_child)
            new_child.parents.append(self)

    

class IrExtractPolyCoeff(IrAst):
    def __init__(self, inputIr):
        super().__init__()
        if(inputIr.irMetadata[-1].type != 'PolyExp'):
            print(inputIr.irMetadata[-1].type)
        assert(inputIr.irMetadata[-1].type == 'PolyExp')

        self.inputIr = inputIr 
        self.irMetadata = copy.deepcopy(inputIr.irMetadata)
        self.irMetadata[-1].type = 'Float'
        self.irMetadata[-1].shape.append(POLYEXP_SIZE)
        self.irMetadata[-1].broadcast.append(1)
        self.children.append(inputIr)

    def __str__(self):
        print(type(self))
        print(self.inputIr)
        return ''
    
    def update_child(self, original_child, new_child):
        if self.inputIr == original_child:
            self.children.remove(self.inputIr)
            self.inputIr = new_child 
        
        # else:
        #     print(self.inputIr)
        #     print(original_child)
        #     print(new_child)
        #     raise Exception('ERROR')

            self.children.append(new_child)
            new_child.parents.append(self)

class IrExtractPolyConst(IrAst):
    def __init__(self, inputIr):
        super().__init__()
        assert(inputIr.irMetadata[-1].type == 'PolyExp')

        self.inputIr = inputIr 
        self.irMetadata = copy.deepcopy(inputIr.irMetadata)
        self.irMetadata[-1].type = 'Float'
        self.children.append(inputIr)
        # self.irMetadata[-1].shape.append(1)
        # self.irMetadata[-1].broadcast.append(1)
        # print(self.irMetadata[-1])
        # print('@@@@@@@@@@@@@@@@@@@')

    def __str__(self):
        print(type(self))
        print(self.inputIr)
        return ''
    
    def update_child(self, original_child, new_child):
        if self.inputIr == original_child:
            self.children.remove(self.inputIr)
            self.inputIr = new_child 

        # else:
        #     raise Exception('ERROR')

            self.children.append(new_child)
            new_child.parents.append(self)


class IrAccess(IrAst):
    def __init__(self, lhsIr, elem, type=None, isMetadata=False):
        super().__init__()
        self.lhsIr = lhsIr
        self.isMetadata = isMetadata
        self.elem = elem

        lhsIrMetadata = lhsIr.irMetadata

        self.irMetadata = copy.deepcopy(lhsIrMetadata)
        self.children.append(lhsIr)

        if isMetadata:
            assert(lhsIrMetadata[-1].type == 'Neuron')
            if elem == 'weight':
                self.irMetadata[-1].shape.append(PREV_SIZE)
                self.irMetadata[-1].broadcast.append(1)
                self.irMetadata[-1].type = 'Float'

            elif elem == 'bias':
                self.irMetadata[-1].type = 'Float'
                # self.irMetadata[-1].shape.append(1)
                # self.irMetadata[-1].broadcast.append(1)

            elif elem == 'layer':
                self.irMetadata[-1].type = 'Int'
        
        else:
            self.irMetadata[-1].type = type

    def __str__(self):
        print(type(self))
        print(self.lhsIr)
        return ''
    
    def hash(self):
        self.hash_str = str(type(self))
        self.hash_str += self.lhsIr.hash()
        self.hash_str += self.elem 
        return self.hash_str
    
    def update_child(self, original_child, new_child):
        if self.lhsIr == original_child:
            self.children.remove(self.lhsIr)
            self.lhsIr = new_child 

            self.children.append(new_child)
            new_child.parents.append(self)



class IrBinaryOp(IrAst):
    def __init__(self, lhsIr, rhsIr, op):
        super().__init__()
        self.lhsIr = lhsIr
        self.rhsIr = rhsIr
        self.op = op
        
        self.lhsIr, self.rhsIr = matchDims(self.lhsIr, self.rhsIr)
        lhsIrMetadata = self.lhsIr.irMetadata
        rhsIrMetadata = self.rhsIr.irMetadata

        self.irMetadata = copy.deepcopy(self.lhsIr.irMetadata)
        new_type = 'Float' if lhsIrMetadata[-1].type!=rhsIrMetadata[-1].type else lhsIrMetadata[-1].type
        if self.op in ['>', '>=', '==', '<', '<=']:
            new_type = 'Bool'
        self.irMetadata[-1].type = new_type
        self.children.append(self.lhsIr)
        self.children.append(self.rhsIr)

    def __str__(self):
        print(type(self))
        print(self.lhsIr)
        print('\nmid\n')
        print(self.rhsIr)
        return ''
    
    def update_child(self, original_child, new_child):
        if self.lhsIr == original_child:
            self.children.remove(self.lhsIr)
            self.lhsIr = new_child 
            self.children.append(new_child)
            new_child.parents.append(self)

        if self.rhsIr == original_child:
            self.children.remove(self.rhsIr)
            self.rhsIr = new_child 

            self.children.append(new_child)
            new_child.parents.append(self)



class IrMult(IrAst):
    def __init__(self, lhsIr, rhsIr, op):
        super().__init__()
        self.lhsIr = lhsIr
        self.rhsIr = rhsIr
        self.op = op 

        self.lhsIr, self.rhsIr = matchDims(self.lhsIr, self.rhsIr)
        lhsIrMetadata = self.lhsIr.irMetadata
        rhsIrMetadata = self.rhsIr.irMetadata

        self.irMetadata = copy.deepcopy(self.lhsIr.irMetadata)
        new_type = 'Float' if lhsIrMetadata[-1].type!=rhsIrMetadata[-1].type else lhsIrMetadata[-1].type
        self.irMetadata[-1].type = new_type
        self.children.append(self.lhsIr)
        self.children.append(self.rhsIr)

    def __str__(self):
        print(type(self))
        print(self.lhsIr)
        print('\nmid\n')
        print(self.rhsIr)
        return ''
    
    def update_child(self, original_child, new_child):
        if self.lhsIr == original_child:
            self.children.remove(self.lhsIr)
            self.lhsIr = new_child 
            self.children.append(new_child)
            new_child.parents.append(self)

        if self.rhsIr == original_child:
            self.children.remove(self.rhsIr)
            self.rhsIr = new_child 

            self.children.append(new_child)
            new_child.parents.append(self)

class IrTernary(IrAst):
    def __init__(self, condIr, lhsIr, rhsIr):
        super().__init__()
        self.condIr = condIr
        self.lhsIr = lhsIr
        self.rhsIr = rhsIr

        self.condIr, self.lhsIr = matchDims(self.condIr, self.lhsIr)
        self.condIr, self.rhsIr = matchDims(self.condIr, self.rhsIr)
        self.lhsIr, self.rhsIr = matchDims(self.lhsIr, self.rhsIr)

        self.irMetadata = self.lhsIr.irMetadata.copy()
        self.children.append(self.condIr)
        self.children.append(self.lhsIr)
        self.children.append(self.rhsIr)

    def __str__(self):
        print(type(self))
        print(self.lhsIr)
        print('\nmid\n')
        print(self.rhsIr)
        return ''
    
    def update_child(self, original_child, new_child):
        if self.lhsIr == original_child:
            self.children.remove(self.lhsIr)
            self.lhsIr = new_child 
            self.children.append(new_child)
            new_child.parents.append(self)

        if self.rhsIr == original_child:
            self.children.remove(self.rhsIr)
            self.rhsIr = new_child 

            self.children.append(new_child)
            new_child.parents.append(self)

        if self.condIr == original_child:
            self.children.remove(self.condIr)
            self.condIr = new_child 

            self.children.append(new_child)
            new_child.parents.append(self)

class IrDot(IrAst):
    def __init__(self, lhsIr, rhsIr):
        super().__init__()
        assert(lhsIr.irMetadata[-1].type == 'Neuron')
        assert(rhsIr.irMetadata[-1].type == 'Float')
        assert(len(lhsIr.irMetadata[-1].shape) == 2)
        assert(check_eq(lhsIr.irMetadata[-1].shape[0]*lhsIr.irMetadata[-1].broadcast[0], rhsIr.irMetadata[-1].shape[0]*rhsIr.irMetadata[-1].broadcast[0]))
        assert(check_eq(lhsIr.irMetadata[-1].shape[1]*lhsIr.irMetadata[-1].broadcast[1], rhsIr.irMetadata[-1].shape[1]*rhsIr.irMetadata[-1].broadcast[1]))

        self.lhsIr = lhsIr
        self.rhsIr = rhsIr
        self.irMetadata = copy.deepcopy(lhsIr.irMetadata)
        self.irMetadata[-1].shape = [lhsIr.irMetadata[-1].shape[0]*lhsIr.irMetadata[-1].broadcast[0]]
        self.irMetadata[-1].broadcast = [1]
        self.irMetadata[-1].type = 'PolyExp'
        self.children.append(self.lhsIr)
        self.children.append(self.rhsIr)

    def __str__(self):
        print(type(self))
        print(self.lhsIr)
        print('\nmid\n')
        print(self.rhsIr)
        return ''
    
    def update_child(self, original_child, new_child):
        if self.lhsIr == original_child:
            self.children.remove(self.lhsIr)
            self.lhsIr = new_child 
            self.children.append(new_child)
            new_child.parents.append(self)

        if self.rhsIr == original_child:
            self.children.remove(self.rhsIr)
            self.rhsIr = new_child 

            self.children.append(new_child)
            new_child.parents.append(self)

class IrCombineToPoly(IrAst):
    def __init__(self, coeffIr, constIr):
        super().__init__()
        if not check_eq(coeffIr.irMetadata[-1].shape[:-1], constIr.irMetadata[-1].shape):
            print(coeffIr.irMetadata[-1].shape[:-1], constIr.irMetadata[-1].shape)
        assert(check_eq(coeffIr.irMetadata[-1].shape[:-1], constIr.irMetadata[-1].shape))
        assert(check_eq(coeffIr.irMetadata[-1].broadcast[:-1], constIr.irMetadata[-1].broadcast))
        assert(coeffIr.irMetadata[-1].isConst == constIr.irMetadata[-1].isConst)

        self.coeffIr = coeffIr
        self.constIr = constIr
        self.irMetadata = copy.deepcopy(constIr.irMetadata)
        self.irMetadata[-1].type = 'PolyExp'
        self.children.append(self.coeffIr)
        self.children.append(self.constIr)

    def __str__(self):
        print(type(self))
        print(self.coeffIr)
        print('\nmid\n')
        print(self.constIr)
        return ''
    
    def update_child(self, original_child, new_child):
        if self.coeffIr == original_child:
            self.children.remove(self.coeffIr)
            self.coeffIr = new_child 
            self.children.append(new_child)
            new_child.parents.append(self)

        if self.constIr == original_child:
            self.children.remove(self.constIr)
            self.constIr = new_child 
            self.children.append(new_child)
            new_child.parents.append(self)


class IrReduce(IrAst):
    def __init__(self, inputIr):
        super().__init__()
        self.inputIr = inputIr
        irMetadata = inputIr.irMetadata 
        assert(len(irMetadata) > 1)
        assert(len(irMetadata[-1].shape) <= 2)

        self.irMetadata = copy.deepcopy(irMetadata)[:-1]
        if len(irMetadata[-1].shape)==1:
            irMetadataElement = copy.deepcopy(inputIr.irMetadata[-1])
            irMetadataElement.shape.append(1)
            irMetadataElement.broadcast.append(1)
            self.irMetadata[-1].type = 'Float'
        else:
            self.irMetadata[-1].type = 'PolyExp'
        self.children.append(self.inputIr)

    def __str__(self):
        print(type(self))
        print(self.inputIr)
        return ''

class IrMapCoeff(IrAst):
    def __init__(self, inputIr):
        super().__init__()
        assert(inputIr.irMetadata[-1].type == 'PolyExp')
        
        self.inputIr = inputIr
        self.irMetadata = copy.deepcopy(inputIr.irMetadata)
        irMetadataElement = IrMetadataElement([POLYEXP_SIZE], 'Float', [1], False)
        self.irMetadata.append(irMetadataElement)
        self.children.append(self.inputIr)

    def __str__(self):
        print(type(self))
        print(self.inputIr)
        return ''

class IrMapNeuron(IrAst):
    def __init__(self, inputIr):
        super().__init__()
        assert(inputIr.irMetadata[-1].type == 'PolyExp')
        
        self.inputIr = inputIr
        self.irMetadata = copy.deepcopy(inputIr.irMetadata)
        
        for i in range(len(self.irMetadata)):
            for j in range(len(self.irMetadata[i].shape)):
                self.irMetadata[i].broadcast[j] *= self.irMetadata[i].shape[j]
                self.irMetadata[i].shape[j] = 1

        irMetadataElement = IrMetadataElement([POLYEXP_SIZE], 'Neuron', [1], False)
        self.irMetadata.append(irMetadataElement)
        self.children.append(self.inputIr)

    def __str__(self):
        print(type(self))
        print(self.inputIr)
        return ''

class IrSymbolic(IrAst):
    def __init__(self, irMetadata):
        super().__init__()
        self.irMetadata = irMetadata


class IrTraverse(IrAst):
    def __init__(self, exprIr, stopSeqIr, priorityIr, prioritySeqIr, funcIr, funcSeqIr, direction):
        super().__init__()
        self.exprIr = exprIr 
        self.stopSeqIr = stopSeqIr 
        self.priorityIr = priorityIr
        self.prioritySeqIr = prioritySeqIr 
        self.funcIr = funcIr
        self.funcSeqIr = funcSeqIr 
        self.direction = direction 
        self.irMetadata = copy.deepcopy(exprIr.irMetadata)
        
        self.children.append(exprIr)
        self.children += stopSeqIr
        self.children += prioritySeqIr
        self.children += funcSeqIr

class IrAssignment(IrAst):
    def __init__(self, varIr, inputIr):
        super().__init__()
        self.varIr = varIr
        self.inputIr = inputIr
        self.irMetadata = inputIr.irMetadata
        self.children.append(self.varIr)
        self.children.append(self.inputIr)

    def __str__(self):
        print(type(self))
        print(self.varIr)
        print(self.inputIr)
        return ''

class IrRet(IrAst):
    def __init__(self, inputIr):
        super().__init__()
        self.inputIr = inputIr
        self.children.append(self.inputIr)

class IrSeq(IrAst):
    def __init__(self, inputIr1, inputIr2=None):
        super().__init__()
        self.inputIr1 = inputIr1
        self.inputIr2 = inputIr2

class IrTransRetBasic(IrAst):
    def __init__(self, exprIrs):
        super().__init__()
        self.exprIrs = exprIrs
        self.children += self.exprIrs

    def __str__(self):
        print(type(self))
        for t in self.exprIrs:
            print(t)
        return ''

class IrTransRetIf(IrAst):
    def __init__(self, condIr, lhsIrs, rhsIrs):
        super().__init__()
        self.condIr = condIr
        self.lhsIrs = lhsIrs
        self.rhsIrs = rhsIrs
        self.children.append(self.condIr)
        self.children.append(self.lhsIr)
        self.children.append(self.rhsIr)

class IrOpStmt(IrAst):
    def __init__(self, op, inputIr):
        super().__init__()
        self.op = op
        self.inputIr = inputIr

    def __str__(self):
        print(type(self))
        print(self.inputIr)
        return ''

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

    def __str__(self):
        print(type(self))
        for t in self.tstore.keys():
            for tt in self.tstore[t]:
                print(tt)
        return ''