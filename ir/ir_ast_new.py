import copy

POLYEXP_SIZE = 1004

class IrMetadata:
    def __init__(self, shape, type, isExpanded = True, isConst = False):
        self.shape = shape  
        self.isExpanded = isExpanded
        self.isConst = isConst
        self.type = type 
    
    def copy(self):
        return copy.deepcopy(self)
    
    def __eq__(self,obj):
        return self.shape == obj.shape and self.type == obj.type and self.isConst == obj.isConst and self.isExpanded == obj.isExpanded


def checkLargerShape(shape1, shape2):
    for i in range(len(shape1)):
        if shape1[i]>shape2[i]:
            return -1
        elif shape1[i]<shape2[i]:
            return 1
    return 0 
        

def matchDims(lhs_ir, rhs_ir):
    lhs_irMetadata = lhs_ir.irMetadata
    rhs_irMetadata = rhs_ir.irMetadata

    if (lhs_irMetadata.isExpanded != rhs_irMetadata.isExpanded):
        if not lhs_irMetadata.isExpanded:
            lhs_ir = IrRepeatAlongDimOuter(IrAddDimensionOuter(lhs_ir))
        else:
            rhs_ir = IrRepeatAlongDimOuter(IrAddDimensionOuter(rhs_ir))
    
    lhs_irMetadata = lhs_ir.irMetadata
    rhs_irMetadata = rhs_ir.irMetadata

    if len(lhs_irMetadata.shape) < len(rhs_irMetadata.shape):
        new_shape = lhs_irMetadata.shape 
        for i in range(len(lhs_irMetadata.shape), len(rhs_irMetadata.shape)):
            new_shape.append(1)
        lhs_ir = IrAddDimensionInner(lhs_ir, new_shape)
        

    elif len(rhs_irMetadata.shape) < len(lhs_irMetadata.shape):
        new_shape = rhs_irMetadata.shape 
        for i in range(len(rhs_irMetadata.shape), len(lhs_irMetadata.shape)):
            new_shape.append(1)
        rhs_ir = IrAddDimensionInner(rhs_ir, new_shape)

    lhs_irMetadata = lhs_ir.irMetadata
    rhs_irMetadata = rhs_ir.irMetadata

    c = checkLargerShape(lhs_irMetadata.shape, rhs_irMetadata.shape)
    
    if c==-1:
        rhs_ir = IrRepeatAlongDimInner(rhs_ir, lhs_irMetadata.shape)
    elif c==1:
        lhs_ir = IrRepeatAlongDimInner(lhs_ir, rhs_irMetadata.shape)

    return lhs_ir, rhs_ir 


class IrAst:
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

class IrConst(IrAst):
    def __init__(self, const, type):
        super().__init__()
        self.const = const 
        self.irMetadata = IrMetadata([1], type, False, True)

class IrVar(IrAst):
    def __init__(self, name, irMetadata):
        super().__init__()
        self.name = name 
        self.irMetadata = irMetadata
        
def canRepeat(shape1, shape2):
    if len(shape1) != len(shape2):
        return False 
    for i in range(len(shape1)):
        if shape1[i]!=shape2[i]:
            if shape1[i]!=1:
                return False 
    return True

class IrRepeatAlongDimInner(IrAst):
    def __init__(self, input_ir, shape):
        super().__init__()
        assert(canRepeat(input_ir.irMetadata.shape, shape))
        assert(input_ir.irMetadata.isConst == False)

        self.irMetadata = input_ir.irMetadata.copy()
        self.irMetadata.shape = shape 
        self.input_ir = input_ir

class IrRepeatAlongDimOuter(IrAst):
    def __init__(self, input_ir):
        super().__init__()
        assert(not input_ir.irMetadata.isExpanded)

        self.irMetadata = input_ir.irMetadata.copy()
        self.irMetadata.isExpanded = True
        self.input_ir = input_ir

def canAddInner(shape1, shape2):
    if len(shape1) <= len(shape2):
        return False 
    for i in range(len(shape1)):
        if shape1[i]!=shape2[i]:
            return False
    for i in range(len(shape1), len(shape2)):
        if shape2[i]!=1:
            return False
    return True

class IrAddDimensionInner(IrAst):
    def __init__(self, input_ir, shape):
        super().__init__()
        assert(canAddInner(input_ir.irMetadata.shape, shape))

        self.irMetadata = input_ir.irMetadata.copy()
        self.irMetadata.shape = shape 
        self.input_ir = input_ir

class IrAddDimensionOuter(IrAst):
    def __init__(self, input_ir):
        super().__init__()
        assert(not input_ir.irMetadata.isExpanded)

        self.irMetadata = input_ir.irMetadata.copy()
        self.irMetadata.isConst = False  
        self.input_ir = input_ir

class IrConvertNeuronToPoly(IrAst):
    def __init__(self, input_ir):
        super().__init__()
        self.input_ir = input_ir 
        input_shape = input_ir.irMetadata.shape
        new_shape = input_shape.copy() + [POLYEXP_SIZE]
        self.irMetadata = IrMetadata(new_shape, 'PolyExp', input_ir.isExpanded, False)

class IrConvertConstToPoly(IrAst):
    def __init__(self, input_ir):
        super().__init__()
        self.input_ir = input_ir 
        input_shape = input_ir.irMetadata.shape
        new_shape = input_shape.copy() + [POLYEXP_SIZE]
        self.irMetadata = IrMetadata(new_shape, 'PolyExp', input_ir.isExpanded, False)

class IrExtractPolyCoeff(IrAst):
    def __init__(self, input_ir):
        super().__init__()
        assert(input_ir.irMetadata.type == 'PolyExp')

        self.input_ir = input_ir 
        self.irMetadata = input_ir.irMetadata.copy()
        self.irMetadata.type = 'Float'

class IrExtractPolyConst(IrAst):
    def __init__(self, input_ir):
        super().__init__()
        assert(input_ir.irMetadata.type == 'PolyExp')

        self.input_ir = input_ir 
        self.irMetadata = input_ir.irMetadata.copy()
        self.irMetadata.type = 'Float'
        self.irMetadata.shape = self.irMetadata.shape[:-1]



class IrAccess(IrAst):
    def __init__(self, lhs_ir, rhs_ir, is_metadata=False):
        super().__init__()
        self.lhs_ir = lhs_ir
        self.rhs_ir = rhs_ir
        self.is_metadata = is_metadata

        lhs_irMetadata = lhs_ir.irMetadata
        rhs_irMetadata = rhs_ir.irMetadata

        if is_metadata:
            assert(isinstance(rhs_ir, IrVar))
            if rhs_ir.name == 'weight':
                

        if rhs_irMetadata.type == 'Int' or rhs_irMetadata.type == 'Float':
            self.irMetadata = lhs_irMetadata.copy()
            self.irMetadata.type = rhs_irMetadata.type
        else:
            new_shape = lhs_irMetadata.shape.copy() + [POLYEXP_SIZE]
            self.irMetadata = IrMetadata(new_shape, 'PolyExp', lhs_irMetadata.irMetadata.isExpanded, False)


class IrBinaryOpSimple(IrAst):
    def __init__(self, lhs_ir, rhs_ir, op):
        super().__init__()
        self.lhs_ir = lhs_ir
        self.rhs_ir = rhs_ir
        self.op = op
        
        lhs_irMetadata = self.lhs_ir.irMetadata
        rhs_irMetadata = self.rhs_ir.irMetadata

        self.lhs_ir, self.rhs_ir = matchDims(self.lhs_ir, self.rhs_ir)

        self.irMetadata = self.lhs_ir.irMetadata.copy()
        new_type = 'Float' if lhs_irMetadata.type!=rhs_irMetadata.type else lhs_irMetadata.type
        if self.op in ['>', '>=', '==', '<', '<=']:
            new_type = 'Bool'
        self.irMetadata.type = new_type



class IrMultSimple(IrAst):
    def __init__(self, lhs_ir, rhs_ir, op):
        super().__init__()
        self.lhs_ir = lhs_ir
        self.rhs_ir = rhs_ir
        self.op = op
        
        lhs_irMetadata = self.lhs_ir.irMetadata
        rhs_irMetadata = self.rhs_ir.irMetadata

        self.lhs_ir, self.rhs_ir = matchDims(self.lhs_ir, self.rhs_ir)

        self.irMetadata = self.lhs_ir.irMetadata.copy()
        new_type = 'Float' if lhs_irMetadata.type!=rhs_irMetadata.type else lhs_irMetadata.type
        self.irMetadata.type = new_type
                    


class IrCombineToPoly(IrAst):
    def __init__(self, coeffIr, constIr):
        super().__init__()

        assert(coeffIr.irMetadata.shape[:-1] == constIr.irMetadata.shape)
        assert(coeffIr.irMetadata.isExpanded == constIr.irMetadata.isExpanded)
        assert(coeffIr.irMetadata.isConst == constIr.irMetadata.isConst)

        self.coeffIr = coeffIr
        self.constIr = constIr
        self.irMetadata = IrMetadata(coeffIr.irMetadata.shape, 'PolyExp', coeffIr.irMetadata.isExpanded, False)


class IrTernarySimple(IrAst):
    def __init__(self, cond_ir, lhs_ir, rhs_ir):
        super().__init__()
        self.cond_ir = cond_ir
        self.lhs_ir = lhs_ir
        self.rhs_ir = rhs_ir

        self.cond_ir, self.lhs_ir = matchDims(self.cond_ir, self.lhs_ir)
        self.cond_ir, self.rhs_ir = matchDims(self.cond_ir, self.rhs_ir)
        self.lhs_ir, self.rhs_ir = matchDims(self.lhs_ir, self.rhs_ir)

        self.irMetadata = self.cond_ir.irMetadata.copy()

class IrTernaryOuter(IrAst):
    def __init__(self, cond_ir, lhs_irs, rhs_irs):
        super().__init__()
        self.cond_ir = cond_ir
        self.lhs_irs = lhs_irs
        self.rhs_irs = rhs_irs

        

class IrReduce(IrAst):
    def __init__(self, input_ir):
        super().__init__()
        assert(isinstance(input_ir, IrFuncCall))
        assert(input_ir.fromMap)
        self.input_ir = input_ir 
        input_shape = input_ir.irMetadata.shape.copy()

        if len(input_shape) == 1:
            new_shape = [1]
            new_type = 'Float'
        else:
            new_shape = input_shape[1:]
            new_type = 'PolyExp'

        self.isMetadata = IrMetadata(new_shape, new_type, input_ir.irMetadata.isExpanded, False)

class IrFuncCall(IrAst):
    def __init__(self, name, argIrs, irMetadata, fromMap = False):
        super().__init__()
        assert(not(fromMap) or (irMetadata == argIrs[0].irMetadata))
        self.name = name
        self.input_ir = argIrs
        self.irMetadata = irMetadata
        self.fromMap = fromMap

class IrMap(IrAst):
    def __init__(self, polyExpIr, funcName, irMetadata):
        super().__init__()
        self.polyExpIr = polyExpIr
        self.funcName = funcName
        self.irMetadata = irMetadata

class IrTraverse(IrAst):
    def __init__(self, polyExpIr, stop, priority, funcName, direction, irMetadata):
        super().__init__()
        (irMetadata == polyExpIr.irMetadata)
        self.polyExpIr = polyExpIr 
        self.stop = stop 
        self.priority = priority 
        self.funcName = funcName 
        self.direction = direction 
        self.irMetadata = irMetadata


class IrOpStmt(IrAst):
    def __init__(self, op, input_ir):
        super().__init__()
        self.op = op
        self.input_ir = input_ir