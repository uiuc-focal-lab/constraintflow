import copy
from z3 import *
from constraintflow.compiler.globals import *

# POLYEXP_SIZE = Int('poly_size')
# CURR_SIZE = Int('curr_size')
# PREV_SIZE = Int('prev_size')

def get_complete_shape_broadcast(irMetadata):
    shape = []
    broadcast = []
    for i in range(len(irMetadata)):
        shape += (irMetadata[i].shape)
        broadcast += (irMetadata[i].broadcast)
    return shape, broadcast

def collect_multiplicands(expr):
    if not (isinstance(expr, IrMult) and (expr.op == '*')):
        return [expr]
    multiplicands = []
    for child in expr.children:
        multiplicands.extend(collect_multiplicands(child))
    return multiplicands

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

def divide_metadata(m1, m2):
    if isinstance(m1, int):
        assert (m1==1) and (m2==1)
        return m1
    if isinstance(m2, int):
        assert(m2==1)
        return m1 
    else:
        lhs_multiplicands = collect_multiplicands(m1)
        rhs_multiplicands = collect_multiplicands(m2)
        # Computes (∏ lhs) / (∏ rhs). Must ensure: rhs ⊆ lhs.
        new_metadata = []
        for i in range(len(lhs_multiplicands)):
            new_metadata.append(lhs_multiplicands[i])
        for i in range(len(rhs_multiplicands)):
            for j in range(len(new_metadata)-1,-1,-1):
                if new_metadata[j] == rhs_multiplicands[i]:
                    del new_metadata[j]
                    break
        new_metadata_elem = 1
        for i in range(len(new_metadata)):
            new_metadata_elem = mult_metadata(new_metadata_elem, new_metadata[i])
        return new_metadata_elem

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
    
    def is_expanded(self):
        for i in self.broadcast:
            if i != 1:
                # Some dimension's broadcast ≠ 1 means not yet expanded.
                return False 
        return True
    
    def __eq__(self,obj):
        return self.shape == obj.shape and self.type == obj.type and self.isConst == obj.isConst and self.broadcast == obj.broadcast

    def __str__(self):
        return ' '

def is_expanded_metadata(irMetadata):
    for i in range(len(irMetadata)):
        if not irMetadata[i].is_expanded():
            return False
    return True

def expand_irMetadata(irMetadata):
    res = copy_metadata(irMetadata)
    for i in range(len(res)):
        for j in range(len(res[i].shape)):
            if check_eq(res[i].broadcast[j], 1):
                continue
            if not check_eq(res[i].shape[j], 1):
                print(check_eq(res[i].shape[j], 1))
                print(check_eq(res[i].broadcast[j], 1))
            # If broadcast[j] ≠ 1, then shape[j] must == 1.
            assert(check_eq(res[i].shape[j], 1))
            res[i].shape[j] = res[i].broadcast[j]
            res[i].broadcast[j] = 1
    return res

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
    """
    Checks that irMetadata1 and 2 have the same expression form of shape, not
      only semantically the same shape.
    That is, their shapes and broadcasts must be equal at each index (same
      expansion status).
    """
    if len(irMetadata1) != len(irMetadata2):
        return False
    for i in range(len(irMetadata1)):
        if len(irMetadata1[i].shape) != len(irMetadata2[i].shape):
            return False 
        for j in range(len(irMetadata1[i].shape)):
            if not check_eq(irMetadata1[i].shape[j], irMetadata2[i].shape[j]):
                return False
            if not check_eq(irMetadata1[i].broadcast[j], y=irMetadata2[i].broadcast[j]):
                return False
    return True

def createLcm(irMetadata1, irMetadata2):
    """
    Make two IR metadata's compatible by expanding any side whose shape is 1 to
      match the other side's non-1 shape, as long as allowed by `broadcast`.
    Does not mutate arguments, returns changed copies.
    Returns
    - flag[1/2]: Whether irMetadata[1/2] is expanded. Caller may need to
      construct IrRepeat node(s) accordingly.
    - newIrMetadata[1/2]: Updated IR metadata copies after expansion.
    """
    flag1 = False 
    flag2 = False
    newIrMetadata1 = copy_metadata(irMetadata1)
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
    """
    Checks if irMetadata1 can be repeated to match irMetadata2's shape (up to
      broadcast semantics), or vice versa.
    """
    if len(irMetadata1) != len(irMetadata2):
        return False 
    for i in range(len(irMetadata1)):
        assert(irMetadata1[i].type == irMetadata2[i].type)
        assert(irMetadata1[i].isConst == irMetadata2[i].isConst)
        assert(len(irMetadata1[i].shape) == len(irMetadata2[i].shape))
        if not irMetadata1[i].isConst:
            for j in range(len(irMetadata1[i].shape)):
                if not check_eq(mult_metadata(irMetadata1[i].shape[j], irMetadata1[i].broadcast[j]), mult_metadata(irMetadata2[i].shape[j], irMetadata2[i].broadcast[j])):
                    return False
    return True

def canAddDimension(irMetadataElement1, irMetadataElement2):
    if len(irMetadataElement1.shape) >= len(irMetadataElement2.shape):
        return False 
    for i in range(len(irMetadataElement1.shape)):
        if not check_eq(mult_metadata(irMetadataElement1.shape[i],irMetadataElement1.broadcast[i]), mult_metadata(irMetadataElement2.shape[i],irMetadataElement2.broadcast[i])):
            return False
    for i in range(len(irMetadataElement1.shape), len(irMetadataElement2.shape)):
        if irMetadataElement2.shape[i]!=1:
            # If the extra dimensions of irMetadataElement2 are not 1, then
            #   they are not just placeholders (have extra meanings).
            # They cannot be directly  added to irMetadataElement1.
            return False
    return True

def matchDims(lhsIr, rhsIr):
    """
    Aligns two IR nodes before operation by wrapping around them.
    After that, the IR nodes are dimension-compatible.
    """
    lhsIrMetadata = lhsIr.irMetadata
    rhsIrMetadata = rhsIr.irMetadata
    if checkEqualMetadata(lhsIr.irMetadata, rhsIr.irMetadata):
        return lhsIr, rhsIr
    if lhsIrMetadata[-1].isConst:
        # QUESTION: What if lhsIrMetadata[-1].isConst but not
        #   lhsIrMetadata[i].isConst for some other i?
        updated_irMetadata = copy_metadata(rhsIrMetadata)
        for i in range(len(updated_irMetadata)):
            updated_irMetadata[i].isConst = True 
            updated_irMetadata[i].type = lhsIrMetadata[-1].type
        lhsIr = IrAddDimensionConst(lhsIr, updated_irMetadata)
        return lhsIr, rhsIr

    if rhsIrMetadata[-1].isConst:
        updated_irMetadata = copy_metadata(lhsIrMetadata)
        for i in range(len(updated_irMetadata)):
            updated_irMetadata[i].isConst = True 
            updated_irMetadata[i].type = rhsIrMetadata[-1].type
        rhsIr = IrAddDimensionConst(rhsIr, updated_irMetadata) 
        return lhsIr, rhsIr

    # TODO: CHECK THIS ASSERT
    assert(len(lhsIrMetadata) == len(rhsIrMetadata))

    if len(lhsIrMetadata[-1].shape) < len(rhsIrMetadata[-1].shape):
        updated_irMetadataElement = rhsIrMetadata[-1].copy()
        for i in range(len(lhsIr.irMetadata[-1].shape)):
            updated_irMetadataElement.shape[i] = lhsIr.irMetadata[-1].shape[i]
            updated_irMetadataElement.broadcast[i] = lhsIr.irMetadata[-1].broadcast[i]
        for i in range(len(lhsIrMetadata[-1].shape), len(rhsIrMetadata[-1].shape)):
            updated_irMetadataElement.broadcast[i] = mult_metadata(updated_irMetadataElement.broadcast[i], updated_irMetadataElement.shape[i])
            updated_irMetadataElement.shape[i] = 1
        lhsIr = IrAddDimension(lhsIr, updated_irMetadataElement)

    elif len(rhsIrMetadata[-1].shape) < len(lhsIrMetadata[-1].shape):
        updated_irMetadataElement = lhsIrMetadata[-1].copy()
        for i in range(len(rhsIrMetadata[-1].shape), len(lhsIrMetadata[-1].shape)):
            updated_irMetadataElement.broadcast[i] = mult_metadata(updated_irMetadataElement.broadcast[i], updated_irMetadataElement.shape[i])
            updated_irMetadataElement.shape[i] = 1
        rhsIr = IrAddDimension(rhsIr, updated_irMetadataElement)

    lhsIrMetadata = lhsIr.irMetadata
    rhsIrMetadata = rhsIr.irMetadata

    if checkEqualMetadata(lhsIrMetadata, rhsIrMetadata):
        return lhsIr, rhsIr
    

    flag1, flag2, newIrMetadata1, newIrMetadata2 = createLcm(lhsIr.irMetadata, rhsIr.irMetadata)
    if flag1:
        lhsIr = IrRepeat(lhsIr, newIrMetadata1)
    if flag2:
        rhsIr = IrRepeat(rhsIr, newIrMetadata2)
    return lhsIr, rhsIr

_EQ_BOOKKEEPING = frozenset((
    'node_name', 'identifier', 'parents', 'children', 'irMetadata',
    'inside_while', 'while_number', 'ttb_counter', 'while_iteration'))


def _value_equal(x, y):
    """One attribute value; nested IR nodes and torch tensors both turn up here."""
    if x is y:
        return True
    if isinstance(x, IrAst) or isinstance(y, IrAst):
        return isinstance(x, IrAst) and isinstance(y, IrAst) and x == y
    if type(x) is not type(y):
        return False
    if isinstance(x, (list, tuple)):
        return len(x) == len(y) and all(_value_equal(p, q) for p, q in zip(x, y))
    if isinstance(x, dict):
        return x.keys() == y.keys() and all(_value_equal(x[k], y[k]) for k in x)
    #FIX FOR: tensor == broadcasts, so IrTensorOnes sizes [100, 100] and [100] compared equal and CSE merged
    # ones([batch, 100]) with ones([100]) whenever the batch size equaled a layer width (reshape crash at runtime).
    if getattr(x, 'shape', None) != getattr(y, 'shape', None):
        return False
    try:
        result = (x == y)
    except Exception:
        return False
    if isinstance(result, bool):
        return result
    try:
        return bool(result.all())
    except Exception:
        return False


def _attrs_equal(a, b):
    keys = {k for k in vars(a) if k not in _EQ_BOOKKEEPING}
    if keys != {k for k in vars(b) if k not in _EQ_BOOKKEEPING}:
        return False
    return all(_value_equal(getattr(a, k), getattr(b, k)) for k in keys)


class IrAst:
    counter = 0
    poly_size = None
    sym_size = None
    curr_size = None
    prev_size = None
    trav_size = None
    batch_size = None

    def __init__(self):
        IrAst.counter += 1
        self.node_name = "IR_ASTNode"
        self.identifier = IrAst.counter
        self.parents = []
        self.children = []
        self.irMetadata = []
        self.inside_while = False
        self.while_number = -1
        
    
    def update_parent_child(self, children):
        self.children = children 
        for child in children:
            if child ==None:
                continue
            if isinstance(child, int):
                continue
            # new_parents = set(child.parents)
            # new_parents.add(self)
            # child.parents = list(new_parents)

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
        if type(self) != type(obj):
            return False
        if len(self.children) != len(obj.children):
            return False
        if not _attrs_equal(self, obj):
            return False
        for i in range(len(self.children)):
            if self.children[i] != obj.children[i]:
                return False
        return checkEqualMetadata(self.irMetadata, obj.irMetadata)


class IrExpression(IrAst):
    def __init__(self):
        super().__init__()
        self.ttb_counter = -1
        self.while_number = -1
        self.while_iteration = -1
        

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
            # IrAst.trav_size = IrAst.poly_size
            IrAst.sym_size = IrVar('sym_size', [IrMetadataElement([1], 'Int', [1], True)])
            IrAst.trav_size = IrVar('trav_size', [IrMetadataElement([1], 'Int', [1], True)])
            IrAst.batch_size = IrVar('batch_size', [IrMetadataElement([1], 'Int', [1], True)])
        
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
    
class IrSparseTensor(IrExpression):
    def __init__(self, start_indices, blocksIr, dims, total_size, end_indices=None, type=None, dense_const=None, delete_indices=None):
        super().__init__()
        self.start_indices = start_indices
        self.dims = dims
        self.total_size = total_size
        if end_indices is not None:
            self.end_indices = end_indices
        if type is not None:
            self.type = type
        if dense_const is not None:
            self.dense_const = dense_const
        # Recorded by SparseTensorConstructor only; the hand-written records predate it.
        if delete_indices is not None:
            self.delete_indices = delete_indices
        self.update_parent_child([blocksIr])


    def __str__(self):
        print(type(self), self.name)
        return ''

    def __hash__(self):
        return 0

    def hash(self):
        self.hash_str = str(type(self))
        self.hash_str += self.start_indices
        # self.hash_str += self.blocks.hash()
        self.hash_str += self.dims
        self.hash_str += self.total_size
        self.hash_str += self.end_indices
        self.hash_str += self.type
        self.hash_str += self.dense_const
        return self.hash_str


class IrConstBlock(IrExpression):
    def __init__(self, inputIr, total_shape, sparse_block_type=None):
        super().__init__()
        self.total_shape = total_shape
        if sparse_block_type is not None:
            self.sparse_block_type = sparse_block_type
        self.update_parent_child([inputIr])

class IrPatchesBlock(IrExpression):
    def __init__(self, blockIr, total_shape, ix, iy, ox, oy, sx, sy, px, py, kx, ky, num_channels, num_kernels,
                 sparse_block_type=None):
        super().__init__()
        self.total_shape = total_shape
        if sparse_block_type is not None:
            self.sparse_block_type = sparse_block_type
        self.ix = ix
        self.iy = iy
        self.ox = ox
        self.oy = oy
        self.sx = sx
        self.sy = sy
        self.px = px
        self.py = py
        self.kx = kx
        self.ky = ky
        self.num_channels = num_channels
        self.num_kernels = num_kernels
        self.update_parent_child([blockIr])

class IrRepeatBlock(IrExpression):
    def __init__(self, blockIr, total_shape, repeat_dims=None, only_one_repeat=None, sparse_block_type=None):
        super().__init__()
        self.total_shape = total_shape
        if repeat_dims is not None:
            self.repeat_dims = repeat_dims
        if only_one_repeat is not None:
            self.only_one_repeat = only_one_repeat
        if sparse_block_type is not None:
            self.sparse_block_type = sparse_block_type
        self.update_parent_child([blockIr])


# sparse_block.get_diagonal simulacrum trace — step 1: torch.diagonal on block tensor payload.
class IrTorchDiagonal(IrExpression):
    def __init__(self, inputIr, dim1, dim2):
        super().__init__()
        self.dim1 = dim1
        self.dim2 = dim2
        self.update_parent_child([inputIr])


# sparse_block.get_diagonal simulacrum trace — step 2: permute after diagonal.
class IrTorchPermute(IrExpression):
    def __init__(self, inputIr, permutation):
        super().__init__()
        self.permutation = permutation
        self.update_parent_child([inputIr])

class IrTorchTranspose(IrExpression):
    def __init__(self, inputIr, dim0, dim1):
        super().__init__()
        self.dim0 = dim0
        self.dim1 = dim1
        self.update_parent_child([inputIr])

class IrTorchMatmul(IrExpression):
    def __init__(self, lhsIr, rhsIr):
        super().__init__()
        self.update_parent_child([lhsIr, rhsIr])

class IrTorchWhere(IrExpression):
    def __init__(self, condIr, lhsIr, rhsIr):
        super().__init__()
        self.update_parent_child([condIr, lhsIr, rhsIr])

class IrTorchUnsqueeze(IrExpression):
    def __init__(self, inputIr, index):
        super().__init__()
        self.index = index
        self.update_parent_child([inputIr])

class IrTorchSqueeze(IrExpression):
    def __init__(self, inputIr, index):
        super().__init__()
        self.index = index
        self.update_parent_child([inputIr])

class IrTorchReshape(IrExpression):
    def __init__(self, inputIr, shape):
        super().__init__()
        self.shape = shape
        self.update_parent_child([inputIr])

class IrTorchView(IrExpression):
    def __init__(self, inputIr, shape):
        super().__init__()
        self.shape = shape
        self.update_parent_child([inputIr])

class IrTorchRepeat(IrExpression):
    def __init__(self, inputIr, repeats):
        super().__init__()
        self.repeats = repeats
        self.update_parent_child([inputIr])

class IrTorchExpand(IrExpression):
    def __init__(self, inputIr, shape):
        super().__init__()
        self.shape = shape
        self.update_parent_child([inputIr])

class IrTorchPad(IrExpression):
    def __init__(self, inputIr, pad):
        super().__init__()
        self.pad = pad
        self.update_parent_child([inputIr])

class IrTorchSum(IrExpression):
    def __init__(self, inputIr, dim):
        super().__init__()
        self.dim = dim
        self.update_parent_child([inputIr])

class IrTorchZeros(IrExpression):
    def __init__(self, size, device=None, dtype=None):
        super().__init__()
        self.size = size
        self.device = device
        self.dtype = dtype

class IrTorchEye(IrExpression):
    def __init__(self, size, device=None, dtype=None):
        super().__init__()
        self.size = size
        self.device = device
        self.dtype = dtype

class IrTorchFloat(IrExpression):
    def __init__(self, inputIr):
        super().__init__()
        self.update_parent_child([inputIr])

class IrTorchDiagEmbed(IrExpression):
    def __init__(self, inputIr):
        super().__init__()
        self.update_parent_child([inputIr])

class IrTorchStride(IrExpression):
    def __init__(self, inputIr):
        super().__init__()
        self.update_parent_child([inputIr])

class IrTorchAsStrided(IrExpression):
    def __init__(self, inputIr, size, stride):
        super().__init__()
        self.size = size
        self.stride = stride
        self.update_parent_child([inputIr])

class IrTorchSlice(IrExpression):
    def __init__(self, inputIr, index):
        super().__init__()
        self.index = index
        self.update_parent_child([inputIr])

class IrPatchesToDense(IrExpression):
    def __init__(self, inputIr, geometry):
        super().__init__()
        self.geometry = geometry
        self.update_parent_child([inputIr])


class IrRowGather(IrExpression):
    """Select axis-one rows, optionally mapping flattened rows to channels/tiles."""
    def __init__(self, inputIr, divisor=1, period=None):
        super().__init__()
        self.divisor, self.period = divisor, period
        self.update_parent_child([inputIr, IrVar('_idx', [])])


class IrSelectedPatchesToDense(IrExpression):
    def __init__(self, inputIr, geometry):
        super().__init__()
        self.geometry = geometry
        self.update_parent_child([inputIr, IrVar('_idx', [])])


class IrTensorScatter(IrExpression):
    def __init__(self, inputIr, valueIr, dim, index):
        super().__init__()
        self.dim = dim
        self.index = index
        self.update_parent_child([inputIr, valueIr])

class IrFConv2d(IrExpression):
    def __init__(self, inputIr, weightIr, stride, padding, biasIr=None):
        super().__init__()
        self.stride = stride
        self.padding = padding
        # Optional per-channel bias, applied by the convolution itself.
        self.update_parent_child([inputIr, weightIr] + ([biasIr] if biasIr is not None else []))

class IrFConvTranspose2d(IrExpression):
    def __init__(self, inputIr, weightIr, stride=None, padding=None, output_padding=None):
        super().__init__()
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.update_parent_child([inputIr, weightIr])

class IrFUnfold(IrExpression):
    def __init__(self, inputIr, kernel_size, padding, stride):
        super().__init__()
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.update_parent_child([inputIr])

class IrFFold(IrExpression):
    def __init__(self, inputIr, output_size, kernel_size, stride, padding=None):
        super().__init__()
        self.output_size = output_size
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.update_parent_child([inputIr])

class IrTorchEinsum(IrExpression):
    def __init__(self, equation, operandIrs):
        super().__init__()
        self.equation = equation
        self.update_parent_child(operandIrs)

class IrTorchCat(IrExpression):
    def __init__(self, inputIrs, dim):
        super().__init__()
        self.dim = dim
        self.update_parent_child(list(inputIrs))

class IrAssignToView(IrStatement):
    def __init__(self, inputIr, index, valueIr):
        super().__init__()
        self.index = index
        self.update_parent_child([inputIr, valueIr])


# create_similar_diagonal trace: wrap tensor payload as DiagonalBlock.
class IrDiagonalBlock(IrExpression):
    def __init__(self, blockIr, total_shape, diag_index, batch_size=None, sparse_block_type=None):
        super().__init__()
        self.total_shape = total_shape
        self.diag_index = diag_index
        if batch_size is not None:
            self.batch_size = batch_size
        if sparse_block_type is not None:
            self.sparse_block_type = sparse_block_type
        self.update_parent_child([blockIr])


class IrEmptyList(IrExpression):
    def __init__(self):
        super().__init__()
        self.update_parent_child([]) 

    def __eq__(self, obj):
        return False

class IrAppendList(IrExpression):
    def __init__(self, listIr, elementIr):
        super().__init__()
        self.update_parent_child([listIr, elementIr])

class IrListExtract(IrExpression):
    def __init__(self, listIr, indexIr):
        super().__init__()
        self.update_parent_child([listIr, indexIr])

class IrBlockExtract(IrExpression):
    def __init__(self, inputIr, indexIr):
        super().__init__()
        self.update_parent_child([inputIr, indexIr])

class IrBlockCopy(IrExpression):
    def __init__(self, inputIr):
        super().__init__()
        self.update_parent_child([inputIr])

class IrGetSparseTensorBlocks(IrExpression):
    def __init__(self, inputIr):
        super().__init__()
        self.update_parent_child([inputIr])

class IrGetAbsElemSparseDKey(IrExpression):
    def __init__(self, inputIr, key: str):
        super().__init__()
        self.key = key
        self.ttb_counter = -1
        self.update_parent_child([inputIr])

class IrGetPolyExpSparseConst(IrExpression):
    def __init__(self, inputIr):
        super().__init__()
        self.update_parent_child([inputIr])

class IrGetPolyExpSparseMat(IrExpression):
    def __init__(self, inputIr):
        super().__init__()
        self.update_parent_child([inputIr])

class IrGetSymExpSparseConst(IrExpression):
    def __init__(self, inputIr):
        super().__init__()
        self.update_parent_child([inputIr])

class IrGetSymExpSparseMat(IrExpression):
    def __init__(self, inputIr):
        super().__init__()
        self.update_parent_child([inputIr])

class IrGetKthLayerNetworkParam(IrExpression):
    def __init__(self, layer_index, param):
        super().__init__()
        self.layer_index = layer_index
        self.param = param
        self.update_parent_child([])

class IrDenseBlock(IrExpression):
    def __init__(self, inputIr, total_shape=None, batch_size=None, sparse_block_type=None):
        super().__init__()
        if total_shape is not None:
            self.total_shape = total_shape
        if batch_size is not None:
            self.batch_size = batch_size
        if sparse_block_type is not None:
            self.sparse_block_type = sparse_block_type
        self.update_parent_child([inputIr])

class IrKernelBlock(IrExpression):
    def __init__(self, inputIr, total_shape: list[int],
                 ix, iy, ox, oy, sx, sy, px, py,
                 kx=None, ky=None, num_channels=None, num_kernels=None,
                 sparse_block_type=None):
        """inputIr: the parameter `block` of the `KernelBlock` constructor"""
        super().__init__()
        self.total_shape = total_shape
        self.ix = ix
        self.iy = iy
        self.ox = ox
        self.oy = oy
        self.sx = sx
        self.sy = sy
        self.px = px
        self.py = py
        if kx is not None:
            self.kx = kx
        if ky is not None:
            self.ky = ky
        if num_channels is not None:
            self.num_channels = num_channels
        if num_kernels is not None:
            self.num_kernels = num_kernels
        if sparse_block_type is not None:
            self.sparse_block_type = sparse_block_type
        self.update_parent_child([inputIr])

class IrEpsilon(IrExpression):
    def __init__(self, num=None):
        super().__init__()
        if num==None:
            num = IrAst.curr_size
        self.num = num
        self.irMetadata = [IrMetadataElement([IrAst.batch_size, num], 'SymExp', [1, 1], False)]
        # self.update_parent_child([IrConst(num)])


# Reconstructs the SymExpSparse produced by get_new_eps from its replayed
# mat and const SparseTensors (the tensor_to_block expansion of IrEpsilon).
class IrNewEps(IrExpression):
    def __init__(self, matIr, constIr):
        super().__init__()
        self.update_parent_child([matIr, constIr])

    
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
    def __init__(self, inputIr, irMetadata=None, repeat_dims = None):
        super().__init__()
        if irMetadata==None and repeat_dims==None:
            self.irMetadata = copy_metadata(inputIr.irMetadata)
            repeat_dims = []
            for i in range(len(inputIr.irMetadata)):
                for j in range(len(inputIr.irMetadata[i].broadcast)):
                    repeat_dims.append(inputIr.irMetadata[i].broadcast[j])
                    self.irMetadata[i].shape[j] = mult_metadata(self.irMetadata[i].shape[j], repeat_dims[-1])
                    self.irMetadata[i].broadcast[j] = 1
            children = [inputIr] + repeat_dims
            self.update_parent_child(children)
        
        elif irMetadata==None:
            self.irMetadata = copy_metadata(inputIr.irMetadata)
            counter = 0
            for i in range(len(self.irMetadata)):
                for j in range(len(self.irMetadata[i].shape)):
                    self.irMetadata[i].shape[j] = mult_metadata(self.irMetadata[i].shape[j], repeat_dims[counter])
                    self.irMetadata[i].broadcast[j] = divide_metadata(self.irMetadata[i].broadcast[j], repeat_dims[counter])
                    counter += 1
            children = [inputIr] + repeat_dims
            self.update_parent_child(children)

        else:
            assert(canRepeat(inputIr.irMetadata, irMetadata))
            
            repeat_dims = []
            for i in range(len(inputIr.irMetadata)):
                for j in range(len(inputIr.irMetadata[i].broadcast)):
                    repeat_dims.append(divide_metadata(irMetadata[i].shape[j], inputIr.irMetadata[i].shape[j]))
            children = [inputIr] + repeat_dims
            self.irMetadata = irMetadata
            self.update_parent_child(children)

class IrConvertBoolToFloat(IrExpression):
    def __init__(self, inputIr):
        super().__init__()
        # self.irMetadata = copy.deepcopy(inputIr.irMetadata)
        self.irMetadata = copy_metadata(inputIr.irMetadata)
        self.irMetadata[-1].type = 'Float'
        self.update_parent_child([inputIr])


class IrGetDefaultStop(IrExpression):
    def __init__(self, inputIr):
        super().__init__()
        self.irMetadata = copy_metadata(inputIr.irMetadata)
        self.irMetadata[-1].shape.append(IrAst.poly_size)
        self.irMetadata[-1].broadcast.append(1)
        self.irMetadata[-1].type = 'Bool'
        self.irMetadata[-1].isConst = False

        repeat_dims = []
        for i in range(len(self.irMetadata)):
            for j in range(len(self.irMetadata[i].shape)):
                repeat_dims.append(self.irMetadata[i].shape[j])
        
        new_children = [inputIr] + repeat_dims
        self.update_parent_child(new_children)

class IrGetPriorityLList(IrExpression):
    def __init__(self, inputIr, stopIr):
        super().__init__()
        self.irMetadata = copy_metadata(inputIr.irMetadata)
        self.irMetadata[-1].type = 'Bool'
        self.irMetadata[-1].isConst = False
        
        new_children = [inputIr, stopIr]
        self.update_parent_child(new_children)

class IrGetPolyexpStop(IrExpression):
    def __init__(self, inputIr, stopIr):
        super().__init__()
        self.irMetadata = copy_metadata(inputIr.irMetadata)
        
        new_children = [inputIr, stopIr]
        self.update_parent_child(new_children)

class IrGetPolyexpNotStop(IrExpression):
    def __init__(self, inputIr, stopIr):
        super().__init__()
        self.irMetadata = copy_metadata(inputIr.irMetadata)
        
        new_children = [inputIr, stopIr]
        self.update_parent_child(new_children)

class IrAddDimension(IrExpression):
    def __init__(self, inputIr, irMetadataElement):
        super().__init__()
        assert(canAddDimension(inputIr.irMetadata[-1], irMetadataElement))

        self.irMetadata = inputIr.irMetadata[:-1].copy()
        self.irMetadata.append(irMetadataElement) 
        self.update_parent_child([inputIr])
        
class IrRemoveDimension(IrExpression):
    def __init__(self, inputIr, numDim):
        super().__init__()
        counter = 0
        checked = False
        self.irMetadata = copy_metadata(inputIr.irMetadata)
        self.numDim = numDim
        for i in range(len(inputIr.irMetadata)):
            for j in range(len(inputIr.irMetadata[i].shape)):
                if counter == numDim:
                    assert(inputIr.irMetadata[i].shape[j] == 1)
                    del self.irMetadata[i].shape[j]
                    del self.irMetadata[i].broadcast[j]
                    checked = True
                    # break
                counter += 1
        assert(checked)
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
                repeat_dims.append(self.irMetadata[i].shape[j])
        new_children = [inputIr] + repeat_dims

        self.update_parent_child(new_children)
        self.irMetadata[-1].isConst = False






        

class IrConvertNeuronToPoly(IrExpression):
    def __init__(self, inputIr):
        super().__init__()
        assert(inputIr.irMetadata[-1].type == 'Neuron')

        self.irMetadata = copy_metadata(inputIr.irMetadata)
        self.irMetadata[-1].type = 'PolyExp'
        self.update_parent_child([inputIr])


class IrConcatStitch(IrExpression):
    """Opaque, traced 2-D Concat block re-stitch (see
    constraintflow/gbcsr/tensor_ops.py:concat_stitch_2d). `prev1Ir`/`prev2Ir`
    are only used by this node's own (non-reuse) codegen -- at reuse time
    tensor_to_block.py discards them and rebuilds the expression entirely
    from the captured tape, exactly like IrGetPriorityLList and friends.

    __eq__ always returns False (matching IrExpandSymExp/IrEmptyList): the
    base IrAst.__eq__ compares only type+children+irMetadata, which does NOT
    see `key`/`source` -- two keys of the same type (e.g. 'l' and 'u', both
    Float) share identical prev1/prev2 children and identical metadata, so
    without this override CSE merges them into one node and silently drops
    one key's data."""
    def __init__(self, prev1Ir, prev2Ir, key, source, irMetadata):
        super().__init__()
        assert source in ('direct', 'const')
        self.key = key
        self.source = source
        self.irMetadata = irMetadata
        self.update_parent_child([prev1Ir, prev2Ir])

    def __eq__(self, obj):
        return False

    def __hash__(self):
        return 0


class IrUpdateReplay(IrExpression):
    """Opaque, traced Abs_elem_sparse.update for one shape field (jit_update capture).

    `operandIr` is the op's abstract-shape output for this field; the tape refers to
    it as `abs_shape_<n>` and to the pre-update state via IrGetAbsElemSparseDKey.
    __eq__ is always False for the same reason as IrConcatStitch: the base __eq__ sees
    neither `key` nor `records`, so CSE would merge two same-typed fields into one."""
    def __init__(self, operandIr, key, records, irMetadata):
        super().__init__()
        self.key = key
        self.records = records
        self.irMetadata = irMetadata
        self.update_parent_child([operandIr])

    def __eq__(self, obj):
        return False

    def __hash__(self):
        return 0


class IrConcatStitchMat(IrExpression):
    """Opaque, traced 3-D Concat block re-stitch for a PolyExp's .mat (see
    constraintflow/gbcsr/tensor_ops.py:concat_stitch_mat). See IrConcatStitch
    for why __eq__ must always return False here too."""
    def __init__(self, prev1Ir, prev2Ir, key, irMetadata):
        super().__init__()
        self.key = key
        self.irMetadata = irMetadata
        self.update_parent_child([prev1Ir, prev2Ir])

    def __eq__(self, obj):
        return False

    def __hash__(self):
        return 0



class IrConvertConstToPoly(IrExpression):
    def __init__(self, inputIr):
        super().__init__()
        assert((inputIr.irMetadata[-1].type == 'Float') or (inputIr.irMetadata[-1].type == 'Int'))
        
        self.irMetadata = copy_metadata(inputIr.irMetadata)
        self.irMetadata[-1].type = 'PolyExp'
        new_children = [inputIr, inputIr.irMetadata[-1].shape[0]]
        self.update_parent_child(new_children)

class IrConvertConstToSym(IrExpression):
    def __init__(self, inputIr):
        super().__init__()
        assert((inputIr.irMetadata[-1].type == 'Float') or (inputIr.irMetadata[-1].type == 'Int'))
        
        self.irMetadata = copy_metadata(inputIr.irMetadata)
        self.irMetadata[-1].type = 'SymExp'
        new_children = [inputIr, inputIr.irMetadata[-1].shape[0]]
        self.update_parent_child(new_children)


class IrExtractPolyCoeff(IrExpression):
    def __init__(self, inputIr):
        super().__init__()
        assert(inputIr.irMetadata[-1].type == 'PolyExp')

        self.irMetadata = copy_metadata(inputIr.irMetadata)
        self.irMetadata[-1].type = 'Float'
        self.irMetadata[-1].shape.append(IrAst.poly_size)
        self.irMetadata[-1].broadcast.append(1)
        self.update_parent_child([inputIr])

class IrExtractSymCoeff(IrExpression):
    def __init__(self, inputIr):
        super().__init__()
        assert(inputIr.irMetadata[-1].type == 'SymExp')

        self.irMetadata = copy_metadata(inputIr.irMetadata)
        self.irMetadata[-1].type = 'Float'
        self.irMetadata[-1].shape.append(IrAst.sym_size)
        self.irMetadata[-1].broadcast.append(1)
        self.update_parent_child([inputIr])


class IrExtractPolyConst(IrExpression):
    def __init__(self, inputIr):
        super().__init__()
        assert(inputIr.irMetadata[-1].type == 'PolyExp')

        self.inputIr = inputIr 
        self.irMetadata = copy_metadata(inputIr.irMetadata)
        self.irMetadata[-1].type = 'Float'
        self.update_parent_child([inputIr])

class IrExtractSymConst(IrExpression):
    def __init__(self, inputIr):
        super().__init__()
        assert(inputIr.irMetadata[-1].type == 'SymExp')

        self.inputIr = inputIr 
        self.irMetadata = copy_metadata(inputIr.irMetadata)
        self.irMetadata[-1].type = 'Float'
        self.update_parent_child([inputIr])

class IrExpandSymExp(IrExpression):
    def __init__(self, inputIr):
        super().__init__()
        self.irMetadata = copy_metadata(inputIr.irMetadata)
        self.update_parent_child([inputIr])

    def __eq__(self, obj):
        return False
    
    def __hash__(self):
        return 0

class IrAccess(IrExpression):
    def __init__(self, lhsIr, elem, type=None, isMetadata=False):
        super().__init__()
        self.isMetadata = isMetadata
        self.elem = elem
        self.ttb_counter = -1

        lhsIrMetadata = lhsIr.irMetadata

        self.irMetadata = copy_metadata(lhsIrMetadata)
        self.update_parent_child([lhsIr])

        if isMetadata:
            assert(lhsIrMetadata[-1].type == 'Neuron')
            if elem == 'weight':
                self.irMetadata[-1].shape.append(IrAst.prev_size)
                self.irMetadata[-1].broadcast.append(1)
                self.irMetadata[-1].type = 'Float'
                for i in range(len(self.irMetadata)):
                    for j in range(len(self.irMetadata[i].shape)):
                        if self.irMetadata[i].broadcast[j] != 1:
                            self.irMetadata[i].shape[j] = mult_metadata(self.irMetadata[i].shape[j], self.irMetadata[i].broadcast[j])
                            self.irMetadata[i].broadcast[j] = 1

            elif elem == 'bias':
                self.irMetadata[-1].type = 'Float'

            elif elem == 'layer':
                self.irMetadata[-1].type = 'Int'
            
        
        else:
            self.irMetadata[-1].type = type
            self.irMetadata[0].shape[0] = IrAst.batch_size
            self.irMetadata[0].broadcast[0] = 1
            
    def __eq__(self, obj):
        if isinstance(obj, IrAccess):
            if self.elem == obj.elem and self.isMetadata == obj.isMetadata:
                return self.children[0] == obj.children[0]
        return False
    
    def __hash__(self):
        return 0

class IrBinaryOp(IrExpression):
    def __init__(self, lhsIr, rhsIr, op):
        super().__init__()
        self.op = op
        lhsIr, rhsIr = matchDims(lhsIr, rhsIr)
        lhsIrMetadata = lhsIr.irMetadata
        rhsIrMetadata = rhsIr.irMetadata
        self.ttb_counter = -1


        self.irMetadata = copy_metadata(lhsIr.irMetadata)
        new_type = 'Float' if lhsIrMetadata[-1].type!=rhsIrMetadata[-1].type else lhsIrMetadata[-1].type
        if self.op in ['>', '>=', '==', '!=', '<', '<=']:
            new_type = 'Bool'
        self.irMetadata[-1].type = new_type
        self.irMetadata[-1].isConst = lhsIrMetadata[-1].isConst and rhsIrMetadata[-1].isConst    
        self.update_parent_child([lhsIr, rhsIr])

    def __hash__(self):
        return 0   
    
    def __eq__(self, obj):
        if type(self)==type(obj) and self.op == obj.op:
            if len(self.children) == len(obj.children):
                for i in range(len(self.children)):
                    if self.children[i] != obj.children[i]:
                        return False 
                if checkEqualMetadata(self.irMetadata, obj.irMetadata):
                    return True 
                else:
                    return False
        return False

class IrBlockBinaryOp(IrExpression):
    def __init__(self, lhsIr, rhsIr, op):
        super().__init__()
        self.op = op  
        self.update_parent_child([lhsIr, rhsIr])

    def __hash__(self):
        return 0   
    
    def __eq__(self, obj):
        if type(self)==type(obj) and self.op == obj.op:
            if len(self.children) == len(obj.children):
                for i in range(len(self.children)):
                    if self.children[i] != obj.children[i]:
                        return False 
                if checkEqualMetadata(self.irMetadata, obj.irMetadata):
                    return True 
                else:
                    return False
        return False

class IrBlockWhereBlock(IrExpression):
    def __init__(self, condIr, lhsIr, rhsIr):
        super().__init__()
        self.update_parent_child([condIr, lhsIr, rhsIr])

class IrBlockUnaryOp(IrExpression):
    def __init__(self, inputIr, op):
        super().__init__()
        self.op = op
        self.update_parent_child([inputIr])

class IrPolyExpMat(IrExpression):
    def __init__(self, inputIr):
        super().__init__()
        self.irMetadata = copy_metadata(inputIr.irMetadata)
        self.update_parent_child([inputIr])

class IrPolyExpNotStopFloat(IrExpression):
    def __init__(self, inputIr):
        super().__init__()
        self.irMetadata = copy_metadata(inputIr.irMetadata)
        self.irMetadata[-1].type = 'Float'
        self.update_parent_child([inputIr])

class IrBlockPolyexpStop(IrExpression):
    def __init__(self, travIr, blockMatIr):
        super().__init__()
        self.irMetadata = copy_metadata(travIr.irMetadata)
        self.update_parent_child([travIr, blockMatIr])

class IrBlockPolyexpNotStop(IrExpression):
    def __init__(self, travIr, blockMatIr, constIr):
        super().__init__()
        self.irMetadata = copy_metadata(travIr.irMetadata)
        self.update_parent_child([travIr, blockMatIr, constIr])

class IrBlockAny(IrExpression):
    def __init__(self, inputIr):
        super().__init__()
        # Is all always results in a bool, this is justified
        self.irMetadata = [IrMetadataElement([1], 'Bool', [1], True)]
        self.update_parent_child([inputIr])

class IrBlockAll(IrExpression):
    def __init__(self, inputIr):
        super().__init__()
        # Is all always results in a bool, this is justified
        self.irMetadata = [IrMetadataElement([1], 'Bool', [1], True)]
        self.update_parent_child([inputIr])

class IrBlockGetDims(IrExpression):
    def __init__(self, inputIr):
        super().__init__()
        self.irMetadata = [IrMetadataElement([1], 'Int', [1], True)]
        self.update_parent_child([inputIr])

class IrUnaryOp(IrExpression):
    def __init__(self, inputIr, op):
        super().__init__()
        self.op = op
        if self.op == 'get_dims':
            self.irMetadata = [IrMetadataElement([1], 'Int', [1], True)]
        elif self.op in ('any', 'all'):
            self.irMetadata = [IrMetadataElement([1], 'Bool', [1], True)]
        else:
            self.irMetadata = copy_metadata(inputIr.irMetadata)
        self.update_parent_child([inputIr])

    def __hash__(self):
        return 0   
    
    def __eq__(self, obj):
        if type(self)==type(obj) and self.op == obj.op:
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

class IrGetSubBlockCustomRange(IrExpression):
    def __init__(self, inputIr, start_index, end_index, block_id, tensor):
        super().__init__()
        self.start_index = start_index
        self.end_index = end_index
        self.block_id = block_id
        self.tensor = tensor
        self.update_parent_child([inputIr])


class IrMult(IrExpression):
    def __init__(self, lhsIr, rhsIr, op):
        super().__init__()
        self.op = op 
        lhsIr, rhsIr = matchDims(lhsIr, rhsIr)
        lhsIrMetadata = lhsIr.irMetadata
        rhsIrMetadata = rhsIr.irMetadata
        self.ttb_counter = -1

        self.irMetadata = copy_metadata(lhsIr.irMetadata)
        new_type = 'Float' if lhsIrMetadata[-1].type!=rhsIrMetadata[-1].type else lhsIrMetadata[-1].type
        self.irMetadata[-1].type = new_type
        self.update_parent_child([lhsIr, rhsIr])

    def __hash__(self):
        return 0   
    
    def __eq__(self, obj):
        if type(self)==type(obj) and self.op == obj.op:
            if len(self.children) == len(obj.children):
                for i in range(len(self.children)):
                    if self.children[i] != obj.children[i]:
                        return False 
                if checkEqualMetadata(self.irMetadata, obj.irMetadata):
                    return True 
                else:
                    return False
        return False


class IrSimpleUnary(IrExpression):
    def __init__(self, inputIr, op):
        super().__init__()
        self.op = op
        self.update_parent_child([inputIr])

class IrLambda(IrExpression):
    def __init__(self, op):
        super().__init__()
        self.op = op
        self.update_parent_child([])
        
class IrSimpleBinary(IrExpression):
    def __init__(self, lhsIr, rhsIr, op):
        super().__init__()
        self.op = op 
        self.update_parent_child([lhsIr, rhsIr])

    def __hash__(self):
        return 0   
    
    def __eq__(self, obj):
        if type(self)==type(obj) and self.op == obj.op:
            if len(self.children) == len(obj.children):
                for i in range(len(self.children)):
                    if self.children[i] != obj.children[i]:
                        return False 
                if checkEqualMetadata(self.irMetadata, obj.irMetadata):
                    return True 
                else:
                    return False
        return False
    
class IrTensorOnes(IrExpression):
    def __init__(self, total_size):
        super().__init__()
        self.total_size = total_size

class IrTensorRepeat(IrExpression):
    def __init__(self, inputIr, repeat_dims):
        super().__init__()
        self.repeat_dims = repeat_dims
        self.update_parent_child([inputIr])

    def __hash__(self):
        return 0   
    
    def __eq__(self, obj):
        if type(self)==type(obj) and self.repeat_dims == obj.repeat_dims:
            if len(self.children) == len(obj.children):
                for i in range(len(self.children)):
                    if self.children[i] != obj.children[i]:
                        return False 
                if checkEqualMetadata(self.irMetadata, obj.irMetadata):
                    return True 
                else:
                    return False
        return False

class IrTensorClamp(IrExpression):
    def __init__(self, inputIr, const, min_true):
        super().__init__()
        self.const = const
        self.min_true = min_true
        self.update_parent_child([inputIr])

    def __hash__(self):
        return 0   
    
    def __eq__(self, obj):
        if type(self)==type(obj) and self.min_true == obj.min_true and self.const == obj.const:
            if len(self.children) == len(obj.children):
                for i in range(len(self.children)):
                    if self.children[i] != obj.children[i]:
                        return False 
                if checkEqualMetadata(self.irMetadata, obj.irMetadata):
                    return True 
                else:
                    return False
        return False
    
class IrBlockClamp(IrExpression):
    def __init__(self, inputIr, const, min_true):
        super().__init__()
        self.const = const
        self.min_true = min_true
        self.update_parent_child([inputIr])

    def __hash__(self):
        return 0   
    
    def __eq__(self, obj):
        if type(self)==type(obj) and self.min_true == obj.min_true and self.const == obj.const:
            if len(self.children) == len(obj.children):
                for i in range(len(self.children)):
                    if self.children[i] != obj.children[i]:
                        return False 
                if checkEqualMetadata(self.irMetadata, obj.irMetadata):
                    return True 
                else:
                    return False
        return False

class IrBlockSqueeze(IrExpression):
    def __init__(self, inputIr, index):
        super().__init__()
        self.index = index
        self.update_parent_child([inputIr])

    def __hash__(self):
        return 0

    def __eq__(self, obj):
        if type(self)==type(obj) and self.index == obj.index:
            if len(self.children) == len(obj.children):
                for i in range(len(self.children)):
                    if self.children[i] != obj.children[i]:
                        return False
                if checkEqualMetadata(self.irMetadata, obj.irMetadata):
                    return True
                else:
                    return False
        return False

class IrInnerProduct(IrExpression):
    def __init__(self, lhsIr, rhsIr):
        super().__init__()
        self.irMetadata = copy_metadata(lhsIr.irMetadata)
        self.irMetadata[-1].shape[-1] = rhsIr.irMetadata[-1].shape[-1]
        self.irMetadata[-1].broadcast[-1] = rhsIr.irMetadata[-1].broadcast[-1]
        self.update_parent_child([lhsIr, rhsIr])

class IrBlockInnerProduct(IrExpression):
    def __init__(self, lhsIr, rhsIr, type='equal_dims'):
        super().__init__()
        self.type = type
        self.update_parent_child([lhsIr, rhsIr])

    def __hash__(self):
        return 0   
    
    def __eq__(self, obj):
        if type(self)==type(obj) and self.type == obj.type:
            if len(self.children) == len(obj.children):
                for i in range(len(self.children)):
                    if self.children[i] != obj.children[i]:
                        return False 
                if checkEqualMetadata(self.irMetadata, obj.irMetadata):
                    return True 
                else:
                    return False
        return False


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

class IrClamp(IrExpression):
    def __init__(self, inputIr, const, min_true=True):
        super().__init__()
        self.min_true = min_true
        self.const = const
        self.irMetadata = inputIr.irMetadata.copy()
        self.update_parent_child([inputIr, const])

    def __eq__(self, obj):
        if isinstance(obj, IrClamp):
            if self.min_true == obj.min_true and self.const == obj.const:
                return self.children[0] == obj.children[0] 
        return False
    
    def __hash__(self):
        return 0 


class IrTranspose(IrExpression):
    def __init__(self, inputIr):
        super().__init__()
        assert(len(inputIr.irMetadata)==1)
        shape = inputIr.irMetadata[-1].shape
        broadcast = inputIr.irMetadata[-1].broadcast
        self.update_parent_child([inputIr])
        assert(len(shape) <= 3)
        self.irMetadata = inputIr.irMetadata.copy()

        if len(shape)==3:
            assert(shape[0] == 1)
            self.irMetadata[-1].shape = [shape[0], shape[-1], shape[1]]
            self.irMetadata[-1].broadcast = [broadcast[0], broadcast[-1], broadcast[1]]
        elif len(shape)==2:
            # assert(shape[0] == 1)
            if isinstance(shape[0], int) and shape[0] == 1:
                return
            if isinstance(shape[0], IrConst) and shape[0].const == 1:
                return
            self.irMetadata[-1].shape = [shape[-1], shape[0]]
            self.irMetadata[-1].broadcast = [broadcast[-1], broadcast[0]]


class IrDot(IrExpression):
    def __init__(self, lhsIr, rhsIr):
        super().__init__()


        # self.irMetadata[-1].shape = [mult_metadata(lhsIr.irMetadata[-1].shape[0], lhsIr.irMetadata[-1].broadcast[0])]
        # self.irMetadata[-1].broadcast = [1]
        if lhsIr.irMetadata[-1].type == 'Neuron' or lhsIr.irMetadata[-1].type == 'PolyExp':
            _, _, self.irMetadata, _ = createLcm(lhsIr.irMetadata, rhsIr.irMetadata)
            self.irMetadata[-1].shape = self.irMetadata[-1].shape[:-1]
            self.irMetadata[-1].broadcast = self.irMetadata[-1].broadcast[:-1]

            assert(check_eq(mult_metadata(lhsIr.irMetadata[-1].shape[0], lhsIr.irMetadata[-1].broadcast[0]), mult_metadata(rhsIr.irMetadata[-1].shape[0], rhsIr.irMetadata[-1].broadcast[0])))
            assert(check_eq(mult_metadata(lhsIr.irMetadata[-1].shape[1], lhsIr.irMetadata[-1].broadcast[1]), mult_metadata(rhsIr.irMetadata[-1].shape[1], rhsIr.irMetadata[-1].broadcast[1])))

            self.irMetadata[-1].type = 'PolyExp'
        # elif lhsIr.irMetadata[-1].type == 'Noise' or lhsIr.irMetadata[-1].type == 'SymExp' or lhsIr.irMetadata[-1].type == 'SymExp':
        #     assert(check_eq(mult_metadata(lhsIr.irMetadata[-1].shape[0], lhsIr.irMetadata[-1].broadcast[0]), mult_metadata(rhsIr.irMetadata[-1].shape[0], rhsIr.irMetadata[-1].broadcast[0])))
        #     assert(check_eq(mult_metadata(lhsIr.irMetadata[-1].shape[1], lhsIr.irMetadata[-1].broadcast[1]), mult_metadata(rhsIr.irMetadata[-1].shape[1], rhsIr.irMetadata[-1].broadcast[1])))
        #     self.irMetadata[-1].type = 'SymExp'
        elif lhsIr.irMetadata[-1].type == rhsIr.irMetadata[-1].type and lhsIr.irMetadata[-1].type == 'Float':
            # lhsShape = []
            # lhsBroadcast = []
            # rhsShape = []
            # rhsBroadcast = []
            # for i in range(len(lhsIr.irMetadata)):
            #     lhsShape += lhsIr.irMetadata[i].shape
            #     lhsBroadcast += lhsIr.irMetadata[i].broadcast
            # for i in range(len(rhsIr.irMetadata)):
            #     rhsShape += rhsIr.irMetadata[i].shape
            #     rhsBroadcast += rhsIr.irMetadata[i].broadcast
            lhsShape, lhsBroadcast = get_complete_shape_broadcast(lhsIr.irMetadata)
            rhsShape, rhsBroadcast = get_complete_shape_broadcast(rhsIr.irMetadata)
            
            # assert(len(lhsIr.irMetadata)==1)



            # if(len(lhsShape) == len(rhsShape)):
            #     print(lhsIr.irMetadata[-1])
            #     print(rhsIr.irMetadata[-1])
            #     assert(check_eq(lhsShape[-2],rhsShape[-2]))
            #     # assert(check_eq(lhsShape[0:-3],rhsShape[0:-3]))
            #     repeat_lhs = []
            #     repeat_rhs = []
            #     for i in range(len(lhsShape)-3):
            #         if(not check_eq(lhsShape[i],rhsShape[i])):
            #             if(lhsShape[i] == 1):
            #                 repeat_lhs.append(lhsBroadcast[i])
            #             else:
            #                 repeat_lhs.append(1)
            #             if(rhsShape[i] == 1):
            #                 repeat_rhs.append(rhsBroadcast[i])
            #             else:
            #                 repeat_rhs.append(1)
            #         else:
            #             repeat_lhs.append(1)
            #             repeat_rhs.append(1)
            #     repeat_lhs.extend([1,1,1])
            #     repeat_rhs.extend([1,1,1])
            #     lhsIr = IrRepeat(lhsIr,repeat_dims=repeat_lhs)
            #     rhsIr = IrRepeat(rhsIr,repeat_dims=repeat_rhs)
            #     if(lhsShape[-1] == 1 and rhsShape[-3] == 1):
            #         lhsIr = IrRemoveDimension(lhsIr, len(lhsShape)-1)
            #         rhsIr = IrRemoveDimension(rhsIr, len(rhsShape)-3)
            #         self.irMetadata[-1].shape = lhsShape[0:-3] + [lhsShape[-3],rhsShape[-1]]
            #         self.irMetadata[-1].broadcast = lhsBroadcast[0:-3] + [1,1] 
            #     elif(lhsShape[-3] == 1 and rhsShape[-1] == 1):
            #         lhsIr = IrRemoveDimension(lhsIr, len(lhsShape)-3)
            #         rhsIr = IrRemoveDimension(rhsIr, len(rhsShape)-1)
            #         self.irMetadata[-1].shape = lhsShape[0:-3] + [rhsShape[-3],lhsShape[-1]]
            #         self.irMetadata[-1].broadcast = lhsBroadcast[0:-3] + [1,1] 
            #     else:
            #         raise NotImplementedError
            # else:
            #     raise NotImplementedError

            # assert(len(lhsShape) <= 3)
            # assert(len(rhsShape) <= 2)
            # assert(sum(lhsBroadcast) == len(lhsBroadcast))
            # assert(sum(rhsBroadcast) == len(rhsBroadcast))
            self.irMetadata = copy_metadata(lhsIr.irMetadata)

            if len(lhsShape)==3 and len(rhsShape)==2:
                assert(check_eq(lhsShape[-1], rhsShape[-1]))
                if(not check_eq(lhsShape[0], rhsShape[0]) and lhsShape[0] == 1):
                    lhsIr = IrRepeat(lhsIr, repeat_dims=[lhsBroadcast[0],1,1])
                if(not check_eq(lhsShape[0], rhsShape[0]) and rhsShape[0] == 1):
                    rhsIr = IrRepeat(rhsIr, repeat_dims=[rhsBroadcast[0],1])
                lhsShape, lhsBroadcast = get_complete_shape_broadcast(lhsIr.irMetadata)
                rhsShape, rhsBroadcast = get_complete_shape_broadcast(rhsIr.irMetadata)
                self.irMetadata[-1].shape = [lhsShape[0], lhsShape[-2]]
                self.irMetadata[-1].broadcast = [1, 1]
            elif len(lhsShape)==3 and len(rhsShape)==3:
                assert(check_eq(lhsShape[-1], rhsShape[-2]))
                if(not check_eq(lhsShape[0], rhsShape[0]) and lhsShape[0] == 1):
                    lhsIr = IrRepeat(lhsIr, repeat_dims=[lhsBroadcast[0],1,1])
                if(not check_eq(lhsShape[0], rhsShape[0]) and rhsShape[0] == 1):
                    rhsIr = IrRepeat(rhsIr, repeat_dims=[rhsBroadcast[0],1,1])
                lhsShape, lhsBroadcast = get_complete_shape_broadcast(lhsIr.irMetadata)
                rhsShape, rhsBroadcast = get_complete_shape_broadcast(rhsIr.irMetadata)
                self.irMetadata[-1].shape = [lhsShape[0], lhsShape[-2], rhsShape[-1]]
                self.irMetadata[-1].broadcast = [1,1,1]

            self.irMetadata[-1].type = 'Float'
        else:
            assert False
        self.update_parent_child([lhsIr, rhsIr])


class IrCombineToPoly(IrExpression):
    def __init__(self, coeffIr, constIr):
        super().__init__()
        if not check_eq(coeffIr.irMetadata[-1].shape[:-1], constIr.irMetadata[-1].shape):
            print(len(coeffIr.irMetadata[-1].shape)-1)
            print(len(constIr.irMetadata[-1].shape))
            for i in range(len(coeffIr.irMetadata[0].shape)-1):
                print(coeffIr.irMetadata[-1].shape[i])
                print(constIr.irMetadata[-1].shape[i])
                print('done\n')
            print(constIr.irMetadata[-1].shape[-1])
            print('done\n')
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



class IrCombineToSym(IrExpression):
    def __init__(self, coeffIr, constIr):
        super().__init__()
        if not check_eq(coeffIr.irMetadata[-1].shape[:-1], constIr.irMetadata[-1].shape):
            print(coeffIr.irMetadata[-1].shape[:-1], constIr.irMetadata[-1].shape)
        assert(check_eq(coeffIr.irMetadata[-1].shape[:-1], constIr.irMetadata[-1].shape))
        assert(check_eq(coeffIr.irMetadata[-1].broadcast[:-1], constIr.irMetadata[-1].broadcast))
        assert(coeffIr.irMetadata[-1].isConst == constIr.irMetadata[-1].isConst)

        # self.irMetadata = copy.deepcopy(constIr.irMetadata)
        self.irMetadata = copy_metadata(constIr.irMetadata)
        self.irMetadata[-1].type = 'SymExp'
        new_children = [coeffIr, constIr, constIr.irMetadata[-1].shape[0]]
        # new_children = [coeffIr, constIr, convert_z3_to_ir(constIr.irMetadata[-1].shape[0])]
        self.update_parent_child(new_children)



class IrObjectLookup(IrExpression):
    def __init__(self, inputIr, object_name):
        super().__init__()
        self.object_name = object_name
        self.update_parent_child([inputIr])

class IrBlockCreateSimilar(IrExpression):
    def __init__(self, inputIr, argIr):
        super().__init__()
        self.update_parent_child([inputIr, argIr])

class IrSetBlockTotalShapeLastDim(IrStatement):
    def __init__(self, inputVarIr, valueIr):
        super().__init__()
        self.update_parent_child([inputVarIr, valueIr])

class IrAssignToBlock(IrStatement):
    def __init__(self, blockVarIr, valueIr):
        super().__init__()
        self.update_parent_child([blockVarIr, valueIr])

class IrBlockGetSubBlockCustomRange(IrExpression):
    def __init__(self, inputIr, start_index, end_index, block_start_index):
        super().__init__()
        self.start_index = start_index
        self.end_index = end_index
        self.block_start_index = block_start_index
        self.update_parent_child([inputIr])

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
        size = 0
        for i in range(len(inputIr.irMetadata)-1):
            size += len(inputIr.irMetadata[i].shape)
        self.reduce_dim = size
        self.update_parent_child([inputIr])

class IrMapCoeff(IrExpression):
    def __init__(self, inputIr, reduce=True):
        super().__init__()
        assert(inputIr.irMetadata[-1].type == 'PolyExp' or inputIr.irMetadata[-1].type == 'SymExp')
        
        # self.irMetadata = copy.deepcopy(inputIr.irMetadata)
        self.irMetadata = copy_metadata(inputIr.irMetadata)
        if inputIr.irMetadata[-1].type == 'PolyExp':
            irMetadataElement = IrMetadataElement([IrAst.poly_size], 'Float', [1], False)
        else:
            irMetadataElement = IrMetadataElement([IrAst.sym_size], 'Float', [1], False)
        if reduce:
            self.irMetadata.append(irMetadataElement)
        else:
            self.irMetadata[-1].shape += irMetadataElement.shape
            self.irMetadata[-1].broadcast += irMetadataElement.broadcast
            self.irMetadata[-1].type = 'Float'
        self.update_parent_child([inputIr])

class IrMapNeuron(IrExpression):
    def __init__(self, inputIr, reduce=True, dims=False, dims_val=None):
        super().__init__()
        self.dims = dims
        assert(inputIr.irMetadata[-1].type == 'PolyExp')
        
        # self.irMetadata = copy.deepcopy(inputIr.irMetadata)
        self.irMetadata = copy_metadata(inputIr.irMetadata)
        
        for i in range(len(self.irMetadata)):
            for j in range(len(self.irMetadata[i].shape)):
                self.irMetadata[i].broadcast[j] = mult_metadata(self.irMetadata[i].broadcast[j], self.irMetadata[i].shape[j])
                # self.irMetadata[i].broadcast[j] *= self.irMetadata[i].shape[j]
                self.irMetadata[i].shape[j] = 1

        irMetadataElement = IrMetadataElement([IrAst.poly_size], 'Neuron', [1], False)
        if reduce:
            self.irMetadata.append(irMetadataElement)
        else:
            self.irMetadata[-1].shape += irMetadataElement.shape
            self.irMetadata[-1].broadcast += irMetadataElement.broadcast
            self.irMetadata[-1].type = 'Neuron'
        # self.irMetadata.append(irMetadataElement)
        if self.dims:
            self.update_parent_child([dims_val])
        else:
            self.update_parent_child([inputIr])

class IrMapNoise(IrExpression):
    def __init__(self, inputIr):
        super().__init__()
        assert(inputIr.irMetadata[-1].type == 'SymExp')
        
        # self.irMetadata = copy.deepcopy(inputIr.irMetadata)
        self.irMetadata = copy_metadata(inputIr.irMetadata)
        
        for i in range(len(self.irMetadata)):
            for j in range(len(self.irMetadata[i].shape)):
                self.irMetadata[i].broadcast[j] = mult_metadata(self.irMetadata[i].broadcast[j], self.irMetadata[i].shape[j])
                # self.irMetadata[i].broadcast[j] *= self.irMetadata[i].shape[j]
                self.irMetadata[i].shape[j] = 1

        irMetadataElement = IrMetadataElement([IrAst.sym_size], 'Noise', [1], False)
        self.irMetadata.append(irMetadataElement)
        self.update_parent_child([inputIr])

    

class IrAssignment(IrStatement):
    def __init__(self, varIr, inputIr):
        super().__init__()
        self.irMetadata = inputIr.irMetadata
        self.update_parent_child([varIr, inputIr])
        self.uses = []

class IrTransRetBasic(IrStatement):
    def __init__(self, exprIrs):
        super().__init__()
        self.exprIrs = exprIrs
        self.update_parent_child(exprIrs)

class IrTransRetIf(IrStatement):
    def __init__(self, condIr, lhsIr, rhsIr):
        super().__init__()
        self.condIr = condIr
        self.lhsIr = lhsIr
        self.rhsIr = rhsIr
        self.update_parent_child([condIr, lhsIr, rhsIr])


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

# For the del statements
class IrDel(IrStatement):
    def __init__(self, var_names):
        super().__init__()
        self.var_names = var_names

class IrBlock(IrAst):
    def __init__(self, ir_list = [], jump = None, inner_jump = None, loopBack = None):
        super().__init__()
        self.inner_jump = inner_jump
        self.jump = jump
        if len(ir_list) == 0:
            ir_list = list()
        self.update_parent_child(ir_list)
        self.inside_while = False 
        self.while_number = -1

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



# The parameter names for the generated method's signature (and the matching
# call-site args). Every DSL-declared op (Affine, Relu, Sigmoid, ...) uses this;
# ops synthesized outside the DSL (Add, Concat -- see builtin_ops.py) carry
# their own params on IrOpStmt because they need more than one `prev`.
DEFAULT_OP_PARAMS = ['abs_elem', 'prev', 'curr', 'poly_size', 'curr_size',
                     'prev_size', 'input_size', 'batch_size']


class IrOpStmt(IrAst):
    def __init__(self, op, cfg, layerwise_cfgs = None, params = None):
        super().__init__()
        self.op = op
        self.cfg = cfg
        self.layerwise_cfgs = layerwise_cfgs
        self.params = params if params is not None else DEFAULT_OP_PARAMS
        # In jit builds the specialized methods also thread the d_<key> state.
        self.layerwise_params = None

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
        # tstore: map { certifier name -> list of abstract transformers of operators of this CF program }
        # tstore type: dict[str -> list[IrOpStmt with elem: op, cfg]]
        self.tstore = tstore
        self.fstore = fstore
        self.irNodes = irNodes
        

