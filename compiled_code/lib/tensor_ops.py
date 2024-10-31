import torch 
import math
from compiled_code.lib.sparse_tensor import *

input_size = 784



types = {bool: torch.bool, int: torch.int, float: torch.float}

def checkTypes(x, y):
    if isinstance(x, SparseTensorBlock):
        if isinstance(y, SparseTensorBlock):
            if x.type != y.type:
                print(x.type, y.type)
                raise Exception('TYPE MISMATCH')
        if isinstance(y, float) or isinstance(y, int) or isinstance(y, bool):
            if type(y) != x.type:
                raise Exception('TYPE MISMATCH')
        if isinstance(y, torch.Tensor):
            if types[x.type] != y.dtype:
                print(x.type, y.dtype)
                raise Exception('TYPE MISMATCH')
    elif isinstance(y, SparseTensorBlock):
        if isinstance(x, float) or isinstance(x, int) or isinstance(x, bool):
            if type(x) != y.type:
                raise Exception('TYPE MISMATCH')
    elif isinstance(x, SparseTensorBlock):
        if isinstance(y, torch.Tensor):
            if x.type != y.dtype:
                raise Exception('TYPE MISMATCH')
    elif isinstance(y, SparseTensorBlock):
        if isinstance(x, torch.Tensor):
            if y.type != x.dtype:
                raise Exception('TYPE MISMATCH')
    elif type(x) != type(y):
        print(type(x), type(y))
        raise Exception('TYPE MISMATCH')

def checkShapes(x, y):
    if isinstance(x, SparseTensorBlock):
        if isinstance(y, float) or isinstance(y, int):
            return
        elif isinstance(y, torch.Tensor):
            if not (x.total_size == torch.tensor(y.shape)).all():
                print(x.total_size, y.shape)
                raise Exception('SHAPE MISMATCH')
        elif isinstance(y, SparseTensorBlock):
            if not (x.total_size == y.total_size).all():
                print(x.total_size, y.total_size)
                raise Exception('SHAPE MISMATCH')
    elif isinstance(y, SparseTensorBlock):
        if isinstance(x, float) or isinstance(x, int):
            return True
        elif isinstance(x, torch.Tensor):
            if not (y.total_size == torch.tensor(x.shape)).all():
                print(x.shape, y.total_size)
                raise Exception('SHAPE MISMATCH')
    
    elif isinstance(x, torch.Tensor) and isinstance(y, torch.Tensor):
        if x.shape != y.shape:
            print(x.shape, y.shape)
            raise Exception('SHAPE MISMATCH')

def sanityCheck(x, y):
    checkTypes(x, y)
    checkShapes(x, y)


def boolNeg(x):
    if isinstance(x, torch.Tensor):
        return ~x 
    if isinstance(x, SparseTensorBlock):
        return x.boolneg()
    y = not(x)
    return y

def neg(x):
    return -x

def any(x):
    if type(x)!=torch.Tensor and type(x)!=SparseTensorBlock:
        raise Exception('TYPE MISMATCH')
    
    return x.any()

def all(x):
    if type(x)!=torch.Tensor:
        raise Exception('TYPE MISMATCH')
    return x.all()

def plus(x, y):
    sanityCheck(x, y)
    if isinstance(x, SparseTensorBlock):
        return x.binary(y, '+')
    if isinstance(y, SparseTensorBlock):
        return y.binary(x, '+')
    return x+y

def cf_max(x, y):
    sanityCheck(x, y)
    return torch.max(x, y)

def cf_min(x, y):
    sanityCheck(x, y)
    return torch.min(x, y)

def minus(x, y):
    sanityCheck(x, y)
    if isinstance(x, SparseTensorBlock):
        return x.binary(y, '-')
    if isinstance(y, SparseTensorBlock):
        return convert_dense_to_sparse(x).binary(y, '-')
    return x-y

def ge(x, y):
    sanityCheck(x, y)
    if isinstance(x, SparseTensorBlock):
        return x.binary(y, '>=')
    if isinstance(y, SparseTensorBlock):
        return convert_dense_to_sparse(x).binary(y, '>=')
    return x>=y

def gt(x, y):
    sanityCheck(x, y)
    if isinstance(x, SparseTensorBlock):
        return x.binary(y, '>')
    if isinstance(y, SparseTensorBlock):
        return convert_dense_to_sparse(x).binary(y, '>')
    return x>y

def le(x, y):
    sanityCheck(x, y)
    if isinstance(x, SparseTensorBlock):
        return x.binary(y, '<=')
    if isinstance(y, SparseTensorBlock):
        return convert_dense_to_sparse(x).binary(y, '<=')
    return x<=y

def lt(x, y):
    sanityCheck(x, y)
    if isinstance(x, SparseTensorBlock):
        return x.binary(y, '<')
    if isinstance(y, SparseTensorBlock):
        return convert_dense_to_sparse(x).binary(y, '<')
    return x<y

def eq(x, y):
    sanityCheck(x, y)
    return x==y

def ne(x, y):
    sanityCheck(x, y)
    if isinstance(x, SparseTensorBlock):
        return x.binary(y, '!=')
    if isinstance(y, SparseTensorBlock):
        return y.binary(x, '!=')
    return x!=y

def mult(x, y):
    sanityCheck(x, y)
    if isinstance(x, SparseTensorBlock):
        return x.binary(y, '*')
    if isinstance(y, SparseTensorBlock):
        return y.binary(x, '*')
    return x*y

def lcm(a, b):
    if isinstance(a, float) or isinstance(a, int) or isinstance(a, bool):
        return b
    if isinstance(b, float) or isinstance(b, int) or isinstance(b, bool):
        return a
    assert(a.shape[0] == b.shape[0])
    total_size = []
    for j in range(len(a)):
        total_size.append(math.lcm(int(a[j].item()), int(b[j].item())))
    return torch.tensor(total_size)

def const_to_sparse(c, total_size):
    return SparseTensorBlock([], [], total_size.shape[0], total_size, type=type(c), dense_const=c)

def where(x, y, z):
    if isinstance(x, torch.Tensor) and isinstance(y, torch.Tensor) and isinstance(z, torch.Tensor):
        checkShapes(x, y)
        sanityCheck(y, z)
        return torch.where(x, y, z)
    if isinstance(x, bool) and isinstance(y, float) and isinstance(z, float):
        return y if x else z
    
    if isinstance(x, SparseTensorBlock):
        x_size = x.total_size
    elif isinstance(x, torch.Tensor):
        x_size = torch.tensor(x.shape)
    else:
        x_size = 0

    if isinstance(y, SparseTensorBlock):
        y_size = y.total_size
    elif isinstance(y, torch.Tensor):
        y_size = torch.tensor(y.shape)
    else:
        y_size = 0

    if isinstance(z, SparseTensorBlock):
        z_size = z.total_size
    elif isinstance(z, torch.Tensor):
        z_size = torch.tensor(z.shape)
    else:
        z_size = 0

    total_size = lcm(x_size, lcm(y_size, z_size))

    if isinstance(x, torch.Tensor):
        x1 = convert_dense_to_sparse(x)
    elif isinstance(x, bool):
        x1 = const_to_sparse(x, total_size)
    else:
        x1 = x

    if isinstance(y, torch.Tensor):
        y1 = convert_dense_to_sparse(y)
    elif isinstance(y, float):
        y1 = const_to_sparse(y, total_size)
    else:
        y1 = y

    if isinstance(z, torch.Tensor):
        z1 = convert_dense_to_sparse(z)
    elif isinstance(z, float):
        z1 = const_to_sparse(z, total_size)
    else:
        z1 = z
    checkShapes(x1, y1)
    sanityCheck(y1, z1)

    return sp_where(x1, y1, z1)

def inner_prod(x, y):
    checkTypes(x, y)
    if isinstance(x, SparseTensorBlock):
        if isinstance(y, SparseTensorBlock):
            if x.total_size.shape[0] == y.total_size.shape[0]:
                if x.total_size[-1] != y.total_size[-2]:
                    print(x.total_size, y.total_size)
                    raise Exception('SHAPE MISMATCH')
                if x.total_size[:-2] != y.total_size[:-2]:
                    print(x.total_size, y.total_size)
                    raise Exception('SHAPE MISMATCH')
            elif x.total_size.shape[0] > y.total_size.shape[0]:
                if x.total_size[-1] != y.total_size[-1]:
                    print(x.total_size, y.total_size)
                    raise Exception('SHAPE MISMATCH')
                if x.total_size[:-2] != y.total_size[:-1]:
                    print(x.total_size, y.total_size)
                    raise Exception('SHAPE MISMATCH')
            else:
                print(x.total_size, y.total_size)
                raise Exception('SHAPE MISMATCH')
            return x.matmul(y)
        else:
            if x.total_size.shape[0] == y.shape.shape[0]:
                if x.total_size[-1] != y.shape[-2]:
                    print(x.total_size, y.shape)
                    raise Exception('SHAPE MISMATCH')
                if x.total_size[:-2] != y.shape[:-2]:
                    print(x.total_size, y.shape)
                    raise Exception('SHAPE MISMATCH')
            elif x.total_size.shape[0] > y.shape.shape[0]:
                if x.total_size[-1] != y.shape[-1]:
                    print(x.total_size, y.shape)
                    raise Exception('SHAPE MISMATCH')
                if x.total_size[:-2] != y.shape[:-1]:
                    print(x.total_size, y.shape)
                    raise Exception('SHAPE MISMATCH')
            else:
                print(x.total_size, y.shape)
                raise Exception('SHAPE MISMATCH')
            return x.matmul(y)
    elif isinstance(y, SparseTensorBlock):
        if x.shape.shape[0] == y.total_size.shape[0]:
            if x.shape[-1] != y.total_size[-2]:
                print(x.shape, y.total_size)
                raise Exception('SHAPE MISMATCH')
            if x.shape[:-2] != y.total_size[:-2]:
                print(x.shape, y.total_size)
                raise Exception('SHAPE MISMATCH')
        elif x.shape.shape[0] > y.total_size.shape[0]:
            if x.shape[-1] != y.total_size[-1]:
                print(x.shape, y.total_size)
                raise Exception('SHAPE MISMATCH')
            if x.shape[:-2] != y.total_size[:-1]:
                print(x.shape, y.total_size)
                raise Exception('SHAPE MISMATCH')
        else:
            print(x.shape, y.total_size)
            raise Exception('SHAPE MISMATCH')
        return convert_dense_to_sparse(x).matmul(y)
    else:
        if x.shape.shape[0] == y.shape.shape[0]:
            if x.shape[-1] != y.shape[-2]:
                print(x.shape, y.shape)
                raise Exception('SHAPE MISMATCH')
            if x.shape[:-2] != y.shape[:-2]:
                print(x.shape, y.shape)
                raise Exception('SHAPE MISMATCH')
        elif x.shape.shape[0] > y.shape.shape[0]:
            if x.shape[-1] != y.shape[-1]:
                print(x.shape, y.shape)
                raise Exception('SHAPE MISMATCH')
            if x.shape[:-2] != y.shape[:-1]:
                print(x.shape, y.shape)
                raise Exception('SHAPE MISMATCH')
        else:
            print(x.shape, y.shape)
            raise Exception('SHAPE MISMATCH')
        return x@y

def divide(x, y):
    sanityCheck(x, y)
    if isinstance(x, SparseTensorBlock):
        return x.binary(y, '/')
    if isinstance(y, SparseTensorBlock):
        return convert_dense_to_sparse(x).binary(y, '/')
    return x/y

def conj(x, y):
    sanityCheck(x, y)
    if isinstance(x, SparseTensorBlock):
        return x.binary(y, '&')
    if isinstance(y, SparseTensorBlock):
        return y.binary(x, '&')
    if isinstance(x, bool):
        return x and y 
    return x & y

def disj(x, y):
    sanityCheck(x, y)
    if isinstance(x, bool):
        return x or y 
    # return x | y
    # sanityCheck(x, y)
    if isinstance(x, SparseTensorBlock):
        return x.binary(y, '|')
    if isinstance(y, SparseTensorBlock):
        return convert_dense_to_sparse(x).binary(y, '|')
    return x | y

# def convert_to_tensor(x, shape):
#     if not isinstance(x, torch.Tensor):
#         return torch.ones(shape)*x 
#     return x

def convert_to_float(x):
    return x.float()

def get_default_stop(shape):
    # print(shape)
    # global input_size
    # vertices_stop_default = torch.zeros(shape)
    # vertices_stop_default[:, :, 0:input_size] = 1
    # vertices_stop_default = vertices_stop_default.bool()
    # print(vertices_stop_default)
    # kjdfs
    return SparseTensorBlock([], [], len(shape), torch.tensor(shape), type=bool, dense_const=False)
    return convert_dense_to_sparse(vertices_stop_default)

def get_default_stop2(shape):
    global input_size
    vertices_stop_default = torch.zeros(shape)
    vertices_stop_default[:, 0:834] = 1
    vertices_stop_default = vertices_stop_default.bool()
    return vertices_stop_default

def get_shape_1(x):
    if isinstance(x, SparseTensorBlock):
        return x.total_size[1]
    if not isinstance(x, torch.Tensor):
        raise Exception('TYPE MISMATCH')
    return x.shape[1]

def get_shape_0(x):
    if (not isinstance(x, torch.Tensor)) or (not isinstance(x, SparseTensorBlock)):
        raise Exception('TYPE MISMATCH')
    if isinstance(x, SparseTensorBlock):
        return x.total_size[0]
    return x.shape[0]


# def convert_to_sparse(mat, dense_const, network_size, batch_size=1):
#     return SparseTensorBlock([torch.tensor([0, 0])], [SparseBlock(mat[:784].reshape(1,-1).repeat(batch_size, 1))], 2, torch.tensor([batch_size, network_size]), dense_const=dense_const)

def repeat(mat, repeat_dims):
    if isinstance(mat, float):
        return mat*torch.ones(*(repeat_dims.tolist()))
    if isinstance(mat, torch.Tensor):
        return mat.repeat(*(repeat_dims.tolist()))
    else:
        return mat.repeat(repeat_dims)