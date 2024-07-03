import torch 

input_size = 5

def checkTypes(x, y):
    if type(x) != type(y):
        print(type(x), type(y))
        raise Exception('TYPE MISMATCH')

def checkShapes(x, y):
    if isinstance(x, torch.Tensor) and isinstance(y, torch.Tensor):
        if x.shape != y.shape:
            print(x.shape, y.shape)
            raise Exception('SHAPE MISMATCH')

def sanityCheck(x, y):
    checkTypes(x, y)
    checkShapes(x, y)


def boolNeg(x):
    if isinstance(x, torch.Tensor):
        return ~x 
    return not(x)

def neg(x):
    return -x

def any(x):
    if type(x)!=torch.Tensor:
        raise Exception('TYPE MISMATCH')
    return x.any()

def all(x):
    if type(x)!=torch.Tensor:
        raise Exception('TYPE MISMATCH')
    return x.all()

def plus(x, y):
    sanityCheck(x, y)
    return x+y

def cf_max(x, y):
    sanityCheck(x, y)
    return torch.max(x, y)

def cf_min(x, y):
    sanityCheck(x, y)
    return torch.min(x, y)

def minus(x, y):
    sanityCheck(x, y)
    return x-y

def ge(x, y):
    sanityCheck(x, y)
    return x>=y

def gt(x, y):
    sanityCheck(x, y)
    return x>y

def le(x, y):
    sanityCheck(x, y)
    return x<=y

def lt(x, y):
    sanityCheck(x, y)
    return x<y

def eq(x, y):
    sanityCheck(x, y)
    return x==y

def ne(x, y):
    sanityCheck(x, y)
    return x!=y

def mult(x, y):
    sanityCheck(x, y)
    return x*y

def inner_prod(x, y):
    checkTypes(x, y)
    if x.shape[1] != y.shape[0]:
        print(x.shape, y.shape)
        raise Exception('SHAPE MISMATCH')
    return x@y

def divide(x, y):
    sanityCheck(x, y)
    return x/y

def conj(x, y):
    sanityCheck(x, y)
    if isinstance(x, bool):
        return x and y 
    return x & y

def disj(x, y):
    sanityCheck(x, y)
    if isinstance(x, bool):
        return x or y 
    return x | y

# def convert_to_tensor(x, shape):
#     if not isinstance(x, torch.Tensor):
#         return torch.ones(shape)*x 
#     return x

def convert_to_float(x):
    return x.float()

def get_default_stop(shape):
    # print(shape)
    global input_size
    vertices_stop_default = torch.zeros(shape)
    vertices_stop_default[:, 0:input_size] = 1
    vertices_stop_default = vertices_stop_default.bool()
    # print(vertices_stop_default)
    # kjdfs
    return vertices_stop_default

def get_default_stop2(shape):
    global input_size
    vertices_stop_default = torch.zeros(shape)
    vertices_stop_default[:, 0:834] = 1
    vertices_stop_default = vertices_stop_default.bool()
    return vertices_stop_default

def get_shape_1(x):
    if not isinstance(x, torch.Tensor):
        raise Exception('TYPE MISMATCH')
    return x.shape[1]

def get_shape_0(x):
    if not isinstance(x, torch.Tensor):
        raise Exception('TYPE MISMATCH')
    return x.shape[0]

def phi(l):
    pass
