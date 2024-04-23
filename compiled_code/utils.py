import torch 

def checkTypes(x, y):
    if type(x) != type(y):
        print(type(x), type(y))
        raise Exception('TYPE MISMATCH')

def checkShapes(x, y):
    if x.shape != y.shape:
        print(x.shape, y.shape)
        raise Exception('SHAPE MISMATCH')

def sanityCheck(x, y):
    checkTypes(x, y)
    checkShapes(x, y)


def plus(x, y):
    sanityCheck(x, y)
    return x+y

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

def mult(x, y):
    sanityCheck(x, y)
    return x*y

def divide(x, y):
    sanityCheck(x, y)
    return x/y

def convert_to_tensor(x, shape):
    if not isinstance(x, torch.Tensor):
        return torch.ones(shape)*x 
    return x

def convert_to_float(x):
    return x.float()

def phi(l):
    pass

print('!!!!!!!!!!!!!!!!!!!!!!')