from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
from lib.polyexp import PolyExp, SymExp
from specs.util import *

def product(iterable):
    result = 1
    for value in iterable:
        result *= value
    return result

def reshape_list(original_list, shape):
    size = len(original_list)

    # Check if the product of the dimensions matches the size
    if size != product(shape):
        raise ValueError("Product of dimensions must equal the size of the original list")

    def recursive_reshape_list(start, shape):
        if len(shape) == 1:
            return original_list[start : start + shape[0]]
        else:
            result = []
            current_dimension_size = shape[0]
            current_size = product(shape[1:])

            for _ in range(current_dimension_size):
                sublist = recursive_reshape_list(start, shape[1:])
                result.append(sublist)
                start += current_size

            return result

    return recursive_reshape_list(0, shape)

def reshape(L, shapes):
    return [reshape_list(L[i], shapes[i]) for i in range(len(L))]

def compute_size(shape):
    s = 1
    while len(shape)>0:
        s *= shape[0]
        shape = shape[1:]
    return s

def create_l(image, eps = 0.02, shapes = []):
    image = image.flatten()
    l = image - eps 
    s = 0
    for i in range(1, len(shapes)):
        s += compute_size(shapes[i])
    l_temp = torch.full((s,), float('-inf'))
    l = torch.cat((l, l_temp))
    return l.reshape(-1,1)
    l = torch.ones((N, N, 1))
    l = [(image - eps).unsqueeze(1)]
    for i in range(1, len(shapes)):
        l.append(torch.full(tuple(shapes[i]), float('-inf')))
    return l 

def create_u(image, eps = 0.02, shapes = []):
    image = image.flatten()
    u = image + eps 
    s = 0
    for i in range(1, len(shapes)):
        s += compute_size(shapes[i])
    u_temp = torch.full((s,), float('inf'))
    u = torch.cat((u, u_temp))
    return u.reshape(-1,1) 
    u = [(image + eps).unsqueeze(1)]
    for i in range(1, len(shapes)):
        u.append(torch.full(tuple(shapes[i]), float('inf')))
    return u 

def create_L(image, eps, shapes = []):
    const = create_l(image, eps, shapes)
    N = const.size(0)
    coeff = torch.zeros((N, N))
    return PolyExpNew(N, coeff, const)

    image = image.flatten()
    L_temp = []
    size = product(shapes[0])
    for i in range(size):
        temp = PolyExp(shapes, const = image[i] - eps)
        L_temp.append(temp.copy())
    L = [L_temp]
    for i in range(1, len(shapes)):
        size = product(shapes[i])
        L.append([PolyExp(shapes, const = float('-inf'))]*size)
    L = reshape(L, shapes)
    return L 

def create_U(image, eps, shapes = []):
    const = create_u(image, eps, shapes)
    N = const.size(0)
    coeff = torch.zeros((N, N))
    return PolyExpNew(N, coeff, const)
    image = image.flatten()
    U_temp = []
    size = product(shapes[0])
    for i in range(size):
        temp = PolyExp(shapes, const = image[i] + eps)
        U_temp.append(temp.copy())
    U = [U_temp]
    for i in range(1, len(shapes)):
        size = product(shapes[i])
        U.append([PolyExp(shapes, const = float('inf'))]*size)
    U = reshape(U, shapes)
    return U 

def create_Z(image, eps, shapes = []):
    l_const = create_l(image, eps, shapes)
    u_const = create_u(image, eps, shapes)
    const = (l_const + u_const) / 2
    coeff = torch.diag((u_const - l_const) / 2)
    return SymExp(l_const.shape[0], l_const.shape[0], coeff, const, 0, l_const.shape[0])
    image = image.flatten()
    Z_temp = []
    size = product(shapes[0])
    for i in range(size):
        Z_temp.append(SymExp(mat = []))
        Z_temp[-1].new_symbol()
        Z_temp[-1].populate(coeff=eps, const=image[i])
    Z = [Z_temp]
    for i in range(1, len(shapes)):
        size = product(shapes[i])
        tmp = []
        for i in range(size):
            temp_sym = SymExp()
            temp_sym.populate()
            tmp.append(temp_sym)
        Z.append(tmp)
    Z = reshape(Z, shapes)
    return Z 

def create_live(image, eps, shapes = []):
    image = image.flatten()
    n = image.size(0)
    const = create_u(image, eps, shapes)
    N = const.size(0)
    t = torch.zeros(N)
    t[0:n] = 1
    return t


def get_input_spec(data_name = './data', n = 0, eps = 0.02, train=True, transformer='ibp', shapes = []):
    transform = transforms.ToTensor()
    data = datasets.MNIST(root=data_name, train=train, download=False, transform=transform)

    image, _ = data[n]
    # image = torch.tensor([1,2])
    # print(list(image.shape))
    if list(image.shape) != shapes[0]:
        image = image.flatten()
        image = image.resize(*shapes[0][1:])

    l = create_l(image, eps, shapes=shapes)
    u = create_u(image, eps, shapes=shapes)
    t = create_live(image, eps, shapes=shapes)

    if transformer == 'deeppoly':
        L = create_L(image, eps, shapes)
        U = create_U(image, eps, shapes)
        return (t, l, u, L, U)

    elif transformer == 'ibp':
        return (t, l, u)
    
    elif transformer == 'deepz':
        Z = create_Z(image, eps, shapes)
        return (t, l, u, Z)