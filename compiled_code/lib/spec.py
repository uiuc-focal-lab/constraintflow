from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
from compiled_code.lib.polyexp import PolyExp, SymExp
from compiled_code.lib.parse import *


def create_l(image, eps = 0.02, shapes = []):
    image = image.flatten()
    l = image - eps 
    s = 0
    for i in range(1, len(shapes)):
        s += compute_size(shapes[i])
    l_temp = torch.full((s,), float('-inf'))
    l = torch.cat((l, l_temp))
    return l

def create_u(image, eps = 0.02, shapes = []):
    image = image.flatten()
    u = image + eps 
    s = 0
    for i in range(1, len(shapes)):
        s += compute_size(shapes[i])
    u_temp = torch.full((s,), float('inf'))
    u = torch.cat((u, u_temp))
    return u

def create_L(image, eps, shapes = []):
    const = create_l(image, eps, shapes)
    N = const.size(0)
    coeff = torch.zeros((N, N))
    return PolyExp(N, N, coeff, const)

def create_U(image, eps, shapes = []):
    const = create_u(image, eps, shapes)
    N = const.size(0)
    coeff = torch.zeros((N, N))
    return PolyExp(N, N, coeff, const)

def create_Z(image, eps, shapes = []):
    l_const = create_l(image, eps, shapes)
    u_const = create_u(image, eps, shapes)
    const = (l_const + u_const) / 2
    coeff = torch.eye(l_const.shape[0]) * (u_const - l_const)/2
    coeff = coeff[:, :784]
    return SymExp(l_const.shape[0], l_const.shape[0], coeff, const, 0, l_const.shape[0]-1)


def get_input_spec(data_name = './data', n = 0, eps = 0.02, train=True, transformer='ibp', shapes = []):
    transform = transforms.ToTensor()
    data = datasets.MNIST(root=data_name, train=train, download=False, transform=transform)

    image, _ = data[n]
    if list(image.shape) != shapes[0]:
        image = image.flatten()

    l = create_l(image, eps, shapes=shapes)
    u = create_u(image, eps, shapes=shapes)

    if transformer == 'deeppoly':
        L = create_L(image, eps, shapes)
        U = create_U(image, eps, shapes)
        return (l, u, L, U)

    elif transformer == 'ibp':
        return (l, u)
    
    elif transformer == 'deepz':
        Z = create_Z(image, eps, shapes)
        return (l, u, Z)