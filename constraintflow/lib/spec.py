from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from constraintflow.lib.polyexp import PolyExpSparse
from constraintflow.lib.symexp import SymExpSparse
from constraintflow.lib.parse import *
from constraintflow.gbcsr.sparse_tensor import *


class ImageDataset:
    mnist_mean = torch.tensor([0.1307]).reshape(-1, 1)
    mnist_std = torch.tensor([0.3081]).reshape(-1, 1)
    cifar10_mean = torch.tensor([0.4914, 0.4822, 0.4465]).reshape(1, -1, 1, 1)
    cifar10_std = torch.tensor([0.2023, 0.1994, 0.2010]).reshape(1, -1, 1, 1)
    
    def __init__(self):
        pass

    def create_l(image, network_size, batch_size, eps = 0.01, dataset = 'mnist', no_sparsity=False):
        l = torch.clip(image - eps, min=0., max=1.)
        if dataset == 'mnist':
            l = (l - ImageDataset.mnist_mean) / ImageDataset.mnist_std
        else:
            mean = ImageDataset.cifar10_mean.expand(l.shape)
            std = ImageDataset.cifar10_std.expand(l.shape)
            l = (l - mean) / std
        l = l.reshape(batch_size,-1)
        l = create_sparse_init(l, float('-inf'), batch_size, network_size, no_sparsity)
        return l

    def create_u(image, network_size, batch_size, eps = 0.01, dataset = 'mnist', no_sparsity=False):
        u = torch.clip(image + eps, min=0., max=1.)
        if dataset == 'mnist':
            u = (u - ImageDataset.mnist_mean) / ImageDataset.mnist_std
        else:
            u = (u - ImageDataset.cifar10_mean) / ImageDataset.cifar10_std
        u = u.reshape(batch_size, -1)
        u = create_sparse_init(u, float('inf'), batch_size, network_size, no_sparsity)
        return u
    
    def get_dataset(data_name = '~/temp/constraintflow/data', n=1, train=False, mnist=True):
        transform = transforms.ToTensor()
        if mnist:
            data = datasets.MNIST(root=data_name, train=train, download=False, transform=transform)
        else:
            data = datasets.CIFAR10(root=data_name, train=train, download=False, transform=transform)
        dataloader = DataLoader(data, batch_size=n, shuffle=False)
        image, _ = next(iter(dataloader))
        true_label = data.targets[:n]
        return image, true_label
    
    def get_output_spec_weight_and_bias(X, y, dataset='mnist'):
        if dataset in ['mnist', 'cifar10', 'cifar']:
            num_classes = 10
        elif dataset in ['tinyimagenet', 'imagenet']:
            num_classes = 200
        weight = (torch.eye(num_classes).type_as(X)[y].unsqueeze(1)
            - torch.eye(num_classes).type_as(X).unsqueeze(0))
        I = (~(y.unsqueeze(1) == torch.arange(num_classes).type_as(y).unsqueeze(0)))
        weight = (weight[I].view(X.size(0), num_classes - 1, num_classes))

        bias = torch.zeros(num_classes - 1, dtype=torch.float32)
        return weight, bias
    

def create_sparse_init(x, dense_const, batch_size, network_size, no_sparsity):
    start_indices = [torch.tensor([0, 0])]
    blocks = [DenseBlock(x)]
    total_size = torch.tensor([batch_size, network_size])
    res = SparseTensor(start_indices, blocks, 2, total_size, dense_const=dense_const)
    if no_sparsity:
        mat = res.get_dense()
        res.blocks = [DenseBlock(mat)]
        res.end_indices = [res.total_size]
    return res

def create_L(l, network, batch_size, no_sparsity):
    L = PolyExpSparse(network, SparseTensor([], [], 3, torch.tensor([batch_size, network.size, network.size])), 0)
    L.const = copy.deepcopy(l)
    if no_sparsity:
        mat = L.mat.get_dense()
        L.blocks = [DenseBlock(mat)]
        L.start_indices = [torch.zeros(3, dtype=int)]
        L.end_indices = [L.mat.total_size]
    return L 

def create_U(u, network, batch_size, no_sparsity):
    U = PolyExpSparse(network, SparseTensor([], [], 3, torch.tensor([batch_size, network.size, network.size])), 0)
    U.const = copy.deepcopy(u)
    if no_sparsity:
        mat = U.mat.get_dense()
        U.blocks = [DenseBlock(mat)]
        U.start_indices = [torch.zeros(3, dtype=int)]
        U.end_indices = [U.mat.total_size]
    return U

def create_Z(l, u, network, no_sparsity):
    if not no_sparsity:
        l, u = l.blocks[0].block, u.blocks[0].block
        mid = (l+u)/2
        dist = (u-l)/2
        mat = DiagonalBlock(dist, torch.tensor([l.shape[0], l.shape[1], l.shape[1]]), diag_index=2)
        mat = SparseTensor([torch.zeros(3, dtype=int)], [mat], 3, torch.tensor([l.shape[0], network.size, l.shape[1]]))
        const = DenseBlock(mid)
        const = SparseTensor([torch.zeros(2, dtype=int)], [const], 2, torch.tensor([l.shape[0], network.size]))
        return SymExpSparse(network, mat, const)
    else:
        l = l.blocks[0].block[:, :network.input_size]
        u = u.blocks[0].block[:, :network.input_size]
        mid = (l+u)/2
        dist = (u-l)/2
        mat = DiagonalBlock(dist, torch.tensor([l.shape[0], l.shape[1], l.shape[1]]), diag_index=2)
        mat = SparseTensor([torch.zeros(3, dtype=int)], [mat], 3, torch.tensor([l.shape[0], network.size, l.shape[1]]))
        const = DenseBlock(mid)
        const = SparseTensor([torch.zeros(2, dtype=int)], [const], 2, torch.tensor([l.shape[0], network.size]))
        dense_mat = mat.get_dense()
        mat.blocks = [DenseBlock(dense_mat)]
        mat.start_indices = [torch.zeros(3, dtype=int)]
        mat.end_indices = [mat.total_size]
        const.blocks = [DenseBlock(const.get_dense())]
        const.start_indices = [torch.zeros(2, dtype=int)]
        const.end_indices = [const.total_size]
        return SymExpSparse(network, mat, const)

def create_llist(network):
    return torch.tensor([True] + [False]*network.num_layers)



def get_network_and_input_spec(network_file, batch_size, X, y, dataset, eps, train=False, no_sparsity=False):
    if dataset == 'tinyimagenet':
        X = X.to(torch.float32) / 255.0

    spec_weight, spec_bias = ImageDataset.get_output_spec_weight_and_bias(X, y, dataset)
    network = get_net(network_file, spec_weight, spec_bias, no_sparsity)
    l = ImageDataset.create_l(X, network.size, batch_size, eps, dataset, no_sparsity)
    u = ImageDataset.create_u(X, network.size, batch_size, eps, dataset, no_sparsity)
    L = create_L(l, network, batch_size, no_sparsity)
    U = create_U(u, network, batch_size, no_sparsity)
    Z = create_Z(l, u, network, no_sparsity)
    llist = create_llist(network)
    return network, l, u, L, U, Z, llist