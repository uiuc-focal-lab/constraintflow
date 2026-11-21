import functools
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from constraintflow.lib.polyexp import PolyExpSparse
from constraintflow.lib.symexp import SymExpSparse
from constraintflow.lib.parse import *
from constraintflow.gbcsr.sparse_tensor import *


#FEAT: most output atoms in one ACAS Xu property (props 7 and 8); shorter specs are zero-padded.
ACAS_SPEC_ROWS = 6
ACAS_INPUT_SHAPE = (1, 1, 5)


#FEAT: other box-input specs (VNN-LIB benchmarks): dataset 'box-<input shape, x-separated>-<outputs>-<spec rows>', e.g. 'box-4-2-2'.
#FEAT: 'vnnlib:<benchmark folder>' datasets: VNN-COMP specs, each batch's parsed inside the timed run.
def vnnlib_specs():
    """The VNN-LIB benchmark reader (spec list, input shape, parser)."""
    from constraintflow.lib import vnnlib
    return vnnlib


def read_vnnlib_batch(folder, y, X):
    """Specs y parsed now: their box [batch, 2 (lo, hi), ...] shaped like X, and margin rows C y + d > 0 (one batch shares d)."""
    import numpy as np
    lo, hi, C, d = vnnlib_specs().read_batch(folder, y.tolist())
    assert (d == d[0]).all(), 'specs in one batch must share their offsets d'
    as_x = lambda a: torch.from_numpy(a).to(device=X.device, dtype=X.dtype)
    return as_x(np.stack([lo, hi], 1)).reshape(X.shape), as_x(C), as_x(d[0])


def vnnlib_path(dataset):
    return dataset[len('vnnlib:'):] if dataset.startswith('vnnlib:') else None


def parse_box_dataset(dataset):
    """(input shape, number of outputs, spec rows) of a 'box-...' dataset name, else None."""
    if not dataset.startswith('box-'):
        return None
    shape, outputs, rows = dataset[len('box-'):].split('-')
    return tuple(int(d) for d in shape.split('x')), int(outputs), int(rows)


class ImageDataset:
    mnist_mean = torch.tensor([0.1307]).reshape(-1, 1)
    mnist_std = torch.tensor([0.3081]).reshape(-1, 1)
    cifar10_mean = torch.tensor([0.4914, 0.4822, 0.4465]).reshape(1, -1, 1, 1)
    cifar10_std = torch.tensor([0.2023, 0.1994, 0.2010]).reshape(1, -1, 1, 1)
    #FEAT: CIFAR-100 (CIFAR100_TinyImageNet_ResNet benchmark's normalizer).
    cifar100_mean = torch.tensor([0.5071, 0.4865, 0.4409]).reshape(1, -1, 1, 1)
    cifar100_std = torch.tensor([0.2673, 0.2564, 0.2761]).reshape(1, -1, 1, 1)
    tinyimagenet_mean = torch.tensor([0.4802, 0.4481, 0.3975]).reshape(1, -1, 1, 1)
    tinyimagenet_std = torch.tensor([0.2302, 0.2265, 0.2262]).reshape(1, -1, 1, 1)
    imagenet_mean = torch.tensor([0.485, 0.456, 0.406]).reshape(1, -1, 1, 1)
    imagenet_std = torch.tensor([0.229, 0.224, 0.225]).reshape(1, -1, 1, 1)
    _normalization_cache = {}
    
    def __init__(self):
        pass

    @staticmethod
    def _normalization(image, dataset):
        if dataset == 'mnist':
            canonical_dataset = 'mnist'
            mean, std = ImageDataset.mnist_mean, ImageDataset.mnist_std
        elif dataset in ['cifar10', 'cifar']:
            canonical_dataset = 'cifar10'
            mean, std = ImageDataset.cifar10_mean, ImageDataset.cifar10_std
        #FIX FOR: TinyImageNet was normalized with the CIFAR-10 mean/std; use the benchmark's.
        elif dataset == 'cifar100':
            canonical_dataset = 'cifar100'
            mean, std = ImageDataset.cifar100_mean, ImageDataset.cifar100_std
        elif dataset == 'tinyimagenet':
            canonical_dataset = 'tinyimagenet'
            mean, std = ImageDataset.tinyimagenet_mean, ImageDataset.tinyimagenet_std
        elif dataset == 'imagenet':
            canonical_dataset = 'imagenet'
            mean, std = ImageDataset.imagenet_mean, ImageDataset.imagenet_std
        else:
            return None, None

        key = (canonical_dataset, str(image.device), image.dtype)
        if key not in ImageDataset._normalization_cache:
            ImageDataset._normalization_cache[key] = (
                mean.to(device=image.device, dtype=image.dtype),
                std.to(device=image.device, dtype=image.dtype),
            )
        return ImageDataset._normalization_cache[key]

    def create_l(image, network_size, batch_size, eps = 0.01, dataset = 'mnist', no_sparsity=False):
        #FEAT: a vnnlib image is its box itself: [batch, 2 (lo, hi), ...], no clipping or normalization.
        if vnnlib_path(dataset):
            return create_sparse_init(image[:, 0].reshape(batch_size, -1), float('-inf'), batch_size, network_size, no_sparsity)
        l = torch.clip(image - eps, min=0., max=1.)
        mean, std = ImageDataset._normalization(l, dataset)
        if mean is not None:
            l = (l - mean) / std
        l = l.reshape(batch_size,-1)
        l = create_sparse_init(l, float('-inf'), batch_size, network_size, no_sparsity)
        return l

    def create_u(image, network_size, batch_size, eps = 0.01, dataset = 'mnist', no_sparsity=False):
        if vnnlib_path(dataset):
            return create_sparse_init(image[:, 1].reshape(batch_size, -1), float('inf'), batch_size, network_size, no_sparsity)
        u = torch.clip(image + eps, min=0., max=1.)
        mean, std = ImageDataset._normalization(u, dataset)
        if mean is not None:
            u = (u - mean) / std
        u = u.reshape(batch_size, -1)
        u = create_sparse_init(u, float('inf'), batch_size, network_size, no_sparsity)
        return u
    
    # TODO: I think this function is not used?
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
        #FEAT: ACAS Xu traces with a placeholder spec of ACAS_SPEC_ROWS rows; SpecBounds swaps in each property's.
        if dataset == 'acasxu':
            margins, _ = ImageDataset.get_output_spec_weight_and_bias(X, y, 'margins5')
            weight = torch.cat([margins, -margins[:, :ACAS_SPEC_ROWS - 4]], dim=1)
            return weight, torch.zeros(ACAS_SPEC_ROWS, device=X.device, dtype=X.dtype)
        #FEAT: vnnlib specs: y indexes the spec file; its margin rows C y + d > 0 (one batch shares d).
        path = vnnlib_path(dataset)
        if path:
            return read_vnnlib_batch(path, y, X)[1:]
        #FEAT: box specs trace with a dense placeholder spec of the right size; SpecBounds swaps in each property's.
        box = parse_box_dataset(dataset)
        if box is not None:
            _, outputs, rows = box
            return (torch.ones(X.shape[0], rows, outputs, device=X.device, dtype=X.dtype),
                    torch.zeros(rows, device=X.device, dtype=X.dtype))
        if dataset == 'margins5':
            num_classes = 5
        elif dataset in ['mnist', 'cifar10', 'cifar']:
            num_classes = 10
        elif dataset == 'cifar100':
            num_classes = 100
        elif dataset in ['tinyimagenet', 'imagenet']:
            num_classes = 200
        y = y.to(device=X.device)
        eye = torch.eye(num_classes, device=X.device, dtype=X.dtype)
        others = torch.arange(num_classes - 1, device=X.device, dtype=y.dtype).unsqueeze(0)
        others = others + (others >= y.unsqueeze(1))
        weight = eye[y].unsqueeze(1) - eye[others]

        bias = torch.zeros(num_classes - 1, device=X.device, dtype=X.dtype)
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



#FEAT: add the folded margin bias that parse_onnx_layers(fold_spec=True) moved out of the last layer.
def apply_spec_offset(network, res):
    offset = getattr(network[-1], 'spec_offset', None)
    # The simulacrum pass returns dummy shapes or meta tensors, not values.
    if offset is None or not all(isinstance(r, torch.Tensor) and not r.is_meta for r in res):
        return res
    return tuple(r + offset for r in res)


#FEAT: input_box=(x_L, x_U), already in the network's input space, replaces the clip(X +- eps) box and dataset normalization.
def get_network_and_input_spec(network_file, batch_size, X, y, dataset, eps, train=False, no_sparsity=False, initializers=None, members=None, input_box=None, fold_spec=False):
    #FIX FOR: TinyImageNet pixels were divided by 255 a second time (ToTensor already gives [0, 1]).
    #FEAT: images follow the run dtype (fp64 under --dtype float64).
    X = X.to(torch.get_default_dtype())

    #FEAT: a VNN-LIB batch is parsed here, inside the timed run; X becomes the specs' box.
    if vnnlib_path(dataset):
        X, spec_weight, spec_bias = read_vnnlib_batch(vnnlib_path(dataset), y, X)
    else:
        spec_weight, spec_bias = ImageDataset.get_output_spec_weight_and_bias(X, y, dataset)
    network = get_net(network_file, spec_weight, spec_bias, no_sparsity, initializers=initializers, fold_spec=fold_spec)
    if input_box is None:
        l = ImageDataset.create_l(X, network.size, batch_size, eps, dataset, no_sparsity)
        u = ImageDataset.create_u(X, network.size, batch_size, eps, dataset, no_sparsity)
    else:
        #FEAT: caller-supplied box (see input_box above).
        x_L, x_U = (t.to(torch.get_default_dtype()).reshape(batch_size, -1) for t in input_box)
        l = create_sparse_init(x_L, float('-inf'), batch_size, network.size, no_sparsity)
        u = create_sparse_init(x_U, float('inf'), batch_size, network.size, no_sparsity)
    L = create_L(l, network, batch_size, no_sparsity) if members is None or 'L' in members else None
    U = create_U(u, network, batch_size, no_sparsity) if members is None or 'U' in members else None
    Z = create_Z(l, u, network, no_sparsity) if members is None or 'Z' in members else None
    # llist type: torch Tensor of bool
    llist = create_llist(network)
    return network, l, u, L, U, Z, llist