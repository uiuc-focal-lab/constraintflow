from compiled_code.specs.networks import FullyConnected, Conv
from compiled_code.specs.models import Models
import compiled_code.specs.parse as parse 

import torch 
import onnx

def get_net_format(net_name):
    net_format = None
    if 'pt' in net_name:
        net_format = 'torch'
    if 'onnx' in net_name:
        net_format = 'onnx'
    return net_format

def get_net_name(net_file):
    if 'pth.tar' in net_file:
        net_name = net_file.split('/')[-1].split('_')[0]
    else:
        net_name = net_file.split('/')[-1].split('.')[-2]
    return net_name

def get_torch_test_net(net_name, path, device='cpu', input_size=28):
    if net_name == 'fc1':
        net = FullyConnected(device, input_size, [50, 10]).to(device)
    elif net_name == 'fc2':
        net = FullyConnected(device, input_size, [100, 50, 10]).to(device)
    elif net_name == 'fc3':
        net = FullyConnected(device, input_size, [100, 100, 10]).to(device)
    elif net_name == 'fc4':
        net = FullyConnected(device, input_size, [100, 100, 50, 10]).to(device)
    elif net_name == 'fc5':
        net = FullyConnected(device, input_size, [100, 100, 100, 10]).to(device)
    elif net_name == 'fc6':
        net = FullyConnected(device, input_size, [100, 100, 100, 100, 10]).to(device)
    elif net_name == 'fc7':
        net = FullyConnected(device, input_size, [100, 100, 100, 100, 100, 10]).to(device)
    elif net_name == 'conv1':
        net = Conv(device, input_size, [(16, 3, 2, 1)], [100, 10], 10).to(device)
    elif net_name == 'conv2':
        net = Conv(device, input_size, [(16, 4, 2, 1), (32, 4, 2, 1)], [100, 10], 10).to(device)
    elif net_name == 'conv3':
        net = Conv(device, input_size, [(16, 4, 2, 1), (64, 4, 2, 1)], [100, 100, 10], 10).to(device)
    else:
        assert False

def get_torch_net(net_file, dataset, device='cpu'):
    net_name = get_net_name(net_file)

    if 'cpt' in net_file:
        return get_torch_test_net(net_name, net_file)

    if dataset == 2:
        model = Models[net_name](in_ch=1, in_dim=28)
    # elif dataset == Dataset.IMAGENET:
    #     model = get_architecture(net_file, dataset)
    # elif dataset == Dataset.CIFAR10 or dataset == Dataset.OVAL_CIFAR:
    #     if 'resnet' in net_file:
    #         model = get_architecture(net_file, dataset)
    #     else:
    #         model = models.Models[net_name](in_ch=3, in_dim=32) 
    else:
        raise ValueError("Unsupported dataset")

    if 'kw' in net_file:
        model.load_state_dict(torch.load(net_file, map_location=torch.device(device))['state_dict'][0])
    elif 'eran' in net_file:
        model.load_state_dict(torch.load(net_file, map_location=torch.device(device))['state_dict'][0])
    else:
        model.load_state_dict(torch.load(net_file, map_location=torch.device(device))['state_dict'])

    return model

def get_net(net_name, dataset=None):
    net_format = get_net_format(net_name)
    if net_format == 'torch':
        # Load the model
        net_torch = get_torch_net(net_name, dataset)
        net = parse.parse_torch_layers(net_torch)

    elif net_format == 'onnx':
        net_onnx = onnx.load(net_name)
        net = parse.parse_onnx_layers(net_onnx)
    else:
        raise ValueError("Unsupported net format!")

    net.net_name = net_name
    return net