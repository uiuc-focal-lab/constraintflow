"""Export the MadryLab robustness CIFAR10 ResNet50 checkpoint (cifar_linf_8.pt) to ONNX (network N40).

The input normalizer is left out (cf-gpu normalizes CIFAR10 with the same mean/std),
and the final 4x4 avg_pool is folded into the Linear so the graph is Conv/BN/Relu/Add/Flatten/Gemm.
BN stays as nodes (PRESERVE): the parser finds conv weights by their parameter names, which
the exporter's own Conv+BN fusion renames; cf-gpu folds BN at load time instead.

usage: python export_resnet50.py <robustness checkout> <checkpoint.pt> <out.onnx>
"""
import os
import sys

import onnx
import torch
import torch.nn as nn


CIFAR10_MEAN = torch.tensor([0.4914, 0.4822, 0.4465])
CIFAR10_STD = torch.tensor([0.2023, 0.1994, 0.2010])


class PoolFoldedResNet(nn.Module):
    """ResNet50 trunk + Flatten + Linear(2048*16, 10) equal to avg_pool2d(4) then Linear(2048, 10)."""

    def __init__(self, net):
        super().__init__()
        self.net = net
        w, b = net.linear.weight.data, net.linear.bias.data
        self.linear = nn.Linear(w.shape[1] * 16, w.shape[0])
        # Flatten order is (c, h, w): column c*16 + hw takes W[:, c] / 16.
        self.linear.weight.data = w.repeat_interleave(16, dim=1) / 16
        self.linear.bias.data = b.clone()

    def forward(self, x):
        n = self.net
        out = torch.relu(n.bn1(n.conv1(x)))
        out = n.layer4(n.layer3(n.layer2(n.layer1(out))))
        return self.linear(torch.flatten(out, 1))


def undedup_initializers(path):
    """The exporter turns equal-valued params (bn3.bias == shortcut.1.bias) into Identity nodes;
    store them as initializers again so BN folding sees constant operands."""
    model = onnx.load(path)
    inits = {i.name: i for i in model.graph.initializer}
    keep = []
    for node in model.graph.node:
        if node.op_type == "Identity" and node.input[0] in inits:
            copy = onnx.TensorProto()
            copy.CopyFrom(inits[node.input[0]])
            copy.name = node.output[0]
            model.graph.initializer.append(copy)
        else:
            keep.append(node)
    del model.graph.node[:]
    model.graph.node.extend(keep)
    onnx.checker.check_model(model)
    onnx.save(model, path)


def main(ckpt_path, out_path):
    from robustness.cifar_models.resnet import ResNet50
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    sd = ckpt["state_dict"]
    assert torch.allclose(sd["module.normalizer.new_mean"].flatten(), CIFAR10_MEAN)
    assert torch.allclose(sd["module.normalizer.new_std"].flatten(), CIFAR10_STD)

    net = ResNet50(num_classes=10)
    prefix = "module.model."
    net.load_state_dict({k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)})
    net.eval()
    folded = PoolFoldedResNet(net).eval()

    x = torch.rand(8, 3, 32, 32)
    x = (x - CIFAR10_MEAN.view(1, 3, 1, 1)) / CIFAR10_STD.view(1, 3, 1, 1)
    with torch.no_grad():
        diff = (net(x) - folded(x)).abs().max().item()
    print(f"pool fold max abs diff: {diff:.2e}")
    assert diff < 1e-4

    torch.onnx.export(folded, torch.zeros(1, 3, 32, 32), out_path, input_names=["input"],
                      output_names=["output"], opset_version=12, do_constant_folding=True,
                      dynamo=False, training=torch.onnx.TrainingMode.PRESERVE)
    undedup_initializers(out_path)
    print(f"wrote {out_path} (epoch {ckpt['epoch']}, nat {float(ckpt['nat_prec1']):.2f}%, "
          f"adv {float(ckpt['adv_prec1']):.2f}%)")


if __name__ == "__main__":
    sys.path.insert(0, os.path.abspath(sys.argv[1]))
    main(sys.argv[2], sys.argv[3])
