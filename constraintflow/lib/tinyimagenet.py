#FEAT: TinyImageNet as the CIFAR100_TinyImageNet_ResNet benchmark loads it (generate_properties.py).
# Shared verbatim by constraintflow-gpu, original constraintflow and auto_LiRPA's harness script.
"""Labeled validation split, 56x56 center crop, benchmark mean/std."""
import os
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets.folder import default_loader

MEAN = (0.4802, 0.4481, 0.3975)
STD = (0.2302, 0.2265, 0.2262)
NUM_CLASSES = 200


class TinyImageNetValidation(Dataset):
    """The official flat val/ layout (images/ + val_annotations.txt) in its own order (val_0, val_1, ...,
    classes mixed), labeled with ImageFolder's class indices (sorted wnids)."""
    def __init__(self, root):
        self.transform = transforms.Compose([transforms.CenterCrop(56), transforms.ToTensor()])
        with open(os.path.join(root, 'wnids.txt')) as f:
            class_to_idx = {name: i for i, name in enumerate(sorted(f.read().split()))}
        val = os.path.join(root, 'val')
        with open(os.path.join(val, 'val_annotations.txt')) as f:
            labels = dict(line.split()[:2] for line in f if line.strip())
        self.samples = [(os.path.join(val, 'images', name), class_to_idx[labels[name]])
                        for name in sorted(labels, key=lambda name: int(name.split('_')[1].split('.')[0]))]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label = self.samples[index]
        return self.transform(default_loader(path)), label
