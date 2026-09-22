"""
Raw CIFAR loaders.

utils.prepare_data() builds the loaders the merging pipeline needs (splits,
fractional loaders, class names).  Pretrain and mutation want something simpler:
plain train/test loaders over the whole dataset, or over a subset of classes.
That is what lives here, so the dataset path is configured once rather than
hardcoded in each script.
"""

import os

import numpy as np
import torch
import torchvision
import torchvision.transforms as T

CIFAR_MEAN = [125.307, 122.961, 113.8575]
CIFAR_STD = [51.5865, 50.847, 51.255]

normalize = T.Normalize(np.array(CIFAR_MEAN) / 255, np.array(CIFAR_STD) / 255)

TRAIN_TRANSFORM = T.Compose([
    T.RandomHorizontalFlip(),
    T.RandomCrop(32, padding=4),
    T.ToTensor(),
    normalize,
])
TEST_TRANSFORM = T.Compose([T.ToTensor(), normalize])


# Where torchvision expects to find each dataset, relative to --data-dir.
_DATASET_SPEC = {
    'cifar10': {
        'wrapper': torchvision.datasets.CIFAR10,
        'subdir': 'cifar-10-python',
        'num_classes': 10,
    },
    'cifar100': {
        'wrapper': torchvision.datasets.CIFAR100,
        'subdir': 'cifar-100-python',
        'num_classes': 100,
    },
}


def get_datasets(args, download=True):
    """Raw train/test datasets for --dataset, with standard augmentation."""
    spec = _DATASET_SPEC[args.dataset]
    root = os.path.join(args.data_dir, spec['subdir'])

    train_dset = spec['wrapper'](root=root, train=True, download=download,
                                 transform=TRAIN_TRANSFORM)
    test_dset = spec['wrapper'](root=root, train=False, download=download,
                                transform=TEST_TRANSFORM)
    return train_dset, test_dset, spec['num_classes']


def get_loaders(args, batch_size=None, workers=None, classes=None):
    """Train/test dataloaders, optionally restricted to `classes`."""
    batch_size = batch_size or args.pretrain_batch_size
    workers = workers if workers is not None else args.pretrain_workers

    train_dset, test_dset, num_classes = get_datasets(args)

    if classes is not None:
        train_dset = subset_by_classes(train_dset, classes)
        test_dset = subset_by_classes(test_dset, classes)

    train_loader = torch.utils.data.DataLoader(
        train_dset, batch_size=batch_size, shuffle=True, num_workers=workers)
    test_loader = torch.utils.data.DataLoader(
        test_dset, batch_size=batch_size, shuffle=False, num_workers=workers)

    return train_loader, test_loader, num_classes


def subset_by_classes(dataset, classes):
    """Restrict a CIFAR dataset to the given class ids, labels unchanged."""
    keep = set(int(c) for c in classes)
    indices = [i for i, label in enumerate(dataset.targets) if label in keep]
    return torch.utils.data.Subset(dataset, indices)
