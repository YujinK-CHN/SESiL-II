"""
Dataset access, and the train / validation / test split.

utils.prepare_data() builds the loaders the merging pipeline needs. Everything
supervised goes through here instead, because it has to respect one boundary:

    train   what agents are finetuned on
    val     what certification ranks agents on -- and therefore what drives
            mating and what each agent is licensed to train on
    test    touched ONLY by sesil.evaluator

Validation is carved out of the training set rather than borrowed from test.
Without that split, certification would rank agents on the test set, putting it
inside the optimisation loop and making any reported test accuracy
optimistically biased.

The split is stratified and seeded by --seed, so a SESiL run and a baseline run
at the same seed see exactly the same training data -- which is what makes a
budget-matched comparison fair -- while different seeds average over the choice
of split.
"""

import os

import numpy as np
import torch
import torchvision
import torchvision.transforms as T
from sklearn.model_selection import train_test_split

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


def train_val_indices(targets, val_fraction, seed):
    """Stratified split of the training set into train and validation indices.

    Stratified so every class is represented in validation -- certification
    ranks per class, so a class missing from validation could not be certified
    at all.
    """
    indices = np.arange(len(targets))
    if val_fraction <= 0:
        return indices, np.array([], dtype=int)

    train_idx, val_idx = train_test_split(
        indices,
        test_size=val_fraction,
        random_state=seed,
        stratify=np.asarray(targets),
    )
    return np.sort(train_idx), np.sort(val_idx)


def subset_by_classes(dataset, classes, indices=None):
    """Restrict a CIFAR dataset to the given class ids, labels unchanged.

    `indices` optionally restricts the candidate pool first, which is how the
    training split is kept clear of validation samples.
    """
    keep = set(int(c) for c in classes) if classes is not None else None
    targets = dataset.targets
    pool = range(len(targets)) if indices is None else indices
    selected = [int(i) for i in pool if keep is None or targets[int(i)] in keep]
    return torch.utils.data.Subset(dataset, selected)


class DataBundle:
    """Train / validation / test, split once and reused for the whole run."""

    def __init__(self, args, download=True):
        self.args = args
        self.train_dset, self.test_dset, self.num_classes = get_datasets(args, download)

        self.train_idx, self.val_idx = train_val_indices(
            self.train_dset.targets, args.val_fraction, args.seed)

        self.batch_size = args.pretrain_batch_size
        self.workers = args.pretrain_workers

    # ------------------------------------------------------------------ #

    @property
    def train_size(self):
        """Samples an agent could train on. The budget's epoch-equivalent unit.

        Defined on the post-split training set, so one epoch-equivalent means
        the same amount of work for SESiL and the baseline alike.
        """
        return len(self.train_idx)

    @property
    def val_size(self):
        return len(self.val_idx)

    @property
    def test_size(self):
        return len(self.test_dset)

    # ------------------------------------------------------------------ #

    def _loader(self, dataset, shuffle, batch_size=None):
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size or self.batch_size,
            shuffle=shuffle,
            num_workers=self.workers,
        )

    def train_loader(self, classes=None, batch_size=None):
        """Training data, optionally restricted to `classes`. Never sees val."""
        subset = subset_by_classes(self.train_dset, classes, self.train_idx)
        return self._loader(subset, shuffle=True, batch_size=batch_size)

    def val_loader(self, batch_size=None):
        """What certification ranks on. Held out of training, never test."""
        subset = subset_by_classes(self.train_dset, None, self.val_idx)
        return self._loader(subset, shuffle=False, batch_size=batch_size)

    def test_loader(self, batch_size=None):
        """Touched only by sesil.evaluator."""
        return self._loader(self.test_dset, shuffle=False, batch_size=batch_size)

    def summary(self):
        return {
            'train_size': self.train_size,
            'val_size': self.val_size,
            'test_size': self.test_size,
            'val_fraction': self.args.val_fraction,
            'split_seed': self.args.seed,
        }
