"""
Pretrain: create the initial population.

Previously this lived in a standalone script with hardcoded paths, and its
output had to be copied into place by hand before evolution could start.  It is
now a normal stage of the pipeline: main.py calls ensure_population() and only
trains what is missing.

Each individual is a model trained on a random subset of --classes-per-model
classes, saved as

    <population dir>/<hash8 of its labels>/<arch>_v0.pth.tar
"""

import os
import time

import numpy as np
import torch
from sklearn.model_selection import train_test_split

from utils import encode_labels, save_model, train_logits

from sesil.data import get_datasets, subset_by_classes
from sesil.population import list_population


def build_model(args, num_classes):
    """Fresh backbone of the configured architecture and width."""
    if args.model_name.startswith('resnet20'):
        from models.resnets import resnet20
        return resnet20(w=args.model_width, num_classes=num_classes).to(args.device)
    if args.model_name.startswith('vgg11'):
        from models.vgg import vgg11
        return vgg11(w=args.model_width, num_classes=num_classes).to(args.device)
    raise NotImplementedError(
        f'No builder for model {args.model_name!r}. Add one in sesil/pretrain.py.'
    )


def ensure_population(args):
    """Create the initial population if it is not already on disk.

    Returns the list of model ids making up generation 0.
    """
    existing = list_population(args.population_dir)

    if existing and not args.force_pretrain:
        print(f'[pretrain] Found {len(existing)} individuals in {args.population_dir}; '
              f'skipping pretrain.')
        if len(existing) != args.pop_size:
            print(f'[pretrain] NOTE: --pop-size is {args.pop_size} but the existing '
                  f'population has {len(existing)} members. Using what is on disk. '
                  f'Pass --force-pretrain to rebuild it.')
        return existing

    print(f'[pretrain] Building a population of {args.pop_size} into {args.population_dir}')
    return pretrain_population(args)


def pretrain_population(args):
    """Train --pop-size individuals, each on its own random class subset."""
    os.makedirs(args.population_dir, exist_ok=True)

    train_dset, test_dset, num_classes = get_datasets(args)

    start = time.time()
    for individual in range(args.pop_size):
        # train_test_split here is just a convenient way to draw a random subset
        # of class ids; the "test" half is discarded.
        split, _ = train_test_split(
            np.arange(num_classes), train_size=args.classes_per_model
        )
        split = sorted(int(c) for c in split)

        train_loader = torch.utils.data.DataLoader(
            subset_by_classes(train_dset, split),
            batch_size=args.pretrain_batch_size, shuffle=True,
            num_workers=args.pretrain_workers)
        test_loader = torch.utils.data.DataLoader(
            subset_by_classes(test_dset, split),
            batch_size=args.pretrain_batch_size, shuffle=False,
            num_workers=args.pretrain_workers)

        print(f'[pretrain] {individual + 1}/{args.pop_size} on classes {split}')

        model = build_model(args, num_classes).train()
        model, final_acc = train_logits(
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            epochs=args.pretrain_epochs,
        )
        print(f'[pretrain] accuracy on {split}: {final_acc}')

        label_key = [str(c) for c in split]
        model_id = encode_labels(label_key)
        save_dir = os.path.join(args.population_dir, model_id)
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(
            save_dir, f'{args.arch}_v{len(os.listdir(save_dir))}.pth.tar')
        save_model(model, save_path)
        print(f'[pretrain] saved -> {save_path}')

    print(f'[pretrain] done in {time.time() - start:.1f}s')
    return list_population(args.population_dir)
