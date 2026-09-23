"""
Pretrain: create the initial population.

Previously this lived in a standalone script with hardcoded paths, and its
output had to be copied into place by hand before evolution could start.  It is
now a normal stage of the pipeline: main.py calls ensure_population() and only
trains what is missing.

Each individual is a model trained on a random subset of --classes-per-model
classes, saved as

    <population dir>/agent_NNN/<arch>_v0.pth.tar

The subset it saw is stored as provenance, not identity. Generation 0 is
evaluated across the whole label space like every later generation and earns its
proficiency certificate from that measurement -- pretrain creates genuine
specialists, and evaluation discovers what they specialise in.
"""

import os
import time

import numpy as np
import torch
from sklearn.model_selection import train_test_split

from utils import train_logits

from sesil.population import list_population, save_agent


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


def ensure_population(args, data, budget=None):
    """Create the initial population if it is not already on disk.

    Returns the list of model ids making up generation 0.

    When a population is reused rather than trained, nothing is charged to the
    budget -- the compute was spent by whichever run created it. That makes
    reuse across seeds cheaper but means the budget figures of two runs are
    only comparable when both trained their own population, so the reuse is
    reported loudly.
    """
    existing = list_population(args.population_dir)

    if existing and not args.force_pretrain:
        print(f'[pretrain] Found {len(existing)} individuals in {args.population_dir}; '
              f'skipping pretrain.')
        if len(existing) != args.pop_size:
            print(f'[pretrain] NOTE: --pop-size is {args.pop_size} but the existing '
                  f'population has {len(existing)} members. Using what is on disk. '
                  f'Pass --force-pretrain to rebuild it.')
        print('[pretrain] NOTE: reused population is NOT charged to this run\'s '
              'budget. For a like-for-like budget comparison, use --force-pretrain '
              'or a fresh --population-dir.')
        return existing

    print(f'[pretrain] Building a population of {args.pop_size} into {args.population_dir}')
    return pretrain_population(args, data, budget)


def pretrain_population(args, data, budget=None):
    """Train --pop-size individuals, each on its own random class subset."""
    os.makedirs(args.population_dir, exist_ok=True)

    num_classes = data.num_classes
    val_loader = data.val_loader()

    start = time.time()
    for individual in range(args.pop_size):
        # train_test_split here is just a convenient way to draw a random subset
        # of class ids; the "test" half is discarded.
        split, _ = train_test_split(
            np.arange(num_classes), train_size=args.classes_per_model
        )
        split = sorted(int(c) for c in split)

        train_loader = data.train_loader(classes=split)

        n_samples = len(train_loader.dataset)
        if budget is not None:
            cost = budget.spend_samples('pretrain', n_samples, epochs=args.pretrain_epochs)
            print(f'[pretrain] {individual + 1}/{args.pop_size} on classes {split}  '
                  f'{n_samples} samples  cost={cost:.3f} epoch-equiv')
        else:
            print(f'[pretrain] {individual + 1}/{args.pop_size} on classes {split}')

        model = build_model(args, num_classes).train()
        model, final_acc = train_logits(
            model=model,
            train_loader=train_loader,
            test_loader=val_loader,
            epochs=args.pretrain_epochs,
        )
        print(f'[pretrain] accuracy on {split}: {final_acc}')

        # The subset is recorded as provenance only. Nothing downstream reads
        # it: generation 0 is evaluated on the whole label space like any other,
        # and earns its certificate the same way. Pretrain creates genuine
        # specialists; evaluation discovers what they specialise in.
        meta = {
            'generation': 0,
            'certificate': [],          # granted at the first evaluation
            'trained_on': split,
            'pretrain_subset': split,
            'parents': [],
            'is_loner': False,
        }
        save_path = save_agent(model, args.population_dir, individual, args.arch, meta)
        print(f'[pretrain] saved -> {save_path}')

    print(f'[pretrain] done in {time.time() - start:.1f}s')
    return list_population(args.population_dir)
