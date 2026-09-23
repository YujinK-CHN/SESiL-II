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

from sesil.population import list_population, read_meta, save_agent


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

    A reused population is still CHARGED, at what it originally cost to build
    (see _charge_reused_population). Reuse saves wall-clock, not budget: the
    compute that produced those specialists was really spent, and letting SESiL
    inherit them for free while the baseline starts from random init would quietly
    unmatch a budget-matched comparison.
    """
    existing = list_population(args.population_dir)

    if existing and not args.force_pretrain:
        print(f'[pretrain] Found {len(existing)} agents in {args.population_dir}; '
              f'skipping pretrain.')
        if len(existing) != args.pop_size:
            print(f'[pretrain] NOTE: --pop-size is {args.pop_size} but the existing '
                  f'population has {len(existing)} members. Using what is on disk. '
                  f'Pass --force-pretrain to rebuild it.')
        _charge_reused_population(args, existing, budget)
        return existing

    print(f'[pretrain] Building a population of {args.pop_size} into {args.population_dir}')
    return pretrain_population(args, data, budget)


def _charge_reused_population(args, agent_ids, budget):
    """Charge a reused population at what it originally cost to create.

    Reuse must not be free. The compute that produced those specialists was
    really spent, and if SESiL inherits them for nothing while the baseline
    starts from random init, the budget-matched comparison is not matched at
    all -- SESiL simply gets a head start the ledger never sees.

    The price is read from each agent's own metadata rather than recomputed, so
    it stays correct even if --pretrain-epochs or --pop-size have changed since
    the population was built.
    """
    if budget is None:
        return 0.0

    recorded = [read_meta(args.population_dir, a).get('pretrain_cost')
                for a in agent_ids]

    if any(c is None for c in recorded):
        from sesil.budget import estimate_pretrain_cost
        total = estimate_pretrain_cost(args)
        print(f'[pretrain] WARNING: this population records no creation cost '
              f'(built before costs were tracked). Falling back to the estimate '
              f'for the CURRENT settings: {total:.2f} epoch-equiv. If those '
              f'settings differ from the ones that built it, the figure is wrong '
              f'-- rebuild with --force-pretrain for exact accounting.')
    else:
        total = float(sum(recorded))
        print(f'[pretrain] Reused population charged at its recorded creation '
              f'cost: {total:.2f} epoch-equiv.')

    budget.spend_samples('pretrain', budget.train_set_size * total, epochs=1)
    return total


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
        # Computed whether or not there is a ledger, because the cost is stored
        # with the agent so a later run can be charged for reusing it.
        cost = (n_samples * args.pretrain_epochs) / max(data.train_size, 1)
        if budget is not None:
            budget.spend_samples('pretrain', n_samples, epochs=args.pretrain_epochs)
        print(f'[pretrain] {individual + 1}/{args.pop_size} on classes {split}  '
              f'{n_samples} samples  cost={cost:.3f} epoch-equiv')

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
            'pretrain_cost': round(cost, 6),   # what reusing this agent costs
            'parents': [],
            'is_loner': False,
        }
        save_path = save_agent(model, args.population_dir, individual, args.arch, meta)
        print(f'[pretrain] saved -> {save_path}')

    print(f'[pretrain] done in {time.time() - start:.1f}s')
    return list_population(args.population_dir)
