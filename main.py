"""
Single entry point for every method.

    python main.py --method sesil    --dataset cifar10 --budget 500 --seed 0
    python main.py --method baseline --dataset cifar10 --budget 500 --seed 0

Normally invoked through run.sh / run_sesil.sh / run_baseline.sh.

--budget is in epoch-equivalents (one backprop pass over the full training
set), which is the unit that makes SESiL and the baseline comparable. Both
runs above are given exactly the same amount of training compute.

For --method sesil this runs the whole pipeline: it creates the initial
population if one does not already exist, then evolves it until the budget is
spent. Nothing has to be copied into place by hand.
"""

import json
import os
import sys

from config import get_config, resolve
from utils import set_seed

from sesil.budget import (
    BudgetTracker,
    estimate_generation_cost,
    estimate_generations,
    estimate_pretrain_cost,
    samples_for,
)


def _make_budget(args):
    """Build the ledger, sized by the actual dataset."""
    from sesil.data import get_datasets

    train_dset, test_dset, _ = get_datasets(args, download=True)
    return BudgetTracker(
        total=args.budget,
        train_set_size=len(train_dset),
        test_set_size=len(test_dset),
    )


def main(argv=None):
    args = resolve(get_config().parse_args(argv))

    set_seed(args.seed)

    budget = _make_budget(args)

    print('=' * 70)
    print(f'method     : {args.method}')
    print(f'seed       : {args.seed}')
    print(f'device     : {args.device}')
    print(f'dataset    : {args.dataset} ({args.num_classes} classes, '
          f'{budget.train_set_size} train samples)')
    print(f'model      : {args.arch}')
    print(f'run dir    : {args.run_dir}')
    print(f'budget     : {args.budget} epoch-equivalents '
          f'(1 = one backprop pass over the full training set)')

    if args.method == 'sesil':
        pre = estimate_pretrain_cost(args)
        per_gen = estimate_generation_cost(args)
        gens = estimate_generations(args)
        per_agent = samples_for(args.individual_budget, budget.train_set_size)
        print(f'merger     : {args.merger}')
        print(f'selection  : {args.selection}')
        print(f'population : {args.pop_size} x {args.classes_per_model} classes')
        print(f'  pretrain        {pre:.2f} epoch-equiv '
              f'({args.pop_size} agents x {args.pretrain_epochs} epochs on '
              f'{args.classes_per_model}/{args.num_classes} of the data)')
        print(f'  per agent/gen   {args.individual_budget:.3f} epoch-equiv '
              f'= {per_agent} sample-presentations')
        print(f'  per generation  {per_gen:.2f} epoch-equiv '
              f'({args.pop_size} agents)')
        print(f'  -> {gens} generations from a budget of {args.budget:g}')
        print(f'population dir: {args.population_dir}')
    else:
        print(f'mode       : {args.baseline_mode}')
        print(f'  -> {args.baseline_epochs} epochs')
    print('=' * 70)

    os.makedirs(args.run_dir, exist_ok=True)

    if args.method == 'sesil':
        result = _run_sesil(args, budget)
    elif args.method == 'baseline':
        from sesil.baseline import run_baseline
        result = run_baseline(args, budget)
    else:
        raise ValueError(f'Unknown method {args.method!r}')

    # Written at the end so the accounting reflects what was actually spent.
    with open(os.path.join(args.run_dir, 'config.json'), 'w') as f:
        json.dump({'config': vars(args), 'budget': budget.summary()},
                  f, indent=2, default=str)

    return result


def _run_sesil(args, budget):
    from sesil.evolution import run_evolution
    from sesil.pretrain import ensure_population

    # Stage 1: pretrain. Skipped when a population is already on disk, unless
    # --force-pretrain. This is what used to be a manual copy step.
    if args.start_gen == 0:
        population = ensure_population(args, budget)
        if not population:
            raise RuntimeError(f'Pretrain produced no individuals in {args.population_dir}')
    else:
        print(f'[main] resuming from generation {args.start_gen}; skipping pretrain.')

    if args.pretrain_only:
        print('[main] --pretrain-only set; stopping after pretrain.')
        print(budget.report())
        return None

    # Stage 2: evolution, until the budget runs out.
    return run_evolution(args, budget)


if __name__ == '__main__':
    sys.exit(main() or 0)
