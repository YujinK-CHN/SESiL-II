"""
Single entry point for every method.

    python main.py --method sesil    --dataset cifar10 --budget 500 --seed 0
    python main.py --method baseline --dataset cifar10 --budget 500 --seed 0

Normally invoked through run.sh / run_sesil.sh / run_baseline.sh.

--budget is in epoch-equivalents (one backprop pass over the full training
set), which is the unit that makes SESiL and the baseline comparable. Both runs
above get exactly the same training compute, the same train/validation split,
and evaluation at the same budget watermarks -- so their curves can be plotted
on one axis without any post-hoc alignment.

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
from sesil.data import DataBundle
from sesil.evaluator import Evaluator
from sesil.logging import RunLogger


def main(argv=None):
    args = resolve(get_config().parse_args(argv))

    set_seed(args.seed)

    # One split for the whole run: train for finetuning, validation for
    # certification, test for the evaluator alone.
    data = DataBundle(args)

    # The budget's unit is an epoch over the TRAINABLE set, so it means the
    # same amount of work whatever --val-fraction is.
    budget = BudgetTracker(
        total=args.budget,
        train_set_size=data.train_size,
        test_set_size=data.test_size,
    )

    os.makedirs(args.run_dir, exist_ok=True)
    logger = RunLogger(args.run_dir)

    evaluator = Evaluator(
        test_loader=data.test_loader(),
        num_classes=args.num_classes,
        eval_interval=args.eval_interval,
        logger=logger,
        budget=budget,
        meta={'method': args.method, 'seed': args.seed, 'dataset': args.dataset,
              'merger': args.merger if args.method == 'sesil' else None},
    )

    _print_banner(args, data, budget)

    try:
        if args.method == 'sesil':
            result = _run_sesil(args, budget, data, logger, evaluator)
        elif args.method == 'baseline':
            from sesil.baseline import run_baseline
            result = run_baseline(args, budget, data, logger, evaluator)
        else:
            raise ValueError(f'Unknown method {args.method!r}')
    finally:
        # Written whatever happens, so a crashed run still records what it spent.
        with open(os.path.join(args.run_dir, 'config.json'), 'w') as f:
            json.dump({'config': vars(args),
                       'data': data.summary(),
                       'budget': budget.summary()},
                      f, indent=2, default=str)
        logger.close()

    return result


def _print_banner(args, data, budget):
    print('=' * 70)
    print(f'method     : {args.method}')
    print(f'seed       : {args.seed}')
    print(f'device     : {args.device}')
    print(f'dataset    : {args.dataset} ({args.num_classes} classes)')
    print(f'  split    : {data.train_size} train / {data.val_size} val / '
          f'{data.test_size} test')
    print(f'             certification ranks on VAL; only the evaluator sees TEST')
    print(f'model      : {args.arch}')
    print(f'run dir    : {args.run_dir}')
    print(f'budget     : {args.budget} epoch-equivalents '
          f'(1 = one backprop pass over {data.train_size} samples)')
    print(f'eval every : {args.eval_interval} epoch-equivalents '
          f'(~{int(args.budget / max(args.eval_interval, 1e-9)) + 1} points)')

    if args.method == 'sesil':
        pre = estimate_pretrain_cost(args)
        per_gen = estimate_generation_cost(args)
        gens = estimate_generations(args)
        per_agent = samples_for(args.individual_budget, budget.train_set_size)
        stop = args.stop_node if args.stop_node is not None else 'none / full merge'
        print(f'merger     : {args.merger}  (stop-node {stop})')
        print(f'population : {args.pop_size} x {args.classes_per_model} classes')
        print(f'  pretrain        {pre:.2f} epoch-equiv')
        print(f'  per agent/gen   {args.individual_budget:.3f} epoch-equiv '
              f'= {per_agent} sample-presentations')
        print(f'  per generation  {per_gen:.2f} epoch-equiv ({args.pop_size} agents)')
        print(f'  -> {gens} generations from a budget of {args.budget:g}')
        print(f'population dir: {args.population_dir}')
    else:
        print(f'mode       : {args.baseline_mode}')
        print(f'  -> {args.baseline_epochs} epochs')
    print('=' * 70)


def _run_sesil(args, budget, data, logger, evaluator):
    from sesil.evolution import run_evolution
    from sesil.pretrain import ensure_population

    # Stage 1: pretrain. Skipped when a population is already on disk, unless
    # --force-pretrain. This is what used to be a manual copy step.
    if args.start_gen == 0:
        population = ensure_population(args, data, budget)
        if not population:
            raise RuntimeError(f'Pretrain produced no individuals in {args.population_dir}')

        # Whatever pretrain cost -- freshly trained or charged for reuse -- is
        # where the evolution curve begins. Recorded on every eval point so the
        # phase boundary can be drawn without reading anything else.
        evaluator.phase_start = budget.spent
        print(f'[main] pretrain phase cost {budget.spent:.2f} epoch-equiv; '
              f'evolution curve starts there.')
    else:
        print(f'[main] resuming from generation {args.start_gen}; skipping pretrain.')

    if args.pretrain_only:
        print('[main] --pretrain-only set; stopping after pretrain.')
        print(budget.report())
        return None

    # Stage 2: evolution, until the budget runs out.
    return run_evolution(args, budget, data, logger, evaluator)


if __name__ == '__main__':
    # Deliberately NOT sys.exit(main()): main() returns the generation count
    # for sesil and a (model, accuracy) tuple for baseline, and using either as
    # an exit status makes a successful run look like a failure to run.sh.
    # A normal return exits 0; an exception still propagates and exits non-zero.
    main()
