"""
Single entry point for every method.

    python main.py --method sesil --merger permute --seed 0 --generations 25
    python main.py --method baseline --baseline-epochs 100

Normally invoked through run.sh / run_sesil.sh / run_baseline.sh rather than
directly.

For --method sesil this runs the whole pipeline: it creates the initial
population if one does not already exist, then evolves it.  Nothing has to be
copied into place by hand.
"""

import json
import os
import sys

from config import get_config, resolve
from utils import set_seed


def main(argv=None):
    args = resolve(get_config().parse_args(argv))

    set_seed(args.seed)

    print('=' * 70)
    print(f'method     : {args.method}')
    print(f'seed       : {args.seed}')
    print(f'device     : {args.device}')
    print(f'dataset    : {args.dataset} ({args.num_classes} classes)')
    print(f'model      : {args.arch}')
    print(f'run dir    : {args.run_dir}')
    if args.method == 'sesil':
        print(f'merger     : {args.merger}')
        print(f'selection  : {args.selection}')
        print(f'budget     : {args.generations} generations from gen {args.start_gen}')
        print(f'population : {args.population_dir}')
    else:
        print(f'mode       : {args.baseline_mode}')
        print(f'budget     : {args.baseline_epochs} epochs')
    print('=' * 70)

    os.makedirs(args.run_dir, exist_ok=True)
    with open(os.path.join(args.run_dir, 'config.json'), 'w') as f:
        json.dump(vars(args), f, indent=2, default=str)

    if args.method == 'sesil':
        return _run_sesil(args)
    if args.method == 'baseline':
        from sesil.baseline import run_baseline
        return run_baseline(args)

    raise ValueError(f'Unknown method {args.method!r}')


def _run_sesil(args):
    from sesil.evolution import run_evolution
    from sesil.pretrain import ensure_population

    # Stage 1: pretrain. Skipped when a population is already on disk, unless
    # --force-pretrain. This is what used to be a manual copy step.
    if args.start_gen == 0:
        population = ensure_population(args)
        if not population:
            raise RuntimeError(f'Pretrain produced no individuals in {args.population_dir}')
    else:
        print(f'[main] resuming from generation {args.start_gen}; skipping pretrain.')

    if args.pretrain_only:
        print('[main] --pretrain-only set; stopping after pretrain.')
        return None

    # Stage 2: evolution.
    return run_evolution(args)


if __name__ == '__main__':
    sys.exit(main() or 0)
