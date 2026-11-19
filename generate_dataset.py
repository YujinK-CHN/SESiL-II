"""
Dataset generator entry point: a copy of main.py that runs datagen/evolution.py
(a copy of sesil/evolution.py that also writes <run_dir>/samples.csv) in place
of sesil/evolution.py. GLOBA mating and GLOBA merging only; the baseline,
probe and phase-C paths of main.py are removed. Launched by
generate_dataset.sh -> submit.sh (or submit_trillium.sh) -> run.sh ->
run_datagen.sh.

--resume (generator only; see datagen/resume.py): continue this run from its
latest complete generation, do nothing if it already finished, or start fresh
if nothing usable is on disk. Lets one run span a chain of time-limited jobs.

The original main.py docstring follows.

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
    # [datagen] --resume is ours, not config.py's.
    argv = list(sys.argv[1:] if argv is None else argv)
    resume = '--resume' in argv
    argv = [a for a in argv if a != '--resume']
    args = resolve(get_config().parse_args(argv))
    # [datagen] the generator is defined for GLOBA mating + GLOBA merging only.
    if (args.method != 'sesil' or args.mating_mode != 'globa'
            or args.merger != 'globa'
            or args.globa_with not in ('D_minus', 'D_plus', 'E')):
        raise SystemExit('[datagen] needs --method sesil --mating-mode globa '
                         '--merger globa --globa-with D_minus|D_plus|E')

    state = None
    args.datagen_resumed_from = None
    if resume:
        from datagen import resume as _resume
        state = _resume.plan(args)
        if state == 'done':
            print(f'[resume] {args.run_dir} already finished (samples.csv '
                  f'exists); nothing to do.')
            return None
        if state is not None:
            args.start_gen = state['generation']
            args.datagen_resumed_from = state['generation']
        else:
            print('[resume] nothing usable on disk; starting fresh.')
            _resume.set_aside(args)

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
        log_predictions=getattr(args, 'log_predictions', False),
        run_dir=args.run_dir,
        # Carried on every eval record so plot_results can separate variants
        # without anyone having to remember which directory was which.
        # Carried on every eval record so plot_results can tell one arm from
        # another. Anything NOT here is invisible to the plot, and two runs
        # that differ only by a missing field silently merge into one series
        # and get averaged as though they were seeds of the same condition --
        # which is exactly how a comparison experiment produces a flat result
        # for the wrong reason. Everything that defines a SESiL-II arm belongs
        # in this dict.
        meta={'method': args.method, 'seed': args.seed, 'dataset': args.dataset,
              'merger': args.merger if args.method == 'sesil' else None,
              'pretrain_mode': args.pretrain_mode if args.method == 'sesil' else None,
              'phase_a_method': (args.phase_a_method
                                 if args.method == 'sesil'
                                 and args.pretrain_mode == 'ssl' else None),
              'mating_mode': args.mating_mode if args.method == 'sesil' else None,
              'cert_with': args.cert_with if args.method == 'sesil' else None,
              'globa_with': (args.globa_with if args.method == 'sesil'
                             and args.mating_mode in ('globa', 'hybrid') else None),
              'mutation_mode': args.mutation_mode if args.method == 'sesil' else None,
              'merge_head': args.merge_head if args.method == 'sesil' else None,
              'globa_preset': (args.globa_preset if args.method == 'sesil'
                               and args.merger == 'globa' else None),
              'subset_mode': args.subset_mode if args.method == 'sesil' else None,
              'mating_rounds': args.mating_rounds if args.method == 'sesil' else None,
              'mate_floor': args.mate_floor if args.method == 'sesil' else None,
              # The model's folder name: mating_models/<name>/pair_quality.pt.
              'mating_model': (os.path.basename(os.path.dirname(
                                   os.path.abspath(args.mating_model)))
                               if args.method == 'sesil' and args.mating_model
                               else None),
              'learn_pairing': (args.learn_pairing
                                if args.method == 'sesil'
                                and args.mating_mode == 'learn' else None),
              'phase_c_rank_weight': (args.phase_c_rank_weight
                                      if args.method == 'sesil'
                                      and args.mating_mode == 'learn' else None),
              # Numeric, but every bit as arm-defining as the modes above, and
              # they were missing: two rounds differing only in
              # --individual-budget carried identical meta, so plot_results
              # grouped them into ONE series and averaged six runs of two
              # different conditions into a single curve. The only visible sign
              # was a budget-grid note.
              'individual_budget': (args.individual_budget
                                    if args.method == 'sesil' else None),
              'certify_top_frac': (args.certify_top_frac
                                   if args.method == 'sesil' else None),
              'baseline_init': (args.baseline_init
                                if args.method == 'baseline' else None),
              # A curriculum baseline is defined by the run it replays -- two
              # baselines from different rounds are different conditions even
              # though every other field matches. Grouping needs this; the
              # legend does not, so plot_results uses it as a key only.
              'baseline_mode': (args.baseline_mode
                                if args.method == 'baseline' else None),
              'curriculum_from': (args.curriculum_from
                                  if args.method == 'baseline'
                                  and args.baseline_mode == 'curriculum'
                                  else None)},
    )

    if state is not None:
        _resume.apply(state, args, budget, evaluator, logger)   # [datagen]

    _print_banner(args, data, budget)

    try:
        result = _run_sesil(args, budget, data, logger, evaluator)
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

    pre = estimate_pretrain_cost(args)
    per_gen = estimate_generation_cost(args)
    gens = estimate_generations(args)
    per_agent = samples_for(args.individual_budget, budget.train_set_size)
    stop = args.stop_node if args.stop_node is not None else 'none / full merge'
    print(f'merger     : {args.merger}  (stop-node {stop})')
    print(f'population : {args.pop_size} x {args.classes_per_model} classes')
    from sesil.pretrain import cost_multiplier, phase_budgets
    from sesil.ssl import uses_labels
    phase_a, phase_b = phase_budgets(args)
    print(f'pretrain   : mode {args.pretrain_mode}, {pre:.2f} epoch-equiv total')
    if phase_a <= 0:
        print(f'    (no phase A -- every agent trained from scratch on its '
              f'own subset, {phase_b / max(args.pop_size, 1):.3f} each)')
    else:
        mult = cost_multiplier(args)
        labels = 'uses labels' if uses_labels(args.phase_a_method) else 'label-free'
        print(f'    phase A       {phase_a:.2f}  shared backbone: '
              f'{args.phase_a_method} ({labels}), cost x{mult:g}')
        print(f'    phase B       {phase_b:.2f}  '
              f'{phase_b / max(args.pop_size, 1):.3f} per agent')
    print(f'  per agent/gen   {args.individual_budget:.3f} epoch-equiv '
          f'= {per_agent} sample-presentations')
    print(f'  per generation  {per_gen:.2f} epoch-equiv ({args.pop_size} agents)')
    print(f'  -> {gens} generations from a budget of {args.budget:g}')
    print('=' * 70)


def _run_sesil(args, budget, data, logger, evaluator):
    from datagen.evolution import run_evolution   # [datagen]
    from sesil.pretrain import build_population

    # Stage 1: pretrain. Always runs -- nothing is inherited from a previous
    # run, so every run's numbers describe only itself.
    # [datagen] a resumed run skips pretrain even when resuming at gen 0.
    if args.start_gen == 0 and args.datagen_resumed_from is None:
        population = build_population(args, data, budget, logger)
        if not population:
            raise RuntimeError(f'Pretrain produced no individuals in {args.population_dir}')

        # What pretrain cost is where the evolution curve begins. Recorded on every eval point so the
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
