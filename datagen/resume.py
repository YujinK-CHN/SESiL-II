"""
Resuming a dataset-generation run across job time limits (generate_dataset.py --resume).

A cluster with a 24 h limit (Trillium) cannot finish one run in one job, so the
same command is submitted as a chain of jobs, each with --resume:

  * run already finished (samples.csv exists)  -> the job does nothing;
  * nothing usable on disk                     -> a fresh start, pretrain
    included; whatever an earlier attempt left (a pretrain cut short) is
    moved aside to <run_dir>.incomplete-<time>, not deleted;
  * otherwise                                  -> continue from the latest
    COMPLETE generation, with the budget, the evaluator's watermark and
    samples.partial.csv put back to where that generation began.

"Complete" is read from the run's own log, not from the checkpoint folders: a
job killed while saving gen_N+1 can leave a folder that mixes two attempts, but
the 'society' record of generation N is written only after every agent of
gen_N+1 is saved. Generation 0 is complete once its population was evaluated.
Its weights must also still be there: --keep-generations prunes old weights
but keeps the folders (and agent.json), so a folder alone proves nothing.
Folders above N are partial work of the killed job; they are cleared, since a
redone generation may breed fewer children and would otherwise inherit stale
agents.

The random state is not restored: a resumed run is a valid run of the same
configuration, not a bit-identical continuation.
"""

import csv
import json
import os
import shutil
import time


def _records(path):
    if not os.path.exists(path):
        return []
    out = []
    with open(path, encoding='utf-8') as f:
        for line in f:
            try:
                out.append(json.loads(line))
            except ValueError:
                pass            # a line cut off by the kill
    return out


def _has_weights(gen_dir):
    """Every agent of the generation still has its weights (not just agent.json)."""
    from sesil.population import AGENT_META, list_population
    agents = list_population(gen_dir)
    return bool(agents) and all(
        any(e != AGENT_META for e in os.listdir(os.path.join(gen_dir, a)))
        for a in agents)


def plan(args):
    """'done', None (fresh start), or {'generation', 'spent', ...}."""
    from datagen.evolution import _samples_paths
    from sesil.population import generation_dir

    partial, final = _samples_paths(args)
    if os.path.exists(final):
        return 'done'

    train = _records(os.path.join(args.run_dir, 'train.jsonl'))
    gen0 = [r['budget'] for r in train
            if r.get('stage') == 'population' and r.get('generation') == 0]
    society = {r['generation']: r['budget'] for r in train
               if r.get('stage') == 'society'}

    candidates = {}
    if gen0:
        candidates[0] = min(gen0)
    for g, spent in society.items():
        candidates[g + 1] = spent
    # The weights must still be on disk (--keep-generations prunes old ones).
    candidates = {g: s for g, s in candidates.items()
                  if _has_weights(generation_dir(args, g))}
    if not candidates:
        return None

    n = max(candidates)
    evals = _records(os.path.join(args.run_dir, 'eval.jsonl'))
    return {
        'generation': n,
        'spent': float(candidates[n]),
        'phase_start': float(evals[-1].get('phase_start', 0.0)) if evals else None,
        'last_eval_budget': float(evals[-1]['budget']) if evals else None,
        'n_points': len(evals),
        'partial': partial,
    }


def apply(state, args, budget, evaluator, logger):
    """Put the budget, evaluator and samples back to the start of generation N."""
    n = state['generation']
    budget.spent = state['spent']
    budget.by_stage['before_resume'] = state['spent']

    if state['phase_start'] is not None:
        evaluator.phase_start = state['phase_start']
    if state['last_eval_budget'] is not None:
        # The next watermark after the last point already recorded.
        step = evaluator.eval_interval
        evaluator.next_watermark = 0.0
        while evaluator.next_watermark <= state['last_eval_budget'] + 1e-9:
            evaluator.next_watermark += step
        evaluator.n_points = state['n_points']

    dropped = drop_samples_from(state['partial'], n)
    cleared = clear_generations_after(args, n)
    logger.log_train({'stage': 'resume', 'generation': n,
                      'budget': round(budget.spent, 4),
                      'dropped_samples': dropped,
                      'cleared_generations': cleared})
    print(f'[resume] from generation {n}: budget {budget.spent:.4f} spent, '
          f'{dropped} sample row(s) of generation >= {n} dropped'
          + (f', partial generation(s) {cleared} cleared' if cleared else ''))


def clear_generations_after(args, generation):
    """Delete checkpoints/gen_K for K > generation (the killed job's partial work)."""
    root = os.path.join(args.run_dir, 'checkpoints')
    cleared = []
    if not os.path.isdir(root):
        return cleared
    for name in sorted(os.listdir(root)):
        try:
            k = int(name[len('gen_'):]) if name.startswith('gen_') else None
        except ValueError:
            k = None
        if k is not None and k > generation:
            shutil.rmtree(os.path.join(root, name))
            cleared.append(k)
    return sorted(cleared)


def set_aside(args):
    """Before a fresh start: move an earlier attempt's leftovers out of the way."""
    if not os.path.isdir(args.run_dir) or not os.listdir(args.run_dir):
        return None
    dest = f"{args.run_dir.rstrip(os.sep)}.incomplete-{time.strftime('%Y%m%d-%H%M%S')}"
    os.replace(args.run_dir, dest)
    print(f'[resume] earlier attempt had no usable generation; moved it to {dest}')
    return dest


def drop_samples_from(partial, generation):
    """Remove rows of `generation` and later (redone by this job)."""
    if not os.path.exists(partial):
        return 0
    with open(partial, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames
        rows = list(reader)
    # A row cut off by the kill has an empty last column ('head'); its
    # 'generation' cell may itself be truncated, so it is never kept.
    keep = [r for r in rows if r.get('head') not in (None, '')
            and int(r['generation']) < generation]
    if len(keep) == len(rows):
        return 0
    with open(partial + '.tmp', 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(keep)
    os.replace(partial + '.tmp', partial)
    return len(rows) - len(keep)
