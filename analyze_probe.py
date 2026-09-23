"""Read the probe's pairs.jsonl and say whether GLOBA predicted good pairs.

    python analyze_probe.py results/globa_probe
    python analyze_probe.py results/globa_probe --outcome retention_total
    python analyze_probe.py results/globa_probe --per-seed

Every pair is measured, so the true ranking is known in full and there is no
random baseline to run: random is the middle of that ranking by construction,
and is printed as a reference line rather than a competitor.

The question is where each predictor's favourite pair lands in the true
ranking, and -- more importantly -- how much retention that choice gives up
against the best pair available. Rank alone can mislead: if every pair scores
about the same, picking last costs nothing and the whole prediction problem is
vacuous. The gap column is what distinguishes "GLOBA works" from "it did not
matter what GLOBA said".
"""

import argparse
import json
import os
from collections import defaultdict

from sesil.probe.driver import PREDICTORS, _per_pair, _spearman


def find_runs(root):
    """Every seed directory under `root` holding a probe result."""
    found = []
    for dirpath, _dirnames, filenames in os.walk(root):
        if 'pairs.jsonl' in filenames:
            found.append(dirpath)
    return sorted(found)


def load(run_dir):
    rows = []
    with open(os.path.join(run_dir, 'pairs.jsonl')) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def evaluate_run(rows, outcome):
    """Rank, gap and rho for every predictor on one seed."""
    pairs = _per_pair(rows)
    n = len(pairs)
    if n < 3:
        return None

    truth = sorted(pairs, key=lambda r: r[outcome], reverse=True)
    best = truth[0][outcome]
    worst = truth[-1][outcome]
    # What you would get on average by choosing blindly.
    expected_random = sum(r[outcome] for r in pairs) / n

    out = {
        'n_pairs': n,
        'best': best,
        'worst': worst,
        'spread': best - worst,
        'random': expected_random,
        'random_gap': best - expected_random,
        'predictors': {},
    }

    for name in PREDICTORS:
        if name not in pairs[0]:
            continue
        pick = max(pairs, key=lambda r: r[name])
        rank = next(i for i, r in enumerate(truth)
                    if r['pair'] == pick['pair']) + 1
        out['predictors'][name] = {
            'rank': rank,
            'value': pick[outcome],
            'gap': best - pick[outcome],
            'rho': _spearman([r[name] for r in pairs], [r[outcome] for r in pairs]),
            'pair': pick['pair'],
        }
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('root', help='Directory to search for pairs.jsonl.')
    ap.add_argument('--outcome', default='balanced',
                    help="What counts as a good merge. 'balanced' (default) is the "
                         "harmonic mean of the two parents' distinctive retention; "
                         "'retention_total' counts shared classes too; "
                         "'child_overall' ignores the parents entirely.")
    ap.add_argument('--per-seed', action='store_true',
                    help='Also print each seed separately.')
    args = ap.parse_args()

    runs = find_runs(args.root)
    if not runs:
        raise SystemExit(f'No pairs.jsonl found under {args.root!r}. '
                         f'Run: bash run_probe.sh --dataset cifar10 --seed 0')

    print(f'{len(runs)} run(s) under {args.root}')
    print(f'outcome: {args.outcome}\n')

    per_run = []
    for run_dir in runs:
        rows = load(run_dir)
        result = evaluate_run(rows, args.outcome)
        if result is None:
            print(f'  skipped {run_dir}: too few pairs')
            continue
        result['dir'] = run_dir
        per_run.append(result)

        if args.per_seed:
            print(f'--- {run_dir}  ({result["n_pairs"]} pairs, '
                  f'spread {result["spread"]:.4f})')
            _table(result['predictors'], result['n_pairs'],
                   result['best'], result['random'])
            print()

    if not per_run:
        raise SystemExit('Nothing to aggregate.')

    # Aggregate: mean rank, mean gap, mean rho across seeds.
    agg = defaultdict(lambda: {'rank': [], 'gap': [], 'rho': [], 'value': []})
    for result in per_run:
        for name, p in result['predictors'].items():
            for key in ('rank', 'gap', 'rho', 'value'):
                agg[name][key].append(p[key])

    n_pairs = per_run[0]['n_pairs']
    mean_best = sum(r['best'] for r in per_run) / len(per_run)
    mean_random = sum(r['random'] for r in per_run) / len(per_run)
    mean_spread = sum(r['spread'] for r in per_run) / len(per_run)

    print('=' * 78)
    print(f'ACROSS {len(per_run)} SEED(S)')
    print(f'best pair averages {mean_best:.4f}; blind choice averages '
          f'{mean_random:.4f}; spread {mean_spread:.4f}')
    print('=' * 78)

    if mean_spread < 0.02:
        print('WARNING: pairs barely differ, so partner choice is almost')
        print('irrelevant here and no predictor can succeed or fail for a real')
        print('reason. Increase --pop-size or --classes-per-model so that pairs')
        print('genuinely differ before reading anything into the table below.')
        print('=' * 78)

    summary = {name: {key: sum(v[key]) / len(v[key]) for key in v}
               for name, v in agg.items()}
    _table(summary, n_pairs, mean_best, mean_random, aggregated=True)

    print()
    _verdict(summary, mean_best, mean_random, n_pairs)


def _table(predictors, n_pairs, best, random_value, aggregated=False):
    label = 'mean rank' if aggregated else 'rank'
    print(f'{"predictor":<20} {label:>10} {"retention":>10} {"gap":>8} {"rho":>7}')
    print('-' * 78)

    order = sorted(predictors.items(), key=lambda kv: kv[1]['gap'])
    printed_random = False
    for name, p in order:
        # The blind-choice reference sits wherever its value falls.
        if not printed_random and p['value'] < random_value:
            print(f'{"  (blind choice)":<20} {n_pairs / 2:>10.1f} '
                  f'{random_value:>10.4f} {best - random_value:>8.4f} '
                  f'{0.0:>7.3f}')
            printed_random = True
        mark = '  <- pre-registered' if name == 'globa_score' else ''
        mark = '  <- SESiL incumbent' if name == 'mating_score' else mark
        print(f'{name:<20} {p["rank"]:>10.1f} {p["value"]:>10.4f} '
              f'{p["gap"]:>8.4f} {p["rho"]:>7.3f}{mark}')
    if not printed_random:
        print(f'{"  (blind choice)":<20} {n_pairs / 2:>10.1f} '
              f'{random_value:>10.4f} {best - random_value:>8.4f} {0.0:>7.3f}')


def _verdict(summary, best, random_value, n_pairs):
    """State the conclusion in the terms the experiment was set up to answer."""
    globa = summary.get('globa_score')
    incumbent = summary.get('mating_score')
    if globa is None:
        return

    blind_gap = best - random_value
    print('VERDICT')
    print('-' * 78)
    print(f'GLOBA picked pair {globa["rank"]:.1f} of {n_pairs} on average, '
          f'giving up {globa["gap"]:.4f} retention.')
    print(f'Choosing blindly would give up {blind_gap:.4f}.')

    if globa['gap'] < blind_gap * 0.5:
        print('-> GLOBA beats blind choice clearly.')
    elif globa['gap'] < blind_gap:
        print('-> GLOBA beats blind choice, but not by much.')
    else:
        print('-> GLOBA is no better than choosing blindly.')

    if incumbent is not None:
        print(f"SESiL's existing rule gives up {incumbent['gap']:.4f}.")
        if globa['gap'] < incumbent['gap']:
            print('-> GLOBA beats it: worth adopting for mate selection.')
        elif globa['gap'] > incumbent['gap']:
            print('-> GLOBA loses to it: real signal or not, we already have '
                  'something better.')
        else:
            print('-> They tie.')

    print()
    print('Reminder: GLOBA reads weights, the merger aligns activations. These')
    print('may simply not see the same notion of compatibility -- a negative')
    print('result here is informative, not a bug.')


if __name__ == '__main__':
    main()
