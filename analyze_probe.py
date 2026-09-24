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
import random
from collections import defaultdict

from sesil.probe.driver import (PREDICTORS, _per_pair, _spearman,
                                default_outcome)


def permutation_p(xs, ys, n_perm=10000, seed=0):
    """How often chance alone beats the observed rank correlation.

    A predictor can post a respectable rho and still be worthless: with 28
    pairs, |rho| below about 0.38 is ordinary noise. Shuffling one side and
    counting how often chance does at least as well puts a number on that
    without pulling in scipy, and without relying on a normal approximation
    that is poor at this sample size.
    """
    observed = abs(_spearman(xs, ys))
    rng = random.Random(seed)
    shuffled = list(ys)
    hits = 0
    for _ in range(n_perm):
        rng.shuffle(shuffled)
        if abs(_spearman(xs, shuffled)) >= observed:
            hits += 1
    return (hits + 1) / (n_perm + 1)


def _rank_normalise(values):
    """Map values to evenly spaced ranks in [0, 1], ties broken by order.

    Seeds differ in the absolute scale of both predictor and outcome -- one
    population is simply better than another -- so only the ordering within a
    seed carries meaning. Rank-normalising before pooling keeps a seed with a
    wide spread from dominating one with a narrow one.
    """
    n = len(values)
    if n < 2:
        return [0.5] * n
    order = sorted(range(n), key=lambda i: values[i])
    out = [0.0] * n
    for position, index in enumerate(order):
        out[index] = position / (n - 1)
    return out


def pool_across_seeds(runs, outcome, n_perm=5000):
    """Rank correlation and significance over every pair from every seed.

    This is where a real but modest effect becomes visible: rho ~ 0.3 on 28
    pairs is indistinguishable from chance, while the same rho on 84 is not.
    """
    pooled = {}
    per_seed = [_per_pair(load(run_dir)) for run_dir in runs]
    per_seed = [pairs for pairs in per_seed if len(pairs) >= 3]
    if not per_seed:
        return pooled

    for name in PREDICTORS:
        if name not in per_seed[0][0]:
            continue
        xs, ys, rhos = [], [], []
        for pairs in per_seed:
            values = [r[name] for r in pairs]
            outs = [r[outcome] for r in pairs]
            xs += _rank_normalise(values)
            ys += _rank_normalise(outs)
            rhos.append(_spearman(values, outs))
        pooled[name] = {
            'rho': _spearman(xs, ys),
            'p': permutation_p(xs, ys, n_perm=n_perm),
            'n': len(xs),
            # Agreement across seeds is the other half of the evidence: a
            # correlation that changes sign between seeds is not a finding
            # however significant the pooled number looks.
            'consistent': all(r > 0 for r in rhos) or all(r < 0 for r in rhos),
            'per_seed_rho': rhos,
        }
    return pooled


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

    # Results written before 'own'/'transfer' existed still carry everything
    # needed to derive them, so old runs stay readable without a re-run.
    for row in rows:
        if 'transfer' not in row:
            own_is_a = row['child_index'] == 0
            row['own'] = row['distinctive_a'] if own_is_a else row['distinctive_b']
            row['transfer'] = row['distinctive_b'] if own_is_a else row['distinctive_a']
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

    outcomes = [r[outcome] for r in pairs]
    for name in PREDICTORS:
        if name not in pairs[0]:
            continue
        values = [r[name] for r in pairs]
        pick = max(pairs, key=lambda r: r[name])
        rank = next(i for i, r in enumerate(truth)
                    if r['pair'] == pick['pair']) + 1
        out['predictors'][name] = {
            'rank': rank,
            'value': pick[outcome],
            'gap': best - pick[outcome],
            'rho': _spearman(values, outcomes),
            'p': permutation_p(values, outcomes),
            'range': max(values) - min(values),
            'pair': pick['pair'],
        }
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('root', help='Directory to search for pairs.jsonl.')
    ap.add_argument('--outcome', default=None,
                    help='What counts as a good merge. Chosen automatically from '
                         'the merge geometry when omitted: a symmetric full merge '
                         "uses 'balanced' (one child has to carry both parents), "
                         "an asymmetric one uses 'transfer' (each child keeps its "
                         'own parent by construction, so only what it picks up '
                         "from the other is earned). Override with 'balanced', "
                         "'transfer', 'own', 'retention_total' or 'child_overall'.")
    ap.add_argument('--per-seed', action='store_true',
                    help='Also print each seed separately.')
    args = ap.parse_args()

    runs = find_runs(args.root)
    if not runs:
        raise SystemExit(f'No pairs.jsonl found under {args.root!r}. '
                         f'Run: bash run_probe.sh --dataset cifar10 --seed 0')

    # The right outcome depends on how the children were built, which is
    # recorded in the rows rather than assumed here.
    outcome = args.outcome or default_outcome(_per_pair(load(runs[0])))

    print(f'{len(runs)} run(s) under {args.root}')
    print(f'outcome: {outcome}'
          + ('' if args.outcome else '  (chosen from the merge geometry)') + '\n')

    per_run = []
    for run_dir in runs:
        rows = load(run_dir)
        result = evaluate_run(rows, outcome)
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
    keys = ('rank', 'gap', 'rho', 'value', 'p', 'range')
    agg = defaultdict(lambda: {k: [] for k in keys})
    for result in per_run:
        for name, p in result['predictors'].items():
            for key in keys:
                agg[name][key].append(p[key])

    # Significance has to be computed on the pooled data, not averaged.
    #
    # The mean of several p-values is not a p-value: three seeds each at p=0.12
    # average to 0.12 and look like noise, when the same effect appearing three
    # times in the same direction is strong evidence. Averaging them threw away
    # exactly the agreement that three seeds were run to establish.
    #
    # Pairs are pooled by rank-normalising within each seed first, since the
    # absolute scale of both predictor and outcome shifts from seed to seed and
    # only the ordering is comparable.
    pooled = pool_across_seeds(runs, outcome)

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
    # Replace the per-seed averages with the pooled statistics, which are the
    # ones that mean anything.
    for name, stats in summary.items():
        if name in pooled:
            stats['rho'] = pooled[name]['rho']
            stats['p'] = pooled[name]['p']
            stats['consistent'] = pooled[name]['consistent']
            stats['per_seed_rho'] = pooled[name]['per_seed_rho']
    _table(summary, n_pairs, mean_best, mean_random, aggregated=True,
           n_pooled=pooled[next(iter(pooled))]['n'] if pooled else 0)

    print()
    _verdict(summary, mean_best, mean_random, n_pairs)


def _table(predictors, n_pairs, best, random_value, aggregated=False,
           n_pooled=0):
    label = 'mean rank' if aggregated else 'rank'
    if n_pooled:
        print(f'rho and p are POOLED over all {n_pooled} pairs from every seed; '
              f'rank and gap are per-seed means.')
    print(f'{"predictor":<20} {label:>9} {"outcome":>8} {"gap":>7} '
          f'{"rho":>7} {"p":>7} {"range":>8}')
    print('-' * 78)

    order = sorted(predictors.items(), key=lambda kv: kv[1]['gap'])
    printed_random = False
    for name, p in order:
        # The blind-choice reference sits wherever its value falls.
        if not printed_random and p['value'] < random_value:
            print(f'{"  (blind choice)":<20} {n_pairs / 2:>9.1f} '
                  f'{random_value:>8.4f} {best - random_value:>7.4f}')
            printed_random = True
        mark = ''
        if name == 'globa_score':
            mark = '  <- pre-registered'
        elif name == 'mating_score':
            mark = '  <- SESiL incumbent'
        # A predictor whose correlation is indistinguishable from chance has
        # not ranked anything, however good its rho looks.
        flag = '' if p['p'] < 0.05 else '  (ns)'
        if p['p'] < 0.05 and not p.get('consistent', True):
            flag = '  (flips)'
        print(f'{name:<20} {p["rank"]:>9.1f} {p["value"]:>8.4f} '
              f'{p["gap"]:>7.4f} {p["rho"]:>7.3f} {p["p"]:>7.3f} '
              f'{p["range"]:>8.4f}{flag}{mark}')
    if not printed_random:
        print(f'{"  (blind choice)":<20} {n_pairs / 2:>9.1f} '
              f'{random_value:>8.4f} {best - random_value:>7.4f}')
    print()
    print('(ns)    = correlation not distinguishable from chance (p >= 0.05).')
    print('(flips) = significant overall but changes sign between seeds -- not a finding.')
    print('range = how much the predictor varies across pairs; near zero means it')
    print('        returns the same number for everything and cannot rank at all.')


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
          f'getting {globa["value"]:.4f} and giving up {globa["gap"]:.4f}.')
    print(f'Choosing blindly gets {random_value:.4f}, giving up {blind_gap:.4f}.')

    if globa['gap'] < blind_gap * 0.5:
        print('-> GLOBA beats blind choice clearly.')
    elif globa['gap'] < blind_gap:
        print('-> GLOBA beats blind choice, but not by much.')
    else:
        print('-> GLOBA is no better than choosing blindly.')

    if incumbent is not None:
        print(f"SESiL's existing rule gives up {incumbent['gap']:.4f}.")
        if globa['gap'] < incumbent['gap']:
            print('-> GLOBA beats it.')
        elif globa['gap'] > incumbent['gap']:
            print('-> GLOBA loses to it.')
        else:
            print('-> They tie.')

    # Whether that verdict is worth anything at all.
    print()
    print('IS THIS CONCLUSIVE?')
    print('-' * 78)
    noisy = globa['p'] >= 0.05
    consistent = globa.get('consistent', True)

    if noisy:
        print(f'NOT YET. GLOBA\'s correlation (rho {globa["rho"]:+.3f}, '
              f'p {globa["p"]:.3f}) is within')
        print('chance even pooled across seeds. More seeds would settle it.')
        if globa['range'] < 0.1:
            print()
            print(f'Note its score spans only {globa["range"]:.4f} across pairs '
                  f'-- it returns nearly')
            print('the same number for every couple, which is the likely reason. '
                  'Agents all')
            print('finetuned from ONE backbone have task vectors pointing the '
                  'same way.')
    elif not consistent:
        rhos = ', '.join(f'{r:+.2f}' for r in globa.get('per_seed_rho', []))
        print(f'NO. The pooled correlation is significant but its SIGN changes '
              f'between')
        print(f'seeds ({rhos}), so it is not a stable effect.')
    elif globa['rho'] > 0:
        print(f'YES. GLOBA predicts merge quality reliably '
              f'(rho {globa["rho"]:+.3f}, p {globa["p"]:.3f},')
        print('same sign in every seed).')
    else:
        print(f'YES, BUT BACKWARDS. GLOBA correlates reliably with merge quality '
              f'(rho')
        print(f'{globa["rho"]:+.3f}, p {globa["p"]:.3f}, same sign in every seed) '
              f'-- in the WRONG direction.')
        print('A HIGHER GLOBA score means a WORSE pair, so taking its argmax, as '
              'the')
        print('pre-registered rule does, actively selects bad partners.')
        print()
        print('The signal is real. The hypothesis behind it (orthogonal parents '
              'merge')
        print('well, conflicting ones merge badly) is inverted. Re-run the pick '
              'on')
        print('MINIMUM globa_score before concluding it is useless.')

    print()
    print('Reminder: GLOBA reads weights, the merger aligns activations. These')
    print('may simply not see the same notion of compatibility -- a negative')
    print('result here is informative, not a bug.')


if __name__ == '__main__':
    main()
