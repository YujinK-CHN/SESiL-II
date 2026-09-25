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


def pool_across_seeds(runs, outcome, n_perm=5000, operator='sesil'):
    """Rank correlation and significance over every pair from every seed.

    This is where a real but modest effect becomes visible: rho ~ 0.3 on 28
    pairs is indistinguishable from chance, while the same rho on 84 is not.
    """
    pooled = {}
    per_seed = [_per_pair(only(load(run_dir), operator)) for run_dir in runs]
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


TYPES = ('A', 'B', 'C', 'E', 'D_plus', 'D_minus')


# --------------------------------------------------------------------------- #
# Extra data: certificates (from train.jsonl) and per-layer stats (layers.jsonl)
# --------------------------------------------------------------------------- #

def certificates(run_dir):
    """Each agent's certified classes, as the probe recorded them."""
    out = {}
    path = os.path.join(run_dir, 'train.jsonl')
    if not os.path.exists(path):
        return out
    with open(path) as f:
        for line in f:
            row = json.loads(line)
            if row.get('stage') == 'probe_parent':
                out[row['agent']] = set(row['certificate'])
    return out


def with_certificates(run_dir, pairs):
    """Annotate each pair with its parents' certificate overlap and union.

    Overlap is the direct measure of the consanguinity risk: a selector that
    prefers overlapping parents is mating near-identical agents, and the
    offspring's inherited certificate is the UNION, so overlap wastes the merge.
    """
    certs = certificates(run_dir)
    for pair in pairs:
        a, b = pair['pair']
        if a in certs and b in certs:
            pair['cert_overlap'] = len(certs[a] & certs[b])
            pair['cert_union'] = len(certs[a] | certs[b])
    return pairs


def layer_rows(run_dir):
    """Per-layer GLOBA statistics, grouped by pair. {} when not recorded."""
    path = os.path.join(run_dir, 'layers.jsonl')
    if not os.path.exists(path):
        return {}
    grouped = defaultdict(list)
    with open(path) as f:
        for line in f:
            row = json.loads(line)
            grouped[tuple(row['pair'])].append(row)
    return grouped


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


def operators_present(runs):
    """Which merge operators produced children in these runs."""
    seen = set()
    for run_dir in runs:
        for row in load(run_dir):
            seen.add(row.get('operator', 'sesil'))
    return sorted(seen)


def only(rows, operator):
    """Rows from one merge operator.

    Mixing operators would average two different merges into one correlation,
    which is meaningless -- the point of running both is to compare them, not
    to blend them.
    """
    return [r for r in rows if r.get('operator', 'sesil') == operator]


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


# --------------------------------------------------------------------------- #
# --strategies
# --------------------------------------------------------------------------- #

def _tied_at_top(pairs, key='mating_score'):
    """The pairs sharing the highest value of `key`.

    mating_score counts complementary certified classes, so it saturates: with
    k classes per agent every fully-disjoint couple scores the same k, and in
    practice about half the population ties for first. Taking its argmax is
    therefore an arbitrary choice among that tied set, not a prediction -- which
    is why the rule can correlate strongly and still pick badly.
    """
    top = max(p[key] for p in pairs)
    return [p for p in pairs if abs(p[key] - top) < 1e-9]


STRATEGIES = (
    ('blind (random pair)', None),
    ('mating_score only', lambda ps: max(ps, key=lambda p: p['mating_score'])),
    ('D- only', lambda ps: max(ps, key=lambda p: p['energy_D_minus'])),
    ('E only', lambda ps: max(ps, key=lambda p: p['energy_E'])),
    ('globa_score MAX (as posed)', lambda ps: max(ps, key=lambda p: p['globa_score'])),
    ('globa_score MIN (inverted)', lambda ps: min(ps, key=lambda p: p['globa_score'])),
    ('mating_score + D- tiebreak',
     lambda ps: max(_tied_at_top(ps), key=lambda p: p['energy_D_minus'])),
    ('mating_score + E tiebreak',
     lambda ps: max(_tied_at_top(ps), key=lambda p: p['energy_E'])),
    ('best possible (oracle)', None),
)


def report_strategies(runs, outcome, operator='sesil'):
    """Compare whole selection rules, not just single predictors.

    A predictor is judged by its correlation; a strategy is judged by the pair
    it actually chooses. The two can disagree sharply, so this table reports the
    decision rather than the ranking, alongside the certificate overlap of the
    chosen pair -- a rule that picks overlapping parents is mating cousins
    however good its outcome looks.
    """
    per_seed = [with_certificates(r, _per_pair(only(load(r), operator)))
                for r in runs]
    per_seed = [p for p in per_seed if len(p) >= 3]
    if not per_seed:
        return

    blind = [sum(p[outcome] for p in ps) / len(ps) for ps in per_seed]
    best = [max(p[outcome] for p in ps) for ps in per_seed]
    has_certs = 'cert_overlap' in per_seed[0][0]

    print()
    print('=' * 78)
    print('SELECTION STRATEGIES -- what each rule actually picks')
    print('=' * 78)
    head = f'{"strategy":<30}{"mean":>8}{"closes":>9}{"wins":>7}'
    if has_certs:
        head += f'{"overlap":>9}{"union":>7}'
    print(head)
    print('-' * 78)

    for label, rule in STRATEGIES:
        if rule is None:
            picks = blind if label.startswith('blind') else best
            chosen = None
        else:
            chosen = [rule(ps) for ps in per_seed]
            picks = [p[outcome] for p in chosen]

        closes = sum((v - b) / (c - b) if c > b else 0.0
                     for v, b, c in zip(picks, blind, best)) / len(picks)
        wins = sum(1 for v, b in zip(picks, blind) if v > b)
        line = (f'{label:<30}{sum(picks) / len(picks):>8.4f}{closes:>+9.1%}'
                f'{(str(wins) + "/" + str(len(picks))) if chosen else "":>7}')
        if has_certs and chosen:
            ov = sum(p.get('cert_overlap', 0) for p in chosen) / len(chosen)
            un = sum(p.get('cert_union', 0) for p in chosen) / len(chosen)
            line += f'{ov:>9.2f}{un:>7.2f}'
        elif has_certs:
            everyone = [p for ps in per_seed for p in ps]
            if label.startswith('blind'):
                ov = sum(p.get('cert_overlap', 0) for p in everyone) / len(everyone)
                un = sum(p.get('cert_union', 0) for p in everyone) / len(everyone)
                line += f'{ov:>9.2f}{un:>7.2f}'
        print(line)

    if has_certs:
        print()
        print('Is complementarity itself predictive, and does each signal seek it?')
        print(f'  {"":<22}{"rho vs outcome":>16}{"rho vs overlap":>16}')
        for name in ('cert_union', 'cert_overlap', 'mating_score',
                     'energy_D_minus', 'energy_E', 'energy_D_plus',
                     'cosine', 'globa_score'):
            if name not in per_seed[0][0]:
                continue
            cols = []
            for target in (outcome, 'cert_overlap'):
                xs, ys = [], []
                for ps in per_seed:
                    xs += _rank_normalise([p[name] for p in ps])
                    ys += _rank_normalise([p[target] for p in ps])
                cols.append(_spearman(xs, ys))
            print(f'  {name:<22}{cols[0]:>+16.3f}{cols[1]:>+16.3f}')
        print()
        print('  overlap = certified classes BOTH parents already hold. A signal')
        print('  with positive rho vs overlap selects redundant parents.')


# --------------------------------------------------------------------------- #
# --types
# --------------------------------------------------------------------------- #

def report_types(runs, outcome, operator='sesil'):
    """How the six GLOBA types are distributed, and what each one predicts.

    The six are fractions of one whole and sum to 1, so they are not
    independent: rewarding five of them is arithmetically the same as punishing
    the sixth. Any composite score has to be read with that in mind.
    """
    per_seed = [_per_pair(only(load(r), operator)) for r in runs]
    every = [p for ps in per_seed for p in ps]
    if not every:
        return

    print()
    print('=' * 78)
    print('GLOBA TYPE BREAKDOWN')
    print('=' * 78)
    print(f'{"type":<10}{"share":>9}{"range":>9}{"rel.var":>9}'
          f'{"rho":>9}{"p":>8}{"picks":>9}')
    print('-' * 78)

    blind = [sum(p[outcome] for p in ps) / len(ps) for ps in per_seed]
    best = [max(p[outcome] for p in ps) for ps in per_seed]

    for t in TYPES:
        key = f'energy_{t}'
        vals = [p[key] for p in every]
        mean = sum(vals) / len(vals)
        rng = max(vals) - min(vals)

        xs, ys = [], []
        for ps in per_seed:
            xs += _rank_normalise([p[key] for p in ps])
            ys += _rank_normalise([p[outcome] for p in ps])
        rho = _spearman(xs, ys)
        pv = permutation_p(xs, ys, n_perm=3000)

        picks = [max(ps, key=lambda p: p[key])[outcome] for ps in per_seed]
        closes = sum((v - b) / (c - b) if c > b else 0.0
                     for v, b, c in zip(picks, blind, best)) / len(picks)
        flag = '' if pv < 0.05 else ' (ns)'
        print(f'{t:<10}{mean:>8.1%}{rng:>9.4f}{rng / mean if mean else 0:>8.0%}'
              f'{rho:>+9.3f}{pv:>8.3f}{closes:>+9.1%}{flag}')

    print()
    print('  share   = fraction of the pair\'s update energy of this type')
    print('  rel.var = range as a share of the type\'s own size -- a type that')
    print('            barely varies cannot rank pairs however large it is')
    print('  picks   = how much of the blind->oracle gap this type closes alone')


# --------------------------------------------------------------------------- #
# --layers
# --------------------------------------------------------------------------- #

LAYER_WEIGHTINGS = {
    'energy': lambda r: r['weight'],
    'uniform': lambda r: 1.0,
    'first_block': lambda r: 1.0 if '.0.conv1' in r['layer'] else 0.0,
    'early': lambda r: 1.0 if r['layer'].startswith('layer1') else 0.0,
    'late': lambda r: 1.0 if r['layer'].startswith('layer3') else 0.0,
}


def report_layers(runs, outcome, signal='energy_D_minus', operator='sesil'):
    """Where in the network the predictive signal actually lives.

    The default whole-network summary weights each layer by its share of update
    energy. That is not obviously right: the layers holding the most energy are
    not necessarily the ones whose structure says anything about the merge, and
    if they are silent then energy weighting dilutes the signal with them.
    """
    grouped = [(layer_rows(r),
                {tuple(p['pair']): p for p in _per_pair(only(load(r), operator))})
               for r in runs]
    grouped = [(L, P) for L, P in grouped if L]
    if not grouped:
        print()
        print('No layers.jsonl found -- per-layer data was not recorded for these '
              'runs.')
        return

    names = [r['layer'] for r in next(iter(grouped[0][0].values()))]

    print()
    print('=' * 78)
    print(f'PER-LAYER SIGNAL  ({signal})')
    print('=' * 78)
    print(f'{"layer":<30}{"weight%":>9}{"mean":>8}{"rho":>9}{"p":>8}')
    print('-' * 78)
    for name in names:
        xs, ys, means, weights = [], [], [], []
        for L, P in grouped:
            vals, outs = [], []
            for pair, rows in L.items():
                row = next(z for z in rows if z['layer'] == name)
                total = sum(z['weight'] for z in rows) or 1.0
                vals.append(row[signal])
                outs.append(P[pair][outcome])
                means.append(row[signal])
                weights.append(row['weight'] / total)
            xs += _rank_normalise(vals)
            ys += _rank_normalise(outs)
        rho = _spearman(xs, ys)
        pv = permutation_p(xs, ys, n_perm=2000)
        flag = '' if pv < 0.05 else ' (ns)'
        print(f'{name:<30}{sum(weights) / len(weights):>8.1%}'
              f'{sum(means) / len(means):>8.3f}{rho:>+9.3f}{pv:>8.3f}{flag}')

    print()
    print('=' * 78)
    print('HOW LAYERS ARE COMBINED INTO ONE NUMBER')
    print('=' * 78)
    print(f'{"weighting":<16}{"closes":>9}{"rho":>9}{"p":>8}   per-seed rho')
    print('-' * 78)

    blind = [sum(P[k][outcome] for k in L) / len(L) for L, P in grouped]
    best = [max(P[k][outcome] for k in L) for L, P in grouped]

    for label, weigh in LAYER_WEIGHTINGS.items():
        picks, xs, ys, rhos = [], [], [], []
        for (L, P), b, c in zip(grouped, blind, best):
            agg = {}
            for pair, rows in L.items():
                w = [weigh(r) for r in rows]
                total = sum(w) or 1.0
                agg[pair] = sum(wi * r[signal] for wi, r in zip(w, rows)) / total
            outs = [P[k][outcome] for k in agg]
            vals = list(agg.values())
            xs += _rank_normalise(vals)
            ys += _rank_normalise(outs)
            rhos.append(_spearman(vals, outs))
            picks.append(P[max(agg, key=agg.get)][outcome])
        closes = sum((v - b) / (c - b) if c > b else 0.0
                     for v, b, c in zip(picks, blind, best)) / len(picks)
        rho = _spearman(xs, ys)
        pv = permutation_p(xs, ys, n_perm=3000)
        consistent = all(r > 0 for r in rhos) or all(r < 0 for r in rhos)
        note = '' if consistent else '  FLIPS between seeds'
        print(f'{label:<16}{closes:>+9.1%}{rho:>+9.3f}{pv:>8.3f}   '
              f'{[round(r, 2) for r in rhos]}{note}')


def report_operator_comparison(runs, outcome):
    """Score the same couples under each merge operator.

    This is the only comparison that isolates the operator: identical parents,
    identical measurement, so a difference in outcome is a difference in the
    merge and not in which pairs happened to be easy. It also answers whether
    the pair RANKING survives the change of operator -- if the two orderings
    agree, "a good pair" is a property of the parents; if they disagree, it is
    a property of the pairing WITH that operator, and a predictor has to be
    validated against whichever operator will actually be used.
    """
    ops = operators_present(runs)
    if len(ops) < 2:
        print()
        print(f'Only one operator present ({ops[0] if ops else "none"}); '
              f'nothing to compare.')
        print('Re-run with --probe-merger both to get the comparison.')
        return

    print()
    print('=' * 78)
    print('MERGE OPERATORS, SAME PAIRS')
    print('=' * 78)
    print(f'{"operator":<22}{"mean":>9}{"best":>9}{"worst":>9}{"wins pair-by-pair":>20}')
    print('-' * 78)

    by_op = {}
    for op in ops:
        per_seed = [_per_pair(only(load(r), op)) for r in runs]
        by_op[op] = {tuple(p['pair']): p[outcome]
                     for ps in per_seed for p in ps}

    shared = set.intersection(*[set(v) for v in by_op.values()])
    for op in ops:
        vals = [by_op[op][k] for k in shared]
        others = [max(by_op[o][k] for o in ops if o != op) for k in shared]
        wins = sum(1 for v, o in zip(vals, others) if v > o)
        print(f'{op:<22}{sum(vals) / len(vals):>9.4f}{max(vals):>9.4f}'
              f'{min(vals):>9.4f}{f"{wins}/{len(shared)}":>20}')

    if len(ops) == 2:
        a, b = ops
        xs = [by_op[a][k] for k in shared]
        ys = [by_op[b][k] for k in shared]
        rho = _spearman(xs, ys)
        print()
        print(f'Do the two operators rank pairs the same way?  rho {rho:+.3f}')
        if rho > 0.7:
            print('  High -- a good pair is mostly a property of the PARENTS, so a')
            print('  predictor validated on one operator should carry to the other.')
        elif rho > 0.3:
            print('  Moderate -- partly the parents, partly the operator.')
        else:
            print('  Low -- "a good pair" depends on WHICH operator merges them, so a')
            print('  predictor must be validated against the operator you will use.')


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
    ap.add_argument('--operator', default=None,
                    help='Which merge operator to analyse when a run used '
                         "--probe-merger both: 'sesil' or 'globa'. Defaults to "
                         'whichever is present, or sesil when both are. They are '
                         'never mixed -- averaging two operators into one '
                         'correlation would answer no question at all.')
    ap.add_argument('--compare-operators', action='store_true',
                    help='Score the SAME pairs under each operator side by side. '
                         'Needs a run made with --probe-merger both.')
    ap.add_argument('--per-seed', action='store_true',
                    help='Also print each seed separately.')
    ap.add_argument('--strategies', action='store_true',
                    help='Compare whole selection RULES by the pair each one '
                         'actually picks, with the certificate overlap of that '
                         'pair -- including two-stage rules that filter on '
                         'complementarity and then tie-break on a GLOBA signal. '
                         'A rule can correlate well and still choose badly, so '
                         'this answers a different question from the main table.')
    ap.add_argument('--types', action='store_true',
                    help='Break down the six GLOBA energy types: how much of the '
                         'update energy each holds, how much it varies, and what '
                         'each predicts on its own.')
    ap.add_argument('--layers', action='store_true',
                    help='Where in the network the signal lives, and how much the '
                         'choice of layer weighting matters. Needs layers.jsonl, '
                         'which probe runs write automatically.')
    ap.add_argument('--layer-signal', default='energy_D_minus',
                    help='Which per-layer statistic --layers examines '
                         "(default energy_D_minus; try energy_E or cosine).")
    ap.add_argument('--all', action='store_true',
                    help='Every section.')
    args = ap.parse_args()

    runs = find_runs(args.root)
    if not runs:
        raise SystemExit(f'No pairs.jsonl found under {args.root!r}. '
                         f'Run: bash run_probe.sh --dataset cifar10 --seed 0')

    # Which operator's children are being judged. Never mixed: with
    # --probe-merger both the same couple appears twice, and blending them
    # would average two different merges into one uninterpretable number.
    available = operators_present(runs)
    operator = args.operator or ('sesil' if 'sesil' in available else available[0])
    if operator not in available:
        raise SystemExit(f'No {operator!r} children in these runs. '
                         f'Present: {available}.')

    # The right outcome depends on how the children were built, which is
    # recorded in the rows rather than assumed here.
    #
    # Read it from the first run that actually HAS children for this operator.
    # Pointing at a directory holding several conditions can otherwise land on
    # one with no rows for the chosen operator, and an empty list silently
    # falls back to the wrong metric -- wrong numbers, no warning.
    sample = next((pairs for pairs in
                   (_per_pair(only(load(r), operator)) for r in runs) if pairs),
                  [])
    if not sample:
        raise SystemExit(f'No {operator!r} children found in any run under '
                         f'{args.root!r}.')
    outcome = args.outcome or default_outcome(sample)

    print(f'{len(runs)} run(s) under {args.root}')
    if len(available) > 1:
        print(f'operator: {operator}  (also present: '
              f'{[o for o in available if o != operator]}; '
              f'--compare-operators scores them side by side)')
    print(f'outcome: {outcome}'
          + ('' if args.outcome else '  (chosen from the merge geometry)') + '\n')

    per_run = []
    for run_dir in runs:
        rows = only(load(run_dir), operator)
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
    pooled = pool_across_seeds(runs, outcome, operator=operator)

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

    if args.strategies or args.all:
        report_strategies(runs, outcome, operator)
    if args.types or args.all:
        report_types(runs, outcome, operator)
    if args.layers or args.all:
        report_layers(runs, outcome, signal=args.layer_signal, operator=operator)
    if args.compare_operators or args.all:
        report_operator_comparison(runs, outcome)


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
