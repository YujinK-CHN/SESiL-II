"""The mate-screening probe: does GLOBA's decomposition predict a good pair?

This is a diagnostic setting, not a learning method. It does not optimise
anything, it has no generations, and it produces no accuracy-versus-budget
curve -- `plot_results.py` is not its consumer. It builds one population, then
exhaustively merges every possible couple and records, for each, what GLOBA
predicted from the weights alone against what the merge actually produced.

    pretrain (phase A backbone + phase B specialists)
    evaluate every agent on all classes            -> reference proficiency
    for every unordered pair:
        GLOBA statistics from (tau_i, tau_j)       -> the prediction, no merge
        merge with the CONFIGURED SESiL merger     -> the ground truth
        evaluate both children, measure retention

GLOBA is used only as a screening instrument. Children are produced by SESiL's
own merger (zipit / permute / wavg), because the question is whether GLOBA can
advise *that* operator -- not whether GLOBA is a good merger, which is a
different experiment.

Every pair is measured, so the full ranking is known and there is nothing to
compare against a random baseline: random is the middle of a ranking we already
have in full. What the analysis asks instead is where GLOBA's pick lands in
that ranking, and how much retention it gives up relative to the best pair.

A shared origin is mandatory. A task vector is `agent - core`, so without a
phase-A backbone there is no core and the decomposition is meaningless; the
driver refuses to run rather than silently analysing noise.
"""

import itertools
import json
import os
import time

import torch
from tqdm import tqdm

from config import build_raw_config
from utils import prepare_experiment_config, reset_bn_stats

from sesil.budget import FORWARD_TRAIN_PASSES_PER_MERGE
from sesil.certificate import certify_population
from sesil.fitness import evaluate_all_classes
from sesil.merge import extract_children, merge_couple, point_at
from sesil.population import generation_dir, list_population, read_meta
from sesil.pretrain import backbone_dir, build_model, build_population
from sesil.globa_merge import merge as globa_merge_pair
from sesil.globa_stats import TYPES, pair_stats
from sesil.probe.retention import pair_retention
from sesil.selection import mating_score
from sesil.ssl import classifier_name


def _require_shared_core(args):
    """A task vector needs an origin. Refuse to guess one."""
    if args.pretrain_mode != 'ssl':
        raise SystemExit(
            'The probe requires --pretrain-mode ssl.\n'
            'GLOBA analyses task vectors (agent - core), and the core is the '
            'phase-A backbone. Under --pretrain-mode none every agent starts '
            'from its own random initialisation, there is no shared origin, '
            'and the decomposition would be describing initialisation noise.')
    if args.phase_a_ratio <= 0:
        raise SystemExit(
            '--phase-a-ratio must be > 0 for the probe: with no phase A there '
            'is no backbone to serve as the core.')


def _load_core(args):
    """The phase-A backbone every agent was finetuned from."""
    path = os.path.join(backbone_dir(args), f'{args.arch}.pth.tar')
    if not os.path.exists(path):
        raise SystemExit(f'No phase-A backbone at {path}. Pretrain did not run?')
    return torch.load(path, map_location='cpu')


def _load_state(population_dir, agent_id, arch):
    path = os.path.join(population_dir, agent_id, f'{arch}.pth.tar')
    if not os.path.exists(path):
        path = os.path.join(population_dir, agent_id, f'{arch}_v0.pth.tar')
    sd = torch.load(path, map_location='cpu')
    return sd['state_dict'] if isinstance(sd, dict) and 'state_dict' in sd else sd


def _load_agent(agent_id, raw_config, data, budget):
    """Materialise one agent with BN statistics recalibrated on the train split."""
    point_at(raw_config, [agent_id])
    config = prepare_experiment_config(raw_config)
    model = config['models']['bases'][0]
    reset_bn_stats(model, data.train_loader())
    return model


def run_probe(args, budget, data, logger, evaluator=None):
    """Build a population, screen every pair, measure every merge."""
    _require_shared_core(args)

    # Symmetric merge: one whole-network blend, weighted equally, so a couple
    # has one distinct child rather than two mirror images.
    symmetric = args.stop_node is None and abs(args.merge_bias - 0.5) < 1e-9

    n_pairs = args.pop_size * (args.pop_size - 1) // 2
    per_couple = (1 if symmetric else 2) if args.probe_merger != 'globa' else 0
    if args.probe_merger in ('globa', 'both'):
        per_couple += 2      # GLOBA is asymmetric: merge(a,b) and merge(b,a)
    print(f'[probe] {args.pop_size} agents -> {n_pairs} pairs -> '
          f'{per_couple * n_pairs} children to evaluate')
    if args.probe_merger in ('sesil', 'both'):
        print(f'[probe] merger under test: {args.merger} '
              f'(stop-node {args.stop_node if args.stop_node is not None else "none / full merge"})')
    if args.probe_merger in ('globa', 'both'):
        print(f'[probe] merger under test: globa '
              f'(preset {args.globa_preset}, head {args.globa_head})')
    if args.probe_merger == 'both':
        print(f'[probe] BOTH operators run on identical parents, so the two are '
              f'directly\n        comparable. Analyse them separately: '
              f'analyze_probe.py --operator globa')
    if symmetric:
        print(f'[probe] symmetric full merge (--merge-bias 0.5): one child per '
              f'couple,\n        an equal blend of both parents. Outcome metric '
              f'is "balanced" -- how\n        well ONE model carries both '
              f'parents.')
    else:
        print(f'[probe] asymmetric merge: two children per couple, each leaning '
              f'to one\n        parent. Outcome metric is "transfer" -- how much '
              f'of the OTHER parent\n        a child picks up, since it keeps its '
              f'own almost by construction.')
    if args.probe_merger == 'sesil':
        print(f'[probe] GLOBA is the predictor only; it does not produce children.')

    # ---------------------------------------------------------------- stage 1
    population = build_population(args, data, budget, logger)
    if not population:
        raise RuntimeError('Pretrain produced no individuals')

    core = _load_core(args)
    head_prefix = None
    population_dir = generation_dir(args, 0)
    agent_ids = list_population(population_dir)

    raw_config = build_raw_config(args)
    raw_config['model']['dir'] = population_dir

    # ---------------------------------------------------------------- stage 2
    # Reference proficiency: every agent on the whole label space. On VAL --
    # the probe is a study, but there is no reason to spend the test split on
    # it, and keeping the convention makes its numbers comparable with the
    # training runs'.
    val_loader = data.val_loader()
    states, accs, overalls = {}, {}, {}

    with torch.no_grad():
        for agent_id in tqdm(agent_ids, desc='probe: evaluating parents'):
            model = _load_agent(agent_id, raw_config, data, budget)
            per_class, overall = evaluate_all_classes(model, val_loader, args.num_classes)
            accs[agent_id] = per_class
            overalls[agent_id] = overall
            states[agent_id] = {k: v.detach().cpu()
                                for k, v in model.state_dict().items()}
            if head_prefix is None:
                head_prefix = classifier_name(model) + '.'
            del model

    # Certificates are NOT used by the retention measure -- proficiency is
    # measured directly from the accuracy vectors. They exist here only to
    # compute SESiL's incumbent mating_score, which is the predictor GLOBA has
    # to beat, and which happens to be defined on certificates.
    certificates = certify_population(
        [accs[a] for a in agent_ids],
        top_frac=args.certify_top_frac,
        floor=args.certify_floor,
        num_classes=args.num_classes,
    )
    cert_by_id = {a: set(c) for a, c in zip(agent_ids, certificates)}

    # Which classes each agent was actually finetuned on. The label-aware head
    # needs this rather than the certificate: "was this parent trained on this
    # class" is a fact about the run, while a certificate is a measurement that
    # moves with --certify-floor. Using the certificate would make the merge
    # operator itself depend on a thresholding choice.
    trained_on = {a: read_meta(population_dir, a).get('trained_on', [])
                  for a in agent_ids}

    for agent_id in agent_ids:
        logger.log_train({
            'stage': 'probe_parent',
            'agent': agent_id,
            'val_overall': overalls[agent_id],
            'val_per_class': [round(v, 5) for v in accs[agent_id]],
            'certificate': sorted(cert_by_id[agent_id]),
            'trained_on': read_meta(population_dir, agent_id).get('trained_on', []),
        })

    # ---------------------------------------------------------------- stage 3
    train_loader = data.train_loader()
    rows = []
    layer_rows = []
    start = time.time()

    pairs = list(itertools.combinations(agent_ids, 2))
    for a, b in tqdm(pairs, desc='probe: screening + merging pairs'):
        # --- the prediction: weights only, no forward passes, no merge -----
        t0 = time.time()
        globa, per_layer = pair_stats(
            states[a], states[b], core,
            eta=args.probe_eta,
            svd_energy=args.probe_svd_energy,
            basis_energy=args.probe_basis_energy,
            head_prefix=head_prefix,
            weighting=args.probe_layer_weighting,
        )
        predict_seconds = time.time() - t0

        # Per-layer breakdown, kept separately from pairs.jsonl.
        #
        # It belongs to the COUPLE, not to either child, so writing it into the
        # child rows would duplicate every number and make a pair count twice
        # in any correlation computed off that file. A separate file also keeps
        # pairs.jsonl small enough to stay pleasant for the common case.
        #
        # 'weight' is what the whole-network summary used to average this layer
        # in, so an analysis can re-weight the layers differently -- or look at
        # one layer alone -- without recomputing anything.
        for layer_name, s in per_layer.items():
            layer_rows.append({
                'pair': [a, b],
                'layer': layer_name,
                'weight': s['norm_p'] ** 2 + s['norm_q'] ** 2,
                **{f'energy_{t}': s['energy_frac'][t] for t in TYPES},
                **{f'merged_energy_{t}': s['energy_frac_merged'][t] for t in TYPES},
                'cancellation': s['cancellation'],
                'typed_frac_p': s['typed_frac_p'],
                'typed_frac_q': s['typed_frac_q'],
                'cosine': s['cosine'],
                'norm_p': s['norm_p'],
                'norm_q': s['norm_q'],
                'basis_u': s['basis_u'],
                'basis_v': s['basis_v'],
                'nnz_p': s['nnz_p'],
                'nnz_q': s['nnz_q'],
            })

        # --- the ground truth: actually merge the pair and score the child ---
        with torch.no_grad():
            produced = []

            if args.probe_merger in ('sesil', 'both'):
                merge, config = merge_couple((a, b), raw_config, args, train_loader)
                budget.count_forward_train(FORWARD_TRAIN_PASSES_PER_MERGE)

                children = extract_children(
                    merge, config, args, args.num_classes, train_loader,
                    parent_classes=[cert_by_id[a], cert_by_id[b]])

                # At --stop-node none with --merge-bias 0.5 the interpolation
                # weights are equal, so both children are the same tensors.
                # Evaluating the second would cost a full pass to reproduce a
                # number we already have, and would write a duplicate row that
                # silently double-weights this couple in the correlations.
                if symmetric:
                    children = children[:1]
                budget.count_forward_train(len(children))

                for index, (child, n_from_trunk, label_rows) in enumerate(children):
                    produced.append((child, index, 'sesil', n_from_trunk,
                                     merge.compute_transform_time))
                del merge, config, children

            if args.probe_merger in ('globa', 'both'):
                # GLOBA is asymmetric: the base parent is kept whole and only
                # selected components of the donor are added. So a couple
                # yields TWO children -- merge(a, b) and merge(b, a) -- one
                # leaning to each parent, with no interpolation weight needed
                # to tell them apart.
                #
                # No alignment pass and no activation statistics: the operator
                # is linear algebra on the task vectors, so it costs nothing
                # from the data budget. Only the BN recalibration touches the
                # training set.
                for index, (base, donor) in enumerate(((a, b), (b, a))):
                    t1 = time.time()
                    child_sd = globa_merge_pair(
                        states[base], states[donor], core,
                        classes_base=trained_on.get(base),
                        classes_donor=trained_on.get(donor),
                        preset=args.globa_preset, head=args.globa_head,
                        eta=args.probe_eta, svd_energy=args.probe_svd_energy,
                        basis_energy=args.probe_basis_energy,
                        head_prefix=head_prefix)
                    globa_seconds = time.time() - t1

                    child = build_model(args, args.num_classes)
                    child.load_state_dict(child_sd, strict=False)
                    # The child owns neither parent's BatchNorm statistics, so
                    # they are recomputed for the network as assembled --
                    # exactly as extract_children does for the SESiL side, so
                    # neither operator is handicapped.
                    reset_bn_stats(child, train_loader)
                    budget.count_forward_train(1)
                    produced.append((child, index, 'globa', 0, globa_seconds))

            for child, child_index, operator, n_from_trunk, merge_seconds in produced:
                per_class, overall = evaluate_all_classes(
                    child, val_loader, args.num_classes)
                ret = pair_retention(accs[a], accs[b], per_class)

                # Split retention into the half that is structural and the half
                # that is earned.
                #
                # With --stop-node N a child is the merged trunk plus ONE
                # parent's remaining layers, so it keeps that parent almost by
                # construction (measured: 0.93-0.99, near the ceiling for every
                # pair). What varies -- and what crossover is actually being
                # asked to deliver -- is how much of the OTHER parent it picks
                # up. Reporting only the combined figure buries that: the
                # harmonic mean ends up driven by the transfer term while
                # looking like it measures balance.
                own, transfer = ((ret['distinctive_a'], ret['distinctive_b'])
                                 if child_index == 0 else
                                 (ret['distinctive_b'], ret['distinctive_a']))

                row = {
                    'own': own,
                    'transfer': transfer,
                    # A GLOBA child keeps its base parent whole and adds parts
                    # of the donor, so it leans to one parent exactly as a
                    # stop-node child does. Not symmetric.
                    'symmetric': symmetric and operator != 'globa',
                    'stage': 'probe_pair',
                    'pair': [a, b],
                    'operator': operator,
                    'child_index': child_index,
                    'biased_toward': [a, b][child_index],
                    'merger': args.merger if operator == 'sesil'
                              else f'globa_{args.globa_preset}_{args.globa_head}',
                    'trunk_params': n_from_trunk,

                    # outcome
                    **ret,
                    'child_overall': overall,
                    'child_per_class': [round(v, 5) for v in per_class],
                    'parent_a_overall': overalls[a],
                    'parent_b_overall': overalls[b],

                    # pre-registered predictor
                    'globa_score': globa['globa_score'],

                    # exploratory predictors
                    **{f'energy_{t}': globa['energy_frac'][t] for t in TYPES},
                    'cancellation': globa['cancellation'],
                    'typed_frac': globa['typed_frac'],
                    'cosine': globa['cosine'],
                    'norm_ratio': globa['norm_ratio'],
                    'basis_u': globa['basis_u'],
                    'basis_v': globa['basis_v'],

                    # the incumbent GLOBA has to beat, symmetrised
                    'mating_score': 0.5 * (
                        mating_score(cert_by_id[a], cert_by_id[b],
                                     [1.0] * args.num_classes)
                        + mating_score(cert_by_id[b], cert_by_id[a],
                                       [1.0] * args.num_classes)),
                    'partner_mean_acc': 0.5 * (overalls[a] + overalls[b]),

                    'predict_seconds': round(predict_seconds, 4),
                    'merge_seconds': merge_seconds,
                }
                rows.append(row)
                logger.log_train(row)

            del produced

        torch.cuda.empty_cache()

    # ---------------------------------------------------------------- output
    out_path = os.path.join(args.run_dir, 'pairs.jsonl')
    with open(out_path, 'w') as f:
        for row in rows:
            f.write(json.dumps(row, default=float) + '\n')

    layer_path = os.path.join(args.run_dir, 'layers.jsonl')
    with open(layer_path, 'w') as f:
        for row in layer_rows:
            f.write(json.dumps(row, default=float) + '\n')

    # One summary PER OPERATOR. Summarising them together would rank a couple's
    # SESiL child against its own GLOBA child as if they were different pairs,
    # so "rank 3 of 28" would silently become "rank 3 of 56" and mean nothing.
    operators = sorted({r['operator'] for r in rows})
    summaries = {op: _summarise([r for r in rows if r['operator'] == op], args)
                 for op in operators}
    summary = summaries if len(operators) > 1 else summaries[operators[0]]

    with open(os.path.join(args.run_dir, 'probe_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2, default=float)

    print(f'\n[probe] {len(pairs)} pairs, {len(rows)} children, '
          f'{time.time() - start:.1f}s')
    print(f'[probe] rows   -> {out_path}')
    print(f'[probe] layers -> {layer_path}  ({len(layer_rows)} rows)')
    for op in operators:
        if len(operators) > 1:
            print(f'\n### operator: {op} ###')
        _print_summary(summaries[op])
    print(budget.report())

    return summary


def _per_pair(rows):
    """Collapse the two children of each pair into one outcome.

    Mean rather than max: a couple's two children differ by which parent they
    lean toward, so taking the better one would reward exactly the asymmetry
    the balanced measure exists to penalise.
    """
    by_pair = {}
    for row in rows:
        # Keyed by operator too: with --probe-merger both, the same couple has
        # a SESiL child and a GLOBA child, and collapsing them together would
        # average two different operators into one meaningless number.
        key = (tuple(row['pair']), row.get('operator', 'sesil'))
        by_pair.setdefault(key, []).append(row)

    out = []
    for (pair, operator), group in by_pair.items():
        merged = dict(group[0])
        merged['pair'] = list(pair)
        merged['operator'] = operator
        for field in ('balanced', 'retention_total', 'child_overall',
                      'retention_a', 'retention_b',
                      'distinctive_a', 'distinctive_b',
                      'own', 'transfer'):
            merged[field] = sum(g[field] for g in group) / len(group)
        merged.pop('child_index', None)
        merged.pop('child_per_class', None)
        out.append(merged)
    return out


PREDICTORS = (['globa_score', 'mating_score', 'partner_mean_acc', 'cosine',
               'cancellation', 'typed_frac', 'norm_ratio']
              + [f'energy_{t}' for t in TYPES])


def default_outcome(pairs):
    """Which measure answers "was this a good pair", given the merge geometry.

    A symmetric full merge produces ONE model that has to carry both parents,
    so the question is whether it managed that -- 'balanced', the harmonic mean
    of the two distinctive retentions, which goes to zero if either parent was
    lost.

    An asymmetric merge produces two children that each lean to one parent and
    keep it near the ceiling whatever the partner. Ranking on 'balanced' there
    would mostly be ranking on a constant, so the measure that varies -- and
    that crossover has to earn -- is 'transfer': how much of the OTHER parent
    each child picked up.
    """
    return 'balanced' if pairs and pairs[0].get('symmetric') else 'transfer'


def _summarise(rows, args):
    """Where each predictor's favourite pair lands in the true ranking."""
    pairs = _per_pair(rows)
    n = len(pairs)
    if n < 2:
        return {'n_pairs': n, 'note': 'too few pairs to rank'}

    outcome = default_outcome(pairs)
    truth = sorted(pairs, key=lambda r: r[outcome], reverse=True)
    best, worst = truth[0], truth[-1]
    spread = best[outcome] - worst[outcome]

    result = {
        'n_pairs': n,
        'merger': args.merger,
        'seed': args.seed,
        'best_pair': best['pair'],
        'outcome': outcome,
        'best_transfer': best[outcome],
        'worst_transfer': worst[outcome],
        'spread': spread,
        'median_transfer': truth[n // 2][outcome],
        'blind': sum(r[outcome] for r in pairs) / n,
        'mean_own': sum(r['own'] for r in pairs) / n,
        'zero_transfer_pairs': sum(1 for r in pairs if r[outcome] < 0.01),
        'predictors': {},
    }

    for name in PREDICTORS:
        if name not in pairs[0]:
            continue
        values = [r[name] for r in pairs]
        pick = max(pairs, key=lambda r: r[name])
        rank = next(i for i, r in enumerate(truth) if r['pair'] == pick['pair']) + 1
        result['predictors'][name] = {
            'picked_pair': pick['pair'],
            'rank_of_pick': rank,
            'n_pairs': n,
            'transfer_of_pick': pick[outcome],
            'gap_to_best': best[outcome] - pick[outcome],
            'spearman': _spearman(values, [r[outcome] for r in pairs]),
            # A predictor that returns nearly the same number for every pair
            # cannot rank them, whatever its correlation happens to be.
            'range': max(values) - min(values),
            'min': min(values),
            'max': max(values),
        }
    return result


def _rank(xs):
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    ranks = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def _spearman(xs, ys):
    """Rank correlation, ties averaged. Written out to avoid a scipy dependency."""
    n = len(xs)
    if n < 3:
        return 0.0
    rx, ry = _rank(xs), _rank(ys)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    dx = sum((a - mx) ** 2 for a in rx) ** 0.5
    dy = sum((b - my) ** 2 for b in ry) ** 0.5
    return float(num / (dx * dy)) if dx > 0 and dy > 0 else 0.0


def _print_summary(summary):
    if 'predictors' not in summary:
        return
    n = summary['n_pairs']
    print()
    print('=' * 78)
    outcome = summary.get('outcome', 'transfer')
    if outcome == 'balanced':
        print('Outcome: balanced -- how well ONE symmetric child carries BOTH '
              'parents.')
        print('(Zero means it kept one parent and lost the other entirely.)')
        verb = 'scored'
    else:
        print('Outcome: transfer -- how much of the OTHER parent a child picks up.')
        print(f'Each child keeps its own parent at {summary["mean_own"]:.3f} on '
              f'average (structural, near the ceiling).')
        verb = 'transferred'
    print(f'Best pair {summary["best_pair"]} {verb} '
          f'{summary["best_transfer"]:.4f}; worst {summary["worst_transfer"]:.4f}; '
          f'blind choice {summary["blind"]:.4f}')
    if summary['zero_transfer_pairs']:
        print(f'{summary["zero_transfer_pairs"]} of {n} pairs scored ~zero.')
    if summary['spread'] < 0.02:
        print('NOTE: every pair scored about the same, so partner choice barely '
              'matters here.')
    print('=' * 78)
    print(f'{"predictor":<20} {"rank":>8} {outcome:>9} {"gap":>8} '
          f'{"rho":>7} {"range":>8}')
    print('-' * 78)
    order = sorted(summary['predictors'].items(), key=lambda kv: kv[1]['rank_of_pick'])
    for name, p in order:
        mark = '  <- pre-registered' if name == 'globa_score' else ''
        print(f'{name:<20} {p["rank_of_pick"]:>4}/{n:<3} '
              f'{p["transfer_of_pick"]:>9.4f} {p["gap_to_best"]:>8.4f} '
              f'{p["spearman"]:>7.3f} {p["range"]:>8.4f}{mark}')
    print('=' * 78)
    print('Run analyze_probe.py for significance testing across seeds.')
