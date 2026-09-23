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
from sesil.pretrain import backbone_dir, build_population
from sesil.probe.globa_stats import TYPES, pair_stats
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

    n_pairs = args.pop_size * (args.pop_size - 1) // 2
    print(f'[probe] {args.pop_size} agents -> {n_pairs} pairs -> '
          f'{2 * n_pairs} children to evaluate')
    print(f'[probe] merger under test: {args.merger} '
          f'(stop-node {args.stop_node if args.stop_node is not None else "none / full merge"})')
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

        # --- the ground truth: the merger SESiL would actually have used ---
        with torch.no_grad():
            merge, config = merge_couple((a, b), raw_config, args, train_loader)
            budget.count_forward_train(FORWARD_TRAIN_PASSES_PER_MERGE)

            children = extract_children(merge, config, args,
                                        args.num_classes, train_loader)
            budget.count_forward_train(len(children))

            for child_index, (child, n_from_trunk) in enumerate(children):
                per_class, overall = evaluate_all_classes(
                    child, val_loader, args.num_classes)
                ret = pair_retention(accs[a], accs[b], per_class)

                row = {
                    'stage': 'probe_pair',
                    'pair': [a, b],
                    'child_index': child_index,
                    'biased_toward': [a, b][child_index],
                    'merger': args.merger,
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
                    'merge_seconds': merge.compute_transform_time,
                }
                rows.append(row)
                logger.log_train(row)

            del merge, config, children

        torch.cuda.empty_cache()

    # ---------------------------------------------------------------- output
    out_path = os.path.join(args.run_dir, 'pairs.jsonl')
    with open(out_path, 'w') as f:
        for row in rows:
            f.write(json.dumps(row, default=float) + '\n')

    summary = _summarise(rows, args)
    with open(os.path.join(args.run_dir, 'probe_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2, default=float)

    print(f'\n[probe] {len(pairs)} pairs, {len(rows)} children, '
          f'{time.time() - start:.1f}s')
    print(f'[probe] rows -> {out_path}')
    _print_summary(summary)
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
        key = tuple(row['pair'])
        by_pair.setdefault(key, []).append(row)

    out = []
    for key, group in by_pair.items():
        merged = dict(group[0])
        merged['pair'] = list(key)
        for field in ('balanced', 'retention_total', 'child_overall',
                      'retention_a', 'retention_b',
                      'distinctive_a', 'distinctive_b'):
            merged[field] = sum(g[field] for g in group) / len(group)
        merged.pop('child_index', None)
        merged.pop('child_per_class', None)
        out.append(merged)
    return out


PREDICTORS = (['globa_score', 'mating_score', 'partner_mean_acc', 'cosine',
               'cancellation', 'typed_frac', 'norm_ratio']
              + [f'energy_{t}' for t in TYPES])


def _summarise(rows, args):
    """Where each predictor's favourite pair lands in the true ranking."""
    pairs = _per_pair(rows)
    n = len(pairs)
    if n < 2:
        return {'n_pairs': n, 'note': 'too few pairs to rank'}

    truth = sorted(pairs, key=lambda r: r['balanced'], reverse=True)
    best, worst = truth[0], truth[-1]
    spread = best['balanced'] - worst['balanced']

    result = {
        'n_pairs': n,
        'merger': args.merger,
        'seed': args.seed,
        'best_pair': best['pair'],
        'best_balanced': best['balanced'],
        'worst_balanced': worst['balanced'],
        'spread': spread,
        'median_balanced': truth[n // 2]['balanced'],
        'predictors': {},
    }

    for name in PREDICTORS:
        if name not in pairs[0]:
            continue
        pick = max(pairs, key=lambda r: r[name])
        rank = next(i for i, r in enumerate(truth) if r['pair'] == pick['pair']) + 1
        result['predictors'][name] = {
            'picked_pair': pick['pair'],
            'rank_of_pick': rank,
            'n_pairs': n,
            'balanced_of_pick': pick['balanced'],
            'gap_to_best': best['balanced'] - pick['balanced'],
            'spearman': _spearman([r[name] for r in pairs],
                                  [r['balanced'] for r in pairs]),
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
    print('=' * 72)
    print(f'Best pair {summary["best_pair"]} scored {summary["best_balanced"]:.4f}; '
          f'worst {summary["worst_balanced"]:.4f}  (spread {summary["spread"]:.4f})')
    if summary['spread'] < 0.02:
        print('NOTE: every pair scored about the same, so partner choice barely '
              'matters here and no predictor can look good or bad for a real '
              'reason. Treat the ranks below as noise.')
    print('=' * 72)
    print(f'{"predictor":<22} {"rank":>8} {"retention":>10} {"gap":>8} {"rho":>7}')
    print('-' * 72)
    order = sorted(summary['predictors'].items(), key=lambda kv: kv[1]['rank_of_pick'])
    for name, p in order:
        mark = '  <- pre-registered' if name == 'globa_score' else ''
        print(f'{name:<22} {p["rank_of_pick"]:>4}/{n:<3} '
              f'{p["balanced_of_pick"]:>10.4f} {p["gap_to_best"]:>8.4f} '
              f'{p["spearman"]:>7.3f}{mark}')
    print('=' * 72)
