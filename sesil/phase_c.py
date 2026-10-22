"""Phase C: fit the mating function before evolution starts.

Phases A and B produce a population. Phase C asks that population a question
neither of them answers: given two agents, how good is their child? It samples
pairs, actually merges them, measures the result, and fits sesil.mating_fn to
the measurements. Evolution then mates on the fitted rule instead of on a cell
type named at the command line.

The samples are DIRECTIONAL, and deliberately so. GLOBA's merge keeps the base
parent whole and adds selected components of the donor, so a couple yields two
different children -- merge(a, b) and merge(b, a) -- and mating asks each agent
"what does this partner bring ME". Training on ordered pairs makes the thing
being fitted the same thing mating consumes. The outcome is 'transfer': a child
keeps its own base parent near the ceiling almost by construction, so what
varies, and what partner choice has to earn, is how much of the OTHER parent it
picked up.

Cost is forward-only -- the merge is linear algebra on task vectors, and only
BatchNorm recalibration and the evaluation touch data -- so phase C spends no
backpropagation budget. It is not free in wall-clock time, which is why it
samples pairs rather than taking all of them.
"""

import itertools
import json
import os
import random
import time

import numpy as np
import torch
from tqdm import tqdm

from config import build_raw_config
from utils import reset_bn_stats

from sesil import mating_fn
from sesil.budget import FORWARD_TRAIN_PASSES_PER_MERGE
from sesil.certificate import certify_population
from sesil.fitness import evaluate_all_classes, summarise
from sesil.globa_merge import merge as globa_merge_pair
from sesil.globa_stats import TYPES, directional_pair_stats
from sesil.population import generation_dir, list_population, read_meta
from sesil.pretrain import build_model
from sesil.probe.retention import pair_retention
from sesil.ssl import classifier_name


def _load_agent(agent_id, raw_config, data, budget):
    from sesil.probe.driver import _load_agent as _la
    return _la(agent_id, raw_config, data, budget)


def _load_core(args):
    from sesil.probe.driver import _load_core as _lc
    return _lc(args)


def sample_pairs(agent_ids, n_pairs, seed):
    """Ordered pairs to measure, drawn without replacement from the unordered
    ones. Both directions of a chosen pair are kept: one decomposition yields
    both, so the second direction is nearly free and doubles the sample."""
    everything = list(itertools.combinations(agent_ids, 2))
    rng = random.Random(seed)
    rng.shuffle(everything)
    if n_pairs and n_pairs < len(everything):
        everything = everything[:int(n_pairs)]
    return everything


def run_phase_c(args, budget, data, logger):
    """Fit whichever mating model this run asked for.

    Two different jobs share the phase, because both need measured merges:

        --mating-mode learn     a pair-quality net, fitted on EVERY ordered
                                pair of SEVERAL populations
        --globa-with learned    coefficients over the six types, fitted on a
                                sample of pairs from THIS population

    Returns the path it wrote, or None when phase C is not applicable.
    """
    if args.mating_mode == 'learn':
        return run_learn_phase_c(args, budget, data, logger)

    population_dir = generation_dir(args, 0)
    agent_ids = list_population(population_dir)
    if len(agent_ids) < 2:
        raise RuntimeError('Phase C needs at least two agents.')

    core = _load_core(args)
    raw_config = build_raw_config(args)
    raw_config['model']['dir'] = population_dir

    val_loader = data.val_loader()
    train_loader = data.train_loader()

    start = time.time()
    states, accs, overalls = {}, {}, {}
    head_prefix = None
    with torch.no_grad():
        for agent_id in tqdm(agent_ids, desc='phase c: evaluating parents'):
            model = _load_agent(agent_id, raw_config, data, budget)
            per_class, overall = evaluate_all_classes(
                model, val_loader, args.num_classes)
            accs[agent_id] = per_class
            overalls[agent_id] = overall
            states[agent_id] = {k: v.detach().cpu()
                                for k, v in model.state_dict().items()}
            if head_prefix is None:
                head_prefix = classifier_name(model) + '.'
            del model

    # Certificates are not used to score a merge -- retention is measured from
    # the accuracy vectors directly. They are built because the CONTEXT vector
    # is defined on them, and it has to be the same context mating will supply.
    certificates = certify_population(
        [accs[a] for a in agent_ids],
        top_frac=args.certify_top_frac,
        floor=args.certify_floor,
        num_classes=args.num_classes,
    )
    cert_by_id = {a: set(c) for a, c in zip(agent_ids, certificates)}
    trained_on = {a: read_meta(population_dir, a).get('trained_on', [])
                  for a in agent_ids}

    population_info = []
    for agent_id in agent_ids:
        record = summarise(accs[agent_id], overalls[agent_id],
                           cert_by_id[agent_id])
        record['Model Name'] = agent_id
        record['Certificate'] = cert_by_id[agent_id]
        population_info.append(record)
    context = mating_fn.context_features(population_info, args)

    pairs = sample_pairs(agent_ids, args.phase_c_pairs, args.seed)
    print(f'[phase c] {len(agent_ids)} agents, measuring {len(pairs)} pairs '
          f'({2 * len(pairs)} directed merges)')

    shares_rows, outcome_rows = [], []
    with torch.no_grad():
        for a, b in tqdm(pairs, desc='phase c: measuring merges'):
            a_sees_b, b_sees_a = directional_pair_stats(
                states[a], states[b], core,
                eta=args.probe_eta,
                svd_energy=args.probe_svd_energy,
                basis_energy=args.probe_basis_energy,
                head_prefix=head_prefix,
                weighting=args.probe_layer_weighting,
            )
            # shares[base] is what the DONOR brings the base, which is the
            # quantity mating scores on, so it is the quantity fitted here.
            for (base, donor), shares in (((a, b), a_sees_b),
                                          ((b, a), b_sees_a)):
                child_sd = globa_merge_pair(
                    states[base], states[donor], core,
                    classes_base=trained_on.get(base),
                    classes_donor=trained_on.get(donor),
                    preset=args.globa_preset, head=args.globa_head,
                    eta=args.probe_eta, svd_energy=args.probe_svd_energy,
                    basis_energy=args.probe_basis_energy,
                    head_prefix=head_prefix)
                child = build_model(args, args.num_classes)
                child.load_state_dict(child_sd, strict=False)
                reset_bn_stats(child, train_loader)
                budget.count_forward_train(1)

                per_class, _overall = evaluate_all_classes(
                    child, val_loader, args.num_classes)
                ret = pair_retention(accs[base], accs[donor], per_class)
                # 'distinctive_b' is the DONOR's distinctive retention when the
                # base is passed first -- how much of the partner came across.
                transfer = ret['distinctive_b']

                shares_rows.append([float(shares[t]) for t in TYPES])
                outcome_rows.append(float(transfer))
                logger.log_train({
                    'stage': 'phase_c_pair',
                    'budget': round(budget.spent, 4),
                    'base': base,
                    'donor': donor,
                    'transfer': transfer,
                    'balanced': ret['balanced'],
                    **{f'energy_{t}': float(shares[t]) for t in TYPES},
                })
                del child, child_sd
            torch.cuda.empty_cache()

    samples = [{
        'context': context,
        'shares': np.array(shares_rows, dtype=np.float64),
        'outcome': np.array(outcome_rows, dtype=np.float64),
    }]

    net, history = mating_fn.train(
        samples, hidden=args.phase_c_hidden, epochs=args.phase_c_epochs,
        lr=args.phase_c_lr, seed=args.seed)

    coeffs = net.coefficients(context)
    path = mating_fn.default_path(args)
    mating_fn.save(net, path, meta={
        'n_pairs': len(pairs),
        'n_samples': len(outcome_rows),
        'context_fields': list(mating_fn.CONTEXT_FIELDS),
        'context': [float(v) for v in context],
        'coefficients': coeffs,
        'outcome': 'transfer',
        'loss_history': history,
        'seconds': round(time.time() - start, 1),
    })

    logger.log_train({
        'stage': 'phase_c',
        'budget': round(budget.spent, 4),
        'n_pairs': len(pairs),
        'n_samples': len(outcome_rows),
        'coefficients': coeffs,
        'context': {k: float(v) for k, v
                    in zip(mating_fn.CONTEXT_FIELDS, context)},
        'seconds': round(time.time() - start, 1),
    })
    print('[phase c] fitted coefficients: '
          + '  '.join(f'{t}={coeffs[t]:+.3f}' for t in TYPES))
    print(f'[phase c] wrote {path}  ({time.time() - start:.1f}s)')
    return path


# --------------------------------------------------------------------------- #
# --mating-mode learn
# --------------------------------------------------------------------------- #

def _measure_population(states, accs, trained_on, core, head_prefix, args,
                        budget, data, label, logger):
    """Every ordered pair of one population: profile in, offspring quality out.

    ALL pairs, both directions. --globa-with learned samples because it only
    needs enough points to fit six coefficients; this fits a network on the
    profile itself and needs the whole ranking, which is also the target.
    """
    agent_ids = list(states)
    val_loader = data.val_loader()
    train_loader = data.train_loader()
    profiles, outcomes = [], []

    pairs = list(itertools.combinations(agent_ids, 2))
    with torch.no_grad():
        for a, b in tqdm(pairs, desc=f'phase c [{label}]: all pairs'):
            a_sees_b, b_sees_a = directional_pair_stats(
                states[a], states[b], core,
                eta=args.probe_eta,
                svd_energy=args.probe_svd_energy,
                basis_energy=args.probe_basis_energy,
                head_prefix=head_prefix,
                weighting=args.probe_layer_weighting,
            )
            for (base, donor), shares in (((a, b), a_sees_b),
                                          ((b, a), b_sees_a)):
                child_sd = globa_merge_pair(
                    states[base], states[donor], core,
                    classes_base=trained_on.get(base),
                    classes_donor=trained_on.get(donor),
                    preset=args.globa_preset, head=args.globa_head,
                    eta=args.probe_eta, svd_energy=args.probe_svd_energy,
                    basis_energy=args.probe_basis_energy,
                    head_prefix=head_prefix)
                child = build_model(args, args.num_classes)
                child.load_state_dict(child_sd, strict=False)
                reset_bn_stats(child, train_loader)
                budget.count_forward_train(1)

                per_class, _overall = evaluate_all_classes(
                    child, val_loader, args.num_classes)
                ret = pair_retention(accs[base], accs[donor], per_class)
                transfer = ret['distinctive_b']

                profiles.append([float(shares[t]) for t in TYPES])
                outcomes.append(float(transfer))
                logger.log_train({
                    'stage': 'phase_c_pair',
                    'population': label,
                    'budget': round(budget.spent, 4),
                    'base': base,
                    'donor': donor,
                    'transfer': transfer,
                    'balanced': ret['balanced'],
                    **{f'energy_{t}': float(shares[t]) for t in TYPES},
                })
                del child, child_sd
            torch.cuda.empty_cache()
    return np.array(profiles, dtype=np.float64), np.array(outcomes, dtype=np.float64)


def _population_from_backbone(args, data, budget, logger, draw_seed, label):
    """A fresh society from the SAME phase-A backbone.

    Scalability is the point. Phase A is the expensive half of pretrain and is
    already paid for, so another society costs only a short phase B: new class
    subsets, a brief finetune, nothing else. Reusing the backbone is also the
    right invariance -- the rule should carry across societies, not have to
    carry across backbones it never saw.
    """
    from sesil.pretrain import backbone_dir, class_subsets, phase_budgets
    from sesil.mutation import mutate
    from sesil.ssl import reset_classifier

    # Same file _load_core reads, loaded the same way, so an extra population
    # starts from exactly the backbone the run's own agents started from.
    path = os.path.join(backbone_dir(args), f'{args.arch}.pth.tar')
    if not os.path.exists(path):
        raise SystemExit(
            f'--mating-mode learn needs the phase-A backbone at {path} to '
            f'build extra populations, and it is not there. It requires '
            f'--pretrain-mode ssl with --phase-a-ratio > 0.')
    backbone = torch.load(path, map_location='cpu')

    _phase_a, phase_b = phase_budgets(args)
    per_agent = phase_b / max(args.pop_size, 1)
    sample_budget = int(round(per_agent * budget.train_set_size))

    # Different subsets per extra population: class_subsets keys its draw off
    # args.seed, so the seed is shifted and put back.
    real_seed = args.seed
    args.seed = draw_seed
    try:
        subsets = class_subsets(args, data.num_classes)
    finally:
        args.seed = real_seed

    val_loader = data.val_loader()
    states, accs, trained_on = {}, {}, {}
    head_prefix = None
    print(f'[phase c] building population {label}: {args.pop_size} agents '
          f'x {per_agent:.3f} epoch-equiv from the phase-A backbone')

    for index in range(args.pop_size):
        split = subsets[index]
        model = build_model(args, args.num_classes)
        model.load_state_dict(backbone, strict=False)
        reset_classifier(model)
        model = model.train()
        train_loader = data.train_loader(classes=split)
        budget.spend_samples('phase_c', sample_budget, epochs=1)
        model, _acc = mutate(model, train_loader, val_loader,
                             sample_budget=sample_budget)
        per_class, _overall = evaluate_all_classes(
            model, val_loader, args.num_classes)
        agent_id = f'{label}_agent_{index:03d}'
        accs[agent_id] = per_class
        trained_on[agent_id] = list(split)
        states[agent_id] = {k: v.detach().cpu()
                            for k, v in model.state_dict().items()}
        if head_prefix is None:
            head_prefix = classifier_name(model) + '.'
        del model
    return states, accs, trained_on, head_prefix


def run_learn_phase_c(args, budget, data, logger):
    """Fit the pair-quality net across several populations."""
    start = time.time()
    population_dir = generation_dir(args, 0)
    agent_ids = list_population(population_dir)
    if len(agent_ids) < 2:
        raise RuntimeError('Phase C needs at least two agents.')

    core = _load_core(args)
    raw_config = build_raw_config(args)
    raw_config['model']['dir'] = population_dir
    val_loader = data.val_loader()

    # Population 0 is the run's own, already paid for.
    states, accs, trained_on = {}, {}, {}
    head_prefix = None
    with torch.no_grad():
        for agent_id in tqdm(agent_ids, desc='phase c [run]: evaluating parents'):
            model = _load_agent(agent_id, raw_config, data, budget)
            per_class, _overall = evaluate_all_classes(
                model, val_loader, args.num_classes)
            accs[agent_id] = per_class
            states[agent_id] = {k: v.detach().cpu()
                                for k, v in model.state_dict().items()}
            if head_prefix is None:
                head_prefix = classifier_name(model) + '.'
            del model
    trained_on = {a: read_meta(population_dir, a).get('trained_on', [])
                  for a in agent_ids}

    n_pop = max(int(args.phase_c_populations), 1)
    print(f'[phase c] learn mode: {n_pop} population(s), '
          f'{args.pop_size * (args.pop_size - 1)} ordered pairs each')

    samples = []
    prof, out = _measure_population(states, accs, trained_on, core,
                                    head_prefix, args, budget, data, 'run',
                                    logger)
    samples.append({'shares': prof, 'outcome': out})

    for extra in range(1, n_pop):
        label = f'extra{extra}'
        st, ac, tr, hp = _population_from_backbone(
            args, data, budget, logger, draw_seed=args.seed + 1000 * extra,
            label=label)
        prof, out = _measure_population(st, ac, tr, core, hp or head_prefix,
                                        args, budget, data, label, logger)
        samples.append({'shares': prof, 'outcome': out})
        del st

    net, history = mating_fn.train_quality(
        samples, hidden=args.phase_c_hidden, epochs=args.phase_c_epochs,
        lr=args.phase_c_lr, seed=args.seed)

    all_out = np.concatenate([s['outcome'] for s in samples])
    with torch.no_grad():
        all_pred = np.concatenate([
            net(torch.tensor(s['shares'], dtype=torch.float32)).numpy()
            for s in samples])
    path = mating_fn.quality_path(args)
    meta = {
        'populations': len(samples),
        'pairs_per_population': [int(len(s['outcome'])) for s in samples],
        'outcome': 'transfer',
        'target_mean': float(all_out.mean()),
        'target_quantiles': {str(q): float(np.percentile(all_out, q))
                             for q in (10, 25, 50, 75, 90)},
        'pred_mean': float(all_pred.mean()),
        'pred_quantiles': {str(q): float(np.percentile(all_pred, q))
                           for q in (10, 25, 50, 75, 90)},
        'loss_history': history,
        'seconds': round(time.time() - start, 1),
    }
    mating_fn.save_quality(net, path, meta=meta)
    logger.log_train({'stage': 'phase_c', 'budget': round(budget.spent, 4),
                      **meta})
    print(f'[phase c] measured quality  '
          + '  '.join(f'p{q}={meta["target_quantiles"][str(q)]:.3f}'
                      for q in (10, 25, 50, 75, 90)))
    print(f'[phase c] predicted quality '
          + '  '.join(f'p{q}={meta["pred_quantiles"][str(q)]:.3f}'
                      for q in (10, 25, 50, 75, 90)))
    print(f'[phase c] wrote {path}  ({time.time() - start:.1f}s)')
    print(f'[phase c] --mate-floor is compared against these predictions; '
          f'a floor above p90 leaves almost everyone a loner.')
    return path
