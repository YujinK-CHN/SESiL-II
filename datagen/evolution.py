"""
Dataset generator: a copy of sesil/evolution.py, modified to collect samples.

The run evolves exactly as sesil/evolution.py would. On top of that, every
generation it builds and measures the child of EVERY ordered pair (base,
donor) and writes one row per pair to <run_dir>/samples.csv:

    generation, n_generations, location, share_A .. share_D_minus,
    child_val_overall, type, head

location = generation / (n_generations - 1): 0 at the first generation, 1 at
the last. The total is known only when the run
ends, so rows go to samples.partial.csv during the run and samples.csv is
written at the end.

Changes from sesil/evolution.py, all marked [datagen]:
  * the six-share profile of every ordered pair is computed once per
    generation and reused by GLOBA mating (same numbers, not done twice);
  * _breed_globa carries each bred child's parents and accuracy, so those
    children are reused rather than rebuilt;
  * _collect_pair_children + the samples.csv writers; called after breeding;
  * only GLOBA mating and GLOBA merging are kept: the other mating modes,
    the other mergers, learn mode and phase C are removed, since
    generate_dataset.py refuses them.

The original module docstring follows.

The SESiL generation loop.

One generation is:

    evaluate   -- every agent on the whole label space, over the VALIDATION split
    certify    -- rank the population per class; grant proficiency certificates
    mate       -- pair agents by complementary certificates
    merge      -- crossover: one couple yields one child per parent
    mutate     -- finetune each agent on what it is licensed to train on

Agents have no names. What an agent may train on comes from its certificate,
which is re-measured from scratch every generation (sesil/certificate.py).
Offspring start from the union of their parents' certificates and are
re-certified from their own performance one generation later, so an inherited
certificate can misdirect at most a single round of training before it
self-corrects.

Loners -- agents nobody reciprocated -- carry forward unchanged and earn one
random uncertified class to explore, which is the only way a class enters an
agent's repertoire other than through merging.

Two splits, two purposes. Everything above runs on VALIDATION, because
certification decides what gets trained and so belongs inside the optimisation
loop. The TEST split is touched only by sesil.evaluator, which fires on budget
watermarks so SESiL and the baseline produce directly comparable curves.

The loop runs until the training budget is exhausted rather than for a fixed
number of generations. See sesil/budget.py.
"""

import os
import time

import numpy as np
import torch
from tqdm.auto import tqdm

from config import build_raw_config
from utils import prepare_experiment_config, reset_bn_stats

from sesil.budget import (
    FORWARD_TRAIN_PASSES_PER_EVAL,
    FORWARD_TRAIN_PASSES_PER_LONER,
    samples_for,
)
from sesil.certificate import (
    certify_population,
    coverage,
    inherit,
    training_classes,
)
from sesil.fitness import evaluate_all_classes, summarise
from sesil.merge import point_at
from sesil.mutation import mutate
from sesil.population import (generation_dir, list_population,
                              prune_generations, save_agent)
from sesil.selection import (_globa_value, pair_profiles,   # [datagen]
                             select_mates, strength_matrix)
from sesil.ssl import classifier_name


def _load_agent(agent_id, raw_config, data, budget):
    """Load one agent from disk, BN statistics recalibrated on the train split."""
    point_at(raw_config, [agent_id])
    config = prepare_experiment_config(raw_config)

    model = config['models']['bases'][0]
    reset_bn_stats(model, data.train_loader())
    budget.count_forward_train(FORWARD_TRAIN_PASSES_PER_EVAL)
    return model, config


# --------------------------------------------------------------------------- #
# Stages
# --------------------------------------------------------------------------- #

def evaluate_and_certify(agent_ids, raw_config, args, data, budget, logger, generation):
    """Measure every agent on all classes, then grant certificates by ranking.

    Evaluation covers the whole label space, not just what an agent was trained
    on: certification ranks agents against each other per class, so an agent's
    accuracy on classes it has never seen is exactly what decides it is not
    proficient in them.

    Runs on VALIDATION -- this ranking drives mating and licensing, so it is
    part of the optimisation and must not touch the test split.
    """
    val_loader = data.val_loader()
    per_class_accuracy = []
    overalls = []
    models = []

    for agent_id in tqdm(agent_ids, desc=f'Gen {generation}: evaluating'):
        model, _config = _load_agent(agent_id, raw_config, data, budget)
        per_class, overall = evaluate_all_classes(model, val_loader, args.num_classes)

        per_class_accuracy.append(per_class)
        overalls.append(overall)
        models.append(model)

    certificates = certify_population(
        per_class_accuracy,
        top_frac=args.certify_top_frac,
        floor=args.certify_floor,
        num_classes=args.num_classes,
    )

    # Per-class weights mate selection will use, resolved once for the whole
    # population so every scoring mode shares one code path.
    strengths = strength_matrix(args.cert_with, per_class_accuracy, args.num_classes)

    population_info = []
    for agent_id, per_class, overall, cert, strength in zip(
            agent_ids, per_class_accuracy, overalls, certificates, strengths):
        record = summarise(per_class, overall, cert)
        logger.log_train({
            'stage': 'population',
            'generation': generation,
            'budget': round(budget.spent, 4),
            'agent': agent_id,
            'certificate': sorted(cert),
            'val_overall': record['Joint'],
            'val_certified_avg': record['Certified Avg'],
            'val_per_class': [round(v, 5) for v in per_class],
        })
        record['Model Name'] = agent_id      # selection keys on the id
        record['Certificate'] = cert
        record['Strength'] = strength
        # Kept so the evaluator can route on validation without measuring it
        # again: this is the same vector, already paid for.
        record['ValPerClass'] = per_class
        population_info.append(record)

    stats = coverage(certificates, args.num_classes)
    logger.log_train({
        'stage': 'certification',
        'generation': generation,
        'budget': round(budget.spent, 4),
        **stats,
    })
    print(f'[gen {generation}] certificates: '
          f'mean {stats["cert_size_mean"]:.1f} classes '
          f'(min {stats["cert_size_min"]}, max {stats["cert_size_max"]}), '
          f'{stats["classes_covered"]}/{args.num_classes} classes covered, '
          f'{stats["uncertified_agents"]} uncertified')

    return population_info, models


SAMPLE_COLUMNS = (['generation', 'n_generations', 'location']
                  + [f'share_{t}' for t in ('A', 'B', 'C', 'E', 'D_plus', 'D_minus')]
                  + ['child_val_overall', 'type', 'head'])


def _samples_paths(args):
    """[datagen] (rows during the run, final file)."""
    return (os.path.join(args.run_dir, 'samples.partial.csv'),
            os.path.join(args.run_dir, 'samples.csv'))


def _append_samples(path, rows):
    """[datagen] Append rows; n_generations / location are filled at the end."""
    import csv
    new = not os.path.exists(path)
    with open(path, 'a', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=SAMPLE_COLUMNS)
        if new:
            w.writeheader()
        w.writerows(rows)


def _finalize_samples(partial, final):
    """[datagen] Fill n_generations and location = generation / (n_generations - 1).

    n_generations is read from the rows (every generation 0 .. last has its
    own), so it is right however many jobs the run was split across. 0 at the
    first generation, 1 at the last; 0 for a one-generation run. Returns
    (rows, n_generations).
    """
    import csv
    if not os.path.exists(partial):
        return 0, 0
    with open(partial, newline='', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    n_generations = max(int(r['generation']) for r in rows) + 1 if rows else 0
    for r in rows:
        r['n_generations'] = n_generations
        r['location'] = (int(r['generation']) / (n_generations - 1)
                         if n_generations > 1 else 0.0)
    with open(final + '.tmp', 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=SAMPLE_COLUMNS)
        w.writeheader()
        w.writerows(rows)
    os.replace(final + '.tmp', final)
    os.remove(partial)
    return len(rows), n_generations


def _collect_pair_children(population_info, models, core, args, data,
                           generation, offspring, profiles):
    """[datagen] Measure the child of every ordered pair; append its samples.

    Each child merge(base, donor) is built exactly as _breed_globa builds a bred
    one -- same certificates, preset and head, BatchNorm recalibrated on the
    train split -- and measured on validation. Children the run just bred are
    reused. Pure observation: after breeding, no budget charged, and the
    Python / NumPy / torch random states are restored, so the run continues
    exactly as sesil/evolution.py would.
    """
    import random
    from sesil.globa_merge import merge as globa_merge
    from sesil.pretrain import build_model

    head_prefix = classifier_name(models[0]) + '.'
    states = {info['Model Name']:
              {k: v.detach().cpu() for k, v in m.state_dict().items()}
              for info, m in zip(population_info, models)}
    certs = {info['Model Name']: set(info['Certificate'])
             for info in population_info}
    ids = [info['Model Name'] for info in population_info]
    reuse = {(o['base_parent'], o['donor_parent']): float(o['val_overall'])
             for o in offspring if not o.get('is_loner') and 'base_parent' in o}

    saved = (random.getstate(), np.random.get_state(), torch.get_rng_state(),
             torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)
    rows = []
    t0 = time.time()
    try:
        train_loader, val_loader = data.train_loader(), data.val_loader()
        for base in tqdm(ids, desc=f'Gen {generation}: every ordered pair'):
            for donor in ids:
                if base == donor:
                    continue
                overall = reuse.get((base, donor))
                if overall is None:
                    child_sd = globa_merge(
                        states[base], states[donor], core,
                        classes_base=certs[base], classes_donor=certs[donor],
                        preset=args.globa_preset, head=args.merge_head,
                        eta=args.probe_eta, svd_energy=args.probe_svd_energy,
                        basis_energy=args.probe_basis_energy,
                        head_prefix=head_prefix)
                    child = build_model(args, args.num_classes)
                    child.load_state_dict(child_sd, strict=False)
                    reset_bn_stats(child, train_loader)
                    _, overall = evaluate_all_classes(child, val_loader,
                                                      args.num_classes)
                    del child, child_sd
                prof = profiles[(base, donor)]
                row = {'generation': generation,
                       'child_val_overall': round(float(overall), 6),
                       'type': args.globa_with, 'head': args.merge_head}
                row.update({f'share_{t}': float(prof[t])
                            for t in ('A', 'B', 'C', 'E', 'D_plus', 'D_minus')})
                rows.append(row)
    finally:
        random.setstate(saved[0])
        np.random.set_state(saved[1])
        torch.set_rng_state(saved[2])
        if saved[3] is not None:
            torch.cuda.set_rng_state_all(saved[3])
        del states
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    _append_samples(_samples_paths(args)[0], rows)
    print(f'[gen {generation}] {len(rows)} samples ({len(reuse)} children '
          f'reused from breeding) in {time.time() - t0:.0f}s')


def _breed_globa(pairs, loners, population_info, models, args, data,
                 budget, logger, generation, core):
    """Crossover with GLOBA's own asymmetric operator.

    Each couple is merged twice, swapping which parent is the base, so the two
    children lean to different parents by construction. No alignment pass and
    no activation statistics are needed -- the operator is linear algebra on
    the task vectors -- so the only training data touched is the BatchNorm
    recalibration each child needs.
    """
    from sesil.globa_merge import merge as globa_merge
    from sesil.pretrain import build_model
    from sesil.ssl import classifier_name

    if core is None:
        raise SystemExit(
            '--merger globa requires --pretrain-mode ssl.\n'
            'GLOBA merges task vectors (agent - core), and the core is the '
            'phase-A backbone. Without one there is no shared origin.')

    by_id = {p['Model Name']: p for p in population_info}
    model_by_id = dict(zip([p['Model Name'] for p in population_info], models))
    head_prefix = classifier_name(models[0]) + '.'

    val_loader = data.val_loader()
    train_loader = data.train_loader()
    offspring = []

    states = {name: {k: v.detach().cpu() for k, v in m.state_dict().items()}
              for name, m in model_by_id.items()}

    for pair in tqdm(pairs, desc=f'Gen {generation}: merging couples (globa)'):
        cert = inherit([by_id[p]['Certificate'] for p in pair])

        # Twice, swapping base and donor: two children, one leaning each way.
        directions = (pair, tuple(reversed(pair)))
        for head_index, (base, donor) in enumerate(directions):
            start = time.time()
            child_sd = globa_merge(
                states[base], states[donor], core,
                classes_base=by_id[base]['Certificate'],
                classes_donor=by_id[donor]['Certificate'],
                preset=args.globa_preset, head=args.merge_head,
                eta=args.probe_eta, svd_energy=args.probe_svd_energy,
                basis_energy=args.probe_basis_energy,
                head_prefix=head_prefix)
            merge_seconds = time.time() - start

            child = build_model(args, args.num_classes)
            child.load_state_dict(child_sd, strict=False)
            # The child owns neither parent's BatchNorm statistics, so they are
            # recomputed for the network as assembled.
            reset_bn_stats(child, train_loader)
            budget.count_forward_train(1)

            per_class, overall = evaluate_all_classes(child, val_loader,
                                                      args.num_classes)
            logger.log_train({
                'stage': 'offspring',
                'generation': generation,
                'budget': round(budget.spent, 4),
                'parents': list(pair),
                'head': head_index,
                'base_parent': base,
                'merger': f'globa_{args.globa_preset}_{args.merge_head}',
                'trunk_params': 0,
                'label_head_rows': None,
                'certificate': sorted(cert),
                'val_overall': overall,
                'val_per_class': [round(v, 5) for v in per_class],
                'merge_seconds': round(merge_seconds, 3),
            })
            offspring.append({
                'model': child,
                'certificate': cert,
                'parents': list(pair),
                'head': head_index,
                'is_loner': False,
                # Carried so --mutation-mode best/worse can pick an
                # exploration class from measured performance rather than at
                # random. Already computed just above, so this costs nothing.
                'per_class': per_class,
                # [datagen] so _collect_pair_children reuses this child.
                'base_parent': base, 'donor_parent': donor,
                'val_overall': overall,
            })

    del states

    for loner in tqdm(loners, desc=f'Gen {generation}: carrying loners'):
        budget.count_forward_train(FORWARD_TRAIN_PASSES_PER_LONER)
        offspring.append({
            'model': model_by_id[loner],
            'certificate': set(by_id[loner]['Certificate']),
            'parents': [loner],
            'head': None,
            'is_loner': True,
            'per_class': by_id[loner].get('Per Class'),
        })

    return offspring


def breed(pairs, loners, population_info, models, raw_config, args, data,
          budget, logger, generation, globa_core=None):
    """Merge every couple (GLOBA, once each way), carry every loner."""
    return _breed_globa(pairs, loners, population_info, models, args, data,
                        budget, logger, generation, globa_core)


def generation_mutation_cost(offspring, args):
    """Epoch-equivalents the next mutation step will cost.

    Every agent is granted --individual-budget regardless of how many classes
    it is licensed for, so the cost is the head count times that grant.
    """
    return len(offspring) * args.individual_budget


def mutate_and_save(offspring, args, data, next_dir, budget, logger, generation):
    """Finetune each agent on its licensed classes, then write the next generation.

    Two things are deliberate:

    - Data is restricted to certified classes, so skill acquisition stays
      attributable: an agent cannot learn a class it has not earned, except
      through the single exploration slot a loner receives.
    - Compute is NOT restricted by coverage. Each agent gets exactly
      --individual-budget sample-presentations, cycling its subset if that
      subset is small, so a narrow agent is not quietly starved relative to a
      broad one.
    """
    os.makedirs(next_dir, exist_ok=True)

    rng = np.random.default_rng(args.seed + generation)
    sample_budget = samples_for(args.individual_budget, budget.train_set_size)
    val_loader = data.val_loader()
    loader_cache = {}

    trained = []

    # The society's task space: every class the population is actually working
    # on this generation. Not the union of CERTIFICATES -- an agent with an
    # empty certificate still trains on its one drawn class, so this is never
    # empty even when certification collapses entirely. It is also what the
    # baseline replays, so it has to be what the society really studied, not
    # what it was licensed to study.
    society_space = set()
    society_explored = set()

    for index, child in enumerate(offspring):
        classes = training_classes(
            child['certificate'], args.num_classes,
            is_loner=child['is_loner'], rng=rng,
            mode=args.mutation_mode, per_class_accuracy=child.get('per_class'),
        )
        explored = sorted(set(classes) - set(child['certificate']))
        society_space.update(classes)
        society_explored.update(explored)

        meta = {
            'generation': generation + 1,
            'certificate': child['certificate'],
            'trained_on': classes,
            'explored': explored,
            'parents': child['parents'],
            'head': child.get('head'),
            'is_loner': child['is_loner'],
        }

        model = child['model']

        if not args.no_mutation:
            key = tuple(classes)
            if key not in loader_cache:
                loader_cache[key] = data.train_loader(classes=classes)
            train_loader = loader_cache[key]

            n_available = len(train_loader.dataset)
            cost = budget.spend_samples('mutation', sample_budget, epochs=1)

            note = f' (+explore {explored})' if explored else ''
            print(f'[mutate] agent_{index:03d}  train_on={classes}{note}  '
                  f'{sample_budget} presentations over {n_available} samples  '
                  f'cost={cost:.3f} epoch-equiv')

            model, _ = mutate(model, train_loader, val_loader,
                              sample_budget=sample_budget)

            logger.log_train({
                'stage': 'mutation',
                'generation': generation,
                'budget': round(budget.spent, 4),
                'agent': f'agent_{index:03d}',
                'trained_on': classes,
                'explored': explored,
                'cost': round(cost, 5),
                'available_samples': n_available,
            })

        save_agent(model, next_dir, index, args.arch, meta)
        trained.append(model)

    certified_space = sorted(society_space - society_explored)
    logger.log_train({
        'stage': 'society',
        'generation': generation,
        # The budget AFTER this generation's mutation, which is the point on
        # the x axis at which the society held this space.
        'budget': round(budget.spent, 4),
        'space': sorted(society_space),
        'size': len(society_space),
        'from_certificates': certified_space,
        'from_exploration': sorted(society_explored),
    })
    print(f'[gen {generation}] society space: {len(society_space)} classes '
          f'({len(certified_space)} certified, {len(society_explored)} explored)')

    return trained


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #

def _load_globa_core(args):
    """The phase-A backbone GLOBA measures task vectors against.

    Refused without a shared origin: a task vector is `agent - core`, and
    under --pretrain-mode none every agent starts from its own random
    initialisation, so there is no core and the decomposition would be
    describing initialisation noise.
    """
    needs = ['--mating-mode globa', '--merger globa']
    if args.pretrain_mode != 'ssl':
        raise SystemExit(
            f'{" and ".join(needs)} requires --pretrain-mode ssl.\n'
            'GLOBA works on task vectors (agent - core), and the core is the '
            'phase-A backbone. Without one there is no shared origin.')

    from sesil.pretrain import backbone_dir
    path = os.path.join(backbone_dir(args), f'{args.arch}.pth.tar')
    if not os.path.exists(path):
        raise SystemExit(f'No phase-A backbone at {path}. Did pretrain run?')
    return torch.load(path, map_location='cpu')


def run_evolution(args, budget, data, logger, evaluator):
    """Evolve until the training budget is exhausted."""
    raw_config = build_raw_config(args)
    globa_core = _load_globa_core(args)

    generation = args.start_gen
    completed = 0

    while completed < args.max_generations:
        if budget.exhausted:
            print(f'\nBudget exhausted after {completed} generation(s).')
            break

        print(f'\n============== Generation {generation} '
              f'({budget.remaining:.2f} epoch-equiv left) ==============\n')

        current_dir = generation_dir(args, generation)
        next_dir = generation_dir(args, generation + 1)

        agent_ids = list_population(current_dir)
        if not agent_ids:
            raise RuntimeError(
                f'No population found in {current_dir}. '
                f'Run pretrain first, or check --start-gen.'
            )
        raw_config['model']['dir'] = current_dir
        print(f'[gen {generation}] population of {len(agent_ids)} from {current_dir}')

        # Merging and evaluation need no gradients; mutation does.
        with torch.no_grad():
            population_info, models = evaluate_and_certify(
                agent_ids, raw_config, args, data, budget, logger, generation)

            # External evaluation on the held-out test split, on the budget
            # watermark -- the x-values the baseline's curve lines up with.
            # The validation matrix goes with the models so the evaluator can
            # decide WHICH agent to trust per class on validation, and only
            # then read that agent's test accuracy. Costs nothing: these
            # numbers were just measured for certification.
            evaluator.maybe_record(
                models, step=generation, step_kind='generation',
                val_per_class=[info['ValPerClass'] for info in population_info],
                certificates=[info['Certificate'] for info in population_info])

            # [datagen] the profile of every ordered pair: the samples' input,
            # and the GLOBA mating scores below are read from it.
            args.head_prefix = classifier_name(models[0]) + '.'
            ids = [info['Model Name'] for info in population_info]
            states = {info['Model Name']:
                      {k: v.detach().cpu() for k, v in m.state_dict().items()}
                      for info, m in zip(population_info, models)}
            t0 = time.time()
            profiles = pair_profiles(ids, states, globa_core, args)
            del states
            print(f'[gen {generation}] {len(profiles)} pair profiles '
                  f'({time.time() - t0:.1f}s)')

            # --mating-mode globa ranks every candidate by its --globa-with
            # share: the same scores as globa_score_matrix, from the profiles.
            # GLOBA ranks on normalised shares, which say nothing about HOW
            # MUCH a donor brings; --mate-floor gates on the unnormalised
            # energy outside D_plus instead.
            novelty = {} if args.mate_floor > 0 else None
            pair_scores = {a: {} for a in ids}
            for (a, b), prof in profiles.items():
                pair_scores[a][b] = _globa_value(prof, args.globa_with)
                if novelty is not None:
                    novelty.setdefault(a, {})[b] = float(prof.get('_novel', 0.0))

            mate_stats = {}
            pairs, loners = select_mates(population_info, args,
                                         scores=pair_scores, tiebreak=None,
                                         stats=mate_stats, gate=novelty)
            print(f'[gen {generation}] {len(pairs)} couples, {len(loners)} loners')
            scored = None

            logger.log_train({
                'stage': 'mating',
                'generation': generation,
                'budget': round(budget.spent, 4),
                'couples': [list(p) for p in pairs],
                'loners': loners,
                'globa_pairs_scored': scored,
                # Separates refusal under --mate-floor from ordinary
                # non-reciprocation: both leave an agent unpaired, and only
                # this distinguishes them after the fact.
                'mate_floor': args.mate_floor,
                'abstained': mate_stats.get('abstained', 0),      # agent-rounds
                'abstainers': mate_stats.get('abstainers', 0),    # agents
                'couples_per_round': mate_stats.get('couples_per_round'),
            })

            offspring = breed(pairs, loners, population_info, models, raw_config,
                              args, data, budget, logger, generation,
                              globa_core=globa_core)
            print(f'[gen {generation}] produced {len(offspring)} agents')

            # [datagen] after breeding, so the run's own children are made
            # exactly as in sesil/evolution.py.
            _collect_pair_children(population_info, models, globa_core, args,
                                   data, generation, offspring, profiles)

        # Stop before mutating if this generation would overrun the budget --
        # a partially-finetuned generation is not a meaningful result.
        cost = generation_mutation_cost(offspring, args)
        if not args.no_mutation and not budget.can_afford(cost):
            print(f'[gen {generation}] mutation would cost {cost:.2f} epoch-equiv '
                  f'but only {budget.remaining:.2f} remain -- stopping here.')
            break

        mutate_and_save(offspring, args, data, next_dir, budget, logger, generation)

        # gen_N+1 is now on disk, so everything below the retention window is
        # dead -- nothing reads a generation once the loop has moved past it.
        dropped = prune_generations(args, generation + 1, args.keep_generations)
        if dropped:
            print(f'[gen {generation}] pruned {len(dropped)} old generation(s) '
                  f'(--keep-generations {args.keep_generations}); gen_0 kept')

        generation += 1
        completed += 1

    # Final point, so a run always ends on a recorded value wherever the budget
    # happened to run out.
    final_dir = generation_dir(args, generation)
    final_ids = list_population(final_dir)
    if final_ids:
        raw_config['model']['dir'] = final_dir
        with torch.no_grad():
            final_models = [
                _load_agent(a, raw_config, data, budget)[0] for a in final_ids
            ]
            # Validation too, so the final point carries the routed metrics like
            # every other point. Without this the last sample of the curve is
            # the only one missing them, which breaks the series exactly where
            # a reader looks first.
            final_val = [
                evaluate_all_classes(m, data.val_loader(), args.num_classes)[0]
                for m in final_models
            ]
            final_certs = certify_population(
                final_val, top_frac=args.certify_top_frac,
                floor=args.certify_floor, num_classes=args.num_classes)
            evaluator.record(final_models, step=generation, step_kind='generation',
                             final=True, val_per_class=final_val,
                             certificates=final_certs)

    # [datagen] the total is known now: write samples.csv.
    n, n_gen = _finalize_samples(*_samples_paths(args))
    if n:
        print(f'[samples] {n} samples, {n_gen} generations -> '
              f'{_samples_paths(args)[1]}')

    print(f'\n{budget.report()}')
    print(f'\nDone after {completed} generation(s). Logs: {args.run_dir}')
    return completed
