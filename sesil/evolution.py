"""
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

import numpy as np
import torch
from tqdm.auto import tqdm

from config import build_raw_config
from utils import prepare_experiment_config, reset_bn_stats

from sesil.budget import (
    FORWARD_TRAIN_PASSES_PER_EVAL,
    FORWARD_TRAIN_PASSES_PER_LONER,
    FORWARD_TRAIN_PASSES_PER_MERGE,
    samples_for,
)
from sesil.certificate import (
    certify_population,
    coverage,
    inherit,
    strength_matrix,
    training_classes,
)
from sesil.fitness import evaluate_all_classes, summarise
from sesil.merge import extract_children, merge_couple, point_at
from sesil.mutation import mutate
from sesil.population import generation_dir, list_population, save_agent
from sesil.selection import select_mates


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
    strengths = strength_matrix(args.mate_score, per_class_accuracy, args.num_classes)

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


def breed(pairs, loners, population_info, models, raw_config, args, data,
          budget, logger, generation):
    """Merge every couple, carry every loner.

    A couple yields one child per parent -- with a stop node they differ by
    which parent's head is spliced onto the merged trunk, without one by how
    the merged weights are interpolated. Either way neither parent's
    contribution is discarded.
    """
    by_id = {p['Model Name']: p for p in population_info}
    model_by_id = dict(zip([p['Model Name'] for p in population_info], models))

    val_loader = data.val_loader()
    train_loader = data.train_loader()
    offspring = []

    for pair in tqdm(pairs, desc=f'Gen {generation}: merging couples'):
        merge, config = merge_couple(pair, raw_config, args, train_loader)
        budget.count_forward_train(FORWARD_TRAIN_PASSES_PER_MERGE)

        children = extract_children(merge, config, args, args.num_classes, train_loader)
        budget.count_forward_train(len(children))   # BN recalibration per child

        cert = inherit([by_id[p]['Certificate'] for p in pair])

        for head_index, (child, n_from_trunk) in enumerate(children):
            # Evaluate the model that will actually be saved, not the composite.
            per_class, overall = evaluate_all_classes(child, val_loader, args.num_classes)

            logger.log_train({
                'stage': 'offspring',
                'generation': generation,
                'budget': round(budget.spent, 4),
                'parents': list(pair),
                'head': head_index,
                'trunk_params': n_from_trunk,
                'certificate': sorted(cert),
                'val_overall': overall,
                'val_per_class': [round(v, 5) for v in per_class],
                'merge_seconds': merge.compute_transform_time,
            })

            offspring.append({
                'model': child,
                'certificate': cert,
                'parents': list(pair),
                'head': head_index,
                'is_loner': False,
            })

    for loner in tqdm(loners, desc=f'Gen {generation}: carrying loners'):
        model = model_by_id[loner]
        budget.count_forward_train(FORWARD_TRAIN_PASSES_PER_LONER)

        cert = set(by_id[loner]['Certificate'])
        logger.log_train({
            'stage': 'loner',
            'generation': generation,
            'budget': round(budget.spent, 4),
            'parents': [loner],
            'certificate': sorted(cert),
            'val_overall': by_id[loner]['Joint'],
        })

        offspring.append({
            'model': model,
            'certificate': cert,
            'parents': [loner],
            'head': None,
            'is_loner': True,
        })

    return offspring


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

    for index, child in enumerate(offspring):
        classes = training_classes(
            child['certificate'], args.num_classes,
            is_loner=child['is_loner'], rng=rng,
        )
        explored = sorted(set(classes) - set(child['certificate']))

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

    return trained


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #

def run_evolution(args, budget, data, logger, evaluator):
    """Evolve until the training budget is exhausted."""
    raw_config = build_raw_config(args)

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
            evaluator.maybe_record(models, step=generation, step_kind='generation')

            pairs, loners = select_mates(population_info, args)
            print(f'[gen {generation}] {len(pairs)} couples, {len(loners)} loners')
            logger.log_train({
                'stage': 'mating',
                'generation': generation,
                'budget': round(budget.spent, 4),
                'couples': [list(p) for p in pairs],
                'loners': loners,
            })

            offspring = breed(pairs, loners, population_info, models, raw_config,
                              args, data, budget, logger, generation)
            print(f'[gen {generation}] produced {len(offspring)} agents')

        # Stop before mutating if this generation would overrun the budget --
        # a partially-finetuned generation is not a meaningful result.
        cost = generation_mutation_cost(offspring, args)
        if not args.no_mutation and not budget.can_afford(cost):
            print(f'[gen {generation}] mutation would cost {cost:.2f} epoch-equiv '
                  f'but only {budget.remaining:.2f} remain -- stopping here.')
            break

        mutate_and_save(offspring, args, data, next_dir, budget, logger, generation)

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
            evaluator.record(final_models, step=generation, step_kind='generation', final=True)

    print(f'\n{budget.report()}')
    print(f'\nDone after {completed} generation(s). Logs: {args.run_dir}')
    return completed
