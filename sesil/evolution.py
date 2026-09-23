"""
The SESiL generation loop.

One generation is:

    evaluate   -- every agent on the WHOLE label space
    certify    -- rank the population per class; grant proficiency certificates
    mate       -- pair agents by complementary certificates
    merge      -- crossover each couple into one offspring
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

The loop runs until the training budget is exhausted rather than for a fixed
number of generations. See sesil/budget.py.
"""

import os

import numpy as np
import torch
from tqdm.auto import tqdm

from config import build_raw_config
from utils import prepare_experiment_config, reset_bn_stats, write_to_csv

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
from sesil.data import get_loaders
from sesil.fitness import evaluate_all_classes, summarise
from sesil.merge import merge_pair, point_at
from sesil.mutation import mutate
from sesil.population import (
    generation_dir,
    inherited_certificate,
    list_population,
    save_agent,
)
from sesil.registry import get_selection_fn


def _log(results, csv_file, **extra):
    """Append one row, flattening anything that would break the csv."""
    row = {}
    for key, value in list(results.items()) + list(extra.items()):
        if isinstance(value, (list, tuple, set, frozenset, np.ndarray)):
            seq = sorted(value) if isinstance(value, (set, frozenset)) else value
            # utils.write_to_csv joins on ',', so values must not contain one.
            row[key] = '|'.join(str(v) for v in np.asarray(seq).ravel())
        else:
            row[key] = value
    write_to_csv(row, csv_file=csv_file)


# --------------------------------------------------------------------------- #
# Stages
# --------------------------------------------------------------------------- #

def evaluate_and_certify(agent_ids, raw_config, args, csv_file, generation, budget):
    """Measure every agent on all classes, then grant certificates by ranking.

    Evaluation has to cover the whole label space, not just what an agent was
    trained on: certification ranks agents against each other per class, so an
    agent's accuracy on classes it has never seen is exactly what decides it is
    not proficient in them.
    """
    per_class_accuracy = []
    overalls = []

    for agent_id in tqdm(agent_ids, desc=f'Gen {generation}: evaluating'):
        point_at(raw_config, [agent_id])
        config = prepare_experiment_config(raw_config)

        train_loader = config['data']['train']['full']
        model = config['models']['bases'][0]
        reset_bn_stats(model, train_loader)
        budget.count_forward_train(FORWARD_TRAIN_PASSES_PER_EVAL)

        per_class, overall = evaluate_all_classes(
            model, config['data']['test']['full'], args.num_classes)
        budget.count_forward_test(1)

        per_class_accuracy.append(per_class)
        overalls.append(overall)

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
        _log(record, csv_file,
             Generation=generation, Stage='population', Agent=agent_id,
             Certificate=cert,
             Merger=args.merger, Selection=args.selection, Seed=args.seed)
        record['Model Name'] = agent_id      # selection keys on the id
        record['Certificate'] = cert
        record['Strength'] = strength
        population_info.append(record)

    stats = coverage(certificates, args.num_classes)
    print(f'[gen {generation}] certificates: '
          f'mean {stats["cert_size_mean"]:.1f} classes '
          f'(min {stats["cert_size_min"]}, max {stats["cert_size_max"]}), '
          f'{stats["classes_covered"]}/{args.num_classes} classes covered, '
          f'{stats["uncertified_agents"]} uncertified')

    return population_info


def breed(pairs, loners, population_info, raw_config, args, csv_file,
          generation, budget):
    """Merge every couple, carry every loner.

    Returns a list of dicts: {model, certificate, parents, is_loner}.
    The certificate an offspring starts with is the union of its parents';
    a loner keeps its own.
    """
    by_id = {p['Model Name']: p for p in population_info}
    offspring = []

    for pair in tqdm(pairs, desc=f'Gen {generation}: merging pairs'):
        merged, config = merge_pair(pair, raw_config, args)
        budget.count_forward_train(FORWARD_TRAIN_PASSES_PER_MERGE)

        per_class, overall = evaluate_all_classes(
            merged, config['data']['test']['full'], args.num_classes)
        budget.count_forward_test(1)

        cert = inherit([by_id[p]['Certificate'] for p in pair])
        record = summarise(per_class, overall, cert)
        _log(record, csv_file,
             Generation=generation, Stage='offspring',
             Parents='+'.join(pair), Certificate=cert,
             Time=merged.compute_transform_time,
             Merger=args.merger, Selection=args.selection, Seed=args.seed)

        offspring.append({
            'model': merged,
            'certificate': cert,
            'parents': list(pair),
            'is_loner': False,
        })

    for loner in tqdm(loners, desc=f'Gen {generation}: carrying loners'):
        point_at(raw_config, [loner])
        config = prepare_experiment_config(raw_config)

        train_loader = config['data']['train']['full']
        model = config['models']['bases'][0]
        reset_bn_stats(model, train_loader)
        budget.count_forward_train(FORWARD_TRAIN_PASSES_PER_LONER)

        cert = set(by_id[loner]['Certificate'])
        record = summarise(by_id[loner]['Per Class'], by_id[loner]['Joint'], cert)
        _log(record, csv_file,
             Generation=generation, Stage='loner', Parents=loner, Certificate=cert,
             Time=0.0, Merger=args.merger, Selection=args.selection, Seed=args.seed)

        offspring.append({
            'model': model,
            'certificate': cert,
            'parents': [loner],
            'is_loner': True,
        })

    return offspring


def generation_mutation_cost(offspring, args):
    """Epoch-equivalents the next mutation step will cost.

    Every agent is granted --individual-budget regardless of how many classes
    it is licensed for, so the cost is the head count times that grant.
    """
    return len(offspring) * args.individual_budget


def mutate_and_save(offspring, args, next_dir, budget, generation, csv_file):
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
    loader_cache = {}

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
            'is_loner': child['is_loner'],
        }

        if args.no_mutation:
            save_agent(child['model'], next_dir, index, args.arch, meta)
            continue

        key = tuple(classes)
        if key not in loader_cache:
            loader_cache[key] = get_loaders(args, classes=classes)
        train_loader, test_loader, _ = loader_cache[key]

        n_available = len(train_loader.dataset)
        cost = budget.spend_samples('mutation', sample_budget, epochs=1)

        note = f' (+explore {explored})' if explored else ''
        print(f'[mutate] agent_{index:03d}  train_on={classes}{note}  '
              f'{sample_budget} presentations over {n_available} samples  '
              f'cost={cost:.3f} epoch-equiv')

        model, _ = mutate(child['model'], train_loader, test_loader,
                          sample_budget=sample_budget, seed=args.seed)
        save_agent(model, next_dir, index, args.arch, meta)

        _log({}, csv_file,
             Generation=generation, Stage='mutation',
             Agent=f'agent_{index:03d}', TrainedOn=classes, Explored=explored,
             Cost=round(cost, 4), Seed=args.seed)


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #

def run_evolution(args, budget):
    """Evolve until the training budget is exhausted."""
    raw_config = build_raw_config(args)
    selection_fn = get_selection_fn(args.selection)

    csv_file = os.path.join(args.run_dir, 'results.csv')
    os.makedirs(args.run_dir, exist_ok=True)

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
            population_info = evaluate_and_certify(
                agent_ids, raw_config, args, csv_file, generation, budget)

            pairs, loners = selection_fn(population_info, args)
            print(f'[gen {generation}] {len(pairs) // 2} couples, {len(loners)} loners')

            offspring = breed(pairs, loners, population_info, raw_config, args,
                              csv_file, generation, budget)
            print(f'[gen {generation}] produced {len(offspring)} agents')

        # Stop before mutating if this generation would overrun the budget --
        # a partially-finetuned generation is not a meaningful result.
        cost = generation_mutation_cost(offspring, args)
        if not args.no_mutation and not budget.can_afford(cost):
            print(f'[gen {generation}] mutation would cost {cost:.2f} epoch-equiv '
                  f'but only {budget.remaining:.2f} remain -- stopping here.')
            break

        mutate_and_save(offspring, args, next_dir, budget, generation, csv_file)

        generation += 1
        completed += 1

    print(f'\n{budget.report()}')
    print(f'\nDone after {completed} generation(s). Results: {csv_file}')
    return completed
