"""
The SESiL generation loop.

One generation is:

    evaluate  -- per-class fitness of every individual in the population
    mate      -- pair individuals by complementary skills (sesil.selection)
    merge     -- crossover each couple into one offspring (sesil.merge)
    mutate    -- finetune each offspring on the labels it inherited

Loners (individuals nobody reciprocated) carry forward unchanged, so the
population size is preserved across generations.

The loop runs until the training budget is exhausted rather than for a fixed
number of generations: each generation costs a different amount depending on
how much of the label space the population covers. See sesil/budget.py.
"""

import os

import numpy as np
import torch
from tqdm.auto import tqdm

from config import build_raw_config
from utils import (
    decode_labels,
    prepare_experiment_config,
    reset_bn_stats,
    split_str_to_ints,
    write_to_csv,
)

from sesil.budget import (
    FORWARD_TRAIN_PASSES_PER_EVAL,
    FORWARD_TRAIN_PASSES_PER_LONER,
    FORWARD_TRAIN_PASSES_PER_MERGE,
)
from sesil.data import get_loaders
from sesil.fitness import evaluate_individual, evaluate_merged
from sesil.merge import merge_pair
from sesil.mutation import mutate
from sesil.population import (
    build_unique_key,
    generation_dir,
    labels_of,
    list_population,
    save_individual,
)
from sesil.registry import get_selection_fn


def inject_model(raw_config, model_id):
    """Point the config at a single individual (the loner / evaluation case)."""
    model_name = raw_config['model']['name']
    raw_config['dataset']['class_splits'] = [split_str_to_ints(decode_labels(model_id))]
    raw_config['model']['bases'] = [
        os.path.join(raw_config['model']['dir'], model_id, f'{model_name}_v0.pth.tar')
    ]
    return raw_config


def _log(results, csv_file, **extra):
    """Append one row, flattening anything that would break the csv."""
    row = {}
    for key, value in list(results.items()) + list(extra.items()):
        if isinstance(value, (list, tuple, np.ndarray)):
            # utils.write_to_csv joins on ',', so lists must not contain one.
            row[key] = '|'.join(f'{float(v):.4f}' for v in np.asarray(value).ravel())
        else:
            row[key] = value
    write_to_csv(row, csv_file=csv_file)


# --------------------------------------------------------------------------- #
# Stages
# --------------------------------------------------------------------------- #

def evaluate_population(model_ids, raw_config, args, csv_file, generation, budget):
    """Per-class fitness for every individual. Drives mate selection."""
    population_info = []

    for model_id in tqdm(model_ids, desc=f'Gen {generation}: evaluating population'):
        inject_model(raw_config, model_id)
        config = prepare_experiment_config(raw_config)

        train_loader = config['data']['train']['full']
        base_model = config['models']['bases'][0]
        reset_bn_stats(base_model, train_loader)
        budget.count_forward_train(FORWARD_TRAIN_PASSES_PER_EVAL)
        budget.count_forward_test(2)   # evaluate_individual walks the test set twice

        labels_str = labels_of(model_id)
        results = evaluate_individual(base_model, config, labels_str, args.num_classes)

        _log(results, csv_file,
             Generation=generation, Stage='population', Name=labels_str,
             Merger=args.merger, Selection=args.selection, Seed=args.seed)

        results['Model Name'] = model_id     # selection keys on the id, not labels
        population_info.append(results)

    return population_info


def breed(pairs, loners, raw_config, args, csv_file, generation, budget):
    """Merge every couple, carry every loner.

    Returns {label_key: (model, labels)}. The label list is carried explicitly
    rather than parsed back out of the key, because build_unique_key may append
    a disambiguating suffix that is not a class id.
    """
    offspring = {}
    seen_keys = set()

    for pair in tqdm(pairs, desc=f'Gen {generation}: merging pairs'):
        merged, config = merge_pair(pair, raw_config, args)
        budget.count_forward_train(FORWARD_TRAIN_PASSES_PER_MERGE)

        results = evaluate_merged(merged, config, eval_type=args.eval_type)
        budget.count_forward_test(1)
        _log(results, csv_file,
             Generation=generation, Stage='offspring',
             Parents='+'.join(labels_of(p) for p in pair),
             Time=merged.compute_transform_time,
             Merger=args.merger, Selection=args.selection, Seed=args.seed)

        labels = '_'.join(labels_of(p) for p in pair).split('_')
        key = build_unique_key(labels, seen_keys)
        offspring[key] = (merged, sorted({int(c) for c in labels}))

    for loner in tqdm(loners, desc=f'Gen {generation}: carrying loners'):
        inject_model(raw_config, loner)
        config = prepare_experiment_config(raw_config)

        train_loader = config['data']['train']['full']
        base_model = config['models']['bases'][0]
        reset_bn_stats(base_model, train_loader)
        budget.count_forward_train(FORWARD_TRAIN_PASSES_PER_LONER)

        results = evaluate_merged(base_model, config, eval_type=args.eval_type)
        budget.count_forward_test(1)
        _log(results, csv_file,
             Generation=generation, Stage='loner', Parents=labels_of(loner),
             Time=0.0, Merger=args.merger, Selection=args.selection, Seed=args.seed)

        labels = labels_of(loner).split('_')
        key = build_unique_key(labels, seen_keys)
        offspring[key] = (base_model, sorted({int(c) for c in labels}))

    return offspring


def generation_mutation_cost(offspring, args):
    """Epoch-equivalents the next mutation step will cost.

    Mutation is restricted to the labels an individual actually inherited, so
    the cost is the summed label coverage rather than one full epoch each.
    CIFAR is class-balanced, so |labels| / num_classes is the exact data
    fraction.
    """
    total_fraction = sum(len(labels) for _, labels in offspring.values()) / args.num_classes
    return total_fraction * args.mutate_epochs


def mutate_and_save(offspring, args, next_dir, budget):
    """Finetune each offspring on its own labels, then write the next generation.

    Restricting the data to inherited labels is what keeps skill acquisition
    attributable to merging: an offspring never sees a class neither parent
    knew, so coverage can only grow through crossover.
    """
    os.makedirs(next_dir, exist_ok=True)

    if args.no_mutation:
        for label_key, (model, _labels) in offspring.items():
            save_individual(model, next_dir, label_key, args.arch)
        return

    loader_cache = {}

    for label_key, (model, labels) in offspring.items():
        key = tuple(labels)
        if key not in loader_cache:
            loader_cache[key] = get_loaders(args, classes=labels)
        train_loader, test_loader, _ = loader_cache[key]

        n_samples = len(train_loader.dataset)
        cost = budget.spend_samples('mutation', n_samples, epochs=args.mutate_epochs)
        print(f'[mutate] {label_key}  classes={labels}  '
              f'{n_samples} samples  cost={cost:.3f} epoch-equiv')

        model, _ = mutate(model, train_loader, test_loader, epochs=args.mutate_epochs)
        path = save_individual(model, next_dir, label_key, args.arch)
        print(f'[mutate] saved -> {path}')


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

        model_ids = list_population(current_dir)
        if not model_ids:
            raise RuntimeError(
                f'No population found in {current_dir}. '
                f'Run pretrain first, or check --start-gen.'
            )
        raw_config['model']['dir'] = current_dir
        print(f'[gen {generation}] population of {len(model_ids)} from {current_dir}')

        # Merging and evaluation need no gradients; mutation does.
        with torch.no_grad():
            population_info = evaluate_population(
                model_ids, raw_config, args, csv_file, generation, budget)

            pairs, loners = selection_fn(population_info, args)
            print(f'[gen {generation}] {len(pairs) // 2} couples, {len(loners)} loners')

            offspring = breed(
                pairs, loners, raw_config, args, csv_file, generation, budget)
            print(f'[gen {generation}] produced {len(offspring)} offspring')

        # Stop before mutating if this generation would overrun the budget --
        # a partially-finetuned generation is not a meaningful result.
        cost = generation_mutation_cost(offspring, args)
        if not args.no_mutation and not budget.can_afford(cost):
            print(f'[gen {generation}] mutation would cost {cost:.2f} epoch-equiv '
                  f'but only {budget.remaining:.2f} remain -- stopping here.')
            break

        mutate_and_save(offspring, args, next_dir, budget)

        generation += 1
        completed += 1

    print(f'\n{budget.report()}')
    print(f'\nDone after {completed} generation(s). Results: {csv_file}')
    return completed
