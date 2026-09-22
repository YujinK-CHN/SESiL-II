"""
The SESiL generation loop.

One generation is:

    evaluate  -- per-class fitness of every individual in the population
    mate      -- pair individuals by complementary skills (sesil.selection)
    merge     -- crossover each couple into one offspring (sesil.merge)
    mutate    -- finetune each offspring (sesil.mutation)

Loners (individuals nobody reciprocated) carry forward unchanged, so the
population size is preserved across generations.

All eight of the original evolutionary_*.py scripts were this same loop with a
different merge function or selection rule hardcoded in the middle; both are now
arguments resolved through sesil.registry.
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

def evaluate_population(model_ids, raw_config, args, csv_file, generation):
    """Per-class fitness for every individual. Drives mate selection."""
    population_info = []

    for model_id in tqdm(model_ids, desc=f'Gen {generation}: evaluating population'):
        inject_model(raw_config, model_id)
        config = prepare_experiment_config(raw_config)

        train_loader = config['data']['train']['full']
        base_model = config['models']['bases'][0]
        reset_bn_stats(base_model, train_loader)

        labels_str = labels_of(model_id)
        results = evaluate_individual(base_model, config, labels_str, args.num_classes)

        _log(results, csv_file,
             Generation=generation, Stage='population', Name=labels_str,
             Merger=args.merger, Selection=args.selection, Seed=args.seed)

        results['Model Name'] = model_id     # selection keys on the id, not labels
        population_info.append(results)

    return population_info


def breed(pairs, loners, raw_config, args, csv_file, generation):
    """Merge every couple, carry every loner. Returns {label_key: model}."""
    offspring = {}
    seen_keys = set()

    for pair in tqdm(pairs, desc=f'Gen {generation}: merging pairs'):
        merged, config = merge_pair(pair, raw_config, args)

        results = evaluate_merged(merged, config, eval_type=args.eval_type)
        _log(results, csv_file,
             Generation=generation, Stage='offspring',
             Parents='+'.join(labels_of(p) for p in pair),
             Time=merged.compute_transform_time,
             Merger=args.merger, Selection=args.selection, Seed=args.seed)

        labels = '_'.join(labels_of(p) for p in pair).split('_')
        offspring[build_unique_key(labels, seen_keys)] = merged

    for loner in tqdm(loners, desc=f'Gen {generation}: carrying loners'):
        inject_model(raw_config, loner)
        config = prepare_experiment_config(raw_config)

        train_loader = config['data']['train']['full']
        base_model = config['models']['bases'][0]
        reset_bn_stats(base_model, train_loader)

        results = evaluate_merged(base_model, config, eval_type=args.eval_type)
        _log(results, csv_file,
             Generation=generation, Stage='loner', Parents=labels_of(loner),
             Time=0.0, Merger=args.merger, Selection=args.selection, Seed=args.seed)

        labels = labels_of(loner).split('_')
        offspring[build_unique_key(labels, seen_keys)] = base_model

    return offspring


def mutate_and_save(offspring, args, next_dir):
    """Finetune each offspring and write it out as the next generation."""
    os.makedirs(next_dir, exist_ok=True)

    if args.no_mutation:
        for label_key, model in offspring.items():
            save_individual(model, next_dir, label_key, args.arch)
        return

    train_loader, test_loader, _ = get_loaders(args)

    for label_key, model in offspring.items():
        print(f'[mutate] {label_key}')
        model, _ = mutate(model, train_loader, test_loader, epochs=args.mutate_epochs)
        path = save_individual(model, next_dir, label_key, args.arch)
        print(f'[mutate] saved -> {path}')


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #

def run_evolution(args):
    """Run --generations generations, starting from --start-gen."""
    raw_config = build_raw_config(args)
    selection_fn = get_selection_fn(args.selection)

    csv_file = os.path.join(args.run_dir, 'results.csv')
    os.makedirs(args.run_dir, exist_ok=True)

    for generation in range(args.start_gen, args.start_gen + args.generations):
        print(f'\n============== Generation {generation} ==============\n')

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
                model_ids, raw_config, args, csv_file, generation)

            pairs, loners = selection_fn(population_info, args)
            print(f'[gen {generation}] {len(pairs) // 2} couples, {len(loners)} loners')

            offspring = breed(pairs, loners, raw_config, args, csv_file, generation)
            print(f'[gen {generation}] produced {len(offspring)} offspring')

        mutate_and_save(offspring, args, next_dir)

    print(f'\nDone. Results: {csv_file}')
