import os
import torch
import random
import time
from copy import deepcopy

from tqdm.auto import tqdm
import numpy as np
import pandas as pd

from utils import *
from model_merger import ModelMerge


torch.manual_seed(0)
random.seed(0)
np.random.seed(0)
import pandas as pd

def inject_model(config, model, ignore_bases=False):
    model_name = config['model']['name']
    config['dataset']['class_splits'] = [split_str_to_ints(decode_labels(model))]
    if not ignore_bases:
        config['model']['bases'] = [os.path.join(config['model']['dir'], model, f'{model_name}_v0.pth.tar')]
    return config

def sort_model_name_unique(name: str) -> str:
    nums = [int(x) for x in name.split("_")]
    nums = sorted(set(nums))
    return "_".join(map(str, nums))

def evaluate_model(eval_type, model, config, **opt_kwargs):
    """ Evaluate methods on arbitrary experiment kinds. """
    if opt_kwargs.get("opt_dataloader", None) is not None:
        loader = opt_kwargs["opt_dataloader"]
        num_classes = opt_kwargs["opt_classes"]
    else:
        loader = config['data']['test']['full']
        num_classes = len(config['data']['test']['class_names'])
        
    if eval_type == 'logits':    
        acc_overall, acc_avg, pertask_acc, perclass_acc = evaluate_logits_alltasks(
            model, loader, 
            splits=config['dataset']['class_splits'], 
            num_classes=num_classes
        )
        print(acc_overall)
        print(acc_avg)
        print(pertask_acc)
        print(perclass_acc)
        
    elif eval_type == 'clip':
        clip_features = load_clip_features(config['data']['test']['class_names'], get_device(model))
        class_vectors = [clip_features[split] for split in config['data']['train']['class_splits']]

        acc_overall, acc_avg, pertask_acc = evaluate_cliphead_alltasks(
            model, 
            loader, 
            class_vectors, config['data']['train']['class_splits'], 
            num_classes=num_classes
        )
    else:
        raise ValueError(f'Invalid eval_type: {eval_type}! Must be one of [logits, clip].')

    results = {'Joint': acc_overall, 'Per Task Avg': acc_avg, 'Per class Acc': perclass_acc}
    for task_idx, task_acc in enumerate(pertask_acc):
        results[f'Task {CONCEPT_TASKS[task_idx]}'] = task_acc

    return results

def evaluate_logits_alltasks(model, loader, splits, num_classes):
    model.eval()
    correct = 0
    total = 0
    
    # always store splits as list of lists
    splits = [list(split) for split in splits]
    print("check splits: ", splits)
    totals = [0] * num_classes
    corrects = [0] * num_classes

    device = get_device(model)

    # flatten all splits into one tensor of valid class ids
    all_splits = torch.tensor([cls for split in splits for cls in split], 
                              device=device, dtype=torch.long)
    print(all_splits)
    
    # map each class → task index
    task_map = {}
    for i, split in enumerate(splits):
        for _cls in split:
            task_map[_cls] = i
    
    task_map = [task_map.get(_cls, -1) for _cls in range(num_classes)]
    task_map = torch.tensor(task_map, device=device, dtype=torch.long)

    # ✅ represent splits as list of tensors (avoids rectangular/ragged problem entirely)
    splits_tensor = [torch.tensor(s, device=device, dtype=torch.long) for s in splits]

    with torch.no_grad(), autocast():
        for inputs, labels in tqdm(loader, 'Evaluating multihead head model'):
            inputs, labels = inputs.to(device), labels.to(device)
            
            # keep only samples belonging to classes in all_splits
            class_selector = torch.isin(labels, all_splits)
            inputs, labels = inputs[class_selector], labels[class_selector]

            batch_size = inputs.shape[0]
            if batch_size == 0:
                continue

            task_idx = task_map[labels]   # task id for each label
            outputs = model(inputs)

            if isinstance(outputs, list):
                # Filter out predictions on classes not in this split
                for i, split in enumerate(splits_tensor):
                    exclude_labels = torch.tensor(
                        [cls for cls in all_splits.tolist() if cls not in split.tolist()],
                        device=device, dtype=torch.long
                    )
                    if exclude_labels.numel() > 0:
                        outputs[i][:, exclude_labels] = -torch.inf
                outputs = torch.stack(outputs, dim=1)
                outputs2 = outputs.softmax(dim=-1).to(outputs.dtype).max(dim=-2)[0]
                outputs2[:, all_splits] += 2
                outputs = outputs[range(batch_size), task_idx, :]
            else:
                outputs2 = outputs.clone()
                for split in splits_tensor:
                    outputs2[:, split] = torch.softmax(
                        outputs2[:, split], dim=-1
                    ).to(outputs.dtype) + 2
            outputs2 = outputs2.argmax(dim=-1)

            # ✅ per-sample task-specific prediction (works for ragged splits)
            preds = []
            for out_vec, split in zip(outputs, [splits_tensor[i] for i in task_idx.tolist()]):
                idx = out_vec[split].argmax()   # find best index within this split
                preds.append(split[idx].item())
            outputs = torch.tensor(preds, device=device, dtype=torch.long)

            # update stats
            for gt, p, p2 in zip(labels, outputs, outputs2):
                totals[gt] += 1
                if gt == p:
                    corrects[gt] += 1
                if gt == p2:
                    correct += 1
                total += 1

    print("acc: ", np.asarray(corrects) / np.asarray(totals))
    split_accs = [0] * len(splits)
    
    for i, split in enumerate(splits):
        print("last split: ", split)
        split_total = 0
        for _cls in split:
            split_accs[i] += corrects[_cls]
            split_total += totals[_cls]
        split_accs[i] /= max(split_total, 1e-4)
    
    print("split_acc: ", split_accs)
                
    return correct / total, sum(split_accs) / len(split_accs), split_accs, np.asarray(corrects) / np.asarray(totals)


def build_unique_key(labels, seen_keys):
    """
    Build a short, unique key from labels with rotation + index suffix if needed.
    """
    sorted_labels = sorted(set(labels))
    base_key = "_".join(sorted_labels)

    key = base_key
    rotation = 0
    # First try rotation
    while key in seen_keys:
        rotation += 1
        rotated = sorted_labels[rotation % len(sorted_labels):] + \
                  sorted_labels[:rotation % len(sorted_labels)]
        key = "_".join(rotated)

        # If we rotated through all possibilities and still duplicate -> add suffix
        if rotation >= len(sorted_labels):
            suffix = 1
            while f"{base_key}_{suffix}" in seen_keys:
                suffix += 1
            key = f"{base_key}_{suffix}"
            break

    seen_keys.add(key)
    return key

def run_auxiliary_experiment(merging_fn, node_config, experiment_config, pairs, loners, device, csv_file):
    offsprings = {}
    seen_keys = set()  # track existing keys to handle duplicates

    # ---------------------------
    # Handle PAIRS
    # ---------------------------
    for pair in tqdm(pairs, desc='Evaluating Pairs...'):
        experiment_config = inject_pair(experiment_config, pair)
        config = prepare_experiment_config(raw_config)

        train_loader = config['data']['train']['full']
        base_models = [reset_bn_stats(base_model, train_loader) 
                       for base_model in config['models']['bases']]

        config['node'] = node_config
        Grapher = config['graph']
        graphs = [Grapher(deepcopy(base_model)).graphify() for base_model in base_models]

        Merge = ModelMerge(*graphs, device=device)

        Merge.transform(
            deepcopy(config['models']['new']),
            train_loader,
            transform_fn=get_merging_fn(merging_fn),
            metric_classes=config['metric_fns'],
            stop_at=node_config['stop_node'],
            **node_config['params']
        )

        reset_bn_stats(Merge, train_loader)

        results = evaluate_model(experiment_config['eval_type'], Merge, config)
        for idx, split in enumerate(pair):
            results[f'Split {CONCEPT_TASKS[idx]}'] = sort_model_name_unique(decode_labels(split))
        results['Time'] = Merge.compute_transform_time
        results['Merging Fn'] = merging_fn
        results['Model Name'] = config['model']['name']
        results.update(flatten_nested_dict(node_config, sep=' '))
        write_to_csv(results, csv_file=csv_file)
        print(results)

        # Build unique key by checking duplicates and inverting if needed
        pair = tuple(decode_labels(p) for p in pair)
        labels = "_".join(pair).split("_")
        merged_key = build_unique_key(labels, seen_keys)
        offsprings[merged_key] = Merge

    # ---------------------------
    # Handle LONERS
    # ---------------------------
    for loner in tqdm(loners, desc='Evaluating Loners...'):
        experiment_config = inject_model(experiment_config, loner)  # like inject_pair
        config = prepare_experiment_config(raw_config)

        train_loader = config['data']['train']['full']

        # Single model from bases
        base_model = reset_bn_stats(config['models']['bases'][0], train_loader)
        config['node'] = node_config

        results = evaluate_model(experiment_config['eval_type'], base_model, config)
        results['Split'] = sort_model_name_unique(decode_labels(loner))
        results['Time'] = 0.0
        results['Merging Fn'] = "None"
        results['Model Name'] = config['model']['name']
        results.update(flatten_nested_dict(node_config, sep=' '))
        write_to_csv(results, csv_file=csv_file)
        print(results)

        # Key is just the label subset (loners don’t duplicate)
        labels = decode_labels(loner).split("_")
        loner_key = build_unique_key(labels, seen_keys)
        offsprings[loner_key] = base_model
        #offsprings[decode_labels(loner)] = base_model

    return offsprings

if __name__ == "__main__":
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    config_name = 'cifar_evolution_resnet20'
    skip_pair_idxs = [0]
    model_width = 4
    raw_config = get_config_from_name(config_name, device=device)
    #print(raw_config)

    model_dir = raw_config['model']['dir']
    print(model_dir)

    experiment_configs = [
        # best experiment
        {'stop_node': 21, 'params':{'a': .0001, 'b': .075}},
        # {'stop_node': 21, 'params':{'a': 1., 'b': 1.}},
        # {'stop_node': None, 'params':{'a': 0.01, 'b': 1.0}, 'dataset': {'train_fraction': .0001, 'no_transform': False}}, 
        # Alpha Ablations
        # {'stop_node': None, 'params': {'a': .0, 'b': 1.}},
    ]
    
    model_name = raw_config['model']['name']
    models_ids = find_runable_pairs(model_dir, model_name, skip_pair_idxs=skip_pair_idxs)
    #run_pairs = find_runable_pairs(model_dir, model_name, skip_pair_idxs=skip_pair_idxs)
    file_id = "permute"
    csv_file = os.path.join(
        f'./checkpoints/merging_baseline',
        'zipit_merging_baseline.csv'
    )

    pairs = [()]
    with torch.no_grad():
        for node_config in experiment_configs:
            raw_config['dataset'].update(node_config.get('dataset', {}))

            # concept merging.
            offsprings = run_auxiliary_experiment(
                merging_fn='match_tensors_permute', # change here
                node_config=node_config, 
                experiment_config=raw_config, 
                pairs=pairs, 
                loners=[],
                device=device,
                csv_file=csv_file
            )
            
            print(f'\nWe collected {len(offsprings)} offsprings.\n')

    for model_name, model in offsprings.items():
        save_dir = os.path.join(f'./checkpoints/merging_baseline/{model_name}')
        print('Saving Base Model to:')
        print(save_dir)
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f'resnet20x{model_width}_v{len(os.listdir(save_dir))}.pth.tar')
        save_model(model, save_path)