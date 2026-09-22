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

###################### Evaluation    #####################
def sort_model_name_unique(name: str) -> str:
    nums = [int(x) for x in name.split("_")]
    nums = sorted(set(nums))
    return "_".join(map(str, nums))


def inject_model(config, model, ignore_bases=False):
    model_name = config['model']['name']
    config['dataset']['class_splits'] = [split_str_to_ints(decode_labels(model))]
    if not ignore_bases:
        config['model']['bases'] = [os.path.join(config['model']['dir'], model, f'{model_name}_v0.pth.tar')]
    return config

def find_pairs(str_splits):
    pairs = []
    for i, str_split_i in enumerate(str_splits):
        try:
            split_i = set([int(k) for k in str_split_i.split('_')])
        except:
            continue
        for str_split_j in str_splits[i+1:]:
            try:
                split_j = set([int(k) for k in str_split_j.split('_')])
            except:
                continue
            if len(split_i.intersection(split_j)) == 0:
                pairs.append((str_split_i, str_split_j))
    return pairs


def find_runable_pairs(model_dir, model_name, skip_pair_idxs=[]):

    run_pairs = []
    '''
    valid_pairs = [pair for pair in find_pairs(os.listdir(model_dir)) if is_valid_pair(model_dir, pair, model_name)]
    for idx, pair in enumerate(valid_pairs):
        if idx in skip_pair_idxs:
            continue
        run_pairs += [pair]
    return run_pairs
    '''
    return os.listdir(model_dir) 


def evaluate_fitness(model_id, model, config):
    device = next(model.parameters()).device
    test_loader = config['data']['test']['full']

    num_classes = 10
    acc_per_class = [0.0] * num_classes

    '''subset_labels = [int(x) for x in model_id.split('_')]
    subset_labels.sort()'''
    subset_labels = sorted(set(int(x) for x in model_id.split('_')))
    print('Consider subset labels: ', subset_labels)

    model.eval()
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)

            # mask for only trained labels
            mask = torch.zeros_like(labels, dtype=torch.bool)
            for lab in subset_labels:
                mask |= (labels == lab)
            if mask.sum() == 0:
                continue

            images_masked = images[mask]
            labels_masked = labels[mask]

            outputs = model(images_masked)
            _, preds = outputs.max(1)

            # compute per-class accuracy only for subset_labels
            for lab in subset_labels:
                lab_mask = (labels_masked == lab)
                if lab_mask.sum() > 0:
                    acc_per_class[lab] += (preds[lab_mask] == labels_masked[lab_mask]).sum().item()

    # divide by total number of samples per class
    total_per_class = torch.zeros(num_classes)
    for images, labels in test_loader:
        for lab in subset_labels:
            total_per_class[lab] += (labels == lab).sum().item()

    for lab in subset_labels:
        if total_per_class[lab] > 0:
            acc_per_class[lab] /= total_per_class[lab]

    # joint accuracy over trained labels
    joint_acc = sum(acc_per_class[lab] for lab in subset_labels) / len(subset_labels)

    results = {
        'Joint': sum(acc_per_class) / len(acc_per_class),
        'Per Task Avg': joint_acc,
        'Model Name': model_id,
        'Per Class': acc_per_class.copy()
    }

    return results

'''
def evaluate_fitness(eval_type, model, config):
    device = next(model.parameters()).device
    test_loader = config['data']['test']['full']
    all_labels = set()
    for _, labels in test_loader:
        all_labels.update(labels.tolist())
    print(sorted(all_labels))
    
    num_classes = 10
    correct_per_class = torch.zeros(num_classes, device=device)
    total_per_class = torch.zeros(num_classes, device=device)
    
    model.eval()
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = outputs.max(1)
            
            for c in range(num_classes):
                mask = (labels == c)
                correct_per_class[c] += (preds[mask] == labels[mask]).sum()
                total_per_class[c] += mask.sum()
    
    acc_per_class = [(correct_per_class[c] / total_per_class[c]).item() if total_per_class[c] > 0 else 0.0
                     for c in range(num_classes)]
    
    results = {
        'Joint': sum(acc_per_class)/num_classes,
        'Per Task Avg': sum(acc_per_class)/num_classes,
        'Model Name': config['model']['name']
    }
    
    # Instead of task-specific splits, just store full per-class list
    results['Per Class'] = acc_per_class.copy()
    
    return results
'''
def run_evolutionary_evaluation(node_config, experiment_config, model_ids, device, csv_file):
    population_info = []
    for model_id in model_ids:
        print(model_id)
        print(decode_labels(model_id))
        ''''''
        experiment_config = inject_model(experiment_config, model_id)

        config = prepare_experiment_config(raw_config)

        train_loader = config['data']['train']['full']
        base_model = [reset_bn_stats(base_model, train_loader) for base_model in config['models']['bases']]
        config['node'] = node_config
        '''
        Grapher = config['graph']
        graphs = [Grapher(deepcopy(base_model)).graphify() for base_model in base_models]
        
        Merge = ModelMerge(*graphs, device=device)
        
        base_model[0].transform(
            deepcopy(config['models']['new']), 
            train_loader, 
            transform_fn=config['merging_fn'], 
            metric_classes=config['metric_fns'],
            stop_at=node_config['stop_node'],
            **node_config['params']
        )
        '''
        reset_bn_stats(base_model[0], train_loader)
        
        results = evaluate_fitness(decode_labels(model_id), base_model[0], config)
        print(results)
        results['Model Name'] = sort_model_name_unique(decode_labels(model_id)) # just for csv
        results.update(flatten_nested_dict(node_config, sep=' '))
        write_to_csv(results, csv_file=csv_file)
        results['Model Name'] = model_id
        population_info.append(results)

    return population_info
###################### Evaluation    #####################

###################### Mating Matrix #####################

def known_classes(fitness, threshold=0.5):
    """Return indices of classes this model knows (above threshold)."""
    return {i for i, acc in enumerate(fitness) if acc > threshold}


def mating_score(fitness_a, fitness_b, 
                 mode="threshold", 
                 threshold=0.5,
                 weight_extra=1.0, 
                 weight_common=0.1):
    """
    Compute how much A values B as a mate.

    Args:
        fitness_a: list of accuracies for model A (per class)
        fitness_b: list of accuracies for model B (per class)
        mode: "threshold" or "soft"
        threshold: cutoff for class considered 'known' (if threshold mode)
        weight_extra: weight for extra skills (B knows what A doesn't)
        weight_common: weight for common skills (overlap)
    """
    if mode == "threshold":
        known_a = known_classes(fitness_a, threshold)
        known_b = known_classes(fitness_b, threshold)

        extra_skills = known_b - known_a
        common_skills = known_a & known_b

        score = weight_extra * sum(fitness_b[i] for i in extra_skills)
        score += weight_common * sum(fitness_b[i] for i in common_skills)

    elif mode == "soft":
        # No hard cutoff, just weight by differences in accuracy
        score = 0.0
        for i, (acc_a, acc_b) in enumerate(zip(fitness_a, fitness_b)):
            if acc_b > 0:
                if acc_a < threshold:  # A is weak here
                    score += weight_extra * acc_b
                else:  # A is decent here too
                    score += weight_common * acc_b
    else:
        raise ValueError("mode must be 'threshold' or 'soft'")

    return score


def build_score_matrix(models, **kwargs):
    """
    Build a matrix of directional mating scores.
    kwargs are passed to mating_score.
    """
    scores = {}
    for model_a in models:
        fa = model_a["Per Class"]
        scores[model_a["Model Name"]] = {}
        for model_b in models:
            if model_a is model_b:
                continue
            fb = model_b["Per Class"]
            scores[model_a["Model Name"]][model_b["Model Name"]] = mating_score(fa, fb, **kwargs)
    return scores


def probabilistic_choice(score_dict):
    """Pick a mate from score_dict probabilistically."""
    if not score_dict:
        return None
    models = list(score_dict.keys())
    weights = np.array(list(score_dict.values()), dtype=float)
    if weights.sum() == 0:
        return random.choice(models)  # fallback
    probs = weights / weights.sum()
    return np.random.choice(models, p=probs)


def bidirectional_selection_pop_shrink(scores, num_pairs=None, max_retries=100):
    """
    Perform bidirectional probabilistic mate selection.
    
    Args:
        scores: dict of dicts [model_a][model_b] = score
        num_pairs: total pairs to produce (default = len(models))
        max_retries: number of times to retry if not enough pairs
    
    Returns:
        list of tuples (A, B) where A and B are model names
    """
    print("\nnum_pairs: ", num_pairs)
    models = list(scores.keys())
    if num_pairs is None:
        num_pairs = len(models)  # default: same size as population
        
    pairs = set()
    
    retries = 0
    while len(pairs) < num_pairs and retries < max_retries:
        retries += 1
        choices = {m: probabilistic_choice(scores[m]) for m in models}
        
        # keep only reciprocated matches
        for a, b in choices.items():
            if b is not None and choices.get(b) == a:
                pair = tuple(sorted((a, b)))
                pairs.add(pair)
    
    pairs = list(pairs)
    
    # If too few pairs, duplicate randomly until reaching num_pairs
    while len(pairs) < num_pairs-7:
        pairs.append(random.choice(pairs))
    
    return pairs

import random

def bidirectional_selection(scores, max_retries=100, return_loners=True):
    """
    Perform bidirectional probabilistic mate selection with population preservation.
    
    Args:
        scores: dict of dicts [model_a][model_b] = score
        max_retries: number of retries to find reciprocal pairs
        return_loners: if True, return both pairs and loners
    
    Returns:
        pairs, loners
            pairs: list of tuples (A, B) where each successful mating is duplicated
            loners: list of individuals that did not mate
    """
    models = list(scores.keys())
    N = len(models)

    pairs = []
    paired = set()
    retries = 0

    while retries < max_retries:
        retries += 1
        choices = {m: probabilistic_choice(scores[m]) for m in models if m not in paired}

        for a, b in choices.items():
            if b is not None and choices.get(b) == a:
                if a not in paired and b not in paired:
                    pair = tuple(sorted((a, b)))
                    # add two offspring (pair counted twice)
                    pairs.append(pair)
                    pairs.append(pair)
                    paired.update([a, b])

    loners = [m for m in models if m not in paired]

    # ✅ check population size invariant
    assert 2 * (len(pairs) // 2) + len(loners) == N, \
        f"Population size mismatch: {2*(len(pairs)//2)+len(loners)} != {N}"

    if return_loners:
        return pairs, loners
    return pairs




def mate_with_population_info(population_info):
    # Build directional score matrix (from our earlier code)
    scores = build_score_matrix(population_info, mode="threshold", threshold=0.5)
    print("Scores:\n", scores)
    # Do bidirectional probabilistic selection
    #pairs = bidirectional_selection_pop_shrink(scores, num_pairs=len(population_info))  # 10 pairs
    pairs, loners = bidirectional_selection(scores)  # 10 pairs
    # Example output:
    # [('5_9_7','0_7_6'), ('3_5_1','6_5_7'), ('2_5_6','1_4_5'), ...]

    return pairs, loners
###################### Mating Matrix #####################

################## Zipit concept merging #################
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


def run_node_experiment_pop_shrink(node_config, experiment_config, pairs, loners, device, csv_file):

    offsprings = {}

    for pair in tqdm(pairs, desc='Evaluating Pairs...'):
        experiment_config = inject_pair(experiment_config, pair)
        config = prepare_experiment_config(raw_config)
        train_loader = config['data']['train']['full']
        base_models = [reset_bn_stats(base_model, train_loader) for base_model in config['models']['bases']]
        config['node'] = node_config
        
        Grapher = config['graph']
        graphs = [Grapher(deepcopy(base_model)).graphify() for base_model in base_models]
        
        Merge = ModelMerge(*graphs, device=device)

        Merge.transform(
            deepcopy(config['models']['new']), 
            train_loader, 
            transform_fn=config['merging_fn'], 
            metric_classes=config['metric_fns'],
            stop_at=node_config['stop_node'],
            **node_config['params']
        )

        reset_bn_stats(Merge, train_loader)
        
        results = evaluate_model(experiment_config['eval_type'], Merge, config)
        for idx, split in enumerate(pair):
            results[f'Split {CONCEPT_TASKS[idx]}'] = split
        results['Time'] = Merge.compute_transform_time
        results['Merging Fn'] = config['merging_fn'].__name__
        results['Model Name'] = config['model']['name']
        results.update(flatten_nested_dict(node_config, sep=' '))
        write_to_csv(results, csv_file=csv_file)
        print(results)

        # Build key from unique sorted labels
        merged_key = "_".join(sorted(set("_".join(pair).split("_"))))
        
        offsprings[merged_key] = Merge
        
    print(offsprings)
    return offsprings

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

def run_node_experiment(merging_fn, node_config, experiment_config, pairs, loners, device, csv_file):
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




################## Zipit concept merging #################


################## Mutation by finetune  #################
import torchvision
import torchvision.transforms as T

def get_CIFAR10_data():
    CIFAR_MEAN = [125.307, 122.961, 113.8575]
    CIFAR_STD = [51.5865, 50.847, 51.255]
    normalize = T.Normalize(np.array(CIFAR_MEAN)/255, np.array(CIFAR_STD)/255)
    denormalize = T.Normalize(-np.array(CIFAR_MEAN)/np.array(CIFAR_STD), 255/np.array(CIFAR_STD))

    data_dir = './data/cifar-100-python/cifar-10-batches-py'
    wrapper = torchvision.datasets.CIFAR10
    batch_size = 500
    train_transform = T.Compose([T.RandomHorizontalFlip(), T.RandomCrop(32, padding=4), T.ToTensor(), normalize])
    test_transform = T.Compose([T.ToTensor(), normalize])
    train_dset = wrapper(root=data_dir, train=True, download=True, transform=train_transform)
    test_dset = wrapper(root=data_dir, train=False, download=True, transform=test_transform)
    
    trainloader = torch.utils.data.DataLoader(train_dset, batch_size=batch_size, shuffle=True, num_workers=8)
    testloader = torch.utils.data.DataLoader(test_dset, batch_size=batch_size, shuffle=False, num_workers=8)
    
    return trainloader, testloader


def evaluate_logits(model, test_loader, return_confusion=False, use_flip_aug=False,
                    remap_class_idxs=None, class_idxs=None, eval_mask=None):
    model.eval()
    correct = 0
    total = 0
    totals = defaultdict(lambda: 0)
    corrects = defaultdict(lambda: 0)
    loss_fn = CrossEntropyLoss()
    device = next(iter(model.parameters())).device
    total_loss = 0
    total_iter = len(test_loader)

    # ✅ no autocast in evaluation
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, 'Evaluating classification model'):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)

            # Handle multi-head case
            if isinstance(outputs, list):
                outputs = outputs[0]

            if use_flip_aug:
                flip_outputs = model(torch.flip(inputs, (3,)))
                if isinstance(flip_outputs, list):
                    flip_outputs = flip_outputs[0]
                outputs += flip_outputs

            if eval_mask is not None:
                outputs[:, eval_mask == 0] = -torch.inf

            pred = outputs.argmax(dim=-1)
            total += pred.shape[0]

            if remap_class_idxs is not None:
                remapped_labels = remap_class_idxs[labels]
            else:
                remapped_labels = labels

            loss = loss_fn(outputs, remapped_labels)
            total_loss += loss

            for gt, p in zip(remapped_labels, pred):
                gt, p = gt.item(), p.item()
                totals[gt] += 1
                if gt == p:
                    correct += 1
                    corrects[gt] += 1

    num_classes = max(totals) + 1 if totals else 0
    totals = [totals[i] for i in range(num_classes)]
    corrects = [corrects[i] for i in range(num_classes)]
    total_loss = total_loss / max(total_iter, 1)

    if return_confusion:
        acc_per_class = [(c / t if t > 0 else 0.0) for c, t in zip(corrects, totals)]
        return correct / sum(totals), acc_per_class
    else:
        return correct / total if total > 0 else 0.0


def train_logits(model, train_loader, test_loader, epochs=2, remap_class_idxs=None):
    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001)
    ne_iters = len(train_loader)
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1, end_factor=1e-7, total_iters=ne_iters
    )
    early_stopper = EarlyStopper(patience=epochs, min_delta=.0001)

    scaler = GradScaler()
    loss_fn = CrossEntropyLoss(reduction='mean')
    device = get_device(model)
    losses = []
    acc = 0.
    best_acc = 0.
    best_epoch = 0
    best_sd = None

    pbar = tqdm(range(epochs), desc=f'Training, prev acc: {acc}: ')
    for epoch in pbar:
        model.train()
        for i, (inputs, labels) in enumerate(train_loader):
            optimizer.zero_grad(set_to_none=True)
            with autocast():  # ✅ autocast only during training
                logits = model(inputs.to(device))
                if isinstance(logits, list):
                    logits = logits[0]
                if remap_class_idxs is not None:
                    remapped_labels = remap_class_idxs[labels].to(device)
                else:
                    remapped_labels = labels.to(device)

                loss = loss_fn(logits, remapped_labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            losses.append(loss.item())

        acc = evaluate_logits(model, test_loader, remap_class_idxs=remap_class_idxs)
        if acc > best_acc:
            best_sd = model.state_dict()
            best_acc = acc
            best_epoch = epoch

        if early_stopper.early_stop(acc):
            print(f'Stopping at Epoch: {epoch}. Best Accuracy {best_acc}, Achieved at Epoch {best_epoch}')
            break
        pbar.set_description(f'Training, prev acc: {acc}: ')

    if best_sd is not None:
        model.load_state_dict(best_sd)
    acc = evaluate_logits(model, test_loader, remap_class_idxs=remap_class_idxs)
    print('Acc at Best Model: {}'.format(acc))
    return model, best_acc


def finetune_merged_model(model, trainloader, testloader, epochs=2):
    model.train()
    model, final_acc = train_logits(
        model=model,
        train_loader=trainloader,
        test_loader=testloader,
        epochs=epochs,
    )
    return model

################## Mutation by finetune  #################

def save_model(model, save_path, head_index=0):
    # If model has head_models, save one of them
    if hasattr(model, "head_models"):
        sd = model.head_models[head_index].state_dict()
    else:
        sd = model.state_dict()
    torch.save(sd, save_path)


if __name__ == "__main__":
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    config_name = 'cifar_evolution_resnet20'
    skip_pair_idxs = [0]
    model_width = 4
    num_generation = 15
    start_from = 10
    raw_config = get_config_from_name(config_name, device=device)
    #print(raw_config)
    

    for g in range(num_generation):

        print('')
        print(f' ============== Generation {g+1+start_from} ==============')
        print('')
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
        
        

        file_id = "zipit"
        csv_file = os.path.join(
            f'./csvs/{time.strftime("%Y-%m-%d")}/{file_id}/{g+1+start_from}',
            'evolutionary_zipit_configurations.csv'
        )

        with torch.no_grad():
            for node_config in experiment_configs:
                raw_config['dataset'].update(node_config.get('dataset', {}))

                # Get population information (i.e., model_names, per_class_accuracy,etc)
                population_info = run_evolutionary_evaluation(
                    node_config=node_config, 
                    experiment_config=raw_config, 
                    model_ids=models_ids, 
                    device=device,
                    csv_file=csv_file
                )

                # Create mating matrix and mate selection.
                #pairs = mate_with_population_info(population_info)
                pairs, loners = mate_with_population_info(population_info)
                print("Parents:\n", pairs)
                print("Loners:\n", loners)

                # Zipit concept merging.
                offsprings = run_node_experiment(
                    merging_fn='match_tensors_zipit', # change here
                    node_config=node_config, 
                    experiment_config=raw_config, 
                    pairs=pairs, 
                    loners=loners,
                    device=device,
                    csv_file=csv_file
                )
                
                print(f'\nWe collected {len(offsprings)} offsprings.\n')

        # Some mutation here...
        trainloader, testloader = get_CIFAR10_data()
        for model_name, model in offsprings.items():
            print(f"Processing model: {model_name}")

            mutated_model = finetune_merged_model(model, trainloader, testloader)
            
            ''''''
            # Update/save offspring as next generation.
            print(encode_labels(model_name))
            save_dir = os.path.join(f'./checkpoints/cifar10_evolution/zipit/gen_{g+1+start_from}/{encode_labels(model_name)}')
            print('Saving Base Model to:')
            print(save_dir)
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f'resnet20x{model_width}_v{len(os.listdir(save_dir))}.pth.tar')
            save_model(mutated_model, save_path)
            raw_config['model']['dir'] = f'./checkpoints/cifar10_evolution/zipit/gen_{g+1+start_from}/'
            