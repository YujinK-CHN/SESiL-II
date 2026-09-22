"""
Fitness evaluation.

Two different questions get asked during a generation:

  evaluate_individual  -- how good is one population member on the classes it
                          was trained for?  Produces the per-class vector that
                          mate selection runs on.
  evaluate_merged      -- how good is an offspring across all the tasks its
                          parents covered?  Produces the numbers that go in the
                          results csv.

The multi-task evaluator below is the ragged-split-safe version: as offspring
accumulate labels, parents stop having equal numbers of classes, so splits
cannot be packed into a rectangular tensor.
"""

import numpy as np
import torch
from torch.cuda.amp import autocast
from tqdm.auto import tqdm

from utils import CONCEPT_TASKS, get_device


def evaluate_individual(model, config, labels_str, num_classes):
    """Per-class accuracy of a single individual, over its own label subset.

    Classes outside the individual's subset stay at 0.0 -- that zero is exactly
    what makes another individual attractive as a mate.
    """
    device = next(model.parameters()).device
    test_loader = config['data']['test']['full']

    acc_per_class = [0.0] * num_classes
    subset_labels = sorted(set(int(x) for x in labels_str.split('_')))

    model.eval()
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)

            mask = torch.zeros_like(labels, dtype=torch.bool)
            for lab in subset_labels:
                mask |= (labels == lab)
            if mask.sum() == 0:
                continue

            images_masked = images[mask]
            labels_masked = labels[mask]

            outputs = model(images_masked)
            _, preds = outputs.max(1)

            for lab in subset_labels:
                lab_mask = (labels_masked == lab)
                if lab_mask.sum() > 0:
                    acc_per_class[lab] += (preds[lab_mask] == labels_masked[lab_mask]).sum().item()

    total_per_class = torch.zeros(num_classes)
    for _, labels in test_loader:
        for lab in subset_labels:
            total_per_class[lab] += (labels == lab).sum().item()

    for lab in subset_labels:
        if total_per_class[lab] > 0:
            acc_per_class[lab] /= total_per_class[lab]

    # Accuracy restricted to the classes this individual actually knows.
    own_acc = sum(acc_per_class[lab] for lab in subset_labels) / len(subset_labels)

    return {
        'Joint': sum(acc_per_class) / len(acc_per_class),
        'Per Task Avg': own_acc,
        'Per Class': acc_per_class.copy(),
    }


def evaluate_merged(model, config, eval_type='logits'):
    """Multi-task evaluation of a merged offspring."""
    loader = config['data']['test']['full']
    num_classes = len(config['data']['test']['class_names'])

    if eval_type == 'logits':
        acc_overall, acc_avg, pertask_acc, perclass_acc = evaluate_logits_alltasks(
            model, loader,
            splits=config['dataset']['class_splits'],
            num_classes=num_classes,
        )
    elif eval_type == 'clip':
        from utils import evaluate_cliphead_alltasks, load_clip_features
        clip_features = load_clip_features(config['data']['test']['class_names'], get_device(model))
        class_vectors = [clip_features[split] for split in config['data']['train']['class_splits']]
        acc_overall, acc_avg, pertask_acc = evaluate_cliphead_alltasks(
            model, loader, class_vectors,
            config['data']['train']['class_splits'],
            num_classes=num_classes,
        )
        perclass_acc = None
    else:
        raise ValueError(f'Invalid eval_type: {eval_type}! Must be one of [logits, clip].')

    results = {'Joint': acc_overall, 'Per Task Avg': acc_avg, 'Per class Acc': perclass_acc}
    for task_idx, task_acc in enumerate(pertask_acc):
        results[f'Task {CONCEPT_TASKS[task_idx]}'] = task_acc
    return results


def evaluate_logits_alltasks(model, loader, splits, num_classes):
    """Per-task and per-class accuracy of a multi-head / merged model.

    Handles ragged splits: each split is kept as its own tensor rather than
    stacked, because offspring accumulate labels and parents end up covering
    different numbers of classes.
    """
    model.eval()
    correct = 0
    total = 0

    splits = [list(split) for split in splits]
    totals = [0] * num_classes
    corrects = [0] * num_classes

    device = get_device(model)

    all_splits = torch.tensor([cls for split in splits for cls in split],
                              device=device, dtype=torch.long)

    # class id -> which task it belongs to (-1 if no task covers it)
    task_map = {}
    for i, split in enumerate(splits):
        for _cls in split:
            task_map[_cls] = i
    task_map = [task_map.get(_cls, -1) for _cls in range(num_classes)]
    task_map = torch.tensor(task_map, device=device, dtype=torch.long)

    splits_tensor = [torch.tensor(s, device=device, dtype=torch.long) for s in splits]

    with torch.no_grad(), autocast():
        for inputs, labels in tqdm(loader, 'Evaluating merged model'):
            inputs, labels = inputs.to(device), labels.to(device)

            # keep only samples belonging to a class some task covers
            class_selector = torch.isin(labels, all_splits)
            inputs, labels = inputs[class_selector], labels[class_selector]

            batch_size = inputs.shape[0]
            if batch_size == 0:
                continue

            task_idx = task_map[labels]
            outputs = model(inputs)

            if isinstance(outputs, list):
                # mask out classes each head is not responsible for
                for i, split in enumerate(splits_tensor):
                    exclude_labels = torch.tensor(
                        [cls for cls in all_splits.tolist() if cls not in split.tolist()],
                        device=device, dtype=torch.long,
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

            # per-sample prediction restricted to that sample's own task
            preds = []
            for out_vec, split in zip(outputs, [splits_tensor[i] for i in task_idx.tolist()]):
                idx = out_vec[split].argmax()
                preds.append(split[idx].item())
            outputs = torch.tensor(preds, device=device, dtype=torch.long)

            for gt, p, p2 in zip(labels, outputs, outputs2):
                totals[gt] += 1
                if gt == p:
                    corrects[gt] += 1
                if gt == p2:
                    correct += 1
                total += 1

    split_accs = [0] * len(splits)
    for i, split in enumerate(splits):
        split_total = 0
        for _cls in split:
            split_accs[i] += corrects[_cls]
            split_total += totals[_cls]
        split_accs[i] /= max(split_total, 1e-4)

    perclass = np.asarray(corrects) / np.maximum(np.asarray(totals), 1e-4)
    return correct / max(total, 1), sum(split_accs) / len(split_accs), split_accs, perclass
