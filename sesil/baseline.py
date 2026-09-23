"""
The learning-based baseline: one classifier, trained conventionally.

This is what SESiL's generation curve is compared against.  Two modes:

    scratch   train a single model on --baseline-classes (or the whole dataset)
    finetune  take an existing checkpoint and extend it onto --finetune-classes

Both log a per-epoch accuracy curve next to the checkpoint, so the baseline can
be plotted on the same axes as an evolution run.
"""

import os

import numpy as np
import torch

from config import parse_int_list
from utils import evaluate_logits, save_model, train_logits

from sesil.pretrain import build_model


def run_baseline(args, budget, data, logger, evaluator):
    """Entry point for --method baseline."""
    os.makedirs(args.run_dir, exist_ok=True)

    if args.baseline_mode == 'scratch':
        model, acc = _train_scratch(args, budget, data, evaluator)
    elif args.baseline_mode == 'finetune':
        model, acc = _train_finetune(args, budget, data, evaluator)
    else:
        raise ValueError(f'Unknown baseline mode {args.baseline_mode!r}')

    save_path = os.path.join(args.run_dir, f'{args.arch}_v0.pth.tar')
    save_model(model, save_path)
    print(f'[baseline] final accuracy {acc:.4f}, saved -> {save_path}')
    if budget is not None:
        print(budget.report())
    return model, acc


def _train_scratch(args, budget, data, evaluator):
    """Train one classifier from random initialisation."""
    classes = parse_int_list(args.baseline_classes)
    train_loader = data.train_loader(classes=classes, batch_size=args.baseline_batch_size)
    val_loader = data.val_loader(batch_size=args.baseline_batch_size)
    num_classes = data.num_classes

    print(f'[baseline] training from scratch on '
          f'{classes if classes is not None else "all classes"} '
          f'for {args.baseline_epochs} epochs')

    model = build_model(args, num_classes).train()
    model, acc = _train_and_log(args, model, train_loader, val_loader,
                                args.baseline_epochs, budget, evaluator)
    return model, acc


def _train_finetune(args, budget, data, evaluator):
    """Extend an existing checkpoint onto additional classes.

    The head keeps its full width, so old classes are not dropped -- the model
    is trained on the union of what it knew and --finetune-classes.
    """
    if args.baseline_load_path is None:
        raise ValueError('--baseline-load-path is required for --baseline-mode finetune')

    old_classes = parse_int_list(args.baseline_classes) or []
    new_classes = parse_int_list(args.finetune_classes)
    if not new_classes:
        raise ValueError('--finetune-classes is required for --baseline-mode finetune')

    classes = sorted(set(old_classes) | set(new_classes))
    train_loader = data.train_loader(classes=classes, batch_size=args.baseline_batch_size)
    val_loader = data.val_loader(batch_size=args.baseline_batch_size)
    num_classes = data.num_classes

    print(f'[baseline] finetuning {args.baseline_load_path} onto {new_classes} '
          f'(training over {classes})')

    model = build_model(args, num_classes)
    state_dict = torch.load(args.baseline_load_path, map_location=args.device)
    if 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']
    model.load_state_dict(state_dict)
    model = model.train()

    model, acc = _train_and_log(args, model, train_loader, val_loader,
                                args.baseline_epochs, budget, evaluator)
    return model, acc


def _train_and_log(args, model, train_loader, val_loader, epochs, budget, evaluator):
    """Train one epoch at a time so the accuracy curve can be recorded.

    utils.train_logits already loops internally, but it only returns the best
    accuracy; calling it per epoch is what makes the curve available for
    plotting against the SESiL generation curve -- and what lets each epoch be
    charged to the budget as it is spent.

    Epochs are charged by actual sample count, so a baseline restricted to a
    class subset costs proportionally less than a full-dataset one.
    """
    curve = []
    best_acc = 0.0
    best_sd = None

    n_samples = len(train_loader.dataset)

    for epoch in range(epochs):
        if budget is not None:
            if not budget.can_afford(n_samples / budget.train_set_size):
                print(f'[baseline] budget exhausted after {epoch} epoch(s).')
                break
            budget.spend_samples('baseline', n_samples, epochs=1)

        model, _ = train_logits(model, train_loader, val_loader, epochs=1)
        acc = evaluate_logits(model, val_loader)
        curve.append(acc)
        if budget is not None:
            budget.count_forward_test(1)
        print(f'[baseline] epoch {epoch + 1}/{epochs}: val acc {acc:.4f}')

        # External evaluation on the SAME budget watermarks SESiL uses, so the
        # two curves share an x-axis by construction.
        evaluator.maybe_record([model], step=epoch + 1, step_kind='epoch')
        if acc > best_acc:
            best_acc = acc
            best_sd = {k: v.detach().clone() for k, v in model.state_dict().items()}

    if best_sd is not None:
        model.load_state_dict(best_sd)

    curve_path = os.path.join(args.run_dir, 'val_accuracy_curve.npy')
    np.save(curve_path, np.asarray(curve))
    print(f'[baseline] validation curve -> {curve_path}')

    # Final point, matching what run_evolution does, so both methods always end
    # on a recorded value.
    evaluator.record([model], step=len(curve), step_kind='epoch', final=True)

    return model, best_acc
