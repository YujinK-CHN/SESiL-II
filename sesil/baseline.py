"""
The learning-based baseline: one classifier, trained conventionally.

This is what SESiL's generation curve is compared against.  Two modes:

    scratch   train a single model on --baseline-classes (or the whole dataset)
    finetune  take an existing checkpoint and extend it onto --finetune-classes

Both log a per-epoch accuracy curve next to the checkpoint, so the baseline can
be plotted on the same axes as an evolution run.
"""

import json
import os

import numpy as np
import torch

from config import parse_int_list
from utils import evaluate_logits, save_model, train_logits

from sesil.mutation import mutate
from sesil.pretrain import backbone_cost, build_model, load_backbone
from sesil.ssl import reset_classifier


def run_baseline(args, budget, data, logger, evaluator):
    """Entry point for --method baseline."""
    os.makedirs(args.run_dir, exist_ok=True)

    if args.baseline_mode == 'scratch':
        model, acc = _train_scratch(args, budget, data, evaluator)
    elif args.baseline_mode == 'finetune':
        model, acc = _train_finetune(args, budget, data, evaluator)
    elif args.baseline_mode == 'curriculum':
        model, acc = _train_curriculum(args, budget, data, evaluator)
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

    model = _init_model(args, num_classes, budget)
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


def _init_model(args, num_classes, budget):
    """Starting point for the baseline, per --baseline-init.

    'backbone' loads the same phase-A backbone SESiL uses AND is charged its
    recorded cost, so the two methods start level on both weights and budget.
    Running the baseline both ways is what separates "the backbone helped" from
    "evolution helped" -- with only the scratch variant, a SESiL win is
    ambiguous.
    """
    model = build_model(args, num_classes)

    if args.baseline_init != 'backbone':
        return model.train()

    if not args.backbone_path:
        raise SystemExit(
            '--baseline-init backbone needs --backbone-path pointing at a '
            'backbone a SESiL run produced, e.g.\n'
            '  --backbone-path results/<exp>/<dataset>/<merger>/seed0/backbone\n'
            'Nothing is discovered implicitly: the comparison only means '
            'something if you name the backbone you intend to share.')

    state = load_backbone(args.backbone_path, args.arch, args.device)
    if state is None:
        raise SystemExit(
            f'No {args.arch} backbone at {args.backbone_path!r}.\n'
            f'Run the SESiL side first (--method sesil --pretrain-mode ssl, '
            f'optionally --pretrain-only) and point --backbone-path at its '
            f'<run dir>/backbone.')

    model.load_state_dict(state, strict=False)
    reset_classifier(model)

    cost = backbone_cost(args.backbone_path)
    if cost and budget is not None:
        budget.spend_samples('phase_a', budget.train_set_size * cost, epochs=1)
        print(f'[baseline] initialised from the phase-A backbone, '
              f'charged {cost:.2f} epoch-equiv for it')
    return model.train()


def read_society_curriculum(run_dir):
    """The society-space record from a SESiL run: (budget, classes) per step.

    Written by sesil.evolution.mutate_and_save as stage 'society'. It is the
    union of what every agent was actually TRAINED ON that generation, not the
    union of their certificates -- an agent whose certificate collapsed still
    trains on its one drawn class, so the space is never empty.
    """
    path = os.path.join(run_dir, 'train.jsonl')
    if not os.path.exists(path):
        raise SystemExit(
            f'No train.jsonl in {run_dir!r}. --curriculum-from must point at a '
            f'finished SESiL run directory (the one holding eval.jsonl and '
            f'train.jsonl), not at the experiment root.')

    steps = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get('stage') == 'society':
                steps.append((float(row['budget']), list(row['space'])))

    if not steps:
        raise SystemExit(
            f'{path} has no society records. It was written before society-space '
            f'logging existed, or the run never reached a mutation step.')

    steps.sort(key=lambda s: s[0])
    return steps


def _train_curriculum(args, budget, data, evaluator):
    """Replay a SESiL run's society space through ONE continuously trained model.

    At each of that run's generations the society was working on some set of
    classes. This trains a single classifier on exactly those classes, for
    exactly the budget that generation cost, then moves to the next set.

    No finetuning machinery: the model is never re-initialised and its head is
    never touched. Only the training set changes -- which is the whole point.
    It isolates the population and the merging from the CURRICULUM, because
    SESiL cannot then win merely by having explored better.

    The society space shrinks as well as grows, so the baseline can stop
    training on a class it previously learned. That is the adaptability test
    working, not a defect, and it is why evaluation stays on the full label
    space for both methods.
    """
    steps = read_society_curriculum(args.curriculum_from)
    val_loader = data.val_loader(batch_size=args.baseline_batch_size)
    num_classes = data.num_classes

    total = steps[-1][0]
    print(f'[baseline] replaying the society curriculum from '
          f'{args.curriculum_from}')
    print(f'[baseline] {len(steps)} steps, budget {steps[0][0]:.2f} -> {total:.2f}')

    model = _init_model(args, num_classes, budget)

    # The baseline's curve starts at zero: an untrained classifier, before any
    # budget is spent. SESiL's starts where pretrain finished, which is what
    # the dashed phase line marks. Without this point the two curves begin at
    # different x and the comparison silently drops the baseline's early
    # progress, which is the part a reader most wants to see.
    evaluator.record([model], step=0, step_kind='step')

    curve = []
    previous = 0.0
    loaders = {}

    for index, (mark, classes) in enumerate(steps):
        delta = mark - previous
        previous = mark
        if delta <= 0 or not classes:
            continue

        key = tuple(sorted(classes))
        if key not in loaders:
            loaders[key] = data.train_loader(
                classes=list(key), batch_size=args.baseline_batch_size)
        train_loader = loaders[key]

        # Charge the same epoch-equivalents SESiL spent reaching this mark,
        # converted to presentations over whatever the society was studying.
        want = int(round(delta * budget.train_set_size))
        if not budget.can_afford(delta):
            print(f'[baseline] budget exhausted at step {index}.')
            break
        budget.spend_samples('baseline', want, epochs=1)

        print(f'[baseline] step {index + 1}/{len(steps)}  '
              f'{len(classes)} classes  {want} presentations  '
              f'(budget {mark:.2f})')

        model, _ = mutate(model, train_loader, val_loader, sample_budget=want)
        acc = evaluate_logits(model, val_loader)
        curve.append(acc)
        budget.count_forward_test(1)

        evaluator.maybe_record([model], step=index + 1, step_kind='step')

    curve_path = os.path.join(args.run_dir, 'val_accuracy_curve.npy')
    np.save(curve_path, np.asarray(curve))
    print(f'[baseline] validation curve -> {curve_path}')
    evaluator.record([model], step=len(curve), step_kind='step', final=True)

    return model, (curve[-1] if curve else 0.0)


def _train_and_log(args, model, train_loader, val_loader, epochs, budget, evaluator):
    """Train one epoch at a time so the accuracy curve can be recorded.

    utils.train_logits already loops internally, but it only returns the best
    accuracy; calling it per epoch is what makes the curve available for
    plotting against the SESiL generation curve -- and what lets each epoch be
    charged to the budget as it is spent.

    Epochs are charged by actual sample count, so a baseline restricted to a
    class subset costs proportionally less than a full-dataset one.
    """
    # Zero point: untrained, nothing spent. See _train_curriculum for why.
    evaluator.record([model], step=0, step_kind='epoch')

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
