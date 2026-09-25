"""
Pretrain: create the initial population, in two phases.

    phase A   train ONE backbone on the whole training set, typically with a
              self-supervised objective (sesil/ssl.py). No agents yet.
    phase B   copy that backbone to every agent, give each a fresh head, and
              finetune it on its own random class subset -- using the same
              routine mutation uses in the evolution stage.

--pretrain-mode none skips phase A and trains every agent from scratch on its
own subset, which is the original SESiL scheme.

Why the shared backbone matters beyond feature quality: two networks trained
from *different* random initialisations occupy unrelated bases, so aligning
them for a merge is doing violence to both and the merged child is worse than
either parent. Agents finetuned from a *common* backbone stay in the same loss
basin, alignment is close to identity, and crossover keeps far more of what
each parent knew.

**Every run pretrains.** Nothing is loaded from a previous run implicitly.
Caching a population looked like a saving but bought only wall-clock -- reuse
was charged at its recorded cost anyway -- while costing a per-agent cost
ledger, a staleness fallback, and shared directories that concurrent seeds
raced on. Fresh pretrain per run is also the *correct* default for independent
seeds: error bands should cover phase A, not hold it fixed.

Everything a run produces therefore lives under its own run directory:

    <run dir>/backbone/<arch>.pth.tar       phase A output (ssl mode only)
    <run dir>/checkpoints/gen_0/agent_NNN/  the initial population
    <run dir>/checkpoints/gen_N/agent_NNN/  later generations

The one deliberate exception is --backbone-path, which lets a baseline run load
the backbone a SESiL run produced so the two start from identical weights. That
sharing is explicit and one-directional; nothing is ever picked up by accident.
"""

import json
import os
import time

import numpy as np
import torch
from sklearn.model_selection import train_test_split

from sesil.mutation import mutate
from sesil.population import list_population, save_agent
from sesil.ssl import (
    get_objective,
    reset_classifier,
    set_backbone_trainable,
    uses_labels,
    views_per_image,
)


BACKBONE_META = 'backbone.json'


def build_model(args, num_classes):
    """Fresh backbone of the configured architecture and width."""
    if args.model_name.startswith('resnet20'):
        from models.resnets import resnet20
        return resnet20(w=args.model_width, num_classes=num_classes).to(args.device)
    if args.model_name.startswith('vgg11'):
        from models.vgg import vgg11
        return vgg11(w=args.model_width, num_classes=num_classes).to(args.device)
    raise NotImplementedError(
        f'No builder for model {args.model_name!r}. Add one in sesil/pretrain.py.'
    )


# --------------------------------------------------------------------------- #
# Budget split
# --------------------------------------------------------------------------- #

def phase_budgets(args):
    """(phase_a, phase_b) epoch-equivalents, from --pretrain-budget.

    In --pretrain-mode none there is no phase A: the whole budget goes to
    training each agent from scratch on its own subset.
    """
    if args.pretrain_mode != 'ssl' or args.phase_a_ratio <= 0:
        return 0.0, float(args.pretrain_budget)
    phase_a = float(args.pretrain_budget) * float(args.phase_a_ratio)
    return phase_a, float(args.pretrain_budget) - phase_a


def cost_multiplier(args):
    """Epoch-equivalents charged per image in phase A.

    Defaults to the objective's own forward passes per image, so switching
    --phase-a-method re-prices phase A automatically rather than charging
    whatever number happened to be left in the flag.
    """
    if args.phase_a_cost_multiplier is not None:
        return float(args.phase_a_cost_multiplier)
    return views_per_image(args.phase_a_method)


def backbone_dir(args):
    """Where this run writes its phase-A backbone: inside its own run dir."""
    return os.path.join(args.run_dir, 'backbone')


# --------------------------------------------------------------------------- #
# Phase A
# --------------------------------------------------------------------------- #

def train_backbone(args, data, budget, logger=None):
    """Train the shared backbone for this run.

    Always trains -- there is no cache to consult. Returns (state_dict, cost);
    (None, 0.0) when phase A is not in play.
    """
    phase_a, _ = phase_budgets(args)
    if phase_a <= 0:
        print(f'[phase A] skipped (--pretrain-mode {args.pretrain_mode}); '
              f'agents will start from random initialisations.')
        return None, 0.0

    # Each image costs `multiplier` epoch-equivalents, so the budget buys
    # proportionally fewer images than a supervised pass would.
    multiplier = max(cost_multiplier(args), 1e-9)
    sample_budget = int(round(phase_a * budget.train_set_size / multiplier))

    labelled = uses_labels(args.phase_a_method)
    print(f'[phase A] {args.phase_a_method} on all classes: '
          f'{phase_a:.2f} epoch-equiv budget, cost multiplier {multiplier:g} '
          f'-> {sample_budget} image-presentations')
    print(f'[phase A] labels: '
          f'{"USED (control, not self-supervised)" if labelled else "not used"}')

    model = build_model(args, data.num_classes)
    objective = get_objective(args.phase_a_method)
    model, info = objective(model, data.train_loader(), sample_budget, args)

    cost = (sample_budget * multiplier) / max(budget.train_set_size, 1)
    budget.spend_samples('phase_a', budget.train_set_size * cost, epochs=1)

    # Written so a baseline run can start from the same weights via
    # --backbone-path. Nothing in this run reads it back.
    path = backbone_dir(args)
    os.makedirs(path, exist_ok=True)
    state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    torch.save(state, os.path.join(path, f'{args.arch}.pth.tar'))
    with open(os.path.join(path, BACKBONE_META), 'w') as f:
        json.dump({'cost': round(cost, 6),
                   'method': args.phase_a_method,
                   'multiplier': multiplier,
                   'uses_labels': labelled,
                   'arch': args.arch,
                   'seed': args.seed,
                   **info}, f, indent=2)

    print(f'[phase A] done: {info}')
    print(f'[phase A] backbone -> {path}  (cost {cost:.2f} epoch-equiv)')
    if logger is not None:
        logger.log_train({'stage': 'phase_a', 'budget': round(budget.spent, 4),
                          'cost': round(cost, 6), **info})

    return state, cost


def load_backbone(path, arch, device):
    """Load a backbone another run produced. Used only by --baseline-init backbone."""
    weights = os.path.join(path, f'{arch}.pth.tar')
    if not os.path.exists(weights):
        return None
    return torch.load(weights, map_location=device)


def backbone_cost(path):
    """What that backbone cost to build, so a run loading it can be charged."""
    meta = os.path.join(path, BACKBONE_META)
    if not os.path.exists(meta):
        return None
    with open(meta) as f:
        return float(json.load(f).get('cost', 0.0))


# --------------------------------------------------------------------------- #
# Phase B
# --------------------------------------------------------------------------- #

def class_subsets(args, num_classes):
    """Which classes each agent is trained on, per --subset-mode.

    'random' draws each agent's subset independently. That is the original
    scheme, and it leaves overlap to chance: with 8 agents taking 3 of 10
    classes, some couples end up nearly identical and others nearly disjoint,
    but nothing controls the distribution and it changes with the seed.

    'disjoint' deals classes round-robin from repeatedly reshuffled
    permutations instead. Every class is used about equally often and pairwise
    overlap is as low as the arithmetic allows, so agents genuinely specialise.
    This matters for anything that reads the *weights*: agents finetuned from
    one backbone on largely the same classes produce task vectors that all
    point the same way, and a weight-space statistic then returns nearly the
    same number for every couple -- it cannot rank what does not differ.
    """
    rng = np.random.default_rng(args.seed)
    k = args.classes_per_model

    if getattr(args, 'subset_mode', 'random') == 'random':
        return [sorted(int(c) for c in train_test_split(
            np.arange(num_classes), train_size=k)[0])
            for _ in range(args.pop_size)]

    if k > num_classes:
        raise ValueError(f'--classes-per-model {k} exceeds {num_classes} classes')

    subsets, pool = [], []
    for _ in range(args.pop_size):
        chosen = []
        while len(chosen) < k:
            if not pool:
                pool = list(rng.permutation(num_classes))
            nxt = pool.pop()
            # Refill rather than repeat: an agent never gets a class twice.
            if nxt in chosen:
                spare = [c for c in range(num_classes) if c not in chosen]
                nxt = int(rng.choice(spare))
            chosen.append(int(nxt))
        subsets.append(sorted(chosen))
    return subsets


def build_population(args, data, budget, logger=None):
    """Phase A then phase B: one backbone, then --pop-size specialists.

    Writes generation 0 into this run's own directory. Always runs; there is no
    population to inherit.
    """
    population_dir = args.population_dir
    os.makedirs(population_dir, exist_ok=True)

    phase_a, phase_b = phase_budgets(args)
    backbone, _backbone_cost = train_backbone(args, data, budget, logger)

    per_agent = phase_b / max(args.pop_size, 1)
    sample_budget = int(round(per_agent * budget.train_set_size))

    # In mode 'none' there is no phase A, so calling this "phase B" in the log
    # would be misleading -- it is simply the whole pretrain stage.
    label = 'phase B' if backbone is not None else 'pretrain'
    origin = 'the shared backbone' if backbone is not None else 'scratch'

    print(f'[{label}] {args.pop_size} agents x {per_agent:.3f} epoch-equiv '
          f'= {sample_budget} presentations each, on {args.classes_per_model} '
          f'of {data.num_classes} classes, from {origin}')

    val_loader = data.val_loader()
    start = time.time()

    subsets = class_subsets(args, data.num_classes)

    for individual in range(args.pop_size):
        split = subsets[individual]

        model = build_model(args, data.num_classes)
        if backbone is not None:
            model.load_state_dict(backbone, strict=False)
            # Every agent starts from identical weights, so a freshly random
            # head is the only diversity they have before their subsets pull
            # them apart.
            reset_classifier(model)
        model = model.train()

        train_loader = data.train_loader(classes=split)
        n_available = len(train_loader.dataset)

        budget.spend_samples('phase_b', sample_budget, epochs=1)

        print(f'[{label}] {individual + 1}/{args.pop_size} on classes {split}  '
              f'{sample_budget} presentations over {n_available} samples')

        # Optional warmup with the backbone held still. See
        # sesil.ssl.set_backbone_trainable for why this matters more here than
        # in ordinary transfer learning.
        #
        # The warmup is charged at the full sample-presentation rate even
        # though a head-only backward pass is cheaper. That is deliberate and
        # conservative: overcharging can only understate the benefit, so a
        # warmup that still wins is not winning on accounting.
        warmup = int(round(sample_budget * getattr(args, 'phase_b_freeze', 0.0)))
        if backbone is not None and warmup > 0:
            n_frozen = set_backbone_trainable(model, False)
            model, _ = mutate(model, train_loader, val_loader,
                              sample_budget=warmup)
            set_backbone_trainable(model, True)
            print(f'[{label}] warmup: {warmup} presentations with '
                  f'{n_frozen} backbone tensors frozen')

        model, final_acc = mutate(model, train_loader, val_loader,
                                  sample_budget=sample_budget - warmup)
        print(f'[{label}] val accuracy: {final_acc}')

        meta = {
            'generation': 0,
            'certificate': [],          # granted at the first evaluation
            'trained_on': split,
            'pretrain_subset': split,   # provenance, not identity
            'pretrain_mode': args.pretrain_mode,
            'phase_a_method': args.phase_a_method if backbone is not None else None,
            'parents': [],
            'head': None,
            'is_loner': False,
        }
        save_path = save_agent(model, population_dir, individual, args.arch, meta)
        print(f'[{label}] saved -> {save_path}')

    print(f'[pretrain] done in {time.time() - start:.1f}s  '
          f'(phase A {phase_a:.2f} + phase B {phase_b:.2f} epoch-equiv)')
    return list_population(population_dir)
