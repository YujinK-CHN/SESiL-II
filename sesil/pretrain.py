"""
Pretrain: create the initial population, in two phases.

    phase A   train ONE backbone on the whole training set, typically with a
              self-supervised objective (sesil/ssl.py). No agents yet.
    phase B   copy that backbone to every agent, give each a fresh head, and
              finetune it on its own random class subset -- using the same
              routine mutation uses in the evolution stage.

--phase-a-ratio splits --pretrain-budget between them; --phase-a-ratio 0.0
skips phase A and reproduces the older behaviour of training every agent from
scratch.

Why the shared backbone matters beyond feature quality: two networks trained
from *different* random initialisations occupy unrelated bases, so aligning
them for a merge is doing violence to both and the merged child is worse than
either parent. Agents finetuned from a *common* backbone stay in the same loss
basin, alignment is close to identity, and crossover keeps far more of what
each parent knew.

Each agent is saved as

    <population dir>/agent_NNN/<arch>_v0.pth.tar
    <population dir>/agent_NNN/agent.json

The class subset it saw is stored as provenance, not identity. Generation 0 is
evaluated across the whole label space like every later generation and earns
its proficiency certificate from that measurement.
"""

import json
import os
import time

import numpy as np
import torch
from sklearn.model_selection import train_test_split

from sesil.mutation import mutate
from sesil.population import list_population, read_meta, save_agent
from sesil.ssl import get_objective, reset_classifier


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
    """(phase_a, phase_b) epoch-equivalents, from --pretrain-budget."""
    if args.phase_a_method == 'none' or args.phase_a_ratio <= 0:
        return 0.0, float(args.pretrain_budget)
    phase_a = float(args.pretrain_budget) * float(args.phase_a_ratio)
    return phase_a, float(args.pretrain_budget) - phase_a


def backbone_dir(args):
    """Where the phase-A backbone is cached."""
    if args.backbone_path:
        return args.backbone_path
    # Beside the population, but keyed only by what affects the backbone --
    # NOT by pop-size or classes-per-model, which are phase-B concerns.
    return os.path.join(os.path.dirname(os.path.normpath(args.population_dir)),
                        f'backbone_{args.phase_a_method}')


# --------------------------------------------------------------------------- #
# Phase A
# --------------------------------------------------------------------------- #

def ensure_backbone(args, data, budget, logger=None):
    """Train the shared backbone, or load and charge for a cached one.

    Returns (state_dict or None, cost). None means phase A was skipped, and
    agents will be built from random initialisations.
    """
    phase_a, _ = phase_budgets(args)
    if phase_a <= 0:
        print('[phase A] skipped (--phase-a-ratio 0 or --phase-a-method none); '
              'agents will start from random initialisations.')
        return None, 0.0

    path = backbone_dir(args)
    weights = os.path.join(path, f'{args.arch}.pth.tar')
    meta_path = os.path.join(path, BACKBONE_META)

    if os.path.exists(weights) and not args.force_pretrain:
        recorded = {}
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                recorded = json.load(f)
        cost = float(recorded.get('cost', phase_a))
        print(f'[phase A] reusing cached backbone from {path}')
        print(f'[phase A] charged at its recorded creation cost: '
              f'{cost:.2f} epoch-equiv')
        budget.spend_samples('phase_a', budget.train_set_size * cost, epochs=1)
        return torch.load(weights, map_location=args.device), cost

    # Each image costs `multiplier` epoch-equivalents, so the budget buys
    # proportionally fewer images than a supervised pass would.
    multiplier = max(float(args.phase_a_cost_multiplier), 1e-9)
    sample_budget = int(round(phase_a * budget.train_set_size / multiplier))

    print(f'[phase A] {args.phase_a_method} on all classes: '
          f'{phase_a:.2f} epoch-equiv budget, cost multiplier {multiplier:g} '
          f'-> {sample_budget} image-presentations')

    model = build_model(args, data.num_classes)
    objective = get_objective(args.phase_a_method)
    model, info = objective(model, data.train_loader(), sample_budget, args)

    cost = (sample_budget * multiplier) / max(budget.train_set_size, 1)
    budget.spend_samples('phase_a', budget.train_set_size * cost, epochs=1)

    os.makedirs(path, exist_ok=True)
    state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    torch.save(state, weights)
    with open(meta_path, 'w') as f:
        json.dump({'cost': round(cost, 6),
                   'method': args.phase_a_method,
                   'multiplier': multiplier,
                   'arch': args.arch,
                   **info}, f, indent=2)

    print(f'[phase A] done: {info}')
    print(f'[phase A] backbone -> {weights}  (cost {cost:.2f} epoch-equiv)')
    if logger is not None:
        logger.log_train({'stage': 'phase_a', 'budget': round(budget.spent, 4),
                          'cost': round(cost, 6), **info})

    return state, cost


def load_backbone(args):
    """The cached backbone state dict, or None. Used by the baseline too."""
    weights = os.path.join(backbone_dir(args), f'{args.arch}.pth.tar')
    if not os.path.exists(weights):
        return None
    return torch.load(weights, map_location=args.device)


def backbone_cost(args):
    """What the cached backbone cost to build, for charging a reuser."""
    meta_path = os.path.join(backbone_dir(args), BACKBONE_META)
    if not os.path.exists(meta_path):
        return None
    with open(meta_path) as f:
        return float(json.load(f).get('cost', 0.0))


# --------------------------------------------------------------------------- #
# Phase B
# --------------------------------------------------------------------------- #

def ensure_population(args, data, budget=None, logger=None):
    """Create the initial population if it is not already on disk.

    Returns the agent ids making up generation 0.

    A reused population is still CHARGED, at what it originally cost to build.
    Reuse saves wall-clock, not budget: letting SESiL inherit a population for
    free while the baseline starts from random init would quietly unmatch a
    budget-matched comparison.
    """
    existing = list_population(args.population_dir)

    if existing and not args.force_pretrain:
        print(f'[pretrain] Found {len(existing)} agents in {args.population_dir}; '
              f'skipping pretrain.')
        if len(existing) != args.pop_size:
            print(f'[pretrain] NOTE: --pop-size is {args.pop_size} but the existing '
                  f'population has {len(existing)} members. Using what is on disk. '
                  f'Pass --force-pretrain to rebuild it.')
        _charge_reused_population(args, existing, budget)
        return existing

    print(f'[pretrain] Building a population of {args.pop_size} into {args.population_dir}')
    return pretrain_population(args, data, budget, logger)


def _charge_reused_population(args, agent_ids, budget):
    """Charge a reused population at what it originally cost to create.

    The price is read from each agent's own metadata rather than recomputed, so
    it stays correct even if the pretrain settings have changed since. Phase A
    is included, because the recorded per-agent cost carries its share.
    """
    if budget is None:
        return 0.0

    recorded = [read_meta(args.population_dir, a).get('pretrain_cost')
                for a in agent_ids]

    if any(c is None for c in recorded):
        total = float(args.pretrain_budget)
        print(f'[pretrain] WARNING: this population records no creation cost '
              f'(built before costs were tracked). Falling back to '
              f'--pretrain-budget = {total:.2f} epoch-equiv, which is wrong if the '
              f'settings differ from the ones that built it. Rebuild with '
              f'--force-pretrain for exact accounting.')
    else:
        total = float(sum(recorded))
        print(f'[pretrain] Reused population charged at its recorded creation '
              f'cost: {total:.2f} epoch-equiv.')

    budget.spend_samples('pretrain', budget.train_set_size * total, epochs=1)
    return total


def pretrain_population(args, data, budget=None, logger=None):
    """Phase A then phase B: one backbone, then --pop-size specialists."""
    os.makedirs(args.population_dir, exist_ok=True)

    phase_a, phase_b = phase_budgets(args)
    backbone, backbone_cost_ = ensure_backbone(args, data, budget, logger)

    per_agent = phase_b / max(args.pop_size, 1)
    sample_budget = int(round(per_agent * budget.train_set_size)) if budget else 0

    # Phase A was paid once but benefits every agent, so its share is recorded
    # per agent -- that way a later run reusing this population is charged for
    # the backbone too, not just the finetuning.
    backbone_share = (backbone_cost_ or 0.0) / max(args.pop_size, 1)

    print(f'[phase B] {args.pop_size} agents x {per_agent:.3f} epoch-equiv '
          f'= {sample_budget} presentations each, on {args.classes_per_model} '
          f'of {data.num_classes} classes')

    val_loader = data.val_loader()
    start = time.time()

    for individual in range(args.pop_size):
        # A random subset of class ids; the discarded half is not used.
        split, _ = train_test_split(
            np.arange(data.num_classes), train_size=args.classes_per_model)
        split = sorted(int(c) for c in split)

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

        if budget is not None:
            budget.spend_samples('phase_b', sample_budget, epochs=1)

        print(f'[phase B] {individual + 1}/{args.pop_size} on classes {split}  '
              f'{sample_budget} presentations over {n_available} samples')

        model, final_acc = mutate(model, train_loader, val_loader,
                                  sample_budget=sample_budget)
        print(f'[phase B] val accuracy: {final_acc}')

        meta = {
            'generation': 0,
            'certificate': [],          # granted at the first evaluation
            'trained_on': split,
            'pretrain_subset': split,
            # What a later run pays to reuse this agent: its own phase-B share
            # plus its slice of the shared backbone.
            'pretrain_cost': round(per_agent + backbone_share, 6),
            'phase_a_method': args.phase_a_method if backbone is not None else None,
            'parents': [],
            'is_loner': False,
        }
        save_path = save_agent(model, args.population_dir, individual, args.arch, meta)
        print(f'[phase B] saved -> {save_path}')

    print(f'[pretrain] done in {time.time() - start:.1f}s  '
          f'(phase A {phase_a:.2f} + phase B {phase_b:.2f} epoch-equiv)')
    return list_population(args.population_dir)
