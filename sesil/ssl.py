"""
Phase-A objectives: how the shared backbone is trained.

Phase A trains ONE model on the whole training set, before any agent exists.
The objective is pluggable so the choice of self-supervised method is a flag
rather than a rewrite -- add a function here and one line to OBJECTIVES.

Every objective has the signature

    fn(model, loader, sample_budget, args) -> (model, info_dict)

and must stop after roughly `sample_budget` image-presentations, so the budget
ledger stays honest regardless of which method is used. `info_dict` is logged.

Three are provided:

    rotation    a self-supervised pretext task -- rotate each image by
                0/90/180/270 degrees and predict which. Cheap and
                dependency-free, but the task is only loosely related to
                object identity.
    cluster     DeepCluster-style self-labelling: extract features, k-means
                them, train on the cluster assignment as a pseudo-label, then
                re-cluster and repeat. Closer in spirit to classification than
                rotation is, and it reuses the ordinary cross-entropy path
                rather than needing a bespoke loss.
    supervised  trains on all classes WITH labels. Not self-supervised, and
                deliberately so: it is the control that separates "a shared
                backbone helps" from "self-supervision helps". Without it, a
                gain from phase A is ambiguous.

Both self-supervised options discard their pretext head afterwards -- only the
trunk is carried into phase B, which re-initialises the classifier per agent.
Cluster ids are arbitrary and do not correspond to real classes, so a
cluster-trained head would be meaningless to keep even if phase B kept it.

Adding a contrastive method (SimCLR, SimSiam, BYOL) means writing one function
here that consumes a sample budget and registering it with its views-per-image.
"""

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from tqdm.auto import tqdm


# Registry. `views` is how many forward passes per image the objective does,
# and becomes the default --phase-a-cost-multiplier so the budget reflects real
# work without anyone having to remember a number. `uses_labels` is enforced,
# not advisory: an objective that declares False is handed images only, so it
# CANNOT read a label even by accident.
_REGISTRY = {}


def register(name, views, uses_labels):
    def wrap(fn):
        _REGISTRY[name] = {'fn': fn, 'views': views, 'uses_labels': uses_labels}
        return fn
    return wrap


def objective_names():
    return sorted(_REGISTRY)


def views_per_image(name):
    """Forward passes per image -- the honest cost multiplier for this method."""
    return float(_REGISTRY[name]['views']) if name in _REGISTRY else 1.0


def uses_labels(name):
    return bool(_REGISTRY[name]['uses_labels']) if name in _REGISTRY else True


def classifier_name(model):
    """Attribute holding the final classification layer.

    resnet20 calls it `linear`, vgg calls it `classifier`. Phase A has to swap
    it out to expose features, and phase B has to re-initialise it per agent,
    so both go through here rather than hardcoding one architecture.
    """
    for name in ('linear', 'classifier', 'fc'):
        if isinstance(getattr(model, name, None), nn.Linear):
            return name
    raise AttributeError(
        f'{type(model).__name__} has no recognised classification head '
        f'(looked for .linear, .classifier, .fc). Add its name in '
        f'sesil/ssl.classifier_name().'
    )


def reset_classifier(model):
    """Give the model a freshly initialised head of the same shape.

    Phase B does this per agent: every agent starts from identical backbone
    weights, so a fresh random head is the one cheap source of diversity left
    before their class subsets pull them apart.
    """
    name = classifier_name(model)
    head = getattr(model, name)
    setattr(model, name, nn.Linear(head.in_features, head.out_features).to(
        head.weight.device))
    return model


def set_backbone_trainable(model, trainable):
    """Freeze or unfreeze everything except the classifier.

    Used for the phase-B warmup. An agent starts with a randomly initialised
    head, and a random head produces large, uninformative gradients that flow
    straight back into a backbone that cost most of the pretrain budget to
    build. Holding the backbone still until the head has something to say keeps
    those gradients off it.

    That matters more here than in ordinary transfer learning, because the
    backbone is not only a feature extractor -- it is the SHARED BASIS that
    makes merging work. Agents finetuned from one backbone stay in the same
    loss basin, so alignment is near-identity and crossover destroys little.
    Every unit of backbone drift is drift away from that basis, which is
    exactly the task vector a merge then has to reconcile.

    Returns the number of frozen parameter tensors.
    """
    head = classifier_name(model)
    frozen = 0
    for name, param in model.named_parameters():
        if name.startswith(head + '.'):
            param.requires_grad_(True)
        else:
            param.requires_grad_(trainable)
            frozen += int(not trainable)
    return frozen


def _iterate_for(loader, sample_budget, with_labels):
    """Yield batches until `sample_budget` images have been presented.

    Cycles the loader when it runs out, so a budget larger than one epoch just
    keeps going, and one smaller than an epoch stops partway.

    When `with_labels` is False the labels are DROPPED here rather than merely
    ignored downstream. A self-supervised objective then has no label to read,
    so "phase A uses no labels" is a property of the code path instead of a
    promise in a docstring.
    """
    seen = 0
    while seen < sample_budget:
        for images, labels in loader:
            yield (images, labels) if with_labels else images
            seen += images.shape[0]
            if seen >= sample_budget:
                return


@register('rotation', views=4, uses_labels=False)
def train_rotation(model, loader, sample_budget, args):
    """Self-supervised: predict which of four rotations was applied.

    The classification head is replaced by a fresh 4-way head for the pretext
    task and discarded afterwards -- only the backbone is kept.

    Genuinely label-free: the iterator hands this function images only, so
    there is no label in scope to use. The targets are rotation indices this
    function generates itself.
    """
    device = args.device

    head_name = classifier_name(model)
    original_head = getattr(model, head_name)
    pretext_head = nn.Linear(original_head.in_features, 4).to(device)

    setattr(model, head_name, nn.Identity())     # expose features
    model = model.to(device).train()

    params = list(model.parameters()) + list(pretext_head.parameters())
    optimiser = torch.optim.Adam(params, lr=1e-3)

    seen, correct, total, loss_sum, steps = 0, 0, 0, 0.0, 0
    pbar = tqdm(total=int(sample_budget), desc='Phase A (rotation)', unit='img')

    # Each image contributes all four rotations, so one image presented is four
    # forward passes -- which is what VIEWS_PER_IMAGE records for the ledger.
    for images in _iterate_for(loader, sample_budget, with_labels=False):
        images = images.to(device)
        batch = images.shape[0]

        views = torch.cat([torch.rot90(images, k, dims=(2, 3)) for k in range(4)])
        targets = torch.arange(4, device=device).repeat_interleave(batch)

        optimiser.zero_grad(set_to_none=True)
        logits = pretext_head(model(views))
        loss = F.cross_entropy(logits, targets)
        loss.backward()
        optimiser.step()

        loss_sum += loss.item()
        steps += 1
        correct += (logits.argmax(1) == targets).sum().item()
        total += targets.numel()
        seen += batch
        pbar.update(batch)

    pbar.close()
    setattr(model, head_name, original_head)     # restore the real head shape

    return model, {
        'objective': 'rotation',
        'images_seen': seen,
        'pretext_accuracy': correct / max(total, 1),
        'mean_loss': loss_sum / max(steps, 1),
    }


@register('cluster', views=2, uses_labels=False)
def train_cluster(model, loader, sample_budget, args):
    """DeepCluster-style self-labelling. Label-free.

    Each round:
        1. forward a slice of the data to get features (no gradients)
        2. k-means those features into --cluster-k groups
        3. train the trunk to predict the cluster assignment, as ordinary
           cross-entropy against a temporary k-way head

    Re-clustering every round is what stops the assignment ossifying around
    whatever the randomly-initialised network happened to encode first.

    The cost multiplier is 2: one forward pass for feature extraction plus one
    forward-backward for training. That rounds the extraction pass up from its
    true ~1/3 of a training step, which errs toward charging SESiL more rather
    than less.

    Images for a round are cached on CPU so the same slice can be clustered and
    then trained on. At CIFAR sizes that is a few hundred MB at most; a larger
    dataset would want index bookkeeping instead.
    """
    from sklearn.cluster import KMeans

    device = args.device
    k = int(getattr(args, 'cluster_k', 0) or args.num_classes)
    rounds = max(int(getattr(args, 'cluster_rounds', 5)), 1)
    per_round = max(int(sample_budget // rounds), 1)

    head_name = classifier_name(model)
    original_head = getattr(model, head_name)
    pretext_head = nn.Linear(original_head.in_features, k).to(device)

    setattr(model, head_name, nn.Identity())     # expose features
    model = model.to(device)

    optimiser = torch.optim.Adam(
        list(model.parameters()) + list(pretext_head.parameters()), lr=1e-3)

    seen, loss_sum, steps, correct, total = 0, 0.0, 0, 0, 0
    cluster_sizes = []
    pbar = tqdm(total=int(sample_budget), desc='Phase A (cluster)', unit='img')

    for _round in range(rounds):
        # ---- 1. features, no gradients ------------------------------- #
        cached, feats = [], []
        model.eval()
        with torch.no_grad():
            for images in _iterate_for(loader, per_round, with_labels=False):
                cached.append(images)
                feats.append(model(images.to(device)).detach().cpu())
        if not cached:
            break
        features = torch.cat(feats).numpy()

        # ---- 2. cluster ---------------------------------------------- #
        n_clusters = min(k, len(features))
        assignment = KMeans(n_clusters=n_clusters, n_init=4,
                            random_state=args.seed).fit_predict(features)
        counts = np.bincount(assignment, minlength=n_clusters)
        cluster_sizes.append([int(c) for c in counts])

        # ---- 3. train on the pseudo-labels --------------------------- #
        model.train()
        offset = 0
        pseudo = torch.from_numpy(assignment).long()
        for images in cached:
            batch = images.shape[0]
            targets = pseudo[offset:offset + batch].to(device)
            offset += batch

            optimiser.zero_grad(set_to_none=True)
            logits = pretext_head(model(images.to(device)))
            loss = F.cross_entropy(logits, targets)
            loss.backward()
            optimiser.step()

            loss_sum += loss.item()
            steps += 1
            correct += (logits.argmax(1) == targets).sum().item()
            total += targets.numel()
            seen += batch
            pbar.update(batch)

    pbar.close()
    setattr(model, head_name, original_head)     # restore the real head shape

    # A near-degenerate clustering -- almost everything in one group -- means the
    # pretext task taught the trunk nothing, so it is worth surfacing.
    largest = max((max(c) / max(sum(c), 1) for c in cluster_sizes), default=0.0)

    return model, {
        'objective': 'cluster',
        'images_seen': seen,
        'clusters': k,
        'rounds': rounds,
        'pseudo_label_accuracy': correct / max(total, 1),
        'largest_cluster_fraction': round(largest, 4),
        'mean_loss': loss_sum / max(steps, 1),
    }


@register('supervised', views=1, uses_labels=True)
def train_supervised(model, loader, sample_budget, args):
    """Control: ordinary supervised training on all classes. USES LABELS.

    Not self-supervised. It exists so a phase-A gain can be attributed -- if
    this does as well as the pretext task, the benefit is the shared
    initialisation rather than self-supervision.
    """
    device = args.device
    model = model.to(device).train()
    optimiser = torch.optim.Adam(model.parameters(), lr=1e-3)

    seen, correct, total, loss_sum, steps = 0, 0, 0, 0.0, 0
    pbar = tqdm(total=int(sample_budget), desc='Phase A (supervised)', unit='img')

    for images, labels in _iterate_for(loader, sample_budget, with_labels=True):
        images, labels = images.to(device), labels.to(device)

        optimiser.zero_grad(set_to_none=True)
        logits = model(images)
        loss = F.cross_entropy(logits, labels)
        loss.backward()
        optimiser.step()

        loss_sum += loss.item()
        steps += 1
        correct += (logits.argmax(1) == labels).sum().item()
        total += labels.numel()
        seen += images.shape[0]
        pbar.update(images.shape[0])

    pbar.close()

    return model, {
        'objective': 'supervised',
        'images_seen': seen,
        'train_accuracy': correct / max(total, 1),
        'mean_loss': loss_sum / max(steps, 1),
    }


def get_objective(name):
    """Look up a phase-A objective by --phase-a-method."""
    if name not in _REGISTRY:
        raise ValueError(
            f'Unknown phase-A method {name!r}. '
            f'Choose one of {objective_names()}, or add one to sesil/ssl.py '
            f'with @register(name, views=..., uses_labels=...).'
        )
    return _REGISTRY[name]['fn']
