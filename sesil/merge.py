"""
Crossover: turn a pair of parents into one merged offspring.

Which alignment function is used (ZipIt!, permutation, weight averaging) is an
argument, resolved through sesil.registry -- it is not baked into the script.
"""

import os
from copy import deepcopy

from model_merger import ModelMerge
from utils import (
    get_merging_fn,
    prepare_experiment_config,
    reset_bn_stats,
)

from sesil.registry import get_merger_name


def point_at(raw_config, agent_ids):
    """Point the config at one or more agents' checkpoints.

    Replaces the old inject_model / inject_pair, which also had to write
    `class_splits` derived from each agent's decoded name. Agents no longer
    carry names, and evaluation now covers the whole label space, so no splits
    are needed -- which also saves prepare_data a full scan of the dataset to
    build per-split loaders it never used.
    """
    model_name = raw_config['model']['name']
    raw_config['dataset'].pop('class_splits', None)
    raw_config['model']['bases'] = [
        os.path.join(raw_config['model']['dir'], agent_id, f'{model_name}_v0.pth.tar')
        for agent_id in agent_ids
    ]
    return raw_config


def node_params(args):
    """Hyper-parameters handed to the alignment function."""
    return {'a': args.merge_alpha, 'b': args.merge_beta}


def merge_pair(pair, raw_config, args):
    """Merge one couple.

    Returns (merged_model, config) -- the config is returned too because it
    carries the loaders evaluation needs.
    """
    point_at(raw_config, pair)
    config = prepare_experiment_config(raw_config)

    train_loader = config['data']['train']['full']
    base_models = [reset_bn_stats(base_model, train_loader)
                   for base_model in config['models']['bases']]

    Grapher = config['graph']
    graphs = [Grapher(deepcopy(base_model)).graphify() for base_model in base_models]

    merge = ModelMerge(*graphs, device=args.device)
    merge.transform(
        deepcopy(config['models']['new']),
        train_loader,
        transform_fn=get_merging_fn(get_merger_name(args.merger)),
        metric_classes=config['metric_fns'],
        stop_at=args.stop_node,
        **node_params(args),
    )

    # The merged model inherits neither parent's BN statistics.
    reset_bn_stats(merge, train_loader)

    return merge, config
