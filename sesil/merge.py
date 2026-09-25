"""
Crossover: turn a couple into offspring.

With partial zipping (`--stop-node`), a ModelMerge is a TWO-HEADED model:

    merged_model   the shared trunk, merged up to the stop node. This is where
                   both parents' knowledge is actually combined -- transform()
                   aligns the parents into a common basis and averages them.
    head_models[i] parent i, expressed in that same basis. At inference the
                   unmerge hooks feed it the trunk's intermediate activation, so
                   its own early layers are BYPASSED and only its head matters.

So the composite's output i is "merged trunk + parent i's head" -- two
genuinely different children from a single merge.

Extracting a child therefore means splicing: trunk parameters from
merged_model, post-stop parameters from head_models[i]. Saving head_models[i]
whole would keep its own early layers instead of the merged trunk, which is to
say it would save the parent back out with none of the merge in it.

Which alignment function is used (ZipIt!, permutation, weight averaging) is an
argument, resolved through sesil.registry.
"""

import os
from copy import deepcopy

from graphs.base_graph import NodeType
from model_merger import ModelMerge
from utils import get_merging_fn, prepare_experiment_config, reset_bn_stats

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


def trunk_layer_names(graph):
    """Module names that were merged into the shared basis.

    MergeHandler.prop_back records every node it transforms in `graph.merged`,
    and apply_transformations only walks back from nodes at or before the stop
    node -- so this set is exactly the trunk.
    """
    names = set()
    for node in graph.merged:
        info = graph.get_node_info(node)
        if info.get('type') == NodeType.MODULE:
            names.add(info['layer'])
    return names


def offspring_state_dict(merge, head_index):
    """Splice child `head_index`: merged trunk + that parent's head.

    Returns (state_dict, n_from_trunk). The count is worth logging -- if it is
    zero, nothing of the merge survived into the child and the run is doing
    plain finetuning with extra steps.
    """
    graph = merge.graphs[head_index]
    trunk = trunk_layer_names(graph)

    merged_sd = merge.merged_model.state_dict()
    head_sd = merge.head_models[head_index].state_dict()

    state_dict = {}
    n_from_trunk = 0

    for key, value in head_sd.items():
        module = key.rsplit('.', 1)[0]
        take_trunk = (
            module in trunk
            and key in merged_sd
            and merged_sd[key].shape == value.shape
        )
        if take_trunk:
            state_dict[key] = merged_sd[key].detach().clone()
            n_from_trunk += 1
        else:
            state_dict[key] = value.detach().clone()

    return state_dict, n_from_trunk


def parent_heads_in_merged_basis(merge, args):
    """Each parent's classifier, expressed in the MERGED feature space.

    A parent's own classifier cannot be spliced into a child directly: after a
    permute or zip the trunk emits features in a different basis, so parent i's
    raw rows expect inputs that no longer exist. These are the parents' rows
    after the same transformation the merge applied, which is what makes them
    safe to mix.

    Full merge: asking for an interpolation that puts all the weight on one
    parent returns that parent alone, transformed.
    Partial zip: head_models[i] IS parent i in the merged basis already.
    """
    n_parents = len(merge.graphs)
    if args.stop_node is None:
        return [merge.get_merged_state_dict(
                    interp_w=[1.0 if j == i else 0.0 for j in range(n_parents)])
                for i in range(n_parents)]
    return [merge.head_models[i].state_dict() for i in range(n_parents)]


def apply_label_head(state_dict, parent_heads, parent_classes,
                     head_prefix, num_classes):
    """Rebuild the classifier row by row, from the parent that knows the class.

    Row c of the classifier answers "does this look like class c". A parent
    trained without class c did not leave that row untouched -- class c never
    appeared as a target, so training pushed the row DOWN. It is an
    anti-detector, and averaging it with the other parent's real detector
    cancels part of the signal and halves what survives. Measured on the probe
    data, parents' rows for a class only one of them knows sit at cosine
    -0.072: actively opposed, not merely unrelated.

    So a class exactly one parent knows takes that parent's row whole. Rows
    both parents know, or neither does, are left as the merge produced them --
    there is no better-informed choice available.

    Returns the number of rows taken from a single parent, which is zero when
    the parents' class sets are identical and there is nothing to disentangle.
    """
    replaced = 0
    for key, tensor in list(state_dict.items()):
        if not key.startswith(head_prefix) or not tensor.is_floating_point():
            continue
        # Both weight [C, F] and bias [C] are indexed by class on axis 0.
        if tensor.shape[0] != num_classes:
            continue

        rebuilt = tensor.detach().clone()
        for class_id in range(num_classes):
            owners = [i for i, classes in enumerate(parent_classes)
                      if class_id in classes]
            if len(owners) == 1 and key in parent_heads[owners[0]]:
                rebuilt[class_id] = parent_heads[owners[0]][key][class_id]
                replaced += 1
        state_dict[key] = rebuilt
    return replaced


def interpolation_weights(n_parents, child_index, bias):
    """Weights leaning a full-merge child toward parent `child_index`.

    `bias` goes to that parent, the rest is shared evenly among the others.
    bias=0.5 with two parents makes both children identical, which defeats the
    purpose; bias=1.0 makes a child its parent, with no merge in it at all.
    """
    rest = (1.0 - bias) / max(n_parents - 1, 1)
    return [bias if i == child_index else rest for i in range(n_parents)]


def extract_children(merge, config, args, num_classes, train_loader,
                     parent_classes=None):
    """Standalone child models from a completed merge -- one per parent.

    A couple always yields as many children as it has parents, so the
    population size is preserved whichever merge mode is in use. What makes the
    siblings differ depends on the mode:

      partial zip (--stop-node N)
          The merge has one head per parent. Each child splices the shared
          merged trunk onto a different parent's head.

      full merge (--stop-node none)
          There are no heads -- the whole network is merged into one model. The
          asymmetry comes from the interpolation weight instead: child i is the
          weighted combination leaning toward parent i, via --merge-bias.

    Either way each child is a plain nn.Module, so what gets evaluated,
    finetuned and saved is one and the same object.
    """
    from sesil.pretrain import build_model
    from sesil.ssl import classifier_name

    n_parents = len(merge.graphs)
    children = []

    # Only computed when it will be used: the extra get_merged_state_dict calls
    # are cheap but not free, and they are pointless in 'average' mode.
    use_label_head = (getattr(args, 'merge_head', 'average') == 'label'
                      and parent_classes is not None)
    parent_heads = (parent_heads_in_merged_basis(merge, args)
                    if use_label_head else None)

    for child_index in range(n_parents):
        child = build_model(args, num_classes)

        if args.stop_node is None:
            weights = interpolation_weights(n_parents, child_index, args.merge_bias)
            state_dict = merge.get_merged_state_dict(interp_w=weights)
            n_from_trunk = len(state_dict)
        else:
            state_dict, n_from_trunk = offspring_state_dict(merge, child_index)

        rows_from_one_parent = 0
        if use_label_head:
            rows_from_one_parent = apply_label_head(
                state_dict, parent_heads, parent_classes,
                classifier_name(child) + '.', num_classes)

        child.load_state_dict(state_dict, strict=False)

        # The child owns neither the composite's nor a parent's BN statistics,
        # so they have to be recomputed for the model as assembled.
        reset_bn_stats(child, train_loader)

        children.append((child, n_from_trunk, rows_from_one_parent))

    return children


def merge_couple(pair, raw_config, args, train_loader=None):
    """Merge one couple. Returns (merge, config).

    The caller extracts children from the result -- one merge serves both.

    `train_loader` should be the post-split training loader. Alignment metrics
    and BN recalibration read inputs only, never labels, but keeping validation
    samples out of them entirely is cheaper than having to argue the point.
    """
    point_at(raw_config, pair)
    config = prepare_experiment_config(raw_config)

    if train_loader is None:
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

    return merge, config
