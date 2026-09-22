"""
Population bookkeeping: where individuals live on disk and how they are named.

An individual is a directory whose name is an 8-character hash of its class
subset (see utils.encode_labels / decode_labels, backed by mapping.json), and
which holds one checkpoint file:

    <population dir>/<hash8>/<arch>_v0.pth.tar

Generation 0 is the `initial` population produced by pretrain; generation N is
written under the run directory so that runs never overwrite each other.
"""

import os

import torch

from utils import decode_labels, encode_labels


def generation_dir(args, generation):
    """Directory holding the population at the start of `generation`.

    Generation 0 is the shared initial population; later generations live inside
    this run's own folder, keyed by seed, so parallel seeds stay independent.
    """
    if generation == 0:
        return args.population_dir
    return os.path.join(args.run_dir, 'checkpoints', f'gen_{generation}')


def list_population(population_dir):
    """Model ids present in a population directory.

    Returns [] if the directory does not exist, which is how main.py detects
    that pretrain still has to run.
    """
    if not os.path.isdir(population_dir):
        return []
    return sorted(
        name for name in os.listdir(population_dir)
        if os.path.isdir(os.path.join(population_dir, name))
    )


def checkpoint_path(population_dir, model_id, arch, version=0):
    """Path of one individual's weights."""
    return os.path.join(population_dir, model_id, f'{arch}_v{version}.pth.tar')


def sort_model_name_unique(name):
    """'5_9_5_7' -> '5_7_9'.  Canonical, de-duplicated, sorted label string."""
    nums = sorted(set(int(x) for x in name.split('_')))
    return '_'.join(map(str, nums))


def labels_of(model_id):
    """Model id (hash or raw label string) -> sorted unique label string."""
    return sort_model_name_unique(decode_labels(model_id))


def build_unique_key(labels, seen_keys):
    """Short unique key for an offspring, from the union of its parents' labels.

    Two different couples can produce the same label set; rotating the label
    order keeps the keys distinct, and a numeric suffix is the last resort.
    """
    sorted_labels = sorted(set(labels), key=int)
    base_key = '_'.join(sorted_labels)

    key = base_key
    rotation = 0
    while key in seen_keys:
        rotation += 1
        idx = rotation % len(sorted_labels)
        rotated = sorted_labels[idx:] + sorted_labels[:idx]
        key = '_'.join(rotated)

        if rotation >= len(sorted_labels):
            suffix = 1
            while f'{base_key}_{suffix}' in seen_keys:
                suffix += 1
            key = f'{base_key}_{suffix}'
            break

    seen_keys.add(key)
    return key


def save_individual(model, population_dir, label_key, arch, head_index=0):
    """Persist one individual under its hashed directory.

    A merged model is a ModelMerge, whose weights live in head_models; a plain
    nn.Module is saved directly.
    """
    model_id = encode_labels(label_key)
    save_dir = os.path.join(population_dir, model_id)
    os.makedirs(save_dir, exist_ok=True)

    version = len([f for f in os.listdir(save_dir) if f.endswith('.pth.tar')])
    save_path = os.path.join(save_dir, f'{arch}_v{version}.pth.tar')

    if hasattr(model, 'head_models'):
        state_dict = model.head_models[head_index].state_dict()
    else:
        state_dict = model.state_dict()
    torch.save(state_dict, save_path)

    return save_path
