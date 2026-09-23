"""
Population bookkeeping: where agents live on disk.

An agent has no name that encodes what it knows. It is a numbered directory
holding weights plus a small metadata file:

    <population dir>/agent_000/<arch>_v0.pth.tar
    <population dir>/agent_000/agent.json

`agent.json` records the certificate the agent was granted, what it was trained
on, and who its parents were -- provenance, not identity. Nothing downstream
depends on the directory name meaning anything, which is the point: the old
scheme hashed the class subset into the name and needed a global mapping.json
to decode it, and that name could disagree with reality.

Generation 0 is the `initial` population produced by pretrain; generation N is
written under the run directory so that runs and seeds never overwrite each
other.
"""

import json
import os

import torch


AGENT_META = 'agent.json'


def generation_dir(args, generation):
    """Directory holding the population at the start of `generation`."""
    if generation == 0:
        return args.population_dir
    return os.path.join(args.run_dir, 'checkpoints', f'gen_{generation}')


def list_population(population_dir):
    """Agent ids present in a population directory, in stable order.

    Returns [] if the directory does not exist, which is how main.py detects
    that pretrain still has to run.
    """
    if not os.path.isdir(population_dir):
        return []
    return sorted(
        name for name in os.listdir(population_dir)
        if os.path.isdir(os.path.join(population_dir, name))
    )


def agent_name(index):
    """Directory name for the index-th agent of a generation."""
    return f'agent_{index:03d}'


def checkpoint_path(population_dir, agent_id, arch, version=0):
    """Path of one agent's weights."""
    return os.path.join(population_dir, agent_id, f'{arch}_v{version}.pth.tar')


def save_agent(model, population_dir, index, arch, meta=None):
    """Persist one agent: weights plus metadata.

    `model` must be a plain nn.Module. A ModelMerge is explicitly refused: with
    partial zipping it is a composite of a merged trunk plus one head per
    parent, and calling .state_dict() on it -- or on one of its head_models --
    does NOT give a usable child. Saving head_models[i] whole keeps that
    parent's own early layers instead of the merged trunk, which silently
    writes the parent back out with none of the merge in it. Children are
    spliced by sesil.merge.extract_children before they reach here.
    """
    if hasattr(model, 'head_models'):
        raise TypeError(
            'save_agent received a ModelMerge. Extract children with '
            'sesil.merge.extract_children() first -- saving a merge directly '
            'discards the merged trunk.'
        )

    agent_id = agent_name(index)
    save_dir = os.path.join(population_dir, agent_id)
    os.makedirs(save_dir, exist_ok=True)

    save_path = os.path.join(save_dir, f'{arch}_v0.pth.tar')
    torch.save(model.state_dict(), save_path)

    write_meta(population_dir, agent_id, meta or {})

    return save_path


def write_meta(population_dir, agent_id, meta):
    """Write an agent's metadata, with sets rendered as sorted lists."""
    serialisable = {
        k: (sorted(v) if isinstance(v, (set, frozenset)) else v)
        for k, v in meta.items()
    }
    path = os.path.join(population_dir, agent_id, AGENT_META)
    with open(path, 'w') as f:
        json.dump(serialisable, f, indent=2)


def read_meta(population_dir, agent_id):
    """Read an agent's metadata; {} when there is none.

    Missing metadata is normal rather than an error -- a population created
    before certificates existed, or one built by hand, simply starts with no
    inherited certificate and gets one from its first evaluation.
    """
    path = os.path.join(population_dir, agent_id, AGENT_META)
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


def inherited_certificate(population_dir, agent_id):
    """The certificate an agent carried in, as a set. Empty when unknown."""
    return set(read_meta(population_dir, agent_id).get('certificate', []))
