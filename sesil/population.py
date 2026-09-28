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

Every generation, including generation 0 from pretrain, is written under the
run directory, so runs and seeds never overwrite each other and nothing is
shared between them.
"""

import json
import os

import torch


AGENT_META = 'agent.json'


def generation_dir(args, generation):
    """Directory holding the population at the start of `generation`.

    Uniform: generation 0 is the first generation, not a special shared
    location. Every run builds its own, so concurrent runs never collide and
    none inherits another's state.
    """
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


def prune_generations(args, newest, keep, protect=(0,)):
    """Delete generation directories that nothing will read again.

    The evolution loop only ever touches three generation directories: the one
    it is reading (`gen_N`), the one it is writing (`gen_N+1`), and the final
    one at the end. Nothing reads `gen_K` for K below the current generation --
    verified by there being exactly three generation_dir() call sites. So every
    older directory is dead weight, and at 20 agents x ~18 MB it is 360 MB per
    generation, 7 GB per seed, 22 GB per round of three seeds.

    `keep` is how many of the most recent generations to retain, counted back
    from `newest`. Two is the minimum that can be correct: the loop needs
    `gen_N` and `gen_N+1` alive at the same moment. `keep <= 0` means keep
    everything, which is the old behaviour.

    `protect` names generations that are never deleted whatever `keep` says.
    Generation 0 is protected because it is the pretrained population -- the
    only thing in the tree that cannot be regenerated from a later state, and
    the starting point anyone re-running the evolution from scratch needs.

    ONLY THE WEIGHTS GO. agent.json stays. The weights are ~18 MB each and the
    metadata is 454 bytes, so keeping it costs 20 kB per generation against 360
    MB recovered -- and it is the only place an agent's parents are recorded
    against its own id. train.jsonl logs a couple's parents but not which
    agent_NNN the child became; that mapping can be inferred from the order
    breed() appends children, but an inference that depends on append order is
    not the same as a record, and it silently becomes wrong if that order ever
    changes.

    The cost of pruning is that --start-gen can only resume from a generation
    whose weights are still on disk, so in practice only from the most recent
    one or two. A pruned generation keeps its directory and its metadata, so it
    still lists as a population -- resuming from one would fail on the missing
    weights rather than look empty.

    Returns the list of generation directories pruned, for logging.
    """

    if keep is None or keep <= 0:
        return []

    root = os.path.join(args.run_dir, 'checkpoints')
    if not os.path.isdir(root):
        return []

    # Never delete anything at or above the cutoff, and never a protected one.
    cutoff = newest - keep + 1
    protect = set(protect)

    removed = []
    for name in sorted(os.listdir(root)):
        if not name.startswith('gen_'):
            continue
        try:
            index = int(name[len('gen_'):])
        except ValueError:
            continue                       # not ours to interpret; leave it
        if index in protect or index >= cutoff:
            continue
        path = os.path.join(root, name)

        freed = False
        for agent in sorted(os.listdir(path)):
            agent_dir = os.path.join(path, agent)
            if not os.path.isdir(agent_dir):
                continue
            for entry in sorted(os.listdir(agent_dir)):
                if entry == AGENT_META:
                    continue               # provenance; costs 454 bytes
                try:
                    os.remove(os.path.join(agent_dir, entry))
                    freed = True
                except OSError:
                    pass                   # a concurrent reader; retry next time
        if freed:
            removed.append(path)

    return removed
