"""The genetic-learning dataset: ordered parent pairs, their profiles, and
the children they actually produced.

Built from finished SESiL runs rather than sampled fresh in phase C, so the
mating network learns from real populations at every generation, certified
the way evolution certifies them (across the whole society), instead of from
a few gen-0 societies of 8.

Two tables, joined on (run_id, generation, base, donor):

    pairs     one row per ORDERED pair (base, donor) of agents alive in the
              same generation of the same run. The genetic profile -- the six
              GLOBA shares of what the donor brings, typed against the base --
              plus the donor's unnormalised novelty and each parent's measured
              state (validation accuracy, certificate). Every pair of every
              generation, mated or not: the profile needs only the weights.

    children  one row per MEASURED child merge(base, donor): its validation
              accuracy right after the merge (BatchNorm recalibrated, before
              any mutation), and how it was made (merger, preset, head).
              source 'evolution' = bred during the run and read back from its
              log; source 'generated' = built afterwards by this module for a
              pair the run never mated.

plus runs.json, one record per source run with the settings that produced it.

Layout follows the Hugging Face multi-config convention so the folder can be
published as is:

    <root>/README.md                 dataset card (YAML header + description)
    <root>/runs.json
    <root>/pairs/<run_key>.parquet   (or .csv.gz without pyarrow)
    <root>/children/<run_key>.<source>.parquet

One shard per source run, so building is resumable and a run can be added or
dropped without touching the rest.
"""

import gzip
import csv
import json
import os
import random
import re
import time
from types import SimpleNamespace

from sesil.globa_stats import TYPES

SCHEMA_VERSION = 1
SHARE_COLUMNS = [f'share_{t}' for t in TYPES]
LIST_COLUMNS = {'base_cert', 'donor_cert', 'child_cert', 'child_val_per_class'}
KEY = ('run_id', 'generation', 'base', 'donor')


# --------------------------------------------------------------------------- #
# Table IO: Parquet when pyarrow is installed, gzipped CSV otherwise
# --------------------------------------------------------------------------- #

def _have_pyarrow():
    try:
        import pyarrow  # noqa: F401
        return True
    except ImportError:
        return False


def write_table(rows, stem):
    """Write rows (list of dicts) to <stem>.parquet, or <stem>.csv.gz."""
    os.makedirs(os.path.dirname(os.path.abspath(stem)), exist_ok=True)
    if _have_pyarrow():
        import pyarrow as pa
        import pyarrow.parquet as pq
        path = stem + '.parquet'
        pq.write_table(pa.Table.from_pylist(rows), path + '.tmp')
    else:
        path = stem + '.csv.gz'
        fields = list(rows[0].keys()) if rows else []
        with gzip.open(path + '.tmp', 'wt', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for r in rows:
                w.writerow({k: (json.dumps(v) if k in LIST_COLUMNS else v)
                            for k, v in r.items()})
    os.replace(path + '.tmp', path)
    return path


def _cast(value):
    if value in ('', None):
        return None
    if value in ('True', 'False'):
        return value == 'True'
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def read_table(path):
    if path.endswith('.parquet'):
        import pyarrow.parquet as pq
        return pq.read_table(path).to_pylist()
    with gzip.open(path, 'rt', newline='', encoding='utf-8') as f:
        rows = []
        for r in csv.DictReader(f):
            rows.append({k: (json.loads(v) if k in LIST_COLUMNS and v else _cast(v))
                         for k, v in r.items()})
        return rows


def shard_stem(root, table, run_key, suffix=''):
    return os.path.join(root, table, run_key + (f'.{suffix}' if suffix else ''))


def existing_shard(stem):
    for ext in ('.parquet', '.csv.gz'):
        if os.path.exists(stem + ext):
            return stem + ext
    return None


def load_table(root, table):
    """Every shard of one table, concatenated."""
    folder = os.path.join(root, table)
    rows = []
    if not os.path.isdir(folder):
        return rows
    for name in sorted(os.listdir(folder)):
        if name.endswith(('.parquet', '.csv.gz')):
            rows.extend(read_table(os.path.join(folder, name)))
    return rows


# --------------------------------------------------------------------------- #
# Finding source runs
# --------------------------------------------------------------------------- #

def run_key(run_id):
    """A file-name-safe key for a run id."""
    return re.sub(r'[^A-Za-z0-9._-]+', '_', run_id).strip('_')


def discover_runs(results_root, include=None):
    """SESiL runs that can contribute: SSL backbone + per-generation weights.

    A profile is a task-vector decomposition against the phase-A backbone, so a
    run without one (supervised or no pretrain) has no profile to give.
    Returns [(run_dir, run_id, config_dict)], sorted by run id.
    """
    found = []
    for dirpath, dirs, files in os.walk(results_root):
        if 'config.json' not in files or 'train.jsonl' not in files:
            continue
        dirs[:] = []
        run_id = os.path.relpath(dirpath, results_root).replace('\\', '/')
        if include and not any(re.search(p, run_id) for p in include):
            continue
        with open(os.path.join(dirpath, 'config.json'), encoding='utf-8') as f:
            cfg = json.load(f).get('config', {})
        if cfg.get('method') != 'sesil' or cfg.get('pretrain_mode') != 'ssl':
            continue
        arch = cfg.get('arch')
        if not os.path.exists(os.path.join(dirpath, 'backbone', f'{arch}.pth.tar')):
            continue
        if not os.path.isdir(os.path.join(dirpath, 'checkpoints')):
            continue
        found.append((dirpath, run_id, cfg))
    return sorted(found, key=lambda r: r[1])


RUN_FIELDS = ('dataset', 'seed', 'arch', 'pop_size', 'phase_a_method',
              'mating_mode', 'cert_with', 'globa_with', 'merger', 'merge_head',
              'globa_preset', 'mutation_mode', 'mating_rounds', 'mate_floor',
              'individual_budget', 'certify_top_frac', 'certify_floor',
              'probe_eta', 'probe_svd_energy', 'probe_basis_energy',
              'probe_layer_weighting', 'val_fraction', 'budget')


def run_record(run_id, cfg):
    rec = {'run_id': run_id, 'experiment': run_id.split('/')[0]}
    rec.update({k: cfg.get(k) for k in RUN_FIELDS})
    return rec


def read_run_log(run_dir):
    """Per-generation agent states and bred children, from train.jsonl."""
    agents = {}          # (generation, agent) -> population record
    offspring = []
    with open(os.path.join(run_dir, 'train.jsonl'), encoding='utf-8') as f:
        for line in f:
            try:
                r = json.loads(line)
            except ValueError:
                continue
            stage = r.get('stage')
            if stage == 'population':
                agents[(r['generation'], r['agent'])] = r
            elif stage == 'offspring':
                offspring.append(r)
    return agents, offspring


def evolution_children(run_id, cfg, agents, offspring):
    """Children the run itself bred, as `children` rows."""
    rows = []
    for o in offspring:
        parents = o.get('parents') or []
        base = o.get('base_parent')
        if len(parents) != 2 or base not in parents:
            continue
        donor = parents[1] if parents[0] == base else parents[0]
        g = o['generation']
        rows.append({
            'run_id': run_id, 'generation': g, 'base': base, 'donor': donor,
            'merger': cfg.get('merger'),
            'globa_preset': cfg.get('globa_preset') if cfg.get('merger') == 'globa' else None,
            'merge_head': cfg.get('merge_head'),
            'cert_pool': sum(1 for (gg, _a) in agents if gg == g),
            'child_val_overall': o.get('val_overall'),
            'child_cert': sorted(o.get('certificate') or []),
            'child_val_per_class': o.get('val_per_class'),
            'source': 'evolution',
        })
    return rows


# --------------------------------------------------------------------------- #
# Loading a run back
# --------------------------------------------------------------------------- #

def run_args(run_dir, cfg, device='cpu', data_dir=None):
    """The run's own resolved settings, pointed at where it lives now."""
    args = SimpleNamespace(**cfg)
    args.run_dir = run_dir
    args.population_dir = os.path.join(run_dir, 'checkpoints', 'gen_0')
    args.device = device
    if data_dir:
        args.data_dir = data_dir
    return args


def load_states(run_dir, generation, arch, device='cpu'):
    from sesil.probe.driver import _load_state
    from sesil.population import list_population
    gdir = os.path.join(run_dir, 'checkpoints', f'gen_{generation}')
    return {a: {k: v.to(device) for k, v in _load_state(gdir, a, arch).items()}
            for a in list_population(gdir)}


def load_core(run_dir, arch, device='cpu'):
    import torch
    core = torch.load(os.path.join(run_dir, 'backbone', f'{arch}.pth.tar'),
                      map_location='cpu')
    if isinstance(core, dict) and 'state_dict' in core:
        core = core['state_dict']
    return {k: v.to(device) for k, v in core.items()}


def head_prefix_for(args):
    from sesil.pretrain import build_model
    from sesil.ssl import classifier_name
    dev, args.device = args.device, 'cpu'
    try:
        return classifier_name(build_model(args, args.num_classes)) + '.'
    finally:
        args.device = dev


def generations_on_disk(run_dir):
    ck = os.path.join(run_dir, 'checkpoints')
    return sorted(int(g.split('_')[1]) for g in os.listdir(ck)
                  if g.startswith('gen_') and g.split('_')[1].isdigit())


# --------------------------------------------------------------------------- #
# Building
# --------------------------------------------------------------------------- #

def pair_rows(run_id, generation, states, core, args, agents, head_prefix):
    """Every ordered pair of one generation, profiles from one SVD per pair."""
    from sesil.globa_stats import directional_pair_stats
    ids = sorted(states)
    rows = []
    for i, a in enumerate(ids):
        for b in ids[i + 1:]:
            a_sees_b, b_sees_a = directional_pair_stats(
                states[a], states[b], core,
                eta=args.probe_eta, svd_energy=args.probe_svd_energy,
                basis_energy=args.probe_basis_energy, head_prefix=head_prefix,
                weighting=args.probe_layer_weighting)
            for base, donor, prof in ((a, b, a_sees_b), (b, a, b_sees_a)):
                pb = agents.get((generation, base), {})
                pd = agents.get((generation, donor), {})
                cb, cd = set(pb.get('certificate') or []), set(pd.get('certificate') or [])
                row = {'run_id': run_id, 'generation': generation,
                       'base': base, 'donor': donor}
                row.update({c: float(prof[t]) for c, t in zip(SHARE_COLUMNS, TYPES)})
                row['novel'] = float(prof.get('_novel', 0.0))
                row.update({
                    'base_val_overall': pb.get('val_overall'),
                    'donor_val_overall': pd.get('val_overall'),
                    'base_cert_size': len(cb), 'donor_cert_size': len(cd),
                    'cert_overlap': len(cb & cd),
                    'base_cert': sorted(cb), 'donor_cert': sorted(cd),
                })
                rows.append(row)
    return rows


def build_pairs(run_dir, run_id, cfg, out_root, device='cpu', log=print):
    """pairs + evolution children shards for one run. Skips a finished shard."""
    key = run_key(run_id)
    agents, offspring = read_run_log(run_dir)

    child_stem = shard_stem(out_root, 'children', key, 'evolution')
    kids = evolution_children(run_id, cfg, agents, offspring)
    if kids and not existing_shard(child_stem):
        write_table(kids, child_stem)

    stem = shard_stem(out_root, 'pairs', key)
    if existing_shard(stem):
        log(f'[dataset] {run_id}: pairs already built, skipping')
        return 0, len(kids)

    args = run_args(run_dir, cfg, device=device)
    core = load_core(run_dir, args.arch, device)
    head_prefix = head_prefix_for(args)
    mated = {(c['generation'], c['base'], c['donor']) for c in kids}

    rows = []
    gens = [g for g in generations_on_disk(run_dir)
            if any(gg == g for (gg, _a) in agents)]
    for g in gens:
        t0 = time.time()
        states = load_states(run_dir, g, args.arch, device)
        new = pair_rows(run_id, g, states, core, args, agents, head_prefix)
        for r in new:
            r['mated'] = (g, r['base'], r['donor']) in mated
        rows.extend(new)
        log(f'[dataset] {run_id} gen {g}: {len(new)} ordered pairs '
            f'({time.time() - t0:.1f}s)')
        del states
    if rows:
        write_table(rows, stem)
    return len(rows), len(kids)


def generate_children(run_dir, run_id, cfg, out_root, per_generation,
                      preset, head, device='cuda', data_dir=None, seed=0,
                      log=print):
    """Build and measure children for pairs the run never bred.

    The child is made exactly as evolution makes it: GLOBA merge(base, donor)
    from that generation's weights, with that generation's certificates
    (certified across the whole society), BatchNorm recalibrated on the run's
    own train split, measured on its own validation split.
    """
    import torch
    from sesil.data import DataBundle
    from sesil.fitness import evaluate_all_classes
    from sesil.globa_merge import merge as globa_merge
    from sesil.pretrain import build_model
    from utils import reset_bn_stats

    key = run_key(run_id)
    suffix = f'generated.{preset}_{head}'
    stem = shard_stem(out_root, 'children', key, suffix)
    done_path = existing_shard(stem)
    rows = read_table(done_path) if done_path else []
    done = {(r['generation'], r['base'], r['donor']) for r in rows}

    agents, offspring = read_run_log(run_dir)
    known = {(c['generation'], c['base'], c['donor'])
             for c in evolution_children(run_id, cfg, agents, offspring)
             if c['merger'] == 'globa' and c['globa_preset'] == preset
             and c['merge_head'] == head}

    args = run_args(run_dir, cfg, device=device, data_dir=data_dir)
    data = DataBundle(args)
    train_loader, val_loader = data.train_loader(), data.val_loader()
    core = load_core(run_dir, args.arch, 'cpu')
    head_prefix = head_prefix_for(args)
    rng = random.Random(f'{run_id}|{seed}|{preset}|{head}')

    for g in [g for g in generations_on_disk(run_dir)
              if any(gg == g for (gg, _a) in agents)]:
        ids = sorted(a for (gg, a) in agents if gg == g)
        candidates = [(g, a, b) for a in ids for b in ids
                      if a != b and (g, a, b) not in known]
        rng.shuffle(candidates)
        have = sum(1 for d in done if d[0] == g)
        todo = [c for c in candidates if c not in done][:max(0, per_generation - have)]
        if not todo:
            continue
        states = load_states(run_dir, g, args.arch, 'cpu')
        t0 = time.time()
        for _g, base, donor in todo:
            cb = set(agents[(g, base)]['certificate'])
            cd = set(agents[(g, donor)]['certificate'])
            child_sd = globa_merge(
                states[base], states[donor], core,
                classes_base=cb, classes_donor=cd, preset=preset, head=head,
                eta=args.probe_eta, svd_energy=args.probe_svd_energy,
                basis_energy=args.probe_basis_energy, head_prefix=head_prefix)
            child = build_model(args, args.num_classes)
            child.load_state_dict(child_sd, strict=False)
            reset_bn_stats(child, train_loader)
            with torch.no_grad():
                per_class, overall = evaluate_all_classes(child, val_loader,
                                                          args.num_classes)
            rows.append({
                'run_id': run_id, 'generation': g, 'base': base, 'donor': donor,
                'merger': 'globa', 'globa_preset': preset, 'merge_head': head,
                'cert_pool': len(ids), 'child_val_overall': float(overall),
                'child_cert': sorted(cb | cd),
                'child_val_per_class': [round(float(v), 5) for v in per_class],
                'source': 'generated',
            })
            done.add((g, base, donor))
            del child, child_sd
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        write_table(rows, stem)      # after every generation: resumable
        log(f'[dataset] {run_id} gen {g}: generated {len(todo)} children '
            f'({time.time() - t0:.1f}s)')
        del states
    return len(rows)


# --------------------------------------------------------------------------- #
# Reading back for training
# --------------------------------------------------------------------------- #

def labelled_groups(root, merger='globa', preset=None, head='label',
                    sources=('evolution', 'generated'), experiments=None):
    """Join pairs with children; group by (run_id, generation).

    Returns {(run_id, generation): [(pair_row, child_row), ...]}. A pair
    measured twice (bred in the run AND generated) keeps the evolution one.
    """
    children = [c for c in load_table(root, 'children')
                if c.get('merger') == merger
                and (preset is None or c.get('globa_preset') == preset)
                and c.get('merge_head') == head
                and c.get('source') in sources
                and c.get('child_val_overall') is not None
                and (not experiments
                     or any(re.search(p, c['run_id']) for p in experiments))]
    by_key = {}
    for c in sorted(children, key=lambda c: c['source'] != 'evolution'):
        by_key.setdefault(tuple(c[k] for k in KEY), c)
    groups = {}
    for p in load_table(root, 'pairs'):
        c = by_key.get(tuple(p[k] for k in KEY))
        if c is not None:
            groups.setdefault((p['run_id'], p['generation']), []).append((p, c))
    return groups


CARD = """---
pretty_name: SESiL genetic pairs
license: cc-by-4.0
task_categories:
- tabular-regression
tags:
- model-merging
- evolutionary-learning
- task-vectors
configs:
- config_name: pairs
  data_files: "pairs/*"
- config_name: children
  data_files: "children/*"
---

# SESiL genetic pairs (schema v{version})

Ordered parent pairs from SESiL populations, their genetic profiles, and the
children they produced. Built by `build_dataset.py` from {n_runs} source runs.

## Tables

**pairs** -- one row per ordered pair (base, donor) of agents alive in the
same generation of the same run.

| column | meaning |
|---|---|
| run_id, generation, base, donor | key; (a, b) and (b, a) are different rows |
| share_A ... share_D_minus | GLOBA profile: share of the donor's task-vector energy of each cell type, typed against the base (sums to 1) |
| novel | donor energy outside D_plus, unnormalised (how much it brings) |
| base_/donor_val_overall | parent's overall validation accuracy that generation |
| base_/donor_cert(_size), cert_overlap | parents' certificates |
| mated | whether the run itself bred merge(base, donor) |

**children** -- one row per measured child merge(base, donor), joined to
pairs on (run_id, generation, base, donor).

| column | meaning |
|---|---|
| merger, globa_preset, merge_head | how the child was built |
| cert_pool | agents the certificates were ranked across |
| child_val_overall | validation accuracy right after the merge (BN recalibrated, before any training) |
| child_cert, child_val_per_class | inherited certificate, per-class validation accuracy |
| source | `evolution` (bred during the run) or `generated` (built afterwards) |

**runs.json** -- the settings of every source run.

## Counts

{counts}
"""


def write_card(root, runs):
    pairs = load_table(root, 'pairs')
    children = load_table(root, 'children')
    by = {}
    for c in children:
        k = (c.get('merger'), c.get('globa_preset'), c.get('merge_head'), c.get('source'))
        by[k] = by.get(k, 0) + 1
    lines = [f'- pairs: {len(pairs)} ordered pairs, '
             f'{len({(p["run_id"], p["generation"]) for p in pairs})} populations']
    for (m, p, h, s), n in sorted(by.items(), key=lambda kv: str(kv[0])):
        lines.append(f'- children ({m}, preset {p}, head {h}, {s}): {n}')
    with open(os.path.join(root, 'runs.json'), 'w', encoding='utf-8') as f:
        json.dump({'schema_version': SCHEMA_VERSION, 'runs': runs}, f, indent=2)
    with open(os.path.join(root, 'README.md'), 'w', encoding='utf-8') as f:
        f.write(CARD.format(version=SCHEMA_VERSION, n_runs=len(runs),
                            counts='\n'.join(lines)))
