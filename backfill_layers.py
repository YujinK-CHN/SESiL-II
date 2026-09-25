"""Write layers.jsonl for probe runs made before the driver logged it.

    python backfill_layers.py results/globa_v3

Recomputes the per-layer GLOBA decomposition from each run's saved agents and
backbone. Nothing is merged and nothing is evaluated -- the expensive half of
the probe is already recorded in pairs.jsonl and is not touched -- so this
costs seconds per seed rather than the ~20 minutes the run itself took.

Only needed for runs whose code predates per-layer logging. Newer runs write
the file themselves and are skipped unless --force is given.

One caveat on exactness. Multi-threaded LAPACK returns slightly different SVDs
at different thread counts, and the pruning threshold turns a tiny basis
difference into a changed type for the odd borderline cell. Backfilled values
therefore differ from what the original process would have written by ~1e-3 --
far below the effect sizes these numbers are used for, but it is why the
re-aggregated totals do not match pairs.jsonl to the last decimal.
"""

import argparse
import itertools
import json
import os
import time

import torch

from sesil.globa_stats import TYPES, pair_stats


def agents_of(gen0):
    return sorted(d for d in os.listdir(gen0)
                  if os.path.isdir(os.path.join(gen0, d)))


def load_state(path):
    sd = torch.load(path, map_location='cpu')
    return sd['state_dict'] if isinstance(sd, dict) and 'state_dict' in sd else sd


def backfill(run_dir, force=False):
    out_path = os.path.join(run_dir, 'layers.jsonl')
    if os.path.exists(out_path) and not force:
        print(f'{run_dir}: layers.jsonl already present, skipping '
              f'(--force to overwrite)')
        return False

    config_path = os.path.join(run_dir, 'config.json')
    if not os.path.exists(config_path):
        print(f'{run_dir}: no config.json, skipping')
        return False
    cfg = json.load(open(config_path))['config']

    arch = cfg['arch']
    backbone = os.path.join(run_dir, 'backbone', f'{arch}.pth.tar')
    gen0 = os.path.join(run_dir, 'checkpoints', 'gen_0')
    if not (os.path.exists(backbone) and os.path.isdir(gen0)):
        print(f'{run_dir}: needs both backbone/ and checkpoints/gen_0/ -- '
              f'were they deleted to save space?')
        return False

    core = load_state(backbone)
    agents = agents_of(gen0)
    states = {}
    for agent in agents:
        path = os.path.join(gen0, agent, f'{arch}_v0.pth.tar')
        if not os.path.exists(path):
            path = os.path.join(gen0, agent, f'{arch}.pth.tar')
        states[agent] = load_state(path)

    # The head is excluded from the decomposition; resnet calls it 'linear',
    # vgg 'classifier'. Detect it rather than assuming.
    head_prefix = next((f'{n}.' for n in ('linear', 'classifier', 'fc')
                        if f'{n}.weight' in core), 'linear.')

    rows, start = [], time.time()
    for a, b in itertools.combinations(agents, 2):
        _summary, per_layer = pair_stats(
            states[a], states[b], core,
            eta=cfg['probe_eta'],
            svd_energy=cfg['probe_svd_energy'],
            basis_energy=cfg['probe_basis_energy'],
            head_prefix=head_prefix,
            weighting=cfg['probe_layer_weighting'])

        for name, s in per_layer.items():
            rows.append({
                'pair': [a, b],
                'layer': name,
                'weight': s['norm_p'] ** 2 + s['norm_q'] ** 2,
                **{f'energy_{t}': s['energy_frac'][t] for t in TYPES},
                **{f'merged_energy_{t}': s['energy_frac_merged'][t] for t in TYPES},
                'cancellation': s['cancellation'],
                'typed_frac_p': s['typed_frac_p'],
                'typed_frac_q': s['typed_frac_q'],
                'cosine': s['cosine'],
                'norm_p': s['norm_p'],
                'norm_q': s['norm_q'],
                'basis_u': s['basis_u'],
                'basis_v': s['basis_v'],
                'nnz_p': s['nnz_p'],
                'nnz_q': s['nnz_q'],
            })

    with open(out_path, 'w') as f:
        for row in rows:
            f.write(json.dumps(row, default=float) + '\n')

    print(f'{run_dir}: {len(agents)} agents, {len(rows)} rows, '
          f'{len(per_layer)} layers/pair, {time.time() - start:.1f}s')
    return True


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('root', help='Directory to search for probe runs.')
    ap.add_argument('--force', action='store_true',
                    help='Rewrite layers.jsonl even where one already exists.')
    args = ap.parse_args()

    runs = []
    for dirpath, _dirnames, filenames in os.walk(args.root):
        if 'pairs.jsonl' in filenames:
            runs.append(dirpath)

    if not runs:
        raise SystemExit(f'No probe runs found under {args.root!r}.')

    done = sum(backfill(run, args.force) for run in sorted(runs))
    print(f'\n{done} of {len(runs)} run(s) backfilled.')


if __name__ == '__main__':
    main()
