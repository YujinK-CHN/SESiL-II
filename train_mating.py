"""
Train the mating network on the genetic-learning dataset, on its own.

    python train_mating.py --dataset dataset/sesil_genetic_pairs \
        --name globa_label_v0 --head label --test-seeds 3

Writes mating_models/<name>/pair_quality.pt (+ .json with the fit report).
A SESiL run then uses it with

    --mating-mode learn --mating-model mating_models/<name>/pair_quality.pt

and skips phase C.

What one sample is: an ordered pair (base, donor) in one generation of one
run -> its six-share genetic profile -> the measured child merge(base, donor)
-> the child's validation accuracy right after the merge. Samples are grouped
by (run, generation), the population they were drawn from, which is what the
ranking term (--rank-weight > 0) compares within.

Test split is by RUN, never by row: rows of one population share parents, so
a row split would test on near-copies of training rows. --test-seeds holds out
every run with those seeds; the network never sees their populations.
"""

import argparse
import json
import os
import sys

import numpy as np


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--dataset', default='dataset/sesil_genetic_pairs')
    p.add_argument('--name', required=True)
    p.add_argument('--out-root', default='mating_models')
    p.add_argument('--merger', default='globa')
    p.add_argument('--preset', default=None,
                   help='GLOBA preset of the children to learn from (default: any)')
    p.add_argument('--head', default='label', choices=['label', 'average'])
    p.add_argument('--sources', default='evolution,generated')
    p.add_argument('--include', nargs='*', default=None,
                   help='regexes on run id; only matching runs are used')
    p.add_argument('--test-seeds', default='3',
                   help='comma list; runs with these seeds are held out ('' = none)')
    p.add_argument('--hidden', type=int, default=16)
    p.add_argument('--epochs', type=int, default=400)
    p.add_argument('--lr', type=float, default=0.01)
    p.add_argument('--rank-weight', type=float, default=0.0)
    p.add_argument('--seed', type=int, default=0)
    a = p.parse_args(argv)

    import torch
    from sesil import genetic_dataset as gd
    from sesil import mating_fn

    groups = gd.labelled_groups(a.dataset, merger=a.merger, preset=a.preset,
                                head=a.head, sources=tuple(a.sources.split(',')),
                                experiments=a.include)
    if not groups:
        print('[train_mating] no labelled pairs match these filters')
        return 1

    with open(os.path.join(a.dataset, 'runs.json'), encoding='utf-8') as f:
        seed_of = {r['run_id']: r.get('seed') for r in json.load(f)['runs']}
    test_seeds = {int(s) for s in a.test_seeds.split(',') if s.strip()}

    def sample(rows):
        return {'shares': np.array([[pr[c] for c in gd.SHARE_COLUMNS]
                                    for pr, _c in rows], dtype=np.float64),
                'outcome': np.array([c['child_val_overall'] for _p, c in rows],
                                    dtype=np.float64)}

    train, test = [], []
    for (run_id, gen), rows in sorted(groups.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        (test if seed_of.get(run_id) in test_seeds else train).append(sample(rows))
    n_tr = sum(len(s['outcome']) for s in train)
    n_te = sum(len(s['outcome']) for s in test)
    print(f'[train_mating] train: {len(train)} populations, {n_tr} children; '
          f'test: {len(test)} populations, {n_te} children')
    if not train:
        print('[train_mating] nothing left to train on after the test split')
        return 1

    net, history, report = mating_fn.train_quality(
        train, hidden=a.hidden, epochs=a.epochs, lr=a.lr, seed=a.seed,
        rank_weight=a.rank_weight)
    if test:
        report['test'] = mating_fn.quality_fit_report(net, test)
        print(f'[train_mating] held-out runs (seeds {sorted(test_seeds)})  '
              f'R2 {report["test"]["r2"]:+.3f}  '
              f'spearman {report["test"]["spearman"]:+.3f}')

    outcomes = np.concatenate([s['outcome'] for s in train])
    with torch.no_grad():
        preds = np.concatenate([
            net(torch.tensor(s['shares'], dtype=torch.float32)).numpy()
            for s in train])
    quant = lambda v: {str(q): float(np.percentile(v, q)) for q in (10, 25, 50, 75, 90)}
    meta = {
        'name': a.name, 'dataset': os.path.abspath(a.dataset),
        'filters': {'merger': a.merger, 'preset': a.preset, 'head': a.head,
                    'sources': a.sources, 'include': a.include,
                    'test_seeds': sorted(test_seeds)},
        'train_populations': len(train), 'train_children': n_tr,
        'test_populations': len(test), 'test_children': n_te,
        'hidden': a.hidden, 'epochs': a.epochs, 'lr': a.lr,
        'rank_weight': a.rank_weight, 'seed': a.seed,
        'outcome': 'child_val_overall',
        'target_quantiles': quant(outcomes), 'pred_quantiles': quant(preds),
        'fit': report, 'loss_history': history,
    }
    path = os.path.join(a.out_root, a.name, 'pair_quality.pt')
    mating_fn.save_quality(net, path, meta=meta)
    print('[train_mating] predicted p10..p90 '
          + '  '.join(f'{v:.3f}' for v in meta['pred_quantiles'].values()))
    print(f'[train_mating] wrote {path}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
