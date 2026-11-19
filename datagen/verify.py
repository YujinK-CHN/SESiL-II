"""
Checks for the dataset generator (datagen/). Run: python -m datagen.verify

  1. samples.csv: columns, n_generations and location filled at the end.
  2. GLOBA mating scores taken from the profiles equal globa_score_matrix's.
  3. (info) how many lines of sesil/evolution.py the copy replaces or removes.
"""

import csv
import difflib
import os
import sys
import tempfile
import types

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def check_samples():
    from datagen import evolution as ev
    root = tempfile.mkdtemp()
    part, fin = os.path.join(root, 'p.csv'), os.path.join(root, 'f.csv')
    for g in (0, 3):
        ev._append_samples(part, [{
            'generation': g, 'share_A': 0.5, 'share_B': 0.1, 'share_C': 0.1,
            'share_E': 0.1, 'share_D_plus': 0.1, 'share_D_minus': 0.1,
            'child_val_overall': 0.3, 'type': 'D_minus', 'head': 'label'}])
    n = ev._finalize_samples(part, fin, 5)
    with open(fin, newline='') as f:
        rows = list(csv.DictReader(f))
    ok = (n == 2 and not os.path.exists(part)
          and list(rows[0]) == ev.SAMPLE_COLUMNS
          and [float(r['location']) for r in rows] == [0.0, 0.75]
          and {r['n_generations'] for r in rows} == {'5'}
          and rows[1]['type'] == 'D_minus' and rows[1]['head'] == 'label')
    return ok, rows


def check_scores():
    import random
    from sesil import globa_stats as gs
    from sesil import selection
    ids = [f'agent_{i:03d}' for i in range(6)]
    rng = random.Random(0)
    cache = {}

    def fake(sa, sb, core, **kw):
        if (sa, sb) not in cache:
            def one():
                v = [rng.random() for _ in gs.TYPES]
                d = {t: x / sum(v) for t, x in zip(gs.TYPES, v)}
                d['_novel'] = rng.random()
                return d
            cache[(sa, sb)] = (one(), one())
        return cache[(sa, sb)]

    real = gs.directional_pair_stats
    gs.directional_pair_stats = fake
    try:
        args = types.SimpleNamespace(
            probe_eta=0.8, probe_svd_energy=0.9, probe_basis_energy=0.999,
            head_prefix='fc.', probe_layer_weighting='energy')
        states = {a: a for a in ids}
        bad = []
        for kind in ('D_minus', 'D_plus', 'E'):
            args.globa_with = kind
            nov_ref = {}
            ref = selection.globa_score_matrix(ids, states, None, args,
                                               novelty=nov_ref)
            profiles = selection.pair_profiles(ids, states, None, args)
            got, nov = {a: {} for a in ids}, {}
            for (a, b), prof in profiles.items():
                got[a][b] = selection._globa_value(prof, kind, None)
                nov.setdefault(a, {})[b] = float(prof.get('_novel', 0.0))
            if got != ref or nov != nov_ref:
                bad.append(kind)
    finally:
        gs.directional_pair_stats = real
    return not bad, bad


def check_copy():
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(here, 'sesil', 'evolution.py'), encoding='utf-8') as f:
        orig = f.read().splitlines()
    with open(os.path.join(here, 'datagen', 'evolution.py'), encoding='utf-8') as f:
        copy = f.read().splitlines()
    removed = [l for l in difflib.ndiff(orig, copy)
               if l.startswith('- ') and l[2:].strip()]
    return removed, len(removed)


def main():
    ok1, rows = check_samples()
    print(('PASS' if ok1 else 'FAIL') + '  samples.csv written and finalised'
          + ('' if ok1 else f': {rows}'))
    ok2, bad = check_scores()
    print(('PASS' if ok2 else 'FAIL') + '  mating scores from profiles == '
          'globa_score_matrix' + ('' if ok2 else f' (differ for {bad})'))
    removed, n = check_copy()
    # Lines of the original replaced or removed in the copy ([datagen] blocks,
    # and the mating modes / mergers the generator never runs).
    print(f'INFO  datagen/evolution.py replaces or removes {n} line(s) of '
          f'sesil/evolution.py')
    return 0 if ok1 and ok2 else 1


if __name__ == '__main__':
    sys.exit(main())
