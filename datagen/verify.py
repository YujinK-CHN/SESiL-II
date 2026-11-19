"""
Checks for the dataset generator (datagen/). Run: python -m datagen.verify

  1. samples.csv: columns, n_generations and location filled at the end.
  2. GLOBA mating scores taken from the profiles equal globa_score_matrix's.
  3. --resume: where a run restarts, the budget and evaluator it restores,
     the sample rows it drops, and 'done' once samples.csv exists.
  4. retention / retention_base / retention_donor on hand-computed cases.
  5. (info) how many lines of sesil/evolution.py the copy replaces or removes.
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
    for g in (0, 1, 2):
        ev._append_samples(part, [{
            'generation': g, 'share_A': 0.5, 'share_B': 0.1, 'share_C': 0.1,
            'share_E': 0.1, 'share_D_plus': 0.1, 'share_D_minus': 0.1,
            'child_val_overall': 0.3, 'type': 'D_minus', 'head': 'label'}])
    n, n_gen = ev._finalize_samples(part, fin)
    with open(fin, newline='') as f:
        rows = list(csv.DictReader(f))
    ok = (n == 3 and n_gen == 3 and not os.path.exists(part)
          and list(rows[0]) == ev.SAMPLE_COLUMNS
          and [float(r['location']) for r in rows] == [0.0, 0.5, 1.0]
          and {r['n_generations'] for r in rows} == {'3'}
          and rows[1]['type'] == 'D_minus' and rows[1]['head'] == 'label')
    return ok, rows


def check_retention():
    """Hand-computed cases for retention / retention_base / retention_donor."""
    from datagen.evolution import retention_scores
    base = [0.8, 0.6, 0.1, 0.0]
    donor = [0.2, 0.9, 0.7, 0.0]
    child = [0.4, 0.6, 0.35, 0.5]
    # base certified on {0, 1}, donor on {1, 2}; class 1 is on both -> max 0.9.
    got = retention_scores(child, base, donor, {0, 1}, {1, 2})
    want = {'retention': round((0.4 + 0.6 + 0.35) / (0.8 + 0.9 + 0.7), 6),
            'retention_base': round((0.4 + 0.6) / (0.8 + 0.6), 6),
            'retention_donor': round((0.6 + 0.35) / (0.9 + 0.7), 6)}
    empty = retention_scores(child, base, donor, set(), set())
    one = retention_scores(child, base, donor, {0}, set())
    ok = (got == want
          and empty == {'retention': '', 'retention_base': '', 'retention_donor': ''}
          and one['retention'] == one['retention_base'] == 0.5
          and one['retention_donor'] == '')
    return ok, (got, want, empty, one)


def check_collect():
    """_collect_pair_children end to end with stub models: every ordered pair
    gets a row with all columns, and a bred child is reused, not rebuilt."""
    import torch
    from datagen import evolution as ev
    import sesil.globa_merge as gm
    import sesil.pretrain as pt

    ids = ['agent_000', 'agent_001', 'agent_002']
    info = [{'Model Name': a, 'Certificate': {i}, 'Joint': 0.5 + i / 10,
             'ValPerClass': [0.9 if c == i else 0.1 for c in range(3)]}
            for i, a in enumerate(ids)]
    models = [torch.nn.Linear(2, 3) for _ in ids]
    offspring = [{'is_loner': False, 'base_parent': 'agent_000',
                  'donor_parent': 'agent_001', 'val_overall': 0.77,
                  'per_class': [0.9, 0.45, 0.0]}]
    profiles = {(a, b): {t: 1 / 6 for t in ('A', 'B', 'C', 'E', 'D_plus', 'D_minus')}
                for a in ids for b in ids if a != b}
    run = tempfile.mkdtemp()
    args = types.SimpleNamespace(
        run_dir=run, globa_preset='globa', merge_head='label', probe_eta=0.8,
        probe_svd_energy=0.9, probe_basis_energy=0.999, num_classes=3,
        globa_with='D_minus')
    data = types.SimpleNamespace(train_loader=lambda: None, val_loader=lambda: None)
    built = []
    saved = (gm.merge, pt.build_model, ev.reset_bn_stats,
             ev.evaluate_all_classes, ev.classifier_name)
    gm.merge = lambda *a, **k: {}
    pt.build_model = lambda *a, **k: built.append(1) or torch.nn.Linear(2, 3)
    ev.reset_bn_stats = lambda *a, **k: None
    ev.evaluate_all_classes = lambda *a, **k: ([0.5, 0.5, 0.5], 0.5)
    ev.classifier_name = lambda m: 'fc'
    try:
        ev._collect_pair_children(info, models, None, args, data, 0,
                                  offspring, profiles)
    finally:
        (gm.merge, pt.build_model, ev.reset_bn_stats,
         ev.evaluate_all_classes, ev.classifier_name) = saved
    with open(ev._samples_paths(args)[0], newline='') as f:
        rows = list(csv.DictReader(f))
    first = rows[0]                       # (agent_000, agent_001): reused
    ok = (len(rows) == 6 and len(built) == 5
          and all(r[c] != '' for r in rows for c in ev.SAMPLE_COLUMNS
                  if c not in ('n_generations', 'location'))
          and float(first['child_val_overall']) == 0.77
          and float(first['base_val_overall']) == 0.5
          and float(first['donor_val_overall']) == 0.6
          and float(first['retention']) == round(1.35 / 1.8, 6)
          and float(first['retention_base']) == 1.0
          and float(first['retention_donor']) == 0.5)
    return ok, rows[:2]


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


def check_resume():
    """A run killed while saving gen_3: resume must pick gen 2, not gen 3."""
    import json
    from collections import defaultdict
    from datagen import evolution as ev
    from datagen import resume
    run = tempfile.mkdtemp()
    args = types.SimpleNamespace(run_dir=run)
    # gen_1 pruned (agent.json only); gen_3 half-written by the killed job.
    for g, files in ((0, ('agent.json', 'model.pt')), (1, ('agent.json',)),
                     (2, ('agent.json', 'model.pt')),
                     (3, ('agent.json', 'model.pt'))):
        for a in ('agent_000', 'agent_001'):
            d = os.path.join(run, 'checkpoints', f'gen_{g}', a)
            os.makedirs(d)
            for name in files:
                open(os.path.join(d, name), 'w').close()
    log = [{'stage': 'phase_a', 'budget': 14.0},
           {'stage': 'population', 'generation': 0, 'budget': 20.0},
           {'stage': 'society', 'generation': 0, 'budget': 50.0},
           {'stage': 'population', 'generation': 1, 'budget': 50.0},
           {'stage': 'society', 'generation': 1, 'budget': 80.0},
           {'stage': 'population', 'generation': 2, 'budget': 80.0},
           {'stage': 'mutation', 'generation': 2, 'budget': 81.5}]
    with open(os.path.join(run, 'train.jsonl'), 'w') as f:
        for r in log:
            f.write(json.dumps(r) + chr(10))
        f.write('{"stage": "mutation", "genera')          # cut by the kill
    with open(os.path.join(run, 'eval.jsonl'), 'w') as f:
        for b in (20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0):
            f.write(json.dumps({'budget': b, 'phase_start': 20.0}) + chr(10))
    part, fin = ev._samples_paths(args)
    for g in (0, 1, 2):
        ev._append_samples(part, [{'generation': g, 'share_A': 0.1,
                                   'share_B': 0.1, 'share_C': 0.1,
                                   'share_E': 0.1, 'share_D_plus': 0.1,
                                   'share_D_minus': 0.5,
                                   'child_val_overall': 0.2,
                                   'type': 'D_minus', 'head': 'label'}])
    with open(part, 'a') as f:
        f.write('1')                                     # truncated row

    bad = []
    state = resume.plan(args)
    if not isinstance(state, dict) or state['generation'] != 2 \
            or state['spent'] != 80.0:
        bad.append(f'plan: {state}')
    else:
        budget = types.SimpleNamespace(spent=0.0, by_stage=defaultdict(float))
        evaluator = types.SimpleNamespace(eval_interval=10.0, phase_start=0.0,
                                          next_watermark=0.0, n_points=0)
        logged = []
        logger = types.SimpleNamespace(log_train=logged.append)
        resume.apply(state, args, budget, evaluator, logger)
        with open(part, newline='') as f:
            gens = [r['generation'] for r in csv.DictReader(f)]
        if budget.spent != 80.0 or evaluator.next_watermark != 90.0 \
                or evaluator.phase_start != 20.0 or evaluator.n_points != 7:
            bad.append(f'restored budget/evaluator wrong: {budget.spent} '
                       f'{evaluator.next_watermark} {evaluator.phase_start}')
        if gens != ['0', '1']:
            bad.append(f'sample rows kept: {gens}')
        if not logged or logged[0]['stage'] != 'resume':
            bad.append('no resume record logged')
        if os.path.exists(os.path.join(run, 'checkpoints', 'gen_3')) or \
                not os.path.exists(os.path.join(run, 'checkpoints', 'gen_2')):
            bad.append('partial gen_3 not cleared (or gen_2 lost)')
    open(fin, 'w').close()
    if resume.plan(args) != 'done':
        bad.append('finished run not recognised')

    # Killed during pretrain: gen_0 has weights but was never evaluated.
    left = os.path.join(tempfile.mkdtemp(), 'seed0')
    os.makedirs(os.path.join(left, 'checkpoints', 'gen_0', 'agent_000'))
    early = types.SimpleNamespace(run_dir=left)
    if resume.plan(early) is not None:
        bad.append('pretrain cut short is not a fresh start')
    else:
        moved = resume.set_aside(early)
        if os.path.exists(left) or not moved or not os.path.isdir(moved):
            bad.append('leftovers of a fresh start not moved aside')
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
    ok4, info4 = check_retention()
    print(('PASS' if ok4 else 'FAIL') + '  retention scores match hand-computed '
          'values' + ('' if ok4 else f': {info4}'))
    ok5, info5 = check_collect()
    print(('PASS' if ok5 else 'FAIL') + '  every ordered pair gets a full row; '
          'bred children reused' + ('' if ok5 else f': {info5}'))
    ok2, bad = check_scores()
    print(('PASS' if ok2 else 'FAIL') + '  mating scores from profiles == '
          'globa_score_matrix' + ('' if ok2 else f' (differ for {bad})'))
    ok3, bad3 = check_resume()
    print(('PASS' if ok3 else 'FAIL') + '  --resume restarts at the last '
          'complete generation and restores its state'
          + ('' if ok3 else f': {bad3}'))
    removed, n = check_copy()
    # Lines of the original replaced or removed in the copy ([datagen] blocks,
    # and the mating modes / mergers the generator never runs).
    print(f'INFO  datagen/evolution.py replaces or removes {n} line(s) of '
          f'sesil/evolution.py')
    return 0 if ok1 and ok2 and ok3 and ok4 and ok5 else 1


if __name__ == '__main__':
    sys.exit(main())
