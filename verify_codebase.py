"""Check the codebase is in a consistent state.

    python verify_codebase.py            # all checks
    python verify_codebase.py --quiet    # only failures

Run it before starting a batch of edits and again after. Before, so a failure
afterwards is known to be yours rather than something you inherited; after, so
a broken state is found now instead of several hours into a run.

WHAT THIS CATCHES AND WHAT IT DOES NOT.

It catches syntax errors, broken imports, references to things that were
renamed or deleted, flag combinations that no longer resolve, and -- the one
that matters most for the blueprint -- defaults drifting away from the SESiL
reference. SESiL-II is defined as a set of flags on top of SESiL's defaults, so
a changed default silently redefines the baseline that every comparison is
measured against.

It does NOT catch an edit that never landed. A scripted find-and-replace whose
pattern did not match leaves the code importable, parseable and wrong, and it
reports success. That has happened here: a guard meant to cover --merger globa
was never widened, so the run spent its whole pretrain budget before failing.
The defence against that is to grep for the new text right after writing it,
not to run this.
"""

import argparse
import ast
import importlib
import os
import subprocess
import sys


# Defaults that define the SESiL reference. SESiL-II is these plus explicit
# flags, so the comparison is exact rather than approximate -- but only while
# these hold. Changing one silently moves the baseline.
REFERENCE_DEFAULTS = {
    'pretrain_mode': 'none',
    'subset_mode': 'random',
    'phase_b_freeze': 0.0,
    'mating_mode': 'certificate',
    'cert_with': 'count',
    'weight_extra': 1.0,
    'weight_common': 0.1,
    'merger': 'permute',
    'merge_head': 'average',
    'stop_node': None,
    'merge_bias': 0.5,
    'certify_top_frac': 0.3,
    'certify_floor': 0.5,
}

# Numeric defaults that size the main experiment. Separate from
# REFERENCE_DEFAULTS above because these do NOT define the SESiL method -- they
# define the run: how big the society is, how much compute it gets, how often it
# is measured. Both arms of every comparison inherit them, so a drift here does
# not bias SESiL against the baseline. It does silently change what a figure
# means, and it makes a command line from an earlier session non-reproducible,
# which is why they are pinned.
EXPERIMENT_DEFAULTS = {
    # Moved here from REFERENCE_DEFAULTS once round 1 showed what it is: not a
    # method-defining mode but a tuned knob, and the most sensitive one in the
    # setup. It sets how many agents pair, which sets how many are left over as
    # loners, and loners are the only source of exploration -- so it sets the
    # size of the society space, which is what the whole-space curve tracks.
    # Measured on 3 seeds x 38 generations: 5 rounds -> 27% of agents paired and
    # 13.6 new classes explored per generation; 15 rounds -> 83% paired and 3.3
    # explored. Set to 15 to stay aligned with the mr15 runs.
    'mating_rounds': 15,
    'keep_generations': 2,
    'pop_size': 20,
    'classes_per_model': 3,
    'society_classes': 40,
    'pretrain_budget': 20.0,
    'individual_budget': 0.5,
    'budget': 400,
    'eval_interval': 10.0,
}

MODULES = [
    'config', 'main', 'analyze_probe', 'backfill_layers', 'plot_results',
    'sesil.baseline', 'sesil.budget', 'sesil.certificate', 'sesil.data',
    'sesil.evaluator', 'sesil.evolution', 'sesil.fitness', 'sesil.globa_merge',
    'sesil.globa_stats', 'sesil.gpu', 'sesil.logging', 'sesil.merge',
    'sesil.mutation', 'sesil.population', 'sesil.pretrain', 'sesil.registry',
    'sesil.selection', 'sesil.ssl', 'sesil.probe.driver', 'sesil.probe.retention',
]

# Names that were removed or moved. A hit means something still refers to the
# old thing, which will fail only when that code path runs.
RETIRED = {
    'rank_strength': 'removed from certificate.py',
    'args.mate_score': 'renamed to args.cert_with',
    'args.max_retries': 'renamed to args.mating_rounds',
    'sesil.probe.globa_stats': 'moved to sesil.globa_stats',
    'sesil.probe.globa_merge': 'moved to sesil.globa_merge',
}

# Combinations that must resolve. Anything a run might plausibly use.
COMBINATIONS = [
    ['--method', 'sesil'],
    ['--method', 'baseline'],
    ['--method', 'probe', '--pretrain-mode', 'ssl'],
    ['--method', 'sesil', '--pretrain-mode', 'ssl'],
    ['--method', 'sesil', '--mating-mode', 'random'],
    ['--method', 'sesil', '--mating-mode', 'globa', '--pretrain-mode', 'ssl'],
    ['--method', 'sesil', '--merger', 'globa', '--pretrain-mode', 'ssl'],
    ['--method', 'sesil', '--merger', 'globa', '--mating-mode', 'globa',
     '--pretrain-mode', 'ssl', '--merge-head', 'label'],
    ['--method', 'probe', '--pretrain-mode', 'ssl', '--probe-merger', 'both'],
    ['--method', 'probe', '--pretrain-mode', 'ssl', '--probe-merger', 'globa',
     '--globa-head', 'average'],
]

# Settings that must be refused, with the substring the message should contain.
MUST_REFUSE = [
    (['--merger', 'globa', '--pretrain-mode', 'none'], 'pretrain-mode ssl'),
    (['--mating-mode', 'globa', '--pretrain-mode', 'none'], 'pretrain-mode ssl'),
    (['--method', 'probe', '--probe-merger', 'globa', '--pretrain-mode', 'none'],
     'pretrain-mode ssl'),
    (['--method', 'baseline', '--baseline-mode', 'curriculum'], '--curriculum-from'),
    (['--method', 'baseline', '--curriculum-from', 'results/x'],
     'without --baseline-mode curriculum'),
    (['--keep-generations', '1'], 'smallest retention window'),
]


def source_files():
    """Project sources, skipping vendored trees we did not write."""
    skip = {'__pycache__', '.git', 'data', 'results', 'datasets', 'models', 'graphs'}
    for root, dirs, files in os.walk('.'):
        dirs[:] = [d for d in dirs if d not in skip and not d.startswith('.')]
        for name in files:
            if name.endswith('.py'):
                yield os.path.join(root, name)


def check_parses(report):
    bad = []
    for path in source_files():
        try:
            with open(path, encoding='utf-8') as f:
                ast.parse(f.read())
        except SyntaxError as e:
            bad.append(f'{path}:{e.lineno}: {e.msg}')
    report('every file parses', not bad, bad)


def check_imports(report):
    bad = []
    for name in MODULES:
        try:
            importlib.import_module(name)
        except Exception as e:                       # noqa: BLE001
            bad.append(f'{name}: {type(e).__name__}: {e}')
    report(f'{len(MODULES)} modules import', not bad, bad)


def check_retired(report):
    bad = []
    for symbol, note in RETIRED.items():
        try:
            out = subprocess.run(
                ['grep', '-rn', '--include=*.py', symbol, '.'],
                capture_output=True, text=True, timeout=60).stdout
        except Exception:                            # noqa: BLE001
            continue
        hits = [ln for ln in out.splitlines()
                if '__pycache__' not in ln
                and 'verify_codebase.py' not in ln
                and 'renamed to' not in ln
                and 'moved to' not in ln
                and 'Replaces' not in ln
                and 'removed' not in ln]
        if hits:
            bad.append(f'{symbol} ({note}): ' + '; '.join(hits[:3]))
    report('no references to retired names', not bad, bad)


def _drift(args, expected):
    return [f'--{k.replace("_", "-")}: expected {v!r}, found {getattr(args, k)!r}'
            for k, v in expected.items() if getattr(args, k) != v]


def check_reference_defaults(report):
    from config import get_config
    args = get_config().parse_args(['--device', 'cpu'])
    report('defaults still reproduce the SESiL reference',
           not _drift(args, REFERENCE_DEFAULTS), _drift(args, REFERENCE_DEFAULTS),
           note='SESiL-II is defined as flags on top of these; a changed '
                'default silently moves the baseline')


def check_experiment_defaults(report):
    """The run's scale, pinned so a command line stays reproducible.

    run.sh supplies --budget itself, so config's default is not what a sweep
    actually uses; both are checked, and they have to agree.
    """
    from config import get_config
    args = get_config().parse_args(['--device', 'cpu'])
    drift = _drift(args, EXPERIMENT_DEFAULTS)

    want = EXPERIMENT_DEFAULTS['budget']
    launcher = None
    if os.path.exists('run.sh'):
        with open('run.sh', encoding='utf-8') as f:
            for line in f:
                if line.startswith('BUDGET='):
                    launcher = line.split('=', 1)[1].strip()
                    break
    if launcher is None:
        drift.append('run.sh has no BUDGET= line to check')
    elif float(launcher) != float(want):
        drift.append(f'run.sh BUDGET={launcher} but config default is {want} '
                     f'-- run.sh always passes --budget, so its value is the '
                     f'one a sweep gets')

    report('experiment-scale defaults are pinned', not drift, drift,
           note='both arms inherit these, so drift does not bias the '
                'comparison -- it changes what the figure means')


def check_combinations(report):
    from config import get_config, validate
    bad = []
    for combo in COMBINATIONS:
        try:
            validate(get_config().parse_args(combo + ['--device', 'cpu']))
        except SystemExit as e:
            bad.append(' '.join(combo) + f' -> refused: {str(e).splitlines()[0]}')
    report(f'{len(COMBINATIONS)} valid combinations resolve', not bad, bad)


def check_refusals(report):
    from config import get_config, validate
    bad = []
    for combo, expect in MUST_REFUSE:
        try:
            validate(get_config().parse_args(combo + ['--device', 'cpu']))
            bad.append(' '.join(combo) + ' -> was NOT refused')
        except SystemExit as e:
            if expect not in str(e):
                bad.append(' '.join(combo) + f' -> refused, but message lacks {expect!r}')
    report(f'{len(MUST_REFUSE)} conflicting combinations are refused', not bad, bad)


def check_second_guard(report):
    """The evolution-level backbone guard, checked separately from config's.

    Two guards cover the same conflict: config.validate refuses it before
    anything runs, and _load_globa_core refuses it again inside the run. That
    redundancy is deliberate -- validate can only see the parsed flags, while
    _load_globa_core is where the file is actually needed -- but it means
    either can stop working without any behaviour changing, so neither is
    covered by testing the other.
    """
    from config import get_config
    from sesil.evolution import _load_globa_core

    bad = []
    for flags, expect in (
        (['--merger', 'globa'], '--merger globa'),
        (['--mating-mode', 'globa'], '--mating-mode globa'),
    ):
        args = get_config().parse_args(
            flags + ['--pretrain-mode', 'none', '--device', 'cpu'])
        args.arch = 'resnet20x4'
        try:
            _load_globa_core(args)
            bad.append(' '.join(flags) + ' -> _load_globa_core did NOT refuse')
        except SystemExit as e:
            if expect not in str(e):
                bad.append(f'{" ".join(flags)} -> refused but did not name {expect!r}')

    # And it must stay quiet when nothing needs a backbone.
    args = get_config().parse_args(['--pretrain-mode', 'none', '--device', 'cpu'])
    args.arch = 'resnet20x4'
    try:
        if _load_globa_core(args) is not None:
            bad.append('non-GLOBA settings loaded a backbone anyway')
    except SystemExit as e:
        bad.append(f'non-GLOBA settings were refused: {str(e).splitlines()[0]}')

    report('evolution-level backbone guard works independently', not bad, bad)


def check_curriculum_seed_pairing(report):
    """--curriculum-from must resolve to THIS seed's SESiL run, or refuse.

    Checked because the failure is silent in every other way: run.sh passes
    unrecognised flags through to all seeds unchanged, so one fixed seed
    directory makes every baseline replay the same curriculum and write
    identical curves under different seed labels. The plot then shows a
    baseline with implausibly tight error bars, which reads as a result.
    """
    import json as _json
    import shutil
    import tempfile
    from config import get_config, resolve_curriculum_from

    root = tempfile.mkdtemp(prefix='verify_curric_')
    bad = []
    try:
        # A parent holding two seed directories, each recording its own seed.
        for seed in (0, 1):
            d = os.path.join(root, f'seed{seed}')
            os.makedirs(d)
            with open(os.path.join(d, 'train.jsonl'), 'w') as f:
                f.write(_json.dumps({'stage': 'society', 'budget': 1.0,
                                     'space': [0, 1]}) + chr(10))
            with open(os.path.join(d, 'config.json'), 'w') as f:
                _json.dump({'config': {'seed': seed}}, f)

        def parse(extra):
            return get_config().parse_args(
                ['--method', 'baseline', '--baseline-mode', 'curriculum',
                 '--device', 'cpu'] + extra)

        # Parent form: each seed must land on its own directory.
        for seed in (0, 1):
            args = parse(['--curriculum-from', root, '--seed', str(seed)])
            got = resolve_curriculum_from(args)
            if os.path.basename(got) != f'seed{seed}':
                bad.append(f'seed {seed} resolved to {got!r}, wanted seed{seed}')

        # Seed directory form, matching: allowed.
        args = parse(['--curriculum-from', os.path.join(root, 'seed0'),
                      '--seed', '0'])
        try:
            resolve_curriculum_from(args)
        except SystemExit as e:
            bad.append(f'matching seed dir was refused: {str(e).splitlines()[0]}')

        # Seed directory form, MISmatching: this is the one that must refuse.
        args = parse(['--curriculum-from', os.path.join(root, 'seed0'),
                      '--seed', '1'])
        try:
            resolve_curriculum_from(args)
            bad.append('seed1 baseline replaying seed0 curriculum was NOT refused')
        except SystemExit as e:
            if 'recorded seed 0' not in str(e):
                bad.append(f'refused but did not name the mismatch: '
                           f'{str(e).splitlines()[0]}')

        # A parent with no matching seed: refuse, and say what is there.
        args = parse(['--curriculum-from', root, '--seed', '7'])
        try:
            resolve_curriculum_from(args)
            bad.append('missing seed7 was NOT refused')
        except SystemExit as e:
            if 'seed0' not in str(e):
                bad.append('refusal does not list the available seed dirs')
    finally:
        shutil.rmtree(root, ignore_errors=True)

    report('curriculum baseline pairs each seed with its own SESiL run', not bad, bad,
           note='an unpaired baseline replays one seed under every seed label')


def check_generation_pruning(report):
    """--keep-generations must never delete a generation the loop still needs.

    Worth a behavioural test rather than a config check: this is the only code
    in the tree that deletes results, and getting the window off by one would
    remove gen_N while the loop is reading it -- a failure that shows up hours
    into a run, as a missing population, not as a wrong number.
    """
    import shutil
    import tempfile
    import types
    from sesil.population import prune_generations

    root = tempfile.mkdtemp(prefix='verify_prune_')
    bad = []
    try:
        def fresh(n):
            ck = os.path.join(root, 'checkpoints')
            shutil.rmtree(ck, ignore_errors=True)
            for g in range(n):
                for a in range(2):
                    d = os.path.join(ck, f'gen_{g}', f'agent_{a:03}')
                    os.makedirs(d)
                    open(os.path.join(d, 'resnet_v0.pth.tar'), 'w').close()
                    open(os.path.join(d, 'agent.json'), 'w').close()
            return types.SimpleNamespace(run_dir=root)

        def left():
            """Generations whose WEIGHTS survive -- what the loop can load."""
            ck = os.path.join(root, 'checkpoints')
            out = []
            for d in os.listdir(ck):
                if not d.startswith('gen_'):
                    continue
                for a in os.listdir(os.path.join(ck, d)):
                    if any(f.endswith('.pth.tar')
                           for f in os.listdir(os.path.join(ck, d, a))):
                        out.append(int(d[4:]))
                        break
            return sorted(out)

        def meta_kept():
            """Generations whose agent.json survives -- provenance, kept always."""
            ck = os.path.join(root, 'checkpoints')
            out = []
            for d in os.listdir(ck):
                if not d.startswith('gen_'):
                    continue
                for a in os.listdir(os.path.join(ck, d)):
                    if 'agent.json' in os.listdir(os.path.join(ck, d, a)):
                        out.append(int(d[4:]))
                        break
            return sorted(out)

        # Loop has just written gen_10; keep 2 -> gen_9 and gen_10 survive,
        # gen_0 survives because it is protected.
        args = fresh(11)
        prune_generations(args, newest=10, keep=2)
        if left() != [0, 9, 10]:
            bad.append(f'keep=2 after gen_10 left {left()}, wanted [0, 9, 10]')

        # The pair the loop holds simultaneously must both survive.
        args = fresh(11)
        prune_generations(args, newest=10, keep=2)
        for needed in (9, 10):
            if needed not in left():
                bad.append(f'keep=2 deleted gen_{needed}, which the loop still reads')

        # Wider window.
        args = fresh(11)
        prune_generations(args, newest=10, keep=4)
        if left() != [0, 7, 8, 9, 10]:
            bad.append(f'keep=4 left {left()}, wanted [0, 7, 8, 9, 10]')

        # 0 means keep everything.
        args = fresh(11)
        if prune_generations(args, newest=10, keep=0):
            bad.append('keep=0 deleted something')
        if left() != list(range(11)):
            bad.append(f'keep=0 left {left()}, wanted all 11')

        # Early generations: nothing to prune yet, and gen_0 must survive.
        args = fresh(2)
        prune_generations(args, newest=1, keep=2)
        if left() != [0, 1]:
            bad.append(f'at gen_1 with keep=2, left {left()}, wanted [0, 1]')

        # Idempotent -- running twice removes nothing more.
        args = fresh(11)
        prune_generations(args, newest=10, keep=2)
        if prune_generations(args, newest=10, keep=2):
            bad.append('second prune removed more directories')

        # Provenance survives pruning. agent.json is 454 bytes against 18 MB of
        # weights, and it is the only record tying an agent's parents to its own
        # id -- train.jsonl logs a couple's parents but not which agent_NNN the
        # child became.
        args = fresh(11)
        prune_generations(args, newest=10, keep=2)
        if meta_kept() != list(range(11)):
            bad.append(f'pruning discarded agent.json: kept {meta_kept()}, '
                       f'wanted all 11 generations')
    finally:
        shutil.rmtree(root, ignore_errors=True)

    report('generation pruning keeps what the loop still reads', not bad, bad,
           note='this is the only code that deletes results')


def check_series_fields_are_logged(report):
    """Every field plot_results groups by must actually be written to eval.jsonl.

    This is the quietest failure mode in the project. plot_results separates
    arms using the meta fields carried on each eval record; a field it consults
    but main.py never logs reads back as None for every run, so two different
    conditions get the same series name, their seeds are pooled, and six runs
    of two conditions are averaged into one curve. Nothing errors. The only
    hint is a note about differing budget grids.

    It happened: --individual-budget and --certify-top-frac defined arms and
    were not logged, so the ib050 and ib100 rounds were indistinguishable in a
    combined plot.
    """
    import ast

    bad = []
    try:
        import plot_results
        wanted = set(plot_results.SESIL_REFERENCE)
    except Exception as e:                               # noqa: BLE001
        report('series-defining fields are all logged', False,
               [f'could not import plot_results: {e}'])
        return

    # Read main.py's meta dict literally rather than running it: constructing an
    # Evaluator needs a dataset and a device.
    with open('main.py', encoding='utf-8') as f:
        tree = ast.parse(f.read())

    logged = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.keyword) and node.arg == 'meta':
            if isinstance(node.value, ast.Dict):
                for k in node.value.keys:
                    if isinstance(k, ast.Constant) and isinstance(k.value, str):
                        logged.add(k.value)

    if not logged:
        bad.append('could not find the meta={...} dict in main.py')
    else:
        for field in sorted(wanted - logged):
            bad.append(f'plot_results groups by {field!r} but main.py never '
                       f'logs it -- arms differing only in it would be merged')

    report('series-defining fields are all logged', not bad, bad,
           note='a field consulted but not logged silently averages '
                'different conditions together')


def check_lazy_globa_matches_eager(report):
    """Lazy GLOBA scoring must choose exactly what eager scoring chooses.

    Hybrid mating consults GLOBA only to break ties, so scoring all 190 pairs
    of a 20-agent population up front spent hours per run on numbers nothing
    read. The scorer now decomposes a pair when asked. That is only safe if it
    is invisible: same couples, same loners. A score nobody reads cannot change
    a decision -- but an indexing bug in the lazy row would change plenty, and
    would show up as a quietly different population rather than an error.

    Uses a stub decomposition so the check needs no backbone and no GPU: what
    is under test is the plumbing, not the linear algebra.
    """
    import types
    from sesil import selection

    bad = []
    ids = [f'agent_{i:03}' for i in range(8)]

    # Deterministic stand-in for directional_pair_stats, keyed by the pair, so
    # eager and lazy must see identical numbers.
    def fake_stats(sa, sb, core, **kw):
        h = (hash(sa) ^ (hash(sb) * 31)) % 1000 / 1000.0
        return ({'D_minus': h}, {'D_minus': 1.0 - h})

    import sesil.globa_stats as gs
    real = gs.directional_pair_stats
    gs.directional_pair_stats = fake_stats
    try:
        args = types.SimpleNamespace(
            probe_eta=0.8, probe_svd_energy=0.9, probe_basis_energy=0.999,
            head_prefix='fc.', probe_layer_weighting='none',
            globa_with='D_minus', mating_mode='hybrid', cert_with='count',
            weight_extra=1.0, weight_common=0.1, mating_rounds=5,
            hybrid_tolerance=0.15)
        states = {a: a for a in ids}

        eager = selection.globa_score_matrix(ids, states, None, args)
        lazy = selection.LazyGlobaScores(ids, states, None, args)

        # Every entry must agree, whichever way it is reached.
        for a in ids:
            for b in ids:
                if a == b:
                    continue
                if abs(eager[a][b] - lazy[a].get(b)) > 1e-12:
                    bad.append(f'lazy[{a}][{b}] != eager: '
                               f'{lazy[a].get(b)!r} vs {eager[a][b]!r}')
                    break

        # And the decision they drive must be identical.
        pop = [{'Model Name': a, 'Certificate': {i, (i + 1) % 10, (i + 2) % 10},
                'Strength': [1.0] * 10} for i, a in enumerate(ids)]
        import random
        for trial in range(5):
            random.seed(trial)
            pe, le_ = selection.select_mates(pop, args, tiebreak=eager)
            random.seed(trial)
            pl, ll = selection.select_mates(pop, args,
                                            tiebreak=selection.LazyGlobaScores(
                                                ids, states, None, args))
            if pe != pl or le_ != ll:
                bad.append(f'trial {trial}: eager chose {pe}/{le_}, '
                           f'lazy chose {pl}/{ll}')

        # And it must actually be lazy -- otherwise the fix does nothing.
        counter = selection.LazyGlobaScores(ids, states, None, args)
        selection.select_mates(pop, args, tiebreak=counter)
        total = len(ids) * (len(ids) - 1) // 2
        if counter.computed >= total:
            bad.append(f'lazy scorer computed {counter.computed} of {total} '
                       f'pairs -- no work was saved')
    finally:
        gs.directional_pair_stats = real

    report('lazy GLOBA scoring picks what eager scoring picks', not bad, bad,
           note='hybrid must be unchanged by the speedup')


def check_mating_inputs(report):
    """Only 'globa' may have the GLOBA matrix as its mating SCORE.

    This is a regression test for a bug that silently changed what an arm was.
    The phase-A backbone is loaded when mating OR merging needs it, so
    --merger globa --mating-mode certificate loaded it for merging, built the
    pair-score matrix, and passed it to select_mates as `scores`. select_mates
    builds the certificate matrix only when `scores is None`, so certificate
    scoring never ran: the population mated on GLOBA D_minus while every log
    and config recorded 'certificate'. Measured on the affected run, 6% of
    couples were top-tier by certificate against ~100% for real certificate
    mating.

    Nothing about that failure was visible without recomputing the scores, so
    it is pinned here rather than left to inspection.
    """
    import types
    from sesil.evolution import mating_inputs, mating_needs_globa

    bad = []
    matrix = {'sentinel': True}

    expected = {
        # mode          -> (scores, tiebreak), None meaning "build your own"
        'certificate': (None, None),
        'random':      (None, None),
        'globa':       (matrix, None),
        'hybrid':      (None, matrix),
    }
    for mode, want in expected.items():
        got = mating_inputs(types.SimpleNamespace(mating_mode=mode), matrix)
        if got != want:
            bad.append(f'{mode}: got {got!r}, wanted {want!r}')

    # And the matrix must not even be BUILT for the modes that cannot use it,
    # which is what made the affected run slow as well as wrong.
    for mode, want in (('certificate', False), ('random', False),
                       ('globa', True), ('hybrid', True)):
        got = mating_needs_globa(types.SimpleNamespace(mating_mode=mode))
        if got != want:
            bad.append(f'mating_needs_globa({mode}) = {got}, wanted {want}')

    report('only globa/hybrid consume the GLOBA pair matrix', not bad, bad,
           note='a matrix handed to the wrong slot replaces the scoring rule '
                'without any error')


def check_val_routed_metric(report):
    """Routing must select on validation and read accuracy from test.

    The obvious bug is selecting and reading from the same matrix, which just
    reproduces oracle_overall and would look entirely plausible in a plot --
    a slightly-too-good curve is not something anyone spots by eye. So the
    check feeds a case where the two matrices disagree on purpose and pins the
    exact expected value.
    """
    from sesil.evaluator import Evaluator

    bad = []
    ev = Evaluator.__new__(Evaluator)          # _reduce needs no I/O state

    # Two agents, three classes. Validation says agent 0 is best on class 0 and
    # agent 1 on classes 1 and 2 -- but on TEST those picks score poorly. A
    # router that peeked at test would score 0.9; one that honestly follows
    # validation scores 0.2.
    test = [[0.1, 0.9, 0.9],
            [0.9, 0.2, 0.2]]
    val = [[0.9, 0.1, 0.1],
           [0.1, 0.9, 0.9]]
    out = ev._reduce(test, [0.6, 0.4], val)

    want_routed = (0.1 + 0.2 + 0.2) / 3
    if abs(out['val_routed_overall'] - want_routed) > 1e-9:
        bad.append(f'val_routed_overall = {out["val_routed_overall"]:.4f}, '
                   f'wanted {want_routed:.4f} -- it is not following validation')
    if abs(out['oracle_overall'] - 0.9) > 1e-9:
        bad.append(f'oracle_overall = {out["oracle_overall"]:.4f}, wanted 0.9')
    if abs(out['val_routed_overall'] - out['oracle_overall']) < 1e-9:
        bad.append('routed equals oracle -- selection is reading the test matrix')

    # The single-agent pick: validation prefers agent 1 (mean 0.633 vs 0.367),
    # whose TEST mean is 0.4333.
    want_single = (0.9 + 0.2 + 0.2) / 3
    if abs(out['val_best_agent_overall'] - want_single) > 1e-9:
        bad.append(f'val_best_agent_overall = {out["val_best_agent_overall"]:.4f}, '
                   f'wanted {want_single:.4f}')

    # One model, no validation supplied: the baseline. Routing is vacuous and
    # must still be filled, or the two methods cannot share a plot axis.
    solo = ev._reduce([[0.3, 0.5]], [0.4], None)
    if 'val_routed_overall' not in solo:
        bad.append('baseline (1 model) got no routed metric -- it cannot be '
                   'plotted against sesil on a routed axis')
    elif abs(solo['val_routed_overall'] - 0.4) > 1e-9:
        bad.append(f'baseline routed = {solo["val_routed_overall"]:.4f}, wanted 0.4')

    # No validation and several models: the fields must be ABSENT, not zero.
    older = ev._reduce(test, [0.6, 0.4], None)
    if 'val_routed_overall' in older:
        bad.append('routed metric invented without a validation matrix')

    report('val-routed metric selects on validation, reads on test', not bad, bad,
           note='selecting on test is leakage and looks fine in a plot')


def check_val_routed_is_wired(report):
    """The routed metric must actually reach the evaluator on every eval.

    check_val_routed_metric proves the arithmetic. This proves the plumbing,
    which is the part a find-and-replace can silently fail to land: if
    evolution stops passing the validation matrix, _reduce quietly omits the
    routed fields and every run from then on produces a log that plots as an
    empty series. Nothing errors, and the gap is only noticed when someone
    tries to draw the figure.
    """
    import ast
    import inspect
    from sesil.evaluator import Evaluator

    bad = []

    sig = inspect.signature(Evaluator.record)
    if 'val_per_class' not in sig.parameters:
        bad.append('Evaluator.record has no val_per_class parameter')

    with open(os.path.join('sesil', 'evolution.py'), encoding='utf-8') as f:
        tree = ast.parse(f.read())

    passes = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        fn = node.func
        name = getattr(fn, 'attr', None)
        if name not in ('record', 'maybe_record'):
            continue
        passes.append((name, any(k.arg == 'val_per_class' for k in node.keywords)))

    if not passes:
        bad.append('no evaluator record call found in evolution.py')
    for name, has in passes:
        if not has:
            bad.append(f'evolution.py calls {name}() without val_per_class -- '
                       f'that eval point will have no routed metric')

    report('val-routed metric is wired into every evaluation', not bad, bad,
           note='an unwired metric logs nothing and plots as an empty series')


def check_ensemble_metrics(report):
    """Ensemble rules must combine the population, not echo one agent.

    Pinned against a hand-built case because every rule here returns a
    plausible-looking accuracy whatever it computes; a wrong one reads as a
    slightly disappointing result rather than a bug.

    The fixture isolates the normalisation, which on real populations was
    worth more than every other choice in this function combined. One sample,
    true class 1. Two agents are certified for class 0 and split evenly; one
    agent is certified for class 1 and mildly prefers it:

        agent 0  cert {0}   [0.5, 0.5]
        agent 1  cert {0}   [0.5, 0.5]
        agent 2  cert {1}   [0.4, 0.6]

    Summing over certified agents gives class 0 a total of 1.0 against class
    1's 0.6 -- class 0 wins purely because two agents hold it. Dividing by the
    number of holders gives 0.5 against 0.6 and the right answer. So
    ensemble_soft_certified must be WRONG here and
    ensemble_soft_certified_norm must be RIGHT, which no implementation that
    forgets to divide can satisfy.
    """
    import numpy as np
    from sesil.evaluator import Evaluator

    ev = Evaluator.__new__(Evaluator)
    bad = []

    probs = np.array([[[0.5, 0.5]], [[0.5, 0.5]], [[0.4, 0.6]]], dtype=np.float32)
    labels = np.array([1])
    certs = [[0], [0], [1]]

    out = ev.ensemble(probs, labels, certs)

    expected = {
        'ensemble_hard': 0.0,              # two votes for class 0, one for 1
        'ensemble_conf_weighted': 0.0,     # 1.0 against 0.6, still class 0
        'ensemble_soft': 1.0,              # means 0.467 against 0.533
        'ensemble_max_confidence': 1.0,    # agent 2 is the most confident
        'ensemble_soft_certified': 0.0,    # 1.0 against 0.6 -- headcount wins
        'ensemble_soft_certified_norm': 1.0,   # 0.5 against 0.6 -- corrected
    }
    for key, want in expected.items():
        if key not in out:
            bad.append(f'{key} missing')
        elif abs(out[key] - want) > 1e-9:
            bad.append(f'{key} = {out[key]:.4f}, wanted {want:.4f}')

    # Without certificates the gated keys must be ABSENT, not zero: a zero
    # plots as a real curve lying on the axis.
    plain = ev.ensemble(probs, labels, None)
    for key in ('ensemble_soft_certified', 'ensemble_soft_certified_norm'):
        if key in plain:
            bad.append(f'{key} produced without any certificates')
    for key in ('ensemble_hard', 'ensemble_soft'):
        if key not in plain:
            bad.append(f'{key} missing when certificates are not supplied')

    report('ensemble rules combine the population correctly', not bad, bad,
           note='a wrong rule still returns a plausible accuracy')


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--quiet', action='store_true', help='Print failures only.')
    args = ap.parse_args()

    sys.path.insert(0, os.getcwd())
    failures = []

    def report(name, ok, details, note=None):
        if ok:
            if not args.quiet:
                print(f'  PASS  {name}')
        else:
            failures.append(name)
            print(f'  FAIL  {name}')
            if note:
                print(f'          ({note})')
            for line in details[:8]:
                print(f'          {line}')
            if len(details) > 8:
                print(f'          ... and {len(details) - 8} more')

    if not args.quiet:
        print('Verifying codebase\n')

    # Ordered so a failure explains the ones after it: nothing else can work if
    # the files do not parse.
    check_parses(report)
    check_imports(report)
    check_retired(report)
    check_reference_defaults(report)
    check_experiment_defaults(report)
    check_combinations(report)
    check_refusals(report)
    check_second_guard(report)
    check_curriculum_seed_pairing(report)
    check_generation_pruning(report)
    check_series_fields_are_logged(report)
    check_lazy_globa_matches_eager(report)
    check_mating_inputs(report)
    check_val_routed_metric(report)
    check_val_routed_is_wired(report)
    check_ensemble_metrics(report)

    print()
    if failures:
        print(f'{len(failures)} check(s) FAILED: ' + ', '.join(failures))
        return 1
    print('All checks passed.')
    print('Note: this cannot tell you an edit never landed -- a find-and-replace '
          'that did not match\nleaves the code valid and wrong. Grep for the new '
          'text right after writing it.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
