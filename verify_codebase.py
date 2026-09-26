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
    'mating_rounds': 5,
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
