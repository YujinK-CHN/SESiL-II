"""
Plot runs on the shared budget axis.

Reads every eval.jsonl under a results directory, groups runs into series,
aggregates across seeds, and draws them against spent budget -- the common
currency that makes SESiL and the baseline comparable.

    python plot_results.py results/check/cifar10
    python plot_results.py results/check/cifar10 --metric oracle_overall
    python plot_results.py results/check --band minmax --table --out fig.png

Each series is one method/configuration; each seed within it is one run. The
vertical dashed line is the phase boundary, read from `phase_start` in the log
itself: for SESiL that is the pretrain cost, so its curve begins there rather
than at zero, and the horizontal distance between curves past that line is a
like-for-like compute comparison.

--list-metrics prints what a given results tree actually contains.
"""

import argparse
import json
import os
from collections import defaultdict

import numpy as np


# --------------------------------------------------------------------------- #
# Palette. Categorical slots assigned in fixed order by series name, so a
# series keeps its colour when another is filtered out. Both modes validated
# against their own surface.
# --------------------------------------------------------------------------- #
THEME = {
    'light': {
        'surface': '#fcfcfb',
        'ink': '#0b0b0b',
        'ink_2': '#52514e',
        'muted': '#8a8985',
        'grid': '#e1e0d9',
        'series': ['#2a78d6', '#eb6834', '#1baf7a', '#eda100',
                   '#e87ba4', '#008300', '#4a3aa7', '#e34948'],
    },
    'dark': {
        'surface': '#1a1a19',
        'ink': '#ffffff',
        'ink_2': '#c3c2b7',
        'muted': '#8a8985',
        'grid': '#2c2c2a',
        'series': ['#3987e5', '#d95926', '#199e70', '#c98500',
                   '#d55181', '#008300', '#9085e9', '#e66767'],
    },
}

DEFAULT_METRIC = 'best_agent_overall'

METRIC_LABELS = {
    'best_agent_overall': 'Test accuracy (best agent)',
    'best_agent_balanced': 'Test accuracy (best agent, class-balanced)',
    'population_mean': 'Test accuracy (population mean)',
    'oracle_overall': 'Test accuracy (per-class oracle)',
    'population_best': 'Test accuracy (population best)',
    'population_worst': 'Test accuracy (population worst)',
    'population_median': 'Test accuracy (population median)',
}


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #

def find_runs(root):
    """Every directory under `root` holding an eval.jsonl."""
    runs = []
    for dirpath, _dirnames, filenames in os.walk(root):
        if 'eval.jsonl' in filenames:
            runs.append(dirpath)
    return sorted(runs)


def load_run(run_dir):
    """Records from one run, plus how it should be labelled and grouped."""
    records = []
    with open(os.path.join(run_dir, 'eval.jsonl'), encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    if not records:
        return None

    head = records[0]
    method = head.get('method', 'unknown')
    merger = head.get('merger')
    series = f'{method} ({merger})' if method == 'sesil' and merger else method

    return {
        'dir': run_dir,
        'series': series,
        'seed': head.get('seed'),
        'phase_start': float(head.get('phase_start', 0.0) or 0.0),
        'records': records,
    }


def group_runs(runs):
    """series name -> list of runs (one per seed)."""
    grouped = defaultdict(list)
    for run in runs:
        grouped[run['series']].append(run)
    return dict(sorted(grouped.items()))


# --------------------------------------------------------------------------- #
# Aggregation across seeds
# --------------------------------------------------------------------------- #

def aggregate(runs, metric, band):
    """Collapse seeds into (x, centre, lo, hi).

    Seeds of one configuration normally share identical x-values, because
    pretrain and per-generation costs are deterministic in budget terms. When
    they do not -- different --pop-size, a resumed run -- each seed is
    interpolated onto the shared grid over the overlapping range, and the
    caller is told.
    """
    curves = []
    for run in runs:
        xs = np.array([r['budget'] for r in run['records']], dtype=float)
        ys = np.array([r[metric] for r in run['records']], dtype=float)
        order = np.argsort(xs)
        curves.append((xs[order], ys[order]))

    x_sets = {tuple(np.round(xs, 6)) for xs, _ in curves}
    interpolated = len(x_sets) > 1

    if interpolated:
        lo_x = max(xs[0] for xs, _ in curves)
        hi_x = min(xs[-1] for xs, _ in curves)
        n = max(len(xs) for xs, _ in curves)
        grid = np.linspace(lo_x, hi_x, n)
        stack = np.vstack([np.interp(grid, xs, ys) for xs, ys in curves])
    else:
        grid = curves[0][0]
        stack = np.vstack([ys for _, ys in curves])

    centre = stack.mean(axis=0)

    if stack.shape[0] == 1 or band == 'none':
        lo = hi = centre
    elif band == 'minmax':
        lo, hi = stack.min(axis=0), stack.max(axis=0)
    elif band == 'iqr':
        lo, hi = np.percentile(stack, 25, axis=0), np.percentile(stack, 75, axis=0)
    else:                                   # std
        sd = stack.std(axis=0)
        lo, hi = centre - sd, centre + sd

    return grid, centre, lo, hi, stack.shape[0], interpolated


# --------------------------------------------------------------------------- #
# Output
# --------------------------------------------------------------------------- #

def print_table(series_data, metric, band):
    """Text alternative to the figure -- the numbers behind every point."""
    print(f'\n{metric}   (centre = mean across seeds, band = {band})\n')
    for name, (x, centre, lo, hi, n_seeds, _interp) in series_data.items():
        print(f'{name}   [{n_seeds} seed(s)]')
        print(f'  {"budget":>9}  {"value":>8}  {"lo":>8}  {"hi":>8}')
        for xi, ci, li, hi_ in zip(x, centre, lo, hi):
            print(f'  {xi:>9.2f}  {ci:>8.4f}  {li:>8.4f}  {hi_:>8.4f}')
        print()


def render(series_data, phase_starts, metric, band, out_path, mode, title, ylabel=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    t = THEME[mode]

    fig, ax = plt.subplots(figsize=(8.4, 4.8), dpi=200)
    fig.patch.set_facecolor(t['surface'])
    ax.set_facecolor(t['surface'])

    # ---- phase boundaries, read from the logs ------------------------- #
    for phase in sorted({round(p, 6) for p in phase_starts if p > 0}):
        ax.axvspan(0, phase, color=t['muted'], alpha=0.07, lw=0, zorder=0)
        ax.axvline(phase, color=t['muted'], ls=(0, (5, 4)), lw=1.4, zorder=1)
        ax.annotate(f'pretrain complete at {phase:g}\nevolution budget starts here',
                    xy=(phase, 0.985), xycoords=('data', 'axes fraction'),
                    xytext=(6, -4), textcoords='offset points',
                    color=t['ink_2'], fontsize=9, va='top', ha='left',
                    linespacing=1.4, zorder=5)

    # ---- one line per series ------------------------------------------ #
    ends = []
    for i, (name, (x, centre, lo, hi, n_seeds, _interp)) in enumerate(series_data.items()):
        colour = t['series'][i % len(t['series'])]
        if not np.allclose(lo, hi):
            ax.fill_between(x, lo, hi, color=colour, alpha=0.16, lw=0, zorder=2)
        ax.plot(x, centre, color=colour, lw=2, marker='o', ms=4.5,
                mec=t['surface'], mew=1.2, solid_capstyle='round',
                zorder=4, label=f'{name}  (n={n_seeds})')
        ends.append((float(centre[-1]), float(x[-1]), name, colour))

    # ---- recessive axes ----------------------------------------------- #
    all_x = np.concatenate([d[0] for d in series_data.values()])
    span = all_x.max() - all_x.min()
    ax.set_xlim(all_x.min() - 0.03 * span, all_x.max() + 0.22 * span)

    # ---- direct labels, nudged apart so they never overlap ------------- #
    # Series that converge end at nearly the same y, and two labels stacked on
    # one another are worse than none. Push them apart in draw order, keeping
    # each beside its own line end.
    lo_y, hi_y = ax.get_ylim()
    min_gap = 0.05 * (hi_y - lo_y)

    y_label = None
    for y_end, x_end, name, colour in sorted(ends):
        y_label = y_end if y_label is None else max(y_end, y_label + min_gap)
        ax.annotate(name, xy=(x_end, y_label), xytext=(8, 0),
                    textcoords='offset points', va='center', ha='left',
                    color=colour, fontsize=9.5, zorder=5,
                    annotation_clip=False)

    ax.set_xlabel('Training budget  (epoch-equivalents)', color=t['ink_2'], fontsize=10)
    ax.set_ylabel(ylabel or METRIC_LABELS.get(metric, metric),
                  color=t['ink_2'], fontsize=10)
    ax.set_title(title, color=t['ink'], fontsize=12, pad=14, loc='left')

    ax.grid(axis='y', color=t['grid'], lw=0.9, zorder=0)
    ax.set_axisbelow(True)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color(t['grid'])
    ax.tick_params(colors=t['muted'], labelsize=9, length=0)

    ax.legend(frameon=False, loc='lower right', fontsize=9,
              labelcolor=t['ink_2'], handlelength=1.6)

    fig.tight_layout()
    fig.savefig(out_path, facecolor=t['surface'])
    print(f'wrote {out_path}')


# --------------------------------------------------------------------------- #

def main():
    p = argparse.ArgumentParser(
        description='Plot SESiL / baseline runs on the shared budget axis.')
    p.add_argument('root', help='results directory to search for eval.jsonl files')
    p.add_argument('--metric', default=DEFAULT_METRIC,
                   help=f'which reduction to plot (default: {DEFAULT_METRIC})')
    p.add_argument('--band', default='std', choices=['std', 'minmax', 'iqr', 'none'],
                   help='spread across seeds (default: std)')
    p.add_argument('--out', default=None, help='output image path')
    p.add_argument('--mode', default='light', choices=['light', 'dark'],
                   help='colour mode; dark is stepped for the dark surface, not flipped')
    p.add_argument('--title', default=None,
                   help='figure title (default: derived from the directory name)')
    p.add_argument('--ylabel', default=None,
                   help='y-axis label (default: derived from --metric)')
    p.add_argument('--label', action='append', default=[], metavar='OLD=NEW',
                   help='rename a series in the legend and its direct label, e.g. '
                        '--label "sesil (permute)=SESiL (permutation)". Repeatable. '
                        'Run without it once to see the series names as detected.')
    p.add_argument('--order', default=None,
                   help='comma-separated series names (original, pre-rename) fixing '
                        'draw and colour order. Series not listed are dropped, which '
                        'is also how you plot a subset.')
    p.add_argument('--table', action='store_true',
                   help='also print the numbers behind every point')
    p.add_argument('--list-metrics', action='store_true',
                   help='list the metrics present in these logs and exit')
    args = p.parse_args()

    run_dirs = find_runs(args.root)
    if not run_dirs:
        raise SystemExit(f'No eval.jsonl found anywhere under {args.root}')

    runs = [r for r in (load_run(d) for d in run_dirs) if r]
    if not runs:
        raise SystemExit('Found eval.jsonl files but they are all empty.')

    if args.list_metrics:
        keys = sorted(
            k for k, v in runs[0]['records'][0].items()
            if isinstance(v, (int, float)) and not isinstance(v, bool)
        )
        print(f'metrics available in {args.root}:')
        for k in keys:
            print(f'  {k}' + (f'   -- {METRIC_LABELS[k]}' if k in METRIC_LABELS else ''))
        return

    grouped = group_runs(runs)

    if args.order:
        wanted = [n.strip() for n in args.order.split(',') if n.strip()]
        missing = [n for n in wanted if n not in grouped]
        if missing:
            raise SystemExit(
                f'--order names series that are not here: {missing}\n'
                f'detected: {sorted(grouped)}')
        grouped = {n: grouped[n] for n in wanted}

    renames = {}
    for pair in args.label:
        if '=' not in pair:
            raise SystemExit(f'--label expects OLD=NEW, got {pair!r}')
        old, new = pair.split('=', 1)
        old = old.strip()
        if old not in grouped:
            raise SystemExit(
                f'--label refers to {old!r}, which is not a series here.\n'
                f'detected: {sorted(grouped)}')
        renames[old] = new.strip()

    print(f'{len(runs)} run(s) in {len(grouped)} series:')
    for name, members in grouped.items():
        seeds = sorted(m['seed'] for m in members if m['seed'] is not None)
        shown = renames.get(name)
        suffix = f'   -> "{shown}"' if shown else ''
        print(f'  {name:<24} seeds {seeds if seeds else "?"}{suffix}')

    series_data = {}
    for name, members in grouped.items():
        missing = [m['dir'] for m in members
                   if args.metric not in m['records'][0]]
        if missing:
            raise SystemExit(
                f'Metric {args.metric!r} not in {missing[0]}. '
                f'Run with --list-metrics to see what is available.')
        series_data[name] = aggregate(members, args.metric, args.band)
        if series_data[name][5]:
            print(f'  NOTE: seeds of {name!r} have different budget grids; '
                  f'interpolated onto their overlap. Compare with care.')

    if renames:
        series_data = {renames.get(k, k): v for k, v in series_data.items()}

    if args.table:
        print_table(series_data, args.metric, args.band)

    out = args.out or os.path.join(args.root, f'{args.metric}.png')
    title = args.title or f'Budget-matched comparison  ({os.path.basename(os.path.normpath(args.root))})'
    render(series_data, [r['phase_start'] for r in runs],
           args.metric, args.band, out, args.mode, title, args.ylabel)


if __name__ == '__main__':
    main()
