"""Add arm-defining fields to eval.jsonl files written before they were logged.

    python backfill_meta.py results            # show what would change
    python backfill_meta.py results --write    # rewrite the files

plot_results groups runs into series by the meta fields carried on each eval
record. A field that is not carried is invisible to the grouping, so two arms
differing only in that field become ONE series and their seeds get averaged
together -- six runs of two conditions collapsing into a single curve, with no
error and only a budget-grid note to hint at it.

--individual-budget and --certify-top-frac were missing. They are read back
here from each run's own config.json, which has recorded them all along.
"""

import argparse
import json
import os

SESIL_FIELDS = ('individual_budget', 'certify_top_frac')
# A curriculum baseline's arm is the run it replays; without this two rounds'
# baselines group together and get averaged.
BASELINE_FIELDS = ('baseline_mode', 'curriculum_from')
# For a single model every ensemble rule IS that model, so these are exactly
# recoverable from a finished baseline run -- no re-run needed to put the two
# methods on one axis.
BASELINE_ENSEMBLE = ('ensemble_hard', 'ensemble_soft', 'ensemble_conf_weighted',
                     'ensemble_max_confidence', 'ensemble_soft_certified',
                     'ensemble_soft_certified_norm')


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('root')
    ap.add_argument('--write', action='store_true',
                    help='Rewrite the files. Without this, only prints.')
    args = ap.parse_args()

    touched = 0
    for dirpath, _dirnames, files in os.walk(args.root):
        if 'eval.jsonl' not in files or 'config.json' not in files:
            continue
        cfg = json.load(open(os.path.join(dirpath, 'config.json')))['config']
        method = cfg.get('method')
        if method == 'sesil':
            fields = SESIL_FIELDS
        elif method == 'baseline':
            fields = BASELINE_FIELDS
        else:
            continue                       # probe runs are not plotted as arms

        path = os.path.join(dirpath, 'eval.jsonl')
        rows = [json.loads(l) for l in open(path, encoding='utf-8') if l.strip()]
        missing = [f for f in fields if f not in rows[0]]
        if method == 'baseline' and any(f not in rows[0] for f in BASELINE_ENSEMBLE):
            missing = missing or ['<ensemble>']
        if not missing:
            print(f'  ok       {path}')
            continue

        vals = {f: cfg.get(f) for f in missing}

        # Ensemble metrics for the baseline are derived per RECORD, not from
        # config -- each eval point has its own accuracy.
        ens_missing = (method == 'baseline'
                       and any(f not in rows[0] for f in BASELINE_ENSEMBLE))
        print(f'  backfill {path}  {vals}  ({len(rows)} records)')
        touched += 1
        if args.write:
            for r in rows:
                r.update(vals)
                if ens_missing and 'best_agent_overall' in r:
                    solo = r['best_agent_overall']
                    for f in BASELINE_ENSEMBLE:
                        r.setdefault(f, solo)
            tmp = path + '.tmp'
            with open(tmp, 'w', encoding='utf-8', newline='\n') as f:
                for r in rows:
                    f.write(json.dumps(r) + '\n')
            os.replace(tmp, path)          # atomic: never a half-written log

    print()
    print(f'{touched} file(s) need backfilling.')
    if touched and not args.write:
        print('Dry run -- nothing written. Re-run with --write.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
