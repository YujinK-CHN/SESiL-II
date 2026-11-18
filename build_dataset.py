"""
Build the genetic-learning dataset from finished SESiL runs.

    # 1. profiles for every ordered pair + the children the runs bred (weights
    #    only, no data). Runs on CPU: the decomposition is float64, which is
    #    ~4x slower on a consumer GPU and differs from evolution's CPU numbers
    #    in the 4th decimal. --jobs runs that many source runs in parallel.
    python build_dataset.py profiles --results results --out dataset/sesil_genetic_pairs --jobs 4

    # 2. more labels: build and measure children for pairs the runs never
    #    bred, N per generation per run (GPU, needs the datasets).
    python build_dataset.py children --results results --out dataset/sesil_genetic_pairs \
        --per-generation 20 --preset globa --head label --device cuda

    # 3. rewrite README.md / runs.json with current counts (both steps do it too).
    python build_dataset.py card --results results --out dataset/sesil_genetic_pairs

Every step writes one shard per source run and skips work already on disk,
so a killed job is resumed by running the same command again.
--include takes regexes on the run id (e.g. 'globa\\(label\\)') to restrict
which runs are used.
"""

import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

from sesil import genetic_dataset as gd


def _profiles_job(run_dir, run_id, cfg, out, threads):
    import torch
    torch.set_num_threads(threads)
    return run_id, gd.build_pairs(run_dir, run_id, cfg, out, device='cpu')


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('step', choices=['profiles', 'children', 'card'])
    p.add_argument('--results', default='results')
    p.add_argument('--out', default='dataset/sesil_genetic_pairs')
    p.add_argument('--include', nargs='*', default=None,
                   help='regexes on run id; a run is used if any matches')
    p.add_argument('--device', default='cpu',
                   help='children step only (profiles always run on CPU)')
    p.add_argument('--jobs', type=int, default=1,
                   help='profiles step: source runs processed in parallel; '
                        'CPU threads are split evenly between them')
    p.add_argument('--data-dir', default=None,
                   help='children step: where the image datasets are, if not '
                        'where the source runs found them')
    p.add_argument('--per-generation', type=int, default=20,
                   help='children step: generated children per generation '
                        'per run (on top of those the run bred)')
    p.add_argument('--preset', default='globa',
                   help='children step: GLOBA preset of the generated merges')
    p.add_argument('--head', default='label', choices=['label', 'average'],
                   help='children step: merge head of the generated merges')
    p.add_argument('--seed', type=int, default=0)
    a = p.parse_args(argv)

    runs = gd.discover_runs(a.results, a.include)
    print(f'[dataset] {len(runs)} usable source runs under {a.results}')
    if not runs:
        return 1

    t0 = time.time()
    if a.step == 'profiles':
        jobs = max(1, a.jobs)
        threads = max(1, (os.cpu_count() or 1) // jobs)
        with ProcessPoolExecutor(max_workers=jobs) as pool:
            futures = [pool.submit(_profiles_job, d, r, c, a.out, threads)
                       for d, r, c in runs]
            for i, f in enumerate(as_completed(futures), 1):
                run_id, (n_pairs, n_kids) = f.result()
                print(f'[dataset] ({i}/{len(runs)}) done {run_id}: '
                      f'{n_pairs} pairs, {n_kids} bred children', flush=True)
    for i, (run_dir, run_id, cfg) in enumerate(runs, 1):
        if a.step == 'children':
            print(f'[dataset] ({i}/{len(runs)}) {run_id}')
            n = gd.generate_children(run_dir, run_id, cfg, a.out,
                                     per_generation=a.per_generation,
                                     preset=a.preset, head=a.head,
                                     device=a.device, data_dir=a.data_dir,
                                     seed=a.seed)
            print(f'[dataset]   {n} generated children on disk')

    gd.write_card(a.out, [gd.run_record(run_id, cfg) for _d, run_id, cfg in runs])
    print(f'[dataset] done in {time.time() - t0:.0f}s; card at {a.out}/README.md')
    return 0


if __name__ == '__main__':
    sys.exit(main())
