"""Bring finished runs into line with --keep-generations.

    python prune_results.py results              # show what would go
    python prune_results.py results --delete     # actually remove it

Runs made before --keep-generations existed hold every generation they ever
wrote. Nothing reads them: the evolution loop only ever touches gen_N, gen_N+1
and the final generation, so once a run is finished everything except the last
generation and gen_0 is dead weight -- and at 20 agents a generation is ~360 MB.

The retention rule is deliberately the same one sesil.population.prune_generations
applies during a run, so a pruned old run and a fresh one look alike:

    keep the `keep` most recent generations, and always keep gen_0,
    and in every pruned generation delete only the WEIGHTS

gen_0 is kept because it is the pretrained population -- the only thing in the
tree that cannot be regenerated from a later state. agent.json is kept because
it is 454 bytes against 18 MB of weights and it is the only record tying an
agent's parents to its own id.

Probe runs have a single gen_0 and are therefore untouched by construction.
"""

import argparse
import os


def generations(checkpoints):
    out = []
    for name in os.listdir(checkpoints):
        if name.startswith('gen_'):
            try:
                out.append((int(name[len('gen_'):]), os.path.join(checkpoints, name)))
            except ValueError:
                pass                       # not ours to interpret; leave it
    return sorted(out)


AGENT_META = 'agent.json'


def strip_weights(path):
    """Delete an agent directory's weights, keeping its metadata."""
    freed = 0
    for agent in sorted(os.listdir(path)):
        agent_dir = os.path.join(path, agent)
        if not os.path.isdir(agent_dir):
            continue
        for entry in sorted(os.listdir(agent_dir)):
            if entry == AGENT_META:
                continue
            target = os.path.join(agent_dir, entry)
            try:
                freed += os.path.getsize(target)
                os.remove(target)
            except OSError:
                pass
    return freed


def dir_bytes(path):
    total = 0
    for root, _dirs, files in os.walk(path):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    return total


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('root', help='results directory to walk')
    ap.add_argument('--keep', type=int, default=2,
                    help='Most recent generations to keep (default 2, matching '
                         'the --keep-generations default). gen_0 is always kept.')
    ap.add_argument('--delete', action='store_true',
                    help='Actually remove them. Without this, only prints.')
    args = ap.parse_args()

    if args.keep < 2:
        raise SystemExit('--keep must be at least 2, matching the run-time rule.')

    doomed, freed = [], 0
    for dirpath, dirnames, _files in os.walk(args.root):
        if os.path.basename(dirpath) != 'checkpoints':
            continue
        dirnames[:] = []
        gens = generations(dirpath)
        if not gens:
            continue
        newest = gens[-1][0]
        cutoff = newest - args.keep + 1
        gone = [(i, p) for i, p in gens if i != 0 and i < cutoff]
        if not gone:
            continue
        size = sum(dir_bytes(p) for _i, p in gone)
        freed += size
        doomed.extend(p for _i, p in gone)
        kept = [i for i, _p in gens if i not in {g for g, _ in gone}]
        print(f'{size / 2**30:7.1f} GiB  {len(gone):3} of {len(gens):3} gens  '
              f'{dirpath}')
        print(f'{"":12}  keeping gen_{kept[0]} and gen_'
              f'{", gen_".join(str(k) for k in kept[1:])}')

    print()
    print(f'{len(doomed)} generation director(ies), {freed / 2**30:.1f} GiB')
    if not args.delete:
        print('Dry run -- nothing removed. Re-run with --delete to remove them.')
        return 0
    for path in doomed:
        strip_weights(path)
    print(f'Stripped weights from {len(doomed)} generation(s); agent.json kept.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
