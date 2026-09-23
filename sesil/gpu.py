"""
GPU assignment for concurrent runs.

Without this, every parallel run resolves --device to plain 'cuda' and lands on
cuda:0 while the other GPUs sit idle.

The mechanism is SEMCS's, ported: each process claims the least-loaded allowed
GPU by dropping a lock file named

    gpu_<id>.pid_<pid>.lock

Counting the live lock files gives the current load per GPU. Locks whose PID is
gone are reaped on sight, so a killed run does not permanently inflate a GPU's
count -- which matters because runs here are long and get interrupted.

It is advisory, not enforced: nothing stops a process outside this scheme from
using a GPU. That is fine for the case it exists for -- several seeds of the
same sweep launched together from run.sh.
"""

import atexit
import os


LOCK_DIRNAME = os.path.join('results', '.gpu_locks')


def _lock_dir():
    return os.path.abspath(
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     LOCK_DIRNAME))


def _live(pid):
    """Whether a process is still running. Windows and POSIX differ here."""
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    except Exception:
        return True          # can't tell; assume alive rather than steal its slot
    return True


def _count_claims(allowed):
    """Live claims per allowed GPU, reaping locks whose process has gone."""
    directory = _lock_dir()
    os.makedirs(directory, exist_ok=True)

    counts = {g: 0 for g in allowed}
    for name in os.listdir(directory):
        if not (name.startswith('gpu_') and name.endswith('.lock')):
            continue
        path = os.path.join(directory, name)
        try:
            gpu_id = int(name.split('_')[1].split('.')[0])
            pid = int(name.split('pid_')[1].split('.')[0])
        except (ValueError, IndexError):
            _quiet_remove(path)          # malformed, not ours to interpret
            continue

        if not _live(pid):
            _quiet_remove(path)
            continue

        if gpu_id in counts:
            counts[gpu_id] += 1

    return counts


def _quiet_remove(path):
    try:
        os.remove(path)
    except OSError:
        pass                              # another process reaped it first


def claim_gpu(allowed=None):
    """Claim the least-loaded allowed GPU. Returns its index, or None on CPU.

    `allowed` is a list of GPU indices, or None for every visible GPU.
    """
    import torch

    if not torch.cuda.is_available():
        return None

    n_visible = torch.cuda.device_count()
    if allowed is None:
        allowed = list(range(n_visible))
    else:
        allowed = [g for g in allowed if 0 <= g < n_visible]
        if not allowed:
            raise SystemExit(
                f'--gpus selects no usable device: this machine exposes '
                f'{n_visible} GPU(s), indices 0..{n_visible - 1}. Note that '
                f'CUDA_VISIBLE_DEVICES renumbers them from 0.')

    counts = _count_claims(allowed)
    gpu_id = min(counts, key=lambda g: (counts[g], g))

    directory = _lock_dir()
    lock_path = os.path.join(directory, f'gpu_{gpu_id}.pid_{os.getpid()}.lock')
    with open(lock_path, 'w') as f:
        f.write(str(os.getpid()))
    atexit.register(_quiet_remove, lock_path)

    busy = ', '.join(f'{g}:{counts[g]}' for g in sorted(counts))
    print(f'[gpu] claimed cuda:{gpu_id}  (existing claims -- {busy})')
    return gpu_id


def parse_gpu_list(value):
    """'0,1,3' -> [0, 1, 3]; None/'' -> None (meaning every visible GPU)."""
    if value is None or str(value).strip() in ('', 'none', 'all'):
        return None
    return [int(v) for v in str(value).split(',') if v.strip() != '']


def resolve_device(args):
    """Turn --device / --gpus into a concrete device string.

    Explicit wins: '--device cuda:2' or '--device cpu' is respected as given.
    Otherwise a GPU is claimed, so concurrent runs spread out instead of all
    piling onto cuda:0.
    """
    import torch

    if args.device and args.device != 'cuda':
        return args.device                       # explicit, including cpu

    if not torch.cuda.is_available():
        return 'cpu'

    allowed = parse_gpu_list(
        getattr(args, 'gpus', None) or os.environ.get('SESIL_GPUS'))
    gpu_id = claim_gpu(allowed)
    if gpu_id is None:
        return 'cpu'

    torch.cuda.set_device(gpu_id)
    return f'cuda:{gpu_id}'
