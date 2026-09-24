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
using a GPU. Free memory is therefore consulted as well as the lock count, so a
card someone else is already filling is avoided even though it holds no lock of
ours.

PORTABILITY. The same command is expected to run on a single-GPU laptop and on
an 8-GPU server. A --gpus list that does not exist on the current machine is
therefore a warning that degrades to what is actually there, not a fatal error
-- pass --strict-gpus when the list is a hard requirement (a shared server
where the other cards are somebody else's) and you would rather fail than be
quietly moved.
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


def own_claim(allowed=None):
    """The GPU this process has already claimed, if any.

    A process must not be blocked by its own lock: counting it would make a
    second resolve_device() in the same run report the card as full and refuse
    it. Claiming is therefore idempotent -- ask twice, get the same GPU.
    """
    directory = _lock_dir()
    if not os.path.isdir(directory):
        return None
    suffix = f'.pid_{os.getpid()}.lock'
    for name in os.listdir(directory):
        if name.startswith('gpu_') and name.endswith(suffix):
            try:
                gpu_id = int(name.split('_')[1].split('.')[0])
            except (ValueError, IndexError):
                continue
            if allowed is None or gpu_id in allowed:
                return gpu_id
    return None


def _count_claims(allowed):
    """Live claims per allowed GPU, reaping locks whose process has gone.

    This process's own locks are skipped -- see own_claim().
    """
    directory = _lock_dir()
    os.makedirs(directory, exist_ok=True)

    self_pid = os.getpid()
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

        if pid == self_pid:
            continue                      # our own claim; see own_claim()

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


def free_gb(gpu_id):
    """Free memory on a GPU, in GiB. 0.0 when it cannot be read.

    Catches what a lock count cannot: another user's job, or a different
    project on the same card. Unreadable is reported as 0.0 so an unknown GPU
    loses a tie rather than winning one.
    """
    try:
        import torch
        free, _total = torch.cuda.mem_get_info(gpu_id)
        return free / (1024 ** 3)
    except Exception:
        return 0.0


def resolve_allowed(allowed, n_visible, strict=False):
    """Reconcile a requested GPU list with the machine actually running this.

    Returns the usable subset. A list naming cards this machine does not have
    is the normal case when a sweep moves between machines, so it degrades with
    a warning instead of failing -- unless `strict`, where being silently moved
    onto a card you excluded on purpose is the worse outcome.
    """
    if allowed is None:
        return list(range(n_visible))

    usable = [g for g in allowed if 0 <= g < n_visible]
    missing = [g for g in allowed if g not in usable]

    if missing and strict:
        raise SystemExit(
            f'--strict-gpus: requested GPU(s) {missing} do not exist here. '
            f'This machine exposes {n_visible} GPU(s), indices '
            f'0..{n_visible - 1}. (CUDA_VISIBLE_DEVICES renumbers from 0.)')

    if not usable:
        print(f'[gpu] WARNING: none of the requested GPUs {allowed} exist on '
              f'this machine, which has {n_visible} '
              f'(indices 0..{n_visible - 1}).')
        print(f'[gpu]          Falling back to all visible GPUs. Pass '
              f'--strict-gpus to make this an error instead.')
        return list(range(n_visible))

    if missing:
        print(f'[gpu] WARNING: requested GPU(s) {missing} do not exist here; '
              f'using {usable}.')
    return usable


def claim_gpu(allowed=None, max_per_gpu=1, min_free_gb=1.0, strict=False):
    """Claim the best allowed GPU. Returns its index, or None on CPU.

    "Best" is the fewest live claims, then the most free memory, then the
    lowest index -- so parallel runs spread out, and within a tie they avoid
    whatever another process is already occupying.

    `max_per_gpu` caps how many runs of this scheme share a card. Exceeding it
    is refused with an explanation rather than allowed to become an
    out-of-memory crash several minutes into a run.
    """
    import torch

    if not torch.cuda.is_available():
        return None

    n_visible = torch.cuda.device_count()
    allowed = resolve_allowed(allowed, n_visible, strict)

    held = own_claim(allowed)
    if held is not None:
        print(f'[gpu] reusing cuda:{held}, already claimed by this process')
        return held

    counts = _count_claims(allowed)
    free = {g: free_gb(g) for g in allowed}

    candidates = [g for g in allowed if counts[g] < max_per_gpu]
    if not candidates:
        busy = ', '.join(f'cuda:{g} {counts[g]} claim(s)' for g in sorted(allowed))
        raise SystemExit(
            f'Every allowed GPU is at the --max-per-gpu limit of {max_per_gpu} '
            f'({busy}).\n'
            f'Either launch fewer runs at once, widen --gpus, or raise '
            f'--max-per-gpu if the card really can hold another.\n'
            f'If a previous run was killed, its lock should have been reaped '
            f'automatically -- check {_lock_dir()}.')

    # Fewest claims, then most free memory, then lowest index.
    gpu_id = min(candidates, key=lambda g: (counts[g], -free[g], g))

    if free[gpu_id] < min_free_gb:
        print(f'[gpu] WARNING: cuda:{gpu_id} has only {free[gpu_id]:.1f} GiB '
              f'free (--min-free-gb {min_free_gb:g}). Proceeding, but an '
              f'out-of-memory failure is likely.')

    directory = _lock_dir()
    lock_path = os.path.join(directory, f'gpu_{gpu_id}.pid_{os.getpid()}.lock')
    with open(lock_path, 'w') as f:
        f.write(str(os.getpid()))
    atexit.register(_quiet_remove, lock_path)

    status = ', '.join(f'{g}:{counts[g]}c/{free[g]:.0f}G' for g in sorted(allowed))
    print(f'[gpu] claimed cuda:{gpu_id}  ({free[gpu_id]:.1f} GiB free; '
          f'all -- {status})')
    return gpu_id


def parse_gpu_list(value):
    """'0,1,3' -> [0, 1, 3]; None/'' -> None (meaning every visible GPU)."""
    if value is None or str(value).strip() in ('', 'none', 'all'):
        return None
    return [int(v) for v in str(value).split(',') if v.strip() != '']


def resolve_device(args):
    """Turn --device / --gpus into a concrete device string.

    Explicit wins: '--device cuda:2' or '--device cpu' is respected as given,
    but is checked against the machine first -- an index that does not exist
    would otherwise fail much later, inside the first forward pass, with an
    error that does not mention the flag that caused it.
    """
    import torch

    if args.device and args.device != 'cuda':
        return _check_explicit(args.device, torch)

    if not torch.cuda.is_available():
        return 'cpu'

    allowed = parse_gpu_list(
        getattr(args, 'gpus', None) or os.environ.get('SESIL_GPUS'))
    gpu_id = claim_gpu(
        allowed,
        max_per_gpu=getattr(args, 'max_per_gpu', 1),
        min_free_gb=getattr(args, 'min_free_gb', 1.0),
        strict=getattr(args, 'strict_gpus', False),
    )
    if gpu_id is None:
        return 'cpu'

    torch.cuda.set_device(gpu_id)
    return f'cuda:{gpu_id}'


def _check_explicit(device, torch):
    """Validate an explicit --device now rather than mid-run."""
    if not device.startswith('cuda'):
        return device                                  # cpu, mps, whatever

    if not torch.cuda.is_available():
        raise SystemExit(
            f'--device {device} was requested but CUDA is not available here. '
            f'Use --device cpu, or drop --device to auto-detect.')

    if ':' not in device:
        return device                                  # plain 'cuda' handled above

    index = int(device.split(':')[1])
    n_visible = torch.cuda.device_count()
    if index >= n_visible:
        raise SystemExit(
            f'--device {device} does not exist: this machine exposes '
            f'{n_visible} GPU(s), indices 0..{n_visible - 1}. '
            f'(CUDA_VISIBLE_DEVICES renumbers them from 0.)')

    torch.cuda.set_device(index)
    return device
