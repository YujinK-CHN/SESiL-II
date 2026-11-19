#!/usr/bin/env bash
#
# Trillium version of submit.sh (same flags, passed to run.sh). Run it on the
# GPU login node (ssh trillium-gpu, trig-login01).
#
# How Trillium differs from nibi, and what this script does about it:
#
#   * Jobs start with --export=NONE: nothing from the login shell carries
#     over, so the job script loads the modules itself before the venv.
#   * A job may ask for 1 GPU or whole 4-GPU nodes, nothing in between. Each
#     seed therefore gets its OWN 1-GPU job (and its own chain): no idle GPU,
#     single GPUs are scheduled sooner, and a seed that fails or finishes does
#     not hold the others. Each GPU comes with a fixed 24 cores and 187.5 GB,
#     so --mem / --cpus-per-task are not set.
#   * HOME is read-only on compute nodes. The code stays in ~/SESiL-II, but
#     results, data and caches go to $SCRATCH.
#   * Every GPU partition has a 24 h limit. With --chain K, K jobs of up to 24 h
#     are queued, each starting when the one before it ends
#     (--dependency=afterany). Every link runs the same command, so the command
#     must be able to resume (generate_dataset.py --resume). A job that finds
#     its run already finished exits at once.
#
#   bash submit_trillium.sh --account def-zinovi --chain 4 --seeds 0,1,3 <run.sh flags...>
#   -> 3 chains of 4 jobs, one per seed.
#
# Slurm settings come from flags or environment:
#   --account   (or $SESIL_ACCOUNT)   required
#   --time      (or $SESIL_TIME)      default 24:00:00 (the maximum)
#   --chain     (or $SESIL_CHAIN)     default 1; jobs queued one after another
#   --partition (or $SESIL_PARTITION) default compute (debug: 1 h, one job
#                                     per user, so one seed and --chain 1)
#   --modules   (or $SESIL_MODULES)   default "StdEnv/2023 python/3.11"; must
#                                     match the modules the venv was made with
#   --venv      (or $SESIL_VENV)      default ~/ENV_sesil
#   --data-dir  (or $SESIL_DATA)      default $SCRATCH/data
#   --out-root  (or $SESIL_OUT)       default $SCRATCH/SESiL-II/results
#   --repo      (or $SESIL_REPO)      default ~/SESiL-II
#   --dry-run                         print the job script, submit nothing
#
set -euo pipefail

SCRATCH="${SCRATCH:-/scratch/$USER}"
ACCOUNT="${SESIL_ACCOUNT:-}"
TIME="${SESIL_TIME:-24:00:00}"
CHAIN="${SESIL_CHAIN:-1}"
PARTITION="${SESIL_PARTITION:-compute}"
MODULES="${SESIL_MODULES:-StdEnv/2023 python/3.11}"
VENV="${SESIL_VENV:-$HOME/ENV_sesil}"
DATA_DIR="${SESIL_DATA:-$SCRATCH/data}"
REPO="${SESIL_REPO:-$HOME/SESiL-II}"
OUT_ROOT="${SESIL_OUT:-$SCRATCH/SESiL-II/results}"
DRY_RUN=0

SEEDS="0"
DATASET="cifar100"
EXP_NAME="check"
PASS=()          # everything else goes to run.sh untouched

while [[ $# -gt 0 ]]; do
  case "$1" in
    --account)   ACCOUNT="$2";   shift 2;;
    --time)      TIME="$2";      shift 2;;
    --chain)     CHAIN="$2";     shift 2;;
    --partition) PARTITION="$2"; shift 2;;
    --modules)   MODULES="$2";   shift 2;;
    --venv)      VENV="$2";      shift 2;;
    --out-root)  OUT_ROOT="$2";  shift 2;;
    --repo)      REPO="$2";      shift 2;;
    --dry-run)   DRY_RUN=1;      shift;;
    # Needed by both this script and run.sh, so read AND passed through.
    # One job per seed: --seeds is given to run.sh per job, below.
    --seeds)     SEEDS="$2";     shift 2;;
    --dataset)   DATASET="$2";   PASS+=("$1" "$2"); shift 2;;
    --exp-name)  EXP_NAME="$2";  PASS+=("$1" "$2"); shift 2;;
    --data-dir)  DATA_DIR="$2";  shift 2;;
    --mem|--cpus|--gpu-type)
      echo "[submit] $1 is ignored on Trillium (fixed per GPU: 24 cores, 187.5 GB, H100)." >&2
      shift 2;;
    *)           PASS+=("$1");   shift;;
  esac
done

if [[ -z "$ACCOUNT" ]]; then
  echo "No --account given and \$SESIL_ACCOUNT is unset (e.g. --account def-zinovi)." >&2
  exit 1
fi
if [[ ! -d "$REPO" ]]; then
  echo "No repository at $REPO (override with --repo)." >&2
  exit 1
fi
if (( CHAIN > 1 )) && [[ " ${PASS[*]} " != *" --resume "* ]]; then
  echo "--chain $CHAIN needs a command that resumes (pass --resume);" >&2
  echo "otherwise every link would start the run again from scratch." >&2
  exit 1
fi

IFS=',' read -ra SEED_LIST <<< "$SEEDS"

mkdir -p "$DATA_DIR" "$OUT_ROOT" "$OUT_ROOT/jobs"

# ---- fetch the dataset here, where there is internet --------------------- #
if [[ $DRY_RUN -eq 0 ]]; then
  echo "[submit] checking $DATASET in $DATA_DIR (compute nodes have no internet)"
  # shellcheck disable=SC2086
  module load $MODULES
  # shellcheck disable=SC1091
  source "$VENV/bin/activate" 2>/dev/null || {
    echo "Could not activate $VENV. Pass --venv or set \$SESIL_VENV." >&2
    exit 1; }
  python - "$DATASET" "$DATA_DIR" "$REPO" <<'PYEOF'
import os
import sys
name, data_dir, repo = sys.argv[1], sys.argv[2], sys.argv[3]
# The same folder the run reads (sesil/data.py: <data-dir>/<subdir>), so the
# job finds it and never tries to download on an offline compute node.
sys.path.insert(0, repo)
from sesil.data import _DATASET_SPEC
spec = _DATASET_SPEC.get(name)
if spec is None:
    print(f'[submit] no pre-fetch rule for {name!r}; make sure it is already '
          f'in {data_dir}')
else:
    root = os.path.join(data_dir, spec['subdir'])
    for train in (True, False):
        spec['wrapper'](root=root, train=train, download=True)
    print(f'[submit] {name} ready in {root}')
PYEOF
fi

# ---- one job script and one chain per seed ------------------------------- #
SAFE_NAME="${EXP_NAME//[^A-Za-z0-9._-]/_}"
CACHE="$SCRATCH/.cache"
ALL_IDS=()

for SEED in "${SEED_LIST[@]}"; do
  JOB="$OUT_ROOT/jobs/${SAFE_NAME}_seed${SEED}.sbatch"

  RUN_ARGS=""
  for a in "${PASS[@]}"; do RUN_ARGS+=" $(printf '%q' "$a")"; done
  RUN_ARGS+=" --seeds $(printf '%q' "$SEED")"
  # One seed on the job's one GPU, pinned (so run.sh writes no lock files to
  # the read-only repo).
  RUN_ARGS+=" --gpus 0"
  RUN_ARGS+=" --run-mode parallel"
  RUN_ARGS+=" --data-dir $(printf '%q' "$DATA_DIR")"
  RUN_ARGS+=" --output-root $(printf '%q' "$OUT_ROOT")"

  cat > "$JOB" <<EOF
#!/bin/bash
#SBATCH --account=${ACCOUNT}
#SBATCH --job-name=${SAFE_NAME}_s${SEED}
#SBATCH --partition=${PARTITION}
#SBATCH --gpus-per-node=1
#SBATCH --time=${TIME}
#SBATCH --output=${OUT_ROOT}/%x-%j.out

set -euo pipefail
module load ${MODULES}
source ${VENV}/bin/activate
cd ${REPO}

# HOME is read-only here: keep Python, torch and matplotlib off it.
export PYTHONDONTWRITEBYTECODE=1
# Unbuffered, so a job killed at its time limit keeps its log up to the kill.
export PYTHONUNBUFFERED=1
export XDG_CACHE_HOME=${CACHE}
export TORCH_HOME=${CACHE}/torch
export MPLCONFIGDIR=${CACHE}/matplotlib
mkdir -p ${CACHE}/torch ${CACHE}/matplotlib

echo "host    : \$(hostname)"
echo "gpus    : \${CUDA_VISIBLE_DEVICES:-unset}  (\$(python -c 'import torch;print(torch.cuda.device_count())') visible to torch)"
echo "started : \$(date)"

bash run.sh${RUN_ARGS}

echo "finished: \$(date)"
EOF
  chmod +x "$JOB"
  echo "[submit] seed ${SEED}: job script -> $JOB"

  if [[ $DRY_RUN -eq 1 ]]; then
    echo "---------------------------------------------------------------"
    cat "$JOB"
    echo "---------------------------------------------------------------"
    continue
  fi

  # afterany: the next link starts whether the previous one ended normally,
  # hit the time limit, or failed -- resume decides what is left to do.
  PREV=""
  IDS=()
  for ((k = 1; k <= CHAIN; k++)); do
    DEP=()
    [[ -n "$PREV" ]] && DEP=(--dependency=afterany:"$PREV")
    if ! PREV=$(sbatch --parsable ${DEP[@]+"${DEP[@]}"} "$JOB"); then
      # e.g. the debug partition accepts only one queued job per user.
      echo "[submit] seed ${SEED}: link $k of $CHAIN was refused; queued so far: ${ALL_IDS[*]:-none} ${IDS[*]:-}" >&2
      echo "[submit] resubmit later with: sbatch $JOB" >&2
      exit 1
    fi
    PREV="${PREV%%;*}"
    IDS+=("$PREV")
  done
  ALL_IDS+=("${IDS[@]}")
  echo "[submit] seed ${SEED}: queued ${CHAIN} job(s): ${IDS[*]}"
done

if [[ $DRY_RUN -eq 1 ]]; then
  echo "[submit] --dry-run: nothing submitted (would queue ${#SEED_LIST[@]} x ${CHAIN} job(s))."
  exit 0
fi
echo "[submit] watch with: squeue -u \$USER     logs in $OUT_ROOT"
echo "[submit] cancel everything with: scancel ${ALL_IDS[*]}"
