#!/usr/bin/env bash
#
# Trillium version of submit.sh (same flags, passed to run.sh). Run it on the
# GPU login node (ssh trillium-gpu, trig-login01).
#
# How Trillium differs from nibi, and what this script does about it:
#
#   * Jobs start with --export=NONE: nothing from the login shell carries
#     over, so the job script loads the modules itself before the venv.
#   * Each GPU comes with a fixed 24 cores and 187.5 GB, so this asks only for
#     --gpus-per-node (at most 4 per node); --mem / --cpus-per-task are not set.
#   * HOME is read-only on compute nodes. The code stays in ~/SESiL-II, but
#     results, data and caches go to $SCRATCH.
#   * Every GPU partition has a 24 h limit. With --chain K, K jobs of up to 24 h
#     are queued, each starting when the one before it ends
#     (--dependency=afterany). Every link runs the same command, so the command
#     must be able to resume (generate_dataset.py --resume). A job that finds
#     its run already finished exits at once.
#
#   bash submit_trillium.sh --account def-zinovi --chain 4 <run.sh flags...>
#
# Slurm settings come from flags or environment:
#   --account   (or $SESIL_ACCOUNT)   required
#   --time      (or $SESIL_TIME)      default 24:00:00 (the maximum)
#   --chain     (or $SESIL_CHAIN)     default 1; jobs queued one after another
#   --partition (or $SESIL_PARTITION) default compute (debug: 2 h, for a test)
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
    --seeds)     SEEDS="$2";     PASS+=("$1" "$2"); shift 2;;
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
N_SEEDS=${#SEED_LIST[@]}
if (( N_SEEDS > 4 )); then
  echo "$N_SEEDS seeds need $N_SEEDS GPUs; a Trillium node has 4. Submit in groups of <= 4." >&2
  exit 1
fi

# Seed i -> cuda:i (pinned in run.sh, so no lock files are written to the
# read-only repo).
GPU_LIST=""
for ((i = 0; i < N_SEEDS; i++)); do
  GPU_LIST+="${i}"
  (( i < N_SEEDS - 1 )) && GPU_LIST+=","
done

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
  python - "$DATASET" "$DATA_DIR" <<'PYEOF'
import sys
import torchvision
name, root = sys.argv[1], sys.argv[2]
cls = {'cifar10': torchvision.datasets.CIFAR10,
       'cifar100': torchvision.datasets.CIFAR100}.get(name)
if cls is None:
    print(f'[submit] no pre-fetch rule for {name!r}; make sure it is already '
          f'in {root}')
else:
    for train in (True, False):
        cls(root=root, train=train, download=True)
    print(f'[submit] {name} ready in {root}')
PYEOF
fi

# ---- build the job script ------------------------------------------------ #
SAFE_NAME="${EXP_NAME//[^A-Za-z0-9._-]/_}"
JOB="$OUT_ROOT/jobs/${SAFE_NAME}.sbatch"
CACHE="$SCRATCH/.cache"

RUN_ARGS=""
for a in "${PASS[@]}"; do RUN_ARGS+=" $(printf '%q' "$a")"; done
RUN_ARGS+=" --gpus $(printf '%q' "$GPU_LIST")"
RUN_ARGS+=" --run-mode parallel"
RUN_ARGS+=" --data-dir $(printf '%q' "$DATA_DIR")"
RUN_ARGS+=" --output-root $(printf '%q' "$OUT_ROOT")"

cat > "$JOB" <<EOF
#!/bin/bash
#SBATCH --account=${ACCOUNT}
#SBATCH --job-name=${SAFE_NAME}
#SBATCH --partition=${PARTITION}
#SBATCH --gpus-per-node=${N_SEEDS}
#SBATCH --time=${TIME}
#SBATCH --output=${OUT_ROOT}/%x-%j.out

set -euo pipefail
module load ${MODULES}
source ${VENV}/bin/activate
cd ${REPO}

# HOME is read-only here: keep Python, torch and matplotlib off it.
export PYTHONDONTWRITEBYTECODE=1
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

echo "[submit] job script -> $JOB"
if [[ $DRY_RUN -eq 1 ]]; then
  echo "---------------------------------------------------------------"
  cat "$JOB"
  echo "---------------------------------------------------------------"
  echo "[submit] --dry-run: nothing submitted (would queue $CHAIN job(s))."
  exit 0
fi

# ---- queue the chain ----------------------------------------------------- #
# afterany: the next link starts whether the previous one ended normally, hit
# the time limit, or failed -- resume decides what is left to do.
PREV=""
IDS=()
for ((k = 1; k <= CHAIN; k++)); do
  DEP=()
  [[ -n "$PREV" ]] && DEP=(--dependency=afterany:"$PREV")
  if ! PREV=$(sbatch --parsable ${DEP[@]+"${DEP[@]}"} "$JOB"); then
    # e.g. the debug partition accepts only one queued job per user.
    echo "[submit] link $k of $CHAIN was refused; queued so far: ${IDS[*]:-none}" >&2
    echo "[submit] resubmit the rest later with: sbatch $JOB" >&2
    exit 1
  fi
  PREV="${PREV%%;*}"
  IDS+=("$PREV")
done
echo "[submit] queued ${CHAIN} job(s): ${IDS[*]}"
echo "[submit] watch with: squeue -u \$USER     logs in $OUT_ROOT"
echo "[submit] cancel the whole chain with: scancel ${IDS[*]}"
