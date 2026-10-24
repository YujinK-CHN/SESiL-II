#!/usr/bin/env bash
#
# Submit a SESiL sweep to Slurm (Digital Research Alliance clusters).
#
# One command, all seeds in parallel, same flags as run.sh:
#
#   bash submit.sh --exp-name "sesil_ssl_learn_globa(label)_mf045" \
#     --dataset cifar100 --budget 400 --seeds 0,1,3 \
#     --method sesil --pretrain-mode ssl \
#     --mating-mode learn --mate-floor 0.45 \
#     --merger globa --merge-head label \
#     --individual-budget 1.5 --certify-top-frac 0.3 --log-predictions
#
# What this adds over calling run.sh directly:
#
#   * Requests one GPU per seed and maps seed i to cuda:i. Slurm renumbers
#     devices inside an allocation -- --gres=gpu:3 always presents cuda:0,1,2
#     whatever the physical cards are -- so the old "--gpus 0,1,3" would ask
#     for a card that does not exist in the job.
#   * Downloads the dataset HERE, on the login node, before submitting.
#     Compute nodes have no internet and torchvision's download=True would
#     fail several minutes into the job.
#   * Writes results to scratch. Home is quota-limited and a sweep of this
#     shape has reached 113 GB.
#   * Quotes the whole command into a generated job script, so an --exp-name
#     containing parentheses survives.
#
# Slurm settings come from flags or environment:
#   --account   (or $SESIL_ACCOUNT)   required by Alliance clusters
#   --time      (or $SESIL_TIME)      default 24:00:00
#   --mem       (or $SESIL_MEM)       default 64G
#   --cpus      (or $SESIL_CPUS)      default 12
#   --gpu-type  (or $SESIL_GPU_TYPE)  e.g. h100; empty asks for any GPU
#   --venv      (or $SESIL_VENV)      default ~/venvs/sesil
#   --data-dir  (or $SESIL_DATA)      default ~/scratch/data
#   --out-root  (or $SESIL_OUT)       default ~/scratch/sesil_results
#   --dry-run                         print the job script, submit nothing
#
set -euo pipefail

ACCOUNT="${SESIL_ACCOUNT:-}"
TIME="${SESIL_TIME:-24:00:00}"
MEM="${SESIL_MEM:-64G}"
CPUS="${SESIL_CPUS:-12}"
GPU_TYPE="${SESIL_GPU_TYPE:-}"
VENV="${SESIL_VENV:-$HOME/venvs/sesil}"
DATA_DIR="${SESIL_DATA:-$HOME/scratch/data}"
OUT_ROOT="${SESIL_OUT:-$HOME/scratch/sesil_results}"
REPO="${SESIL_REPO:-$HOME/SESiL-II}"
DRY_RUN=0

SEEDS="0"
DATASET="cifar100"
EXP_NAME="check"
PASS=()          # everything else goes to run.sh untouched

while [[ $# -gt 0 ]]; do
  case "$1" in
    --account)  ACCOUNT="$2";  shift 2;;
    --time)     TIME="$2";     shift 2;;
    --mem)      MEM="$2";      shift 2;;
    --cpus)     CPUS="$2";     shift 2;;
    --gpu-type) GPU_TYPE="$2"; shift 2;;
    --venv)     VENV="$2";     shift 2;;
    --out-root) OUT_ROOT="$2"; shift 2;;
    --repo)     REPO="$2";     shift 2;;
    --dry-run)  DRY_RUN=1;     shift;;
    # Needed by both this script and run.sh, so read AND passed through.
    --seeds)    SEEDS="$2";    PASS+=("$1" "$2"); shift 2;;
    --dataset)  DATASET="$2";  PASS+=("$1" "$2"); shift 2;;
    --exp-name) EXP_NAME="$2"; PASS+=("$1" "$2"); shift 2;;
    # Consumed here because the compute node needs an absolute path.
    --data-dir) DATA_DIR="$2"; shift 2;;
    *)          PASS+=("$1");  shift;;
  esac
done

if [[ -z "$ACCOUNT" ]]; then
  echo "No --account given and \$SESIL_ACCOUNT is unset." >&2
  echo "Alliance clusters reject jobs without one, e.g. --account def-yourpi" >&2
  exit 1
fi
if [[ ! -d "$REPO" ]]; then
  echo "No repository at $REPO (override with --repo)." >&2
  exit 1
fi

IFS=',' read -ra SEED_LIST <<< "$SEEDS"
N_SEEDS=${#SEED_LIST[@]}

# Seed i -> cuda:i. The seed LABELS stay whatever the user asked for; only the
# device index is renumbered, because that is what Slurm hands us.
GPU_LIST=""
for ((i = 0; i < N_SEEDS; i++)); do
  GPU_LIST+="${i}"
  (( i < N_SEEDS - 1 )) && GPU_LIST+=","
done

if [[ -n "$GPU_TYPE" ]]; then
  GRES="gpu:${GPU_TYPE}:${N_SEEDS}"
else
  GRES="gpu:${N_SEEDS}"
fi

mkdir -p "$DATA_DIR" "$OUT_ROOT" "$HOME/scratch/sesil_jobs"

# ---- fetch the dataset here, where there is internet --------------------- #
if [[ $DRY_RUN -eq 0 ]]; then
  echo "[submit] checking $DATASET in $DATA_DIR (login node has internet; "
  echo "         compute nodes do not, so this cannot wait until the job runs)"
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
# Slurm reads #SBATCH lines literally -- no shell quoting -- so the job
# name is sanitised rather than escaped. The real --exp-name, parentheses
# and all, still reaches run.sh below.
SAFE_NAME="${EXP_NAME//[^A-Za-z0-9._-]/_}"
JOB="$HOME/scratch/sesil_jobs/${SAFE_NAME}.sbatch"

# printf %q so parentheses in --exp-name survive the round trip through a file.
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
#SBATCH --gres=${GRES}
#SBATCH --cpus-per-task=${CPUS}
#SBATCH --mem=${MEM}
#SBATCH --time=${TIME}
#SBATCH --output=${OUT_ROOT}/%x-%j.out

set -euo pipefail
source ${VENV}/bin/activate
cd ${REPO}

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
  echo "[submit] --dry-run: nothing submitted."
  exit 0
fi

sbatch "$JOB"
echo "[submit] watch with: squeue -u \$USER     logs in $OUT_ROOT"
