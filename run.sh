#!/usr/bin/env bash
#
# Master launcher. Controls what is shared across every method: seeds, the
# training budget, the dataset, and which methods to dispatch.
#
# Usage:
#   bash run.sh --seeds 0,1,2 [--run-mode sequential] [--method sesil] [extra flags...]
#
# Examples:
#   bash run.sh --seeds 0,1,2
#   bash run.sh --seeds 0 --method sesil --merger zipit --generations 40
#   bash run.sh --seeds 0,1,2,3 --run-mode parallel --dataset cifar100
#
# Any flag this script does not recognise is passed straight through to
# run_<method>.sh and on to main.py, so every knob in config.py is reachable
# from the command line without editing Python.

set -euo pipefail

# ───────────────────────── defaults ─────────────────────────
SEEDS="0"
RUN_MODE="sequential"      # sequential | parallel
EXP_NAME="check"
DATASET="cifar10"

# Training budget
GENERATIONS=25             # SESiL: generations
BASELINE_EPOCHS=100        # baseline: epochs

# Population
POP_SIZE=10
CLASSES_PER_MODEL=3
PRETRAIN_EPOCHS=20

# Which methods to run. Comment out what you do not need.
METHODS=(
  sesil
  # baseline
)

EXTRA_ARGS=()

# ───────────────────────── arg parsing ──────────────────────
while [[ $# -gt 0 ]]; do
  case "$1" in
    --seeds)             SEEDS="$2"; shift 2;;
    --run-mode)          RUN_MODE="$2"; shift 2;;
    --exp-name)          EXP_NAME="$2"; shift 2;;
    --dataset)           DATASET="$2"; shift 2;;
    --generations)       GENERATIONS="$2"; shift 2;;
    --baseline-epochs)   BASELINE_EPOCHS="$2"; shift 2;;
    --pop-size)          POP_SIZE="$2"; shift 2;;
    --classes-per-model) CLASSES_PER_MODEL="$2"; shift 2;;
    --pretrain-epochs)   PRETRAIN_EPOCHS="$2"; shift 2;;
    --method)            METHODS=("$2"); shift 2;;
    *)                   EXTRA_ARGS+=("$1"); shift;;
  esac
done

IFS=',' read -ra SEED_LIST <<< "$SEEDS"

echo "Methods : ${METHODS[*]}"
echo "Seeds   : ${SEED_LIST[*]}"
echo "Mode    : $RUN_MODE"
echo "Dataset : $DATASET"
echo "Extra   : ${EXTRA_ARGS[*]-}"

# Settings every method shares.
COMMON_ARGS=(
  --exp-name "$EXP_NAME"
  --dataset "$DATASET"
  --pop-size "$POP_SIZE"
  --classes-per-model "$CLASSES_PER_MODEL"
  --pretrain-epochs "$PRETRAIN_EPOCHS"
)

# ───────────────────────── dispatch ─────────────────────────
PIDS=()
LABELS=()

for method in "${METHODS[@]}"; do
  script="run_${method}.sh"
  if [[ ! -f "$script" ]]; then
    echo "No launcher $script for method '$method'." >&2
    exit 1
  fi

  # Per-method budget flag.
  case "$method" in
    sesil)    BUDGET_ARGS=(--generations "$GENERATIONS");;
    baseline) BUDGET_ARGS=(--baseline-epochs "$BASELINE_EPOCHS");;
    *)        BUDGET_ARGS=();;
  esac

  for seed in "${SEED_LIST[@]}"; do
    echo ">> $method  seed=$seed"
    if [[ "$RUN_MODE" == "parallel" ]]; then
      bash "$script" --seed "$seed" "${COMMON_ARGS[@]}" "${BUDGET_ARGS[@]}" ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"} &
      PIDS+=($!)
      LABELS+=("$method/seed$seed")
      sleep 5
    else
      bash "$script" --seed "$seed" "${COMMON_ARGS[@]}" "${BUDGET_ARGS[@]}" ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"} \
        || echo "!! $method seed=$seed FAILED, continuing."
    fi
  done
done

# ───────────────────────── wait ─────────────────────────────
if [[ "$RUN_MODE" == "parallel" ]]; then
  echo "Launched ${#PIDS[@]} jobs. Waiting..."
  FAILED=0
  for i in "${!PIDS[@]}"; do
    if wait "${PIDS[$i]}"; then
      echo "${LABELS[$i]} finished OK."
    else
      echo "${LABELS[$i]} FAILED."
      FAILED=$((FAILED + 1))
    fi
  done
  if (( FAILED > 0 )); then
    echo "$FAILED job(s) failed."
    exit 1
  fi
fi

echo "All runs done."
