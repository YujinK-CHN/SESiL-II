#!/usr/bin/env bash
#
# Master launcher.
#
# Three knobs, because only three things change between runs of a suite:
#
#   --dataset   which environment   (cifar10 | cifar100)
#   --budget    training budget in EPOCH-EQUIVALENTS -- one unit is a single
#               backprop pass over the full training set. This is the common
#               currency that makes SESiL and the baseline comparable: at the
#               same --budget both methods get the same training compute.
#               SESiL spends it on pretrain + per-generation mutation and stops
#               when it runs out; the baseline spends it as epochs.
#   --seeds     comma-separated seeds
#
# Everything else -- merger, selection, population shape, hyper-parameters --
# lives in config.py. Edit it there.
#
# Usage:
#   bash run.sh --dataset cifar10  --budget 500 --seeds 0
#   bash run.sh --dataset cifar100 --budget 500 --seeds 0,1,2
#
#   # same compute, for a fair comparison:
#   bash run.sh --dataset cifar10 --budget 500 --seeds 0 --method sesil
#   bash run.sh --dataset cifar10 --budget 500 --seeds 0 --method baseline
#
# Any unrecognised flag is passed through to config.py, so a one-off override is
# still possible without editing anything:
#   bash run.sh --dataset cifar10 --budget 500 --seeds 0 --merger zipit

set -euo pipefail

# ───────────────────────── the three knobs ──────────────────────────
DATASET="cifar10"
BUDGET=500
SEEDS="0"

# ───────────────────────── run control ──────────────────────────────
METHODS=(sesil)            # sesil | baseline  (override with --method)
RUN_MODE="sequential"      # sequential | parallel
EXP_NAME="check"

EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset)   DATASET="$2";   shift 2;;
    --budget)    BUDGET="$2";    shift 2;;
    --seeds)     SEEDS="$2";     shift 2;;
    --method)    METHODS=("$2"); shift 2;;
    --run-mode)  RUN_MODE="$2";  shift 2;;
    --exp-name)  EXP_NAME="$2";  shift 2;;
    *)           EXTRA_ARGS+=("$1"); shift;;
  esac
done

IFS=',' read -ra SEED_LIST <<< "$SEEDS"

echo "dataset : $DATASET"
echo "budget  : $BUDGET"
echo "seeds   : ${SEED_LIST[*]}"
echo "methods : ${METHODS[*]}"
echo "mode    : $RUN_MODE"

COMMON_ARGS=(--dataset "$DATASET" --budget "$BUDGET" --exp-name "$EXP_NAME")

# ───────────────────────── dispatch ─────────────────────────────────
PIDS=()
LABELS=()

for method in "${METHODS[@]}"; do
  script="run_${method}.sh"
  if [[ ! -f "$script" ]]; then
    echo "No launcher $script for method '$method'." >&2
    exit 1
  fi

  for seed in "${SEED_LIST[@]}"; do
    echo ">> $method  seed=$seed"
    if [[ "$RUN_MODE" == "parallel" ]]; then
      bash "$script" --seed "$seed" "${COMMON_ARGS[@]}" ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"} &
      PIDS+=($!)
      LABELS+=("$method/seed$seed")
      sleep 5
    else
      bash "$script" --seed "$seed" "${COMMON_ARGS[@]}" ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"} \
        || echo "!! $method seed=$seed FAILED, continuing."
    fi
  done
done

# ───────────────────────── wait ─────────────────────────────────────
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
