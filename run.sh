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
# For several seeds at once:
#   bash run.sh --seeds 0,1,2 --run-mode parallel --gpus 0,1,2
# Each run claims the least-loaded of the listed GPUs (lock files under
# results/.gpu_locks), so seeds spread across devices instead of all landing on
# cuda:0. Without --gpus they use every visible GPU.
#
# Everything else -- merger, certification, population shape, hyper-parameters --
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
# PAIRED SWEEP (curriculum baseline). Run SESiL first, then point the baseline
# at the PARENT of its seed directories -- not at one seed -- so each baseline
# seed replays the SESiL seed of the same number:
#
#   bash run.sh --exp-name main --dataset cifar100 --budget 400 --seeds 0,1,2 #        --method sesil
#   bash run.sh --exp-name main --dataset cifar100 --budget 400 --seeds 0,1,2 #        --method baseline --baseline-mode curriculum #        --curriculum-from results/main/cifar100/permute
#
# Every unrecognised flag is passed through to ALL seeds unchanged, which is
# why the parent form matters: a fixed .../seed0 path would make every baseline
# replay seed 0 and write identical curves under different seed labels. config.py
# refuses that outright rather than letting it through.
#
# Any unrecognised flag is passed through to config.py, so a one-off override is
# still possible without editing anything:
#   bash run.sh --dataset cifar10 --budget 500 --seeds 0 --merger zipit

set -euo pipefail

# ───────────────────────── the three knobs ──────────────────────────
DATASET="cifar10"
BUDGET=400
SEEDS="0"

# ───────────────────────── run control ──────────────────────────────
METHODS=(sesil)            # sesil | baseline  (override with --method)
RUN_MODE="sequential"      # sequential | parallel
EXP_NAME="check"
GPUS=""                    # e.g. "0,1,2"; empty = every visible GPU

EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset)   DATASET="$2";   shift 2;;
    --budget)    BUDGET="$2";    shift 2;;
    --seeds)     SEEDS="$2";     shift 2;;
    --method)    METHODS=("$2"); shift 2;;
    --run-mode)  RUN_MODE="$2";  shift 2;;
    --exp-name)  EXP_NAME="$2";  shift 2;;
    --gpus)      GPUS="$2";      shift 2;;
    *)           EXTRA_ARGS+=("$1"); shift;;
  esac
done

IFS=',' read -ra SEED_LIST <<< "$SEEDS"

echo "dataset : $DATASET"
echo "budget  : $BUDGET"
echo "seeds   : ${SEED_LIST[*]}"
echo "methods : ${METHODS[*]}"
echo "mode    : $RUN_MODE"
IFS=',' read -ra GPU_LIST <<< "$GPUS"

# Does the caller pin the device themselves? Then leave it alone.
EXPLICIT_DEVICE=0
for a in ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}; do
  [[ "$a" == "--device" ]] && EXPLICIT_DEVICE=1
done

# PINNED MODE. When --seeds and --gpus are the same length, seed i goes to
# gpu i, by position, and nothing is negotiated at run time.
#
# Worth having over the claim-a-free-card scheme for two reasons. The mapping
# is knowable before launch, so "seed 3 died" tells you which card to look at.
# And --device is then explicit, which makes sesil/gpu.py skip claim_gpu()
# entirely -- including the free_gb() sweep that reads every allowed card's
# memory and, as a side effect of asking, leaves a ~400 MiB CUDA context on
# each one. Five runs over four cards stranded about 2 GiB per card that way.
#
# Any other combination keeps the old behaviour: each run claims the
# least-loaded allowed card, which is what you want when the counts do not
# line up.
PIN_GPUS=0
if [[ -n "$GPUS" && ${#GPU_LIST[@]} -eq ${#SEED_LIST[@]} && $EXPLICIT_DEVICE -eq 0 ]]; then
  PIN_GPUS=1
fi

if [[ -n "$GPUS" ]]; then
  if [[ $PIN_GPUS -eq 1 ]]; then
    MAP=""
    for i in "${!SEED_LIST[@]}"; do
      MAP+="seed${SEED_LIST[$i]}->cuda:${GPU_LIST[$i]} "
    done
    echo "gpus    : $GPUS  (pinned one-to-one: $MAP)"
  else
    # Read by sesil/gpu.py when --device is not given explicitly.
    export SESIL_GPUS="$GPUS"
    echo "gpus    : $GPUS  (${#GPU_LIST[@]} gpu(s) for ${#SEED_LIST[@]} seed(s)"
    echo "          -> not pinned; each run claims the least-loaded allowed card)"
  fi
else
  echo "gpus    : all visible"
fi

# In parallel mode every seed launches at once, so more seeds than GPUs means
# they stack. sesil/gpu.py refuses past --max-per-gpu, but it does so one run
# at a time and several minutes in; saying it here costs nothing and stops the
# whole sweep before any of it starts.
if [[ "$RUN_MODE" == "parallel" && -n "$GPUS" && $PIN_GPUS -eq 0 ]]; then
  N_GPUS=$(awk -F, '{print NF}' <<< "$GPUS")
  N_JOBS=$(( ${#SEED_LIST[@]} * ${#METHODS[@]} ))
  if (( N_JOBS > N_GPUS )); then
    echo "!! $N_JOBS parallel job(s) but only $N_GPUS GPU(s) in --gpus." >&2
    echo "   Runs past the first per GPU will stop with a --max-per-gpu error." >&2
    echo "   Use --run-mode sequential, fewer seeds, or raise --max-per-gpu." >&2
  fi
fi

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

  for i in "${!SEED_LIST[@]}"; do
    seed="${SEED_LIST[$i]}"

    # Pinned mode passes the device outright; otherwise nothing is added and
    # sesil/gpu.py claims a card as before.
    DEVICE_ARGS=()
    if [[ $PIN_GPUS -eq 1 ]]; then
      DEVICE_ARGS=(--device "cuda:${GPU_LIST[$i]}")
      echo ">> $method  seed=$seed  on cuda:${GPU_LIST[$i]}"
    else
      echo ">> $method  seed=$seed"
    fi

    if [[ "$RUN_MODE" == "parallel" ]]; then
      bash "$script" --seed "$seed" "${COMMON_ARGS[@]}" \
        ${DEVICE_ARGS[@]+"${DEVICE_ARGS[@]}"} \
        ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"} &
      PIDS+=($!)
      LABELS+=("$method/seed$seed")
      sleep 5
    else
      bash "$script" --seed "$seed" "${COMMON_ARGS[@]}" \
        ${DEVICE_ARGS[@]+"${DEVICE_ARGS[@]}"} \
        ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"} \
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
