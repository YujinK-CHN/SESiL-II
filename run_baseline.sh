#!/usr/bin/env bash
#
# Learning-based baseline: one classifier trained conventionally, for comparison
# against a SESiL evolution curve.
#
# Usage (normally via run.sh):
#   bash run_baseline.sh --seed 0
#   bash run_baseline.sh --seed 0 --baseline-classes 0,1,2,3,4,5,6,7
#   bash run_baseline.sh --seed 0 --baseline-mode finetune \
#        --baseline-classes 0,1,2,3,4,5,6,7 --finetune-classes 8,9 \
#        --baseline-load-path ./results/check/baseline_scratch/seed0/resnet20x4_v0.pth.tar

set -euo pipefail

# ──────────────── baseline arguments ────────────────
BASELINE_MODE="scratch"     # scratch | finetune
BASELINE_CLASSES=""         # empty = the whole dataset
FINETUNE_CLASSES=""
BASELINE_LOAD_PATH=""

PASSTHROUGH=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --baseline-mode)      BASELINE_MODE="$2"; shift 2;;
    --baseline-classes)   BASELINE_CLASSES="$2"; shift 2;;
    --finetune-classes)   FINETUNE_CLASSES="$2"; shift 2;;
    --baseline-load-path) BASELINE_LOAD_PATH="$2"; shift 2;;
    *)                    PASSTHROUGH+=("$1"); shift;;
  esac
done

ARGS=(--method baseline --baseline-mode "$BASELINE_MODE")
[[ -n "$BASELINE_CLASSES"   ]] && ARGS+=(--baseline-classes "$BASELINE_CLASSES")
[[ -n "$FINETUNE_CLASSES"   ]] && ARGS+=(--finetune-classes "$FINETUNE_CLASSES")
[[ -n "$BASELINE_LOAD_PATH" ]] && ARGS+=(--baseline-load-path "$BASELINE_LOAD_PATH")

echo "[run_baseline] mode=$BASELINE_MODE classes=${BASELINE_CLASSES:-all}"

python main.py "${ARGS[@]}" ${PASSTHROUGH[@]+"${PASSTHROUGH[@]}"}
