#!/usr/bin/env bash
#
# SESiL: pretrain a population, then evolve it.
#
# The merge operator and the mate-selection rule are ARGUMENTS to SESiL, not
# separate methods -- set them below or override from the command line.
#
# Usage (normally via run.sh):
#   bash run_sesil.sh --seed 0
#   bash run_sesil.sh --seed 0 --merger zipit --selection guided
#
# Pretrain runs automatically if the population does not exist yet.

set -euo pipefail

# ──────────────── SESiL arguments ────────────────
MERGER="permute"            # zipit | permute | wavg
SELECTION="bidirectional"   # bidirectional | breed | guided | hard

# Mutation (the per-generation training budget)
MUTATE_EPOCHS=2

# Merge operator hyper-parameters (ZipIt! alpha/beta, partial-zipping depth)
STOP_NODE=21
MERGE_ALPHA=0.0001
MERGE_BETA=0.075

# Mate-selection hyper-parameters
TAU=0.5                     # accuracy above which a class counts as "known"
WEIGHT_EXTRA=1.0            # value of skills the mate has and you lack
WEIGHT_COMMON=0.1           # value of skills you already share

PASSTHROUGH=()

# run.sh forwards --merger/--selection/etc; catch them so they override the
# defaults above instead of being passed twice.
while [[ $# -gt 0 ]]; do
  case "$1" in
    --merger)        MERGER="$2"; shift 2;;
    --selection)     SELECTION="$2"; shift 2;;
    --mutate-epochs) MUTATE_EPOCHS="$2"; shift 2;;
    --stop-node)     STOP_NODE="$2"; shift 2;;
    --merge-alpha)   MERGE_ALPHA="$2"; shift 2;;
    --merge-beta)    MERGE_BETA="$2"; shift 2;;
    --tau)           TAU="$2"; shift 2;;
    --weight-extra)  WEIGHT_EXTRA="$2"; shift 2;;
    --weight-common) WEIGHT_COMMON="$2"; shift 2;;
    *)               PASSTHROUGH+=("$1"); shift;;
  esac
done

echo "[run_sesil] merger=$MERGER selection=$SELECTION mutate_epochs=$MUTATE_EPOCHS"

python main.py \
  --method sesil \
  --merger "$MERGER" \
  --selection "$SELECTION" \
  --mutate-epochs "$MUTATE_EPOCHS" \
  --stop-node "$STOP_NODE" \
  --merge-alpha "$MERGE_ALPHA" \
  --merge-beta "$MERGE_BETA" \
  --tau "$TAU" \
  --weight-extra "$WEIGHT_EXTRA" \
  --weight-common "$WEIGHT_COMMON" \
  ${PASSTHROUGH[@]+"${PASSTHROUGH[@]}"}
