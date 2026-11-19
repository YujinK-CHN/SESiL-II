#!/usr/bin/env bash
#
# Dataset generator: the run sesil_ssl_globa(D_minus)_globa(label)_mr6_ib150,
# with <type> (globa_with) and <head> (merge_head) free. Runs
# generate_dataset.py (datagen/evolution.py, a copy of sesil/evolution.py):
# the run evolves as usual and every generation it also measures the child of
# EVERY ordered pair (380 for pop 20). Mating mode is always globa.
#
# Output, one file per run, one row per sample:
#   results/datagen/<exp>/cifar100/<variant>/seed<N>/samples.csv
#   generation, n_generations, location (= generation / (n_generations - 1)),
#   share_A .. share_D_minus, child_val_overall, type, head
# While the run is going it is samples.partial.csv; location is filled in when
# the run ends (the total number of generations is only known then).
#
#   bash generate_dataset.sh <type> <head> <seeds> [submit.sh flags...]
#   bash generate_dataset.sh D_minus label 0,1,3 --account def-xxx --time 48:00:00
#
#   <type>  D_minus | D_plus | E
#   <head>  label | average
#
set -euo pipefail

if [[ $# -lt 3 ]]; then
  sed -n 2,21p "$0"; exit 1
fi
TYPE="$1"; HEAD="$2"; SEEDS="$3"; shift 3
case "$TYPE" in D_minus|D_plus|E) ;; *) echo "type must be D_minus, D_plus or E"; exit 1;; esac
case "$HEAD" in label|average) ;; *) echo "head must be label or average"; exit 1;; esac

HERE="$(cd "$(dirname "$0")" && pwd)"

bash "$HERE/submit.sh" \
  --exp-name "sesil_ssl_globa(${TYPE})_globa(${HEAD})_mr6_ib150" \
  --out-root "$HERE/results/datagen" \
  --dataset cifar100 --budget 400 --seeds "$SEEDS" \
  --method datagen --pretrain-mode ssl \
  --mating-mode globa --globa-with "$TYPE" --mating-rounds 6 \
  --merger globa --globa-preset globa --merge-head "$HEAD" \
  --individual-budget 1.5 --certify-top-frac 0.3 \
  --keep-generations 2 --log-predictions \
  "$@"
