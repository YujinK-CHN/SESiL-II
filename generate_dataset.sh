#!/usr/bin/env bash
#
# Dataset generator: the run sesil_ssl_globa(D_minus)_globa(label)_mr6_ib150,
# with <type> (globa_with) and <head> (merge_head) free. Runs
# generate_dataset.py (datagen/evolution.py, a copy of sesil/evolution.py):
# the run evolves as usual and every generation it also measures the child of
# EVERY ordered pair (380 for pop 20). Mating mode is always globa.
#
# Output, one file per run, one row per sample:
#   <out-root>/<exp>/cifar100/<variant>/seed<N>/samples.csv
#   generation, n_generations, location (= generation / (n_generations - 1)),
#   share_A .. share_D_minus, child_val_overall, type, head
# While the run is going it is samples.partial.csv; location is filled in when
# the run ends (the total number of generations is only known then).
#
#   bash generate_dataset.sh <type> <head> <seeds> [--cluster nibi|trillium] [submit flags...]
#
#   nibi (default): one job via submit.sh, results in <repo>/results/datagen.
#     bash generate_dataset.sh D_minus label 0,1,3 --account def-xxx --time 72:00:00
#
#   trillium: submit_trillium.sh, results in $SCRATCH/SESiL-II/results/datagen,
#   and a chain of 24 h jobs (--chain, default 6) that each continue the run
#   with --resume.
#     bash generate_dataset.sh D_minus label 0,1,3 --cluster trillium --account def-zinovi
#
#   <type>  D_minus | D_plus | E
#   <head>  label | average
#
set -euo pipefail

if [[ $# -lt 3 ]]; then
  sed -n 2,28p "$0"; exit 1
fi
TYPE="$1"; HEAD="$2"; SEEDS="$3"; shift 3
case "$TYPE" in D_minus|D_plus|E) ;; *) echo "type must be D_minus, D_plus or E"; exit 1;; esac
case "$HEAD" in label|average) ;; *) echo "head must be label or average"; exit 1;; esac

HERE="$(cd "$(dirname "$0")" && pwd)"

CLUSTER="nibi"
REST=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --cluster) CLUSTER="$2"; shift 2;;
    *)         REST+=("$1"); shift;;
  esac
done

case "$CLUSTER" in
  nibi)
    SUBMIT=("$HERE/submit.sh" --out-root "$HERE/results/datagen");;
  trillium)
    # Defaults go first, so a --chain or --out-root given in REST wins.
    SUBMIT=("$HERE/submit_trillium.sh" --chain 6
            --out-root "${SCRATCH:-/scratch/$USER}/SESiL-II/results/datagen"
            --resume);;
  *) echo "cluster must be nibi or trillium"; exit 1;;
esac

bash "${SUBMIT[@]}" \
  --exp-name "sesil_ssl_globa(${TYPE})_globa(${HEAD})_mr6_ib150" \
  --dataset cifar100 --budget 400 --seeds "$SEEDS" \
  --method datagen --pretrain-mode ssl \
  --mating-mode globa --globa-with "$TYPE" --mating-rounds 6 \
  --merger globa --globa-preset globa --merge-head "$HEAD" \
  --individual-budget 1.5 --certify-top-frac 0.3 \
  --keep-generations 2 --log-predictions \
  ${REST[@]+"${REST[@]}"}
