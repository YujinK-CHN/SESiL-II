#!/usr/bin/env bash
#
# Learning-based baseline: one classifier trained conventionally, for comparison
# against a SESiL evolution curve. --budget is spent as epochs.
#
# Mode, class subsets and everything else default from config.py.
#
# Normally invoked via run.sh:
#   bash run.sh --dataset cifar10 --budget 100 --seeds 0 --method baseline
#
# Directly, for a one-off:
#   bash run_baseline.sh --dataset cifar10 --budget 100 --seed 0 \
#        --baseline-mode finetune --baseline-classes 0,1,2,3,4,5,6,7 \
#        --finetune-classes 8,9 --baseline-load-path <ckpt>

set -euo pipefail

python main.py --method baseline "$@"
