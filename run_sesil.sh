#!/usr/bin/env bash
#
# SESiL: pretrain a population, then evolve it.
#
# Pretrain runs automatically if the population does not exist yet.
#
# The merge operator (--merger) is an ARGUMENT to SESiL, not a separate method.
# Its default -- and every other hyper-parameter -- lives in config.py. Nothing
# is duplicated here.
#
# Normally invoked via run.sh:
#   bash run.sh --dataset cifar10 --budget 25 --seeds 0
#
# Directly, for a one-off:
#   bash run_sesil.sh --dataset cifar10 --budget 500 --seed 0 --merger zipit

set -euo pipefail

python main.py --method sesil "$@"
