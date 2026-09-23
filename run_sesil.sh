#!/usr/bin/env bash
#
# SESiL: pretrain a population, then evolve it.
#
# Pretrain runs automatically if the population does not exist yet.
#
# The merge operator (--merger) and mate-selection rule (--selection) are
# ARGUMENTS to SESiL, not separate methods. Their defaults -- and every other
# hyper-parameter -- live in config.py. Nothing is duplicated here.
#
# Normally invoked via run.sh:
#   bash run.sh --dataset cifar10 --budget 25 --seeds 0
#
# Directly, for a one-off:
#   bash run_sesil.sh --dataset cifar10 --budget 25 --seed 0 --merger zipit

set -euo pipefail

python main.py --method sesil "$@"
