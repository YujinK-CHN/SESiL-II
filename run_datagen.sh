#!/usr/bin/env bash
#
# Dataset generator: a copy of run_sesil.sh that runs generate_dataset.py.
# Normally invoked via generate_dataset.sh -> submit.sh -> run.sh --method datagen.

set -euo pipefail

python generate_dataset.py --method sesil "$@"
