#!/usr/bin/env bash
#
# Mate-screening probe: can GLOBA predict which pairs merge well?
#
# This is NOT a learning method. It builds one population, then exhaustively
# merges every possible couple and records, for each, what GLOBA predicted from
# the weights alone against what the merge actually produced. There are no
# generations and no accuracy curve; the output is pairs.jsonl, read by
# analyze_probe.py rather than plot_results.py.
#
# Children are built by the configured --merger (zipit / permute / wavg), the
# same way SESiL would build them. GLOBA only ever supplies a prediction.
#
# Requires --pretrain-mode ssl: a task vector is agent - core, and the core is
# the phase-A backbone. The driver refuses to run without one.
#
# --budget only sizes pretrain here. The probe does not train, so its real cost
# scales with --pop-size squared.
#
#   bash run_probe.sh --dataset cifar10 --budget 10 --seed 0
#   bash run_probe.sh --dataset cifar10 --budget 10 --seed 0 --merger permute

set -euo pipefail

python main.py --method probe --pretrain-mode ssl "$@"
