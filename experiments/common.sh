#!/usr/bin/env bash
# Shared paths/config for the train scripts. Edit here, not in each script.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export ROOT

# Force CUDA device numbering to match nvidia-smi (PCI bus order). Without this
# torch can enumerate the GPUs in the opposite order, which is how you end up
# launching two runs onto the same card.
export CUDA_DEVICE_ORDER=PCI_BUS_ID

export SG3="$ROOT/stylegan3"
export PRETRAINED="$ROOT/models/stylegan3-r-ffhqu-256x256.pkl"
export RUNS="$ROOT/runs"
export OUTPUTS="$ROOT/outputs"

export DATA_BASELINE="$ROOT/data/celebvhq256_baseline.zip"        # full ~35,610   (exp1, exp3)
export DATA_FILTERED="$ROOT/data/celebvhq256_filtered.zip"        # filtered 12,403 (exp2)
export DATA_CONDITIONAL="$ROOT/data/celebvhq256_conditional.zip"  # full + labels   (exp3)

export SEED=0

# GPU to use. Set on the command line to pick a card, e.g. `GPU=1 bash train_*.sh`.
# With PCI_BUS_ID above: 0 = A5000, 1 = 3090 on this machine.
export GPU=${GPU:-0}
