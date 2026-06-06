#!/usr/bin/env bash
# exp3: full data + brightness/sharpness bucket labels, conditional (--cond=1).
# Needs data/celebvhq256_conditional.zip (build it with make_conditional_zip.py)
# and the conditional-resume patch (run setup.sh once). Best snapshot was 001000.
set -euo pipefail
source "$(dirname "$0")/common.sh"
cd "$SG3"
CUDA_VISIBLE_DEVICES=$GPU python train.py \
  --outdir="$RUNS/exp3_conditional" \
  --cfg=stylegan3-r \
  --data="$DATA_CONDITIONAL" \
  --cond=1 \
  --gpus=1 --batch=32 --gamma=2 --cbase=16384 \
  --resume="$PRETRAINED" \
  --seed=$SEED \
  --snap=10 --metrics=fid50k_full --kimg=1000
