set -euo pipefail
source "$(dirname "$0")/common.sh"
cd "$SG3"
CUDA_VISIBLE_DEVICES=$GPU python train.py \
  --outdir="$RUNS/exp1_baseline" \
  --cfg=stylegan3-r \
  --data="$DATA_BASELINE" \
  --gpus=1 --batch=32 --gamma=2 --cbase=16384 \
  --resume="$PRETRAINED" \
  --seed=$SEED \
  --snap=10 --metrics=fid50k_full --kimg=1000
