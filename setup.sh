#!/usr/bin/env bash
# One-time local setup: apply the exp3 patch, fetch the pretrained checkpoint,
# and make the working directories.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

mkdir -p models runs outputs

# stylegan3 training code must already be here (moved from your old layout)
if [ ! -f stylegan3/train.py ]; then
  echo "ERROR: stylegan3/train.py not found."
  echo "  Move your existing stylegan3 code into $ROOT/stylegan3 first (this is the"
  echo "  exact code that produced your results — do not re-clone)."
  exit 1
fi

# conditional-resume patch (needed only for exp3; harmless for exp1/exp2)
if grep -q "shape mismatch -> random init" stylegan3/torch_utils/misc.py; then
  echo "[setup] patch already applied"
else
  if ( cd stylegan3 && { git apply "$ROOT/patches/conditional_resume.patch" 2>/dev/null \
        || patch -p1 < "$ROOT/patches/conditional_resume.patch" ; } ); then
    echo "[setup] conditional_resume.patch applied"
  else
    echo "[setup] WARNING: patch did not apply cleanly — apply manually (patches/README.md)"
  fi
fi

# FFHQ-U 256 pretrained (resume base for every experiment)
PKL="models/stylegan3-r-ffhqu-256x256.pkl"
if [ -f "$PKL" ]; then
  echo "[setup] pretrained present: $PKL"
else
  echo "[setup] downloading FFHQ-U 256 pretrained ..."
  curl -L -o "$PKL" \
    "https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/stylegan3-r-ffhqu-256x256.pkl"
fi

echo "[setup] done. Create the env, then run experiments/train_exp1_baseline.sh etc."
