ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export ROOT

export CUDA_DEVICE_ORDER=PCI_BUS_ID

export SG3="$ROOT/stylegan3"
export PRETRAINED="$ROOT/models/stylegan3-r-ffhqu-256x256.pkl"
export RUNS="$ROOT/runs"
export OUTPUTS="$ROOT/outputs"

export DATA_BASELINE="$ROOT/data/celebvhq256_baseline.zip"        
export DATA_FILTERED="$ROOT/data/celebvhq256_filtered.zip"        
export DATA_CONDITIONAL="$ROOT/data/celebvhq256_conditional.zip"  

export SEED=0

export GPU=${GPU:-0}
