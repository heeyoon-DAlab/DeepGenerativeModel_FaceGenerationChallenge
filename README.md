# StyleGAN3 Fine-tuning on CelebV-HQ

This repository provides the implementation for fine-tuning the FFHQ-U pretrained StyleGAN3-R generator on CelebV-HQ video frames for the DGM face-generation challenge. The project demonstrates an efficient conditional generation approach to maintain high diversity and quality without discarding training data.

## 1. Setup

Follow these steps to prepare your environment and codebase:

```bash
# Clone the repository
git clone [https://github.com/heeyoon-DAlab/DeepGenerativeModel_FaceGenerationChallenge](https://github.com/heeyoon-DAlab/DeepGenerativeModel_FaceGenerationChallenge)
cd DeepGenerativeModel_FaceGenerationChallenge

# Setup environment
conda env create -f environment.yml
conda activate stylegan3

# Prepare StyleGAN3 source
git clone [https://github.com/NVlabs/stylegan3](https://github.com/NVlabs/stylegan3)
( cd stylegan3 && git apply ../patches/conditional_resume.patch )

# Create directory for model weights
mkdir -p models
```
---

## 2. Inference
Download the network-snapshot-XXXXXX.pkl files from the GitHub Release page and place them into the models/ directory.

```bash
python src/generate.py --network models/exp1_network-snapshot-000960.pkl \
  --outdir outputs/exp1/images --seeds 0-999 --trunc 1.0
python src/make_submission.py --images outputs/exp1/images --out submission_exp1.zip

python src/generate.py --network models/exp2_network-snapshot-000760.pkl \
  --outdir outputs/exp2/images --seeds 0-999 --trunc 1.0
python src/make_submission.py --images outputs/exp2/images --out submission_exp2.zip

python src/generate.py --network models/exp3_network-snapshot-001000.pkl \
  --outdir outputs/exp3/images --seeds 0-999 --trunc 1.0 \
  --proportions 0.3664,0.1336,0.1336,0.3664 --label-seed 0
python src/make_submission.py --images outputs/exp3/images --out submission_exp3.zip
```
---

## 3. Reproducibility Notes
Weights: To ensure reproducibility, you must use the exact .pkl weights provided in the GitHub Release.

Determinism: The generation process is fixed via seeds and weights. Ensure generate.py is executed from an environment where the stylegan3/ directory is importable.

Conditional Sampling: For exp3, the --proportions and --label-seed parameters are critical to match the training-set bucket distribution.

---
