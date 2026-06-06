# StyleGAN3 fine-tuning on CelebV-HQ

Fine-tuning the FFHQ-U pretrained StyleGAN3-R generator on CelebV-HQ video frames
for the DGM face-generation challenge. The evaluation generates 1,000 images and
scores them against the 32,550-image CelebV-HQ reference using FID, IS, KID and
TopP&R, averaged into one rank.

The whole thing is a plain fine-tune. No architecture or loss changes. The only
modification to the upstream StyleGAN3 code is a small patch that lets a
conditional model resume from the unconditional FFHQ-U checkpoint (see
`patches/`). What I actually study here is how the *quality* of the training
frames (brightness, sharpness) should be handled, which is what the three
experiments below are about.

## Experiments

| name | data | model | notes |
|------|------|-------|-------|
| exp1_baseline | full, ~35,610 frames | StyleGAN3-R, unconditional | the straightforward baseline |
| exp2_filtered | filtered, 12,403 frames | StyleGAN3-R, unconditional | drop dark/blurry frames |
| exp3_conditional | full + quality labels | StyleGAN3-R, conditional | keep everything, condition on quality instead |

exp1 and exp3 train on the same full set of frames; exp3 just adds a
brightness×sharpness bucket label per frame. exp2 is the smaller filtered set.
Results and the short writeup of what came out of this are in `docs/RESULTS.md`,
and the full report is in `report/`.

## Setup

```bash
conda env create -f environment.yml
conda activate stylegan3
```

You also need the StyleGAN3 source. I trained against my own working copy and
keep it at `stylegan3/` in the repo root so the results come from the exact code
I ran. If you are setting this up fresh, either drop that same `stylegan3/`
directory in here, or clone NVlabs/stylegan3 and apply the patch:

```bash
git clone https://github.com/NVlabs/stylegan3
( cd stylegan3 && git apply ../patches/conditional_resume.patch )
```

Then run the one-time setup, which applies the patch (if not already applied),
fetches the FFHQ-U checkpoint into `models/`, and makes the working dirs:

```bash
bash setup.sh
```

## Data

Frames are extracted from CelebV-HQ videos to 256×256 and packed into zips that
sit under `data/` (not in the repo; too large). The training scripts expect:

- `data/celebvhq256_baseline.zip` — full set, used by exp1 and exp3
- `data/celebvhq256_filtered.zip` — filtered set (12,403), used by exp2
- `data/celebvhq256_conditional.zip` — full set + labels, used by exp3

The filtered and conditional zips are built from the baseline zip:

```bash
# exp2 filtered set (provenance; the experiment uses the existing zip)
python src/preprocessing/filter_brightness_blur.py \
  --in data/celebvhq256_baseline.zip --out data/celebvhq256_filtered.zip \
  --min_bright <B> --min_sharp <S>

# exp3 conditional set (adds dataset.json with bucket labels; pixels untouched)
python src/preprocessing/make_conditional_zip.py \
  --in data/celebvhq256_baseline.zip --out data/celebvhq256_conditional.zip
```

`make_conditional_zip.py` prints the realized bucket proportions at the end; you
pass those to `generate.py` for exp3 (see below).

## Training

One GPU per run, 1,000 kimg each, seed 0. Long runs, so background them.

```bash
GPU=0 bash experiments/train_exp1_baseline.sh
GPU=0 bash experiments/train_exp2_filtered.sh
GPU=0 bash experiments/train_exp3_conditional.sh
```

`fid50k_full` is logged every 40 kimg; pick the snapshot with the lowest value.
For my runs that was 000960 (exp1), 000760 (exp2) and 001000 (exp3).

## Generating the submission images

This is the part that has to be reproducible. Generation is seeded, so the same
weights and the same command give the same 1,000 images. The weights are
attached to the GitHub release (they are too big for the repo). Download the
three snapshots, then:

```bash
# exp1
python src/generate.py \
  --network models/exp1_network-snapshot-000960.pkl \
  --outdir outputs/exp1/images --seeds 0-999 --trunc 1.0
python src/make_submission.py \
  --images outputs/exp1/images --out outputs/exp1/submission_exp1.zip

# exp2
python src/generate.py \
  --network models/exp2_network-snapshot-000760.pkl \
  --outdir outputs/exp2/images --seeds 0-999 --trunc 1.0
python src/make_submission.py \
  --images outputs/exp2/images --out outputs/exp2/submission_exp2.zip

# exp3 (conditional: also fix the per-image bucket sampling)
python src/generate.py \
  --network models/exp3_network-snapshot-001000.pkl \
  --outdir outputs/exp3/images --seeds 0-999 --trunc 1.0 \
  --proportions 0.3664,0.1336,0.1336,0.3664 --label-seed 0
python src/make_submission.py \
  --images outputs/exp3/images --out outputs/exp3/submission_exp3.zip
```

`generate.py` needs the StyleGAN3 code on the path; either run it from a shell
where `stylegan3/` is importable or prefix with `PYTHONPATH=stylegan3`. For exp3,
the run prints the bucket counts it sampled, which is a quick check that the
labels came out right. With `--label-seed 0` and the proportions above the draw
is fixed at 376 / 141 / 125 / 358 (dark+blurry / dark+sharp / bright+blurry /
bright+sharp); the sampler uses numpy's legacy RandomState, so the same seed
gives those exact counts regardless of numpy version.

## Reproducibility notes

- Seeds: training uses `--seed 0`; generation uses `--seeds 0-999`. For the
  conditional model the per-image bucket is drawn from `RandomState(--label-seed)`
  with `--label-seed 0` (the default), so the `(z, label)` pairs are fixed.
- The exact `--proportions` for exp3 are `0.3664,0.1336,0.1336,0.3664` (the
  training-set bucket distribution printed by `make_conditional_zip.py`).
- Environment is pinned in `environment.yml` / `requirements.txt`. A couple of
  things matter: `face-alignment==1.3.5` (1.5.0 broke), `ninja` (needed to build
  the StyleGAN3 CUDA ops), and the cu113 PyTorch build.
- The trained weights used for each submission are in the release. Keep them;
  the images are regenerated from those exact pickles.
- One caveat: results are deterministic given the weights and seed, but pixel
  values are not guaranteed bit-identical across different GPUs / driver versions
  because of floating-point nondeterminism. The released weights are the source
  of truth.

## Acknowledgements

Built on NVIDIA's StyleGAN3 (https://github.com/NVlabs/stylegan3), which keeps
its own license under `stylegan3/`. Pretrained checkpoint: `stylegan3-r-ffhqu-256x256`.
TopP&R follows Kim et al., NeurIPS 2023. Data: CelebV-HQ (Zhu et al., ECCV 2022).
