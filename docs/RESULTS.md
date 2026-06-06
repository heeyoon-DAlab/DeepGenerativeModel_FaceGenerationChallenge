# Results

All three runs use the same model, seed and training length (1,000 kimg) and
differ only in how the training frames are handled. Each is generated from its
lowest-`fid50k_full` snapshot (000960 / 000760 / 001000), 1,000 images at
truncation 1.0, scored against the full CelebV-HQ reference.

| run | data | FID | IS | KID | TopP&R |
|-----|------|-----|-----|-----|--------|
| exp1_baseline | full, ~35,610 | 37.06 | 3.81 | 0.0089 | 0.8989 |
| exp2_filtered | filtered, 12,403 | 42.99 | 4.49 | 0.0134 | 0.7718 |
| exp3_conditional | full + labels | 37.27 | 3.90 | 0.0082 | 0.8887 |

(FID and KID lower is better; IS and TopP&R higher is better.)

For reference, the best local `fid50k_full` (50k samples) was 7.64 for exp1,
10.73 for exp2 and 7.88 for exp3. The local number is much lower than the
leaderboard FID because the leaderboard only scores 1,000 images and FID is
biased upward at small sample sizes, so the two are not directly comparable. I
use the local metric only to pick a snapshot within a run, where that bias is
shared.

## Filtering hurts

Dropping dark and blurry frames (35,610 → 12,403) gives exp2 the best IS by a
clear margin, but the worst FID, KID and TopP&R. The reason is that the
evaluation target is the full, messy CelebV-HQ distribution. Three of the four
metrics reward matching that distribution, including its dark and blurry modes;
only IS is reference-free and rewards per-image sharpness/confidence regardless
of the target. Removing those frames moves the generated distribution away from
the reference (FID and KID up) and drops coverage, which shows up most clearly in
TopP&R falling from 0.90 to 0.77. The IS gain is real but it is the only metric
that benefits.

## Conditioning instead of filtering

exp3 keeps every frame and conditions on the same quality information instead of
filtering by it. It stays level with the full-data baseline on the
distribution-matching metrics (FID 37.27 vs 37.06, TopP&R 0.8887 vs 0.8989, both
far above the filtered run) and improves KID (0.0082, the best of the three) and
IS (3.90 vs 3.81) over exp1. So the per-image quality that filtering was chasing
can be recovered without the distribution and coverage damage, because all the
modes are still there, just organized by label.

It does not beat the baseline on everything, and it is not meant to. The point is
narrower: when the target distribution is messy, conditioning on frame quality is
a better way to use that information than throwing frames away.

## The proportion knob

Because exp3 is conditional, the bucket mix at generation time is a free
parameter. The numbers above use the training distribution
(`--proportions 0.3664,0.1336,0.1336,0.3664`), which matches the reference. The
two quality axes turn out to be strongly correlated on CelebV-HQ: dark frames
tend to be blurry and bright frames tend to be sharp, so the two off-diagonal
buckets are only ~13% each. Biasing the mix toward the sharp buckets should trade
some coverage for IS, which is room to push the score further without retraining.
