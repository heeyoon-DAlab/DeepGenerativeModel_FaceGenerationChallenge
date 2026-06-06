# conditional_resume.patch

This patches `copy_params_and_buffers` in `stylegan3/torch_utils/misc.py`.

exp3 trains a conditional model (`--cond=1`). That adds label layers that the
unconditional FFHQ-U checkpoint does not have: the generator's first mapping
layer changes shape and the discriminator gets a label projection. Stock
StyleGAN3 raises on the shape mismatch when you try to resume. The patch copies
every tensor whose shape matches (so the whole synthesis network and all the
learned image content transfers) and leaves only the label-specific layers
randomly initialized, printing each one it skips:

```
[resume] shape mismatch -> random init: mapping.fc0.weight (... vs ...)
```

It is a no-op for exp1 and exp2, since unconditional resume has no mismatch, so
it is fine to apply once and use it for all three runs.

`setup.sh` applies it for you. To do it by hand:

```bash
( cd stylegan3 && git apply ../patches/conditional_resume.patch )
```

If it does not apply cleanly against your checkout, open
`stylegan3/torch_utils/misc.py`, find `copy_params_and_buffers`, and change the
copy so it only runs when the shapes match, with an `elif` that prints the
skipped tensor. The diff in `conditional_resume.patch` is the exact change.
