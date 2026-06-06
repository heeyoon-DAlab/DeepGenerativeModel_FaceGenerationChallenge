#!/usr/bin/env python3
import os, argparse, math
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

EXTS = (".png", ".jpg", ".jpeg")


def load_montage(folder, n, grid):
    files = sorted(f for f in os.listdir(folder) if f.lower().endswith(EXTS))[:n]
    if len(files) < n:
        raise SystemExit(f"{folder}: found {len(files)} images, need {n}")
    imgs = [np.asarray(Image.open(os.path.join(folder, f)).convert("RGB")) for f in files]
    h, w, _ = imgs[0].shape
    canvas = np.full((grid * h, grid * w, 3), 255, np.uint8)
    for i, im in enumerate(imgs):
        r, c = divmod(i, grid)
        canvas[r * h:(r + 1) * h, c * w:(c + 1) * w] = im
    return canvas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outputs", default="outputs")
    ap.add_argument("--out", default="figures/comparison_grid.png")
    ap.add_argument("--n", type=int, default=16)
    args = ap.parse_args()

    grid = int(math.isqrt(args.n))
    if grid * grid != args.n:
        raise SystemExit("--n must be a perfect square (e.g. 9, 16, 25)")

    exps = [
        ("exp1_baseline",    "exp1: full (unconditional)"),
        ("exp2_filtered",    "exp2: filtered (unconditional)"),
        ("exp3_conditional", "exp3: conditional"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.4))
    for ax, (d, title) in zip(axes, exps):
        folder = os.path.join(args.outputs, d, "images")
        ax.imshow(load_montage(folder, args.n, grid))
        ax.set_title(title, fontsize=13)
        ax.axis("off")
    plt.tight_layout()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    plt.savefig(args.out, dpi=200, bbox_inches="tight")
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
