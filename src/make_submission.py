#!/usr/bin/env python3
"""
make_submission.py — package generated images into the submission zip.

  - converts each image to RGB JPEG (quality 95)
  - renames to img_0000.jpg ... img_NNNN.jpg, FLAT at the zip root
  - asserts exactly --count images and final zip size < --max-mb

Usage:
  python src/make_submission.py \
    --images outputs/exp1_baseline/images \
    --out    outputs/exp1_baseline/submission_exp1_baseline.zip
"""
import argparse, io, os, sys, glob, zipfile
import PIL.Image


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--images', required=True, help='folder of generated images')
    ap.add_argument('--out', required=True, help='output .zip path')
    ap.add_argument('--count', type=int, default=1000)
    ap.add_argument('--quality', type=int, default=95)
    ap.add_argument('--max-mb', dest='max_mb', type=float, default=200.0)
    args = ap.parse_args()

    files = sorted(
        glob.glob(os.path.join(args.images, '*.png'))
        + glob.glob(os.path.join(args.images, '*.jpg'))
        + glob.glob(os.path.join(args.images, '*.jpeg'))
    )
    if len(files) != args.count:
        sys.exit(f'expected {args.count} images, found {len(files)} in {args.images}')

    out_dir = os.path.dirname(os.path.abspath(args.out))
    os.makedirs(out_dir, exist_ok=True)

    with zipfile.ZipFile(args.out, 'w', zipfile.ZIP_DEFLATED) as z:
        for i, fp in enumerate(files):
            im = PIL.Image.open(fp).convert('RGB')
            buf = io.BytesIO()
            im.save(buf, 'JPEG', quality=args.quality)
            z.writestr(f'img_{i:04d}.jpg', buf.getvalue())  # flat -> zip root

    size_mb = os.path.getsize(args.out) / 1e6
    if size_mb >= args.max_mb:
        sys.exit(f'zip is {size_mb:.1f} MB (>= {args.max_mb} MB limit). lower --quality.')
    print(f'OK: {args.out}  ({args.count} images, {size_mb:.1f} MB)')


if __name__ == '__main__':
    main()
