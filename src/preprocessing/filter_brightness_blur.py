#!/usr/bin/env python3
import argparse, io, zipfile, sys
import numpy as np
from PIL import Image
from image_quality import brightness, sharpness

IMG_EXT = ('.png', '.jpg', '.jpeg')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--in',  dest='inp', required=True)
    ap.add_argument('--out', dest='out', required=True)
    ap.add_argument('--min_bright', type=float, required=True, help='drop frames with brightness < this')
    ap.add_argument('--min_sharp',  type=float, required=True, help='drop frames with sharpness  < this')
    args = ap.parse_args()

    zin = zipfile.ZipFile(args.inp, 'r')
    names = [n for n in zin.namelist() if n.lower().endswith(IMG_EXT)]
    if not names:
        sys.exit('no image inside zip.')
    print(f' {len(names)}images examining...')

    keep = []
    for i, n in enumerate(names):
        arr = np.asarray(Image.open(io.BytesIO(zin.read(n))).convert('RGB'), dtype=np.float64)
        if brightness(arr) >= args.min_bright and sharpness(arr) >= args.min_sharp:
            keep.append(n)
        if (i + 1) % 2000 == 0:
            print(f'  {i+1}/{len(names)}')

    print(f'remain {len(keep)} / {len(names)}  ({100*len(keep)/len(names):.1f}%)  '
          f'[min_bright={args.min_bright}, min_sharp={args.min_sharp}]')

    with zipfile.ZipFile(args.out, 'w', zipfile.ZIP_STORED) as zout:
        for n in keep:
            zout.writestr(n, zin.read(n))
    zin.close()
    print(f'Done: {args.out}  ({len(keep)} images)')


if __name__ == '__main__':
    main()
