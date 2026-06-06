#!/usr/bin/env python3
import argparse, io, json, zipfile, sys
import numpy as np
from PIL import Image
from image_quality import brightness, sharpness  

IMG_EXT = ('.png', '.jpg', '.jpeg')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--in',  dest='inp', required=True)
    ap.add_argument('--out', dest='out', required=True)
    ap.add_argument('--bright_thr', default='auto')
    ap.add_argument('--sharp_thr',  default='auto')
    args = ap.parse_args()

    zin = zipfile.ZipFile(args.inp, 'r')
    names = [n for n in zin.namelist() if n.lower().endswith(IMG_EXT)]
    if not names:
        sys.exit('No images found in zip. Check file structure.')
    print(f'Measuring {len(names)} images...')

    b_vals, s_vals = np.empty(len(names)), np.empty(len(names))
    for i, n in enumerate(names):
        arr = np.asarray(Image.open(io.BytesIO(zin.read(n))).convert('RGB'), dtype=np.float64)
        b_vals[i] = brightness(arr)
        s_vals[i] = sharpness(arr)
        if (i + 1) % 2000 == 0:
            print(f'  {i+1}/{len(names)}')

    b_thr = float(np.median(b_vals)) if args.bright_thr == 'auto' else float(args.bright_thr)
    s_thr = float(np.median(s_vals)) if args.sharp_thr  == 'auto' else float(args.sharp_thr)
    print(f'Thresholds -> Brightness {b_thr:.2f}, Sharpness {s_thr:.2f}')

    bright_bit = (b_vals >= b_thr).astype(int)
    sharp_bit  = (s_vals >= s_thr).astype(int)
    labels = (bright_bit * 2 + sharp_bit)  # 0..3

    name2label = {n: int(l) for n, l in zip(names, labels)}
    hist = np.bincount(labels, minlength=4)
    tag = ['dark+blurry', 'dark+sharp', 'bright+blurry', 'bright+sharp']
    print('Bucket distribution (use these ratios for generate.py --proportions):')
    for k in range(4):
        print(f'  {k} {tag[k]:14s}: {hist[k]:6d}  ({100*hist[k]/len(labels):5.1f}%)')
    print('  proportions=' + ','.join(f'{hist[k]/len(labels):.4f}' for k in range(4)))
    if (hist == 0).any():
        print('  ! Empty bucket found -> c_dim < 4. Adjust thresholds.')

    print(f'Writing new zip: {args.out}')
    written = 0
    with zipfile.ZipFile(args.out, 'w', zipfile.ZIP_STORED) as zout:
        for n in zin.namelist():
            if n.lower().endswith('dataset.json'):
                continue
            zout.writestr(n, zin.read(n))
            written += 1
        zout.writestr('dataset.json', json.dumps({'labels': [[n, name2label[n]] for n in names]}))
    zin.close()
    print(f'Done. Entries {written} + dataset.json. Train with --data={args.out} --cond=1')


if __name__ == '__main__':
    main()