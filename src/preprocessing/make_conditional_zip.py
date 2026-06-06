#!/usr/bin/env python3
"""
make_conditional_zip.py  (exp3)

Reads the (unconditional) full dataset zip and writes a CONDITIONAL zip:
every image is bucketed by brightness x sharpness (2x2 = 4 classes) and a
dataset.json with integer class labels is added. Pixels are NOT touched
(original bytes copied verbatim); only dataset.json is added.

Bucket order (index = bright_bit*2 + sharp_bit):
  0: dark   + blurry
  1: dark   + sharp
  2: bright + blurry
  3: bright + sharp

Usage:
  python src/preprocessing/make_conditional_zip.py \
    --in  data/celebvhq256_baseline.zip \
    --out data/celebvhq256_conditional.zip
  # thresholds default to 'auto' (per-axis median -> balanced buckets);
  # to use perceptual thresholds:  --bright_thr 60 --sharp_thr 100
"""
import argparse, io, json, zipfile, sys
import numpy as np
from PIL import Image
from image_quality import brightness, sharpness  # same dir on sys.path when run as script

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
        sys.exit('zip 안에 이미지가 없음. 구조 확인 필요.')
    print(f'이미지 {len(names)}장 측정 중...')

    b_vals, s_vals = np.empty(len(names)), np.empty(len(names))
    for i, n in enumerate(names):
        arr = np.asarray(Image.open(io.BytesIO(zin.read(n))).convert('RGB'), dtype=np.float64)
        b_vals[i] = brightness(arr)
        s_vals[i] = sharpness(arr)
        if (i + 1) % 2000 == 0:
            print(f'  {i+1}/{len(names)}')

    b_thr = float(np.median(b_vals)) if args.bright_thr == 'auto' else float(args.bright_thr)
    s_thr = float(np.median(s_vals)) if args.sharp_thr  == 'auto' else float(args.sharp_thr)
    print(f'임계값 -> 밝기 {b_thr:.2f}, 선명도 {s_thr:.2f}')

    bright_bit = (b_vals >= b_thr).astype(int)
    sharp_bit  = (s_vals >= s_thr).astype(int)
    labels = (bright_bit * 2 + sharp_bit)  # 0..3

    name2label = {n: int(l) for n, l in zip(names, labels)}
    hist = np.bincount(labels, minlength=4)
    tag = ['dark+blurry', 'dark+sharp', 'bright+blurry', 'bright+sharp']
    print('버킷 분포 (이 비율을 generate.py --proportions 에 그대로 쓰면 됨):')
    for k in range(4):
        print(f'  {k} {tag[k]:14s}: {hist[k]:6d}  ({100*hist[k]/len(labels):5.1f}%)')
    print('  proportions=' + ','.join(f'{hist[k]/len(labels):.4f}' for k in range(4)))
    if (hist == 0).any():
        print('  ! 빈 버킷 있음 -> c_dim 이 4 미만. 임계값 조정 필요.')

    print(f'새 zip 작성: {args.out}')
    written = 0
    with zipfile.ZipFile(args.out, 'w', zipfile.ZIP_STORED) as zout:
        for n in zin.namelist():
            if n.lower().endswith('dataset.json'):
                continue
            zout.writestr(n, zin.read(n))
            written += 1
        zout.writestr('dataset.json', json.dumps({'labels': [[n, name2label[n]] for n in names]}))
    zin.close()
    print(f'완료. 엔트리 {written} + dataset.json. 학습 시 --data={args.out} --cond=1')


if __name__ == '__main__':
    main()
