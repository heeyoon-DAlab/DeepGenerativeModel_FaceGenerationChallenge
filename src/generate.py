#!/usr/bin/env python3
"""
Generate images from a trained snapshot with fixed seeds.

Works for both the unconditional runs (exp1/exp2) and the conditional one
(exp3). z is drawn per seed the same way stylegan3's gen_images.py does it, so a
given snapshot and seed range always produce the same images. For a conditional
model the per-image bucket label is sampled from RandomState(--label-seed) using
--proportions, so the (z, label) pairs are fixed too.

Needs the stylegan3 code importable, e.g. run with PYTHONPATH=stylegan3.
Writes img_0000.png ... (RGB), no translate/rotate.
"""
import argparse, os, re
import numpy as np
import torch
import PIL.Image

import dnnlib
import legacy

BUCKETS = ['dark+blurry', 'dark+sharp', 'bright+blurry', 'bright+sharp']


def parse_seeds(s):
    out = []
    for part in s.split(','):
        m = re.match(r'^(\d+)-(\d+)$', part)
        if m:
            out += list(range(int(m.group(1)), int(m.group(2)) + 1))
        else:
            out.append(int(part))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--network', required=True, help='snapshot .pkl')
    ap.add_argument('--outdir', required=True)
    ap.add_argument('--seeds', default='0-999')
    ap.add_argument('--trunc', type=float, default=1.0)
    ap.add_argument('--proportions', default=None,
                    help='conditional only: comma list len=c_dim, e.g. "0.30,0.20,0.20,0.30". '
                         'order: 0=dark+blurry 1=dark+sharp 2=bright+blurry 3=bright+sharp')
    ap.add_argument('--label-seed', type=int, default=0,
                    help='conditional only: seed for per-image bucket sampling')
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit('CUDA not available.')
    device = torch.device('cuda')

    print(f'Loading {args.network}')
    with dnnlib.util.open_url(args.network) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device)
    G.eval()

    os.makedirs(args.outdir, exist_ok=True)
    seeds = parse_seeds(args.seeds)
    n = len(seeds)

    if G.c_dim == 0:
        if args.proportions is not None:
            print('warn: --proportions ignored (unconditional network)')
        class_ids = [None] * n
    else:
        if args.proportions is None:
            raise SystemExit(f'Conditional model (c_dim={G.c_dim}); pass --proportions '
                             f'with {G.c_dim} values.')
        p = np.array([float(x) for x in args.proportions.split(',')], dtype=np.float64)
        if len(p) != G.c_dim:
            raise SystemExit(f'--proportions has {len(p)} values but c_dim={G.c_dim}')
        p = p / p.sum()
        class_ids = np.random.RandomState(args.label_seed).choice(G.c_dim, size=n, p=p).tolist()
        cnt = np.bincount(class_ids, minlength=G.c_dim)
        print('bucket counts: ' + ', '.join(
            f'{BUCKETS[k] if k < len(BUCKETS) else k}={cnt[k]}' for k in range(G.c_dim)))

    for i, seed in enumerate(seeds):
        z = torch.from_numpy(np.random.RandomState(seed).randn(1, G.z_dim)).to(device)
        label = torch.zeros([1, G.c_dim], device=device)
        if class_ids[i] is not None:
            label[0, class_ids[i]] = 1
        img = G(z, label, truncation_psi=args.trunc, noise_mode='const')
        img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
        PIL.Image.fromarray(img[0].cpu().numpy(), 'RGB').save(
            os.path.join(args.outdir, f'img_{seed:04d}.png'))
        if (i + 1) % 200 == 0:
            print(f'  {i+1}/{n}')
    print(f'done -> {args.outdir}  ({n} images)')


if __name__ == '__main__':
    main()
