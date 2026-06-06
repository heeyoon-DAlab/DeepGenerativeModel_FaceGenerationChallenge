"""
image_quality.py — shared brightness & sharpness measures.

Used by BOTH filter_brightness_blur.py (exp2) and make_conditional_zip.py (exp3)
so the "quality axes" are defined identically across experiments. Pure numpy
(no cv2), so it runs in any environment.
"""
import numpy as np


def _luma(arr):
    # arr: HxWx3, any numeric dtype
    return 0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]


def brightness(arr):
    """Mean luma in [0, 255]. Higher = brighter."""
    return float(_luma(arr).mean())


def sharpness(arr):
    """Variance of the discrete Laplacian of luma. Higher = sharper / less blurry."""
    l = _luma(arr).astype(np.float64)
    lap = (-4 * l
           + np.roll(l, 1, 0) + np.roll(l, -1, 0)
           + np.roll(l, 1, 1) + np.roll(l, -1, 1))
    return float(lap[1:-1, 1:-1].var())  # drop wrap-around border
