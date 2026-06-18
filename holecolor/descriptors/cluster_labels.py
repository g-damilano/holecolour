from __future__ import annotations

import math

import numpy as np

from holecolor.descriptors.color_spaces import rgb_to_hsl

try:  # pragma: no cover - exercised when numba is installed
    from numba import njit, prange
    _HAS_NUMBA = True
except Exception:  # pragma: no cover - keeps the package importable without numba
    _HAS_NUMBA = False

    def njit(*args, **kwargs):
        def deco(fn):
            return fn
        return deco

    def prange(*args):
        return range(*args)


@njit(cache=True, parallel=True, nogil=True)
def _predict_hsl_cluster_labels_kernel(image: np.ndarray, valid_mask: np.ndarray, centers: np.ndarray) -> np.ndarray:
    h_img = image.shape[0]
    w_img = image.shape[1]
    n_centers = centers.shape[0]
    labels = np.full((h_img, w_img), -1, dtype=np.int32)
    two_pi = 2.0 * math.pi
    for y in prange(h_img):
        for x in range(w_img):
            if not valid_mask[y, x]:
                continue
            r = float(image[y, x, 0]) / 255.0
            g = float(image[y, x, 1]) / 255.0
            b = float(image[y, x, 2]) / 255.0
            mx = r
            if g > mx:
                mx = g
            if b > mx:
                mx = b
            mn = r
            if g < mn:
                mn = g
            if b < mn:
                mn = b
            c = mx - mn
            lightness = 0.5 * (mx + mn)
            saturation = 0.0
            hue = 0.0
            if c > 0.0:
                denom = 1.0 - abs(2.0 * lightness - 1.0) + 1e-6
                saturation = c / denom
                if mx == r:
                    hue = (g - b) / c
                    while hue < 0.0:
                        hue += 6.0
                    while hue >= 6.0:
                        hue -= 6.0
                elif mx == g:
                    hue = (b - r) / c + 2.0
                else:
                    hue = (r - g) / c + 4.0
                hue /= 6.0
            hx = saturation * math.cos(two_pi * hue)
            hy = saturation * math.sin(two_pi * hue)
            best_idx = -1
            best_d2 = 1.0e30
            for ci in range(n_centers):
                dx = hx - float(centers[ci, 0])
                dy = hy - float(centers[ci, 1])
                dl = lightness - float(centers[ci, 2])
                d2 = dx * dx + dy * dy + dl * dl
                if d2 < best_d2:
                    best_d2 = d2
                    best_idx = ci
            labels[y, x] = best_idx
    return labels


def predict_hsl_cluster_labels(frame_rgb: np.ndarray, valid_mask: np.ndarray, centers: np.ndarray) -> np.ndarray:
    centers = np.asarray(centers, dtype=np.float32)
    if centers.ndim != 2 or centers.shape[0] == 0 or centers.shape[1] < 3:
        return np.full(np.asarray(valid_mask).shape, -1, dtype=np.int32)
    valid = np.asarray(valid_mask, dtype=np.bool_)
    if not np.any(valid):
        return np.full(valid.shape, -1, dtype=np.int32)
    if _HAS_NUMBA:
        image = np.ascontiguousarray(frame_rgb[..., :3], dtype=np.uint8)
        return _predict_hsl_cluster_labels_kernel(image, np.ascontiguousarray(valid), np.ascontiguousarray(centers[:, :3]))

    labels = np.full(valid.shape, -1, dtype=np.int32)
    hsl = rgb_to_hsl(frame_rgb)
    emb = np.empty((*valid.shape, 3), dtype=np.float32)
    emb[..., 0] = hsl[..., 1] * np.cos(2 * np.pi * hsl[..., 0])
    emb[..., 1] = hsl[..., 1] * np.sin(2 * np.pi * hsl[..., 0])
    emb[..., 2] = hsl[..., 2]
    yy, xx = np.where(valid)
    pts = emb[yy, xx]
    d2 = np.sum((pts[:, None, :] - centers[None, :, :3]) ** 2, axis=2)
    labels[yy, xx] = np.argmin(d2, axis=1).astype(np.int32)
    return labels
