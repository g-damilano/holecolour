from __future__ import annotations

import math

import cv2
import numpy as np


def make_synthetic_grid(shape=(512, 512), rows=6, cols=8, radius_px=18, spacing_px=60, rotation_deg=8.0, blur_sigma=0.0, noise_sigma=0.0, vignetting=0.0, missing_frac=0.0):
    h, w = shape
    img = np.full((h, w, 3), 180, dtype=np.float32)
    yy, xx = np.indices((h, w))
    cx0, cy0 = w / 2.0, h / 2.0
    th = math.radians(rotation_deg)
    rot = np.array([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]], dtype=np.float32)
    centers = []
    for r in range(rows):
        for c in range(cols):
            p = np.array([(c - (cols - 1) / 2.0) * spacing_px, (r - (rows - 1) / 2.0) * spacing_px], dtype=np.float32)
            x, y = rot @ p + np.array([cx0, cy0], dtype=np.float32)
            centers.append((float(x), float(y)))
    keep = np.ones(len(centers), dtype=bool)
    if missing_frac > 0:
        rng = np.random.default_rng(42)
        drop = rng.choice(len(centers), size=int(len(centers) * missing_frac), replace=False)
        keep[drop] = False
    for i, (x, y) in enumerate(centers):
        if not keep[i]:
            continue
        cv2.circle(img, (int(round(x)), int(round(y))), radius_px, (40, 40, 40), -1)
    if vignetting > 0:
        rr = ((xx - cx0) / (w / 2)) ** 2 + ((yy - cy0) / (h / 2)) ** 2
        vign = 1 - vignetting * np.clip(rr, 0, 1)
        img *= vign[..., None]
    if blur_sigma > 0:
        img = cv2.GaussianBlur(img, (0, 0), blur_sigma)
    if noise_sigma > 0:
        img += np.random.default_rng(1).normal(0, noise_sigma, img.shape)
    img = np.clip(img, 0, 255).astype(np.uint8)
    gt = {"centers": centers, "radius_px": radius_px, "spacing_px": spacing_px}
    return img, gt
