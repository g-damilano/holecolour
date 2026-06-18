from __future__ import annotations

import numpy as np

from holecolor.synth.grid import make_synthetic_grid


def make_synthetic_radial_front_video(n_frames=30, front_speed_px_per_frame=1.5, hotspot_mode='ring'):
    base, gt = make_synthetic_grid()
    frames = []
    h, w, _ = base.shape
    yy, xx = np.indices((h, w))
    centers = gt['centers']
    radius = gt['radius_px']
    for t in range(n_frames):
        img = base.astype(np.float32).copy()
        for (cx, cy) in centers:
            rr = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
            front = radius + t * front_speed_px_per_frame
            ring = np.exp(-((rr - front) ** 2) / (2 * (4.0 ** 2)))
            img[..., 0] += 20 * ring
            img[..., 1] += 5 * ring
            img[..., 2] += 30 * ring
        frames.append(np.clip(img, 0, 255).astype(np.uint8))
    return frames, gt
