from __future__ import annotations

import cv2
import numpy as np


from holecolor.registration.rigid import Transform2D


def add_photometric_artifacts(frames: list[np.ndarray], brightness_ramp: float = 0.2, vignette_strength: float = 0.3, channel_gain=(1.0, 1.1, 0.9)) -> list[np.ndarray]:
    out = []
    h, w = frames[0].shape[:2]
    yy, xx = np.indices((h, w))
    rr = ((xx - w / 2) / (w / 2)) ** 2 + ((yy - h / 2) / (h / 2)) ** 2
    vign = 1 - vignette_strength * np.clip(rr, 0, 1)
    for i, frame in enumerate(frames):
        gain_t = 1 + brightness_ramp * i / max(len(frames) - 1, 1)
        arr = frame.astype(np.float32) * gain_t
        arr *= np.array(channel_gain, dtype=np.float32)[None, None, :]
        arr *= vign[..., None]
        out.append(np.clip(arr, 0, 255).astype(np.uint8))
    return out


def add_known_drift(frames: list[np.ndarray], dx_per_frame: float = 0.5, dy_per_frame: float = -0.3):
    out = []
    tfms = []
    for i, frame in enumerate(frames):
        tfm = Transform2D(dx=i * dx_per_frame, dy=i * dy_per_frame)
        mat = np.array([[1.0, 0.0, tfm.dx], [0.0, 1.0, tfm.dy]], dtype=np.float32)
        moved = cv2.warpAffine(frame, mat, (frame.shape[1], frame.shape[0]))
        out.append(moved)
        tfms.append(tfm)
    return out, tfms
