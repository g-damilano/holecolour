from __future__ import annotations

import cv2
import numpy as np


def edge_energy_ratio(image: np.ndarray, border_width_px: int) -> float:
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx * gx + gy * gy)
    h, w = mag.shape
    b = max(1, min(border_width_px, min(h, w) // 4))
    border_mask = np.zeros_like(mag, dtype=bool)
    border_mask[:b, :] = True
    border_mask[-b:, :] = True
    border_mask[:, :b] = True
    border_mask[:, -b:] = True
    center_mask = ~border_mask
    border_energy = float(mag[border_mask].mean())
    center_energy = float(mag[center_mask].mean()) if np.any(center_mask) else 1.0
    return border_energy / max(center_energy, 1e-6)


def positive_prominence(image: np.ndarray) -> float:
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32)
    blur = cv2.GaussianBlur(gray, (0, 0), 5.0)
    resid = gray - blur
    resid = resid[resid > 0]
    return float(resid.mean()) if resid.size else 0.0


def hole_control_drift(image: np.ndarray, hole_mask: np.ndarray | None, baseline_image: np.ndarray | None) -> float | None:
    if hole_mask is None or baseline_image is None:
        return None
    a = image[hole_mask].astype(np.float32)
    b = baseline_image[hole_mask].astype(np.float32)
    if a.size == 0 or b.size == 0:
        return None
    return float(np.mean(np.abs(a - b)))
