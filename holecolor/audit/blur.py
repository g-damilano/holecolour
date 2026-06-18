from __future__ import annotations

import cv2
import numpy as np

try:
    import pywt  # type: ignore
except Exception:  # pragma: no cover - optional dependency fallback
    pywt = None


def wavelet_blur_score(gray: np.ndarray) -> float:
    gray_f = gray.astype(np.float32) / 255.0
    if pywt is None:
        lap = cv2.Laplacian(gray_f, cv2.CV_32F)
        return float(np.var(lap))
    coeffs = pywt.dwt2(gray_f, "haar")
    _, (lh, hl, hh) = coeffs
    energy = float(np.mean(np.abs(lh)) + np.mean(np.abs(hl)) + np.mean(np.abs(hh)))
    return energy


def laplacian_variance_score(gray: np.ndarray) -> float:
    lap = cv2.Laplacian(gray, cv2.CV_32F)
    return float(lap.var())


def classify_blur(score: float, threshold: float) -> bool:
    return score >= threshold
