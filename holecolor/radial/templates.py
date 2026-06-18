from __future__ import annotations

import cv2
import numpy as np


def gradient_orientation_map(gray: np.ndarray) -> np.ndarray:
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    return np.arctan2(gy, gx)


def radial_template_response(gray: np.ndarray, template: np.ndarray) -> np.ndarray:
    return cv2.matchTemplate(gray.astype(np.float32), template.astype(np.float32), cv2.TM_CCOEFF_NORMED)
