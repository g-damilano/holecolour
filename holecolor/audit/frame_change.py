from __future__ import annotations

import cv2
import numpy as np


def histogram_jump_score(a: np.ndarray, b: np.ndarray, bins: int = 64) -> float:
    a_gray = cv2.cvtColor(a, cv2.COLOR_RGB2GRAY)
    b_gray = cv2.cvtColor(b, cv2.COLOR_RGB2GRAY)
    ha = cv2.calcHist([a_gray], [0], None, [bins], [0, 256])
    hb = cv2.calcHist([b_gray], [0], None, [bins], [0, 256])
    ha = cv2.normalize(ha, None).ravel()
    hb = cv2.normalize(hb, None).ravel()
    return float(np.linalg.norm(ha - hb))
