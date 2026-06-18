from __future__ import annotations

import numpy as np


def overlap_fraction(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.logical_and(a, b).sum()
    denom = min(a.sum(), b.sum())
    return float(inter / denom) if denom else 0.0
