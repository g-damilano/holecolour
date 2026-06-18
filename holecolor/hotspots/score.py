from __future__ import annotations

import numpy as np


def hotspot_score_map(descriptor_image: np.ndarray, baseline_descriptor: np.ndarray | None = None) -> np.ndarray:
    arr = descriptor_image.astype(np.float32)
    if baseline_descriptor is None:
        return arr
    return np.abs(arr - baseline_descriptor.astype(np.float32))
