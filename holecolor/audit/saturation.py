from __future__ import annotations

import numpy as np


def saturation_fraction(channel: np.ndarray, threshold: int = 250) -> float:
    return float(np.mean(channel >= threshold))
