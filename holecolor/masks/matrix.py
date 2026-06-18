from __future__ import annotations

import cv2
import numpy as np

from holecolor.core.types import HoleGeometry


def make_global_hole_union(shape: tuple[int, int], holes: list[HoleGeometry], expand_px: int = 0) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    for h in holes:
        r = int(round(h.radius_outer_px + expand_px))
        cv2.circle(mask, (int(round(h.x)), int(round(h.y))), r, 1, -1)
    return mask.astype(bool)


def make_matrix_bulk_mask(roi_mask: np.ndarray, hole_union_mask: np.ndarray, hotspot_mask: np.ndarray | None = None) -> np.ndarray:
    mask = roi_mask.astype(bool) & ~hole_union_mask.astype(bool)
    if hotspot_mask is not None:
        mask &= ~hotspot_mask.astype(bool)
    return mask
