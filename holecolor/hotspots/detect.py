from __future__ import annotations

import cv2
import numpy as np

from dataclasses import dataclass

from holecolor.config.schema import HotspotConfig
from holecolor.core.types import HoleGeometry


@dataclass(slots=True)
class Hotspot:
    frame_id: int
    hotspot_id: int
    cx: float
    cy: float
    area_px: int
    score: float
    nearest_hole_id: int | None
    dist_to_hole_px: float | None
    mean_r: float | None = None
    mean_g: float | None = None
    mean_b: float | None = None
    mean_h: float | None = None
    mean_s: float | None = None
    bbox_x: int | None = None
    bbox_y: int | None = None
    bbox_w: int | None = None
    bbox_h: int | None = None


def _threshold(score_map: np.ndarray, cfg: HotspotConfig) -> np.ndarray:
    vals = score_map[np.isfinite(score_map)]
    if vals.size == 0:
        return np.zeros_like(score_map, dtype=bool)
    if cfg.threshold_mode == "percentile":
        thr = np.percentile(vals, cfg.threshold_value)
    elif cfg.threshold_mode == "otsu":
        score_u8 = np.clip(score_map, 0, np.percentile(vals, 99.5) if vals.size else 255)
        score_u8 = (255 * score_u8 / max(float(score_u8.max()), 1e-6)).astype(np.uint8)
        thr, _ = cv2.threshold(score_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        thr = float(thr) / 255.0 * max(float(score_map.max()), 1e-6)
    else:
        thr = vals.mean() + 2.0 * vals.std()
    return score_map >= thr


def _nearest_hole(cx: float, cy: float, holes: list[HoleGeometry]) -> tuple[int | None, float | None]:
    if not holes:
        return None, None
    d = [((cx - h.x) ** 2 + (cy - h.y) ** 2) ** 0.5 for h in holes]
    idx = int(np.argmin(d))
    return holes[idx].hole_id, float(d[idx])


def detect_hotspots(
    frame_id: int,
    score_map: np.ndarray,
    matrix_mask: np.ndarray,
    cfg: HotspotConfig,
    holes: list[HoleGeometry] | None = None,
    image_rgb: np.ndarray | None = None,
    image_hsv: np.ndarray | None = None,
) -> list[Hotspot]:
    mask = _threshold(score_map, cfg) & matrix_mask.astype(bool)
    mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, np.ones((3, 3), np.uint8))
    n, labels, stats, cent = cv2.connectedComponentsWithStats(mask, connectivity=8)
    out = []
    holes = holes or []
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < cfg.min_area_px:
            continue
        comp = labels == i
        cx, cy = map(float, cent[i])
        hid, dist = _nearest_hole(cx, cy, holes)
        x = int(stats[i, cv2.CC_STAT_LEFT])
        y = int(stats[i, cv2.CC_STAT_TOP])
        w = int(stats[i, cv2.CC_STAT_WIDTH])
        h = int(stats[i, cv2.CC_STAT_HEIGHT])
        kw = dict(mean_r=None, mean_g=None, mean_b=None, mean_h=None, mean_s=None)
        if image_rgb is not None:
            rgb_vals = image_rgb[comp]
            if rgb_vals.size:
                kw.update(mean_r=float(rgb_vals[:, 0].mean()), mean_g=float(rgb_vals[:, 1].mean()), mean_b=float(rgb_vals[:, 2].mean()))
        if image_hsv is not None:
            hsv_vals = image_hsv[comp]
            if hsv_vals.size:
                kw.update(mean_h=float(hsv_vals[:, 0].mean()), mean_s=float(hsv_vals[:, 1].mean()))
        out.append(Hotspot(frame_id, len(out), cx, cy, area, float(score_map[comp].mean()), hid, dist, bbox_x=x, bbox_y=y, bbox_w=w, bbox_h=h, **kw))
    return out
