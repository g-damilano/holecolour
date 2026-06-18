from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class RegionStats:
    frame_id: int
    region_id: str
    mean_r: float
    mean_g: float
    mean_b: float
    mean_h: float
    mean_s: float
    area_px: int


def _stats_from_coords(frame_id: int, region_id: str, image_rgb: np.ndarray, image_hsv: np.ndarray | None, yy: np.ndarray, xx: np.ndarray) -> RegionStats:
    area = int(yy.size)
    if area == 0:
        return RegionStats(frame_id, region_id, np.nan, np.nan, np.nan, np.nan, np.nan, 0)
    rgb = image_rgb[yy, xx].astype(np.float32)
    if image_hsv is None:
        h = s = np.array([np.nan], dtype=np.float32)
    else:
        h = image_hsv[..., 0][yy, xx].astype(np.float32)
        s = image_hsv[..., 1][yy, xx].astype(np.float32)
    return RegionStats(frame_id, region_id, float(rgb[:, 0].mean()), float(rgb[:, 1].mean()), float(rgb[:, 2].mean()), float(np.nanmean(h)), float(np.nanmean(s)), area)


def compute_region_stats(frame_id: int, image_rgb: np.ndarray, image_hsv: np.ndarray | None, region_mask: np.ndarray, region_id: str) -> RegionStats:
    mask = region_mask.astype(bool)
    yy, xx = np.nonzero(mask)
    return _stats_from_coords(frame_id, region_id, image_rgb, image_hsv, yy, xx)


def compute_region_stats_from_coords(frame_id: int, image_rgb: np.ndarray, image_hsv: np.ndarray | None, yy: np.ndarray, xx: np.ndarray, region_id: str) -> RegionStats:
    return _stats_from_coords(frame_id, region_id, image_rgb, image_hsv, yy.astype(int, copy=False), xx.astype(int, copy=False))


def compute_stats_for_masks(frame_id: int, image_rgb: np.ndarray, image_hsv: np.ndarray | None, masks: dict[str, np.ndarray]) -> list[RegionStats]:
    return [compute_region_stats(frame_id, image_rgb, image_hsv, mask, name) for name, mask in masks.items()]


def compute_region_stats_from_region(frame_id: int, image_rgb: np.ndarray, image_hsv: np.ndarray | None, region, region_id: str) -> RegionStats:
    if not hasattr(region, "mask"):
        raise TypeError("region must expose a mask attribute")
    yy, xx = np.nonzero(np.asarray(region.mask).astype(bool, copy=False))
    if yy.size == 0:
        return RegionStats(frame_id, region_id, np.nan, np.nan, np.nan, np.nan, np.nan, 0)
    yy = yy + int(region.y0)
    xx = xx + int(region.x0)
    return _stats_from_coords(frame_id, region_id, image_rgb, image_hsv, yy, xx)


def mean_from_region(image: np.ndarray, region) -> float:
    if not hasattr(region, "mask"):
        raise TypeError("region must expose a mask attribute")
    mask = np.asarray(region.mask).astype(bool, copy=False)
    if mask.size == 0 or not np.any(mask):
        return float("nan")
    view = image[int(region.y0):int(region.y1), int(region.x0):int(region.x1)]
    vals = view[mask]
    return float(np.nanmean(vals.astype(np.float32, copy=False))) if vals.size else float("nan")
