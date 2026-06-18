from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class RadialCurve:
    frame_id: int
    hole_id: int
    descriptor_name: str
    terrace_values: list[float]


def _terrace_coords(terrace) -> tuple[np.ndarray, np.ndarray]:
    if hasattr(terrace, "global_coords"):
        return terrace.global_coords()
    yy, xx = np.nonzero(np.asarray(terrace).astype(bool))
    return yy, xx


def _terrace_values(descriptor_image: np.ndarray, terrace) -> np.ndarray:
    yy, xx = _terrace_coords(terrace)
    if yy.size == 0:
        return np.asarray([], dtype=float)
    return descriptor_image[yy, xx].astype(float, copy=False)


def compute_radial_curve(frame_id: int, hole_id: int, descriptor_image: np.ndarray, terraces, descriptor_name: str) -> RadialCurve:
    vals = []
    for t in terraces:
        px = _terrace_values(descriptor_image, t)
        vals.append(float(np.nanmean(px)) if px.size else float('nan'))
    return RadialCurve(frame_id, hole_id, descriptor_name, vals)


def compute_all_radial_curves(frame_id: int, descriptor_image: np.ndarray, terraces_by_hole, descriptor_name: str) -> list[RadialCurve]:
    return [compute_radial_curve(frame_id, hid, descriptor_image, terraces, descriptor_name) for hid, terraces in terraces_by_hole.items()]
