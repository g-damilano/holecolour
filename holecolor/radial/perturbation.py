from __future__ import annotations

import numpy as np

from holecolor.config.schema import RadialConfig
from holecolor.core.types import HoleGeometry
from holecolor.masks.terraces import make_hole_terraces
from holecolor.radial.curves import compute_radial_curve


def perturb_hole_geometry(hole: HoleGeometry, radius_pct: float = 0.05) -> list[HoleGeometry]:
    delta = hole.radius_outer_px * radius_pct
    out = []
    for s in (-1.0, 1.0):
        out.append(HoleGeometry(hole.hole_id, hole.x, hole.y, max(hole.radius_inner_px + s * delta, 1.0), max(hole.radius_outer_px + s * delta, 1.0), hole.confidence))
    return out


def radial_curve_stability(descriptor_image: np.ndarray, hole: HoleGeometry, cfg: RadialConfig) -> dict[str, float]:
    curves = []
    for variant in perturb_hole_geometry(hole, cfg.perturb_radius_pct):
        terraces = make_hole_terraces(descriptor_image.shape[:2], variant, 8, variant.radius_outer_px + 16)
        curve = compute_radial_curve(0, variant.hole_id, descriptor_image, terraces, 'test')
        curves.append(np.array(curve.terrace_values, dtype=float))
    if len(curves) < 2:
        return {"mae": 0.0, "max_abs": 0.0}
    diff = np.abs(curves[0] - curves[1])
    mae = np.nanmean(diff)
    return {"mae": float(mae), "max_abs": float(np.nanmax(diff))}
