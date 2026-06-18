from __future__ import annotations

from collections import defaultdict
from itertools import groupby
from typing import Any

import numpy as np


def _safe_float(value: Any, default: float = np.nan) -> float:
    try:
        return float(value) if value is not None else float(default)
    except Exception:
        return float(default)


def _safe_float_array(values) -> np.ndarray:
    return np.asarray([_safe_float(v) for v in values], dtype=float)


def _finite_nanmean(values) -> float | None:
    arr = _safe_float_array(values)
    arr = arr[np.isfinite(arr)]
    return float(np.mean(arr)) if arr.size else None




def _acc_init(fields: tuple[str, ...]) -> dict[str, Any]:
    return {"n": 0, **{f"sum__{f}": 0.0 for f in fields}, **{f"count__{f}": 0 for f in fields}}


def _acc_add(acc: dict[str, Any], field: str, value: Any) -> None:
    v = _safe_float(value)
    if np.isfinite(v):
        acc[f"sum__{field}"] += float(v)
        acc[f"count__{field}"] += 1


def _acc_mean(acc: dict[str, Any], field: str) -> float | None:
    n = int(acc.get(f"count__{field}", 0))
    if n <= 0:
        return None
    return float(acc[f"sum__{field}"] / n)



def _hotspot_stats_by_hole(hotspot_rows: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    stats: dict[int, dict[str, Any]] = {}
    for row in hotspot_rows:
        hid = row.get("nearest_hole_id")
        if hid is None:
            continue
        hole_id = int(hid)
        acc = stats.setdefault(hole_id, {"n": 0, "sum_dist": 0.0, "count_dist": 0, "min_dist": np.inf})
        acc["n"] += 1
        dist = _safe_float(row.get("dist_to_hole_px", np.nan))
        if np.isfinite(dist):
            acc["sum_dist"] += float(dist)
            acc["count_dist"] += 1
            if dist < acc["min_dist"]:
                acc["min_dist"] = float(dist)
    out: dict[int, dict[str, Any]] = {}
    for hole_id, acc in stats.items():
        count_dist = int(acc.get("count_dist", 0))
        out[hole_id] = {
            "n_hotspots": int(acc.get("n", 0)),
            "mean_hotspot_distance_px": None if count_dist <= 0 else float(acc["sum_dist"] / count_dist),
            "min_hotspot_distance_px": None if not np.isfinite(acc.get("min_dist", np.inf)) else float(acc["min_dist"]),
        }
    return out

def _r2(y: np.ndarray, yhat: np.ndarray) -> float | None:
    y = np.asarray(y, dtype=float)
    yhat = np.asarray(yhat, dtype=float)
    mask = np.isfinite(y) & np.isfinite(yhat)
    if int(mask.sum()) < 2:
        return None
    y = y[mask]
    yhat = yhat[mask]
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    if ss_tot <= 1e-12:
        return 1.0
    return 1.0 - ss_res / ss_tot


def fit_radial_models(radial_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Fit lightweight radial profile models per (hole, frame).

    The goal is descriptive stability rather than physical exactness.
    We fit linear and quadratic profiles over annulus index and export
    fit quality plus a front-radius proxy from the derivative of the
    quadratic fit when available.
    """
    if not radial_rows:
        return [], []
    fit_rows: list[dict[str, Any]] = []
    by_hole: dict[int, list[dict[str, Any]]] = defaultdict(list)
    ordered = sorted(radial_rows, key=lambda r: (int(r.get('hole_id', 0)), int(r.get('frame_id', 0)), int(r.get('annulus_id', 0))))
    for (hole_id, frame_id), grp in groupby(ordered, key=lambda r: (int(r.get('hole_id', 0)), int(r.get('frame_id', 0)))):
        rows = list(grp)
        x = np.asarray([int(r.get('annulus_id', 0)) for r in rows], dtype=float)
        y = _safe_float_array([r.get('descriptor_value', np.nan) for r in rows])
        valid = np.isfinite(x) & np.isfinite(y)
        if int(valid.sum()) < 2:
            continue
        x = x[valid]
        y = y[valid]
        meta = rows[0]
        lin = np.polyfit(x, y, deg=1)
        y_lin = np.polyval(lin, x)
        lin_r2 = _r2(y, y_lin)
        quad_a = quad_b = quad_c = quad_r2 = front_radius = None
        if int(valid.sum()) >= 3:
            quad = np.polyfit(x, y, deg=2)
            quad_a, quad_b, quad_c = [float(v) for v in quad]
            y_quad = np.polyval(quad, x)
            quad_r2 = _r2(y, y_quad)
            if abs(quad_a) > 1e-8:
                fr = -quad_b / (2.0 * quad_a)
                if np.isfinite(fr):
                    front_radius = float(fr)
        row = {
            'hole_id': int(hole_id),
            'frame_id': int(frame_id),
            'lattice_u': meta.get('lattice_u'),
            'lattice_v': meta.get('lattice_v'),
            'linear_slope': float(lin[0]),
            'linear_intercept': float(lin[1]),
            'linear_r2': lin_r2,
            'quadratic_a': quad_a,
            'quadratic_b': quad_b,
            'quadratic_c': quad_c,
            'quadratic_r2': quad_r2,
            'front_radius_quadratic': front_radius,
            'n_valid_annuli': int(len(x)),
        }
        fit_rows.append(row)
        by_hole[int(hole_id)].append(row)

    def _vals(rows, key):
        out = []
        for r in rows:
            v = r.get(key, np.nan)
            try:
                out.append(float(v) if v is not None else np.nan)
            except Exception:
                out.append(np.nan)
        return np.asarray(out, dtype=float)

    hole_summary: list[dict[str, Any]] = []
    for hole_id, rows in sorted(by_hole.items()):
        quad_r2 = _vals(rows, 'quadratic_r2')
        front_r = _vals(rows, 'front_radius_quadratic')
        hole_summary.append({
            'hole_id': int(hole_id),
            'lattice_u': rows[0].get('lattice_u'),
            'lattice_v': rows[0].get('lattice_v'),
            'mean_linear_slope': float(np.nanmean(_vals(rows, 'linear_slope'))),
            'mean_linear_r2': float(np.nanmean(_vals(rows, 'linear_r2'))),
            'mean_quadratic_r2': float(np.nanmean(quad_r2)) if np.any(np.isfinite(quad_r2)) else None,
            'mean_front_radius_quadratic': float(np.nanmean(front_r)) if np.any(np.isfinite(front_r)) else None,
            'front_radius_range': float(np.nanmax(front_r) - np.nanmin(front_r)) if np.any(np.isfinite(front_r)) else None,
            'n_frames': int(len(rows)),
        })
    return fit_rows, hole_summary

def compute_sector_radial_timeseries(
    frame_id: int,
    descriptor_image: np.ndarray,
    holes,
    terraces_by_hole,
    n_sectors: int = 8,
    lattice_indices: dict[int, tuple[int, int]] | None = None,
) -> list[dict[str, Any]]:
    n_sectors = max(4, int(n_sectors))
    out: list[dict[str, Any]] = []
    for hole in holes:
        masks = terraces_by_hole.get(int(hole.hole_id), [])
        if not masks:
            continue
        uv = lattice_indices.get(int(hole.hole_id), (None, None)) if lattice_indices is not None else (None, None)
        for annulus_id, terrace in enumerate(masks):
            yy, xx = terrace.global_coords() if hasattr(terrace, "global_coords") else np.nonzero(np.asarray(terrace).astype(bool))
            if yy.size == 0:
                continue
            vals = descriptor_image[yy, xx].astype(float, copy=False)
            angles = np.arctan2(yy - float(hole.y), xx - float(hole.x))
            bins = np.floor((angles + np.pi) / (2.0 * np.pi) * n_sectors).astype(int)
            bins = np.clip(bins, 0, n_sectors - 1)
            for sector_id in range(n_sectors):
                mask = bins == sector_id
                if not np.any(mask):
                    continue
                out.append(
                    {
                        "frame_id": int(frame_id),
                        "hole_id": int(hole.hole_id),
                        "lattice_u": uv[0],
                        "lattice_v": uv[1],
                        "annulus_id": int(annulus_id),
                        "sector_id": int(sector_id),
                        "descriptor_value": float(np.nanmean(vals[mask])),
                        "area_px": int(mask.sum()),
                    }
                )
    return out


def summarize_sector_fronts(sector_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Summarize anisotropic radial fronts by sector and over time."""
    if not sector_rows:
        return [], []
    sector_front_rows: list[dict[str, Any]] = []
    per_frame_hole: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    ordered = sorted(sector_rows, key=lambda r: (int(r.get('hole_id', 0)), int(r.get('frame_id', 0)), int(r.get('sector_id', 0)), int(r.get('annulus_id', 0))))
    for (hole_id, frame_id, sector_id), grp in groupby(ordered, key=lambda r: (int(r.get('hole_id', 0)), int(r.get('frame_id', 0)), int(r.get('sector_id', 0)))):
        rows = list(grp)
        ann = np.asarray([int(r.get('annulus_id', 0)) for r in rows], dtype=float)
        vals = _safe_float_array([r.get('descriptor_value', np.nan) for r in rows])
        valid = np.isfinite(ann) & np.isfinite(vals)
        if int(valid.sum()) == 0:
            continue
        ann = ann[valid]
        vals = vals[valid]
        shifted = vals - float(np.nanmin(vals)) + 1e-6
        denom = float(np.sum(shifted))
        com = float(np.sum(ann * shifted) / denom) if denom > 0 else float(np.nanmean(ann))
        peak_annulus = int(ann[int(np.nanargmax(vals))]) if vals.size else None
        row = {
            'hole_id': int(hole_id),
            'frame_id': int(frame_id),
            'sector_id': int(sector_id),
            'lattice_u': rows[0].get('lattice_u'),
            'lattice_v': rows[0].get('lattice_v'),
            'sector_front_radius': com,
            'sector_peak_annulus': peak_annulus,
            'n_valid_annuli': int(len(vals)),
        }
        sector_front_rows.append(row)
        per_frame_hole[(int(hole_id), int(frame_id))].append(row)
    frame_summary_rows: list[dict[str, Any]] = []
    for (hole_id, frame_id), rows in sorted(per_frame_hole.items()):
        fronts = _safe_float_array([r.get('sector_front_radius', np.nan) for r in rows])
        valid = fronts[np.isfinite(fronts)]
        if valid.size == 0:
            continue
        mean_front = float(np.mean(valid))
        std_front = float(np.std(valid))
        frame_summary_rows.append({
            'hole_id': int(hole_id),
            'frame_id': int(frame_id),
            'lattice_u': rows[0].get('lattice_u'),
            'lattice_v': rows[0].get('lattice_v'),
            'mean_sector_front_radius': mean_front,
            'std_sector_front_radius': std_front,
            'max_sector_front_radius': float(np.max(valid)),
            'min_sector_front_radius': float(np.min(valid)),
            'front_anisotropy_ratio': float(std_front / (abs(mean_front) + 1e-6)),
            'n_valid_sectors': int(valid.size),
        })
    return sector_front_rows, frame_summary_rows

def build_hotspot_reticulum_comparison(
    hotspot_rows: list[dict[str, Any]],
    per_hole_radial_summary_rows: list[dict[str, Any]],
    radial_archetype_rows: list[dict[str, Any]],
    zone_by_hole: dict[int, str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not per_hole_radial_summary_rows:
        return [], []
    hotspot_stats = _hotspot_stats_by_hole(hotspot_rows)
    arche_by_hole = {int(r.get("hole_id", 0)): r for r in radial_archetype_rows}
    out: list[dict[str, Any]] = []
    for row in per_hole_radial_summary_rows:
        hole_id = int(row.get("hole_id", 0))
        hs = hotspot_stats.get(hole_id, {})
        mean_dist = hs.get("mean_hotspot_distance_px")
        min_dist = hs.get("min_hotspot_distance_px")
        n_hotspots = int(hs.get("n_hotspots", 0))
        proximity_bucket = "none"
        if min_dist is not None:
            if min_dist <= 10:
                proximity_bucket = "near"
            elif min_dist <= 24:
                proximity_bucket = "mid"
            else:
                proximity_bucket = "far"
        arche = arche_by_hole.get(hole_id, {})
        out.append(
            {
                "hole_id": hole_id,
                "lattice_u": row.get("lattice_u"),
                "lattice_v": row.get("lattice_v"),
                "reticulum_zone": zone_by_hole.get(hole_id, "unknown"),
                "radial_conclusion_label": row.get("radial_conclusion_label"),
                "radial_archetype_label": arche.get("radial_archetype_label"),
                "delta_center_of_mass": row.get("delta_center_of_mass"),
                "mean_inner_minus_outer": row.get("mean_inner_minus_outer"),
                "mean_angular_asymmetry": row.get("mean_angular_asymmetry"),
                "n_hotspots": n_hotspots,
                "mean_hotspot_distance_px": mean_dist,
                "min_hotspot_distance_px": min_dist,
                "hotspot_proximity_bucket": proximity_bucket,
            }
        )
    group_acc: dict[tuple[str, str], dict[str, Any]] = {}
    for row in out:
        key = (str(row.get("reticulum_zone", "unknown")), str(row.get("hotspot_proximity_bucket", "none")))
        acc = group_acc.setdefault(key, _acc_init(("delta_center_of_mass", "mean_inner_minus_outer", "mean_angular_asymmetry", "mean_hotspot_distance_px")))
        acc["n"] += 1
        _acc_add(acc, "delta_center_of_mass", row.get("delta_center_of_mass"))
        _acc_add(acc, "mean_inner_minus_outer", row.get("mean_inner_minus_outer"))
        _acc_add(acc, "mean_angular_asymmetry", row.get("mean_angular_asymmetry"))
        _acc_add(acc, "mean_hotspot_distance_px", row.get("mean_hotspot_distance_px"))
    group_rows: list[dict[str, Any]] = []
    for (zone, bucket), acc in sorted(group_acc.items()):
        group_rows.append(
            {
                "reticulum_zone": zone,
                "hotspot_proximity_bucket": bucket,
                "n_holes": int(acc.get("n", 0)),
                "mean_delta_center_of_mass": _acc_mean(acc, "delta_center_of_mass"),
                "mean_inner_minus_outer": _acc_mean(acc, "mean_inner_minus_outer"),
                "mean_angular_asymmetry": _acc_mean(acc, "mean_angular_asymmetry"),
                "mean_hotspot_distance_px": _acc_mean(acc, "mean_hotspot_distance_px"),
            }
        )
    return out, group_rows

def build_rdf_hotspot_reticulum_comparison(
    hotspot_rows: list[dict[str, Any]],
    rdf_archetype_rows: list[dict[str, Any]],
    rdf_dynamics_rows: list[dict[str, Any]],
    zone_by_hole: dict[int, str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not rdf_archetype_rows:
        return [], []
    hotspot_stats = _hotspot_stats_by_hole(hotspot_rows)
    dyn_by_hole = {int(r.get("hole_id", 0)): r for r in rdf_dynamics_rows}
    out: list[dict[str, Any]] = []
    for row in rdf_archetype_rows:
        hole_id = int(row.get("hole_id", 0))
        hs = hotspot_stats.get(hole_id, {})
        mean_dist = hs.get("mean_hotspot_distance_px")
        min_dist = hs.get("min_hotspot_distance_px")
        proximity_bucket = "none"
        if min_dist is not None:
            if min_dist <= 10:
                proximity_bucket = "near"
            elif min_dist <= 24:
                proximity_bucket = "mid"
            else:
                proximity_bucket = "far"
        dyn = dyn_by_hole.get(hole_id, {})
        out.append({
            "hole_id": hole_id,
            "lattice_u": row.get("lattice_u"),
            "lattice_v": row.get("lattice_v"),
            "reticulum_zone": zone_by_hole.get(hole_id, "unknown"),
            "rdf_archetype_canonical_id": row.get("rdf_archetype_canonical_id"),
            "rdf_archetype_canonical_label": row.get("rdf_archetype_canonical_label"),
            "mean_front_radius_norm": row.get("mean_front_radius_norm"),
            "delta_front_radius_norm": row.get("delta_front_radius_norm"),
            "rdf_front_acceleration_per_frame2": dyn.get("rdf_front_acceleration_per_frame2"),
            "rdf_front_nonlinearity_gain": dyn.get("rdf_front_nonlinearity_gain"),
            "n_hotspots": int(hs.get("n_hotspots", 0)),
            "mean_hotspot_distance_px": mean_dist,
            "min_hotspot_distance_px": min_dist,
            "hotspot_proximity_bucket": proximity_bucket,
        })
    group_acc: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in out:
        key = (str(row.get("reticulum_zone", "unknown")), str(row.get("hotspot_proximity_bucket", "none")), str(row.get("rdf_archetype_canonical_label", "unknown")))
        acc = group_acc.setdefault(key, _acc_init(("mean_front_radius_norm", "delta_front_radius_norm", "rdf_front_acceleration_per_frame2", "mean_hotspot_distance_px")))
        acc["n"] += 1
        _acc_add(acc, "mean_front_radius_norm", row.get("mean_front_radius_norm"))
        _acc_add(acc, "delta_front_radius_norm", row.get("delta_front_radius_norm"))
        _acc_add(acc, "rdf_front_acceleration_per_frame2", row.get("rdf_front_acceleration_per_frame2"))
        _acc_add(acc, "mean_hotspot_distance_px", row.get("mean_hotspot_distance_px"))
    group_rows: list[dict[str, Any]] = []
    for (zone, bucket, arche), acc in sorted(group_acc.items()):
        group_rows.append({
            "reticulum_zone": zone,
            "hotspot_proximity_bucket": bucket,
            "rdf_archetype_canonical_label": arche,
            "n_holes": int(acc.get("n", 0)),
            "mean_front_radius_norm": _acc_mean(acc, "mean_front_radius_norm"),
            "mean_delta_front_radius_norm": _acc_mean(acc, "delta_front_radius_norm"),
            "mean_rdf_front_acceleration_per_frame2": _acc_mean(acc, "rdf_front_acceleration_per_frame2"),
            "mean_hotspot_distance_px": _acc_mean(acc, "mean_hotspot_distance_px"),
        })
    return out, group_rows

