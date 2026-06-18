from __future__ import annotations

import math
from collections import defaultdict
from itertools import groupby
from typing import Any

import numpy as np

try:  # pragma: no cover - exercised when numba is installed
    from numba import njit
except Exception:  # pragma: no cover - keeps the package importable without numba
    def njit(*args, **kwargs):
        def deco(fn):
            return fn
        return deco


RADIAL_FEATURE_KEYS = (
    "delta_center_of_mass",
    "delta_peak_annulus",
    "mean_inner_minus_outer",
    "com_range",
    "mean_angular_asymmetry",
    "mean_angular_vector_strength",
)


def _safe_float(value: Any, default: float = np.nan) -> float:
    try:
        return float(value) if value is not None else float(default)
    except Exception:
        return float(default)


def _finite_mean(vals) -> float | None:
    arr = np.asarray([_safe_float(v) for v in vals], dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(arr.mean()) if arr.size else None


def _iter_grouped_rows(rows: list[dict[str, Any]], key_fn):
    ordered = sorted(rows, key=key_fn)
    for key, grp in groupby(ordered, key=key_fn):
        yield key, list(grp)


def _mode_int(values) -> int | None:
    counts: dict[int, int] = {}
    for value in values:
        if value is None:
            continue
        try:
            iv = int(value)
        except Exception:
            continue
        counts[iv] = counts.get(iv, 0) + 1
    if not counts:
        return None
    return max(counts.items(), key=lambda kv: (kv[1], -kv[0]))[0]


@njit(cache=True)
def _angular_sector_stats_kernel(yy: np.ndarray, xx: np.ndarray, vals: np.ndarray, hole_y: float, hole_x: float, n_sectors: int):
    sector_sum = np.zeros(n_sectors, dtype=np.float64)
    sector_count = np.zeros(n_sectors, dtype=np.int64)
    min_val = np.inf
    n_finite = 0
    for i in range(vals.size):
        v = float(vals[i])
        if np.isfinite(v):
            n_finite += 1
            if v < min_val:
                min_val = v
    sector_means = np.empty(n_sectors, dtype=np.float64)
    for b in range(n_sectors):
        sector_means[b] = np.nan
    if n_finite == 0:
        return sector_means, 0, np.nan, np.nan, np.nan, -1

    vec_x = 0.0
    vec_y = 0.0
    wsum = 0.0
    two_pi = 2.0 * math.pi
    for i in range(vals.size):
        v = float(vals[i])
        if not np.isfinite(v):
            continue
        angle = math.atan2(float(yy[i]) - hole_y, float(xx[i]) - hole_x)
        b = int(math.floor((angle + math.pi) / two_pi * float(n_sectors)))
        if b < 0:
            b = 0
        elif b >= n_sectors:
            b = n_sectors - 1
        sector_sum[b] += v
        sector_count[b] += 1
        weight = v - min_val + 1e-6
        if np.isfinite(weight):
            vec_x += weight * math.cos(angle)
            vec_y += weight * math.sin(angle)
            wsum += abs(weight)

    valid_count = 0
    valid_sum = 0.0
    valid_min = np.inf
    valid_max = -np.inf
    dominant_sector = -1
    for b in range(n_sectors):
        if sector_count[b] > 0:
            mean = sector_sum[b] / float(sector_count[b])
            sector_means[b] = mean
            valid_count += 1
            valid_sum += mean
            if mean < valid_min:
                valid_min = mean
            if mean > valid_max:
                valid_max = mean
                dominant_sector = b
    if valid_count == 0:
        return sector_means, 0, np.nan, np.nan, np.nan, -1

    mean_val = valid_sum / float(valid_count)
    var = 0.0
    for b in range(n_sectors):
        if sector_count[b] > 0:
            diff = sector_means[b] - mean_val
            var += diff * diff
    std_val = math.sqrt(var / float(valid_count))
    asymmetry = std_val / (abs(mean_val) + 1e-6)
    contrast = valid_max - valid_min
    vector_strength = np.nan
    if wsum > 0.0:
        vector_strength = math.sqrt(vec_x * vec_x + vec_y * vec_y) / (wsum + 1e-6)
    return sector_means, valid_count, asymmetry, contrast, vector_strength, dominant_sector


def per_hole_radial_frame_summary(radial_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not radial_rows:
        return []
    ordered = sorted(radial_rows, key=lambda r: (int(r.get("hole_id", 0)), int(r.get("frame_id", 0)), int(r.get("annulus_id", 0))))
    out: list[dict[str, Any]] = []
    for (hole_id, frame_id), grp in groupby(ordered, key=lambda r: (int(r.get("hole_id", 0)), int(r.get("frame_id", 0)))):
        rows = list(grp)
        ann = np.asarray([int(r.get("annulus_id", 0)) for r in rows], dtype=float)
        vals = np.asarray([_safe_float(r.get("descriptor_value", np.nan)) for r in rows], dtype=float)
        finite = np.isfinite(vals)
        if np.any(finite):
            ann = ann[finite]
            vals = vals[finite]
            shifted = vals - float(np.nanmin(vals)) + 1e-6
            denom = float(np.sum(shifted))
            com = float(np.sum(ann * shifted) / denom) if denom > 0 else float(np.nanmean(ann))
            peak_annulus = int(ann[int(np.nanargmax(vals))]) if vals.size else None
            inner_minus_outer = float(vals[0] - vals[-1]) if vals.size >= 2 else None
            mean_descriptor = float(np.nanmean(vals)) if vals.size else None
            std_descriptor = float(np.nanstd(vals)) if vals.size else None
            n_valid = int(vals.size)
        else:
            com = None
            peak_annulus = None
            inner_minus_outer = None
            mean_descriptor = None
            std_descriptor = None
            n_valid = 0
        uv = (rows[0].get("lattice_u"), rows[0].get("lattice_v"))
        out.append(
            {
                "hole_id": int(hole_id),
                "frame_id": int(frame_id),
                "lattice_u": uv[0],
                "lattice_v": uv[1],
                "center_of_mass_annulus": com,
                "peak_annulus": peak_annulus,
                "inner_minus_outer": inner_minus_outer,
                "mean_descriptor": mean_descriptor,
                "std_descriptor": std_descriptor,
                "n_valid_annuli": n_valid,
            }
        )
    return out

def summarize_hole_radial_evolution(frame_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not frame_rows:
        return []
    ordered = sorted(frame_rows, key=lambda r: (int(r.get("hole_id", 0)), int(r.get("frame_id", 0))))
    out: list[dict[str, Any]] = []
    for hole_id, grp in groupby(ordered, key=lambda r: int(r.get("hole_id", 0))):
        rows = list(grp)
        com_vals = np.asarray([_safe_float(r.get("center_of_mass_annulus", np.nan)) for r in rows], dtype=float)
        peak_vals = np.asarray([_safe_float(r.get("peak_annulus", np.nan)) for r in rows], dtype=float)
        im_vals = np.asarray([_safe_float(r.get("inner_minus_outer", np.nan)) for r in rows], dtype=float)
        finite_com = np.isfinite(com_vals)
        finite_peak = np.isfinite(peak_vals)
        start_com = float(com_vals[finite_com][0]) if np.any(finite_com) else None
        end_com = float(com_vals[finite_com][-1]) if np.any(finite_com) else None
        start_peak = float(peak_vals[finite_peak][0]) if np.any(finite_peak) else None
        end_peak = float(peak_vals[finite_peak][-1]) if np.any(finite_peak) else None
        delta_com = None if start_com is None or end_com is None else float(end_com - start_com)
        delta_peak = None if start_peak is None or end_peak is None else float(end_peak - start_peak)
        com_range = None if not np.any(finite_com) else float(np.nanmax(com_vals) - np.nanmin(com_vals))
        mean_inner_outer = float(np.nanmean(im_vals)) if np.any(np.isfinite(im_vals)) else None
        if delta_com is None:
            label = "unknown"
        elif delta_com > 0.35:
            label = "outward_shift"
        elif delta_com < -0.35:
            label = "inward_shift"
        else:
            label = "stable"
        uv = (rows[0].get("lattice_u"), rows[0].get("lattice_v"))
        out.append(
            {
                "hole_id": int(hole_id),
                "lattice_u": uv[0],
                "lattice_v": uv[1],
                "start_center_of_mass": start_com,
                "end_center_of_mass": end_com,
                "delta_center_of_mass": delta_com,
                "start_peak_annulus": start_peak,
                "end_peak_annulus": end_peak,
                "delta_peak_annulus": delta_peak,
                "com_range": com_range,
                "mean_inner_minus_outer": mean_inner_outer,
                "radial_conclusion_label": label,
                "n_frames": int(len(rows)),
            }
        )
    return out

def compute_frame_angular_asymmetry(
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
        yy_parts = []
        xx_parts = []
        for terrace in masks:
            yy, xx = terrace.global_coords() if hasattr(terrace, "global_coords") else np.nonzero(np.asarray(terrace).astype(bool))
            if yy.size:
                yy_parts.append(yy)
                xx_parts.append(xx)
        if not yy_parts:
            continue
        yy = np.concatenate(yy_parts)
        xx = np.concatenate(xx_parts)
        vals = descriptor_image[yy, xx].astype(np.float64, copy=False)
        finite, valid_count, asymmetry_val, contrast_val, vector_strength_val, dominant_sector_val = _angular_sector_stats_kernel(
            yy.astype(np.float64, copy=False),
            xx.astype(np.float64, copy=False),
            vals,
            float(hole.y),
            float(hole.x),
            int(n_sectors),
        )
        asymmetry = None if not np.isfinite(asymmetry_val) else float(asymmetry_val)
        contrast = None if not np.isfinite(contrast_val) else float(contrast_val)
        vector_strength = None if not np.isfinite(vector_strength_val) else float(vector_strength_val)
        dominant_sector = None if int(dominant_sector_val) < 0 else int(dominant_sector_val)
        uv = lattice_indices.get(int(hole.hole_id), (None, None)) if lattice_indices is not None else (None, None)
        row = {
            "frame_id": int(frame_id),
            "hole_id": int(hole.hole_id),
            "lattice_u": uv[0],
            "lattice_v": uv[1],
            "n_valid_sectors": int(valid_count),
            "angular_asymmetry": asymmetry,
            "sector_contrast": contrast,
            "vector_strength": vector_strength,
            "dominant_sector": dominant_sector,
        }
        for i in range(n_sectors):
            row[f"sector_{i}_mean"] = None if not np.isfinite(finite[i]) else float(finite[i])
        out.append(row)
    return out


def aggregate_angular_asymmetry_rows(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not rows:
        return [], []
    frame_acc: dict[int, dict[str, Any]] = {}
    hole_acc: dict[int, dict[str, Any]] = {}
    hole_meta: dict[int, tuple[Any, Any]] = {}
    for row in rows:
        frame_id = int(row.get("frame_id", 0))
        hole_id = int(row.get("hole_id", 0))
        hole_meta.setdefault(hole_id, (row.get("lattice_u"), row.get("lattice_v")))
        facc = frame_acc.setdefault(frame_id, {"n": 0, "angular": [], "contrast": [], "vector": []})
        facc["n"] += 1
        facc["angular"].append(row.get("angular_asymmetry"))
        facc["contrast"].append(row.get("sector_contrast"))
        facc["vector"].append(row.get("vector_strength"))
        hacc = hole_acc.setdefault(hole_id, {"n": 0, "angular": [], "contrast": [], "vector": [], "sector_counts": {}})
        hacc["n"] += 1
        hacc["angular"].append(row.get("angular_asymmetry"))
        hacc["contrast"].append(row.get("sector_contrast"))
        hacc["vector"].append(row.get("vector_strength"))
        ds = row.get("dominant_sector")
        if ds is not None:
            try:
                ids = int(ds)
                hacc["sector_counts"][ids] = hacc["sector_counts"].get(ids, 0) + 1
            except Exception:
                pass
    frame_out = [
        {
            "frame_id": int(frame_id),
            "mean_angular_asymmetry": _finite_mean(acc["angular"]),
            "mean_sector_contrast": _finite_mean(acc["contrast"]),
            "mean_vector_strength": _finite_mean(acc["vector"]),
            "n_holes": int(acc["n"]),
        }
        for frame_id, acc in sorted(frame_acc.items())
    ]
    hole_out = []
    for hole_id, acc in sorted(hole_acc.items()):
        uv = hole_meta.get(hole_id, (None, None))
        dominant = None
        if acc["sector_counts"]:
            dominant = max(acc["sector_counts"].items(), key=lambda kv: (kv[1], -kv[0]))[0]
        hole_out.append(
            {
                "hole_id": int(hole_id),
                "lattice_u": uv[0],
                "lattice_v": uv[1],
                "mean_angular_asymmetry": _finite_mean(acc["angular"]),
                "mean_sector_contrast": _finite_mean(acc["contrast"]),
                "mean_vector_strength": _finite_mean(acc["vector"]),
                "dominant_sector_mode": dominant,
                "n_frames": int(acc["n"]),
            }
        )
    return frame_out, hole_out

def merge_hole_radial_and_asymmetry(
    hole_summary_rows: list[dict[str, Any]],
    hole_asymmetry_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    asym_by_hole = {int(r.get("hole_id", 0)): r for r in hole_asymmetry_rows}
    out: list[dict[str, Any]] = []
    for row in hole_summary_rows:
        merged = dict(row)
        asym = asym_by_hole.get(int(row.get("hole_id", 0)), {})
        merged["mean_angular_asymmetry"] = asym.get("mean_angular_asymmetry")
        merged["mean_angular_vector_strength"] = asym.get("mean_vector_strength")
        merged["dominant_sector_mode"] = asym.get("dominant_sector_mode")
        out.append(merged)
    return out


def _prepare_matrix(rows: list[dict[str, Any]]) -> tuple[np.ndarray, list[str]]:
    feats = list(RADIAL_FEATURE_KEYS)
    X = np.asarray([[_safe_float(r.get(f)) for f in feats] for r in rows], dtype=float)
    if X.size == 0:
        return X, feats
    means = np.asarray([np.nanmean(col) if np.any(np.isfinite(col)) else 0.0 for col in X.T], dtype=float)
    inds = np.where(~np.isfinite(X))
    X[inds] = np.take(means, inds[1])
    std = np.asarray([np.nanstd(col) if np.nanstd(col) > 1e-8 else 1.0 for col in X.T], dtype=float)
    X = (X - means) / std
    return X, feats


def _simple_kmeans(X: np.ndarray, k: int, n_iter: int = 25) -> tuple[np.ndarray, np.ndarray]:
    n = X.shape[0]
    if n == 0:
        return np.empty((0,), dtype=int), np.empty((0, X.shape[1] if X.ndim == 2 else 0), dtype=float)
    k = max(1, min(int(k), n))
    order = np.linspace(0, n - 1, k, dtype=int)
    centers = X[order].copy()
    labels = np.zeros(n, dtype=int)
    for _ in range(max(1, n_iter)):
        d2 = ((X[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        new_labels = np.argmin(d2, axis=1)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        for j in range(k):
            mask = labels == j
            if np.any(mask):
                centers[j] = X[mask].mean(axis=0)
    return labels, centers


def assign_radial_archetypes(hole_summary_rows: list[dict[str, Any]], k: int = 3) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not hole_summary_rows:
        return [], []
    X, feats = _prepare_matrix(hole_summary_rows)
    labels, _centers = _simple_kmeans(X, k=min(k, len(hole_summary_rows)))
    cluster_rows: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row, lab in zip(hole_summary_rows, labels):
        cluster_rows[int(lab)].append(row)
    centroid_rows: list[dict[str, Any]] = []
    semantic_map: dict[int, str] = {}
    for cid, rows in sorted(cluster_rows.items()):
        delta = _finite_mean(r.get("delta_center_of_mass") for r in rows)
        asym = _finite_mean(r.get("mean_angular_asymmetry") for r in rows)
        if delta is None:
            semantic = "unknown"
        elif delta > 0.35:
            semantic = "outward_shift"
        elif delta < -0.35:
            semantic = "inward_shift"
        elif asym is not None and asym > 0.25:
            semantic = "stable_asymmetric"
        else:
            semantic = "stable"
        semantic_map[cid] = semantic
        centroid = {"cluster_id": int(cid), "semantic_label": semantic, "count": int(len(rows))}
        for f in feats:
            centroid[f] = _finite_mean(r.get(f) for r in rows)
        centroid_rows.append(centroid)
    centroid_rows = sorted(centroid_rows, key=lambda r: (str(r.get("semantic_label")), float(r.get("delta_center_of_mass") or 0.0)))
    canon_map: dict[int, tuple[int, str]] = {}
    for i, row in enumerate(centroid_rows, start=1):
        canon = f"R{i}_{row['semantic_label']}"
        canon_map[int(row["cluster_id"])] = (i, canon)
        row["canonical_id"] = int(i)
        row["canonical_label"] = canon
    out_rows: list[dict[str, Any]] = []
    for row, lab in zip(hole_summary_rows, labels):
        merged = dict(row)
        merged["archetype_cluster_id"] = int(lab)
        merged["radial_archetype_semantic"] = semantic_map[int(lab)]
        merged["radial_archetype_id"] = canon_map[int(lab)][0]
        merged["radial_archetype_label"] = canon_map[int(lab)][1]
        out_rows.append(merged)
    return out_rows, centroid_rows


def reticulum_zone_by_hole(rows: list[dict[str, Any]]) -> dict[int, str]:
    uv = [(int(r["hole_id"]), r.get("lattice_u"), r.get("lattice_v")) for r in rows if r.get("lattice_u") is not None and r.get("lattice_v") is not None]
    if not uv:
        return {int(r.get("hole_id", 0)): "all" for r in rows}
    us = [int(u) for _, u, _ in uv]
    vs = [int(v) for _, _, v in uv]
    min_u, max_u = min(us), max(us)
    min_v, max_v = min(vs), max(vs)
    out: dict[int, str] = {}
    for hole_id, u, v in uv:
        on_u = int(u) in {min_u, max_u}
        on_v = int(v) in {min_v, max_v}
        if on_u and on_v:
            zone = "corner"
        elif on_u or on_v:
            zone = "edge"
        else:
            zone = "interior"
        out[int(hole_id)] = zone
    return out


def build_reticulum_group_rows(radial_rows: list[dict[str, Any]], zone_by_hole: dict[int, str]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not radial_rows:
        return [], []
    buckets: dict[tuple[str, int, int], list[float]] = defaultdict(list)
    for row in radial_rows:
        zone = zone_by_hole.get(int(row.get("hole_id", 0)), "all")
        buckets[(zone, int(row.get("frame_id", 0)), int(row.get("annulus_id", 0)))].append(_safe_float(row.get("descriptor_value", np.nan)))
    group_rows: list[dict[str, Any]] = []
    for (zone, frame_id, annulus_id), vals in sorted(buckets.items()):
        arr = np.asarray(vals, dtype=float)
        arr = arr[np.isfinite(arr)]
        group_rows.append(
            {
                "reticulum_zone": zone,
                "frame_id": int(frame_id),
                "annulus_id": int(annulus_id),
                "mean_descriptor": float(arr.mean()) if arr.size else None,
                "std_descriptor": float(arr.std()) if arr.size else None,
                "n_holes": int(arr.size),
            }
        )
    summary_rows: list[dict[str, Any]] = []
    zone_buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in group_rows:
        zone_buckets[str(row["reticulum_zone"])].append(row)
    for zone, rows in sorted(zone_buckets.items()):
        by_frame: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            by_frame[int(row["frame_id"])] += [row]
        frame_summaries = []
        for frame_id, frows in sorted(by_frame.items()):
            ann = np.asarray([int(r["annulus_id"]) for r in frows], dtype=float)
            vals = np.asarray([float(r.get("mean_descriptor")) if r.get("mean_descriptor") is not None else np.nan for r in frows], dtype=float)
            finite = np.isfinite(vals)
            if not np.any(finite):
                continue
            ann = ann[finite]
            vals = vals[finite]
            shifted = vals - float(np.nanmin(vals)) + 1e-6
            com = float(np.sum(ann * shifted) / np.sum(shifted)) if np.sum(shifted) > 0 else float(np.nanmean(ann))
            frame_summaries.append((frame_id, com))
        if frame_summaries:
            summary_rows.extend(
                {
                    "reticulum_zone": zone,
                    "frame_id": int(fid),
                    "center_of_mass_annulus": float(com),
                }
                for fid, com in frame_summaries
            )
    return group_rows, summary_rows
