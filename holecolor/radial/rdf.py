from __future__ import annotations

from collections import defaultdict
from itertools import groupby
from typing import Any

import numpy as np

from holecolor.config.schema import ParallelConfig
from holecolor.core.parallel import parallel_map

try:  # pragma: no cover - exercised when numba is installed
    from numba import njit
except Exception:  # pragma: no cover - keeps the package importable without numba
    def njit(*args, **kwargs):
        def deco(fn):
            return fn
        return deco


def _safe_float(x):
    try:
        return float(x)
    except Exception:
        return np.nan


def _iter_sorted_group_rows(rows: list[dict[str, Any]], sort_key, group_key=None):
    ordered = sorted(rows, key=sort_key)
    group_key = sort_key if group_key is None else group_key
    for key, grp in groupby(ordered, key=group_key):
        yield key, list(grp)


def _feature_names() -> list[str]:
    return [
        'mean_front_radius_norm',
        'delta_front_radius_norm',
        'front_radius_std_norm',
        'mean_rdf_spread_norm',
        'mean_rdf_total_positive_delta',
        'mean_abs_inner_minus_outer_delta',
    ]


def _mean_if_any(arr: np.ndarray) -> float:
    return float(np.nanmean(arr)) if np.isfinite(arr).any() else np.nan


def _hotspot_distance_stats_by_hole(hotspot_rows: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    stats: dict[int, dict[str, Any]] = {}
    for row in hotspot_rows:
        hid = row.get('nearest_hole_id')
        if hid is None:
            continue
        hole_id = int(hid)
        acc = stats.setdefault(hole_id, {'n_hotspots': 0, 'sum_dist': 0.0, 'count_dist': 0, 'min_dist': np.inf})
        acc['n_hotspots'] += 1
        dist = _safe_float(row.get('dist_to_hole_px', np.nan))
        if np.isfinite(dist):
            acc['sum_dist'] += float(dist)
            acc['count_dist'] += 1
            if dist < acc['min_dist']:
                acc['min_dist'] = float(dist)
    out: dict[int, dict[str, Any]] = {}
    for hole_id, acc in stats.items():
        count = int(acc.get('count_dist', 0))
        out[hole_id] = {
            'n_hotspots': int(acc.get('n_hotspots', 0)),
            'mean_hotspot_distance_px': None if count <= 0 else float(acc['sum_dist'] / count),
            'min_hotspot_distance_px': None if not np.isfinite(acc.get('min_dist', np.inf)) else float(acc['min_dist']),
        }
    return out


def _acc_init(fields: tuple[str, ...]) -> dict[str, Any]:
    return {'n': 0, **{f'sum__{f}': 0.0 for f in fields}, **{f'count__{f}': 0 for f in fields}}


def _acc_add(acc: dict[str, Any], field: str, value: Any) -> None:
    v = _safe_float(value)
    if np.isfinite(v):
        acc[f'sum__{field}'] += float(v)
        acc[f'count__{field}'] += 1


def _acc_mean(acc: dict[str, Any], field: str) -> float | None:
    n = int(acc.get(f'count__{field}', 0))
    if n <= 0:
        return None
    return float(acc[f'sum__{field}'] / n)


def build_per_hole_rdf_evolution(radial_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Build per-hole/frame radial distribution summaries from annulus rows.

    The radial distribution function here is a normalized distribution over annuli
    derived from the change relative to each hole/annulus baseline at the first
    available frame. Negative shifts are clipped to zero so the resulting PDF
    reflects where positive colour variation is concentrated radially.
    """
    if not radial_rows:
        return [], [], []
    baseline: dict[tuple[int, int], float] = {}
    baseline_frame: dict[int, int] = {}
    for row in sorted(radial_rows, key=lambda r: (int(r.get('hole_id', 0)), int(r.get('annulus_id', 0)), int(r.get('frame_id', 0)))):
        hole_id = int(row.get('hole_id', 0))
        annulus_id = int(row.get('annulus_id', 0))
        frame_id = int(row.get('frame_id', 0))
        key = (hole_id, annulus_id)
        if key not in baseline:
            baseline[key] = _safe_float(row.get('descriptor_value', np.nan))
            baseline_frame[hole_id] = min(frame_id, baseline_frame.get(hole_id, frame_id))
    evolution_rows: list[dict[str, Any]] = []
    frame_summary_rows: list[dict[str, Any]] = []
    velocity_rows: list[dict[str, Any]] = []
    front_by_hole: dict[int, list[tuple[int, float]]] = {}
    ordered = sorted(radial_rows, key=lambda r: (int(r.get('hole_id', 0)), int(r.get('frame_id', 0)), int(r.get('annulus_id', 0))))
    for (hole_id, frame_id), grp in groupby(ordered, key=lambda r: (int(r.get('hole_id', 0)), int(r.get('frame_id', 0)))):
        rows = list(grp)
        ann = np.asarray([int(r.get('annulus_id', 0)) for r in rows], dtype=float)
        vals = np.asarray([_safe_float(r.get('descriptor_value', np.nan)) for r in rows], dtype=float)
        baselines = np.asarray([baseline.get((hole_id, int(a)), np.nan) for a in ann], dtype=float)
        delta = vals - baselines
        positive = np.clip(delta, 0.0, None)
        total = float(np.nansum(positive))
        norm_r = ann / max(float(np.nanmax(ann)) if ann.size else 0.0, 1.0) if ann.size else ann
        pdf = positive / total if total > 0 else np.full_like(positive, np.nan, dtype=float)
        cdf = np.cumsum(np.nan_to_num(pdf, nan=0.0)) if pdf.size else pdf
        if total > 0 and np.isfinite(pdf).any():
            front_r = float(np.nansum(norm_r * pdf))
            second_m = float(np.nansum(((norm_r - front_r) ** 2) * pdf))
            spread = float(np.sqrt(max(second_m, 0.0)))
            peak_annulus = int(ann[int(np.nanargmax(pdf))])
        else:
            front_r = np.nan
            spread = np.nan
            peak_annulus = None
        inner_delta = float(delta[0]) if delta.size else np.nan
        outer_delta = float(delta[-1]) if delta.size else np.nan
        frame_summary_rows.append({
            'hole_id': int(hole_id),
            'frame_id': int(frame_id),
            'baseline_frame_id': int(baseline_frame.get(hole_id, frame_id)),
            'rdf_front_radius_norm': None if not np.isfinite(front_r) else front_r,
            'rdf_spread_norm': None if not np.isfinite(spread) else spread,
            'rdf_total_positive_delta': total,
            'rdf_peak_annulus': peak_annulus,
            'rdf_inner_delta': None if not np.isfinite(inner_delta) else inner_delta,
            'rdf_outer_delta': None if not np.isfinite(outer_delta) else outer_delta,
            'rdf_inner_minus_outer_delta': None if not (np.isfinite(inner_delta) and np.isfinite(outer_delta)) else float(inner_delta - outer_delta),
        })
        if np.isfinite(front_r):
            front_by_hole.setdefault(int(hole_id), []).append((int(frame_id), float(front_r)))
        for i, row in enumerate(rows):
            evolution_rows.append({
                'hole_id': int(hole_id),
                'frame_id': int(frame_id),
                'annulus_id': int(row.get('annulus_id', 0)),
                'normalized_radius': None if not (norm_r.size and np.isfinite(norm_r[i])) else float(norm_r[i]),
                'descriptor_value': None if not np.isfinite(vals[i]) else float(vals[i]),
                'baseline_descriptor_value': None if not np.isfinite(baselines[i]) else float(baselines[i]),
                'delta_descriptor_value': None if not np.isfinite(delta[i]) else float(delta[i]),
                'positive_delta_descriptor': None if not np.isfinite(positive[i]) else float(positive[i]),
                'rdf_pdf': None if not (pdf.size and np.isfinite(pdf[i])) else float(pdf[i]),
                'rdf_cdf': None if not (cdf.size and np.isfinite(cdf[i])) else float(cdf[i]),
            })
    for hole_id, series in sorted(front_by_hole.items()):
        series = sorted(series)
        frames = np.asarray([s[0] for s in series], dtype=float)
        radii = np.asarray([s[1] for s in series], dtype=float)
        if radii.size >= 2 and np.ptp(frames) > 0:
            slope, intercept = np.polyfit(frames, radii, 1)
            pred = slope * frames + intercept
            ss_res = float(np.sum((radii - pred) ** 2))
            ss_tot = float(np.sum((radii - np.mean(radii)) ** 2))
            r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
        elif radii.size == 1:
            slope, intercept, r2 = 0.0, float(radii[0]), 1.0
        else:
            slope, intercept, r2 = np.nan, np.nan, np.nan
        velocity_rows.append({
            'hole_id': int(hole_id),
            'rdf_front_velocity_per_frame': None if not np.isfinite(slope) else float(slope),
            'rdf_front_intercept': None if not np.isfinite(intercept) else float(intercept),
            'rdf_front_velocity_r2': None if not np.isfinite(r2) else float(r2),
            'start_front_radius_norm': None if radii.size == 0 else float(radii[0]),
            'end_front_radius_norm': None if radii.size == 0 else float(radii[-1]),
            'delta_front_radius_norm': None if radii.size == 0 else float(radii[-1] - radii[0]),
            'n_frames': int(radii.size),
        })
    return evolution_rows, frame_summary_rows, velocity_rows

def _simple_kmeans(features: np.ndarray, k: int, n_iter: int = 24) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(features, dtype=float)
    if x.ndim != 2 or x.shape[0] == 0:
        return np.zeros((0,), dtype=int), np.zeros((0, 0), dtype=float)
    finite_mask = np.isfinite(x)
    col_means = []
    for j in range(x.shape[1]):
        col = x[:, j]
        mask = np.isfinite(col)
        col_means.append(float(np.mean(col[mask])) if np.any(mask) else 0.0)
    col_means = np.asarray(col_means, dtype=float)
    x = np.where(np.isfinite(x), x, col_means)
    k = max(1, min(int(k), x.shape[0]))
    order = np.argsort(np.nanmean(x, axis=1))
    seed_idx = np.linspace(0, len(order) - 1, num=k).round().astype(int)
    centroids = x[order[seed_idx]].copy()
    labels = np.zeros((x.shape[0],), dtype=int)
    for _ in range(max(1, int(n_iter))):
        d2 = np.sum((x[:, None, :] - centroids[None, :, :]) ** 2, axis=2)
        new_labels = np.argmin(d2, axis=1)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        for j in range(k):
            mask = labels == j
            if np.any(mask):
                centroids[j] = np.mean(x[mask], axis=0)
    return labels, centroids


def build_per_hole_rdf_archetypes(rdf_frame_rows: list[dict[str, Any]], k: int = 3) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not rdf_frame_rows:
        return [], {}
    feature_names = _feature_names()
    feature_rows: list[dict[str, Any]] = []
    ordered = sorted(rdf_frame_rows, key=lambda r: (int(r.get('hole_id', 0)), int(r.get('frame_id', 0))))
    for hole_id, grp in groupby(ordered, key=lambda r: int(r.get('hole_id', 0))):
        rows = list(grp)
        front = np.asarray([_safe_float(r.get('rdf_front_radius_norm', np.nan)) for r in rows], dtype=float)
        spread = np.asarray([_safe_float(r.get('rdf_spread_norm', np.nan)) for r in rows], dtype=float)
        total = np.asarray([_safe_float(r.get('rdf_total_positive_delta', np.nan)) for r in rows], dtype=float)
        io = np.asarray([_safe_float(r.get('rdf_inner_minus_outer_delta', np.nan)) for r in rows], dtype=float)
        feature_rows.append({
            'hole_id': int(hole_id),
            'mean_front_radius_norm': _mean_if_any(front),
            'delta_front_radius_norm': float(front[-1] - front[0]) if front.size and np.isfinite(front[[0, -1]]).all() else np.nan,
            'front_radius_std_norm': float(np.nanstd(front)) if np.isfinite(front).any() else np.nan,
            'mean_rdf_spread_norm': _mean_if_any(spread),
            'mean_rdf_total_positive_delta': _mean_if_any(total),
            'mean_abs_inner_minus_outer_delta': float(np.nanmean(np.abs(io))) if np.isfinite(io).any() else np.nan,
            'n_frames': int(len(rows)),
        })
    x = np.asarray([[r.get(name, np.nan) for name in feature_names] for r in feature_rows], dtype=float)
    labels, centroids = _simple_kmeans(x, k=max(1, int(k)))
    out_rows: list[dict[str, Any]] = []
    centroid_payload: dict[str, Any] = {}
    for i, row in enumerate(feature_rows):
        label = int(labels[i]) if labels.size else 0
        out = dict(row)
        out['rdf_archetype_id'] = label
        out['rdf_archetype_label'] = f'rdf_archetype_{label}'
        out_rows.append(out)
    for idx, centroid in enumerate(np.asarray(centroids)):
        centroid_payload[f'rdf_archetype_{idx}'] = {name: None if not np.isfinite(float(val)) else float(val) for name, val in zip(feature_names, centroid)}
    return out_rows, centroid_payload

def build_per_hole_rdf_front_dynamics(rdf_frame_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not rdf_frame_rows:
        return []
    out: list[dict[str, Any]] = []
    ordered = sorted(rdf_frame_rows, key=lambda r: (int(r.get('hole_id', 0)), int(r.get('frame_id', 0))))
    for hole_id, grp in groupby(ordered, key=lambda r: int(r.get('hole_id', 0))):
        rows = list(grp)
        t = np.asarray([int(r.get('frame_id', 0)) for r in rows], dtype=float)
        y = np.asarray([_safe_float(r.get('rdf_front_radius_norm', np.nan)) for r in rows], dtype=float)
        valid = np.isfinite(t) & np.isfinite(y)
        t = t[valid]
        y = y[valid]
        if t.size == 0:
            continue
        linear_slope = linear_r2 = np.nan
        quad_a = quad_b = quad_c = quad_r2 = acceleration = nonlinear_gain = np.nan
        if t.size >= 2 and np.ptp(t) > 0:
            lin = np.polyfit(t, y, 1)
            ylin = np.polyval(lin, t)
            linear_slope = float(lin[0])
            ss_res = float(np.sum((y - ylin) ** 2))
            ss_tot = float(np.sum((y - np.mean(y)) ** 2))
            linear_r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
        if t.size >= 3 and np.ptp(t) > 0:
            quad = np.polyfit(t, y, 2)
            yquad = np.polyval(quad, t)
            quad_a, quad_b, quad_c = [float(v) for v in quad]
            ss_res = float(np.sum((y - yquad) ** 2))
            ss_tot = float(np.sum((y - np.mean(y)) ** 2))
            quad_r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
            acceleration = float(2.0 * quad_a)
            if np.isfinite(linear_r2):
                nonlinear_gain = float(quad_r2 - linear_r2)
        out.append({
            'hole_id': int(hole_id),
            'rdf_linear_velocity_per_frame': None if not np.isfinite(linear_slope) else float(linear_slope),
            'rdf_linear_velocity_r2': None if not np.isfinite(linear_r2) else float(linear_r2),
            'rdf_quadratic_a': None if not np.isfinite(quad_a) else float(quad_a),
            'rdf_quadratic_b': None if not np.isfinite(quad_b) else float(quad_b),
            'rdf_quadratic_c': None if not np.isfinite(quad_c) else float(quad_c),
            'rdf_quadratic_r2': None if not np.isfinite(quad_r2) else float(quad_r2),
            'rdf_front_acceleration_per_frame2': None if not np.isfinite(acceleration) else float(acceleration),
            'rdf_front_nonlinearity_gain': None if not np.isfinite(nonlinear_gain) else float(nonlinear_gain),
            'n_frames': int(t.size),
        })
    return out

def build_sector_rdf_evolution(sector_radial_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Build sector-resolved RDF evolution around holes.

    This mirrors the per-hole RDF but keeps sector identity, so anisotropic fronts can
    be compared sector-by-sector around the same hole.
    """
    if not sector_radial_rows:
        return [], []
    baseline: dict[tuple[int, int, int], float] = {}
    ordered_for_baseline = sorted(
        sector_radial_rows,
        key=lambda r: (int(r.get('hole_id', 0)), int(r.get('sector_id', 0)), int(r.get('annulus_id', 0)), int(r.get('frame_id', 0))),
    )
    for row in ordered_for_baseline:
        hole_id = int(row.get('hole_id', 0))
        sector_id = int(row.get('sector_id', 0))
        annulus_id = int(row.get('annulus_id', 0))
        key = (hole_id, sector_id, annulus_id)
        if key not in baseline:
            baseline[key] = _safe_float(row.get('descriptor_value', np.nan))

    evo_rows: list[dict[str, Any]] = []
    frame_rows: list[dict[str, Any]] = []
    ordered = sorted(
        sector_radial_rows,
        key=lambda r: (int(r.get('hole_id', 0)), int(r.get('frame_id', 0)), int(r.get('sector_id', 0)), int(r.get('annulus_id', 0))),
    )
    for (hole_id, frame_id, sector_id), grp in groupby(
        ordered,
        key=lambda r: (int(r.get('hole_id', 0)), int(r.get('frame_id', 0)), int(r.get('sector_id', 0))),
    ):
        rows = list(grp)
        ann = np.asarray([int(r.get('annulus_id', 0)) for r in rows], dtype=float)
        vals = np.asarray([_safe_float(r.get('descriptor_value', np.nan)) for r in rows], dtype=float)
        baselines = np.asarray([baseline.get((hole_id, sector_id, int(a)), np.nan) for a in ann], dtype=float)
        delta = vals - baselines
        positive = np.clip(delta, 0.0, None)
        total = float(np.nansum(positive))
        max_ann = max(float(np.nanmax(ann)) if ann.size else 0.0, 1.0)
        norm_r = ann / max_ann if ann.size else ann
        pdf = positive / total if total > 0 else np.full_like(positive, np.nan, dtype=float)
        cdf = np.cumsum(np.nan_to_num(pdf, nan=0.0)) if pdf.size else pdf
        front = float(np.nansum(norm_r * pdf)) if total > 0 and np.isfinite(pdf).any() else np.nan
        frame_rows.append({
            'hole_id': int(hole_id),
            'frame_id': int(frame_id),
            'sector_id': int(sector_id),
            'sector_rdf_front_radius_norm': None if not np.isfinite(front) else float(front),
            'sector_rdf_total_positive_delta': total,
            'lattice_u': rows[0].get('lattice_u'),
            'lattice_v': rows[0].get('lattice_v'),
        })
        for i, row in enumerate(rows):
            evo_rows.append({
                'hole_id': int(hole_id),
                'frame_id': int(frame_id),
                'sector_id': int(sector_id),
                'annulus_id': int(row.get('annulus_id', 0)),
                'normalized_radius': None if not (norm_r.size and np.isfinite(norm_r[i])) else float(norm_r[i]),
                'descriptor_value': None if not np.isfinite(vals[i]) else float(vals[i]),
                'baseline_descriptor_value': None if not np.isfinite(baselines[i]) else float(baselines[i]),
                'delta_descriptor_value': None if not np.isfinite(delta[i]) else float(delta[i]),
                'positive_delta_descriptor': None if not np.isfinite(positive[i]) else float(positive[i]),
                'sector_rdf_pdf': None if not (pdf.size and np.isfinite(pdf[i])) else float(pdf[i]),
                'sector_rdf_cdf': None if not (cdf.size and np.isfinite(cdf[i])) else float(cdf[i]),
                'lattice_u': row.get('lattice_u'),
                'lattice_v': row.get('lattice_v'),
            })
    return evo_rows, frame_rows

def canonicalize_rdf_archetypes(archetype_rows: list[dict[str, Any]], centroid_payload: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not archetype_rows:
        return [], {}
    items = []
    for raw_label, feats in (centroid_payload or {}).items():
        feats = feats or {}
        items.append((
            str(raw_label),
            float(feats.get("mean_front_radius_norm")) if feats.get("mean_front_radius_norm") is not None else np.nan,
            float(feats.get("delta_front_radius_norm")) if feats.get("delta_front_radius_norm") is not None else np.nan,
            float(feats.get("mean_rdf_total_positive_delta")) if feats.get("mean_rdf_total_positive_delta") is not None else np.nan,
        ))
    if not items:
        labels = sorted({str(r.get("rdf_archetype_label", f"rdf_archetype_{r.get('rdf_archetype_id',0)}")) for r in archetype_rows})
        items = [(lbl, np.nan, np.nan, np.nan) for lbl in labels]
    def _sort_key(item):
        lbl, a, b, c = item
        return (np.inf if not np.isfinite(a) else a, np.inf if not np.isfinite(b) else b, -np.inf if not np.isfinite(c) else -c, lbl)
    ordered = sorted(items, key=_sort_key)
    label_map = {raw: i for i, (raw, *_rest) in enumerate(ordered)}
    out_rows=[]
    for row in archetype_rows:
        raw = str(row.get("rdf_archetype_label", f"rdf_archetype_{row.get('rdf_archetype_id',0)}"))
        cid = int(label_map.get(raw, 0))
        new = dict(row)
        new["rdf_archetype_canonical_id"] = cid
        new["rdf_archetype_canonical_label"] = f"rdf_archetype_canonical_{cid}"
        out_rows.append(new)
    out_payload={}
    for raw, *_ in ordered:
        cid=int(label_map[raw])
        feats = dict(centroid_payload.get(raw, {}) if centroid_payload else {})
        feats["raw_label"]=raw
        feats["rdf_archetype_canonical_id"]=cid
        feats["rdf_archetype_canonical_label"]=f"rdf_archetype_canonical_{cid}"
        out_payload[f"rdf_archetype_canonical_{cid}"]=feats
    return out_rows, out_payload


def build_sector_front_lag_rows(sector_rdf_frame_rows: list[dict[str, Any]], onset_threshold: float = 0.01) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not sector_rdf_frame_rows:
        return [], []
    lag_rows: list[dict[str, Any]] = []
    by_hole: dict[int, list[dict[str, Any]]] = defaultdict(list)
    ordered = sorted(sector_rdf_frame_rows, key=lambda r: (int(r.get('hole_id', 0)), int(r.get('sector_id', 0)), int(r.get('frame_id', 0))))
    for (hole_id, sector_id), grp in groupby(ordered, key=lambda r: (int(r.get('hole_id', 0)), int(r.get('sector_id', 0)))):
        rows = list(grp)
        frames = np.asarray([int(r.get('frame_id', 0)) for r in rows], dtype=float)
        delta = np.asarray([_safe_float(r.get('sector_rdf_total_positive_delta', np.nan)) for r in rows], dtype=float)
        front = np.asarray([_safe_float(r.get('sector_rdf_front_radius_norm', np.nan)) for r in rows], dtype=float)
        finite_delta = np.isfinite(delta)
        finite_front = np.isfinite(front)
        onset_frame = None
        if np.any(finite_delta):
            idx = np.where((delta > float(onset_threshold)) & finite_delta)[0]
            if idx.size:
                onset_frame = int(frames[idx[0]])
        peak_frame = None
        if np.any(finite_front):
            peak_frame = int(frames[np.nanargmax(front)])
        row0 = rows[0]
        lag_row = {
            'hole_id': int(hole_id),
            'sector_id': int(sector_id),
            'lattice_u': row0.get('lattice_u'),
            'lattice_v': row0.get('lattice_v'),
            'sector_onset_frame': onset_frame,
            'sector_peak_frame': peak_frame,
            'sector_peak_front_radius_norm': None if not np.any(finite_front) else float(np.nanmax(front)),
            'sector_mean_front_radius_norm': None if not np.any(finite_front) else float(np.nanmean(front)),
            'valid_frames': int(np.sum(finite_front)),
        }
        lag_rows.append(lag_row)
        by_hole[int(hole_id)].append(lag_row)
    summary_rows = []
    for hole_id, rows in sorted(by_hole.items()):
        onset_vals = [r['sector_onset_frame'] for r in rows if r.get('sector_onset_frame') is not None]
        peak_vals = [r['sector_peak_frame'] for r in rows if r.get('sector_peak_frame') is not None]
        min_on = min(onset_vals) if onset_vals else None
        min_pk = min(peak_vals) if peak_vals else None
        for r in rows:
            r['sector_onset_lag'] = None if min_on is None or r.get('sector_onset_frame') is None else int(r['sector_onset_frame'] - min_on)
            r['sector_peak_lag'] = None if min_pk is None or r.get('sector_peak_frame') is None else int(r['sector_peak_frame'] - min_pk)
        onset_lags = [r['sector_onset_lag'] for r in rows if r.get('sector_onset_lag') is not None]
        peak_lags = [r['sector_peak_lag'] for r in rows if r.get('sector_peak_lag') is not None]
        summary_rows.append({
            'hole_id': int(hole_id),
            'lattice_u': rows[0].get('lattice_u'),
            'lattice_v': rows[0].get('lattice_v'),
            'min_sector_onset_frame': min_on,
            'min_sector_peak_frame': min_pk,
            'max_sector_onset_lag': None if not onset_lags else int(max(onset_lags)),
            'max_sector_peak_lag': None if not peak_lags else int(max(peak_lags)),
            'mean_sector_onset_lag': None if not onset_lags else float(np.mean(onset_lags)),
            'mean_sector_peak_lag': None if not peak_lags else float(np.mean(peak_lags)),
            'valid_onset_sector_fraction': float(len(onset_lags) / len(rows)) if rows else 0.0,
            'valid_peak_sector_fraction': float(len(peak_lags) / len(rows)) if rows else 0.0,
        })
    return lag_rows, summary_rows

def _rdf_feature_row_from_rows(hole_id: int, rows: list[dict[str, Any]]) -> dict[str, Any]:
    rows = sorted(rows, key=lambda r: int(r.get('frame_id', 0)))
    front = np.asarray([_safe_float(r.get('rdf_front_radius_norm', np.nan)) for r in rows], dtype=float)
    spread = np.asarray([_safe_float(r.get('rdf_spread_norm', np.nan)) for r in rows], dtype=float)
    total = np.asarray([_safe_float(r.get('rdf_total_positive_delta', np.nan)) for r in rows], dtype=float)
    io = np.asarray([_safe_float(r.get('rdf_inner_minus_outer_delta', np.nan)) for r in rows], dtype=float)
    return {
        'hole_id': int(hole_id),
        'mean_front_radius_norm': float(np.nanmean(front)) if np.isfinite(front).any() else np.nan,
        'delta_front_radius_norm': float(front[-1] - front[0]) if front.size and np.isfinite(front[[0, -1]]).all() else np.nan,
        'front_radius_std_norm': float(np.nanstd(front)) if np.isfinite(front).any() else np.nan,
        'mean_rdf_spread_norm': float(np.nanmean(spread)) if np.isfinite(spread).any() else np.nan,
        'mean_rdf_total_positive_delta': float(np.nanmean(total)) if np.isfinite(total).any() else np.nan,
        'mean_abs_inner_minus_outer_delta': float(np.nanmean(np.abs(io))) if np.isfinite(io).any() else np.nan,
        'n_frames': int(len(rows)),
    }


def _canonical_centroid_matrix(canonical_centroids: dict[str, Any]) -> tuple[list[str], np.ndarray]:
    feature_names = [
        'mean_front_radius_norm',
        'delta_front_radius_norm',
        'front_radius_std_norm',
        'mean_rdf_spread_norm',
        'mean_rdf_total_positive_delta',
        'mean_abs_inner_minus_outer_delta',
    ]
    labels = sorted(canonical_centroids.keys())
    if not labels:
        return [], np.zeros((0, len(feature_names)), dtype=float)
    mat = []
    for lbl in labels:
        feats = canonical_centroids.get(lbl, {}) or {}
        mat.append([_safe_float(feats.get(name, np.nan)) for name in feature_names])
    arr = np.asarray(mat, dtype=float)
    if arr.size:
        col_means = np.nanmean(np.where(np.isfinite(arr), arr, np.nan), axis=0)
        col_means = np.where(np.isfinite(col_means), col_means, 0.0)
        arr = np.where(np.isfinite(arr), arr, col_means)
    return labels, arr


def _assign_feature_to_centroids(feature_row: dict[str, Any], canonical_centroids: dict[str, Any]) -> tuple[str | None, int | None]:
    labels, centroids = _canonical_centroid_matrix(canonical_centroids)
    if centroids.size == 0:
        return None, None
    feat = np.asarray([[
        _safe_float(feature_row.get('mean_front_radius_norm', np.nan)),
        _safe_float(feature_row.get('delta_front_radius_norm', np.nan)),
        _safe_float(feature_row.get('front_radius_std_norm', np.nan)),
        _safe_float(feature_row.get('mean_rdf_spread_norm', np.nan)),
        _safe_float(feature_row.get('mean_rdf_total_positive_delta', np.nan)),
        _safe_float(feature_row.get('mean_abs_inner_minus_outer_delta', np.nan)),
    ]], dtype=float)
    col_means = np.nanmean(centroids, axis=0)
    col_means = np.where(np.isfinite(col_means), col_means, 0.0)
    feat = np.where(np.isfinite(feat), feat, col_means)
    d2 = np.sum((centroids - feat) ** 2, axis=1)
    idx = int(np.argmin(d2))
    lbl = labels[idx]
    cid = None
    try:
        cid = int((canonical_centroids.get(lbl, {}) or {}).get('rdf_archetype_canonical_id'))
    except Exception:
        cid = idx
    return lbl, cid


@njit(cache=True)
def _bootstrap_front_summary_kernel(unique_frames: np.ndarray, frame_slot: np.ndarray, front: np.ndarray, sample_idx: np.ndarray):
    n_boot = sample_idx.shape[0]
    n_sample = sample_idx.shape[1]
    n_unique = unique_frames.size
    mean_front = np.empty(n_boot, dtype=np.float64)
    delta_front = np.empty(n_boot, dtype=np.float64)
    velocity = np.empty(n_boot, dtype=np.float64)
    for boot in range(n_boot):
        sums = np.zeros(n_unique, dtype=np.float64)
        counts = np.zeros(n_unique, dtype=np.int64)
        for j in range(n_sample):
            src = int(sample_idx[boot, j])
            slot = int(frame_slot[src])
            sums[slot] += float(front[src])
            counts[slot] += 1

        used = 0
        sum_y = 0.0
        sum_x = 0.0
        sum_xx = 0.0
        sum_xy = 0.0
        first_y = 0.0
        last_y = 0.0
        have_first = False
        for slot in range(n_unique):
            if counts[slot] <= 0:
                continue
            x = float(unique_frames[slot])
            y = sums[slot] / float(counts[slot])
            if not have_first:
                first_y = y
                have_first = True
            last_y = y
            used += 1
            sum_y += y
            sum_x += x
            sum_xx += x * x
            sum_xy += x * y

        if used > 0:
            mean_front[boot] = sum_y / float(used)
            delta_front[boot] = last_y - first_y
        else:
            mean_front[boot] = np.nan
            delta_front[boot] = np.nan
        if used >= 2:
            den = float(used) * sum_xx - sum_x * sum_x
            if abs(den) > 1e-12:
                velocity[boot] = (float(used) * sum_xy - sum_x * sum_y) / den
            else:
                velocity[boot] = np.nan
        else:
            velocity[boot] = np.nan
    return mean_front, delta_front, velocity


def _bootstrap_summary_for_hole(task: tuple[int, list[dict[str, Any]], int, int]) -> dict[str, Any] | None:
    hole_id, rows, n_boot, rng_seed = task
    rows = sorted(rows, key=lambda r: int(r.get('frame_id', 0)))
    frames = np.asarray([int(r.get('frame_id', 0)) for r in rows], dtype=float)
    front = np.asarray([_safe_float(r.get('rdf_front_radius_norm', np.nan)) for r in rows], dtype=float)
    valid = np.isfinite(front)
    front = front[valid]
    frames = frames[valid]
    if front.size == 0:
        return None
    rng = np.random.default_rng(int(rng_seed) + int(hole_id))
    n_boot_eff = max(8, int(n_boot))
    sample_idx = rng.integers(0, len(front), size=(n_boot_eff, len(front))).astype(np.int64, copy=False)
    unique_frames, frame_slot = np.unique(frames, return_inverse=True)
    mean_front_samples, delta_front_samples, vel_samples = _bootstrap_front_summary_kernel(
        unique_frames.astype(np.float64, copy=False),
        frame_slot.astype(np.int64, copy=False),
        front.astype(np.float64, copy=False),
        sample_idx,
    )
    vel_samples = vel_samples[np.isfinite(vel_samples)]
    def _q(arr, q):
        arr = np.asarray(arr, dtype=float)
        arr = arr[np.isfinite(arr)]
        return float(np.nanquantile(arr, q)) if arr.size else np.nan
    mean_front_samples = mean_front_samples[np.isfinite(mean_front_samples)]
    delta_front_samples = delta_front_samples[np.isfinite(delta_front_samples)]
    return {
        'hole_id': int(hole_id),
        'n_boot': int(n_boot_eff),
        'bootstrap_mean_front_radius_norm': float(np.nanmean(mean_front_samples)) if mean_front_samples.size else None,
        'bootstrap_mean_front_radius_ci_low': None if not mean_front_samples.size else _q(mean_front_samples, 0.025),
        'bootstrap_mean_front_radius_ci_high': None if not mean_front_samples.size else _q(mean_front_samples, 0.975),
        'bootstrap_delta_front_radius_norm': float(np.nanmean(delta_front_samples)) if delta_front_samples.size else None,
        'bootstrap_delta_front_radius_ci_low': None if not delta_front_samples.size else _q(delta_front_samples, 0.025),
        'bootstrap_delta_front_radius_ci_high': None if not delta_front_samples.size else _q(delta_front_samples, 0.975),
        'bootstrap_front_velocity_per_frame': float(np.nanmean(vel_samples)) if vel_samples.size else None,
        'bootstrap_front_velocity_ci_low': None if not vel_samples.size else _q(vel_samples, 0.025),
        'bootstrap_front_velocity_ci_high': None if not vel_samples.size else _q(vel_samples, 0.975),
        'bootstrap_front_velocity_ci_width': None if not vel_samples.size else float(_q(vel_samples, 0.975) - _q(vel_samples, 0.025)),
    }


def _bootstrap_support_for_hole(task: tuple[int, list[dict[str, Any]], dict[str, Any], int, int]) -> dict[str, Any] | None:
    hole_id, rows, canonical_centroids, n_boot, rng_seed = task
    rows = sorted(rows, key=lambda r: int(r.get('frame_id', 0)))
    rng = np.random.default_rng(int(rng_seed) + int(hole_id))
    support_counts: dict[str, int] = defaultdict(int)
    total = 0
    n_boot_eff = max(8, int(n_boot))
    for _ in range(n_boot_eff):
        idx = rng.integers(0, len(rows), size=len(rows))
        bs_rows = [rows[int(i)] for i in idx]
        feat = _rdf_feature_row_from_rows(hole_id, bs_rows)
        lbl, cid = _assign_feature_to_centroids(feat, canonical_centroids)
        if lbl is not None:
            support_counts[str(lbl)] += 1
            total += 1
    if total == 0:
        return None
    best_label, best_count = sorted(support_counts.items(), key=lambda kv: (-kv[1], kv[0]))[0]
    best_payload = canonical_centroids.get(best_label, {}) or {}
    return {
        'hole_id': int(hole_id),
        'n_boot': int(total),
        'bootstrap_rdf_archetype_label': str(best_label),
        'bootstrap_rdf_archetype_canonical_id': best_payload.get('rdf_archetype_canonical_id'),
        'bootstrap_rdf_archetype_support_fraction': float(best_count / total),
        'bootstrap_rdf_archetype_n_labels': int(len(support_counts)),
    }


def build_per_hole_rdf_bootstrap_summary(rdf_frame_rows: list[dict[str, Any]], n_boot: int = 128, rng_seed: int = 0, parallel_cfg: ParallelConfig | dict[str, Any] | None = None) -> list[dict[str, Any]]:
    if not rdf_frame_rows:
        return []
    by_hole: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rdf_frame_rows:
        by_hole[int(row.get('hole_id', 0))].append(dict(row))
    tasks = [(int(hole_id), rows, int(n_boot), int(rng_seed)) for hole_id, rows in sorted(by_hole.items())]
    out = parallel_map(_bootstrap_summary_for_hole, tasks, parallel_cfg, desc='RDF bootstrap summary')
    return [row for row in out if row is not None]


def build_rdf_archetype_bootstrap_support(rdf_frame_rows: list[dict[str, Any]], canonical_centroids: dict[str, Any], n_boot: int = 128, rng_seed: int = 0, parallel_cfg: ParallelConfig | dict[str, Any] | None = None) -> list[dict[str, Any]]:
    if not rdf_frame_rows or not canonical_centroids:
        return []
    by_hole: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rdf_frame_rows:
        by_hole[int(row.get('hole_id', 0))].append(dict(row))
    tasks = [(int(hole_id), rows, canonical_centroids, int(n_boot), int(rng_seed)) for hole_id, rows in sorted(by_hole.items())]
    out = parallel_map(_bootstrap_support_for_hole, tasks, parallel_cfg, desc='RDF archetype support')
    return [row for row in out if row is not None]


def build_sector_front_propagation(sector_rdf_frame_rows: list[dict[str, Any]], onset_threshold: float = 0.01) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not sector_rdf_frame_rows:
        return [], []
    by_hole_sector: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in sector_rdf_frame_rows:
        by_hole_sector[(int(row.get('hole_id', 0)), int(row.get('sector_id', 0)))].append(dict(row))
    sector_rows = []
    by_hole_summary: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for (hole_id, sector_id), rows in sorted(by_hole_sector.items()):
        rows = sorted(rows, key=lambda r: int(r.get('frame_id', 0)))
        frames = np.asarray([int(r.get('frame_id', 0)) for r in rows], dtype=float)
        front = np.asarray([_safe_float(r.get('sector_rdf_front_radius_norm', np.nan)) for r in rows], dtype=float)
        total = np.asarray([_safe_float(r.get('sector_rdf_total_positive_delta', np.nan)) for r in rows], dtype=float)
        valid = np.isfinite(frames) & np.isfinite(front)
        slope = r2 = np.nan
        if np.sum(valid) >= 2 and np.ptp(frames[valid]) > 0:
            coeff = np.polyfit(frames[valid], front[valid], 1)
            pred = np.polyval(coeff, frames[valid])
            ss_res = float(np.sum((front[valid] - pred) ** 2))
            ss_tot = float(np.sum((front[valid] - np.mean(front[valid])) ** 2))
            r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
            slope = float(coeff[0])
        onset = None
        tv = np.isfinite(total)
        if np.any(tv):
            idx = np.where((total > float(onset_threshold)) & tv)[0]
            if idx.size:
                onset = int(frames[idx[0]])
        row0 = rows[0]
        out = {
            'hole_id': int(hole_id),
            'sector_id': int(sector_id),
            'lattice_u': row0.get('lattice_u'),
            'lattice_v': row0.get('lattice_v'),
            'sector_front_velocity_per_frame': None if not np.isfinite(slope) else float(slope),
            'sector_front_velocity_r2': None if not np.isfinite(r2) else float(r2),
            'sector_front_onset_frame': onset,
            'sector_front_start_radius_norm': None if not np.any(valid) else float(front[valid][0]),
            'sector_front_end_radius_norm': None if not np.any(valid) else float(front[valid][-1]),
            'sector_front_delta_radius_norm': None if not np.any(valid) else float(front[valid][-1] - front[valid][0]),
            'valid_frames': int(np.sum(valid)),
        }
        sector_rows.append(out)
        by_hole_summary[int(hole_id)].append(out)
    hole_rows = []
    for hole_id, rows in sorted(by_hole_summary.items()):
        vel = np.asarray([_safe_float(r.get('sector_front_velocity_per_frame', np.nan)) for r in rows], dtype=float)
        velv = vel[np.isfinite(vel)]
        onsets = [int(r['sector_front_onset_frame']) for r in rows if r.get('sector_front_onset_frame') is not None]
        lags = [o - min(onsets) for o in onsets] if onsets else []
        hole_rows.append({
            'hole_id': int(hole_id),
            'lattice_u': rows[0].get('lattice_u'),
            'lattice_v': rows[0].get('lattice_v'),
            'mean_sector_front_velocity_per_frame': None if velv.size == 0 else float(np.mean(velv)),
            'std_sector_front_velocity_per_frame': None if velv.size == 0 else float(np.std(velv)),
            'sector_front_velocity_anisotropy': None if velv.size == 0 else float(np.std(velv) / max(abs(np.mean(velv)), 1e-6)),
            'valid_sector_velocity_fraction': float(velv.size / max(len(rows), 1)),
            'max_sector_onset_lag': None if not lags else int(max(lags)),
            'mean_sector_onset_lag': None if not lags else float(np.mean(lags)),
        })
    return sector_rows, hole_rows



def build_rdf_uncertainty_reticulum_rows(
    rdf_archetype_rows: list[dict[str, Any]],
    rdf_bootstrap_rows: list[dict[str, Any]],
    rdf_bootstrap_support_rows: list[dict[str, Any]],
    rdf_dynamics_rows: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    if not rdf_archetype_rows:
        return []
    boot_by_hole = {int(r.get('hole_id', 0)): dict(r) for r in rdf_bootstrap_rows}
    support_by_hole = {int(r.get('hole_id', 0)): dict(r) for r in rdf_bootstrap_support_rows}
    dyn_by_hole = {int(r.get('hole_id', 0)): dict(r) for r in (rdf_dynamics_rows or [])}
    out = []
    for row in rdf_archetype_rows:
        hole_id = int(row.get('hole_id', 0))
        boot = boot_by_hole.get(hole_id, {})
        sup = support_by_hole.get(hole_id, {})
        dyn = dyn_by_hole.get(hole_id, {})
        support = _safe_float(sup.get('bootstrap_rdf_archetype_support_fraction', np.nan))
        vel_ci = _safe_float(boot.get('bootstrap_front_velocity_ci_width', np.nan))
        front_ci = _safe_float(boot.get('bootstrap_mean_front_radius_ci_width', np.nan))
        delta_ci = _safe_float(boot.get('bootstrap_delta_front_radius_ci_width', np.nan))
        uncertainty = np.nan
        pieces = []
        if np.isfinite(vel_ci): pieces.append(abs(vel_ci))
        if np.isfinite(front_ci): pieces.append(abs(front_ci))
        if np.isfinite(delta_ci): pieces.append(abs(delta_ci))
        if np.isfinite(support): pieces.append(max(0.0, 1.0 - support))
        if pieces:
            uncertainty = float(np.nanmean(np.asarray(pieces, dtype=float)))
        out.append({
            'hole_id': hole_id,
            'lattice_u': row.get('lattice_u'),
            'lattice_v': row.get('lattice_v'),
            'rdf_archetype_canonical_id': row.get('rdf_archetype_canonical_id'),
            'rdf_archetype_canonical_label': row.get('rdf_archetype_canonical_label'),
            'bootstrap_rdf_archetype_support_fraction': None if not np.isfinite(support) else float(support),
            'bootstrap_mean_front_radius_ci_width': None if not np.isfinite(front_ci) else float(front_ci),
            'bootstrap_delta_front_radius_ci_width': None if not np.isfinite(delta_ci) else float(delta_ci),
            'bootstrap_front_velocity_ci_width': None if not np.isfinite(vel_ci) else float(vel_ci),
            'rdf_front_acceleration_per_frame2': dyn.get('rdf_front_acceleration_per_frame2'),
            'rdf_front_nonlinearity_gain': dyn.get('rdf_front_nonlinearity_gain'),
            'rdf_uncertainty_score': None if not np.isfinite(uncertainty) else float(uncertainty),
        })
    return out


def build_rdf_uncertainty_hotspot_comparison(
    hotspot_rows: list[dict[str, Any]],
    rdf_bootstrap_rows: list[dict[str, Any]],
    rdf_bootstrap_support_rows: list[dict[str, Any]],
    rdf_archetype_rows: list[dict[str, Any]],
    zone_by_hole: dict[int, str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not rdf_archetype_rows:
        return [], []
    hotspot_stats = _hotspot_distance_stats_by_hole(hotspot_rows)
    boot_by_hole = {int(r.get('hole_id', 0)): dict(r) for r in rdf_bootstrap_rows}
    support_by_hole = {int(r.get('hole_id', 0)): dict(r) for r in rdf_bootstrap_support_rows}
    out = []
    group_acc: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rdf_archetype_rows:
        hole_id = int(row.get('hole_id', 0))
        hs = hotspot_stats.get(hole_id, {})
        mean_dist = hs.get('mean_hotspot_distance_px')
        min_dist = hs.get('min_hotspot_distance_px')
        bucket = 'none'
        if min_dist is not None:
            if min_dist <= 10:
                bucket = 'near'
            elif min_dist <= 24:
                bucket = 'mid'
            else:
                bucket = 'far'
        boot = boot_by_hole.get(hole_id, {})
        sup = support_by_hole.get(hole_id, {})
        out_row = {
            'hole_id': hole_id,
            'lattice_u': row.get('lattice_u'),
            'lattice_v': row.get('lattice_v'),
            'reticulum_zone': zone_by_hole.get(hole_id, 'unknown'),
            'rdf_archetype_canonical_id': row.get('rdf_archetype_canonical_id'),
            'rdf_archetype_canonical_label': row.get('rdf_archetype_canonical_label'),
            'n_hotspots': int(hs.get('n_hotspots', 0)),
            'mean_hotspot_distance_px': mean_dist,
            'min_hotspot_distance_px': min_dist,
            'hotspot_proximity_bucket': bucket,
            'bootstrap_rdf_archetype_support_fraction': sup.get('bootstrap_rdf_archetype_support_fraction'),
            'bootstrap_front_velocity_ci_width': boot.get('bootstrap_front_velocity_ci_width'),
            'bootstrap_mean_front_radius_ci_width': boot.get('bootstrap_mean_front_radius_ci_width'),
            'bootstrap_delta_front_radius_ci_width': boot.get('bootstrap_delta_front_radius_ci_width'),
        }
        out.append(out_row)
        key = (str(out_row.get('reticulum_zone', 'unknown')), str(bucket))
        acc = group_acc.setdefault(key, _acc_init((
            'bootstrap_rdf_archetype_support_fraction',
            'bootstrap_front_velocity_ci_width',
            'bootstrap_mean_front_radius_ci_width',
        )))
        acc['n'] += 1
        _acc_add(acc, 'bootstrap_rdf_archetype_support_fraction', out_row.get('bootstrap_rdf_archetype_support_fraction'))
        _acc_add(acc, 'bootstrap_front_velocity_ci_width', out_row.get('bootstrap_front_velocity_ci_width'))
        _acc_add(acc, 'bootstrap_mean_front_radius_ci_width', out_row.get('bootstrap_mean_front_radius_ci_width'))
    group_rows = []
    for (zone, bucket), acc in sorted(group_acc.items()):
        group_rows.append({
            'reticulum_zone': zone,
            'hotspot_proximity_bucket': bucket,
            'n_holes': int(acc.get('n', 0)),
            'mean_bootstrap_support_fraction': _acc_mean(acc, 'bootstrap_rdf_archetype_support_fraction'),
            'mean_bootstrap_front_velocity_ci_width': _acc_mean(acc, 'bootstrap_front_velocity_ci_width'),
            'mean_bootstrap_front_radius_ci_width': _acc_mean(acc, 'bootstrap_mean_front_radius_ci_width'),
        })
    return out, group_rows

def build_sector_front_acceleration(sector_rdf_frame_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not sector_rdf_frame_rows:
        return [], []
    sector_rows = []
    by_hole = defaultdict(list)
    ordered = sorted(sector_rdf_frame_rows, key=lambda r: (int(r.get('hole_id', 0)), int(r.get('sector_id', 0)), int(r.get('frame_id', 0))))
    for (hole_id, sector_id), grp in groupby(ordered, key=lambda r: (int(r.get('hole_id', 0)), int(r.get('sector_id', 0)))):
        rows = list(grp)
        frames = np.asarray([int(r.get('frame_id', 0)) for r in rows], dtype=float)
        front = np.asarray([_safe_float(r.get('sector_rdf_front_radius_norm', np.nan)) for r in rows], dtype=float)
        valid = np.isfinite(frames) & np.isfinite(front)
        if int(np.sum(valid)) < 3 or np.ptp(frames[valid]) <= 0:
            continue
        xf = frames[valid]
        yf = front[valid]
        lin = np.polyfit(xf, yf, 1)
        ylin = np.polyval(lin, xf)
        ss_res = float(np.sum((yf - ylin) ** 2))
        ss_tot = float(np.sum((yf - np.mean(yf)) ** 2))
        lin_r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
        quad = np.polyfit(xf, yf, 2)
        yquad = np.polyval(quad, xf)
        ss_res_q = float(np.sum((yf - yquad) ** 2))
        quad_r2 = 1.0 - ss_res_q / ss_tot if ss_tot > 0 else 1.0
        a, b, c = [float(v) for v in quad]
        out = {
            'hole_id': int(hole_id),
            'sector_id': int(sector_id),
            'lattice_u': rows[0].get('lattice_u'),
            'lattice_v': rows[0].get('lattice_v'),
            'sector_front_linear_velocity_per_frame': float(lin[0]),
            'sector_front_linear_r2': float(lin_r2),
            'sector_front_quadratic_a': a,
            'sector_front_quadratic_b': b,
            'sector_front_quadratic_c': c,
            'sector_front_quadratic_r2': float(quad_r2),
            'sector_front_acceleration_per_frame2': float(2.0 * a),
            'sector_front_curvature': float(a),
            'sector_front_nonlinearity_gain': float(quad_r2 - lin_r2),
            'valid_frames': int(np.sum(valid)),
        }
        sector_rows.append(out)
        by_hole[int(hole_id)].append(out)
    summary_rows = []
    for hole_id, rows in sorted(by_hole.items()):
        acc = np.asarray([_safe_float(r.get('sector_front_acceleration_per_frame2', np.nan)) for r in rows], dtype=float)
        curv = np.asarray([_safe_float(r.get('sector_front_curvature', np.nan)) for r in rows], dtype=float)
        nlg = np.asarray([_safe_float(r.get('sector_front_nonlinearity_gain', np.nan)) for r in rows], dtype=float)
        summary_rows.append({
            'hole_id': int(hole_id),
            'lattice_u': rows[0].get('lattice_u'),
            'lattice_v': rows[0].get('lattice_v'),
            'mean_sector_front_acceleration_per_frame2': None if not np.isfinite(acc).any() else float(np.nanmean(acc)),
            'sector_front_acceleration_spread': None if not np.isfinite(acc).any() else float(np.nanstd(acc)),
            'mean_abs_sector_front_curvature': None if not np.isfinite(curv).any() else float(np.nanmean(np.abs(curv))),
            'mean_sector_front_nonlinearity_gain': None if not np.isfinite(nlg).any() else float(np.nanmean(nlg)),
            'valid_sector_acceleration_fraction': float(np.mean(np.isfinite(acc))) if acc.size else 0.0,
            'n_valid_sectors': int(np.sum(np.isfinite(acc))),
        })
    return sector_rows, summary_rows
