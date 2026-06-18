from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np


def _safe_float(value: Any, default: float = np.nan) -> float:
    try:
        return float(value) if value is not None else float(default)
    except Exception:
        return float(default)


FEATURE_KEYS = [
    "onset_slope_per_annulus",
    "peak_slope_per_annulus",
    "onset_monotonic_fraction",
    "peak_monotonic_fraction",
    "n_valid_annuli_onset",
    "n_valid_annuli_peak",
]

SEMANTIC_ORDER = {
    "outward_propagating": 0,
    "near_synchronous": 1,
    "outer_first": 2,
    "irregular": 3,
}


def _finite_or_nan(value: Any) -> float:
    if value is None:
        return float("nan")
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out if np.isfinite(out) else float("nan")


def build_propagation_feature_rows(
    per_hole_summary_rows: list[dict[str, Any]],
    per_hole_detail_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    detail_by_hole: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in per_hole_detail_rows:
        hid = row.get("hole_id")
        if hid is None:
            continue
        detail_by_hole[int(hid)].append(row)

    out: list[dict[str, Any]] = []
    for row in per_hole_summary_rows:
        hid = int(row["hole_id"])
        detail = sorted(detail_by_hole.get(hid, []), key=lambda r: int(r.get("annulus_id", 0)))
        onset_vals = np.asarray([_finite_or_nan(r.get("onset_frame")) for r in detail], dtype=float)
        peak_vals = np.asarray([_finite_or_nan(r.get("peak_frame")) for r in detail], dtype=float)

        def _monotonic_fraction(vals: np.ndarray) -> float:
            vals = vals[np.isfinite(vals)]
            if vals.size < 2:
                return 1.0 if vals.size == 1 else 0.0
            diffs = np.diff(vals)
            return float(np.mean(diffs >= 0.0))

        onset_mon_frac = _monotonic_fraction(onset_vals)
        peak_mon_frac = _monotonic_fraction(peak_vals)
        onset_lag = _finite_or_nan(row.get("onset_lag_frames"))
        peak_lag = _finite_or_nan(row.get("peak_lag_frames"))
        valid_onsets = int(row.get("n_valid_annuli_onset", 0))
        valid_peaks = int(row.get("n_valid_annuli_peak", 0))
        onset_slope = onset_lag / max(valid_onsets - 1, 1) if np.isfinite(onset_lag) else float("nan")
        peak_slope = peak_lag / max(valid_peaks - 1, 1) if np.isfinite(peak_lag) else float("nan")
        neg_lag = bool((np.isfinite(onset_lag) and onset_lag < 0) or (np.isfinite(peak_lag) and peak_lag < 0))
        out.append(
            {
                "hole_id": hid,
                "descriptor": row.get("descriptor"),
                "onset_lag_frames": None if not np.isfinite(onset_lag) else float(onset_lag),
                "peak_lag_frames": None if not np.isfinite(peak_lag) else float(peak_lag),
                "onset_slope_per_annulus": None if not np.isfinite(onset_slope) else float(onset_slope),
                "peak_slope_per_annulus": None if not np.isfinite(peak_slope) else float(peak_slope),
                "n_valid_annuli_onset": valid_onsets,
                "n_valid_annuli_peak": valid_peaks,
                "monotonic_onset": bool(row.get("monotonic_onset", False)),
                "monotonic_peak": bool(row.get("monotonic_peak", False)),
                "onset_monotonic_fraction": float(onset_mon_frac),
                "peak_monotonic_fraction": float(peak_mon_frac),
                "negative_lag_flag": neg_lag,
            }
        )
    return out


def _prepare_matrix(rows: list[dict[str, Any]]) -> tuple[np.ndarray, list[str], np.ndarray, np.ndarray]:
    feats = list(FEATURE_KEYS)
    X = np.asarray([[float(r.get(f)) if r.get(f) is not None else np.nan for f in feats] for r in rows], dtype=float)
    if X.size == 0:
        return X, feats, np.empty((0,), dtype=float), np.empty((0,), dtype=float)
    col_means = []
    for j in range(X.shape[1]):
        col = X[:, j]
        finite = col[np.isfinite(col)]
        col_means.append(float(finite.mean()) if finite.size else 0.0)
    col_means = np.asarray(col_means, dtype=float)
    inds = np.where(~np.isfinite(X))
    X[inds] = np.take(col_means, inds[1])
    std = np.nanstd(X, axis=0)
    std = np.where(std > 1e-8, std, 1.0)
    X = (X - col_means) / std
    return X, feats, col_means, std


def _simple_kmeans(X: np.ndarray, k: int, n_iter: int = 20, rng: np.random.Generator | None = None) -> tuple[np.ndarray, np.ndarray]:
    n = X.shape[0]
    if n == 0:
        return np.empty((0,), dtype=int), np.empty((0, X.shape[1] if X.ndim == 2 else 0), dtype=float)
    k = max(1, min(int(k), n))
    if rng is None:
        order = np.linspace(0, n - 1, k, dtype=int)
    else:
        order = np.asarray(rng.choice(n, size=k, replace=False), dtype=int)
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


def _label_centroid(raw_rows: list[dict[str, Any]]) -> str:
    onset_vals = np.asarray([_finite_or_nan(r.get("onset_slope_per_annulus")) for r in raw_rows], dtype=float)
    peak_vals = np.asarray([_finite_or_nan(r.get("peak_slope_per_annulus")) for r in raw_rows], dtype=float)
    mon_vals = np.asarray([_finite_or_nan(r.get("onset_monotonic_fraction")) for r in raw_rows], dtype=float)
    onset = float(onset_vals[np.isfinite(onset_vals)].mean()) if np.any(np.isfinite(onset_vals)) else float("nan")
    peak = float(peak_vals[np.isfinite(peak_vals)].mean()) if np.any(np.isfinite(peak_vals)) else float("nan")
    mon = float(mon_vals[np.isfinite(mon_vals)].mean()) if np.any(np.isfinite(mon_vals)) else float("nan")
    if np.isfinite(onset) and np.isfinite(peak) and onset >= 0.5 and peak >= 0.5 and mon >= 0.75:
        return "outward_propagating"
    if np.isfinite(onset) and np.isfinite(peak) and abs(onset) < 0.35 and abs(peak) < 0.35:
        return "near_synchronous"
    if np.isfinite(onset) and onset < 0:
        return "outer_first"
    return "irregular"


def _canonical_sort_key(row: dict[str, Any]) -> tuple[Any, ...]:
    label = str(row.get("label", "irregular"))
    return (
        SEMANTIC_ORDER.get(label, 99),
        _finite_or_nan(row.get("onset_slope_per_annulus")),
        _finite_or_nan(row.get("peak_slope_per_annulus")),
        -_finite_or_nan(row.get("onset_monotonic_fraction")),
        int(row.get("cluster_id", 0)),
    )


def _feature_vector(row: dict[str, Any]) -> np.ndarray:
    return np.asarray([_finite_or_nan(row.get(k)) for k in FEATURE_KEYS], dtype=float)


def _feature_distance(a: dict[str, Any], b: dict[str, Any]) -> float:
    va = _feature_vector(a)
    vb = _feature_vector(b)
    mask = np.isfinite(va) & np.isfinite(vb)
    if not np.any(mask):
        return float("inf")
    return float(np.sqrt(np.mean((va[mask] - vb[mask]) ** 2)))


def canonicalize_phenotype_outputs(
    phenotype_rows: list[dict[str, Any]],
    centroid_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[int, dict[str, Any]]]:
    if not centroid_rows:
        return phenotype_rows, centroid_rows, {}
    ordered = sorted(centroid_rows, key=_canonical_sort_key)
    mapping: dict[int, dict[str, Any]] = {}
    new_centroids: list[dict[str, Any]] = []
    for canonical_id, row in enumerate(ordered, start=1):
        semantic = str(row.get("label", "irregular"))
        canonical_name = f"P{canonical_id}_{semantic}"
        updated = dict(row)
        updated["canonical_id"] = int(canonical_id)
        updated["canonical_label"] = canonical_name
        mapping[int(row.get("cluster_id", canonical_id - 1))] = {
            "canonical_id": int(canonical_id),
            "canonical_label": canonical_name,
            "semantic_label": semantic,
        }
        new_centroids.append(updated)
    new_rows: list[dict[str, Any]] = []
    for row in phenotype_rows:
        info = mapping.get(int(row.get("cluster_id", -1)), {"canonical_id": None, "canonical_label": None, "semantic_label": row.get("phenotype_label")})
        updated = dict(row)
        updated["semantic_label"] = str(updated.get("phenotype_label", info.get("semantic_label", "unknown")))
        updated["canonical_id"] = info["canonical_id"]
        updated["canonical_label"] = info["canonical_label"]
        updated["phenotype_label"] = info["canonical_label"]
        new_rows.append(updated)
    return new_rows, new_centroids, mapping


def assign_hole_phenotypes(feature_rows: list[dict[str, Any]], k: int = 3) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not feature_rows:
        return [], []
    X, feats, _means, _std = _prepare_matrix(feature_rows)
    labels, _ = _simple_kmeans(X, k=min(k, len(feature_rows)))
    cluster_raw: dict[int, list[dict[str, Any]]] = {}
    for row, lab in zip(feature_rows, labels):
        cluster_raw.setdefault(int(lab), []).append(row)
    centroid_rows: list[dict[str, Any]] = []
    label_map: dict[int, str] = {}
    for cid in sorted(cluster_raw):
        raw_rows = cluster_raw[cid]
        phenotype_label = _label_centroid(raw_rows)
        label_map[cid] = phenotype_label
        payload = {
            "cluster_id": int(cid),
            "label": phenotype_label,
            "count": int(len(raw_rows)),
        }
        for key in feats:
            vals = np.asarray([_finite_or_nan(r.get(key)) for r in raw_rows], dtype=float)
            payload[key] = float(vals[np.isfinite(vals)].mean()) if np.any(np.isfinite(vals)) else None
        centroid_rows.append(payload)
    phenotype_rows: list[dict[str, Any]] = []
    for row, lab in zip(feature_rows, labels):
        out = dict(row)
        out["cluster_id"] = int(lab)
        out["phenotype_label"] = label_map[int(lab)]
        phenotype_rows.append(out)
    phenotype_rows, centroid_rows, _mapping = canonicalize_phenotype_outputs(phenotype_rows, centroid_rows)
    return phenotype_rows, centroid_rows


def _match_rerun_centroids_to_base(
    base_centroids: list[dict[str, Any]],
    rerun_centroids: list[dict[str, Any]],
) -> dict[int, dict[str, Any]]:
    if not rerun_centroids:
        return {}
    remaining = {int(r.get("cluster_id", i)): dict(r) for i, r in enumerate(rerun_centroids)}
    mapping: dict[int, dict[str, Any]] = {}
    for base in sorted(base_centroids, key=_canonical_sort_key):
        base_sem = str(base.get("label", "irregular"))
        candidates = [r for r in remaining.values() if str(r.get("label", "irregular")) == base_sem]
        if not candidates:
            candidates = list(remaining.values())
        if not candidates:
            break
        best = min(candidates, key=lambda r: _feature_distance(base, r))
        cid = int(best.get("cluster_id", -1))
        mapping[cid] = {
            "canonical_id": int(base.get("canonical_id", -1)),
            "canonical_label": str(base.get("canonical_label", base_sem)),
            "semantic_label": base_sem,
            "distance_to_base": _feature_distance(base, best),
        }
        remaining.pop(cid, None)
    for cid, row in list(remaining.items()):
        sem = str(row.get("label", "irregular"))
        mapping[cid] = {
            "canonical_id": None,
            "canonical_label": f"UNMATCHED_{sem}",
            "semantic_label": sem,
            "distance_to_base": None,
        }
    return mapping


def phenotype_stability_across_reruns(
    feature_rows: list[dict[str, Any]],
    base_phenotype_rows: list[dict[str, Any]],
    base_centroid_rows: list[dict[str, Any]] | None = None,
    k: int = 3,
    n_reruns: int = 5,
    jitter_scale: float = 0.03,
) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    if not feature_rows or not base_phenotype_rows or n_reruns <= 0:
        return [], {"mean_stability_fraction": None, "n_reruns": int(max(n_reruns, 0)), "agreement_fraction": None, "canonical_agreement_fraction": None}, []
    base_centroid_rows = list(base_centroid_rows or [])
    base_map = {int(r["hole_id"]): str(r.get("phenotype_label", "unknown")) for r in base_phenotype_rows}
    base_sem_map = {int(r["hole_id"]): str(r.get("semantic_label", r.get("phenotype_label", "unknown"))) for r in base_phenotype_rows}
    X, _feats, _means, _std = _prepare_matrix(feature_rows)
    votes: dict[int, list[str]] = defaultdict(list)
    semantic_votes: dict[int, list[str]] = defaultdict(list)
    agree_flags: list[bool] = []
    canonical_agree_flags: list[bool] = []
    rerun_rows: list[dict[str, Any]] = []
    for rerun_idx in range(int(n_reruns)):
        rng = np.random.default_rng(1000 + rerun_idx)
        order = np.asarray(rng.permutation(len(feature_rows)), dtype=int)
        X_run = X[order].copy()
        if jitter_scale > 0:
            X_run = X_run + rng.normal(0.0, jitter_scale, size=X_run.shape)
        rows_run = [dict(feature_rows[i]) for i in order]
        labels, _ = _simple_kmeans(X_run, k=min(k, len(rows_run)), rng=rng)
        cluster_raw: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for row, lab in zip(rows_run, labels):
            cluster_raw[int(lab)].append(row)
        rerun_centroids = []
        for cid, raw_rows in cluster_raw.items():
            payload = {"cluster_id": int(cid), "label": _label_centroid(raw_rows)}
            for key in FEATURE_KEYS:
                vals = np.asarray([_finite_or_nan(r.get(key)) for r in raw_rows], dtype=float)
                payload[key] = float(vals[np.isfinite(vals)].mean()) if np.any(np.isfinite(vals)) else None
            rerun_centroids.append(payload)
        centroid_alignment = _match_rerun_centroids_to_base(base_centroid_rows, rerun_centroids) if base_centroid_rows else {}
        for row, lab in zip(rows_run, labels):
            hid = int(row["hole_id"])
            semantic = str(_label_centroid(cluster_raw[int(lab)]))
            canonical = centroid_alignment.get(int(lab), {}).get("canonical_label", semantic)
            votes[hid].append(str(canonical))
            semantic_votes[hid].append(semantic)
            agree_flags.append(semantic == base_sem_map.get(hid))
            canonical_agree_flags.append(str(canonical) == base_map.get(hid))
            rerun_rows.append(
                {
                    "rerun_id": int(rerun_idx),
                    "hole_id": hid,
                    "rerun_cluster_id": int(lab),
                    "rerun_semantic_label": semantic,
                    "rerun_canonical_label": canonical,
                    "base_canonical_label": base_map.get(hid),
                    "base_semantic_label": base_sem_map.get(hid),
                    "matches_base_canonical": bool(str(canonical) == base_map.get(hid)),
                    "matches_base_semantic": bool(semantic == base_sem_map.get(hid)),
                    "distance_to_base_centroid": centroid_alignment.get(int(lab), {}).get("distance_to_base"),
                }
            )
    stability_rows: list[dict[str, Any]] = []
    stability_fracs: list[float] = []
    for hid in sorted(votes):
        counts = Counter(votes[hid])
        dominant_label, dominant_count = counts.most_common(1)[0]
        frac = float(dominant_count / max(len(votes[hid]), 1))
        stability_fracs.append(frac)
        sem_counts = Counter(semantic_votes[hid])
        dominant_sem, dominant_sem_count = sem_counts.most_common(1)[0]
        stability_rows.append(
            {
                "hole_id": int(hid),
                "base_phenotype_label": base_map.get(hid),
                "base_semantic_label": base_sem_map.get(hid),
                "dominant_rerun_label": dominant_label,
                "dominant_rerun_semantic_label": dominant_sem,
                "stability_fraction": frac,
                "semantic_stability_fraction": float(dominant_sem_count / max(len(semantic_votes[hid]), 1)),
                "n_unique_labels": int(len(counts)),
                "reruns": int(len(votes[hid])),
                "base_matches_dominant": bool(base_map.get(hid) == dominant_label),
            }
        )
    summary = {
        "mean_stability_fraction": float(np.mean(stability_fracs)) if stability_fracs else None,
        "min_stability_fraction": float(np.min(stability_fracs)) if stability_fracs else None,
        "agreement_fraction": float(np.mean(agree_flags)) if agree_flags else None,
        "canonical_agreement_fraction": float(np.mean(canonical_agree_flags)) if canonical_agree_flags else None,
        "n_reruns": int(n_reruns),
    }
    return stability_rows, summary, rerun_rows


def phenotype_neighbor_coherence(phenotype_rows: list[dict[str, Any]]) -> tuple[float | None, list[dict[str, Any]]]:
    if not phenotype_rows:
        return None, []
    by_uv: dict[tuple[int, int], dict[str, Any]] = {}
    for row in phenotype_rows:
        u = row.get("lattice_u")
        v = row.get("lattice_v")
        if u is None or v is None:
            continue
        by_uv[(int(u), int(v))] = row
    if not by_uv:
        return None, []
    pairs: list[dict[str, Any]] = []
    seen: set[tuple[tuple[int, int], tuple[int, int]]] = set()
    for (u, v), row in by_uv.items():
        for du, dv in [(1, 0), (0, 1), (-1, 0), (0, -1)]:
            nb = (u + du, v + dv)
            if nb not in by_uv:
                continue
            key = tuple(sorted(((u, v), nb)))
            if key in seen:
                continue
            seen.add(key)
            row2 = by_uv[nb]
            same = str(row.get("phenotype_label")) == str(row2.get("phenotype_label"))
            pairs.append(
                {
                    "hole_id_a": int(row["hole_id"]),
                    "hole_id_b": int(row2["hole_id"]),
                    "u_a": int(u),
                    "v_a": int(v),
                    "u_b": int(nb[0]),
                    "v_b": int(nb[1]),
                    "same_label": bool(same),
                    "label_a": row.get("phenotype_label"),
                    "label_b": row2.get("phenotype_label"),
                }
            )
    if not pairs:
        return None, []
    frac = float(np.mean([bool(p["same_label"]) for p in pairs]))
    return frac, pairs


def phenotype_spatial_smoothness(
    phenotype_rows: list[dict[str, Any]],
    radius: int = 2,
) -> tuple[float | None, list[dict[str, Any]], dict[str, Any]]:
    if not phenotype_rows:
        return None, [], {"mean_spatial_smoothness": None, "radius": int(radius), "mean_neighbor_count": 0.0}
    by_uv: dict[tuple[int, int], dict[str, Any]] = {}
    for row in phenotype_rows:
        u = row.get("lattice_u")
        v = row.get("lattice_v")
        if u is None or v is None:
            continue
        by_uv[(int(u), int(v))] = row
    if not by_uv:
        return None, [], {"mean_spatial_smoothness": None, "radius": int(radius), "mean_neighbor_count": 0.0}
    smooth_rows: list[dict[str, Any]] = []
    smoothness_vals: list[float] = []
    neighbor_counts: list[int] = []
    for (u, v), row in by_uv.items():
        weights = []
        same_weights = []
        for du in range(-radius, radius + 1):
            for dv in range(-radius, radius + 1):
                if du == 0 and dv == 0:
                    continue
                dist = abs(du) + abs(dv)
                if dist == 0 or dist > radius:
                    continue
                nb = (u + du, v + dv)
                if nb not in by_uv:
                    continue
                w = 1.0 / float(dist)
                weights.append(w)
                same_weights.append(w if str(by_uv[nb].get("phenotype_label")) == str(row.get("phenotype_label")) else 0.0)
        total_w = float(np.sum(weights)) if weights else 0.0
        smooth = float(np.sum(same_weights) / total_w) if total_w > 0 else float("nan")
        neighbor_counts.append(len(weights))
        if np.isfinite(smooth):
            smoothness_vals.append(smooth)
        smooth_rows.append(
            {
                "hole_id": int(row["hole_id"]),
                "lattice_u": int(u),
                "lattice_v": int(v),
                "phenotype_label": row.get("phenotype_label"),
                "spatial_smoothness": None if not np.isfinite(smooth) else smooth,
                "neighbor_count": int(len(weights)),
                "radius": int(radius),
            }
        )
    summary = {
        "mean_spatial_smoothness": float(np.mean(smoothness_vals)) if smoothness_vals else None,
        "min_spatial_smoothness": float(np.min(smoothness_vals)) if smoothness_vals else None,
        "radius": int(radius),
        "mean_neighbor_count": float(np.mean(neighbor_counts)) if neighbor_counts else 0.0,
    }
    mean_smooth = summary["mean_spatial_smoothness"]
    return mean_smooth, smooth_rows, summary


def build_phenotype_archetype_rows(
    radial_rows: list[dict[str, Any]],
    phenotype_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if not radial_rows or not phenotype_rows:
        return []
    label_by_hole = {int(r["hole_id"]): str(r.get("phenotype_label", "unknown")) for r in phenotype_rows}
    buckets: dict[tuple[str, int, int], list[float]] = defaultdict(list)
    for row in radial_rows:
        hid = row.get("hole_id")
        if hid is None or int(hid) not in label_by_hole:
            continue
        key = (label_by_hole[int(hid)], int(row.get("frame_id", 0)), int(row.get("annulus_id", 0)))
        buckets[key].append(_safe_float(row.get("descriptor_value", np.nan)))
    out: list[dict[str, Any]] = []
    for (label, frame_id, annulus_id), vals in sorted(buckets.items(), key=lambda kv: (kv[0][0], kv[0][1], kv[0][2])):
        arr = np.asarray(vals, dtype=float)
        finite = arr[np.isfinite(arr)]
        out.append(
            {
                "phenotype_label": label,
                "frame_id": int(frame_id),
                "annulus_id": int(annulus_id),
                "mean_descriptor": float(finite.mean()) if finite.size else None,
                "std_descriptor": float(finite.std()) if finite.size else None,
                "n_holes": int(finite.size),
            }
        )
    return out


def write_centroids_json(path: Path, centroid_rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(centroid_rows, indent=2), encoding="utf-8")
