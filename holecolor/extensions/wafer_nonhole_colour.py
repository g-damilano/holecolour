
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np
import pandas as pd

from holecolor.core.types import FrameRecord, HoleGeometry
from holecolor.descriptors.cluster_labels import predict_hsl_cluster_labels
from holecolor.descriptors.color_spaces import rgb_to_hsl
from holecolor.core.types import VideoMeta
from holecolor.masks.matrix import make_global_hole_union
from holecolor.qc.reports import save_image, write_json, write_table


@dataclass(slots=True)
class WaferNonholeColourBundle:
    status: str
    support_mask: np.ndarray | None
    frame_region_rows: list[dict[str, Any]]
    cluster_rows: list[dict[str, Any]]
    frame_cluster_summary_rows: list[dict[str, Any]]
    frame_cluster_hard_rows: list[dict[str, Any]]
    frame_cluster_soft_rows: list[dict[str, Any]]
    global_context_rows: list[dict[str, Any]]
    local_context_rows: list[dict[str, Any]]
    selected_k: int | None
    message: str | None = None
    model_selection_rows: list[dict[str, Any]] = field(default_factory=list)


def _circle_mask(shape: tuple[int, int], circle: tuple[int, int, int]) -> np.ndarray:
    h, w = shape
    yy, xx = np.indices((h, w))
    xc, yc, rc = circle
    return ((xx - xc) ** 2 + (yy - yc) ** 2) <= (float(rc) ** 2)


def support_mask_from_debug(shape: tuple[int, int], support_mask: np.ndarray | None, support_circle: tuple[int, int, int] | None) -> np.ndarray:
    if support_mask is not None:
        mask = np.asarray(support_mask, dtype=bool)
        if mask.shape == shape:
            return mask
    if support_circle is not None:
        return _circle_mask(shape, support_circle)
    return np.ones(shape, dtype=bool)


def _sample_indices(mask: np.ndarray, max_points: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    yy, xx = np.where(mask)
    if len(xx) <= max_points:
        return yy, xx
    idx = rng.choice(len(xx), size=max_points, replace=False)
    return yy[idx], xx[idx]


def _overlay_region(image: np.ndarray, support_mask: np.ndarray, hole_union: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    overlay = image.copy()
    if overlay.ndim == 2:
        overlay = np.repeat(overlay[..., None], 3, axis=2)
    overlay = overlay.astype(np.float32)
    support_edge = cv2.morphologyEx((support_mask.astype(np.uint8) * 255), cv2.MORPH_GRADIENT, np.ones((3,3), np.uint8)) > 0
    hole_edge = cv2.morphologyEx((hole_union.astype(np.uint8) * 255), cv2.MORPH_GRADIENT, np.ones((3,3), np.uint8)) > 0
    tint = overlay.copy()
    tint[..., 1] = np.where(valid_mask, np.clip(tint[..., 1] * 0.6 + 120, 0, 255), tint[..., 1])
    overlay = np.where(valid_mask[..., None], 0.55 * overlay + 0.45 * tint, overlay)
    overlay[support_edge] = np.array([255, 255, 255], dtype=np.float32)
    overlay[hole_edge] = np.array([255, 64, 64], dtype=np.float32)
    return np.clip(overlay, 0, 255).astype(np.uint8)


def extract_wafer_nonhole_region(
    frames: list[FrameRecord],
    holes_by_frame: dict[int, list[HoleGeometry]],
    support_mask: np.ndarray,
    max_points_per_frame: int,
    rng_seed: int,
) -> tuple[list[dict[str, Any]], dict[int, dict[str, np.ndarray]], np.ndarray]:
    rng = np.random.default_rng(rng_seed)
    frame_rows: list[dict[str, Any]] = []
    samples_by_frame: dict[int, dict[str, np.ndarray]] = {}
    ref_overlay: np.ndarray | None = None
    for frame in frames:
        holes = holes_by_frame[frame.frame_id]
        hole_union = make_global_hole_union(frame.image.shape[:2], holes)
        valid_mask = support_mask & (~hole_union)
        n_valid = int(valid_mask.sum())
        yy, xx = _sample_indices(valid_mask, max_points_per_frame, rng)
        if len(xx) > 0:
            hsl = rgb_to_hsl(frame.image)
            pts_hsl = hsl[yy, xx]
            h = pts_hsl[:, 0].astype(np.float32)
            s = pts_hsl[:, 1].astype(np.float32)
            l = pts_hsl[:, 2].astype(np.float32)
            hx = s * np.cos(2 * np.pi * h)
            hy = s * np.sin(2 * np.pi * h)
            samples_by_frame[frame.frame_id] = {
                'x': xx.astype(np.int32),
                'y': yy.astype(np.int32),
                'r': frame.image[yy, xx, 0].astype(np.uint8),
                'g': frame.image[yy, xx, 1].astype(np.uint8),
                'b': frame.image[yy, xx, 2].astype(np.uint8),
                'h': h,
                's': s,
                'l': l,
                'hx': hx.astype(np.float32),
                'hy': hy.astype(np.float32),
            }
        else:
            samples_by_frame[frame.frame_id] = {
                'x': np.empty((0,), dtype=np.int32),
                'y': np.empty((0,), dtype=np.int32),
                'r': np.empty((0,), dtype=np.uint8),
                'g': np.empty((0,), dtype=np.uint8),
                'b': np.empty((0,), dtype=np.uint8),
                'h': np.empty((0,), dtype=np.float32),
                's': np.empty((0,), dtype=np.float32),
                'l': np.empty((0,), dtype=np.float32),
                'hx': np.empty((0,), dtype=np.float32),
                'hy': np.empty((0,), dtype=np.float32),
            }
        frame_rows.append({
            'frame_id': int(frame.frame_id),
            'time_s': float(frame.time_s),
            'n_support_px': int(support_mask.sum()),
            'n_hole_px': int(hole_union.sum()),
            'n_valid_px': n_valid,
            'n_sampled_px': int(len(xx)),
            'valid_fraction_of_support': float(n_valid / max(1, int(support_mask.sum()))),
        })
        if ref_overlay is None:
            ref_overlay = _overlay_region(frame.image, support_mask, hole_union, valid_mask)
    if ref_overlay is None:
        ref_overlay = np.zeros((*support_mask.shape, 3), dtype=np.uint8)
    return frame_rows, samples_by_frame, ref_overlay


def _diag_covariances(gmm: Any, n_features: int) -> np.ndarray:
    cov = np.asarray(getattr(gmm, "covariances_", np.ones((int(gmm.n_components), n_features))), dtype=np.float64)
    if cov.ndim == 1:
        return np.repeat(cov[:, None], n_features, axis=1)
    if cov.ndim == 2:
        if cov.shape[0] == int(gmm.n_components):
            return cov[:, :n_features]
        return np.repeat(np.diag(cov)[None, :n_features], int(gmm.n_components), axis=0)
    return np.stack([np.diag(c)[:n_features] for c in cov], axis=0)


def _minimum_center_sigma_separation(gmm: Any) -> float:
    means = np.asarray(gmm.means_, dtype=np.float64)
    k, n_features = means.shape
    if k <= 1:
        return float("inf")
    cov = np.maximum(_diag_covariances(gmm, n_features), 1e-9)
    best = float("inf")
    for i in range(k):
        for j in range(i + 1, k):
            pooled = 0.5 * (cov[i] + cov[j])
            diff = means[i] - means[j]
            sep = float(np.sqrt(np.mean((diff * diff) / pooled)))
            best = min(best, sep)
    return best


def _choose_gmm_candidate(candidates: list[dict[str, Any]], k_min: int) -> dict[str, Any] | None:
    if not candidates:
        return None
    eligible = [c for c in candidates if bool(c.get("is_relevant", False))]
    if not eligible:
        eligible = [min(candidates, key=lambda c: float(c["icl"]))]
    best_icl = min(float(c["icl"]) for c in eligible)
    strong_evidence_window = 10.0
    for candidate in sorted(eligible, key=lambda c: int(c["k"])):
        if float(candidate["icl"]) <= best_icl + strong_evidence_window:
            return candidate
    return min(eligible, key=lambda c: (float(c["icl"]), int(c["k"])))


def _fit_gmm(emb: np.ndarray, k_min: int, k_max: int, random_state: int) -> tuple[Any | None, list[dict[str, Any]], int | None, str | None]:
    try:
        from sklearn.mixture import GaussianMixture
    except Exception as exc:  # pragma: no cover
        return None, [], None, f'sklearn_unavailable: {exc}'
    if emb.shape[0] < max(8, k_min):
        return None, [], None, 'insufficient_points_for_clustering'
    candidates: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    upper = max(k_min, min(k_max, emb.shape[0]))
    n_samples = int(emb.shape[0])
    relevance_weight_floor = float(1.0 / max(1.0, np.sqrt(float(n_samples))))
    for k in range(k_min, upper + 1):
        try:
            gmm = GaussianMixture(n_components=k, covariance_type='diag', random_state=random_state, reg_covar=1e-6, n_init=1, max_iter=80)
            gmm.fit(emb)
            bic = float(gmm.bic(emb))
            probs = np.asarray(gmm.predict_proba(emb), dtype=np.float64)
            entropy = float(-np.sum(probs * np.log(np.clip(probs, 1e-12, 1.0))))
            icl = float(bic + 2.0 * entropy)
            min_weight = float(np.min(gmm.weights_))
            min_sep = _minimum_center_sigma_separation(gmm)
            candidate = {
                'k': int(k),
                'bic': bic,
                'classification_entropy': entropy,
                'icl': icl,
                'min_component_weight': min_weight,
                'component_weight_floor': relevance_weight_floor,
                'min_center_sigma_separation': min_sep,
                'center_separation_floor': None,
                'is_relevant': False,
                'gmm': gmm,
            }
            candidates.append(candidate)
        except Exception:
            continue
    base_candidates = [c for c in candidates if int(c['k']) == int(k_min)]
    base_sep = float(base_candidates[0]['min_center_sigma_separation']) if base_candidates else float(candidates[0]['min_center_sigma_separation']) if candidates else 0.0
    sep_floor = 0.8 * base_sep if np.isfinite(base_sep) and base_sep > 0 else 0.0
    for candidate in candidates:
        candidate['center_separation_floor'] = sep_floor
        candidate['is_relevant'] = bool(
            int(candidate['k']) <= int(k_min)
            or (
                float(candidate['min_component_weight']) >= relevance_weight_floor
                and float(candidate['min_center_sigma_separation']) >= sep_floor
            )
        )
    selected = _choose_gmm_candidate(candidates, k_min)
    if selected is None:
        return None, rows, None, 'gmm_fit_failed'
    selected_k = int(selected['k'])
    best_bic = min(float(c['bic']) for c in candidates)
    best_icl = min(float(c['icl']) for c in candidates)
    rows = []
    for candidate in candidates:
        rows.append({
            'k': int(candidate['k']),
            'bic': float(candidate['bic']),
            'bic_delta_from_best': float(candidate['bic'] - best_bic),
            'classification_entropy': float(candidate['classification_entropy']),
            'icl': float(candidate['icl']),
            'icl_delta_from_best': float(candidate['icl'] - best_icl),
            'min_component_weight': float(candidate['min_component_weight']),
            'component_weight_floor': float(candidate['component_weight_floor']),
            'min_center_sigma_separation': float(candidate['min_center_sigma_separation']),
            'center_separation_floor': float(candidate['center_separation_floor']),
            'is_relevant': bool(candidate['is_relevant']),
            'selected': bool(int(candidate['k']) == selected_k),
        })
    return selected['gmm'], rows, int(selected['gmm'].n_components), None


def _cluster_center_rows(gmm) -> list[dict[str, Any]]:
    rows = []
    for i, (mu, w) in enumerate(zip(gmm.means_, gmm.weights_)):
        hx, hy, l = [float(v) for v in mu[:3]]
        s = float(np.clip(np.hypot(hx, hy), 0.0, 1.0))
        h = float((np.arctan2(hy, hx) / (2 * np.pi)) % 1.0)
        rows.append({
            'cluster_id': int(i),
            'weight': float(w),
            'center_h': h,
            'center_s': s,
            'center_l': float(np.clip(l, 0.0, 1.0)),
            'center_hx': hx,
            'center_hy': hy,
        })
    return rows


def _update_cluster_rows_with_sample_observed_rgb(
    samples_by_frame: dict[int, dict[str, np.ndarray]],
    cluster_rows: list[dict[str, Any]],
    max_sample_points_per_cluster: int = 20000,
) -> bool:
    if not cluster_rows:
        return False
    required = {'hx', 'hy', 'l', 'r', 'g', 'b'}
    if not all(required.issubset(set(samples.keys())) for samples in samples_by_frame.values()):
        return False
    n_clusters = len(cluster_rows)
    centers = np.asarray([[float(r['center_hx']), float(r['center_hy']), float(r['center_l'])] for r in cluster_rows], dtype=np.float32)
    pixel_counts = np.zeros(n_clusters, dtype=np.int64)
    pixel_sums = np.zeros((n_clusters, 3), dtype=np.float64)
    sampled_counts = np.zeros(n_clusters, dtype=np.int64)
    sampled_chunks: list[list[np.ndarray]] = [[] for _ in range(n_clusters)]
    for samples in samples_by_frame.values():
        n = int(len(samples.get('hx', [])))
        if n <= 0:
            continue
        emb = np.column_stack([samples['hx'], samples['hy'], samples['l']]).astype(np.float32)
        d2 = np.sum((emb[:, None, :] - centers[None, :, :]) ** 2, axis=2)
        labels = np.argmin(d2, axis=1).astype(np.int32)
        rgb = np.column_stack([samples['r'], samples['g'], samples['b']]).astype(np.uint8)
        for label_idx in range(n_clusters):
            idx = labels == label_idx
            count = int(np.sum(idx))
            if count <= 0:
                continue
            cluster_rgb = rgb[idx]
            pixel_counts[label_idx] += count
            pixel_sums[label_idx] += cluster_rgb.astype(np.float64).sum(axis=0)
            remaining = int(max_sample_points_per_cluster - sampled_counts[label_idx])
            if remaining <= 0:
                continue
            take = min(remaining, count)
            if count > take:
                step = max(1, int(np.ceil(count / max(1, take))))
                sample = cluster_rgb[::step][:take]
            else:
                sample = cluster_rgb[:take]
            sampled_chunks[label_idx].append(sample.astype(np.uint8, copy=False))
            sampled_counts[label_idx] += int(len(sample))
    updated = False
    for label_idx, row in enumerate(cluster_rows):
        n = int(pixel_counts[label_idx])
        row['analysis_n_pixels'] = n
        row['analysis_sampled_pixels'] = int(sampled_counts[label_idx])
        if n <= 0:
            row.setdefault('display_source', 'hsl_centroid_rgb')
            continue
        mean_rgb = pixel_sums[label_idx] / float(n)
        sample_rgb = np.vstack(sampled_chunks[label_idx]).astype(np.float32) if sampled_chunks[label_idx] else mean_rgb[None, :]
        median_rgb = np.median(sample_rgb, axis=0)
        for name, values in (('analysis_mean', mean_rgb), ('analysis_median', median_rgb)):
            row[f'{name}_r'] = int(np.clip(round(float(values[0])), 0, 255))
            row[f'{name}_g'] = int(np.clip(round(float(values[1])), 0, 255))
            row[f'{name}_b'] = int(np.clip(round(float(values[2])), 0, 255))
        row['display_r'] = int(row['analysis_median_r'])
        row['display_g'] = int(row['analysis_median_g'])
        row['display_b'] = int(row['analysis_median_b'])
        row['display_source'] = 'analysis_observed_sample_median_rgb'
        row['display_colour_space'] = 'photometry_corrected_registered_rgb'
        updated = True
    return updated


def _balanced_fit_embedding(samples_by_frame: dict[int, dict[str, np.ndarray]], max_total_points: int, rng: np.random.Generator) -> np.ndarray:
    nonempty = [(fid, s) for fid, s in samples_by_frame.items() if len(s['hx']) > 0]
    if not nonempty:
        return np.empty((0, 3), dtype=np.float32)
    per_frame_cap = max(1, max_total_points // max(1, len(nonempty)))
    chunks = []
    for _, s in nonempty:
        n = len(s['hx'])
        take = min(n, per_frame_cap)
        if n > take:
            idx = rng.choice(n, size=take, replace=False)
        else:
            idx = np.arange(n)
        chunks.append(np.column_stack([s['hx'][idx], s['hy'][idx], s['l'][idx]]).astype(np.float32))
    emb = np.vstack(chunks) if chunks else np.empty((0, 3), dtype=np.float32)
    if len(emb) > max_total_points:
        idx = rng.choice(len(emb), size=max_total_points, replace=False)
        emb = emb[idx]
    return emb


def build_wafer_nonhole_colour_bundle_from_samples(
    frame_rows: list[dict[str, Any]],
    samples_by_frame: dict[int, dict[str, np.ndarray]],
    support_mask: np.ndarray,
    max_total_fit_points: int = 4000,
    min_total_points: int = 500,
    k_min: int = 2,
    k_max: int = 5,
    random_state: int = 0,
) -> WaferNonholeColourBundle:
    rng = np.random.default_rng(random_state)
    total_points = int(sum(len(s['hx']) for s in samples_by_frame.values()))
    if total_points < min_total_points:
        return WaferNonholeColourBundle('skipped', support_mask, frame_rows, [], [], [], [], [], [], None, message='insufficient_total_points')
    emb = _balanced_fit_embedding(samples_by_frame, max_total_points=max_total_fit_points, rng=rng)
    if len(emb) < min_total_points:
        return WaferNonholeColourBundle('skipped', support_mask, frame_rows, [], [], [], [], [], [], None, message='insufficient_fit_sample_points')
    gmm, gmm_rows, selected_k, msg = _fit_gmm(emb, k_min=k_min, k_max=k_max, random_state=random_state)
    if gmm is None:
        return WaferNonholeColourBundle('skipped', support_mask, frame_rows, [], [], [], [], [], [], selected_k, message=msg, model_selection_rows=gmm_rows)
    cluster_rows = _cluster_center_rows(gmm)
    _update_cluster_rows_with_sample_observed_rgb(samples_by_frame, cluster_rows)
    frame_cluster_summary_rows = []
    frame_cluster_hard_rows = []
    frame_cluster_soft_rows = []
    for row in frame_rows:
        fid = int(row['frame_id'])
        s = samples_by_frame[fid]
        if len(s['hx']) == 0:
            continue
        emb_f = np.column_stack([s['hx'], s['hy'], s['l']]).astype(np.float32)
        probs = gmm.predict_proba(emb_f)
        hard = np.argmax(probs, axis=1)
        hard_counts = np.bincount(hard, minlength=gmm.n_components).astype(np.float64)
        soft_counts = probs.sum(axis=0).astype(np.float64)
        hard_freq = hard_counts / max(1.0, hard_counts.sum())
        soft_freq = soft_counts / max(1.0, soft_counts.sum())
        entropy = float(-np.sum(np.where(soft_freq > 0, soft_freq * np.log(soft_freq + 1e-12), 0.0)))
        dom = int(np.argmax(soft_freq))
        dom_row = cluster_rows[dom]
        frame_cluster_summary_rows.append({
            'frame_id': int(fid),
            'time_s': float(row.get('time_s', fid)),
            'n_sampled_px': int(len(emb_f)),
            'selected_k': int(gmm.n_components),
            'dominant_cluster_id': dom,
            'dominant_soft_prevalence': float(soft_freq[dom]),
            'cluster_entropy': entropy,
            'dominant_center_h': dom_row['center_h'],
            'dominant_center_s': dom_row['center_s'],
            'dominant_center_l': dom_row['center_l'],
        })
        for cid in range(gmm.n_components):
            frame_cluster_hard_rows.append({'frame_id': int(fid), 'cluster_id': int(cid), 'hard_prevalence': float(hard_freq[cid])})
            frame_cluster_soft_rows.append({'frame_id': int(fid), 'cluster_id': int(cid), 'soft_prevalence': float(soft_freq[cid])})
    by_frame = {int(r['frame_id']): r for r in frame_cluster_summary_rows}
    global_context_rows: list[dict[str, Any]] = []
    local_context_rows: list[dict[str, Any]] = []
    return WaferNonholeColourBundle(
        'ok',
        support_mask,
        frame_rows,
        cluster_rows,
        frame_cluster_summary_rows,
        frame_cluster_hard_rows,
        frame_cluster_soft_rows,
        global_context_rows,
        local_context_rows,
        int(gmm.n_components),
        message='ok',
        model_selection_rows=gmm_rows,
    )


def enrich_global_matrix_rows(matrix_rows: list[dict[str, Any]], frame_cluster_summary_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_frame = {int(r['frame_id']): r for r in frame_cluster_summary_rows}
    out = []
    for row in matrix_rows:
        fid = int(row['frame_id'])
        ctx = by_frame.get(fid, {})
        rec = dict(row)
        rec.update({
            'dominant_cluster_id': ctx.get('dominant_cluster_id'),
            'dominant_soft_prevalence': ctx.get('dominant_soft_prevalence'),
            'cluster_entropy': ctx.get('cluster_entropy'),
            'dominant_center_h': ctx.get('dominant_center_h'),
            'dominant_center_s': ctx.get('dominant_center_s'),
            'dominant_center_l': ctx.get('dominant_center_l'),
        })
        out.append(rec)
    return out


def enrich_local_compartment_rows(compartment_rows: list[dict[str, Any]], frame_cluster_summary_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_frame = {int(r['frame_id']): r for r in frame_cluster_summary_rows}
    out = []
    for row in compartment_rows:
        fid = int(row['frame_id'])
        ctx = by_frame.get(fid, {})
        rec = dict(row)
        rec.update({
            'dominant_cluster_id': ctx.get('dominant_cluster_id'),
            'dominant_soft_prevalence': ctx.get('dominant_soft_prevalence'),
            'cluster_entropy': ctx.get('cluster_entropy'),
            'dominant_center_h': ctx.get('dominant_center_h'),
            'dominant_center_s': ctx.get('dominant_center_s'),
            'dominant_center_l': ctx.get('dominant_center_l'),
        })
        if ctx:
            mh = rec.get('mean_H')
            ms = rec.get('mean_S')
            if mh is not None and ctx.get('dominant_center_h') is not None:
                dh = float(((float(mh) - float(ctx['dominant_center_h']) + 0.5) % 1.0) - 0.5)
                rec['delta_to_dominant_cluster_h'] = dh
            if ms is not None and ctx.get('dominant_center_s') is not None:
                rec['delta_to_dominant_cluster_s'] = float(ms) - float(ctx['dominant_center_s'])
        out.append(rec)
    return out


def write_wafer_nonhole_colour_artifacts(base_dir: Path, bundle: WaferNonholeColourBundle, overlay: np.ndarray | None = None, sampled_points: dict[int, dict[str, np.ndarray]] | None = None) -> None:
    write_json(base_dir / 'stage_status.json', {
        'status': bundle.status,
        'message': bundle.message,
        'selected_k': bundle.selected_k,
        'n_frame_rows': len(bundle.frame_region_rows),
        'n_cluster_rows': len(bundle.cluster_rows),
        'n_frame_cluster_summary_rows': len(bundle.frame_cluster_summary_rows),
        'model_selection': bundle.model_selection_rows,
    })
    write_table(base_dir / 'frame_region_summary.csv', bundle.frame_region_rows)
    if bundle.model_selection_rows:
        write_table(base_dir / 'cluster_model_selection.csv', bundle.model_selection_rows)
    if bundle.cluster_rows:
        write_table(base_dir / 'cluster_model_summary.csv', bundle.cluster_rows)
    if bundle.frame_cluster_summary_rows:
        write_table(base_dir / 'frame_cluster_summary.csv', bundle.frame_cluster_summary_rows)
    if bundle.frame_cluster_hard_rows:
        write_table(base_dir / 'frame_cluster_prevalence_hard.csv', bundle.frame_cluster_hard_rows)
    if bundle.frame_cluster_soft_rows:
        write_table(base_dir / 'frame_cluster_prevalence_soft.csv', bundle.frame_cluster_soft_rows)
    if bundle.global_context_rows:
        write_table(base_dir / 'global_buffer_cluster_context.csv', bundle.global_context_rows)
    if bundle.local_context_rows:
        write_table(base_dir / 'local_hole_cluster_context.csv', bundle.local_context_rows)
    if overlay is not None:
        save_image(base_dir / 'frame0_region_overlay.png', overlay)
    if sampled_points is not None:
        arrays = {}
        for fid, vals in sampled_points.items():
            for key, arr in vals.items():
                arrays[f'frame_{int(fid):04d}_{key}'] = np.asarray(arr)
        np.savez_compressed(base_dir / 'canonical_hsl_samples.npz', **arrays)


def build_wafer_nonhole_colour_bundle(
    frames: list[FrameRecord],
    holes_by_frame: dict[int, list[HoleGeometry]],
    support_mask: np.ndarray,
    max_points_per_frame: int = 5000,
    max_total_fit_points: int = 4000,
    min_total_points: int = 500,
    k_min: int = 2,
    k_max: int = 5,
    random_state: int = 0,
) -> WaferNonholeColourBundle:
    frame_rows, samples_by_frame, _ = extract_wafer_nonhole_region(frames, holes_by_frame, support_mask, max_points_per_frame=max_points_per_frame, rng_seed=random_state)
    return build_wafer_nonhole_colour_bundle_from_samples(
        frame_rows, samples_by_frame, support_mask,
        max_total_fit_points=max_total_fit_points,
        min_total_points=min_total_points,
        k_min=k_min, k_max=k_max, random_state=random_state,
    )


def _estimate_fps(frames: list[FrameRecord]) -> float:
    if len(frames) < 2:
        return 1.0
    dt = []
    for a, b in zip(frames[:-1], frames[1:]):
        d = float(b.time_s) - float(a.time_s)
        if d > 1e-9:
            dt.append(d)
    if not dt:
        return 1.0
    return float(1.0 / max(1e-9, float(np.median(np.asarray(dt, dtype=np.float64)))))


def _hsl_to_rgb_uint8(h: np.ndarray, s: np.ndarray, l: np.ndarray) -> np.ndarray:
    h = np.asarray(h, dtype=np.float32)
    s = np.asarray(s, dtype=np.float32)
    l = np.asarray(l, dtype=np.float32)
    c = (1.0 - np.abs(2.0 * l - 1.0)) * s
    hp = (h % 1.0) * 6.0
    x = c * (1.0 - np.abs((hp % 2.0) - 1.0))
    z = np.zeros_like(hp, dtype=np.float32)
    r1 = np.select([(0 <= hp) & (hp < 1), (1 <= hp) & (hp < 2), (2 <= hp) & (hp < 3), (3 <= hp) & (hp < 4), (4 <= hp) & (hp < 5), (5 <= hp) & (hp < 6)], [c, x, z, z, x, c], default=z)
    g1 = np.select([(0 <= hp) & (hp < 1), (1 <= hp) & (hp < 2), (2 <= hp) & (hp < 3), (3 <= hp) & (hp < 4), (4 <= hp) & (hp < 5), (5 <= hp) & (hp < 6)], [x, c, c, x, z, z], default=z)
    b1 = np.select([(0 <= hp) & (hp < 1), (1 <= hp) & (hp < 2), (2 <= hp) & (hp < 3), (3 <= hp) & (hp < 4), (4 <= hp) & (hp < 5), (5 <= hp) & (hp < 6)], [z, z, x, c, c, x], default=z)
    m = l - 0.5 * c
    rgb = np.stack([r1 + m, g1 + m, b1 + m], axis=-1)
    return np.clip(np.round(rgb * 255.0), 0, 255).astype(np.uint8)


def _row_rgb_triplet(row: dict[str, Any], prefix: str) -> tuple[int, int, int] | None:
    keys = (f'{prefix}_r', f'{prefix}_g', f'{prefix}_b')
    if not all(k in row and row[k] is not None for k in keys):
        return None
    try:
        rgb = tuple(int(np.clip(round(float(row[k])), 0, 255)) for k in keys)
    except (TypeError, ValueError):
        return None
    return rgb  # type: ignore[return-value]


def _cluster_palette_rows(cluster_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not cluster_rows:
        return []
    h = np.asarray([float(r['center_h']) for r in cluster_rows], dtype=np.float32)
    s = np.asarray([float(r['center_s']) for r in cluster_rows], dtype=np.float32)
    l = np.asarray([float(r['center_l']) for r in cluster_rows], dtype=np.float32)
    centroid_rgb = _hsl_to_rgb_uint8(h, s, l)
    rows = []
    label_palette = np.array([
        [230, 25, 75], [60, 180, 75], [0, 130, 200], [245, 130, 48],
        [145, 30, 180], [70, 240, 240], [240, 50, 230], [210, 245, 60],
    ], dtype=np.uint8)
    for i, row in enumerate(cluster_rows):
        observed_rgb = _row_rgb_triplet(row, 'analysis_median')
        display_rgb = _row_rgb_triplet(row, 'display') or observed_rgb
        display_source = str(row.get('display_source') or ('analysis_observed_median_rgb' if observed_rgb is not None else 'hsl_centroid_rgb'))
        if display_rgb is None:
            display_rgb = (int(centroid_rgb[i, 0]), int(centroid_rgb[i, 1]), int(centroid_rgb[i, 2]))
        record = {
            'cluster_id': int(row['cluster_id']),
            'display_r': int(display_rgb[0]),
            'display_g': int(display_rgb[1]),
            'display_b': int(display_rgb[2]),
            'display_source': display_source,
            'centroid_r': int(centroid_rgb[i, 0]),
            'centroid_g': int(centroid_rgb[i, 1]),
            'centroid_b': int(centroid_rgb[i, 2]),
            'label_r': int(label_palette[i % len(label_palette), 0]),
            'label_g': int(label_palette[i % len(label_palette), 1]),
            'label_b': int(label_palette[i % len(label_palette), 2]),
        }
        for prefix in ('analysis_median', 'analysis_mean', 'display_median', 'display_mean'):
            rgb = _row_rgb_triplet(row, prefix)
            if rgb is not None:
                record[f'{prefix}_r'] = int(rgb[0])
                record[f'{prefix}_g'] = int(rgb[1])
                record[f'{prefix}_b'] = int(rgb[2])
        for key in ('analysis_n_pixels', 'analysis_sampled_pixels', 'display_n_pixels', 'display_sampled_pixels'):
            if key in row:
                record[key] = int(row[key])
        if 'display_colour_space' in row:
            record['display_colour_space'] = row['display_colour_space']
        rows.append({
            **record,
        })
    return rows


def _update_cluster_rows_with_observed_rgb(
    frames: list[FrameRecord],
    holes_by_frame: dict[int, list[HoleGeometry]],
    support_mask: np.ndarray,
    bundle: WaferNonholeColourBundle,
    max_sample_points_per_cluster: int = 20000,
) -> None:
    if bundle.status != 'ok' or not bundle.cluster_rows or not frames:
        return
    rows = bundle.cluster_rows
    n_clusters = len(rows)
    pixel_counts = np.zeros(n_clusters, dtype=np.int64)
    pixel_sums = np.zeros((n_clusters, 3), dtype=np.float64)
    sampled_counts = np.zeros(n_clusters, dtype=np.int64)
    sampled_chunks: list[list[np.ndarray]] = [[] for _ in range(n_clusters)]
    for frame in frames:
        holes = holes_by_frame.get(frame.frame_id, [])
        hole_union = make_global_hole_union(frame.image.shape[:2], holes)
        valid_mask = support_mask & (~hole_union)
        labels = _predict_cluster_labels_for_frame(frame.image, valid_mask, rows)
        labelled = labels >= 0
        if not np.any(labelled):
            continue
        yy, xx = np.where(labelled)
        lab = labels[yy, xx]
        pix = frame.image[yy, xx]
        for label_idx in range(n_clusters):
            idx = lab == label_idx
            n = int(np.sum(idx))
            if n <= 0:
                continue
            cluster_pix = pix[idx]
            pixel_counts[label_idx] += n
            pixel_sums[label_idx] += cluster_pix.astype(np.float64).sum(axis=0)
            remaining = int(max_sample_points_per_cluster - sampled_counts[label_idx])
            if remaining <= 0:
                continue
            take = min(remaining, n)
            if n > take:
                step = max(1, int(np.ceil(n / max(1, take))))
                sample = cluster_pix[::step][:take]
            else:
                sample = cluster_pix[:take]
            sampled_chunks[label_idx].append(sample.astype(np.uint8, copy=False))
            sampled_counts[label_idx] += int(len(sample))
    for label_idx, row in enumerate(rows):
        n = int(pixel_counts[label_idx])
        row['analysis_n_pixels'] = n
        row['analysis_sampled_pixels'] = int(sampled_counts[label_idx])
        if n <= 0:
            row.setdefault('display_source', 'hsl_centroid_rgb')
            continue
        mean_rgb = pixel_sums[label_idx] / float(n)
        if sampled_chunks[label_idx]:
            sample_rgb = np.vstack(sampled_chunks[label_idx]).astype(np.float32)
            median_rgb = np.median(sample_rgb, axis=0)
        else:
            median_rgb = mean_rgb
        for name, values in (('analysis_mean', mean_rgb), ('analysis_median', median_rgb)):
            row[f'{name}_r'] = int(np.clip(round(float(values[0])), 0, 255))
            row[f'{name}_g'] = int(np.clip(round(float(values[1])), 0, 255))
            row[f'{name}_b'] = int(np.clip(round(float(values[2])), 0, 255))
        row['display_r'] = int(row['analysis_median_r'])
        row['display_g'] = int(row['analysis_median_g'])
        row['display_b'] = int(row['analysis_median_b'])
        row['display_source'] = 'analysis_observed_pixel_median_rgb'
        row['display_colour_space'] = 'photometry_corrected_registered_rgb'


def _update_cluster_rows_with_display_rgb(
    label_frames: list[FrameRecord],
    colour_frames: list[FrameRecord],
    holes_by_frame: dict[int, list[HoleGeometry]],
    support_mask: np.ndarray,
    bundle: WaferNonholeColourBundle,
    source_label: str,
    colour_space_label: str,
    max_sample_points_per_cluster: int = 20000,
) -> None:
    if bundle.status != 'ok' or not bundle.cluster_rows or not label_frames or not colour_frames:
        return
    by_frame = {int(frame.frame_id): frame for frame in colour_frames}
    rows = bundle.cluster_rows
    n_clusters = len(rows)
    pixel_counts = np.zeros(n_clusters, dtype=np.int64)
    pixel_sums = np.zeros((n_clusters, 3), dtype=np.float64)
    sampled_counts = np.zeros(n_clusters, dtype=np.int64)
    sampled_chunks: list[list[np.ndarray]] = [[] for _ in range(n_clusters)]
    for frame in label_frames:
        colour_frame = by_frame.get(int(frame.frame_id))
        if colour_frame is None or colour_frame.image.shape[:2] != frame.image.shape[:2]:
            continue
        holes = holes_by_frame.get(frame.frame_id, [])
        hole_union = make_global_hole_union(frame.image.shape[:2], holes)
        valid_mask = support_mask & (~hole_union)
        labels = _predict_cluster_labels_for_frame(frame.image, valid_mask, rows)
        labelled = labels >= 0
        if not np.any(labelled):
            continue
        yy, xx = np.where(labelled)
        lab = labels[yy, xx]
        pix = colour_frame.image[yy, xx]
        for label_idx in range(n_clusters):
            idx = lab == label_idx
            n = int(np.sum(idx))
            if n <= 0:
                continue
            cluster_pix = pix[idx]
            pixel_counts[label_idx] += n
            pixel_sums[label_idx] += cluster_pix.astype(np.float64).sum(axis=0)
            remaining = int(max_sample_points_per_cluster - sampled_counts[label_idx])
            if remaining <= 0:
                continue
            take = min(remaining, n)
            if n > take:
                step = max(1, int(np.ceil(n / max(1, take))))
                sample = cluster_pix[::step][:take]
            else:
                sample = cluster_pix[:take]
            sampled_chunks[label_idx].append(sample.astype(np.uint8, copy=False))
            sampled_counts[label_idx] += int(len(sample))
    for label_idx, row in enumerate(rows):
        n = int(pixel_counts[label_idx])
        row['display_n_pixels'] = n
        row['display_sampled_pixels'] = int(sampled_counts[label_idx])
        if n <= 0:
            continue
        mean_rgb = pixel_sums[label_idx] / float(n)
        if sampled_chunks[label_idx]:
            sample_rgb = np.vstack(sampled_chunks[label_idx]).astype(np.float32)
            median_rgb = np.median(sample_rgb, axis=0)
        else:
            median_rgb = mean_rgb
        for name, values in (('display_mean', mean_rgb), ('display_median', median_rgb)):
            row[f'{name}_r'] = int(np.clip(round(float(values[0])), 0, 255))
            row[f'{name}_g'] = int(np.clip(round(float(values[1])), 0, 255))
            row[f'{name}_b'] = int(np.clip(round(float(values[2])), 0, 255))
        row['display_r'] = int(row['display_median_r'])
        row['display_g'] = int(row['display_median_g'])
        row['display_b'] = int(row['display_median_b'])
        row['display_source'] = source_label
        row['display_colour_space'] = colour_space_label


def _predict_cluster_labels_for_frame(frame_rgb: np.ndarray, valid_mask: np.ndarray, cluster_rows: list[dict[str, Any]]) -> np.ndarray:
    if not cluster_rows or not np.any(valid_mask):
        return np.full(valid_mask.shape, -1, dtype=np.int32)
    centers = np.asarray([[float(r['center_hx']), float(r['center_hy']), float(r['center_l'])] for r in cluster_rows], dtype=np.float32)
    return predict_hsl_cluster_labels(frame_rgb, valid_mask, centers)


def _recolour_frame(frame_rgb: np.ndarray, labels: np.ndarray, cluster_palette_rows: list[dict[str, Any]], preserve_mode: str = 'original_context', label_mode: bool = False) -> np.ndarray:
    out = frame_rgb.copy()
    if out.ndim == 2:
        out = np.repeat(out[..., None], 3, axis=2)
    pal = np.asarray([
        [int(r['label_r']), int(r['label_g']), int(r['label_b'])] if label_mode else [int(r['display_r']), int(r['display_g']), int(r['display_b'])]
        for r in cluster_palette_rows
    ], dtype=np.uint8)
    valid = labels >= 0
    if preserve_mode == 'black_background':
        out[...] = 0
    elif preserve_mode == 'dim_background':
        out = np.clip(out.astype(np.float32) * 0.35, 0, 255).astype(np.uint8)
    if np.any(valid):
        yy, xx = np.where(valid)
        out[yy, xx] = pal[labels[yy, xx]]
    return out


def _neutral_rgb(level: int = 245) -> np.ndarray:
    value = int(np.clip(level, 0, 255))
    return np.array([value, value, value], dtype=np.float32)


def _write_baseline_activity_rows(
    base_dir: Path,
    frames: list[FrameRecord],
    frame_cluster_count_rows: list[dict[str, Any]],
    cluster_ids: list[int],
    baseline_frames: int,
) -> tuple[list[dict[str, Any]], dict[tuple[int, int], float]]:
    if not frames or not cluster_ids:
        write_table(base_dir / 'frame_cluster_baseline_activity.csv', [])
        return [], {}
    counts_df = pd.DataFrame(frame_cluster_count_rows)
    if counts_df.empty:
        write_table(base_dir / 'frame_cluster_baseline_activity.csv', [])
        return [], {}
    frame_order = [int(frame.frame_id) for frame in frames]
    time_by_frame = {int(frame.frame_id): float(frame.time_s) for frame in frames}
    counts_df['pixel_fraction'] = counts_df['pixel_count'] / counts_df['n_valid_px'].clip(lower=1)
    baseline_frame_ids = frame_order[:max(1, min(int(baseline_frames), len(frame_order)))]
    baseline_df = counts_df[counts_df['frame_id'].isin(baseline_frame_ids)]
    baseline_by_cluster = {
        int(cid): float(baseline_df[baseline_df['cluster_id'] == int(cid)]['pixel_fraction'].mean())
        for cid in cluster_ids
    }
    rows: list[dict[str, Any]] = []
    max_active = 0.0
    for frame_id in frame_order:
        frame_df = counts_df[counts_df['frame_id'] == frame_id]
        for cluster_id in cluster_ids:
            sub = frame_df[frame_df['cluster_id'] == cluster_id]
            if sub.empty:
                pixel_count = 0
                n_valid = 0
                raw_fraction = 0.0
            else:
                first = sub.iloc[0]
                pixel_count = int(first['pixel_count'])
                n_valid = int(first['n_valid_px'])
                raw_fraction = float(first['pixel_fraction'])
            baseline = float(baseline_by_cluster.get(cluster_id, 0.0) or 0.0)
            active = float(max(0.0, raw_fraction - baseline))
            max_active = max(max_active, active)
            rows.append({
                'frame_id': int(frame_id),
                'time_s': float(time_by_frame.get(frame_id, frame_id)),
                'cluster_id': int(cluster_id),
                'pixel_count': int(pixel_count),
                'n_valid_px': int(n_valid),
                'raw_fraction': float(raw_fraction),
                'baseline_fraction': baseline,
                'active_fraction': active,
                'baseline_frame_count': int(len(baseline_frame_ids)),
            })
    alpha_by_frame_cluster: dict[tuple[int, int], float] = {}
    for row in rows:
        alpha = float(row['active_fraction'] / max_active) if max_active > 0 else 0.0
        row['normalized_active_fraction'] = alpha
        row['alpha'] = alpha
        alpha_by_frame_cluster[(int(row['frame_id']), int(row['cluster_id']))] = alpha
    write_table(base_dir / 'frame_cluster_baseline_activity.csv', rows)
    write_json(base_dir / 'cluster_baseline_activity_status.json', {
        'status': 'ok',
        'baseline_frame_ids': baseline_frame_ids,
        'baseline_frame_count': int(len(baseline_frame_ids)),
        'max_active_fraction': float(max_active),
        'alpha_normalisation': 'active_fraction_divided_by_run_max_active_fraction',
    })
    return rows, alpha_by_frame_cluster


def _recolour_baseline_activity_frame(
    labels: np.ndarray,
    palette_rows: list[dict[str, Any]],
    cluster_ids: list[int],
    alpha_by_cluster: dict[int, float],
    neutral_level: int = 245,
) -> np.ndarray:
    out = np.empty((*labels.shape, 3), dtype=np.float32)
    out[...] = _neutral_rgb(neutral_level)
    palette_by_cluster = {
        int(row['cluster_id']): np.array([float(row['display_r']), float(row['display_g']), float(row['display_b'])], dtype=np.float32)
        for row in palette_rows
    }
    for label_idx, cluster_id in enumerate(cluster_ids):
        alpha = float(np.clip(alpha_by_cluster.get(int(cluster_id), 0.0), 0.0, 1.0))
        if alpha <= 0:
            continue
        mask = labels == int(label_idx)
        if not np.any(mask):
            continue
        colour = palette_by_cluster.get(int(cluster_id))
        if colour is None:
            continue
        out[mask] = (1.0 - alpha) * out[mask] + alpha * colour
    return np.clip(np.round(out), 0, 255).astype(np.uint8)


def _open_video_writer(path: Path, shape_hw: tuple[int, int], fps: float) -> cv2.VideoWriter:
    h, w = shape_hw
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    writer = cv2.VideoWriter(str(path), fourcc, max(float(fps), 1.0), (int(w), int(h)))
    return writer



def _write_cluster_prevalence_plots(
    base_dir: Path,
    palette_rows: list[dict[str, Any]],
    frame_cluster_count_rows: list[dict[str, Any]],
) -> None:
    if not frame_cluster_count_rows or not palette_rows:
        return
    counts_df = pd.DataFrame(frame_cluster_count_rows)
    if counts_df.empty:
        return
    counts_df = counts_df.sort_values(['frame_id', 'cluster_id']).reset_index(drop=True)
    counts_df['pixel_fraction'] = counts_df['pixel_count'] / counts_df['n_valid_px'].clip(lower=1)
    write_table(base_dir / 'frame_cluster_pixel_counts.csv', counts_df[['frame_id', 'cluster_id', 'pixel_count', 'n_valid_px']].to_dict(orient='records'))
    write_table(base_dir / 'frame_cluster_pixel_fractions.csv', counts_df[['frame_id', 'cluster_id', 'pixel_fraction', 'n_valid_px']].to_dict(orient='records'))

    import matplotlib
    matplotlib.use('Agg', force=True)
    import matplotlib.pyplot as plt

    palette = {int(r['cluster_id']): (float(r['display_r'])/255.0, float(r['display_g'])/255.0, float(r['display_b'])/255.0) for r in palette_rows}
    cluster_ids = sorted(int(v) for v in counts_df['cluster_id'].unique())
    frame_ids = sorted(int(v) for v in counts_df['frame_id'].unique())

    # counts plot
    fig, ax = plt.subplots(figsize=(8, 4.8))
    for cid in cluster_ids:
        sub = counts_df[counts_df['cluster_id'] == cid].sort_values('frame_id')
        ax.plot(sub['frame_id'].to_numpy(), sub['pixel_count'].to_numpy(), label=f'cluster {cid}', color=palette.get(cid, None))
    ax.set_xlabel('frame')
    ax.set_ylabel('pixels')
    ax.set_title('Cluster pixel counts by frame')
    ax.legend(loc='best', fontsize=8)
    fig.tight_layout()
    fig.savefig(base_dir / 'frame_cluster_pixel_counts.png', dpi=160, bbox_inches='tight')
    plt.close(fig)

    # fractions plot
    fig, ax = plt.subplots(figsize=(8, 4.8))
    for cid in cluster_ids:
        sub = counts_df[counts_df['cluster_id'] == cid].sort_values('frame_id')
        ax.plot(sub['frame_id'].to_numpy(), sub['pixel_fraction'].to_numpy(), label=f'cluster {cid}', color=palette.get(cid, None))
    ax.set_xlabel('frame')
    ax.set_ylabel('fraction of wafer-nonhole pixels')
    ax.set_title('Cluster pixel fractions by frame')
    ax.legend(loc='best', fontsize=8)
    fig.tight_layout()
    fig.savefig(base_dir / 'frame_cluster_pixel_fractions.png', dpi=160, bbox_inches='tight')
    plt.close(fig)


def write_wafer_nonhole_cluster_videos(
    base_dir: Path,
    frames: list[FrameRecord],
    holes_by_frame: dict[int, list[HoleGeometry]],
    support_mask: np.ndarray,
    bundle: WaferNonholeColourBundle,
    display_frames: list[FrameRecord] | None = None,
    write_recolour_video: bool = True,
    write_side_by_side_video: bool = True,
    write_labelmap_video: bool = True,
    write_baseline_activity_video: bool = True,
    baseline_frames: int = 3,
    preserve_mode: str = 'original_context',
    progress_callback: Callable[[int, int, str], None] | None = None,
) -> dict[str, Any]:
    status = {'status': 'skipped', 'message': None, 'fps': _estimate_fps(frames), 'n_frames': len(frames)}
    if bundle.status != 'ok' or not bundle.cluster_rows:
        status['message'] = 'bundle_not_ready'
        write_json(base_dir / 'cluster_video_status.json', status)
        return status
    if not all(_row_rgb_triplet(row, 'analysis_median') is not None for row in bundle.cluster_rows):
        _update_cluster_rows_with_observed_rgb(frames, holes_by_frame, support_mask, bundle)
    if display_frames is not None:
        _update_cluster_rows_with_display_rgb(
            frames,
            display_frames,
            holes_by_frame,
            support_mask,
            bundle,
            source_label='uncorrected_registered_observed_pixel_median_rgb',
            colour_space_label='raw_registered_rgb',
        )
    palette_rows = _cluster_palette_rows(bundle.cluster_rows)
    if palette_rows:
        write_table(base_dir / 'cluster_display_palette.csv', palette_rows)
        write_json(base_dir / 'cluster_display_palette.json', palette_rows)
        write_table(base_dir / 'cluster_model_summary.csv', bundle.cluster_rows)
    fps = float(status['fps'])
    h, w = frames[0].image.shape[:2]
    recol_writer = _open_video_writer(base_dir / 'video_cluster_recoloured.avi', (h, w), fps) if write_recolour_video else None
    label_writer = _open_video_writer(base_dir / 'video_cluster_labels.avi', (h, w), fps) if write_labelmap_video else None
    activity_writer = _open_video_writer(base_dir / 'video_cluster_baseline_activity.avi', (h, w), fps) if write_baseline_activity_video else None
    side_writer = _open_video_writer(base_dir / 'video_cluster_side_by_side.avi', (h, 2 * w), fps) if write_side_by_side_video else None
    if any(wr is not None and not wr.isOpened() for wr in [recol_writer, label_writer, activity_writer, side_writer]):
        for wr in [recol_writer, label_writer, activity_writer, side_writer]:
            if wr is not None:
                wr.release()
        status.update({'status': 'error', 'message': 'video_writer_open_failed'})
        write_json(base_dir / 'cluster_video_status.json', status)
        return status
    progress_total = max(1, len(frames) * (1 + int(activity_writer is not None)) + 2)
    progress_step = 0

    def report(message: str, increment: int = 1) -> None:
        nonlocal progress_step
        progress_step = min(progress_total, progress_step + max(0, int(increment)))
        if progress_callback is not None:
            progress_callback(progress_step, progress_total, message)

    frame_cluster_label_rows = []
    frame_cluster_count_rows = []
    n_clusters = int(len(bundle.cluster_rows))
    cluster_ids = [int(row['cluster_id']) for row in bundle.cluster_rows]
    for frame_i, frame in enumerate(frames, start=1):
        holes = holes_by_frame[frame.frame_id]
        hole_union = make_global_hole_union(frame.image.shape[:2], holes)
        valid_mask = support_mask & (~hole_union)
        labels = _predict_cluster_labels_for_frame(frame.image, valid_mask, bundle.cluster_rows)
        n_valid_px = int(np.sum(valid_mask))
        n_labelled_px = int(np.sum(labels >= 0))
        frame_cluster_label_rows.append({'frame_id': int(frame.frame_id), 'n_valid_px': n_valid_px, 'n_labelled_px': n_labelled_px})
        if n_clusters > 0:
            hard_counts = np.bincount(labels[labels >= 0], minlength=n_clusters).astype(np.int64) if np.any(labels >= 0) else np.zeros((n_clusters,), dtype=np.int64)
            for label_idx, cluster_id in enumerate(cluster_ids):
                frame_cluster_count_rows.append({
                    'frame_id': int(frame.frame_id),
                    'cluster_id': int(cluster_id),
                    'pixel_count': int(hard_counts[label_idx]),
                    'n_valid_px': n_valid_px,
                })
        recol = _recolour_frame(frame.image, labels, palette_rows, preserve_mode=preserve_mode, label_mode=False)
        label = _recolour_frame(frame.image, labels, palette_rows, preserve_mode='black_background', label_mode=True)
        if recol_writer is not None:
            recol_writer.write(cv2.cvtColor(recol, cv2.COLOR_RGB2BGR))
        if label_writer is not None:
            label_writer.write(cv2.cvtColor(label, cv2.COLOR_RGB2BGR))
        if side_writer is not None:
            side = np.concatenate([frame.image, recol], axis=1)
            side_writer.write(cv2.cvtColor(side, cv2.COLOR_RGB2BGR))
        report(f"Rendered cluster colour frame {frame_i}/{len(frames)}")
    activity_rows, alpha_by_frame_cluster = _write_baseline_activity_rows(
        base_dir,
        frames,
        frame_cluster_count_rows,
        cluster_ids,
        baseline_frames=baseline_frames,
    )
    report("Built baseline activity table")
    if activity_writer is not None:
        for frame_i, frame in enumerate(frames, start=1):
            holes = holes_by_frame[frame.frame_id]
            hole_union = make_global_hole_union(frame.image.shape[:2], holes)
            valid_mask = support_mask & (~hole_union)
            labels = _predict_cluster_labels_for_frame(frame.image, valid_mask, bundle.cluster_rows)
            alpha_by_cluster = {
                int(cluster_id): float(alpha_by_frame_cluster.get((int(frame.frame_id), int(cluster_id)), 0.0))
                for cluster_id in cluster_ids
            }
            active = _recolour_baseline_activity_frame(labels, palette_rows, cluster_ids, alpha_by_cluster)
            activity_writer.write(cv2.cvtColor(active, cv2.COLOR_RGB2BGR))
            report(f"Rendered baseline activity frame {frame_i}/{len(frames)}")
    for wr in [recol_writer, label_writer, activity_writer, side_writer]:
        if wr is not None:
            wr.release()
    write_table(base_dir / 'frame_cluster_label_counts.csv', frame_cluster_label_rows)
    _write_cluster_prevalence_plots(base_dir, palette_rows, frame_cluster_count_rows)
    report("Wrote cluster video summaries")
    status.update({
        'status': 'ok',
        'message': 'ok',
        'wrote_recolour_video': bool(write_recolour_video),
        'wrote_side_by_side_video': bool(write_side_by_side_video),
        'wrote_labelmap_video': bool(write_labelmap_video),
        'wrote_baseline_activity_video': bool(write_baseline_activity_video),
        'n_baseline_activity_rows': int(len(activity_rows)),
        'baseline_activity_video_palette': 'uncorrected registered observed RGB' if display_frames is not None else 'analysis observed RGB',
        'wrote_cluster_prevalence_plots': True,
    })
    write_json(base_dir / 'cluster_video_status.json', status)
    return status
