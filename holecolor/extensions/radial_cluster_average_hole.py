from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Callable

import matplotlib
matplotlib.use('Agg', force=True)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from holecolor.config.schema import ParallelConfig
from holecolor.core.parallel import parallel_map, prefer_thread_for_image_tasks
from holecolor.core.types import FrameRecord, HoleGeometry
from holecolor.descriptors.cluster_labels import predict_hsl_cluster_labels
from holecolor.masks.matrix import make_global_hole_union
from holecolor.masks.terraces import make_nonoverlapping_hole_terraces
from holecolor.qc.reports import write_json, write_table, write_table_columns

try:  # pragma: no cover - exercised when numba is installed
    from numba import njit, prange
except Exception:  # pragma: no cover - keeps the package importable without numba
    def njit(*args, **kwargs):
        def deco(fn):
            return fn
        return deco

    def prange(*args):
        return range(*args)


_TENSOR_COLUMNS = (
    'frame_id',
    'time_s',
    'hole_id',
    'lattice_u',
    'lattice_v',
    'terrace_index',
    'sector_id',
    'theta_center_deg',
    'cluster_id',
    'pixel_count',
    'pixel_fraction',
    'n_valid_pixels',
)


def _hsl_to_rgb_uint8(h: np.ndarray, s: np.ndarray, l: np.ndarray) -> np.ndarray:
    h = np.mod(np.asarray(h, dtype=np.float32), 1.0)
    s = np.clip(np.asarray(s, dtype=np.float32), 0.0, 1.0)
    l = np.clip(np.asarray(l, dtype=np.float32), 0.0, 1.0)
    c = (1.0 - np.abs(2.0 * l - 1.0)) * s
    hp = h * 6.0
    x = c * (1.0 - np.abs(np.mod(hp, 2.0) - 1.0))
    z = np.zeros_like(c)
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
        return tuple(int(np.clip(round(float(row[k])), 0, 255)) for k in keys)  # type: ignore[return-value]
    except (TypeError, ValueError):
        return None


def _cluster_palette_rows(cluster_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not cluster_rows:
        return []
    h = np.asarray([float(r['center_h']) for r in cluster_rows], dtype=np.float32)
    s = np.asarray([float(r['center_s']) for r in cluster_rows], dtype=np.float32)
    l = np.asarray([float(r['center_l']) for r in cluster_rows], dtype=np.float32)
    centroid_rgb = _hsl_to_rgb_uint8(h, s, l)
    rows = []
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
        }
        for prefix in ('analysis_median', 'analysis_mean'):
            rgb = _row_rgb_triplet(row, prefix)
            if rgb is not None:
                record[f'{prefix}_r'] = int(rgb[0])
                record[f'{prefix}_g'] = int(rgb[1])
                record[f'{prefix}_b'] = int(rgb[2])
        for key in ('analysis_n_pixels', 'analysis_sampled_pixels', 'display_colour_space'):
            if key in row:
                record[key] = row[key]
        rows.append({
            **record,
        })
    return rows


def _predict_cluster_labels_for_frame(frame_rgb: np.ndarray, valid_mask: np.ndarray, cluster_rows: list[dict[str, Any]]) -> np.ndarray:
    if not cluster_rows or not np.any(valid_mask):
        return np.full(valid_mask.shape, -1, dtype=np.int32)
    centers = np.asarray([[float(r['center_hx']), float(r['center_hy']), float(r['center_l'])] for r in cluster_rows], dtype=np.float32)
    return predict_hsl_cluster_labels(frame_rgb, valid_mask, centers)


@njit(cache=True, parallel=True, nogil=True)
def _predict_hsl_cluster_labels_at_coords_kernel(
    image: np.ndarray,
    yy: np.ndarray,
    xx: np.ndarray,
    centers: np.ndarray,
    cluster_ids: np.ndarray,
) -> np.ndarray:
    labels = np.full(yy.size, -1, dtype=np.int32)
    n_centers = centers.shape[0]
    two_pi = 2.0 * math.pi
    for i in prange(yy.size):
        y = int(yy[i])
        x = int(xx[i])
        if y < 0 or x < 0 or y >= image.shape[0] or x >= image.shape[1]:
            continue
        r = float(image[y, x, 0]) / 255.0
        g = float(image[y, x, 1]) / 255.0
        b = float(image[y, x, 2]) / 255.0
        mx = r
        if g > mx:
            mx = g
        if b > mx:
            mx = b
        mn = r
        if g < mn:
            mn = g
        if b < mn:
            mn = b
        c = mx - mn
        lightness = 0.5 * (mx + mn)
        saturation = 0.0
        hue = 0.0
        if c > 0.0:
            denom = 1.0 - abs(2.0 * lightness - 1.0) + 1e-6
            saturation = c / denom
            if mx == r:
                hue = (g - b) / c
                while hue < 0.0:
                    hue += 6.0
                while hue >= 6.0:
                    hue -= 6.0
            elif mx == g:
                hue = (b - r) / c + 2.0
            else:
                hue = (r - g) / c + 4.0
            hue /= 6.0
        hx = saturation * math.cos(two_pi * hue)
        hy = saturation * math.sin(two_pi * hue)
        best_idx = -1
        best_d2 = 1.0e30
        for ci in range(n_centers):
            dx = hx - float(centers[ci, 0])
            dy = hy - float(centers[ci, 1])
            dl = lightness - float(centers[ci, 2])
            d2 = dx * dx + dy * dy + dl * dl
            if d2 < best_d2:
                best_d2 = d2
                best_idx = ci
        if best_idx >= 0:
            labels[i] = int(cluster_ids[best_idx])
    return labels


def _predict_cluster_labels_for_coords(frame_rgb: np.ndarray, yy: np.ndarray, xx: np.ndarray, cluster_rows: list[dict[str, Any]]) -> np.ndarray:
    if not cluster_rows or yy.size == 0:
        return np.full(yy.size, -1, dtype=np.int32)
    ordered = sorted(cluster_rows, key=lambda r: int(r['cluster_id']))
    centers = np.asarray([[float(r['center_hx']), float(r['center_hy']), float(r['center_l'])] for r in ordered], dtype=np.float32)
    cluster_ids = np.asarray([int(r['cluster_id']) for r in ordered], dtype=np.int32)
    image = np.ascontiguousarray(frame_rgb[..., :3], dtype=np.uint8)
    return _predict_hsl_cluster_labels_at_coords_kernel(
        image,
        np.ascontiguousarray(yy.astype(np.int64, copy=False)),
        np.ascontiguousarray(xx.astype(np.int64, copy=False)),
        np.ascontiguousarray(centers),
        np.ascontiguousarray(cluster_ids),
    )


@njit(cache=True, nogil=True)
def _sector_cluster_counts_kernel(
    yy: np.ndarray,
    xx: np.ndarray,
    lab: np.ndarray,
    hole_y: float,
    hole_x: float,
    angle_ref: float,
    n_sectors: int,
    n_label_slots: int,
) -> tuple[np.ndarray, np.ndarray]:
    counts = np.zeros((n_sectors, n_label_slots), dtype=np.int64)
    totals = np.zeros(n_sectors, dtype=np.int64)
    two_pi = 2.0 * math.pi
    for i in range(lab.size):
        label = int(lab[i])
        if label < 0 or label >= n_label_slots:
            continue
        theta = math.atan2(float(yy[i]) - hole_y, float(xx[i]) - hole_x) - angle_ref
        theta = theta - math.floor(theta / two_pi) * two_pi
        sector_id = int(math.floor(theta / two_pi * float(n_sectors)))
        if sector_id < 0:
            sector_id = 0
        elif sector_id >= n_sectors:
            sector_id = n_sectors - 1
        counts[sector_id, label] += 1
        totals[sector_id] += 1
    return counts, totals


def _empty_tensor_columns() -> dict[str, list[Any]]:
    return {key: [] for key in _TENSOR_COLUMNS}


def _extend_tensor_columns(dst: dict[str, list[Any]], src: dict[str, list[Any]]) -> None:
    for key in _TENSOR_COLUMNS:
        dst[key].extend(src.get(key, []))


def _append_tensor_row(
    cols: dict[str, list[Any]],
    frame_id: int,
    time_s: float,
    hole_id: int,
    uv: tuple[Any, Any],
    terrace_index: int,
    sector_id: int,
    cluster_id: int,
    pixel_count: int,
    n_valid: int,
    n_angle_sectors: int,
) -> None:
    cols['frame_id'].append(int(frame_id))
    cols['time_s'].append(float(time_s))
    cols['hole_id'].append(int(hole_id))
    cols['lattice_u'].append(uv[0])
    cols['lattice_v'].append(uv[1])
    cols['terrace_index'].append(int(terrace_index))
    cols['sector_id'].append(int(sector_id))
    cols['theta_center_deg'].append(float((int(sector_id) + 0.5) * 360.0 / int(n_angle_sectors)))
    cols['cluster_id'].append(int(cluster_id))
    cols['pixel_count'].append(int(pixel_count))
    cols['pixel_fraction'].append(float(pixel_count / max(int(n_valid), 1)))
    cols['n_valid_pixels'].append(int(n_valid))


def _radial_cluster_frame_tensor_task(task: dict[str, Any]) -> dict[str, list[Any]]:
    frame: FrameRecord = task['frame']
    frame_holes: list[HoleGeometry] = task['frame_holes']
    support_mask: np.ndarray = task['support_mask']
    cluster_rows: list[dict[str, Any]] = task['cluster_rows']
    cluster_ids: list[int] = task['cluster_ids']
    lattice_indices: dict[int, tuple[int, int]] | None = task['lattice_indices']
    angle_ref = float(task['angle_ref'])
    n_terraces = int(task['n_terraces'])
    n_angle_sectors = int(task['n_angle_sectors'])
    terrace_width_mode = str(task['terrace_width_mode'])
    terrace_gap_basis = str(task['terrace_gap_basis'])
    terrace_min_width_px = float(task['terrace_min_width_px'])

    cols = _empty_tensor_columns()
    if not frame_holes:
        return cols
    shape = frame.image.shape[:2]
    hole_union = make_global_hole_union(shape, frame_holes)
    terraces_by_hole = make_nonoverlapping_hole_terraces(
        shape,
        frame_holes,
        int(n_terraces),
        lattice_indices=lattice_indices,
        width_mode=terrace_width_mode,
        gap_basis=terrace_gap_basis,
        min_width_px=terrace_min_width_px,
    )
    valid_support_mask = np.asarray(support_mask, dtype=bool) & (~hole_union)
    terrace_records: list[tuple[HoleGeometry, int, np.ndarray, np.ndarray]] = []
    yy_parts: list[np.ndarray] = []
    xx_parts: list[np.ndarray] = []
    for hole in frame_holes:
        hole_id = int(hole.hole_id)
        for terrace_index, terrace in enumerate(terraces_by_hole.get(hole_id, [])):
            yy, xx = terrace.global_coords()
            if yy.size == 0:
                continue
            valid = valid_support_mask[yy, xx]
            if not np.any(valid):
                continue
            yy_valid = yy[valid].astype(np.int64, copy=False)
            xx_valid = xx[valid].astype(np.int64, copy=False)
            terrace_records.append((hole, int(terrace_index), yy_valid, xx_valid))
            yy_parts.append(yy_valid)
            xx_parts.append(xx_valid)
    if not terrace_records:
        return cols
    yy_all = np.concatenate(yy_parts).astype(np.int64, copy=False)
    xx_all = np.concatenate(xx_parts).astype(np.int64, copy=False)
    labels_all = _predict_cluster_labels_for_coords(frame.image, yy_all, xx_all, cluster_rows)
    n_label_slots = max(max(cluster_ids) + 1 if cluster_ids else 0, 1)
    offset = 0
    for hole, terrace_index, yy_valid, xx_valid in terrace_records:
        hole_id = int(hole.hole_id)
        uv = lattice_indices.get(hole_id, (None, None)) if lattice_indices is not None else (None, None)
        n = int(yy_valid.size)
        lab_valid = labels_all[offset:offset + n].astype(np.int64, copy=False)
        offset += n
        valid = lab_valid >= 0
        if not np.any(valid):
            continue
        counts, sector_totals = _sector_cluster_counts_kernel(
            yy_valid[valid].astype(np.float64, copy=False),
            xx_valid[valid].astype(np.float64, copy=False),
            lab_valid[valid],
            float(hole.y),
            float(hole.x),
            angle_ref,
            int(n_angle_sectors),
            int(n_label_slots),
        )
        for sector_id in range(int(n_angle_sectors)):
            n_valid = int(sector_totals[sector_id])
            if n_valid <= 0:
                continue
            for cluster_id in cluster_ids:
                cid = int(cluster_id)
                pixel_count = int(counts[sector_id, cid]) if cid < counts.shape[1] else 0
                _append_tensor_row(
                    cols,
                    int(frame.frame_id),
                    float(frame.time_s),
                    hole_id,
                    uv,
                    int(terrace_index),
                    int(sector_id),
                    cid,
                    pixel_count,
                    n_valid,
                    int(n_angle_sectors),
                )
    return cols


def _normalized_js_similarity(p: np.ndarray, q: np.ndarray) -> float:
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    p = np.clip(p, 0.0, None)
    q = np.clip(q, 0.0, None)
    ps = p.sum()
    qs = q.sum()
    if ps <= 0 or qs <= 0:
        return float('nan')
    p = p / ps
    q = q / qs
    m = 0.5 * (p + q)
    def _kl(a: np.ndarray, b: np.ndarray) -> float:
        mask = (a > 0) & (b > 0)
        if not np.any(mask):
            return 0.0
        return float(np.sum(a[mask] * np.log2(a[mask] / b[mask])))
    jsd = 0.5 * _kl(p, m) + 0.5 * _kl(q, m)
    jsd = float(np.clip(jsd, 0.0, 1.0))
    return float(1.0 - jsd)


def _dominant_rgb_image(cluster_ids: np.ndarray, dominance: np.ndarray, palette_map: dict[int, tuple[float, float, float]]) -> np.ndarray:
    h, w = cluster_ids.shape
    img = np.ones((h, w, 3), dtype=np.float32)
    dom = np.nan_to_num(dominance, nan=0.0, posinf=0.0, neginf=0.0)
    dom = np.clip(dom, 0.0, 1.0)
    for cid, color in palette_map.items():
        mask = cluster_ids == cid
        if not np.any(mask):
            continue
        col = np.asarray(color, dtype=np.float32)
        alpha = dom[mask][:, None]
        img[mask] = (1.0 - alpha) * 1.0 + alpha * col
    return np.clip(img, 0.0, 1.0)


def _save_dominant_cluster_map(path: Path, cluster_ids: np.ndarray, dominance: np.ndarray, palette_map: dict[int, tuple[float, float, float]], title: str, xlabel: str, ylabel: str, yticklabels: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rgb = _dominant_rgb_image(cluster_ids, dominance, palette_map)
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    ax.imshow(rgb, aspect='auto', interpolation='nearest', origin='lower')
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if yticklabels is not None:
        ax.set_yticks(range(len(yticklabels)))
        ax.set_yticklabels(yticklabels)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches='tight')
    plt.close(fig)


def _front_metric_rows(per_hole_df: pd.DataFrame, cluster_ids: list[int], n_terraces: int, threshold_fraction: float) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if per_hole_df.empty:
        return rows
    for (frame_id, cluster_id), grp in per_hole_df.groupby(['frame_id', 'cluster_id'], sort=True):
        grp = grp.sort_values('terrace_index')
        terrace_idx = grp['terrace_index'].to_numpy(dtype=float)
        vals = grp['pixel_fraction'].to_numpy(dtype=float)
        if vals.size == 0 or not np.any(np.isfinite(vals)):
            continue
        mean_w = float(np.nansum((terrace_idx + 1.0) * vals) / max(np.nansum(vals), 1e-12))
        peak_t = int(grp.iloc[int(np.nanargmax(vals))]['terrace_index']) + 1 if np.any(np.isfinite(vals)) else None
        valid_outer = grp.loc[grp['pixel_fraction'] >= float(threshold_fraction), 'terrace_index'].to_numpy(dtype=int)
        outer_t = int(valid_outer.max()) + 1 if valid_outer.size else None
        by_hole_metric = []
        for _, hgrp in grp.groupby('hole_id'):
            hvals = hgrp.sort_values('terrace_index')['pixel_fraction'].to_numpy(dtype=float)
            hterr = hgrp.sort_values('terrace_index')['terrace_index'].to_numpy(dtype=float)
            if hvals.size == 0 or np.nansum(hvals) <= 0:
                continue
            by_hole_metric.append(float(np.nansum((hterr + 1.0) * hvals) / max(np.nansum(hvals), 1e-12)))
        arr = np.asarray(by_hole_metric, dtype=float)
        rows.append({
            'frame_id': int(frame_id),
            'cluster_id': int(cluster_id),
            'weighted_mean_terrace': mean_w,
            'peak_terrace': peak_t,
            'outer_front_terrace': outer_t,
            'hole_mean_weighted_mean_terrace': float(np.nanmean(arr)) if arr.size else None,
            'hole_sd_weighted_mean_terrace': float(np.nanstd(arr)) if arr.size else None,
            'hole_q10_weighted_mean_terrace': float(np.nanquantile(arr, 0.10)) if arr.size else None,
            'hole_q90_weighted_mean_terrace': float(np.nanquantile(arr, 0.90)) if arr.size else None,
            'n_holes': int(arr.size),
            'n_terraces': int(n_terraces),
        })
    return rows


def _radial_cluster_input_guard(
    frames: list[FrameRecord],
    holes_by_frame: dict[int, list[HoleGeometry]],
    support_mask: np.ndarray,
) -> dict[str, Any]:
    if not frames:
        return {'ok': False, 'reason': 'no_frames'}
    counts: list[int] = []
    radii: list[float] = []
    for frame in frames:
        holes = holes_by_frame.get(int(frame.frame_id), [])
        counts.append(int(len(holes)))
        for hole in holes:
            radius = float(getattr(hole, 'radius_outer_px', 0.0))
            if np.isfinite(radius) and radius > 0.0:
                radii.append(radius)
    max_holes = max(counts) if counts else 0
    median_radius = float(np.median(np.asarray(radii, dtype=np.float32))) if radii else 0.0
    support_area = int(np.count_nonzero(np.asarray(support_mask, dtype=bool)))
    physical_max_holes = 0
    if median_radius > 0.0 and support_area > 0:
        physical_max_holes = max(1, int(math.floor(float(support_area) / (math.pi * median_radius * median_radius))))
    if max_holes <= 0:
        return {'ok': False, 'reason': 'no_holes', 'max_frame_holes': int(max_holes)}
    if physical_max_holes > 0 and max_holes > physical_max_holes:
        return {
            'ok': False,
            'reason': 'hole_count_exceeds_physical_nonoverlap_limit',
            'max_frame_holes': int(max_holes),
            'physical_max_holes': int(physical_max_holes),
            'median_hole_radius_px': float(median_radius),
            'support_area_px': int(support_area),
        }
    return {
        'ok': True,
        'max_frame_holes': int(max_holes),
        'physical_max_holes': int(physical_max_holes),
        'median_hole_radius_px': float(median_radius),
        'support_area_px': int(support_area),
    }


_RADIAL_CLUSTER_NUMBA_WARMED = False


def warmup_radial_cluster_numba() -> None:
    global _RADIAL_CLUSTER_NUMBA_WARMED
    if _RADIAL_CLUSTER_NUMBA_WARMED:
        return
    image = np.zeros((4, 4, 3), dtype=np.uint8)
    yy = np.asarray([0, 1], dtype=np.int64)
    xx = np.asarray([0, 1], dtype=np.int64)
    centers = np.asarray([[0.0, 0.0, 0.0]], dtype=np.float32)
    cluster_ids = np.asarray([0], dtype=np.int32)
    _predict_hsl_cluster_labels_at_coords_kernel(image, yy, xx, centers, cluster_ids)
    _sector_cluster_counts_kernel(
        yy.astype(np.float64),
        xx.astype(np.float64),
        np.asarray([0, 0], dtype=np.int64),
        0.0,
        0.0,
        0.0,
        1,
        1,
    )
    _RADIAL_CLUSTER_NUMBA_WARMED = True


def write_radial_cluster_average_hole_artifacts(
    base_dir: Path,
    frames: list[FrameRecord],
    holes_by_frame: dict[int, list[HoleGeometry]],
    support_mask: np.ndarray,
    cluster_rows: list[dict[str, Any]],
    lattice_indices: dict[int, tuple[int, int]] | None,
    lattice_angle_deg: float,
    n_terraces: int,
    n_angle_sectors: int,
    terrace_width_mode: str = 'half_gap',
    terrace_gap_basis: str = 'border_gap',
    terrace_min_width_px: float = 0.0,
    front_threshold_fraction: float = 0.15,
    parallel_cfg: ParallelConfig | dict[str, Any] | None = None,
    progress_callback: Callable[[int, int, str], None] | None = None,
) -> dict[str, Any]:
    base_dir.mkdir(parents=True, exist_ok=True)
    status: dict[str, Any] = {'status': 'skipped', 'message': None, 'n_tensor_rows': 0}
    if not frames or not cluster_rows:
        status['message'] = 'cluster_rows_unavailable'
        write_json(base_dir / 'radial_cluster_status.json', status)
        return status
    guard = _radial_cluster_input_guard(frames, holes_by_frame, support_mask)
    if progress_callback is not None:
        progress_callback(1, max(5, int(len(frames)) + 4), "Validated radial cluster physical inputs")
    if not bool(guard.get('ok')):
        status.update({'status': 'skipped', 'message': str(guard.get('reason', 'invalid_radial_cluster_inputs')), **guard})
        write_json(base_dir / 'radial_cluster_status.json', status)
        return status
    warmup_radial_cluster_numba()
    cluster_ids = sorted(int(r['cluster_id']) for r in cluster_rows)
    palette_rows = _cluster_palette_rows(cluster_rows)
    palette_map = {int(r['cluster_id']): (float(r['display_r']) / 255.0, float(r['display_g']) / 255.0, float(r['display_b']) / 255.0) for r in palette_rows}
    write_table(base_dir / 'radial_cluster_palette.csv', palette_rows)
    write_json(base_dir / 'radial_cluster_palette.json', palette_rows)

    angle_ref = np.deg2rad(float(lattice_angle_deg))
    progress_total = int(len(frames)) + 4
    frame_tasks = [
        {
            'frame': frame,
            'frame_holes': holes_by_frame.get(int(frame.frame_id), []),
            'support_mask': support_mask,
            'cluster_rows': cluster_rows,
            'cluster_ids': cluster_ids,
            'lattice_indices': lattice_indices,
            'angle_ref': float(angle_ref),
            'n_terraces': int(n_terraces),
            'n_angle_sectors': int(n_angle_sectors),
            'terrace_width_mode': str(terrace_width_mode),
            'terrace_gap_basis': str(terrace_gap_basis),
            'terrace_min_width_px': float(terrace_min_width_px),
        }
        for frame in frames
    ]
    task_cfg = prefer_thread_for_image_tasks(parallel_cfg)

    def _frame_progress(current: int, total: int) -> None:
        if progress_callback is not None:
            progress_callback(int(current), progress_total, f"Building radial cluster tensor frame {current}/{total}")

    frame_columns = parallel_map(
        _radial_cluster_frame_tensor_task,
        frame_tasks,
        task_cfg,
        desc='Radial cluster tensor',
        progress_callback=_frame_progress,
    )
    tensor_columns = _empty_tensor_columns()
    for cols in frame_columns:
        _extend_tensor_columns(tensor_columns, cols)
    n_tensor_rows = int(len(tensor_columns['frame_id']))
    if progress_callback is not None:
        progress_callback(int(len(frames)) + 1, progress_total, "Writing radial cluster tensor table")
    write_table_columns(base_dir / 'hole_terrace_sector_cluster_tensor.csv', tensor_columns)
    status['n_tensor_rows'] = n_tensor_rows
    if n_tensor_rows <= 0:
        status.update({'status': 'skipped', 'message': 'no_valid_tensor_rows'})
        write_json(base_dir / 'radial_cluster_status.json', status)
        return status

    if progress_callback is not None:
        progress_callback(int(len(frames)) + 2, progress_total, "Aggregating average-hole radial cluster tables")
    tensor_df = pd.DataFrame(tensor_columns)
    # average-hole terrace cluster fractions
    pooled_counts = tensor_df.groupby(['frame_id', 'terrace_index', 'cluster_id'], as_index=False)['pixel_count'].sum()
    pooled_totals = pooled_counts.groupby(['frame_id', 'terrace_index'], as_index=False)['pixel_count'].sum().rename(columns={'pixel_count': 'terrace_total_pixels'})
    pooled = pooled_counts.merge(pooled_totals, on=['frame_id', 'terrace_index'], how='left')
    pooled['pixel_fraction'] = pooled['pixel_count'] / pooled['terrace_total_pixels'].clip(lower=1)
    write_table(base_dir / 'average_hole_terrace_cluster_fractions.csv', pooled.to_dict(orient='records'))

    summary_rows: list[dict[str, Any]] = []
    frames_sorted = sorted(int(v) for v in pooled['frame_id'].unique())
    terraces_sorted = sorted(int(v) for v in pooled['terrace_index'].unique())
    avg_dom = np.full((len(terraces_sorted), len(frames_sorted)), -1, dtype=int)
    avg_dom_strength = np.zeros((len(terraces_sorted), len(frames_sorted)), dtype=float)
    for fi, frame_id in enumerate(frames_sorted):
        for ti, terrace_index in enumerate(terraces_sorted):
            sub = pooled[(pooled['frame_id'] == frame_id) & (pooled['terrace_index'] == terrace_index)].sort_values('cluster_id')
            if sub.empty:
                continue
            p = sub['pixel_fraction'].to_numpy(dtype=float)
            cids = sub['cluster_id'].to_numpy(dtype=int)
            idx = int(np.argmax(p))
            dominant_cluster = int(cids[idx])
            dominant_fraction = float(p[idx])
            entropy = float(-np.sum(np.where(p > 0, p * np.log2(np.clip(p, 1e-12, None)), 0.0)))
            summary_rows.append({
                'frame_id': int(frame_id),
                'terrace_index': int(terrace_index),
                'dominant_cluster_id': dominant_cluster,
                'dominant_cluster_fraction': dominant_fraction,
                'cluster_entropy': entropy,
                'n_valid_pixels': int(sub['terrace_total_pixels'].iloc[0]),
            })
            avg_dom[ti, fi] = dominant_cluster
            avg_dom_strength[ti, fi] = dominant_fraction
    write_table(base_dir / 'average_hole_terrace_summary.csv', summary_rows)
    _save_dominant_cluster_map(
        base_dir / 'average_hole_terrace_chronogram.png',
        avg_dom,
        avg_dom_strength,
        palette_map,
        title='Average-hole terrace chronogram',
        xlabel='frame',
        ylabel='terrace',
        yticklabels=[str(t + 1) for t in terraces_sorted],
    )

    # terrace-specific angular chronograms
    terrace_angle = tensor_df.groupby(['frame_id', 'terrace_index', 'sector_id', 'cluster_id'], as_index=False)['pixel_count'].sum()
    ta_tot = terrace_angle.groupby(['frame_id', 'terrace_index', 'sector_id'], as_index=False)['pixel_count'].sum().rename(columns={'pixel_count': 'sector_total_pixels'})
    terrace_angle = terrace_angle.merge(ta_tot, on=['frame_id', 'terrace_index', 'sector_id'], how='left')
    terrace_angle['pooled_fraction'] = terrace_angle['pixel_count'] / terrace_angle['sector_total_pixels'].clip(lower=1)
    terrace_angle_records = terrace_angle.to_dict(orient='records')
    write_table(base_dir / 'terrace_angle_cluster_pooled.csv', terrace_angle_records)
    write_table(base_dir / 'terrace_angle_cluster_fractions.csv', terrace_angle_records)
    sectors_sorted = sorted(int(v) for v in terrace_angle['sector_id'].unique())
    for terrace_index in terraces_sorted:
        dom = np.full((len(sectors_sorted), len(frames_sorted)), -1, dtype=int)
        strength = np.zeros((len(sectors_sorted), len(frames_sorted)), dtype=float)
        sub_terr = terrace_angle[terrace_angle['terrace_index'] == terrace_index]
        for fi, frame_id in enumerate(frames_sorted):
            for si, sector_id in enumerate(sectors_sorted):
                sub = sub_terr[(sub_terr['frame_id'] == frame_id) & (sub_terr['sector_id'] == sector_id)].sort_values('cluster_id')
                if sub.empty:
                    continue
                p = sub['pooled_fraction'].to_numpy(dtype=float)
                cids = sub['cluster_id'].to_numpy(dtype=int)
                idx = int(np.argmax(p))
                dom[si, fi] = int(cids[idx])
                strength[si, fi] = float(p[idx])
        _save_dominant_cluster_map(
            base_dir / f'terrace_{int(terrace_index)+1:02d}_angular_chronogram.png',
            dom,
            strength,
            palette_map,
            title=f'Terrace {int(terrace_index)+1} angular chronogram',
            xlabel='frame',
            ylabel='sector',
            yticklabels=[str(int(s)) for s in sectors_sorted],
        )

    # per-hole terrace cluster fractions for front metrics and consistency
    per_hole = tensor_df.groupby(['frame_id', 'hole_id', 'terrace_index', 'cluster_id'], as_index=False)['pixel_count'].sum()
    per_hole_tot = per_hole.groupby(['frame_id', 'hole_id', 'terrace_index'], as_index=False)['pixel_count'].sum().rename(columns={'pixel_count': 'terrace_total_pixels'})
    per_hole = per_hole.merge(per_hole_tot, on=['frame_id', 'hole_id', 'terrace_index'], how='left')
    per_hole['pixel_fraction'] = per_hole['pixel_count'] / per_hole['terrace_total_pixels'].clip(lower=1)

    front_rows = _front_metric_rows(per_hole, cluster_ids, int(n_terraces), float(front_threshold_fraction))
    write_table(base_dir / 'cluster_front_metrics.csv', front_rows)

    # consistency against average-hole distribution
    consistency_rows: list[dict[str, Any]] = []
    pooled_map = {
        (int(r.frame_id), int(r.terrace_index), int(r.cluster_id)): float(r.pixel_fraction)
        for r in pooled.itertuples(index=False)
    }
    for (frame_id, terrace_index), grp in per_hole.groupby(['frame_id', 'terrace_index'], sort=True):
        pooled_vec = np.asarray([pooled_map.get((int(frame_id), int(terrace_index), int(cid)), 0.0) for cid in cluster_ids], dtype=float)
        sims = []
        for hole_id, hgrp in grp.groupby('hole_id'):
            hole_vec = np.asarray([
                float(hgrp.loc[hgrp['cluster_id'] == int(cid), 'pixel_fraction'].iloc[0]) if np.any(hgrp['cluster_id'] == int(cid)) else 0.0
                for cid in cluster_ids
            ], dtype=float)
            sim = _normalized_js_similarity(hole_vec, pooled_vec)
            if np.isfinite(sim):
                sims.append(sim)
        arr = np.asarray(sims, dtype=float)
        consistency_rows.append({
            'frame_id': int(frame_id),
            'terrace_index': int(terrace_index),
            'consistency_score': float(np.nanmean(arr)) if arr.size else None,
            'mean_js_similarity': float(np.nanmean(arr)) if arr.size else None,
            'sd_js_similarity': float(np.nanstd(arr)) if arr.size else None,
            'n_holes': int(arr.size),
        })
    write_table(base_dir / 'hole_consistency_by_terrace.csv', consistency_rows)

    # plots
    if progress_callback is not None:
        progress_callback(int(len(frames)) + 3, progress_total, "Rendering radial cluster plots")
    # front trajectories
    if front_rows:
        front_df = pd.DataFrame(front_rows).sort_values(['cluster_id', 'frame_id'])
        fig, ax = plt.subplots(figsize=(8.4, 4.8))
        for cid in cluster_ids:
            sub = front_df[front_df['cluster_id'] == cid].sort_values('frame_id')
            if sub.empty:
                continue
            x = sub['frame_id'].to_numpy(dtype=float)
            y = sub['weighted_mean_terrace'].to_numpy(dtype=float)
            q10 = sub['hole_q10_weighted_mean_terrace'].to_numpy(dtype=float)
            q90 = sub['hole_q90_weighted_mean_terrace'].to_numpy(dtype=float)
            color = palette_map.get(int(cid), (0.4, 0.4, 0.4))
            if np.any(np.isfinite(q10)) and np.any(np.isfinite(q90)):
                ax.fill_between(x, q10, q90, color=color, alpha=0.18)
            ax.plot(x, y, label=f'cluster {cid}', color=color, linewidth=2.0)
        ax.set_title('Cluster front trajectories')
        ax.set_xlabel('frame')
        ax.set_ylabel('weighted mean terrace')
        ax.set_ylim(0.75, float(int(n_terraces)) + 0.25)
        ax.grid(True, alpha=0.25)
        ax.legend(loc='best', fontsize=8)
        fig.tight_layout()
        fig.savefig(base_dir / 'cluster_front_trajectories.png', dpi=160, bbox_inches='tight')
        plt.close(fig)

    # consistency map
    if consistency_rows:
        cons_df = pd.DataFrame(consistency_rows)
        cons_mat = np.full((len(terraces_sorted), len(frames_sorted)), np.nan, dtype=float)
        for _, row in cons_df.iterrows():
            fi = frames_sorted.index(int(row['frame_id']))
            ti = terraces_sorted.index(int(row['terrace_index']))
            cons_mat[ti, fi] = float(row['consistency_score']) if row['consistency_score'] is not None else np.nan
        fig, ax = plt.subplots(figsize=(8.4, 4.8))
        im = ax.imshow(cons_mat, aspect='auto', interpolation='nearest', origin='lower', vmin=0.0, vmax=1.0)
        ax.set_title('Hole consistency by terrace')
        ax.set_xlabel('frame')
        ax.set_ylabel('terrace')
        ax.set_yticks(range(len(terraces_sorted)))
        ax.set_yticklabels([str(t + 1) for t in terraces_sorted])
        fig.colorbar(im, ax=ax, shrink=0.85, label='JS similarity to average hole')
        fig.tight_layout()
        fig.savefig(base_dir / 'hole_consistency_terrace_map.png', dpi=160, bbox_inches='tight')
        plt.close(fig)

    status.update({
        'status': 'ok',
        'message': 'ok',
        'n_frames': int(len(frames)),
        'n_clusters': int(len(cluster_ids)),
        'n_terraces': int(n_terraces),
        'n_angle_sectors': int(n_angle_sectors),
    })
    if progress_callback is not None:
        progress_callback(int(len(frames)) + 4, progress_total, "Radial cluster average-hole artifacts complete")
    write_json(base_dir / 'radial_cluster_status.json', status)
    return status
