from __future__ import annotations

import math
import time
from dataclasses import dataclass, field, replace
from typing import Callable, Sequence

import cv2
import numpy as np
from scipy import ndimage as ndi
from scipy.spatial import cKDTree

from holecolor.config.schema import GeometryConfig
from holecolor.core.types import HoleCandidate, LatticeModel
from holecolor.geometry.lattice_fit import estimate_lattice_basis

try:  # pragma: no cover - exercised when numba is installed
    from numba import njit, prange
except Exception:  # pragma: no cover - keeps the package importable without numba
    def njit(*args, **kwargs):
        def _decorator(func):
            return func
        if args and callable(args[0]) and len(args) == 1 and not kwargs:
            return args[0]
        return _decorator

    def prange(*args):
        return range(*args)


@dataclass(slots=True)
class StableGridDetectionDebug:
    support_circle: tuple[int, int, int] | None
    support_mask: np.ndarray | None
    raw_count: int
    filtered_count: int
    anchor_count: int
    recovered_strong_count: int
    predicted_only_full_count: int
    predicted_only_partial_count: int
    completed_count: int
    mode: str
    common_radius_px: float = 0.0
    tiers: list[dict[str, object]] = field(default_factory=list)
    predicted_only: list[dict[str, object]] = field(default_factory=list)
    sequence_frame_count: int = 0
    sequence_sampled_count: int = 0
    sequence_sample_indices: list[int] = field(default_factory=list)
    sequence_sampling_history: list[dict[str, object]] = field(default_factory=list)
    watchdog_events: list[dict[str, object]] = field(default_factory=list)


@dataclass(slots=True)
class StableGridDetectionResult:
    accepted_candidates: list[HoleCandidate]
    lattice: LatticeModel
    lattice_indices: dict[int, tuple[int, int]]
    debug: StableGridDetectionDebug


class DetectorWatchdogTimeout(RuntimeError):
    def __init__(self, label: str, phase: str, elapsed_s: float, limit_s: float) -> None:
        self.label = str(label)
        self.phase = str(phase)
        self.elapsed_s = float(elapsed_s)
        self.limit_s = float(limit_s)
        super().__init__(
            f"detector watchdog timeout in {self.label} at {self.phase}: "
            f"elapsed={self.elapsed_s:.1f}s limit={self.limit_s:.1f}s"
        )


@dataclass(slots=True)
class _DetectorWatchdog:
    label: str
    limit_s: float
    started_at: float = field(default_factory=time.perf_counter)

    def check(self, phase: str) -> None:
        limit = float(self.limit_s)
        if limit <= 0.0:
            return
        elapsed = float(time.perf_counter() - self.started_at)
        if elapsed > limit:
            raise DetectorWatchdogTimeout(self.label, phase, elapsed, limit)

    def event(self, reason: str) -> dict[str, object]:
        return {
            "label": str(self.label),
            "reason": str(reason),
            "elapsed_s": float(time.perf_counter() - self.started_at),
            "limit_s": float(self.limit_s),
        }


def _robust_scale(x: np.ndarray, mask: np.ndarray | None = None, lo: float = 1.0, hi: float = 99.0) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float32)
    if mask is not None and np.any(mask):
        vals = arr[mask]
    else:
        vals = arr.reshape(-1)
    if vals.size == 0:
        return np.zeros_like(arr, dtype=np.float32)
    a, b = np.percentile(vals, [lo, hi])
    den = max(float(b - a), 1e-9)
    out = (arr - float(a)) / den
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def _gradient_magnitude(im: np.ndarray) -> np.ndarray:
    gx = cv2.Sobel(im.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(im.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
    return np.sqrt(gx * gx + gy * gy)


def _sequence_gray_stack(images: Sequence[np.ndarray]) -> np.ndarray:
    return np.stack([cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32) for img in images], axis=0)


def _geometry_sequence_sample_target(n_frames: int) -> int:
    n = int(n_frames)
    if n <= 16:
        return max(1, n)
    temporal_scale = 2 * int(math.ceil(math.log2(float(n) + 1.0)))
    return max(8, min(n, min(24, temporal_scale)))


def _representative_sequence_indices(n_frames: int, reference_index: int) -> list[int]:
    n = int(n_frames)
    if n <= 0:
        return []
    reference_index = int(np.clip(reference_index, 0, n - 1))
    target = _geometry_sequence_sample_target(n)
    if target >= n:
        return list(range(n))
    idx = {0, n - 1, reference_index}
    idx.update(int(i) for i in np.linspace(0, n - 1, num=target, dtype=int))
    while len(idx) < target:
        gaps = sorted((b - a, a, b) for a, b in zip(sorted(idx)[:-1], sorted(idx)[1:]) if b - a > 1)
        if not gaps:
            break
        _gap, a, b = gaps[-1]
        idx.add((a + b) // 2)
    return sorted(idx)


def _homogeneous_midpoint_order(n_frames: int, reference_index: int) -> list[int]:
    n = int(n_frames)
    if n <= 0:
        return []
    reference_index = int(np.clip(reference_index, 0, n - 1))
    selected: set[int] = {0, n - 1, reference_index}
    order: list[int] = []
    for idx in (0, n - 1, reference_index):
        if idx not in order:
            order.append(idx)
    while len(selected) < n:
        sorted_idx = sorted(selected)
        gaps = [(b - a, a, b) for a, b in zip(sorted_idx[:-1], sorted_idx[1:]) if b - a > 1]
        if not gaps:
            break
        _gap, a, b = sorted(gaps, key=lambda item: (item[0], -item[1]), reverse=True)[0]
        midpoint = int((a + b) // 2)
        if midpoint in selected:
            midpoint = next((i for i in range(a + 1, b) if i not in selected), -1)
            if midpoint < 0:
                break
        selected.add(midpoint)
        order.append(midpoint)
    return order


def _scale_geometry_cfg(cfg: GeometryConfig, scale: float) -> GeometryConfig:
    if scale <= 1.0:
        return cfg
    return replace(
        cfg,
        min_radius_px=max(2.0, float(cfg.min_radius_px) / float(scale)),
        max_radius_px=max(3.0, float(cfg.max_radius_px) / float(scale)),
        duplicate_suppression_px=max(0.2, float(cfg.duplicate_suppression_px) / float(scale)),
    )


def _sampling_gray_frame(image: np.ndarray, target_max_dim: int = 420) -> tuple[np.ndarray, float]:
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32)
    h, w = gray.shape
    max_dim = max(h, w)
    if max_dim <= int(target_max_dim):
        return gray, 1.0
    scale = float(max_dim) / float(target_max_dim)
    out_shape = (max(16, int(round(float(h) / scale))), max(16, int(round(float(w) / scale))))
    small = cv2.resize(gray, (out_shape[1], out_shape[0]), interpolation=cv2.INTER_AREA).astype(np.float32)
    return small, scale


def _sampling_gray_cache(images: Sequence[np.ndarray], target_max_dim: int = 420) -> tuple[list[np.ndarray], float]:
    grays: list[np.ndarray] = []
    scale = 1.0
    for image in images:
        gray, frame_scale = _sampling_gray_frame(image, target_max_dim=target_max_dim)
        grays.append(gray)
        scale = max(scale, float(frame_scale))
    return grays, scale


def _sampling_geometry_signature_from_gray_cache(
    gray_cache: Sequence[np.ndarray],
    scale: float,
    sample_indices: list[int],
    reference_index: int,
    cfg: GeometryConfig,
) -> dict[str, object]:
    if not sample_indices:
        return {'ok': False, 'accepted_count': 0}
    gray_stack = np.stack([gray_cache[int(idx)] for idx in sample_indices], axis=0)
    sampled_ref = sample_indices.index(reference_index) if reference_index in sample_indices else 0
    reference_gray = gray_stack[sampled_ref]
    support_circle, wafer_mask, mean_gray, std_gray, mad_gray = _detect_support_from_sequence(gray_stack)
    scaled_cfg = _scale_geometry_cfg(cfg, scale)
    evidence = _build_evidence_maps(mean_gray, std_gray, mad_gray, reference_gray, wafer_mask, scaled_cfg)
    seeds = _extract_consensus_seeds(evidence, wafer_mask, scaled_cfg)
    accepted, _provisional = _initial_fitted_candidates(evidence, seeds, scaled_cfg)
    accepted = _select_common_radius_rows(accepted)
    radii = np.asarray([float(row['r']) for row in accepted], dtype=np.float32)
    common_radius_small = float(np.median(radii)) if radii.size else 0.5 * sum(_effective_initial_radius_bounds(mean_gray.shape, scaled_cfg))
    basis = _estimate_probe_basis(accepted)
    spacing_u = spacing_v = angle = None
    if basis is not None:
        _xy, u, v, _origin = basis
        spacing_u = float(np.linalg.norm(u)) * float(scale)
        spacing_v = float(np.linalg.norm(v)) * float(scale)
        angle = float(math.degrees(math.atan2(float(u[1]), float(u[0]))))
    if support_circle is None:
        support_full = None
    else:
        support_full = (
            float(support_circle[0]) * float(scale),
            float(support_circle[1]) * float(scale),
            float(support_circle[2]) * float(scale),
        )
    return {
        'ok': bool(len(accepted) >= 8 and support_full is not None),
        'accepted_count': int(len(accepted)),
        'seed_count': int(len(seeds)),
        'support_circle': support_full,
        'common_radius_px': float(common_radius_small * float(scale)),
        'spacing_u_px': spacing_u,
        'spacing_v_px': spacing_v,
        'angle_deg': angle,
    }


def _sampling_signatures_stable(prev: dict[str, object] | None, cur: dict[str, object]) -> bool:
    if prev is None or not bool(prev.get('ok')) or not bool(cur.get('ok')):
        return False
    prev_circle = prev.get('support_circle')
    cur_circle = cur.get('support_circle')
    if prev_circle is None or cur_circle is None:
        return False
    px, py, pr = [float(v) for v in prev_circle]
    cx, cy, cr = [float(v) for v in cur_circle]
    r_scale = max(2.0, float(cur.get('common_radius_px') or prev.get('common_radius_px') or 2.0))
    if math.hypot(cx - px, cy - py) > max(2.0, 0.75 * r_scale):
        return False
    if abs(cr - pr) > max(2.0, 0.75 * r_scale):
        return False
    prev_count = int(prev.get('accepted_count') or 0)
    cur_count = int(cur.get('accepted_count') or 0)
    if abs(cur_count - prev_count) > max(3, int(round(0.15 * max(prev_count, cur_count, 1)))):
        return False
    prev_rad = float(prev.get('common_radius_px') or 0.0)
    cur_rad = float(cur.get('common_radius_px') or 0.0)
    if prev_rad > 0 and cur_rad > 0 and abs(cur_rad - prev_rad) > max(1.0, 0.30 * r_scale):
        return False
    for key in ('spacing_u_px', 'spacing_v_px'):
        a = prev.get(key)
        b = cur.get(key)
        if a is None or b is None:
            continue
        spacing_scale = max(4.0, abs(float(a)), abs(float(b)))
        if abs(float(a) - float(b)) > max(2.0, 0.08 * spacing_scale):
            return False
    a0 = prev.get('angle_deg')
    a1 = cur.get('angle_deg')
    if a0 is not None and a1 is not None:
        delta = abs((float(a1) - float(a0) + 90.0) % 180.0 - 90.0)
        if delta > 3.0:
            return False
    return True


def _adaptive_sequence_indices(
    images: Sequence[np.ndarray],
    reference_index: int,
    cfg: GeometryConfig,
    *,
    min_fraction: float = 0.05,
    stable_steps_required: int = 5,
    post_stable_refinement_rounds: int = 2,
    watchdog: _DetectorWatchdog | None = None,
) -> tuple[list[int], list[dict[str, object]]]:
    n = int(len(images))
    if n <= 0:
        return [], []
    if n <= 16:
        idx = list(range(n))
        return idx, [{'sampled_count': len(idx), 'stable_run': 0, 'stable': True, 'reason': 'short_sequence_all_frames'}]
    if watchdog is not None:
        watchdog.check("adaptive sampling grayscale cache")
    gray_cache, gray_scale = _sampling_gray_cache(images)
    if watchdog is not None:
        watchdog.check("adaptive sampling grayscale cache complete")
    order = _homogeneous_midpoint_order(n, reference_index)
    min_samples = max(3, int(math.ceil(float(min_fraction) * float(n))))
    first_signature_samples = max(min_samples, int(stable_steps_required) + 1)
    selected: list[int] = []
    selected_set: set[int] = set()
    history: list[dict[str, object]] = []
    prev_signature: dict[str, object] | None = None
    stable_run = 0
    refinement_remaining: int | None = None
    for idx in order:
        if int(idx) in selected_set:
            continue
        selected.append(int(idx))
        selected_set.add(int(idx))
        if len(selected) < 2:
            continue
        sample_sorted = sorted(selected)
        if len(sample_sorted) < first_signature_samples:
            continue
        if watchdog is not None:
            watchdog.check(f"adaptive sampling signature before {len(sample_sorted)} frames")
        signature = _sampling_geometry_signature_from_gray_cache(gray_cache, gray_scale, sample_sorted, reference_index, cfg)
        if watchdog is not None:
            watchdog.check(f"adaptive sampling signature after {len(sample_sorted)} frames")
        stable = _sampling_signatures_stable(prev_signature, signature)
        stable_run = stable_run + 1 if stable else 0
        reason = None
        row = {
            'sampled_count': int(len(sample_sorted)),
            'sample_indices': [int(i) for i in sample_sorted],
            'stable': bool(stable),
            'stable_run': int(stable_run),
            'accepted_count': int(signature.get('accepted_count') or 0),
            'common_radius_px': float(signature.get('common_radius_px') or 0.0),
            'support_circle': None if signature.get('support_circle') is None else [float(v) for v in signature['support_circle']],
        }
        prev_signature = signature
        if len(sample_sorted) >= min_samples and stable_run >= int(stable_steps_required):
            if refinement_remaining is None:
                refinement_remaining = max(0, int(post_stable_refinement_rounds))
                reason = 'stability_reached_starting_refinement_rounds' if refinement_remaining > 0 else 'stability_reached'
            elif stable:
                refinement_remaining -= 1
                reason = f'post_stability_refinement_round_{max(0, int(post_stable_refinement_rounds) - refinement_remaining)}'
            if refinement_remaining is not None and refinement_remaining <= 0:
                row['reason'] = reason or 'stability_reached_after_refinement_rounds'
                history.append(row)
                break
        elif refinement_remaining is not None and not stable:
            refinement_remaining = None
            reason = 'stability_lost_during_refinement'
        if reason is not None:
            row['reason'] = reason
            row['refinement_remaining'] = None if refinement_remaining is None else int(refinement_remaining)
        history.append(row)
    return sorted(selected_set), history


def _next_midpoint_expansion_indices(sample_indices: list[int], n_frames: int, reference_index: int, count: int) -> list[int]:
    selected = set(int(i) for i in sample_indices)
    for idx in _homogeneous_midpoint_order(n_frames, reference_index):
        if idx not in selected:
            selected.add(int(idx))
            if len(selected) >= len(sample_indices) + int(count):
                break
    return sorted(selected)


def _full_detection_consistent_with_sampling(result: StableGridDetectionResult, sample_history: list[dict[str, object]]) -> bool:
    if not sample_history:
        return True
    last = sample_history[-1]
    signature_count = int(last.get('accepted_count') or 0)
    if signature_count >= 16:
        min_completed = max(8, int(round(0.68 * float(signature_count))))
        if int(result.debug.completed_count) < min_completed:
            return False
    if float(result.lattice.confidence) < 0.70 and int(result.debug.anchor_count) < 8:
        return False
    support = last.get('support_circle')
    if support is not None and result.debug.support_circle is not None:
        sx, sy, sr = [float(v) for v in support]
        cx, cy, cr = [float(v) for v in result.debug.support_circle]
        r_scale = max(2.0, float(last.get('common_radius_px') or result.debug.common_radius_px or 2.0))
        if math.hypot(cx - sx, cy - sy) > max(20.0, 4.0 * r_scale):
            return False
        if abs(cr - sr) > max(24.0, 4.0 * r_scale):
            return False
    return True


def _annotate_sequence_sampling_debug(result: StableGridDetectionResult, n_frames: int, sample_indices: list[int], history: list[dict[str, object]] | None = None) -> StableGridDetectionResult:
    result.debug.sequence_frame_count = int(n_frames)
    result.debug.sequence_sampled_count = int(len(sample_indices))
    result.debug.sequence_sample_indices = [int(i) for i in sample_indices]
    result.debug.sequence_sampling_history = list(history or [])
    return result


def _scale_geometry_rows(rows: list[dict[str, object]], scale: float) -> list[dict[str, object]]:
    if scale == 1.0:
        return [dict(row) for row in rows]
    spatial_keys = {
        'x',
        'y',
        'x_final',
        'y_final',
        'x_pred',
        'y_pred',
        'x_rec',
        'y_rec',
        'seed_x',
        'seed_y',
    }
    radius_keys = {'r', 'r_final', 'r_rec', 'radius_px', 'r_loose'}
    out: list[dict[str, object]] = []
    for row in rows:
        nr = dict(row)
        for key in spatial_keys | radius_keys:
            if key in nr:
                try:
                    nr[key] = float(nr[key]) / float(scale)
                except Exception:
                    pass
        out.append(nr)
    return out


def _scale_detection_result(result: StableGridDetectionResult, scale: float, out_shape: tuple[int, int]) -> StableGridDetectionResult:
    if scale == 1.0:
        return result
    candidates = [
        HoleCandidate(
            float(c.x) * float(scale),
            float(c.y) * float(scale),
            float(c.radius_px) * float(scale),
            float(c.ellipticity),
            float(c.boundary_contrast),
            float(c.confidence),
        )
        for c in result.accepted_candidates
    ]
    lattice = result.lattice
    scaled_lattice = LatticeModel(
        float(lattice.origin_x) * float(scale),
        float(lattice.origin_y) * float(scale),
        (float(lattice.basis_u[0]) * float(scale), float(lattice.basis_u[1]) * float(scale)),
        (float(lattice.basis_v[0]) * float(scale), float(lattice.basis_v[1]) * float(scale)),
        float(lattice.angle_deg),
        float(lattice.spacing_u_px) * float(scale),
        float(lattice.spacing_v_px) * float(scale),
        float(lattice.confidence),
    )
    debug = result.debug
    support_circle = None
    if debug.support_circle is not None:
        support_circle = (
            int(round(float(debug.support_circle[0]) * float(scale))),
            int(round(float(debug.support_circle[1]) * float(scale))),
            int(round(float(debug.support_circle[2]) * float(scale))),
        )
    support_mask = None
    if debug.support_mask is not None:
        support_mask = _resize_bool_mask(debug.support_mask, out_shape)
    scaled_tiers = _scale_geometry_rows(list(debug.tiers), 1.0 / float(scale))
    scaled_predicted = _scale_geometry_rows(list(debug.predicted_only), 1.0 / float(scale))
    scaled_debug = StableGridDetectionDebug(
        support_circle,
        support_mask,
        int(debug.raw_count),
        int(debug.filtered_count),
        int(debug.anchor_count),
        int(debug.recovered_strong_count),
        int(debug.predicted_only_full_count),
        int(debug.predicted_only_partial_count),
        int(debug.completed_count),
        str(debug.mode),
        float(debug.common_radius_px) * float(scale),
        tiers=scaled_tiers,
        predicted_only=scaled_predicted,
        sequence_frame_count=int(debug.sequence_frame_count),
        sequence_sampled_count=int(debug.sequence_sampled_count),
        sequence_sample_indices=list(debug.sequence_sample_indices),
        sequence_sampling_history=list(debug.sequence_sampling_history),
        watchdog_events=list(debug.watchdog_events),
    )
    return StableGridDetectionResult(candidates, scaled_lattice, dict(result.lattice_indices), scaled_debug)


def _resize_bool_mask(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    return cv2.resize(mask.astype(np.uint8), (int(shape[1]), int(shape[0])), interpolation=cv2.INTER_NEAREST).astype(bool)


def _fit_circle_least_squares(points: np.ndarray) -> tuple[float, float, float] | None:
    if points.shape[0] < 3:
        return None
    x = points[:, 0].astype(np.float64)
    y = points[:, 1].astype(np.float64)
    A = np.column_stack([2.0 * x, 2.0 * y, np.ones_like(x)])
    b = x * x + y * y
    try:
        sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    except np.linalg.LinAlgError:
        return None
    cx, cy, c = [float(v) for v in sol]
    r2 = c + cx * cx + cy * cy
    if r2 <= 1.0:
        return None
    return cx, cy, float(np.sqrt(r2))


def _fit_circle_weighted_least_squares(points: np.ndarray, weights: np.ndarray | None = None) -> tuple[float, float, float] | None:
    if points.shape[0] < 3:
        return None
    x = points[:, 0].astype(np.float64)
    y = points[:, 1].astype(np.float64)
    A = np.column_stack([2.0 * x, 2.0 * y, np.ones_like(x)])
    b = x * x + y * y
    if weights is not None:
        ww = np.sqrt(np.clip(np.asarray(weights, dtype=np.float64), 1e-6, None))
        A = A * ww[:, None]
        b = b * ww
    try:
        sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    except np.linalg.LinAlgError:
        return None
    cx, cy, c = [float(v) for v in sol]
    r2 = c + cx * cx + cy * cy
    if r2 <= 1.0:
        return None
    return cx, cy, float(np.sqrt(r2))


def _extreme_boundary_points(mask: np.ndarray, side: str) -> np.ndarray:
    H, W = mask.shape
    pts: list[tuple[float, float]] = []
    if side == 'left':
        for y in range(H):
            xs = np.flatnonzero(mask[y])
            if xs.size:
                pts.append((float(xs.min()), float(y)))
    elif side == 'right':
        for y in range(H):
            xs = np.flatnonzero(mask[y])
            if xs.size:
                pts.append((float(xs.max()), float(y)))
    elif side == 'top':
        for x in range(W):
            ys = np.flatnonzero(mask[:, x])
            if ys.size:
                pts.append((float(x), float(ys.min())))
    elif side == 'bottom':
        for x in range(W):
            ys = np.flatnonzero(mask[:, x])
            if ys.size:
                pts.append((float(x), float(ys.max())))
    return np.asarray(pts, dtype=np.float32)


def _fit_partial_support_circle(mask: np.ndarray) -> tuple[float, float, float] | None:
    H, W = mask.shape
    touched = {
        'left': bool(np.any(mask[:, 0])),
        'right': bool(np.any(mask[:, W - 1])),
        'top': bool(np.any(mask[0, :])),
        'bottom': bool(np.any(mask[H - 1, :])),
    }
    if not any(touched.values()):
        return None
    fit_mask = cv2.dilate(mask.astype(np.uint8), np.ones((31, 31), np.uint8), iterations=1).astype(bool)
    yy, xx = np.indices(mask.shape)
    opposite = {'left': 'right', 'right': 'left', 'top': 'bottom', 'bottom': 'top'}
    outside_checks = {
        'left': lambda cx, cy: cx < 0.0,
        'right': lambda cx, cy: cx > float(W - 1),
        'top': lambda cx, cy: cy < 0.0,
        'bottom': lambda cx, cy: cy > float(H - 1),
    }
    point_sets: list[tuple[str, np.ndarray]] = []
    for border_side, is_touched in touched.items():
        if not is_touched:
            continue
        pts = _extreme_boundary_points(fit_mask, opposite[border_side])
        if pts.shape[0] >= 12:
            point_sets.append((border_side, pts))
    if len(point_sets) >= 2:
        combo = np.concatenate([pts for _, pts in point_sets], axis=0)
        combo = np.unique(np.round(combo).astype(np.int32), axis=0).astype(np.float32)
        if combo.shape[0] >= 20:
            point_sets.append(('combo', combo))
    best: tuple[float, tuple[float, float, float]] | None = None
    for border_side, pts in point_sets:
        fit = _fit_circle_least_squares(pts)
        if fit is None:
            continue
        cx, cy, r = fit
        if r <= 0.0:
            continue
        radial = np.sqrt((pts[:, 0] - cx) ** 2 + (pts[:, 1] - cy) ** 2)
        residual = float(np.median(np.abs(radial - r)))
        circle_mask = ((xx - cx) ** 2 + (yy - cy) ** 2) <= (1.03 * r) ** 2
        coverage = float(np.sum(circle_mask & mask) / max(int(mask.sum()), 1))
        if coverage < 0.80:
            continue
        if border_side == 'combo':
            outside_bonus = 0.05 if any(check(cx, cy) for check in outside_checks.values()) else 0.0
        else:
            outside_bonus = 0.10 if outside_checks[border_side](cx, cy) else 0.0
        residual_score = float(np.clip(1.0 - residual / max(r, 1e-6), 0.0, 1.0))
        score = float(coverage + residual_score + outside_bonus)
        if best is None or score > best[0]:
            best = (score, (cx, cy, r))
    return None if best is None else best[1]


def _circle_arc_span(points: np.ndarray, cx: float, cy: float) -> float:
    if points.shape[0] < 2:
        return 0.0
    angles = np.sort(np.mod(np.arctan2(points[:, 1] - cy, points[:, 0] - cx), 2.0 * np.pi))
    gaps = np.diff(np.r_[angles, angles[0] + 2.0 * np.pi])
    return float(2.0 * np.pi - np.max(gaps))


def _boundary_envelope_points(mask: np.ndarray, side: str) -> np.ndarray:
    h, w = mask.shape
    pts: list[tuple[float, float]] = []
    if side == 'top':
        for x in range(w):
            ys = np.flatnonzero(mask[:, x])
            if ys.size:
                pts.append((float(x), float(ys.min())))
    elif side == 'bottom':
        for x in range(w):
            ys = np.flatnonzero(mask[:, x])
            if ys.size:
                pts.append((float(x), float(ys.max())))
    elif side == 'left':
        for y in range(h):
            xs = np.flatnonzero(mask[y, :])
            if xs.size:
                pts.append((float(xs.min()), float(y)))
    elif side == 'right':
        for y in range(h):
            xs = np.flatnonzero(mask[y, :])
            if xs.size:
                pts.append((float(xs.max()), float(y)))
    return np.asarray(pts, dtype=np.float32)


def _visible_boundary_circle_candidates(mask: np.ndarray) -> list[tuple[float, tuple[float, float, float]]]:
    h, w = mask.shape
    if int(mask.sum()) < 12:
        return []
    work = cv2.dilate(mask.astype(np.uint8), np.ones((7, 7), np.uint8), iterations=1).astype(bool)
    candidates: list[tuple[float, tuple[float, float, float]]] = []
    for side in ('top', 'bottom', 'left', 'right'):
        pts = _boundary_envelope_points(work, side)
        if pts.shape[0] < 40:
            continue
        margin = max(2, int(round(0.006 * float(min(h, w)))))
        pts = pts[
            (pts[:, 0] > margin)
            & (pts[:, 0] < w - 1 - margin)
            & (pts[:, 1] > margin)
            & (pts[:, 1] < h - 1 - margin)
        ]
        n_pts = int(pts.shape[0])
        if n_pts < 40:
            continue
        sizes = sorted(
            {
                n_pts,
                int(round(0.85 * n_pts)),
                int(round(0.72 * n_pts)),
                int(round(0.60 * n_pts)),
                int(round(0.50 * n_pts)),
                int(round(0.40 * n_pts)),
                int(round(0.32 * n_pts)),
            },
            reverse=True,
        )
        for size in sizes:
            if size < 35:
                continue
            step = max(1, size // 12)
            for start in range(0, n_pts - size + 1, step):
                run = pts[start:start + size]
                fit = _fit_circle_least_squares(run)
                if fit is None:
                    continue
                cx, cy, r = fit
                if r < 0.12 * float(min(h, w)) or r > 2.0 * float(max(h, w)):
                    continue
                radial = np.sqrt((run[:, 0] - cx) ** 2 + (run[:, 1] - cy) ** 2)
                residual = np.abs(radial - r)
                median_residual = float(np.median(residual))
                span = _circle_arc_span(run, cx, cy)
                score = float(span * math.sqrt(float(size)) / (1.0 + median_residual))
                candidates.append((score, (cx, cy, r)))
    candidates.sort(key=lambda item: item[0], reverse=True)
    deduped: list[tuple[float, tuple[float, float, float]]] = []
    for score, circle in candidates:
        cx, cy, r = circle
        if any(math.hypot(cx - ox, cy - oy) < 0.03 * max(r, orad) and abs(r - orad) < 0.05 * max(r, orad) for _, (ox, oy, orad) in deduped):
            continue
        deduped.append((score, circle))
        if len(deduped) >= 12:
            break
    return deduped


def _fit_visible_boundary_circle(mask: np.ndarray) -> tuple[float, float, float] | None:
    candidates = _visible_boundary_circle_candidates(mask)
    return None if not candidates else candidates[0][1]


def _detect_dark_support_from_mean(mean_gray: np.ndarray) -> tuple[tuple[int, int, int], np.ndarray] | None:
    h, w = mean_gray.shape
    border = max(8, min(h, w) // 20)
    border_mask = np.zeros((h, w), dtype=bool)
    border_mask[:border, :] = True
    border_mask[h - border:, :] = True
    border_mask[:, :border] = True
    border_mask[:, w - border:] = True

    blur = cv2.GaussianBlur(mean_gray.astype(np.float32), (0, 0), 9.0)
    border_level = float(np.median(blur[border_mask])) if np.any(border_mask) else float(np.median(blur))
    low_level = float(np.percentile(blur, 10))
    if border_level - low_level < 12.0:
        return None
    threshold = 0.5 * (border_level + low_level)
    dark = (blur < threshold).astype(np.uint8)
    dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, np.ones((31, 31), np.uint8))
    dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, np.ones((9, 9), np.uint8))

    n, labels, stats, _ = cv2.connectedComponentsWithStats(dark, 8)
    if n <= 1:
        return None
    yy, xx = np.indices((h, w))
    best: tuple[float, tuple[float, float, float], np.ndarray] | None = None
    min_area = max(200, int(round(0.018 * h * w)))
    for lab in range(1, n):
        area = int(stats[lab, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        comp_mask = labels == lab
        comp_vals = blur[comp_mask]
        if comp_vals.size == 0:
            continue
        contrast = border_level - float(np.median(comp_vals))
        if contrast < 10.0:
            continue
        fit = _fit_visible_boundary_circle(comp_mask)
        if fit is None:
            fit = _fit_partial_support_circle(comp_mask)
        if fit is None:
            contours, _ = cv2.findContours(comp_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            if not contours:
                continue
            (xc, yc), r = cv2.minEnclosingCircle(max(contours, key=cv2.contourArea))
            fit = (float(xc), float(yc), float(r))
        xc, yc, r = fit
        if r < 0.28 * min(h, w):
            continue
        circle_mask = ((xx - xc) ** 2 + (yy - yc) ** 2) <= r ** 2
        coverage = float(np.sum(circle_mask & comp_mask) / max(area, 1))
        if coverage < 0.45:
            continue
        exact_mask = circle_mask & comp_mask
        n_exact, exact_labels, exact_stats, _ = cv2.connectedComponentsWithStats(exact_mask.astype(np.uint8), 8)
        if n_exact > 1:
            largest_exact = 1 + int(np.argmax(exact_stats[1:, cv2.CC_STAT_AREA]))
            exact_mask = exact_labels == largest_exact
        score = float(np.sum(exact_mask) / max(h * w, 1)) + 0.012 * contrast + 0.25 * coverage
        if best is None or score > best[0]:
            best = (score, (xc, yc, r), exact_mask)
    if best is None:
        return None
    _, (xc, yc, r), wafer_mask = best
    return (int(round(xc)), int(round(yc)), int(round(r))), wafer_mask.astype(bool)


def _longest_true_run(mask_1d: np.ndarray) -> tuple[int, int, int]:
    best_start = 0
    best_len = 0
    cur_start = 0
    cur_len = 0
    for idx, val in enumerate(mask_1d.astype(bool)):
        if val:
            if cur_len == 0:
                cur_start = int(idx)
            cur_len += 1
            if cur_len > best_len:
                best_start = cur_start
                best_len = cur_len
        else:
            cur_len = 0
    return int(best_start), int(best_start + best_len - 1), int(best_len)


def _detect_rectangular_valid_content_support_from_mean(mean_gray: np.ndarray) -> tuple[tuple[int, int, int], np.ndarray] | None:
    h, w = mean_gray.shape
    if h < 64 or w < 64:
        return None
    blur = cv2.GaussianBlur(mean_gray.astype(np.float32), (0, 0), 5.0)
    low = float(np.percentile(blur, 2))
    high = float(np.percentile(blur, 72))
    if high - low < 10.0:
        return None
    valid_threshold = low + max(6.0, 0.08 * (high - low))
    pixel_valid = blur > valid_threshold
    row_valid = np.mean(pixel_valid, axis=1) > 0.50
    y0, y1, y_len = _longest_true_run(row_valid)
    if y_len <= 0:
        return None
    col_valid = np.mean(pixel_valid[y0:y1 + 1, :], axis=0) > 0.82
    x0, x1, x_len = _longest_true_run(col_valid)
    if y_len <= 0 or x_len <= 0:
        return None
    if y_len < 0.45 * h or x_len < 0.45 * w:
        return None
    if y_len > 0.97 * h and x_len > 0.97 * w:
        return None
    bbox_valid = pixel_valid[y0:y1 + 1, x0:x1 + 1]
    if bbox_valid.size == 0:
        return None
    fill = float(np.mean(bbox_valid))
    # A circular wafer silhouette fills only about pi/4 of its bounding box.
    # This path is for rectangular acquisition content, such as letterboxed microscopy.
    if fill < 0.88:
        return None
    mask = np.zeros((h, w), dtype=bool)
    mask[y0:y1 + 1, x0:x1 + 1] = True
    cx = 0.5 * float(x0 + x1)
    cy = 0.5 * float(y0 + y1)
    r = 0.5 * math.hypot(float(x1 - x0 + 1), float(y1 - y0 + 1))
    return (int(round(cx)), int(round(cy)), int(math.ceil(r))), mask


def _support_mask_is_rectangular_content(mask: np.ndarray) -> bool:
    if mask.size == 0 or not np.any(mask):
        return False
    h, w = mask.shape
    ys, xs = np.where(mask)
    y0, y1 = int(ys.min()), int(ys.max())
    x0, x1 = int(xs.min()), int(xs.max())
    area = float(np.count_nonzero(mask))
    bbox_area = float(max(1, (y1 - y0 + 1) * (x1 - x0 + 1)))
    fill = area / bbox_area
    touches_frame = x0 <= 1 or y0 <= 1 or x1 >= w - 2 or y1 >= h - 2
    return bool(fill >= 0.94 and touches_frame)


def _circle_center_outside_frame(shape: tuple[int, int], circle: tuple[int, int, int] | None) -> bool:
    if circle is None:
        return False
    h, w = shape
    cx, cy, _r = circle
    return bool(float(cx) < 0.0 or float(cy) < 0.0 or float(cx) >= float(w) or float(cy) >= float(h))


def _support_circle_has_visible_boundary(
    mean_gray: np.ndarray,
    reference_gray: np.ndarray,
    circle: tuple[float, float, float] | tuple[int, int, int] | None,
) -> bool:
    if circle is None:
        return False
    h, w = mean_gray.shape
    cx, cy, r = float(circle[0]), float(circle[1]), float(circle[2])
    if r <= 1.0:
        return False
    delta = float(np.clip(0.018 * float(min(h, w)), 5.0, max(6.0, 0.055 * r)))
    mean_scaled = _robust_scale(mean_gray)
    ref_scaled = _robust_scale(reference_gray)
    grad_scaled = _robust_scale(0.65 * _gradient_magnitude(mean_scaled) + 0.35 * _gradient_magnitude(ref_scaled))
    n = max(160, min(960, int(round(2.0 * math.pi * r / max(delta, 1.0)))))
    visible = 0
    strong = 0
    contrasts: list[float] = []
    gradients: list[float] = []
    for angle in np.linspace(0.0, 2.0 * math.pi, n, endpoint=False):
        ca = float(math.cos(float(angle)))
        sa = float(math.sin(float(angle)))
        xb = cx + r * ca
        yb = cy + r * sa
        xi = cx + (r - delta) * ca
        yi = cy + (r - delta) * sa
        xo = cx + (r + delta) * ca
        yo = cy + (r + delta) * sa
        if not (
            1.0 <= xi < w - 2
            and 1.0 <= yi < h - 2
            and 1.0 <= xo < w - 2
            and 1.0 <= yo < h - 2
            and 1.0 <= xb < w - 2
            and 1.0 <= yb < h - 2
        ):
            continue
        visible += 1
        inside = 0.5 * (_bilinear_sample(mean_scaled, xi, yi) + _bilinear_sample(ref_scaled, xi, yi))
        outside = 0.5 * (_bilinear_sample(mean_scaled, xo, yo) + _bilinear_sample(ref_scaled, xo, yo))
        contrast = abs(outside - inside)
        gradient = _bilinear_sample(grad_scaled, xb, yb)
        contrasts.append(float(contrast))
        gradients.append(float(gradient))
        if (contrast >= 0.08 and gradient >= 0.08) or contrast >= 0.13 or gradient >= 0.18:
            strong += 1
    if visible < max(18, int(0.04 * n)):
        return False
    contrast_arr = np.asarray(contrasts, dtype=np.float32)
    grad_arr = np.asarray(gradients, dtype=np.float32)
    visible_fraction = float(visible / max(n, 1))
    supported_arc_fraction = visible_fraction * float(strong / max(visible, 1))
    rim_score = _rim_model_score_from_maps(mean_scaled, ref_scaled, grad_scaled, (cx, cy, r), delta)
    background_similarity = _background_similarity_map(mean_gray, reference_gray)
    if background_similarity is not None:
        background_transition = _circle_background_transition_score(background_similarity, (cx, cy, r), delta)
        return bool(supported_arc_fraction >= 0.025 and background_transition >= 0.035)
    return bool(
        supported_arc_fraction >= 0.080
        and rim_score >= 0.100
        and (
            float(np.percentile(contrast_arr, 75)) >= 0.16
            or float(np.percentile(grad_arr, 75)) >= 0.24
        )
    )


def _detect_support_from_sequence(gray_stack: np.ndarray):
    mean_gray = gray_stack.mean(axis=0).astype(np.float32)
    median_gray = np.median(gray_stack, axis=0).astype(np.float32)
    std_gray = gray_stack.std(axis=0).astype(np.float32)
    mad_gray = np.median(np.abs(gray_stack - median_gray[None, ...]), axis=0).astype(np.float32)

    H, W = mean_gray.shape
    dark_support = _detect_dark_support_from_mean(mean_gray)
    if dark_support is not None:
        support_circle, wafer_mask = dark_support
        if _support_circle_has_visible_boundary(mean_gray, median_gray, support_circle):
            return support_circle, wafer_mask, mean_gray, std_gray, mad_gray

    rectangular_content = _detect_rectangular_valid_content_support_from_mean(mean_gray)
    if rectangular_content is not None:
        support_circle, wafer_mask = rectangular_content
        if _support_circle_has_visible_boundary(mean_gray, median_gray, support_circle):
            return support_circle, wafer_mask, mean_gray, std_gray, mad_gray
        return None, np.ones_like(mean_gray, dtype=bool), mean_gray, std_gray, mad_gray

    border = max(8, min(H, W) // 20)
    border_mask = np.zeros((H, W), dtype=bool)
    border_mask[:border, :] = True
    border_mask[H - border:, :] = True
    border_mask[:, :border] = True
    border_mask[:, W - border:] = True
    border_vals = mean_gray[border_mask]
    border_level = float(np.median(border_vals)) if border_vals.size else float(np.median(mean_gray))
    contrast_to_border = _robust_scale(np.abs(mean_gray - border_level))
    variability = _robust_scale(std_gray)
    support_score = 0.60 * contrast_to_border + 0.40 * variability
    support_smooth = cv2.GaussianBlur((support_score * 255).astype(np.uint8), (0, 0), 9).astype(np.float32) / 255.0
    thr = max(0.35, float(np.percentile(support_smooth, 82)))
    support_bin = (support_smooth > thr).astype(np.uint8)
    support_bin = cv2.morphologyEx(support_bin, cv2.MORPH_CLOSE, np.ones((31, 31), np.uint8))
    support_bin = cv2.morphologyEx(support_bin, cv2.MORPH_OPEN, np.ones((9, 9), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(support_bin, 8)
    if n <= 1:
        return None, np.ones_like(mean_gray, dtype=bool), mean_gray, std_gray, mad_gray
    areas = stats[1:, cv2.CC_STAT_AREA]
    largest = 1 + int(np.argmax(areas))
    comp = (labels == largest).astype(np.uint8)
    contours, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None, np.ones_like(mean_gray, dtype=bool), mean_gray, std_gray, mad_gray
    comp_mask = comp.astype(bool)
    partial_fit = _fit_partial_support_circle(comp_mask)
    yy, xx = np.indices(mean_gray.shape)
    if partial_fit is not None:
        xc, yc, r = partial_fit
        if r < 0.28 * min(H, W):
            return None, np.ones_like(mean_gray, dtype=bool), mean_gray, std_gray, mad_gray
        wafer_mask = (((xx - xc) ** 2 + (yy - yc) ** 2) <= (1.03 * r) ** 2) | comp_mask
        support_circle = (int(round(xc)), int(round(yc)), int(round(r)))
        if _support_circle_has_visible_boundary(mean_gray, median_gray, support_circle):
            return support_circle, wafer_mask, mean_gray, std_gray, mad_gray
        return None, np.ones_like(mean_gray, dtype=bool), mean_gray, std_gray, mad_gray
    cnt = max(contours, key=cv2.contourArea)
    (xc, yc), r = cv2.minEnclosingCircle(cnt)
    if r < 0.28 * min(H, W):
        return None, np.ones_like(mean_gray, dtype=bool), mean_gray, std_gray, mad_gray
    wafer_mask = ((xx - xc) ** 2 + (yy - yc) ** 2) <= (0.985 * r) ** 2
    support_circle = (int(round(xc)), int(round(yc)), int(round(r)))
    if _support_circle_has_visible_boundary(mean_gray, median_gray, support_circle):
        return support_circle, wafer_mask, mean_gray, std_gray, mad_gray
    return None, np.ones_like(mean_gray, dtype=bool), mean_gray, std_gray, mad_gray


def _evidence_sigma_values(shape: tuple[int, int], cfg: GeometryConfig | None) -> list[float]:
    if cfg is None:
        rmin = max(3.0, 0.008 * float(min(shape)))
        rmax = max(rmin + 1.0, 0.045 * float(min(shape)))
    else:
        rmin_i, rmax_i = _effective_initial_radius_bounds(shape, cfg)
        rmin, rmax = float(rmin_i), float(rmax_i)
    radii = np.linspace(max(3.0, rmin), max(rmin + 1.0, rmax), num=6)
    sigmas = sorted({float(np.clip(0.45 * r, 1.5, 18.0)) for r in radii})
    return sigmas or [3.5, 5.5, 7.5]


def _build_evidence_maps(mean_gray: np.ndarray, std_gray: np.ndarray, mad_gray: np.ndarray, frame0_gray: np.ndarray, wafer_mask: np.ndarray, cfg: GeometryConfig | None = None):
    mean_n = _robust_scale(mean_gray, wafer_mask)
    inv_std = _robust_scale(1.0 / (std_gray + 1e-3), wafer_mask)
    inv_mad = _robust_scale(1.0 / (mad_gray + 1e-3), wafer_mask)

    bright_blob_maps = []
    dark_blob_maps = []
    sigma_values = _evidence_sigma_values(mean_gray.shape, cfg)
    for sigma in sigma_values:
        log_resp = ndi.gaussian_laplace(mean_n, sigma=sigma)
        scale_norm = float(sigma) ** 2
        bright_blob_maps.append(np.clip(-log_resp * scale_norm, 0, None))
        dark_blob_maps.append(np.clip(log_resp * scale_norm, 0, None))
    bright_blob = _robust_scale(np.max(np.stack(bright_blob_maps), axis=0), wafer_mask)
    dark_blob = _robust_scale(np.max(np.stack(dark_blob_maps), axis=0), wafer_mask)

    ring_bright_maps = []
    ring_dark_maps = []
    for sigma in sigma_values:
        g1 = cv2.GaussianBlur(mean_n.astype(np.float32), (0, 0), max(1.0, 0.55 * float(sigma)))
        g2 = cv2.GaussianBlur(mean_n.astype(np.float32), (0, 0), max(2.0, 1.45 * float(sigma)))
        dog = g1 - g2
        ring_bright_maps.append(np.clip(dog, 0, None))
        ring_dark_maps.append(np.clip(-dog, 0, None))
    ring_bright = _robust_scale(np.max(np.stack(ring_bright_maps), axis=0), wafer_mask)
    ring_dark = _robust_scale(np.max(np.stack(ring_dark_maps), axis=0), wafer_mask)

    f0n = _robust_scale(frame0_gray.astype(np.float32), wafer_mask)
    frame_ring_bright_maps = []
    frame_ring_dark_maps = []
    for sigma in sigma_values:
        fg1 = cv2.GaussianBlur(f0n.astype(np.float32), (0, 0), max(1.0, 0.55 * float(sigma)))
        fg2 = cv2.GaussianBlur(f0n.astype(np.float32), (0, 0), max(2.0, 1.45 * float(sigma)))
        fdog = fg1 - fg2
        frame_ring_bright_maps.append(np.clip(fdog, 0, None))
        frame_ring_dark_maps.append(np.clip(-fdog, 0, None))
    frame_ring_bright = _robust_scale(np.max(np.stack(frame_ring_bright_maps), axis=0), wafer_mask)
    frame_ring_dark = _robust_scale(np.max(np.stack(frame_ring_dark_maps), axis=0), wafer_mask)

    texture_maps = []
    for sigma in sigma_values:
        tex_sigma = max(1.2, 0.55 * float(sigma))
        local_mean = cv2.GaussianBlur(mean_n.astype(np.float32), (0, 0), tex_sigma)
        local_sq = cv2.GaussianBlur((mean_n.astype(np.float32) ** 2), (0, 0), tex_sigma)
        local_std = np.sqrt(np.clip(local_sq - local_mean * local_mean, 0.0, None))
        texture_maps.append(local_std)
    spatial_texture = _robust_scale(np.max(np.stack(texture_maps), axis=0), wafer_mask)
    gradient_texture = _robust_scale(_gradient_magnitude(cv2.GaussianBlur(mean_n.astype(np.float32), (0, 0), 1.2)), wafer_mask)
    texture_void = _robust_scale((1.0 - spatial_texture) * (1.0 - 0.55 * gradient_texture), wafer_mask)

    if float(np.mean(wafer_mask)) > 0.985:
        edge_penalty = np.zeros_like(mean_gray, dtype=np.float32)
    else:
        yy, xx = np.indices(mean_gray.shape)
        ys, xs = np.where(wafer_mask)
        xc = 0.5 * (float(xs.min()) + float(xs.max()))
        yc = 0.5 * (float(ys.min()) + float(ys.max()))
        r = max(float(xs.max() - xs.min()), float(ys.max() - ys.min())) / 2.0
        dist_to_center = np.sqrt((xx - xc) ** 2 + (yy - yc) ** 2)
        edge_frac = np.clip((dist_to_center / max(0.985 * r, 1e-6) - 0.88) / 0.12, 0, 1)
        edge_penalty = edge_frac ** 2

    blob_score = np.maximum(bright_blob, dark_blob)
    ring_score = np.maximum.reduce([ring_bright, ring_dark, frame_ring_bright, frame_ring_dark])
    stability_score = np.maximum(inv_std, inv_mad)
    hole_score = (
        0.21 * inv_std
        + 0.17 * inv_mad
        + 0.18 * texture_void
        + 0.17 * blob_score
        + 0.13 * np.maximum(ring_bright, ring_dark)
        + 0.11 * np.maximum(frame_ring_bright, frame_ring_dark)
        + 0.03 * np.maximum(mean_n, 1.0 - mean_n)
    )
    hole_score = hole_score * wafer_mask * (1 - 0.25 * edge_penalty)
    hole_score = _robust_scale(hole_score, wafer_mask)
    smoothed = cv2.GaussianBlur((hole_score * 255).astype(np.uint8), (0, 0), 2.0).astype(np.float32) / 255.0
    return {
        'mean_gray': mean_gray.astype(np.float32),
        'frame0_gray': frame0_gray.astype(np.float32),
        'mean_scaled': mean_n,
        'frame0_scaled': f0n,
        'stability_std': inv_std,
        'stability_mad': inv_mad,
        'blob_score': blob_score.astype(np.float32),
        'ring_score': ring_score.astype(np.float32),
        'stability_score': stability_score.astype(np.float32),
        'texture_void_score': texture_void.astype(np.float32),
        'spatial_texture_score': spatial_texture.astype(np.float32),
        'hole_score': hole_score,
        'smoothed_score': smoothed,
        'evidence_sigma_values': np.asarray(sigma_values, dtype=np.float32),
    }


def _extract_seeds_from_score(smoothed_score: np.ndarray, wafer_mask: np.ndarray, cfg: GeometryConfig, min_score: float) -> list[tuple[float, float, float]]:
    peak_size = max(9, int(round(2.0 * float(cfg.min_radius_px) + 1.0)))
    if peak_size % 2 == 0:
        peak_size += 1
    mx = ndi.maximum_filter(smoothed_score, size=peak_size)
    cand = (smoothed_score >= mx - 1e-6) & wafer_mask & (smoothed_score > min_score)
    ys, xs = np.where(cand)
    scores = smoothed_score[ys, xs]
    return [(float(x), float(y), float(s)) for x, y, s in zip(xs, ys, scores)]


def _extract_seeds(smoothed_score: np.ndarray, wafer_mask: np.ndarray, cfg: GeometryConfig) -> np.ndarray:
    rows = _extract_seeds_from_score(smoothed_score, wafer_mask, cfg, 0.58)
    order = np.argsort([r[2] for r in rows])[::-1]
    seed_list: list[tuple[float, float, float]] = []
    min_dist = max(8.0, 1.8 * float(cfg.min_radius_px))
    for idx in order:
        x, y, s = rows[int(idx)]
        if all((x - sx) ** 2 + (y - sy) ** 2 >= min_dist ** 2 for sx, sy, _ in seed_list):
            seed_list.append((x, y, s))
    return np.asarray(seed_list, dtype=float)


def _extract_consensus_seeds(evidence: dict[str, np.ndarray], wafer_mask: np.ndarray, cfg: GeometryConfig) -> np.ndarray:
    if not np.any(wafer_mask):
        return np.empty((0, 3), dtype=float)
    main_seeds = _extract_seeds(evidence['smoothed_score'], wafer_mask, cfg)
    maps = [
        (evidence['smoothed_score'], 0.58, 1.00),
        (evidence['blob_score'], float(np.percentile(evidence['blob_score'][wafer_mask], 92)), 0.72),
        (evidence['ring_score'], float(np.percentile(evidence['ring_score'][wafer_mask], 92)), 0.72),
        (evidence['stability_score'], float(np.percentile(evidence['stability_score'][wafer_mask], 94)), 0.48),
    ]
    candidate_map: dict[tuple[int, int], dict[str, float]] = {}
    min_dist = max(8.0, 1.8 * float(cfg.min_radius_px))
    merge_r2 = (0.55 * min_dist) ** 2
    for score_map, threshold, weight in maps:
        for x, y, score in _extract_seeds_from_score(score_map, wafer_mask, cfg, threshold):
            key = None
            for existing in candidate_map:
                if (float(existing[0]) - x) ** 2 + (float(existing[1]) - y) ** 2 <= merge_r2:
                    key = existing
                    break
            weighted = float(weight * score)
            if key is None:
                candidate_map[(int(round(x)), int(round(y)))] = {
                    'x_sum': x * weighted,
                    'y_sum': y * weighted,
                    'w_sum': weighted,
                    'score': weighted,
                    'support': 1.0,
                }
            else:
                row = candidate_map[key]
                row['x_sum'] += x * weighted
                row['y_sum'] += y * weighted
                row['w_sum'] += weighted
                row['score'] += weighted
                row['support'] += 1.0
    rows: list[tuple[float, float, float]] = []
    for row in candidate_map.values():
        wsum = max(float(row['w_sum']), 1e-6)
        support = float(row['support'])
        score = float(row['score']) * (1.0 + 0.10 * min(support - 1.0, 3.0))
        rows.append((float(row['x_sum']) / wsum, float(row['y_sum']) / wsum, score))
    order = np.argsort([r[2] for r in rows])[::-1]
    seed_list: list[tuple[float, float, float]] = [(float(x), float(y), float(s)) for x, y, s in np.asarray(main_seeds, dtype=float)]
    extra_limit = max(24, int(round(0.30 * max(len(seed_list), 1))))
    max_count = len(seed_list) + extra_limit
    for idx in order:
        if len(seed_list) >= max_count:
            break
        x, y, s = rows[int(idx)]
        if all((x - sx) ** 2 + (y - sy) ** 2 >= min_dist ** 2 for sx, sy, _ in seed_list):
            seed_list.append((x, y, s))
    return np.asarray(seed_list, dtype=float)


def _fit_local_circle(x0: float, y0: float, mean_gray: np.ndarray, stability_std: np.ndarray, rmin: int, rmax: int, search: int = 4):
    H, W = mean_gray.shape
    x0i, y0i = int(round(x0)), int(round(y0))
    xmin, xmax = max(0, x0i - 28), min(W, x0i + 29)
    ymin, ymax = max(0, y0i - 28), min(H, y0i + 29)
    patch_g = mean_gray[ymin:ymax, xmin:xmax]
    patch_s = stability_std[ymin:ymax, xmin:xmax]
    py, px = np.indices(patch_g.shape)
    absx = px + xmin
    absy = py + ymin
    best = None
    for dy in range(-search, search + 1):
        for dx in range(-search, search + 1):
            cx = x0 + dx
            cy = y0 + dy
            rr = np.sqrt((absx - cx) ** 2 + (absy - cy) ** 2)
            for r0 in range(rmin, rmax + 1):
                core = rr <= r0
                ring = (rr >= r0 + 2) & (rr < r0 + 7)
                if int(core.sum()) < 20 or int(ring.sum()) < 20:
                    continue
                core_mean = float(patch_g[core].mean())
                ring_mean = float(patch_g[ring].mean())
                contrast = abs(core_mean - ring_mean)
                polarity_bonus = max(core_mean - ring_mean, 0.0) * 0.25
                stab = float(patch_s[core].mean() - 0.3 * patch_s[ring].mean())
                score = contrast + 35.0 * stab + polarity_bonus
                if best is None or score > best[0]:
                    best = (float(score), float(cx), float(cy), int(r0), core_mean, ring_mean, stab)
    return best


def _effective_initial_radius_bounds(shape: tuple[int, int], cfg: GeometryConfig) -> tuple[int, int]:
    h, w = shape
    rmin = max(3, int(math.floor(float(cfg.min_radius_px))))
    configured_rmax = max(rmin + 1, int(math.ceil(float(cfg.max_radius_px))))
    frame_scaled_cap = max(rmin + 1, int(round(0.050 * float(min(h, w)))))
    search_cap = max(rmin + 1, min(32, frame_scaled_cap))
    return rmin, min(configured_rmax, search_cap)


@njit(cache=True, parallel=True)
def _fit_initial_circles_kernel(
    seeds: np.ndarray,
    mean_gray: np.ndarray,
    stability_std: np.ndarray,
    rmin: int,
    rmax: int,
    search: int,
    patch_radius: int,
) -> np.ndarray:
    h, w = mean_gray.shape
    out = np.zeros((seeds.shape[0], 8), dtype=np.float32)
    for si in prange(seeds.shape[0]):
        x0 = float(seeds[si, 0])
        y0 = float(seeds[si, 1])
        x0i = int(round(x0))
        y0i = int(round(y0))
        xmin = max(0, x0i - patch_radius)
        xmax = min(w, x0i + patch_radius + 1)
        ymin = max(0, y0i - patch_radius)
        ymax = min(h, y0i + patch_radius + 1)
        best_score = -1.0e30
        best_x = 0.0
        best_y = 0.0
        best_r = 0.0
        best_core = 0.0
        best_ring = 0.0
        best_stab = 0.0

        for dy in range(-search, search + 1):
            cy = y0 + float(dy)
            for dx in range(-search, search + 1):
                cx = x0 + float(dx)
                for r0 in range(rmin, rmax + 1):
                    r_core2 = float(r0 * r0)
                    ring_inner = float((r0 + 2) * (r0 + 2))
                    ring_outer = float((r0 + 7) * (r0 + 7))
                    core_sum = 0.0
                    ring_sum = 0.0
                    core_stab_sum = 0.0
                    ring_stab_sum = 0.0
                    core_n = 0
                    ring_n = 0

                    for py in range(ymin, ymax):
                        yy = float(py) - cy
                        yy2 = yy * yy
                        for px in range(xmin, xmax):
                            xx = float(px) - cx
                            d2 = xx * xx + yy2
                            if d2 <= r_core2:
                                core_sum += float(mean_gray[py, px])
                                core_stab_sum += float(stability_std[py, px])
                                core_n += 1
                            elif d2 >= ring_inner and d2 < ring_outer:
                                ring_sum += float(mean_gray[py, px])
                                ring_stab_sum += float(stability_std[py, px])
                                ring_n += 1

                    if core_n < 20 or ring_n < 20:
                        continue
                    core_mean = core_sum / float(core_n)
                    ring_mean = ring_sum / float(ring_n)
                    gap = core_mean - ring_mean
                    contrast = abs(gap)
                    polarity_bonus = max(gap, 0.0) * 0.25
                    stab = core_stab_sum / float(core_n) - 0.3 * ring_stab_sum / float(ring_n)
                    score = contrast + 35.0 * stab + polarity_bonus
                    if score > best_score:
                        best_score = score
                        best_x = cx
                        best_y = cy
                        best_r = float(r0)
                        best_core = core_mean
                        best_ring = ring_mean
                        best_stab = stab

        if best_score > -1.0e20:
            out[si, 0] = 1.0
            out[si, 1] = best_score
            out[si, 2] = best_x
            out[si, 3] = best_y
            out[si, 4] = best_r
            out[si, 5] = best_core
            out[si, 6] = best_ring
            out[si, 7] = best_stab
    return out


def _initial_fitted_candidates(evidence: dict[str, np.ndarray], seeds: np.ndarray, cfg: GeometryConfig) -> tuple[list[dict[str, float | str]], list[dict[str, float | str]]]:
    fitted: list[dict[str, float | str]] = []
    mean_gray = evidence['mean_gray']
    stability_std = evidence['stability_std']
    if seeds.shape[0] > 1600:
        order = np.argsort(seeds[:, 2])[::-1][:1600]
        seeds = np.asarray(seeds[order], dtype=np.float32)
    rmin, rmax = _effective_initial_radius_bounds(mean_gray.shape, cfg)
    search = max(3, min(6, int(round(0.25 * rmax + 1.0))))
    patch_radius = max(16, rmax + search + 8)
    fit_rows = _fit_initial_circles_kernel(
        np.asarray(seeds, dtype=np.float32),
        np.asarray(mean_gray, dtype=np.float32),
        np.asarray(stability_std, dtype=np.float32),
        int(rmin),
        int(rmax),
        int(search),
        int(patch_radius),
    )
    for seed, fit in zip(seeds, fit_rows):
        x, y, s = float(seed[0]), float(seed[1]), float(seed[2])
        if float(fit[0]) <= 0.0:
            continue
        score, cx2, cy2, r2, core_mean, ring_mean, stab = [float(v) for v in fit[1:]]
        conf = float(0.55 * float(s) + 0.45 * min(float(score) / 120.0, 1.0))
        status = 'accepted' if (conf >= 0.50 and rmin <= int(r2) <= rmax) else 'provisional'
        fitted.append({
            'seed_x': float(x), 'seed_y': float(y), 'seed_score': float(s),
            'x': float(cx2), 'y': float(cy2), 'r': float(r2),
            'fit_score': float(score), 'confidence': conf,
            'core_mean': float(core_mean), 'ring_mean': float(ring_mean),
            'stability_center': float(stab), 'status': status,
        })
    accepted = [d for d in fitted if str(d['status']) == 'accepted']
    provisional = [d for d in fitted if str(d['status']) != 'accepted']
    return _deduplicate_fitted_rows(accepted), provisional


def _local_circle_contrast_score(score_image: np.ndarray, texture_void: np.ndarray, wafer_mask: np.ndarray, x0: float, y0: float, radius: float) -> tuple[float, float, float] | None:
    h, w = score_image.shape
    r = max(1.0, float(radius))
    pad = max(4, int(math.ceil(1.75 * r)))
    xi = int(round(float(x0)))
    yi = int(round(float(y0)))
    x_start = max(0, xi - pad)
    x_stop = min(w, xi + pad + 1)
    y_start = max(0, yi - pad)
    y_stop = min(h, yi + pad + 1)
    if x_start >= x_stop or y_start >= y_stop:
        return None
    crop = score_image[y_start:y_stop, x_start:x_stop]
    local_mask = wafer_mask[y_start:y_stop, x_start:x_stop]
    yy, xx = np.indices(crop.shape)
    d = np.sqrt((xx + x_start - float(x0)) ** 2 + (yy + y_start - float(y0)) ** 2)
    core = (d <= 0.82 * r) & local_mask
    ring = (d >= 1.12 * r) & (d <= 1.58 * r) & local_mask
    inner_edge = (d >= 0.82 * r) & (d <= r) & local_mask
    outer_edge = (d >= r) & (d <= 1.20 * r) & local_mask
    if int(core.sum()) < max(10, int(round(0.35 * math.pi * r * r))):
        return None
    if int(ring.sum()) < max(10, int(round(0.25 * math.pi * r * r))):
        return None
    if int(inner_edge.sum()) < 8 or int(outer_edge.sum()) < 8:
        return None
    core_level = float(np.mean(crop[core]))
    ring_level = float(np.mean(crop[ring]))
    edge_gap = float(abs(float(np.mean(crop[inner_edge])) - float(np.mean(crop[outer_edge]))))
    tex_crop = texture_void[y_start:y_stop, x_start:x_stop]
    core_texture_void = float(np.mean(tex_crop[core]))
    ring_texture_void = float(np.mean(tex_crop[ring]))
    dark_contrast = max(0.0, ring_level - core_level)
    texture_gap = max(0.0, core_texture_void - ring_texture_void)
    if dark_contrast <= 0.0 and texture_gap <= 0.0:
        return None
    return dark_contrast + 0.35 * edge_gap + 0.75 * texture_gap + 0.20 * core_texture_void, core_level, ring_level


def _hough_round_rim_candidates(evidence: dict[str, np.ndarray], wafer_mask: np.ndarray, cfg: GeometryConfig) -> list[dict[str, float | str]]:
    mean_gray = evidence['mean_gray']
    if mean_gray.size == 0 or not np.any(wafer_mask):
        return []
    rmin, rmax = _effective_initial_radius_bounds(mean_gray.shape, cfg)
    min_hough_r = max(int(rmin), int(round(0.45 * float(rmax))))
    max_hough_r = max(min_hough_r + 2, int(round(max(float(rmax) + 8.0, 0.055 * float(min(mean_gray.shape))))))
    if max_hough_r <= min_hough_r:
        return []
    score_image = _robust_scale(mean_gray, wafer_mask)
    texture_void = evidence.get('texture_void_score')
    if texture_void is None:
        texture_void = np.zeros_like(score_image, dtype=np.float32)
    texture_void = np.asarray(texture_void, dtype=np.float32)
    u8 = np.clip(score_image * 255.0, 0, 255).astype(np.uint8)
    min_dist = max(12, int(round(1.55 * float(min_hough_r + max_hough_r))))
    best_rows: list[dict[str, float | str]] = []
    for blur_size in (9, 7, 5):
        blurred = cv2.medianBlur(u8, int(blur_size))
        for accumulator_threshold in (44, 38, 32, 26):
            circles = cv2.HoughCircles(
                blurred,
                cv2.HOUGH_GRADIENT,
                dp=1.2,
                minDist=float(min_dist),
                param1=80.0,
                param2=float(accumulator_threshold),
                minRadius=int(min_hough_r),
                maxRadius=int(max_hough_r),
            )
            if circles is None:
                continue
            rows: list[dict[str, float | str]] = []
            for x, y, r in np.asarray(circles[0], dtype=np.float32):
                xi = int(round(float(x)))
                yi = int(round(float(y)))
                if yi < 0 or xi < 0 or yi >= wafer_mask.shape[0] or xi >= wafer_mask.shape[1] or not wafer_mask[yi, xi]:
                    continue
                score = _local_circle_contrast_score(score_image, texture_void, wafer_mask, float(x), float(y), float(r))
                if score is None:
                    continue
                contrast, core_level, ring_level = score
                if contrast < 0.080:
                    continue
                confidence = float(np.clip(0.35 + 2.2 * contrast, 0.0, 1.0))
                rows.append({
                    'seed_x': float(x),
                    'seed_y': float(y),
                    'seed_score': float(contrast),
                    'x': float(x),
                    'y': float(y),
                    'r': float(r),
                    'fit_score': float(contrast),
                    'confidence': confidence,
                    'core_mean': float(core_level),
                    'ring_mean': float(ring_level),
                    'stability_center': 0.0,
                    'status': 'accepted',
                })
            rows = _select_common_radius_rows(_deduplicate_fitted_rows(rows))
            if len(rows) > len(best_rows):
                best_rows = rows
    return best_rows


def _connected_texture_void_candidates(evidence: dict[str, np.ndarray], wafer_mask: np.ndarray, cfg: GeometryConfig) -> list[dict[str, float | str]]:
    if not np.any(wafer_mask):
        return []
    mean_n = np.asarray(evidence.get('mean_scaled', evidence['mean_gray']), dtype=np.float32)
    texture_void = np.asarray(evidence.get('texture_void_score', np.zeros_like(mean_n)), dtype=np.float32)
    void_score = _robust_scale((1.0 - mean_n) * texture_void, wafer_mask)
    rmin, rmax = _effective_initial_radius_bounds(mean_n.shape, cfg)
    min_area = 0.20 * math.pi * float(rmin) * float(rmin)
    max_area = 2.50 * math.pi * float(rmax + 10) * float(rmax + 10)
    best_rows: list[dict[str, float | str]] = []
    best_quality = -float('inf')
    for percentile in (80.0, 84.0, 88.0, 90.0, 92.0, 94.0, 96.0):
        vals = void_score[wafer_mask]
        if vals.size == 0:
            continue
        threshold = float(np.percentile(vals, percentile))
        binary = ((void_score >= threshold) & wafer_mask).astype(np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
        n_labels, labels, stats, _centroids = cv2.connectedComponentsWithStats(binary, 8)
        rows: list[dict[str, float | str]] = []
        for lab in range(1, n_labels):
            area = float(stats[lab, cv2.CC_STAT_AREA])
            if area < min_area or area > max_area:
                continue
            comp = (labels == lab).astype(np.uint8)
            contours, _hier = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            if not contours:
                continue
            contour = max(contours, key=cv2.contourArea)
            contour_area = float(cv2.contourArea(contour))
            if contour_area <= 0.0:
                continue
            (x, y), enclosing_r = cv2.minEnclosingCircle(contour)
            eq_r = math.sqrt(max(contour_area, 1.0) / math.pi)
            radius = 0.45 * float(enclosing_r) + 0.55 * float(eq_r)
            if radius < 0.45 * float(rmax) or radius > 1.60 * float(rmax):
                continue
            circularity = contour_area / (math.pi * max(float(enclosing_r), 1.0) ** 2)
            if circularity < 0.35:
                continue
            local_score = _local_circle_contrast_score(mean_n, texture_void, wafer_mask, float(x), float(y), float(radius))
            if local_score is None:
                continue
            score, core_level, ring_level = local_score
            if score < 0.05:
                continue
            confidence = float(np.clip(0.40 + 2.0 * score, 0.0, 1.0))
            rows.append({
                'seed_x': float(x),
                'seed_y': float(y),
                'seed_score': float(score),
                'x': float(x),
                'y': float(y),
                'r': float(radius),
                'fit_score': float(score),
                'confidence': confidence,
                'core_mean': float(core_level),
                'ring_mean': float(ring_level),
                'stability_center': 0.0,
                'status': 'accepted',
            })
        rows = _select_common_radius_rows(_deduplicate_fitted_rows(rows))
        if not rows:
            continue
        residual = _candidate_basis_residual(rows)
        quality = float(len(rows)) - 12.0 * min(float(residual), 1.0)
        if residual < 0.12:
            quality += 4.0
        if len(rows) >= 8 and quality > best_quality:
            best_rows = rows
            best_quality = quality
    return best_rows


def _candidate_basis_residual(rows: list[dict[str, float | str]]) -> float:
    basis = _estimate_probe_basis(rows)
    if basis is None:
        return float('inf')
    _pts, u, v, origin = basis
    return _basis_residual_for_rows(rows, u, v, origin)


def _deduplicate_fitted_rows(rows: list[dict[str, float | str]]) -> list[dict[str, float | str]]:
    if not rows:
        return []
    radii = np.asarray([float(r['r']) for r in rows], dtype=np.float32)
    min_sep = max(8.0, 1.8 * float(np.median(radii)))
    ordered = sorted(rows, key=lambda r: (float(r['confidence']), float(r['fit_score'])), reverse=True)
    kept: list[dict[str, float | str]] = []
    for row in ordered:
        x = float(row['x'])
        y = float(row['y'])
        if any((x - float(k['x'])) ** 2 + (y - float(k['y'])) ** 2 < min_sep ** 2 for k in kept):
            continue
        kept.append(row)
    return kept


def _otsu_split_1d(values: np.ndarray) -> float | None:
    if values.size < 2:
        return None
    lo = int(math.floor(float(np.min(values))))
    hi = int(math.ceil(float(np.max(values))))
    if hi <= lo:
        return None
    edges = np.arange(lo, hi + 2, dtype=np.float32)
    hist, _ = np.histogram(values.astype(np.float32), bins=edges)
    centers = 0.5 * (edges[:-1] + edges[1:])
    total = int(hist.sum())
    if total <= 1:
        return None
    best_idx = -1
    best_score = -1.0
    for idx in range(1, len(hist)):
        w0 = float(hist[:idx].sum())
        w1 = float(hist[idx:].sum())
        if w0 <= 0.0 or w1 <= 0.0:
            continue
        m0 = float(np.dot(hist[:idx], centers[:idx]) / w0)
        m1 = float(np.dot(hist[idx:], centers[idx:]) / w1)
        score = w0 * w1 * (m0 - m1) * (m0 - m1)
        if score > best_score:
            best_score = score
            best_idx = idx
    if best_idx < 0:
        return None
    return float(edges[best_idx])


def _select_common_radius_rows(rows: list[dict[str, float | str]]) -> list[dict[str, float | str]]:
    if len(rows) < 8:
        return rows
    radii = np.asarray([float(r['r']) for r in rows], dtype=np.float32)
    split = _otsu_split_1d(radii)
    if split is None:
        return rows
    upper = [r for r in rows if float(r['r']) >= split]
    if len(upper) >= 8:
        return upper
    return rows


def _row_median_radius(rows: list[dict[str, float | str]]) -> float:
    if not rows:
        return 0.0
    radii = np.asarray([float(r.get('r', 0.0)) for r in rows], dtype=np.float32)
    radii = radii[np.isfinite(radii) & (radii > 0)]
    return float(np.median(radii)) if radii.size else 0.0


def _basis_min_spacing_px(basis_u: np.ndarray, basis_v: np.ndarray) -> float:
    return float(min(float(np.linalg.norm(basis_u)), float(np.linalg.norm(basis_v))))


def _basis_respects_hole_nonoverlap(basis_u: np.ndarray, basis_v: np.ndarray, hole_radius_px: float) -> bool:
    radius = float(hole_radius_px)
    spacing = _basis_min_spacing_px(basis_u, basis_v)
    if radius <= 0.0 or spacing <= 0.0:
        return False
    # First-principles guard: neighbouring circular holes cannot overlap.
    return bool(spacing > (2.0 * radius + 1.0))


def _max_nonoverlapping_holes_in_mask(mask: np.ndarray, hole_radius_px: float) -> int:
    radius = float(hole_radius_px)
    if radius <= 0.0:
        return 0
    support_area = int(np.count_nonzero(mask))
    hole_area = math.pi * radius * radius
    if support_area <= 0 or hole_area <= 0.0:
        return 0
    return max(1, int(math.floor(float(support_area) / hole_area)))


def _estimate_probe_basis(accepted_rows: list[dict[str, float | str]]):
    if len(accepted_rows) < 8:
        return None
    candidates = [
        HoleCandidate(float(row['x']), float(row['y']), float(row['r']), 0.0, float(row.get('fit_score', 0.0)), float(row.get('confidence', 0.0)))
        for row in accepted_rows
    ]
    lattice = estimate_lattice_basis(candidates)
    if lattice.spacing_u_px <= 0.0 or lattice.spacing_v_px <= 0.0:
        return None
    basis_u = np.asarray(lattice.basis_u, dtype=np.float32)
    basis_v = np.asarray(lattice.basis_v, dtype=np.float32)
    if abs(float(basis_u[0] * basis_v[1] - basis_u[1] * basis_v[0])) < 0.15 * max(float(np.linalg.norm(basis_u) * np.linalg.norm(basis_v)), 1e-6):
        return None
    if not _basis_respects_hole_nonoverlap(basis_u, basis_v, _row_median_radius(accepted_rows)):
        return None
    xy = np.array([(float(d['x']), float(d['y'])) for d in accepted_rows], dtype=np.float32)
    origin = np.array([lattice.origin_x, lattice.origin_y], dtype=np.float32)
    return xy, basis_u, basis_v, origin


def _basis_origin_and_residual(pts: np.ndarray, basis_u: np.ndarray, basis_v: np.ndarray) -> tuple[np.ndarray, float]:
    B = np.column_stack([basis_u, basis_v]).astype(np.float32)
    inv = np.linalg.pinv(B)
    best_origin = pts[0].astype(np.float32)
    best_residual = float('inf')
    for origin in pts:
        uv = (pts - origin) @ inv.T
        residual = float(np.median(np.linalg.norm(uv - np.round(uv), axis=1)))
        if residual < best_residual:
            best_residual = residual
            best_origin = origin.astype(np.float32)
    return best_origin, best_residual


@njit(cache=True, parallel=True)
def _phase_origin_score_kernel(
    score_map: np.ndarray,
    mask: np.ndarray,
    basis_u: np.ndarray,
    basis_v: np.ndarray,
    origin_seed: np.ndarray,
    offsets: np.ndarray,
    umin: int,
    umax: int,
    vmin: int,
    vmax: int,
) -> np.ndarray:
    h, w = score_map.shape
    out = np.zeros(offsets.shape[0], dtype=np.float32)
    for oi in prange(offsets.shape[0]):
        du = float(offsets[oi, 0])
        dv = float(offsets[oi, 1])
        ox = float(origin_seed[0]) + du * float(basis_u[0]) + dv * float(basis_v[0])
        oy = float(origin_seed[1]) + du * float(basis_u[1]) + dv * float(basis_v[1])
        total = 0.0
        count = 0
        for iu in range(umin, umax + 1):
            bux = float(iu) * float(basis_u[0])
            buy = float(iu) * float(basis_u[1])
            for iv in range(vmin, vmax + 1):
                x = ox + bux + float(iv) * float(basis_v[0])
                y = oy + buy + float(iv) * float(basis_v[1])
                if x < 0.0 or y < 0.0 or x >= float(w - 1) or y >= float(h - 1):
                    continue
                xi = int(round(x))
                yi = int(round(y))
                if xi < 0 or yi < 0 or xi >= w or yi >= h or not mask[yi, xi]:
                    continue
                x0 = int(math.floor(x))
                y0 = int(math.floor(y))
                dx = x - float(x0)
                dy = y - float(y0)
                val = (
                    (1.0 - dx) * (1.0 - dy) * float(score_map[y0, x0])
                    + dx * (1.0 - dy) * float(score_map[y0, x0 + 1])
                    + (1.0 - dx) * dy * float(score_map[y0 + 1, x0])
                    + dx * dy * float(score_map[y0 + 1, x0 + 1])
                )
                total += val
                count += 1
        if count > 0:
            out[oi] = float(total / float(count) + 0.0025 * math.sqrt(float(count)))
    return out


_EXACT_SEQUENCE_NUMBA_WARMED = False


def warmup_exact_sequence_numba() -> None:
    global _EXACT_SEQUENCE_NUMBA_WARMED
    if _EXACT_SEQUENCE_NUMBA_WARMED:
        return
    seeds = np.asarray([[16.0, 16.0, 1.0]], dtype=np.float32)
    image = np.ones((40, 40), dtype=np.float32)
    _fit_initial_circles_kernel(seeds, image, image, 3, 4, 1, 8)
    score_map = np.ones((24, 24), dtype=np.float32)
    mask = np.ones((24, 24), dtype=np.bool_)
    basis_u = np.asarray([6.0, 0.0], dtype=np.float32)
    basis_v = np.asarray([3.0, 5.0], dtype=np.float32)
    origin = np.asarray([8.0, 8.0], dtype=np.float32)
    offsets = np.asarray([[0.0, 0.0], [0.1, -0.1]], dtype=np.float32)
    _phase_origin_score_kernel(score_map, mask, basis_u, basis_v, origin, offsets, -1, 1, -1, 1)
    templates = _build_common_radius_templates(3)
    R = int(templates['R'])
    node_x = np.asarray([12.0, 18.0], dtype=np.float32)
    node_y = np.asarray([12.0, 18.0], dtype=np.float32)
    present_flags = np.asarray([1, 0], dtype=np.uint8)
    proto = np.ones((2 * R + 1, 2 * R + 1), dtype=np.float32)
    _refine_common_radius_kernel(
        node_x,
        node_y,
        present_flags,
        score_map,
        score_map,
        score_map,
        score_map,
        mask,
        _template_offsets(templates['core'], R),
        _template_offsets(templates['ring'], R),
        _template_offsets(templates['inner_edge'], R),
        _template_offsets(templates['outer_edge'], R),
        _square_template_offsets(R),
        proto,
        proto,
        proto,
        proto,
        1,
        1,
    )
    _EXACT_SEQUENCE_NUMBA_WARMED = True


def _phase_score_map(evidence: dict[str, np.ndarray], wafer_mask: np.ndarray) -> np.ndarray:
    score = (
        0.45 * evidence['smoothed_score'].astype(np.float32)
        + 0.25 * evidence['ring_score'].astype(np.float32)
        + 0.20 * evidence['blob_score'].astype(np.float32)
        + 0.10 * evidence['stability_score'].astype(np.float32)
    )
    return _robust_scale(score, wafer_mask)


def _refine_lattice_origin_by_phase(evidence: dict[str, np.ndarray], wafer_mask: np.ndarray, basis_u: np.ndarray, basis_v: np.ndarray, origin_seed: np.ndarray) -> np.ndarray:
    if not np.any(wafer_mask):
        return origin_seed.astype(np.float32)
    score_map = _phase_score_map(evidence, wafer_mask)
    h, w = score_map.shape
    B = np.column_stack([basis_u, basis_v]).astype(np.float32)
    inv = np.linalg.pinv(B)
    corners = np.asarray([[0.0, 0.0], [float(w - 1), 0.0], [0.0, float(h - 1)], [float(w - 1), float(h - 1)]], dtype=np.float32)
    uv = (corners - origin_seed.astype(np.float32)) @ inv.T
    umin = int(math.floor(float(uv[:, 0].min()))) - 2
    umax = int(math.ceil(float(uv[:, 0].max()))) + 2
    vmin = int(math.floor(float(uv[:, 1].min()))) - 2
    vmax = int(math.ceil(float(uv[:, 1].max()))) + 2

    best_origin = origin_seed.astype(np.float32)
    for step_count in (13, 9):
        half = 0.5 if step_count == 13 else 1.0 / 12.0
        vals = np.linspace(-half, half, step_count, dtype=np.float32)
        offsets = np.asarray([(float(a), float(b)) for a in vals for b in vals], dtype=np.float32)
        scores = _phase_origin_score_kernel(
            np.asarray(score_map, dtype=np.float32),
            np.asarray(wafer_mask, dtype=np.bool_),
            np.asarray(basis_u, dtype=np.float32),
            np.asarray(basis_v, dtype=np.float32),
            np.asarray(best_origin, dtype=np.float32),
            offsets,
            int(umin),
            int(umax),
            int(vmin),
            int(vmax),
        )
        if scores.size == 0:
            break
        best_idx = int(np.argmax(scores))
        du = float(offsets[best_idx, 0])
        dv = float(offsets[best_idx, 1])
        best_origin = (best_origin + du * basis_u + dv * basis_v).astype(np.float32)
    return best_origin


def _autocorrelation_lattice_basis(evidence: dict[str, np.ndarray], wafer_mask: np.ndarray, accepted_rows: list[dict[str, float | str]]):
    if len(accepted_rows) < 8 or not np.any(wafer_mask):
        return None
    pts = np.asarray([[float(row['x']), float(row['y'])] for row in accepted_rows], dtype=np.float32)
    diffs = pts[None, :, :] - pts[:, None, :]
    dist_pts = np.linalg.norm(diffs, axis=2)
    dist_pts[dist_pts == 0] = np.inf
    spacing = float(np.median(np.min(dist_pts, axis=1)))
    if not np.isfinite(spacing) or spacing <= 1.0:
        return None
    signal = evidence.get('hole_score', evidence['smoothed_score']).astype(np.float32) * wafer_mask.astype(np.float32)
    vals = signal[wafer_mask]
    if vals.size < 16:
        return None
    threshold = float(np.percentile(vals, 65))
    signal = np.clip(signal - threshold, 0.0, None)
    signal = signal - float(np.mean(signal[wafer_mask]))
    h, w = signal.shape
    signal = signal * np.outer(np.hanning(h), np.hanning(w)).astype(np.float32)
    corr = np.fft.fftshift(np.fft.ifft2(np.abs(np.fft.fft2(signal)) ** 2).real).astype(np.float32)
    cy, cx = h // 2, w // 2
    corr[max(0, cy - 4):min(h, cy + 5), max(0, cx - 4):min(w, cx + 5)] = 0.0
    corr_min = float(np.min(corr))
    corr_max = float(np.max(corr))
    if corr_max <= corr_min:
        return None
    corr = (corr - corr_min) / (corr_max - corr_min)
    yy, xx = np.indices((h, w))
    dx = xx - cx
    dy = yy - cy
    dist = np.sqrt(dx * dx + dy * dy)
    annulus = (dist > 0.45 * spacing) & (dist < 1.55 * spacing)
    ys, xs = np.where(annulus)
    if ys.size == 0:
        return None
    peak_values = corr[ys, xs]
    order = np.argsort(peak_values)[::-1]
    peaks: list[tuple[float, float, float, float]] = []
    min_sep2 = (0.35 * spacing) ** 2
    for idx in order:
        x = float(xs[int(idx)] - cx)
        y = float(ys[int(idx)] - cy)
        value = float(peak_values[int(idx)])
        if value < 0.05:
            break
        if all((x - px) ** 2 + (y - py) ** 2 >= min_sep2 for _pv, px, py, _pd in peaks):
            peaks.append((value, x, y, float(math.hypot(x, y))))
        if len(peaks) >= 24:
            break
    if len(peaks) < 2:
        return None
    best = None
    for i, peak_a in enumerate(peaks):
        for peak_b in peaks[i + 1:]:
            u = np.asarray([peak_a[1], peak_a[2]], dtype=np.float32)
            v = np.asarray([peak_b[1], peak_b[2]], dtype=np.float32)
            lu = float(np.linalg.norm(u))
            lv = float(np.linalg.norm(v))
            if lu <= 1e-6 or lv <= 1e-6:
                continue
            det = float(u[0] * v[1] - u[1] * v[0])
            if abs(det) < 0.45 * lu * lv:
                continue
            ratio = lu / lv
            if ratio < 0.65 or ratio > 1.55:
                continue
            if det < 0:
                v = -v
            origin, residual = _basis_origin_and_residual(pts, u, v)
            score = float((peak_a[0] + peak_b[0]) / (1.0 + residual))
            if best is None or score > best[0]:
                best = (score, u, v, origin, residual)
    if best is None:
        return None
    score, basis_u, basis_v, origin, residual = best
    if not _basis_respects_hole_nonoverlap(basis_u, basis_v, _row_median_radius(accepted_rows)):
        return None
    origin = _refine_lattice_origin_by_phase(evidence, wafer_mask, basis_u, basis_v, origin)
    B = np.column_stack([basis_u, basis_v]).astype(np.float32)
    inv = np.linalg.pinv(B)
    uv = (pts - origin) @ inv.T
    residual = float(np.median(np.linalg.norm(uv - np.round(uv), axis=1)))
    confidence = float(np.clip(score / 2.0, 0.0, 1.0) * (1.0 / (1.0 + residual)))
    return pts, basis_u.astype(np.float32), basis_v.astype(np.float32), origin.astype(np.float32), confidence


def _periodicity_signal(evidence: dict[str, np.ndarray], wafer_mask: np.ndarray) -> np.ndarray:
    mean_scaled = evidence['mean_scaled'].astype(np.float32)
    grad = _robust_scale(_gradient_magnitude(mean_scaled), wafer_mask)
    signal = (
        0.34 * evidence['ring_score'].astype(np.float32)
        + 0.30 * evidence['blob_score'].astype(np.float32)
        + 0.22 * evidence['smoothed_score'].astype(np.float32)
        + 0.14 * grad.astype(np.float32)
    )
    signal = _robust_scale(signal, wafer_mask)
    if np.any(wafer_mask):
        vals = signal[wafer_mask]
        signal = np.clip(signal - float(np.median(vals)), 0.0, None)
    return (signal * wafer_mask.astype(np.float32)).astype(np.float32)


def _corr_value_at_vector(corr: np.ndarray, vx: float, vy: float) -> float:
    cy, cx = corr.shape[0] // 2, corr.shape[1] // 2
    x = int(round(float(cx) + float(vx)))
    y = int(round(float(cy) + float(vy)))
    if x < 0 or y < 0 or x >= corr.shape[1] or y >= corr.shape[0]:
        return 0.0
    return float(corr[y, x])


def _basis_phase_mean_score(evidence: dict[str, np.ndarray], wafer_mask: np.ndarray, basis_u: np.ndarray, basis_v: np.ndarray, origin: np.ndarray) -> float:
    if not np.any(wafer_mask):
        return 0.0
    score_map = _phase_score_map(evidence, wafer_mask)
    h, w = score_map.shape
    B = np.column_stack([basis_u, basis_v]).astype(np.float32)
    inv = np.linalg.pinv(B)
    corners = np.asarray([[0.0, 0.0], [float(w - 1), 0.0], [0.0, float(h - 1)], [float(w - 1), float(h - 1)]], dtype=np.float32)
    uv = (corners - origin.astype(np.float32)) @ inv.T
    offsets = np.asarray([[0.0, 0.0]], dtype=np.float32)
    scores = _phase_origin_score_kernel(
        np.asarray(score_map, dtype=np.float32),
        np.asarray(wafer_mask, dtype=np.bool_),
        np.asarray(basis_u, dtype=np.float32),
        np.asarray(basis_v, dtype=np.float32),
        np.asarray(origin, dtype=np.float32),
        offsets,
        int(math.floor(float(uv[:, 0].min()))) - 2,
        int(math.ceil(float(uv[:, 0].max()))) + 2,
        int(math.floor(float(uv[:, 1].min()))) - 2,
        int(math.ceil(float(uv[:, 1].max()))) + 2,
    )
    return float(scores[0]) if scores.size else 0.0


def _global_autocorrelation_lattice_basis(evidence: dict[str, np.ndarray], wafer_mask: np.ndarray, accepted_rows: list[dict[str, float | str]], hole_radius_px: float):
    if not np.any(wafer_mask) or float(hole_radius_px) <= 0.0:
        return None
    signal_full = _periodicity_signal(evidence, wafer_mask)
    h0, w0 = signal_full.shape
    max_dim = max(h0, w0)
    target_dim = 760.0
    scale = float(max_dim) / target_dim if max_dim > target_dim else 1.0
    if scale > 1.0:
        small_shape = (max(32, int(round(float(h0) / scale))), max(32, int(round(float(w0) / scale))))
        signal = cv2.resize(signal_full, (small_shape[1], small_shape[0]), interpolation=cv2.INTER_AREA).astype(np.float32)
        mask = cv2.resize(wafer_mask.astype(np.uint8), (small_shape[1], small_shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)
    else:
        signal = signal_full.astype(np.float32)
        mask = wafer_mask.astype(bool)
    if int(np.count_nonzero(mask)) < 64:
        return None
    vals = signal[mask]
    if vals.size < 64 or float(np.max(vals)) <= float(np.min(vals)):
        return None
    signal = np.clip(signal - float(np.percentile(vals, 55)), 0.0, None)
    signal = signal - float(np.mean(signal[mask]))
    h, w = signal.shape
    window = np.outer(np.hanning(h), np.hanning(w)).astype(np.float32)
    signal = signal * window * mask.astype(np.float32)
    corr = np.fft.fftshift(np.fft.ifft2(np.abs(np.fft.fft2(signal)) ** 2).real).astype(np.float32)
    cy, cx = h // 2, w // 2
    corr[cy, cx] = 0.0
    cmin, cmax = float(np.min(corr)), float(np.max(corr))
    if cmax <= cmin:
        return None
    corr = (corr - cmin) / (cmax - cmin)
    yy, xx = np.indices((h, w))
    dx = xx - cx
    dy = yy - cy
    dist = np.sqrt(dx * dx + dy * dy)
    radius_small = float(hole_radius_px) / float(scale)
    min_period = max(2.0 * radius_small + 1.0, 0.025 * float(min(h, w)))
    max_period = max(min_period + 1.0, 0.42 * float(min(h, w)))
    annulus = (dist >= min_period) & (dist <= max_period)
    if not np.any(annulus):
        return None
    local_max = ndi.maximum_filter(corr, size=9)
    peak_mask = annulus & (corr >= local_max - 1e-6)
    ys, xs = np.where(peak_mask)
    if ys.size < 6:
        ys, xs = np.where(annulus)
    values = corr[ys, xs]
    if values.size == 0:
        return None
    order = np.argsort(values)[::-1]
    peaks: list[tuple[float, float, float, float]] = []
    min_sep2 = max(4.0, (0.35 * min_period) ** 2)
    for idx in order:
        x = float(xs[int(idx)] - cx)
        y = float(ys[int(idx)] - cy)
        value = float(values[int(idx)])
        if value < 0.04:
            break
        if all((x - px) ** 2 + (y - py) ** 2 >= min_sep2 for _pv, px, py, _pd in peaks):
            peaks.append((value, x, y, float(math.hypot(x, y))))
        if len(peaks) >= 48:
            break
    if len(peaks) < 2:
        return None
    pts = np.asarray([[float(row['x']), float(row['y'])] for row in accepted_rows], dtype=np.float32)
    if pts.shape[0] > 260:
        order_rows = np.argsort([float(row.get('confidence', row.get('fit_score', 0.0))) for row in accepted_rows])[::-1][:260]
        pts = pts[order_rows].astype(np.float32)
    best: tuple[float, np.ndarray, np.ndarray, np.ndarray, float, float] | None = None
    for i, peak_a in enumerate(peaks):
        for peak_b in peaks[i + 1:]:
            u_small = np.asarray([peak_a[1], peak_a[2]], dtype=np.float32)
            v_small = np.asarray([peak_b[1], peak_b[2]], dtype=np.float32)
            lu = float(np.linalg.norm(u_small))
            lv = float(np.linalg.norm(v_small))
            if lu <= 1e-6 or lv <= 1e-6:
                continue
            det = float(u_small[0] * v_small[1] - u_small[1] * v_small[0])
            if abs(det) < 0.45 * lu * lv:
                continue
            ratio = lu / lv
            if ratio < 0.55 or ratio > 1.80:
                continue
            if det < 0:
                v_small = -v_small
            u = (u_small * float(scale)).astype(np.float32)
            v = (v_small * float(scale)).astype(np.float32)
            if not _basis_respects_hole_nonoverlap(u, v, hole_radius_px):
                continue
            cross = _corr_value_at_vector(corr, float(u_small[0] - v_small[0]), float(u_small[1] - v_small[1]))
            if pts.shape[0] >= 3:
                origin, residual = _basis_origin_and_residual(pts, u, v)
            else:
                origin = np.asarray([0.5 * float(w0 - 1), 0.5 * float(h0 - 1)], dtype=np.float32)
                residual = 0.35
            origin = _refine_lattice_origin_by_phase(evidence, wafer_mask, u, v, origin)
            phase_score = _basis_phase_mean_score(evidence, wafer_mask, u, v, origin)
            peak_score = 0.5 * (float(peak_a[0]) + float(peak_b[0])) + 0.25 * float(cross)
            score = float((peak_score + 0.75 * phase_score) / (1.0 + residual))
            if best is None or score > best[0]:
                best = (score, u, v, origin.astype(np.float32), residual, phase_score)
    if best is None:
        return None
    score, basis_u, basis_v, origin, residual, phase_score = best
    confidence = float(np.clip(0.55 * score + 0.45 * phase_score, 0.0, 1.0) * (1.0 / (1.0 + 0.5 * residual)))
    return pts, basis_u.astype(np.float32), basis_v.astype(np.float32), origin.astype(np.float32), confidence


def _basis_residual_for_rows(rows: list[dict[str, float | str]], basis_u: np.ndarray, basis_v: np.ndarray, origin: np.ndarray) -> float:
    if len(rows) < 3:
        return 1.0
    pts = np.asarray([[float(row['x']), float(row['y'])] for row in rows], dtype=np.float32)
    B = np.column_stack([basis_u, basis_v]).astype(np.float32)
    inv = np.linalg.pinv(B)
    uv = (pts - origin.astype(np.float32)) @ inv.T
    return float(np.median(np.linalg.norm(uv - np.round(uv), axis=1)))


def _estimate_consensus_probe_basis(accepted_rows: list[dict[str, float | str]], evidence: dict[str, np.ndarray], wafer_mask: np.ndarray):
    spatial = _estimate_probe_basis(accepted_rows)
    fourier = _autocorrelation_lattice_basis(evidence, wafer_mask, accepted_rows)
    global_fourier = _global_autocorrelation_lattice_basis(evidence, wafer_mask, accepted_rows, _row_median_radius(accepted_rows))
    if spatial is None:
        candidates = [cand for cand in (global_fourier, fourier) if cand is not None]
        if not candidates:
            return None
        return max(candidates, key=lambda cand: float(cand[4]))[:4]
    if fourier is None:
        if global_fourier is not None:
            _gpts, gu, gv, go, gconf = global_fourier
            _xy, su, sv, so = spatial
            spatial_residual = _basis_residual_for_rows(accepted_rows, su, sv, so)
            global_residual = _basis_residual_for_rows(accepted_rows, gu, gv, go)
            if gconf >= 0.30 and (global_residual <= spatial_residual + 0.05 or spatial_residual > 0.18):
                return _gpts, gu, gv, go
        if float(np.mean(wafer_mask)) > 0.985:
            xy, su, sv, so = spatial
            so = _refine_lattice_origin_by_phase(evidence, wafer_mask, su, sv, so)
            return xy, su, sv, so
        return spatial
    pts = np.asarray([[float(row['x']), float(row['y'])] for row in accepted_rows], dtype=np.float32)
    _xy, su, sv, so = spatial
    if float(np.mean(wafer_mask)) > 0.985:
        so = _refine_lattice_origin_by_phase(evidence, wafer_mask, su, sv, so)
    B = np.column_stack([su, sv]).astype(np.float32)
    inv = np.linalg.pinv(B)
    uv = (pts - so) @ inv.T
    spatial_residual = float(np.median(np.linalg.norm(uv - np.round(uv), axis=1)))
    _fpts, fu, fv, fo, fconf = fourier
    Bf = np.column_stack([fu, fv]).astype(np.float32)
    invf = np.linalg.pinv(Bf)
    uvf = (pts - fo) @ invf.T
    fourier_residual = float(np.median(np.linalg.norm(uvf - np.round(uvf), axis=1)))
    if global_fourier is not None:
        _gpts, gu, gv, go, gconf = global_fourier
        global_residual = _basis_residual_for_rows(accepted_rows, gu, gv, go)
        if gconf >= 0.30 and (global_residual <= min(spatial_residual, fourier_residual) + 0.05 or spatial_residual > 0.18):
            return _gpts, gu, gv, go
    if float(np.mean(wafer_mask)) > 0.985 and fconf >= 0.35:
        return pts, fu, fv, fo
    if fconf >= 0.35 and (fourier_residual <= spatial_residual + 0.025 or spatial_residual > 0.14):
        return pts, fu, fv, fo
    return spatial


def _enumerate_expected_nodes(origin: np.ndarray, basis_u: np.ndarray, basis_v: np.ndarray, present_cells: set[tuple[int, int]], xc: float, yc: float, wr: float, r_common: float, pad: int) -> list[dict[str, object]]:
    if present_cells:
        us = [c[0] for c in present_cells]
        vs = [c[1] for c in present_cells]
        umin, umax = min(us) - pad, max(us) + pad
        vmin, vmax = min(vs) - pad, max(vs) + pad
    else:
        umin = vmin = -4
        umax = vmax = 4
    rows: list[dict[str, object]] = []
    for iu in range(umin, umax + 1):
        for iv in range(vmin, vmax + 1):
            p = origin + iu * basis_u + iv * basis_v
            x, y = float(p[0]), float(p[1])
            d = float(math.hypot(x - xc, y - yc))
            if d > wr + r_common + 2:
                continue
            geometry_class = 'full' if d <= wr - r_common - 1 else 'partial'
            rows.append({
                'lattice_i': int(iu), 'lattice_j': int(iv), 'x_pred': x, 'y_pred': y,
                'geometry_class': geometry_class,
                'previous_status': 'present' if (int(iu), int(iv)) in present_cells else 'missing',
            })
    return rows


def _build_common_radius_templates(r0: int):
    ring_inner = int(round(1.25 * r0))
    ring_outer = int(round(1.75 * r0))
    edge_inner = max(1, int(round(0.85 * r0)))
    edge_outer = max(edge_inner + 1, int(round(1.15 * r0)))
    R = ring_outer
    g = np.arange(-R, R + 1)
    gy, gx = np.meshgrid(g, g, indexing='ij')
    dist = np.sqrt(gx ** 2 + gy ** 2)
    return {
        'r0': int(r0),
        'R': int(R),
        'core': dist <= r0,
        'ring': (dist >= ring_inner) & (dist <= ring_outer),
        'inner_edge': (dist >= edge_inner) & (dist < r0),
        'outer_edge': (dist >= r0) & (dist <= edge_outer),
    }


def _build_average_hole_templates(accepted_rows: list[dict[str, float | str]], templates: dict[str, np.ndarray], mean_scaled: np.ndarray, frame0_scaled: np.ndarray, inv_std: np.ndarray, inv_mad: np.ndarray, wafer_mask: np.ndarray):
    R = int(templates['R'])
    H, W = mean_scaled.shape
    by_conf = sorted(accepted_rows, key=lambda r: float(r['confidence']), reverse=True)
    proto_mean = []
    proto_std = []
    proto_mad = []
    proto_f0 = []
    for row in by_conf[:50]:
        x = int(round(float(row['x'])))
        y = int(round(float(row['y'])))
        if x < R or x >= W - R or y < R or y >= H - R:
            continue
        sl_y = slice(y - R, y + R + 1)
        sl_x = slice(x - R, x + R + 1)
        local_wafer = wafer_mask[sl_y, sl_x]
        cm = templates['core'] & local_wafer
        rm = templates['ring'] & local_wafer
        if int(cm.sum()) < 0.85 * int(templates['core'].sum()):
            continue
        if int(rm.sum()) < 0.65 * int(templates['ring'].sum()):
            continue
        proto_mean.append(mean_scaled[sl_y, sl_x])
        proto_std.append(inv_std[sl_y, sl_x])
        proto_mad.append(inv_mad[sl_y, sl_x])
        proto_f0.append(frame0_scaled[sl_y, sl_x])
    if not proto_mean:
        shape = templates['core'].shape
        zeros = np.zeros(shape, dtype=np.float32)
        return zeros, zeros, zeros, zeros
    return (
        np.mean(np.stack(proto_mean, axis=0), axis=0),
        np.mean(np.stack(proto_std, axis=0), axis=0),
        np.mean(np.stack(proto_mad, axis=0), axis=0),
        np.mean(np.stack(proto_f0, axis=0), axis=0),
    )


def _corr_masked(a: np.ndarray, b: np.ndarray, m: np.ndarray) -> float:
    av = a[m].ravel().astype(np.float32)
    bv = b[m].ravel().astype(np.float32)
    if av.size == 0 or bv.size == 0:
        return 0.0
    av = av - float(av.mean())
    bv = bv - float(bv.mean())
    den = float(np.linalg.norm(av) * np.linalg.norm(bv) + 1e-8)
    return float(np.dot(av, bv) / den)


def _local_template_window(shape: tuple[int, int], x0: int, y0: int, radius: int):
    h, w = shape
    y_start = y0 - radius
    y_stop = y0 + radius + 1
    x_start = x0 - radius
    x_stop = x0 + radius + 1
    y0_img = max(0, y_start)
    y1_img = min(h, y_stop)
    x0_img = max(0, x_start)
    x1_img = min(w, x_stop)
    if y0_img >= y1_img or x0_img >= x1_img:
        return None
    y0_tpl = y0_img - y_start
    y1_tpl = y0_tpl + (y1_img - y0_img)
    x0_tpl = x0_img - x_start
    x1_tpl = x0_tpl + (x1_img - x0_img)
    return (
        slice(y0_img, y1_img),
        slice(x0_img, x1_img),
        slice(y0_tpl, y1_tpl),
        slice(x0_tpl, x1_tpl),
    )


def _template_offsets(mask: np.ndarray, radius: int) -> np.ndarray:
    ys, xs = np.where(mask)
    if ys.size == 0:
        return np.empty((0, 4), dtype=np.int32)
    return np.column_stack((ys - int(radius), xs - int(radius), ys, xs)).astype(np.int32)


def _square_template_offsets(radius: int) -> np.ndarray:
    side = 2 * int(radius) + 1
    ys, xs = np.indices((side, side))
    return np.column_stack((ys.ravel() - int(radius), xs.ravel() - int(radius), ys.ravel(), xs.ravel())).astype(np.int32)


@njit(cache=True)
def _offset_mean4_kernel(
    mean_scaled: np.ndarray,
    frame0_scaled: np.ndarray,
    inv_std: np.ndarray,
    inv_mad: np.ndarray,
    wafer_mask: np.ndarray,
    offsets: np.ndarray,
    cx: int,
    cy: int,
) -> tuple[int, float, float, float, float]:
    h, w = mean_scaled.shape
    n = 0
    sum_mean = 0.0
    sum_frame0 = 0.0
    sum_std = 0.0
    sum_mad = 0.0
    for oi in range(offsets.shape[0]):
        py = cy + int(offsets[oi, 0])
        px = cx + int(offsets[oi, 1])
        if py < 0 or px < 0 or py >= h or px >= w:
            continue
        if not wafer_mask[py, px]:
            continue
        sum_mean += float(mean_scaled[py, px])
        sum_frame0 += float(frame0_scaled[py, px])
        sum_std += float(inv_std[py, px])
        sum_mad += float(inv_mad[py, px])
        n += 1
    if n <= 0:
        return 0, 0.0, 0.0, 0.0, 0.0
    den = 1.0 / float(n)
    return n, sum_mean * den, sum_frame0 * den, sum_std * den, sum_mad * den


@njit(cache=True)
def _corr_offsets_kernel(
    values: np.ndarray,
    proto: np.ndarray,
    wafer_mask: np.ndarray,
    offsets: np.ndarray,
    cx: int,
    cy: int,
) -> float:
    h, w = values.shape
    n = 0
    sum_a = 0.0
    sum_b = 0.0
    sum_aa = 0.0
    sum_bb = 0.0
    sum_ab = 0.0
    for oi in range(offsets.shape[0]):
        py = cy + int(offsets[oi, 0])
        px = cx + int(offsets[oi, 1])
        if py < 0 or px < 0 or py >= h or px >= w:
            continue
        if not wafer_mask[py, px]:
            continue
        ty = int(offsets[oi, 2])
        tx = int(offsets[oi, 3])
        a = float(values[py, px])
        b = float(proto[ty, tx])
        sum_a += a
        sum_b += b
        sum_aa += a * a
        sum_bb += b * b
        sum_ab += a * b
        n += 1
    if n <= 1:
        return 0.0
    nf = float(n)
    cov = sum_ab - (sum_a * sum_b) / nf
    va = sum_aa - (sum_a * sum_a) / nf
    vb = sum_bb - (sum_b * sum_b) / nf
    if va <= 1.0e-12 or vb <= 1.0e-12:
        return 0.0
    return float(cov / (math.sqrt(va * vb) + 1.0e-8))


@njit(cache=True)
def _common_radius_score_at_kernel(
    cx: int,
    cy: int,
    mean_scaled: np.ndarray,
    frame0_scaled: np.ndarray,
    inv_std: np.ndarray,
    inv_mad: np.ndarray,
    wafer_mask: np.ndarray,
    core_offsets: np.ndarray,
    ring_offsets: np.ndarray,
    inner_edge_offsets: np.ndarray,
    outer_edge_offsets: np.ndarray,
    all_offsets: np.ndarray,
    proto_mean: np.ndarray,
    proto_std: np.ndarray,
    proto_mad: np.ndarray,
    proto_f0: np.ndarray,
) -> tuple[float, float, float, float, float, float, float, float, float, float, float]:
    core_n, core_mean, core_f0, core_std, core_mad = _offset_mean4_kernel(mean_scaled, frame0_scaled, inv_std, inv_mad, wafer_mask, core_offsets, cx, cy)
    ring_n, ring_mean, ring_f0, ring_std, ring_mad = _offset_mean4_kernel(mean_scaled, frame0_scaled, inv_std, inv_mad, wafer_mask, ring_offsets, cx, cy)
    inner_n, inner_mean, inner_f0, _inner_std, _inner_mad = _offset_mean4_kernel(mean_scaled, frame0_scaled, inv_std, inv_mad, wafer_mask, inner_edge_offsets, cx, cy)
    outer_n, outer_mean, outer_f0, _outer_std, _outer_mad = _offset_mean4_kernel(mean_scaled, frame0_scaled, inv_std, inv_mad, wafer_mask, outer_edge_offsets, cx, cy)
    if core_n < 0.85 * float(core_offsets.shape[0]) or ring_n < 0.65 * float(ring_offsets.shape[0]):
        return -1.0e30, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    if inner_n < 0.75 * float(inner_edge_offsets.shape[0]) or outer_n < 0.75 * float(outer_edge_offsets.shape[0]):
        return -1.0e30, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    contrast_mean = core_mean - ring_mean
    contrast_f0 = core_f0 - ring_f0
    gap_std = core_std - ring_std
    gap_mad = core_mad - ring_mad
    edge_mean = inner_mean - outer_mean
    edge_f0 = inner_f0 - outer_f0
    corr_mean = _corr_offsets_kernel(mean_scaled, proto_mean, wafer_mask, all_offsets, cx, cy)
    corr_std = _corr_offsets_kernel(inv_std, proto_std, wafer_mask, all_offsets, cx, cy)
    corr_mad = _corr_offsets_kernel(inv_mad, proto_mad, wafer_mask, all_offsets, cx, cy)
    corr_f0 = _corr_offsets_kernel(frame0_scaled, proto_f0, wafer_mask, all_offsets, cx, cy)
    score_local = (
        0.18 * contrast_mean
        + 0.10 * contrast_f0
        + 0.15 * gap_std
        + 0.10 * gap_mad
        + 0.16 * edge_mean
        + 0.08 * edge_f0
        + 0.13 * corr_mean
        + 0.05 * corr_std
        + 0.03 * corr_mad
        + 0.02 * corr_f0
    )
    return (
        float(score_local),
        float(contrast_mean),
        float(contrast_f0),
        float(gap_std),
        float(gap_mad),
        float(edge_mean),
        float(edge_f0),
        float(corr_mean),
        float(corr_std),
        float(corr_mad),
        float(corr_f0),
    )


@njit(cache=True, parallel=True)
def _refine_common_radius_kernel(
    node_x: np.ndarray,
    node_y: np.ndarray,
    present_flags: np.ndarray,
    mean_scaled: np.ndarray,
    frame0_scaled: np.ndarray,
    inv_std: np.ndarray,
    inv_mad: np.ndarray,
    wafer_mask: np.ndarray,
    core_offsets: np.ndarray,
    ring_offsets: np.ndarray,
    inner_edge_offsets: np.ndarray,
    outer_edge_offsets: np.ndarray,
    all_offsets: np.ndarray,
    proto_mean: np.ndarray,
    proto_std: np.ndarray,
    proto_mad: np.ndarray,
    proto_f0: np.ndarray,
    present_search: int,
    missing_search: int,
) -> np.ndarray:
    out = np.zeros((node_x.shape[0], 14), dtype=np.float32)
    for ni in prange(node_x.shape[0]):
        search_r = present_search if present_flags[ni] > 0 else missing_search
        best_score = -1.0e30
        best_x = 0.0
        best_y = 0.0
        best = np.zeros(10, dtype=np.float32)
        for dy in range(-search_r, search_r + 1):
            cy = int(round(float(node_y[ni]) + float(dy)))
            for dx in range(-search_r, search_r + 1):
                cx = int(round(float(node_x[ni]) + float(dx)))
                score, contrast_mean, contrast_f0, gap_std, gap_mad, edge_mean, edge_f0, corr_mean, corr_std, corr_mad, corr_f0 = _common_radius_score_at_kernel(
                    cx,
                    cy,
                    mean_scaled,
                    frame0_scaled,
                    inv_std,
                    inv_mad,
                    wafer_mask,
                    core_offsets,
                    ring_offsets,
                    inner_edge_offsets,
                    outer_edge_offsets,
                    all_offsets,
                    proto_mean,
                    proto_std,
                    proto_mad,
                    proto_f0,
                )
                if score > best_score:
                    best_score = score
                    best_x = float(cx)
                    best_y = float(cy)
                    best[0] = float(contrast_mean)
                    best[1] = float(contrast_f0)
                    best[2] = float(gap_std)
                    best[3] = float(gap_mad)
                    best[4] = float(edge_mean)
                    best[5] = float(edge_f0)
                    best[6] = float(corr_mean)
                    best[7] = float(corr_std)
                    best[8] = float(corr_mad)
                    best[9] = float(corr_f0)
        if best_score > -1.0e20:
            out[ni, 0] = 1.0
            out[ni, 1] = best_x
            out[ni, 2] = best_y
            out[ni, 3] = float(best_score)
            for bi in range(10):
                out[ni, 4 + bi] = best[bi]
    return out


def _local_common_radius_score(x0: float, y0: float, templates: dict[str, np.ndarray], mean_scaled: np.ndarray, frame0_scaled: np.ndarray, inv_std: np.ndarray, inv_mad: np.ndarray, wafer_mask: np.ndarray, proto_mean: np.ndarray, proto_std: np.ndarray, proto_mad: np.ndarray, proto_f0: np.ndarray):
    R = int(templates['R'])
    H, W = mean_scaled.shape
    x0 = int(round(float(x0)))
    y0 = int(round(float(y0)))
    window = _local_template_window((H, W), x0, y0, R)
    if window is None:
        return None
    sl_y, sl_x, tpl_y, tpl_x = window
    local_wafer = wafer_mask[sl_y, sl_x]
    core_tpl = templates['core'][tpl_y, tpl_x]
    ring_tpl = templates['ring'][tpl_y, tpl_x]
    inner_edge_tpl = templates['inner_edge'][tpl_y, tpl_x]
    outer_edge_tpl = templates['outer_edge'][tpl_y, tpl_x]
    cm = core_tpl & local_wafer
    rm = ring_tpl & local_wafer
    ie = inner_edge_tpl & local_wafer
    oe = outer_edge_tpl & local_wafer
    core_total = int(core_tpl.sum())
    ring_total = int(ring_tpl.sum())
    inner_edge_total = int(inner_edge_tpl.sum())
    outer_edge_total = int(outer_edge_tpl.sum())
    if min(core_total, ring_total, inner_edge_total, outer_edge_total) <= 0:
        return None
    if int(cm.sum()) < 0.85 * core_total or int(rm.sum()) < 0.65 * ring_total:
        return None
    if int(ie.sum()) < 0.75 * inner_edge_total or int(oe.sum()) < 0.75 * outer_edge_total:
        return None
    p_mean = mean_scaled[sl_y, sl_x]
    p_std = inv_std[sl_y, sl_x]
    p_mad = inv_mad[sl_y, sl_x]
    p_f0 = frame0_scaled[sl_y, sl_x]
    contrast_mean = float(p_mean[cm].mean() - p_mean[rm].mean())
    contrast_f0 = float(p_f0[cm].mean() - p_f0[rm].mean())
    gap_std = float(p_std[cm].mean() - p_std[rm].mean())
    gap_mad = float(p_mad[cm].mean() - p_mad[rm].mean())
    edge_mean = float(p_mean[ie].mean() - p_mean[oe].mean())
    edge_f0 = float(p_f0[ie].mean() - p_f0[oe].mean())
    corr_mean = _corr_masked(p_mean, proto_mean[tpl_y, tpl_x], local_wafer)
    corr_std = _corr_masked(p_std, proto_std[tpl_y, tpl_x], local_wafer)
    corr_mad = _corr_masked(p_mad, proto_mad[tpl_y, tpl_x], local_wafer)
    corr_f0 = _corr_masked(p_f0, proto_f0[tpl_y, tpl_x], local_wafer)
    score_local = (
        0.18 * contrast_mean
        + 0.10 * contrast_f0
        + 0.15 * gap_std
        + 0.10 * gap_mad
        + 0.16 * edge_mean
        + 0.08 * edge_f0
        + 0.13 * corr_mean
        + 0.05 * corr_std
        + 0.03 * corr_mad
        + 0.02 * corr_f0
    )
    return {
        'score_local': float(score_local),
        'contrast_mean': float(contrast_mean),
        'contrast_frame0': float(contrast_f0),
        'stability_gap_std': float(gap_std),
        'stability_gap_mad': float(gap_mad),
        'edge_mean': float(edge_mean),
        'edge_frame0': float(edge_f0),
        'corr_mean': float(corr_mean),
        'corr_std': float(corr_std),
        'corr_mad': float(corr_mad),
        'corr_frame0': float(corr_f0),
    }


def _refine_expected_node_common_radius(x0: float, y0: float, templates: dict[str, np.ndarray], mean_scaled: np.ndarray, frame0_scaled: np.ndarray, inv_std: np.ndarray, inv_mad: np.ndarray, wafer_mask: np.ndarray, proto_mean: np.ndarray, proto_std: np.ndarray, proto_mad: np.ndarray, proto_f0: np.ndarray, search_r: int = 6):
    best = None
    for dy in range(-search_r, search_r + 1):
        y = int(round(float(y0 + dy)))
        for dx in range(-search_r, search_r + 1):
            x = int(round(float(x0 + dx)))
            res = _local_common_radius_score(x, y, templates, mean_scaled, frame0_scaled, inv_std, inv_mad, wafer_mask, proto_mean, proto_std, proto_mad, proto_f0)
            if res is None:
                continue
            if best is None or res['score_local'] > best['score_local']:
                best = {'x': float(x), 'y': float(y), **res}
    return best


def _common_radius_anchor_stage(
    expected_nodes: list[dict[str, object]],
    initial_accepted: list[dict[str, float | str]],
    mean_gray: np.ndarray,
    frame0_gray: np.ndarray,
    std_gray: np.ndarray,
    mad_gray: np.ndarray,
    wafer_mask: np.ndarray,
    watchdog: _DetectorWatchdog | None = None,
):
    r0 = int(round(float(np.median([float(r['r']) for r in initial_accepted])))) if initial_accepted else 13
    templates = _build_common_radius_templates(r0)
    mean_scaled = _robust_scale(mean_gray, wafer_mask)
    frame0_scaled = _robust_scale(frame0_gray, wafer_mask)
    inv_std = _robust_scale(1.0 / (std_gray + 1e-6), wafer_mask)
    inv_mad = _robust_scale(1.0 / (mad_gray + 1e-6), wafer_mask)
    proto_mean, proto_std, proto_mad, proto_f0 = _build_average_hole_templates(initial_accepted, templates, mean_scaled, frame0_scaled, inv_std, inv_mad, wafer_mask)

    rows = []
    present_search = max(1, min(3, int(round(0.18 * float(r0)))))
    missing_search = max(3, min(6, int(round(0.45 * float(r0)))))
    if watchdog is not None:
        watchdog.check("common-radius anchor refinement prepare")
    if expected_nodes:
        node_x = np.asarray([float(row.get('x_seed', row['x_pred'])) for row in expected_nodes], dtype=np.float32)
        node_y = np.asarray([float(row.get('y_seed', row['y_pred'])) for row in expected_nodes], dtype=np.float32)
        present_flags = np.asarray([1 if str(row.get('previous_status', 'missing')) == 'present' else 0 for row in expected_nodes], dtype=np.uint8)
        R = int(templates['R'])
        fit_rows = _refine_common_radius_kernel(
            node_x,
            node_y,
            present_flags,
            np.asarray(mean_scaled, dtype=np.float32),
            np.asarray(frame0_scaled, dtype=np.float32),
            np.asarray(inv_std, dtype=np.float32),
            np.asarray(inv_mad, dtype=np.float32),
            wafer_mask.astype(np.bool_, copy=False),
            _template_offsets(templates['core'], R),
            _template_offsets(templates['ring'], R),
            _template_offsets(templates['inner_edge'], R),
            _template_offsets(templates['outer_edge'], R),
            _square_template_offsets(R),
            np.asarray(proto_mean, dtype=np.float32),
            np.asarray(proto_std, dtype=np.float32),
            np.asarray(proto_mad, dtype=np.float32),
            np.asarray(proto_f0, dtype=np.float32),
            int(present_search),
            int(missing_search),
        )
        if watchdog is not None:
            watchdog.check("common-radius anchor refinement")
    else:
        fit_rows = np.empty((0, 14), dtype=np.float32)
    for row, fit in zip(expected_nodes, fit_rows):
        if float(fit[0]) <= 0.0:
            continue
        is_present = str(row.get('previous_status', 'missing')) == 'present'
        rows.append({
            'lattice_i': int(row['lattice_i']),
            'lattice_j': int(row['lattice_j']),
            'previous_status': 'present' if is_present else str(row.get('previous_status', 'missing')),
            'x': float(fit[1]),
            'y': float(fit[2]),
            'r': float(r0),
            'score_local': float(fit[3]),
            'contrast_mean': float(fit[4]),
            'contrast_frame0': float(fit[5]),
            'stability_gap_std': float(fit[6]),
            'stability_gap_mad': float(fit[7]),
            'edge_mean': float(fit[8]),
            'edge_frame0': float(fit[9]),
            'corr_mean': float(fit[10]),
            'corr_std': float(fit[11]),
            'corr_mad': float(fit[12]),
            'corr_frame0': float(fit[13]),
        })
    if not rows:
        return [], r0
    present_local = np.asarray([r['score_local'] for r in rows if r['previous_status'] == 'present'], dtype=np.float32)
    prelim_accept = float(np.percentile(present_local, 5)) if len(present_local) else 0.0
    prelim_map = {(int(r['lattice_i']), int(r['lattice_j'])): bool(r['score_local'] >= prelim_accept) for r in rows}

    def neighbor_support(i: int, j: int) -> float:
        ncoords = [(i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1), (i - 1, j + 1), (i + 1, j - 1)]
        vals = [1.0 if prelim_map.get(c, False) else 0.0 for c in ncoords]
        return float(np.mean(vals)) if vals else 0.0

    for r in rows:
        r['neighbor_support_frac'] = neighbor_support(int(r['lattice_i']), int(r['lattice_j']))
        r['score_total'] = float(r['score_local'] + 0.06 * r['neighbor_support_frac'])

    present_total = np.asarray([r['score_total'] for r in rows if r['previous_status'] == 'present'], dtype=np.float32)
    accepted_threshold = float(np.percentile(present_total, 8)) if len(present_total) else 0.0
    provisional_threshold = float(np.percentile(present_total, 3)) if len(present_total) else accepted_threshold
    p05, p95 = np.percentile(present_total, [5, 95]) if len(present_total) else (0.0, 1.0)
    den = max(float(p95 - p05), 1e-6)

    accepted_rows = []
    for r in rows:
        score = float(r['score_total'])
        if score >= accepted_threshold:
            status = 'accepted'
        elif score >= provisional_threshold:
            status = 'provisional'
        else:
            status = 'rejected'
        r['status'] = status
        r['confidence'] = float(np.clip((score - p05) / den, 0.0, 1.0))
        if status == 'accepted':
            accepted_rows.append(r)
    return accepted_rows, r0


def _build_overflow_mask(shape: tuple[int, int], centers: np.ndarray, radius: int, wafer_mask: np.ndarray, margin_px: int = 8) -> np.ndarray:
    h, w = shape
    mask = np.zeros((h, w), dtype=np.uint8)
    rr = max(1, int(round(radius + margin_px)))
    for x, y in centers:
        cv2.circle(mask, (int(round(float(x))), int(round(float(y)))), rr, 1, thickness=-1)
    return (mask > 0) & wafer_mask


def _build_border_templates(radius_values: Sequence[int]) -> dict[int, dict[str, np.ndarray]]:
    radius_values = [int(r) for r in radius_values]
    max_r = max(radius_values)
    R = max_r + 4
    g = np.arange(-R, R + 1)
    gy, gx = np.meshgrid(g, g, indexing='ij')
    dist = np.sqrt(gx * gx + gy * gy)
    templates: dict[int, dict[str, np.ndarray]] = {}
    for r in radius_values:
        templates[int(r)] = {
            'edge_band': (dist >= (r - 1.2)) & (dist <= (r + 1.2)),
            'inner_band': dist <= (r - 1.2),
            'outer_band': (dist >= (r + 1.2)) & (dist <= (r + 4.0)),
            'R': int(R),
        }
    return templates


def _local_border_score(x0: float, y0: float, radius: int, mean_scaled: np.ndarray, frame0_scaled: np.ndarray, grad_scaled: np.ndarray, wafer_mask: np.ndarray, overflow_mask: np.ndarray, templates: dict[int, dict[str, np.ndarray]], allow_partial: bool):
    tpl = templates[int(radius)]
    R = int(tpl['R'])
    xi, yi = int(round(float(x0))), int(round(float(y0)))
    H, W = mean_scaled.shape
    window = _local_template_window((H, W), xi, yi, R)
    if window is None:
        return None
    sl_y, sl_x, tpl_y, tpl_x = window
    valid = wafer_mask[sl_y, sl_x] & overflow_mask[sl_y, sl_x]
    edge_tpl = tpl['edge_band'][tpl_y, tpl_x]
    inner_tpl = tpl['inner_band'][tpl_y, tpl_x]
    outer_tpl = tpl['outer_band'][tpl_y, tpl_x]
    edge = edge_tpl & valid
    inner = inner_tpl & valid
    outer = outer_tpl & valid
    edge_total = int(edge_tpl.sum())
    inner_total = int(inner_tpl.sum())
    outer_total = int(outer_tpl.sum())
    if min(edge_total, inner_total, outer_total) <= 0:
        return None
    edge_cov = float(edge.sum() / edge_total)
    inner_cov = float(inner.sum() / inner_total)
    outer_cov = float(outer.sum() / outer_total)
    if allow_partial:
        if edge_cov < 0.25 or inner_cov < 0.20 or outer_cov < 0.15:
            return None
    else:
        if edge_cov < 0.60 or inner_cov < 0.55 or outer_cov < 0.45:
            return None
    patch_grad = grad_scaled[sl_y, sl_x]
    patch_mean = mean_scaled[sl_y, sl_x]
    patch_f0 = frame0_scaled[sl_y, sl_x]
    return {
        'edge_grad': float(patch_grad[edge].mean()),
        'edge_contrast_mean': float(patch_mean[inner].mean() - patch_mean[outer].mean()),
        'edge_contrast_frame0': float(patch_f0[inner].mean() - patch_f0[outer].mean()),
    }


def _border_refine_anchors(
    anchor_rows: list[dict[str, object]],
    mean_gray: np.ndarray,
    frame0_gray: np.ndarray,
    wafer_mask: np.ndarray,
    watchdog: _DetectorWatchdog | None = None,
):
    if not anchor_rows:
        return [], 0
    r0_int = int(round(float(np.median([float(r['r']) for r in anchor_rows]))))
    mean_scaled = _robust_scale(mean_gray, wafer_mask)
    frame0_scaled = _robust_scale(frame0_gray, wafer_mask)
    grad_combo = _robust_scale(0.7 * _gradient_magnitude(mean_scaled) + 0.3 * _gradient_magnitude(frame0_scaled), wafer_mask)
    centers = np.asarray([[float(r['x']), float(r['y'])] for r in anchor_rows], dtype=np.float32)
    overflow_mask = _build_overflow_mask(mean_gray.shape, centers, r0_int, wafer_mask, margin_px=8)
    radius_candidates = list(range(max(3, r0_int - 2), r0_int + 3))
    templates = _build_border_templates(radius_candidates)
    search_xy = max(1, min(3, int(round(0.15 * float(r0_int)))))
    loose_rows = []
    for row_i, row in enumerate(anchor_rows):
        if watchdog is not None and row_i % 16 == 0:
            watchdog.check("border-refine anchors loose pass")
        x_init = float(row['x'])
        y_init = float(row['y'])
        best = None
        for dy in range(-search_xy, search_xy + 1):
            for dx in range(-search_xy, search_xy + 1):
                x = x_init + dx
                y = y_init + dy
                for r in radius_candidates:
                    res = _local_border_score(x, y, r, mean_scaled, frame0_scaled, grad_combo, wafer_mask, overflow_mask, templates, allow_partial=False)
                    if res is None:
                        continue
                    score = 0.62 * res['edge_grad'] + 0.23 * res['edge_contrast_mean'] + 0.15 * res['edge_contrast_frame0'] - 0.018 * math.hypot(dx, dy) - 0.028 * abs(r - r0_int)
                    if best is None or score > best['score_loose']:
                        best = {'x_loose': float(x), 'y_loose': float(y), 'r_loose': float(r), 'score_loose': float(score), **res}
        if best is not None:
            loose_rows.append({**row, **best})
    if not loose_rows:
        return anchor_rows, r0_int
    r_common_refined = float(np.median([float(r['r_loose']) for r in loose_rows]))
    r_common_int = int(round(r_common_refined))
    if r_common_int not in templates:
        templates = _build_border_templates(radius_candidates + [r_common_int])
    final_rows = []
    for row_i, row in enumerate(loose_rows):
        if watchdog is not None and row_i % 16 == 0:
            watchdog.check("border-refine anchors final pass")
        x_init = float(row['x'])
        y_init = float(row['y'])
        best = None
        for dy in range(-search_xy, search_xy + 1):
            for dx in range(-search_xy, search_xy + 1):
                x = x_init + dx
                y = y_init + dy
                res = _local_border_score(x, y, r_common_int, mean_scaled, frame0_scaled, grad_combo, wafer_mask, overflow_mask, templates, allow_partial=False)
                if res is None:
                    continue
                score = 0.68 * res['edge_grad'] + 0.20 * res['edge_contrast_mean'] + 0.12 * res['edge_contrast_frame0'] - 0.020 * math.hypot(dx, dy)
                if best is None or score > best['score_final']:
                    best = {'x_final': float(x), 'y_final': float(y), 'r_final': float(r_common_int), 'score_final': float(score), **res}
        if best is not None:
            final_rows.append({**row, **best})
    return final_rows, r_common_int


def _local_outer_contour_fit(
    mean_gray: np.ndarray,
    frame0_gray: np.ndarray,
    wafer_mask: np.ndarray,
    x0: float,
    y0: float,
    r_guess: float,
    max_radius: float,
) -> tuple[float, float, float, float] | None:
    h, w = mean_gray.shape
    pad = max(8, int(math.ceil(max_radius + 4.0)))
    xi = int(round(float(x0)))
    yi = int(round(float(y0)))
    x_start = max(0, xi - pad)
    x_stop = min(w, xi + pad + 1)
    y_start = max(0, yi - pad)
    y_stop = min(h, yi + pad + 1)
    if x_start >= x_stop or y_start >= y_stop:
        return None
    crop = (0.65 * mean_gray[y_start:y_stop, x_start:x_stop] + 0.35 * frame0_gray[y_start:y_stop, x_start:x_stop]).astype(np.float32)
    local_mask = wafer_mask[y_start:y_stop, x_start:x_stop].astype(bool)
    if int(local_mask.sum()) < 16:
        return None
    cy = float(y0) - float(y_start)
    cx = float(x0) - float(x_start)
    yy, xx = np.indices(crop.shape)
    d = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    core = (d <= max(2.0, 0.45 * float(r_guess))) & local_mask
    ring = (d >= max(float(r_guess) + 2.0, 0.55 * float(max_radius))) & (d <= float(max_radius)) & local_mask
    if int(core.sum()) < 6 or int(ring.sum()) < 12:
        return None
    core_level = float(np.median(crop[core]))
    ring_level = float(np.median(crop[ring]))
    crop_u8 = _robust_scale(crop, local_mask) * 255.0
    crop_u8 = cv2.GaussianBlur(crop_u8.astype(np.uint8), (5, 5), 0)
    if core_level >= ring_level:
        _thr, binary = cv2.threshold(crop_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        _thr, binary = cv2.threshold(crop_u8, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    binary = ((binary > 0) & local_mask).astype(np.uint8)
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, 8)
    if n <= 1:
        return None
    center_xi = int(np.clip(round(cx), 0, labels.shape[1] - 1))
    center_yi = int(np.clip(round(cy), 0, labels.shape[0] - 1))
    label = int(labels[center_yi, center_xi])
    if label == 0:
        best_label = 0
        best_dist = float('inf')
        for lab in range(1, n):
            area = float(stats[lab, cv2.CC_STAT_AREA])
            if area < math.pi * max(2.0, 0.35 * float(r_guess)) ** 2:
                continue
            lx, ly = float(centroids[lab][0]), float(centroids[lab][1])
            dist = math.hypot(lx - cx, ly - cy)
            if dist < best_dist:
                best_dist = dist
                best_label = lab
        label = best_label
    if label == 0:
        return None
    comp = (labels == label).astype(np.uint8)
    contours, _hier = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    area = float(cv2.contourArea(contour))
    if area < math.pi * max(2.0, 0.35 * float(r_guess)) ** 2:
        return None
    (fit_x, fit_y), enclosing_r = cv2.minEnclosingCircle(contour)
    eq_r = math.sqrt(max(area, 1.0) / math.pi)
    radius = 0.65 * float(enclosing_r) + 0.35 * float(eq_r)
    if radius < 0.55 * float(r_guess) or radius > max(float(max_radius), 1.45 * float(r_guess)):
        return None
    return float(fit_x + x_start), float(fit_y + y_start), float(radius), float(area)


def _refine_rows_to_outer_contours(
    anchor_rows: list[dict[str, object]],
    strong_rows: list[dict[str, object]],
    mean_gray: np.ndarray,
    frame0_gray: np.ndarray,
    wafer_mask: np.ndarray,
    r_common: int,
    watchdog: _DetectorWatchdog | None = None,
) -> tuple[list[dict[str, object]], list[dict[str, object]], int]:
    if not anchor_rows and not strong_rows:
        return anchor_rows, strong_rows, r_common
    if anchor_rows:
        try:
            _origin, u_ref, v_ref, _res = _least_squares_lattice(anchor_rows)
            lattice_spacing = min(float(np.linalg.norm(u_ref)), float(np.linalg.norm(v_ref)))
        except Exception:
            lattice_spacing = 4.0 * float(r_common)
    else:
        lattice_spacing = 4.0 * float(r_common)
    max_radius = max(float(r_common) + 8.0, min(0.42 * lattice_spacing, 2.2 * float(r_common)))
    fits: list[tuple[dict[str, object], str, tuple[float, float, float, float]]] = []
    for row_i, row in enumerate(anchor_rows):
        if watchdog is not None and row_i % 16 == 0:
            watchdog.check("outer-contour anchor alignment")
        fit = _local_outer_contour_fit(mean_gray, frame0_gray, wafer_mask, float(row['x_final']), float(row['y_final']), float(r_common), max_radius)
        if fit is not None:
            fits.append((row, 'anchor', fit))
    for row_i, row in enumerate(strong_rows):
        if watchdog is not None and row_i % 16 == 0:
            watchdog.check("outer-contour recovered-hole alignment")
        fit = _local_outer_contour_fit(mean_gray, frame0_gray, wafer_mask, float(row['x_rec']), float(row['y_rec']), float(r_common), max_radius)
        if fit is not None:
            fits.append((row, 'strong', fit))
    if not fits:
        return anchor_rows, strong_rows, r_common
    radii = np.asarray([fit[2] for _row, _kind, fit in fits], dtype=np.float32)
    outer_r = int(round(float(np.median(radii))))
    outer_r = max(int(r_common), outer_r)
    fit_by_id = {id(row): fit for row, _kind, fit in fits}

    def _apply_anchor(row: dict[str, object]) -> dict[str, object]:
        out = dict(row)
        fit = fit_by_id.get(id(row))
        if fit is not None:
            fx, fy, fr, _area = fit
            if math.hypot(fx - float(row['x_final']), fy - float(row['y_final'])) <= max(2.0, 0.55 * float(outer_r)):
                out['x_final'] = float(fx)
                out['y_final'] = float(fy)
                out['score_final'] = float(out.get('score_final', out.get('score_total', 0.0))) + 0.02
        out['r_final'] = float(outer_r)
        return out

    def _apply_strong(row: dict[str, object]) -> dict[str, object]:
        out = dict(row)
        fit = fit_by_id.get(id(row))
        if fit is not None:
            fx, fy, fr, _area = fit
            if math.hypot(fx - float(row['x_rec']), fy - float(row['y_rec'])) <= max(2.0, 0.55 * float(outer_r)):
                out['x_rec'] = float(fx)
                out['y_rec'] = float(fy)
        out['r_rec'] = float(outer_r)
        return out

    return [_apply_anchor(row) for row in anchor_rows], [_apply_strong(row) for row in strong_rows], outer_r


def _least_squares_lattice(rows: list[dict[str, object]]) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    pts = np.asarray([[float(r['x_final']), float(r['y_final'])] for r in rows], dtype=np.float32)
    ij = np.asarray([[int(r['lattice_i']), int(r['lattice_j'])] for r in rows], dtype=np.int32)
    A = np.column_stack([np.ones(len(ij), dtype=np.float32), ij[:, 0].astype(np.float32), ij[:, 1].astype(np.float32)])
    coef_x, *_ = np.linalg.lstsq(A, pts[:, 0], rcond=None)
    coef_y, *_ = np.linalg.lstsq(A, pts[:, 1], rcond=None)
    origin_ref = np.array([coef_x[0], coef_y[0]], dtype=np.float32)
    u_ref = np.array([coef_x[1], coef_y[1]], dtype=np.float32)
    v_ref = np.array([coef_x[2], coef_y[2]], dtype=np.float32)
    pred = np.column_stack([A @ coef_x, A @ coef_y]).astype(np.float32)
    residual = float(np.mean(np.linalg.norm(pts - pred, axis=1)))
    return origin_ref, u_ref, v_ref, residual


def _recover_strong_missing(
    anchor_rows: list[dict[str, object]],
    expected_nodes: list[dict[str, object]],
    mean_gray: np.ndarray,
    frame0_gray: np.ndarray,
    wafer_circle: tuple[int, int, int],
    r_common: int,
    wafer_mask_visible: np.ndarray | None = None,
    watchdog: _DetectorWatchdog | None = None,
):
    if not anchor_rows:
        z = np.zeros(2, dtype=np.float32)
        return [], [], [], [], z, z.copy(), z.copy()
    xc, yc, wr = float(wafer_circle[0]), float(wafer_circle[1]), float(wafer_circle[2])
    origin_ref, u_ref, v_ref, _ = _least_squares_lattice(anchor_rows)
    mean_scaled = _robust_scale(mean_gray)
    frame0_scaled = _robust_scale(frame0_gray)
    grad_combo = _robust_scale(0.7 * _gradient_magnitude(mean_scaled) + 0.3 * _gradient_magnitude(frame0_scaled))
    H, W = mean_gray.shape
    if wafer_mask_visible is None:
        yy, xx = np.indices((H, W))
        wafer_mask = ((xx - xc) ** 2 + (yy - yc) ** 2) <= (0.995 * wr) ** 2
    else:
        wafer_mask = wafer_mask_visible.astype(bool)

    expected_refit = []
    anchor_ids = {(int(r['lattice_i']), int(r['lattice_j'])) for r in anchor_rows}
    for row in expected_nodes:
        i = int(row['lattice_i'])
        j = int(row['lattice_j'])
        x = float(origin_ref[0] + u_ref[0] * i + v_ref[0] * j)
        y = float(origin_ref[1] + u_ref[1] * i + v_ref[1] * j)
        d = float(math.hypot(x - xc, y - yc))
        geometry_class = 'full' if d <= wr - r_common - 1 else ('partial' if d <= wr + r_common - 1 else 'outside')
        expected_refit.append({'lattice_i': i, 'lattice_j': j, 'x_pred': x, 'y_pred': y, 'geometry_class': geometry_class, 'anchor_detected': (i, j) in anchor_ids})
    candidate_missing = [r for r in expected_refit if not bool(r['anchor_detected']) and str(r['geometry_class']) != 'outside']

    max_r = r_common + 2
    R = max_r + 4
    g = np.arange(-R, R + 1)
    gy, gx = np.meshgrid(g, g, indexing='ij')
    dist = np.sqrt(gx ** 2 + gy ** 2)
    radius_candidates = list(range(max(3, r_common - 2), r_common + 3))
    templates = {}
    for r in radius_candidates:
        templates[int(r)] = {
            'edge_band': (dist >= (r - 1.2)) & (dist <= (r + 1.2)),
            'inner_band': dist <= (r - 1.2),
            'outer_band': (dist >= (r + 1.2)) & (dist <= (r + 4.0)),
            'R': int(R),
        }

    def local_missing_score(x0: float, y0: float, r: int, allow_partial: bool):
        tpl = templates[int(r)]
        RR = int(tpl['R'])
        xi, yi = int(round(float(x0))), int(round(float(y0)))
        window = _local_template_window((H, W), xi, yi, RR)
        if window is None:
            return None
        sl_y, sl_x, tpl_y, tpl_x = window
        local_wafer = wafer_mask[sl_y, sl_x]
        edge_tpl = tpl['edge_band'][tpl_y, tpl_x]
        inner_tpl = tpl['inner_band'][tpl_y, tpl_x]
        outer_tpl = tpl['outer_band'][tpl_y, tpl_x]
        edge = edge_tpl & local_wafer
        inner = inner_tpl & local_wafer
        outer = outer_tpl & local_wafer
        edge_total = int(edge_tpl.sum())
        inner_total = int(inner_tpl.sum())
        outer_total = int(outer_tpl.sum())
        if min(edge_total, inner_total, outer_total) <= 0:
            return None
        edge_cov = float(edge.sum() / edge_total)
        inner_cov = float(inner.sum() / inner_total)
        outer_cov = float(outer.sum() / outer_total)
        if allow_partial:
            if edge_cov < 0.25 or inner_cov < 0.20 or outer_cov < 0.15:
                return None
        else:
            if edge_cov < 0.60 or inner_cov < 0.55 or outer_cov < 0.45:
                return None
        patch_grad = grad_combo[sl_y, sl_x]
        patch_mean = mean_scaled[sl_y, sl_x]
        patch_f0 = frame0_scaled[sl_y, sl_x]
        return {
            'edge_grad': float(patch_grad[edge].mean()),
            'edge_contrast_mean': float(patch_mean[inner].mean() - patch_mean[outer].mean()),
            'edge_contrast_frame0': float(patch_f0[inner].mean() - patch_f0[outer].mean()),
        }

    anchor_scores = []
    for row in anchor_rows:
        res = local_missing_score(float(row['x_final']), float(row['y_final']), r_common, allow_partial=False)
        if res is not None:
            anchor_scores.append(0.66 * res['edge_grad'] + 0.22 * res['edge_contrast_mean'] + 0.12 * res['edge_contrast_frame0'])
    anchor_scores = np.asarray(anchor_scores, dtype=np.float32)
    strong_thr = float(np.percentile(anchor_scores, 9)) if len(anchor_scores) else 0.0
    weak_thr = float(np.percentile(anchor_scores, 2)) if len(anchor_scores) else 0.0

    anchor_support = {(int(r['lattice_i']), int(r['lattice_j'])): True for r in anchor_rows}

    def neighbor_support(i: int, j: int) -> float:
        ncoords = [(i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1), (i - 1, j + 1), (i + 1, j - 1)]
        vals = [1.0 if anchor_support.get(c, False) else 0.0 for c in ncoords]
        return float(np.mean(vals)) if vals else 0.0

    strong = []
    weak = []
    still = []
    rows = []
    search_xy = 4
    for row_i, row in enumerate(candidate_missing):
        if watchdog is not None and row_i % 16 == 0:
            watchdog.check("recover missing lattice holes")
        x0 = float(row['x_pred'])
        y0 = float(row['y_pred'])
        allow_partial = str(row['geometry_class']) == 'partial'
        n_support = neighbor_support(int(row['lattice_i']), int(row['lattice_j']))
        best = None
        for dy in range(-search_xy, search_xy + 1):
            for dx in range(-search_xy, search_xy + 1):
                x = x0 + dx
                y = y0 + dy
                for r in radius_candidates:
                    xi = int(round(float(x)))
                    yi = int(round(float(y)))
                    if not (0 <= xi < W and 0 <= yi < H) or not wafer_mask[yi, xi]:
                        continue
                    res = local_missing_score(x, y, r, allow_partial=allow_partial)
                    if res is None:
                        continue
                    score_local = 0.66 * res['edge_grad'] + 0.22 * res['edge_contrast_mean'] + 0.12 * res['edge_contrast_frame0']
                    score_total = score_local + 0.05 * n_support - 0.018 * math.hypot(dx, dy) - 0.020 * abs(r - r_common)
                    if best is None or score_total > best['score_total']:
                        best = {
                            'lattice_i': int(row['lattice_i']), 'lattice_j': int(row['lattice_j']),
                            'x_pred': x0, 'y_pred': y0, 'geometry_class': str(row['geometry_class']),
                            'x_rec': float(x), 'y_rec': float(y), 'r_rec': float(r),
                            'score_local': float(score_local), 'score_total': float(score_total),
                            'neighbor_support': float(n_support),
                        }
        if best is None:
            continue
        if best['score_local'] >= strong_thr:
            best['status'] = 'recovered_strong'
            strong.append(best)
        elif best['score_local'] >= weak_thr:
            best['status'] = 'recovered_weak'
            weak.append(best)
        else:
            best['status'] = 'still_missing'
            still.append(best)
        rows.append(best)
    return strong, weak, still, rows, origin_ref, u_ref, v_ref


def _circle_mask(shape: tuple[int, int], circle: tuple[float, float, float]) -> np.ndarray:
    h, w = shape
    yy, xx = np.indices((h, w))
    cx, cy, r = float(circle[0]), float(circle[1]), float(circle[2])
    return ((xx - cx) ** 2 + (yy - cy) ** 2) <= r ** 2


def _bilinear_sample(arr: np.ndarray, x: float, y: float) -> float:
    h, w = arr.shape
    if x < 0.0 or y < 0.0 or x >= w - 1 or y >= h - 1:
        return 0.0
    x0 = int(math.floor(x))
    y0 = int(math.floor(y))
    dx = float(x - x0)
    dy = float(y - y0)
    return float(
        (1.0 - dx) * (1.0 - dy) * arr[y0, x0]
        + dx * (1.0 - dy) * arr[y0, x0 + 1]
        + (1.0 - dx) * dy * arr[y0 + 1, x0]
        + dx * dy * arr[y0 + 1, x0 + 1]
    )


def _rim_model_score_from_maps(mean_scaled: np.ndarray, ref_scaled: np.ndarray, grad_scaled: np.ndarray, circle: tuple[float, float, float], scale_px: float) -> float:
    h, w = mean_scaled.shape
    cx, cy, r = float(circle[0]), float(circle[1]), float(circle[2])
    if r <= 1.0:
        return 0.0
    delta = float(np.clip(scale_px, 3.0, max(4.0, 0.05 * r)))
    n = max(96, min(900, int(round(2.0 * math.pi * r / max(delta, 1.0)))))
    contrasts: list[float] = []
    gradients: list[float] = []
    visible = 0
    for angle in np.linspace(0.0, 2.0 * math.pi, n, endpoint=False):
        ca = float(math.cos(float(angle)))
        sa = float(math.sin(float(angle)))
        xb = cx + r * ca
        yb = cy + r * sa
        xi = cx + (r - delta) * ca
        yi = cy + (r - delta) * sa
        xo = cx + (r + delta) * ca
        yo = cy + (r + delta) * sa
        if not (1.0 <= xi < w - 2 and 1.0 <= yi < h - 2 and 1.0 <= xo < w - 2 and 1.0 <= yo < h - 2 and 1.0 <= xb < w - 2 and 1.0 <= yb < h - 2):
            continue
        visible += 1
        inside = 0.5 * (_bilinear_sample(mean_scaled, xi, yi) + _bilinear_sample(ref_scaled, xi, yi))
        outside = 0.5 * (_bilinear_sample(mean_scaled, xo, yo) + _bilinear_sample(ref_scaled, xo, yo))
        contrasts.append(abs(outside - inside))
        gradients.append(_bilinear_sample(grad_scaled, xb, yb))
    if not contrasts:
        return 0.0
    arc_fraction = float(visible / max(n, 1))
    contrast_arr = np.asarray(contrasts, dtype=np.float32)
    grad_arr = np.asarray(gradients, dtype=np.float32)
    return float(math.sqrt(max(arc_fraction, 1e-6)) * (0.45 * np.median(contrast_arr) + 0.30 * np.percentile(contrast_arr, 75) + 0.25 * np.median(grad_arr)))


def _rim_model_score(mean_gray: np.ndarray, frame0_gray: np.ndarray, circle: tuple[float, float, float], scale_px: float) -> float:
    mean_scaled = _robust_scale(mean_gray)
    ref_scaled = _robust_scale(frame0_gray)
    grad_scaled = _robust_scale(0.65 * _gradient_magnitude(mean_scaled) + 0.35 * _gradient_magnitude(ref_scaled))
    return _rim_model_score_from_maps(mean_scaled, ref_scaled, grad_scaled, circle, scale_px)


def _lattice_expected_cells_for_circle(anchor_rows: list[dict[str, object]], circle: tuple[float, float, float], shape: tuple[int, int], r_common: float) -> set[tuple[int, int]]:
    if not anchor_rows:
        return set()
    h, w = shape
    origin_ref, u_ref, v_ref, _ = _least_squares_lattice(anchor_rows)
    B = np.column_stack([u_ref, v_ref]).astype(np.float32)
    inv = np.linalg.pinv(B)
    frame_corners = np.asarray([[0.0, 0.0], [w - 1.0, 0.0], [0.0, h - 1.0], [w - 1.0, h - 1.0]], dtype=np.float32)
    cx, cy, rad = float(circle[0]), float(circle[1]), float(circle[2])
    circle_corners = np.asarray([[cx - rad, cy - rad], [cx + rad, cy - rad], [cx - rad, cy + rad], [cx + rad, cy + rad]], dtype=np.float32)
    uv_bounds = (np.vstack([frame_corners, circle_corners]) - origin_ref) @ inv.T
    pad = 2
    umin = int(math.floor(float(uv_bounds[:, 0].min()))) - pad
    umax = int(math.ceil(float(uv_bounds[:, 0].max()))) + pad
    vmin = int(math.floor(float(uv_bounds[:, 1].min()))) - pad
    vmax = int(math.ceil(float(uv_bounds[:, 1].max()))) + pad
    cells: set[tuple[int, int]] = set()
    for iu in range(umin, umax + 1):
        for iv in range(vmin, vmax + 1):
            p = origin_ref + iu * u_ref + iv * v_ref
            x, y = float(p[0]), float(p[1])
            if not (0.0 <= x < w and 0.0 <= y < h):
                continue
            if math.hypot(x - cx, y - cy) <= rad + max(1.0, 0.5 * float(r_common)):
                cells.add((int(iu), int(iv)))
    return cells


def _anchor_surface_similarity_map(mean_gray: np.ndarray, frame0_gray: np.ndarray, anchor_rows: list[dict[str, object]], r_common: float) -> np.ndarray | None:
    if len(anchor_rows) < 3:
        return None
    h, w = mean_gray.shape
    surface_mask = _anchor_surface_mask((h, w), anchor_rows, r_common)
    if int(surface_mask.sum()) < 64:
        return None
    mean_scaled = _robust_scale(mean_gray)
    ref_scaled = _robust_scale(frame0_gray)
    grad_scaled = _robust_scale(0.65 * _gradient_magnitude(mean_scaled) + 0.35 * _gradient_magnitude(ref_scaled))
    feats = np.stack([mean_scaled, ref_scaled, grad_scaled], axis=-1).astype(np.float32)
    samples = feats[surface_mask]
    med = np.median(samples, axis=0)
    mad = np.median(np.abs(samples - med[None, :]), axis=0)
    scale = np.maximum(mad, np.percentile(np.abs(samples - med[None, :]), 75, axis=0) * 0.5)
    scale = np.maximum(scale, 0.035)
    z = (feats - med[None, None, :]) / scale[None, None, :]
    dist = np.sqrt(np.mean(z * z, axis=-1))
    return (1.0 / (1.0 + dist)).astype(np.float32)


def _anchor_surface_mask(shape: tuple[int, int], anchor_rows: list[dict[str, object]], r_common: float) -> np.ndarray:
    h, w = shape
    surface_mask = np.zeros((h, w), dtype=bool)
    if len(anchor_rows) < 3:
        return surface_mask
    try:
        _origin, u_ref, v_ref, _residual = _least_squares_lattice(anchor_rows)
        lattice_spacing = min(float(np.linalg.norm(u_ref)), float(np.linalg.norm(v_ref)))
    except Exception:
        lattice_spacing = 4.0 * float(r_common)
    inner = max(float(r_common) + 2.0, 1.25 * float(r_common))
    outer = max(inner + 2.0, min(0.45 * lattice_spacing, 2.8 * float(r_common)))
    for row in anchor_rows:
        x = float(row.get('x_final', row.get('x', 0.0)))
        y = float(row.get('y_final', row.get('y', 0.0)))
        xi = int(round(x))
        yi = int(round(y))
        pad = int(math.ceil(outer)) + 1
        x0 = max(0, xi - pad)
        x1 = min(w, xi + pad + 1)
        y0 = max(0, yi - pad)
        y1 = min(h, yi + pad + 1)
        if x0 >= x1 or y0 >= y1:
            continue
        yy, xx = np.indices((y1 - y0, x1 - x0))
        d = np.sqrt((xx + x0 - x) ** 2 + (yy + y0 - y) ** 2)
        surface_mask[y0:y1, x0:x1] |= (d >= inner) & (d <= outer)
    return surface_mask


def _frame_border_mask(shape: tuple[int, int]) -> np.ndarray:
    h, w = shape
    border = max(8, int(round(0.045 * float(min(h, w)))))
    mask = np.zeros((h, w), dtype=bool)
    mask[:border, :] = True
    mask[h - border:, :] = True
    mask[:, :border] = True
    mask[:, w - border:] = True
    return mask


def _background_similarity_map(mean_gray: np.ndarray, frame0_gray: np.ndarray) -> np.ndarray | None:
    h, w = mean_gray.shape
    border_mask = _frame_border_mask((h, w))
    if int(border_mask.sum()) < 64:
        return None
    mean_scaled = _robust_scale(mean_gray)
    ref_scaled = _robust_scale(frame0_gray)
    grad_scaled = _robust_scale(0.65 * _gradient_magnitude(mean_scaled) + 0.35 * _gradient_magnitude(ref_scaled))
    feats = np.stack([mean_scaled, ref_scaled, grad_scaled], axis=-1).astype(np.float32)
    samples = feats[border_mask]
    if samples.shape[0] < 64:
        return None
    med = np.median(samples, axis=0)
    mad = np.median(np.abs(samples - med[None, :]), axis=0)
    spread = np.percentile(np.abs(samples - med[None, :]), 75, axis=0)
    scale = np.maximum(np.maximum(mad, 0.5 * spread), 0.035)
    z = (feats - med[None, None, :]) / scale[None, None, :]
    dist = np.sqrt(np.mean(z * z, axis=-1))
    similarity = (1.0 / (1.0 + dist)).astype(np.float32)
    border_self = float(np.median(similarity[border_mask]))
    interior = ~border_mask
    if int(interior.sum()) > 64:
        interior_low = float(np.percentile(similarity[interior], 15))
        if border_self - interior_low < 0.05:
            return None
    return similarity


def _background_boundary_circle_candidates(
    background_similarity: np.ndarray | None,
    anchor_rows: list[dict[str, object]],
    r_common: float,
    rough_mask: np.ndarray,
) -> list[tuple[float, float, float]]:
    if background_similarity is None:
        return []
    h, w = background_similarity.shape
    border_mask = _frame_border_mask((h, w))
    bg_vals = background_similarity[border_mask]
    if bg_vals.size < 64:
        return []
    bg_level = float(np.median(bg_vals))
    surface_mask = _anchor_surface_mask((h, w), anchor_rows, r_common)
    wafer_probe = surface_mask & rough_mask.astype(bool)
    if int(wafer_probe.sum()) < 64:
        wafer_probe = rough_mask.astype(bool) & (~border_mask)
    if int(wafer_probe.sum()) < 64:
        return []
    wafer_level = float(np.median(background_similarity[wafer_probe]))
    if bg_level <= wafer_level + 0.05:
        return []
    threshold = 0.5 * (bg_level + wafer_level)
    non_background = (background_similarity <= threshold).astype(np.uint8)
    non_background = cv2.morphologyEx(non_background, cv2.MORPH_CLOSE, np.ones((31, 31), np.uint8))
    non_background = cv2.morphologyEx(non_background, cv2.MORPH_OPEN, np.ones((9, 9), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(non_background, 8)
    if n <= 1:
        return []
    anchor_centers = [(float(r.get('x_final', r.get('x', 0.0))), float(r.get('y_final', r.get('y', 0.0)))) for r in anchor_rows]
    components: list[tuple[float, np.ndarray]] = []
    for lab in range(1, n):
        area = int(stats[lab, cv2.CC_STAT_AREA])
        if area < max(64, int(round(0.01 * h * w))):
            continue
        comp_mask = labels == lab
        anchor_hits = 0
        for x, y in anchor_centers:
            xi = int(round(x))
            yi = int(round(y))
            if 0 <= xi < w and 0 <= yi < h and comp_mask[yi, xi]:
                anchor_hits += 1
        anchor_frac = float(anchor_hits / max(len(anchor_centers), 1)) if anchor_centers else 0.0
        score = float(anchor_frac + area / max(h * w, 1))
        components.append((score, comp_mask))
    if not components:
        return []
    components.sort(key=lambda item: item[0], reverse=True)
    candidates: list[tuple[float, float, float]] = []
    for _score, comp_mask in components[:3]:
        candidates.extend(circle for _s, circle in _visible_boundary_circle_candidates(comp_mask))
        partial_fit = _fit_partial_support_circle(comp_mask)
        if partial_fit is not None:
            candidates.append(tuple(float(v) for v in partial_fit))
        contours, _ = cv2.findContours(comp_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if contours:
            (xc, yc), rr = cv2.minEnclosingCircle(max(contours, key=cv2.contourArea))
            candidates.append((float(xc), float(yc), float(rr)))
    return _dedupe_circle_candidates(candidates)


def _circle_material_exclusion_score(material_similarity: np.ndarray | None, circle: tuple[float, float, float], r_common: float, grid: tuple[np.ndarray, np.ndarray] | None = None) -> float:
    if material_similarity is None:
        return 0.5
    h, w = material_similarity.shape
    cx, cy, r = float(circle[0]), float(circle[1]), float(circle[2])
    if r <= 1.0:
        return 0.0
    if grid is None:
        yy, xx = np.indices((h, w))
    else:
        yy, xx = grid
    d = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    gap = max(2.0, 0.45 * float(r_common))
    band = max(4.0, 2.2 * float(r_common))
    inside_band = (d >= max(0.0, r - band)) & (d <= max(0.0, r - gap))
    outside_band = (d >= r + gap) & (d <= r + band)
    interior = d <= max(0.0, r - gap)
    if int(inside_band.sum()) >= 24:
        inside_match = float(np.median(material_similarity[inside_band]))
    elif int(interior.sum()) >= 24:
        inside_match = float(np.median(material_similarity[interior]))
    else:
        inside_match = 0.0
    if int(outside_band.sum()) >= 24:
        outside_reject = float(1.0 - np.median(material_similarity[outside_band]))
    else:
        outside_reject = 0.5
    return float(np.clip(0.62 * inside_match + 0.38 * outside_reject, 0.0, 1.0))


def _circle_background_exclusion_score(background_similarity: np.ndarray | None, circle: tuple[float, float, float], r_common: float, grid: tuple[np.ndarray, np.ndarray] | None = None) -> float:
    if background_similarity is None:
        return 0.5
    h, w = background_similarity.shape
    cx, cy, r = float(circle[0]), float(circle[1]), float(circle[2])
    if r <= 1.0:
        return 0.0
    if grid is None:
        yy, xx = np.indices((h, w))
    else:
        yy, xx = grid
    d = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    gap = max(2.0, 0.45 * float(r_common))
    band = max(4.0, 2.4 * float(r_common))
    inside_band = (d >= max(0.0, r - band)) & (d <= max(0.0, r - gap))
    outside_band = (d >= r + gap) & (d <= r + band)
    interior = d <= max(0.0, r - gap)
    exterior = d >= r + gap
    if int(outside_band.sum()) >= 24:
        outside_background = float(np.median(background_similarity[outside_band]))
    elif int(exterior.sum()) >= 24:
        outside_background = float(np.median(background_similarity[exterior]))
    else:
        outside_background = 0.5
    if int(inside_band.sum()) >= 24:
        inside_reject = float(1.0 - np.median(background_similarity[inside_band]))
    elif int(interior.sum()) >= 24:
        inside_reject = float(1.0 - np.median(background_similarity[interior]))
    else:
        inside_reject = 0.5
    return float(np.clip(0.58 * outside_background + 0.42 * inside_reject, 0.0, 1.0))


def _circle_background_transition_score(background_similarity: np.ndarray | None, circle: tuple[float, float, float], r_common: float) -> float:
    if background_similarity is None:
        return 0.5
    h, w = background_similarity.shape
    cx, cy, r = float(circle[0]), float(circle[1]), float(circle[2])
    if r <= 1.0:
        return 0.0
    delta = float(np.clip(0.75 * float(r_common), 3.0, max(4.0, 0.05 * r)))
    n = max(96, min(900, int(round(2.0 * math.pi * r / max(delta, 1.0)))))
    transitions: list[float] = []
    visible = 0
    for angle in np.linspace(0.0, 2.0 * math.pi, n, endpoint=False):
        ca = float(math.cos(float(angle)))
        sa = float(math.sin(float(angle)))
        xi = cx + (r - delta) * ca
        yi = cy + (r - delta) * sa
        xo = cx + (r + delta) * ca
        yo = cy + (r + delta) * sa
        if not (1.0 <= xi < w - 2 and 1.0 <= yi < h - 2 and 1.0 <= xo < w - 2 and 1.0 <= yo < h - 2):
            continue
        visible += 1
        inside = _bilinear_sample(background_similarity, xi, yi)
        outside = _bilinear_sample(background_similarity, xo, yo)
        transitions.append(max(0.0, outside - inside))
    if not transitions:
        return 0.0
    arr = np.asarray(transitions, dtype=np.float32)
    arc_fraction = float(visible / max(n, 1))
    return float(np.clip(math.sqrt(max(arc_fraction, 1e-6)) * (0.58 * np.median(arr) + 0.42 * np.percentile(arr, 80)), 0.0, 1.0))


def _fine_refine_circle_from_visible_rim(
    rim_maps: tuple[np.ndarray, np.ndarray, np.ndarray],
    background_similarity: np.ndarray | None,
    circle: tuple[float, float, float],
    r_common: float,
) -> tuple[float, float, float] | None:
    mean_scaled, ref_scaled, grad_scaled = rim_maps
    h, w = mean_scaled.shape
    cx, cy, r = float(circle[0]), float(circle[1]), float(circle[2])
    if r <= 1.0:
        return None
    span = max(4.0, min(2.0 * float(r_common), 0.10 * r))
    delta = float(np.clip(0.70 * float(r_common), 3.0, max(4.0, 0.05 * r)))
    n_angles = max(128, min(960, int(round(2.0 * math.pi * r / max(delta, 1.0)))))
    n_radial = max(13, int(round(2.0 * span)) + 1)
    offsets = np.linspace(-span, span, n_radial, dtype=np.float32)
    points: list[tuple[float, float]] = []
    weights: list[float] = []
    transition_values: list[float] = []
    for angle in np.linspace(0.0, 2.0 * math.pi, n_angles, endpoint=False):
        ca = float(math.cos(float(angle)))
        sa = float(math.sin(float(angle)))
        best: tuple[float, float, float] | None = None
        for off in offsets:
            rr = r + float(off)
            if rr <= 1.0:
                continue
            xb = cx + rr * ca
            yb = cy + rr * sa
            xi = cx + (rr - delta) * ca
            yi = cy + (rr - delta) * sa
            xo = cx + (rr + delta) * ca
            yo = cy + (rr + delta) * sa
            if not (1.0 <= xi < w - 2 and 1.0 <= yi < h - 2 and 1.0 <= xo < w - 2 and 1.0 <= yo < h - 2 and 1.0 <= xb < w - 2 and 1.0 <= yb < h - 2):
                continue
            inside = 0.5 * (_bilinear_sample(mean_scaled, xi, yi) + _bilinear_sample(ref_scaled, xi, yi))
            outside = 0.5 * (_bilinear_sample(mean_scaled, xo, yo) + _bilinear_sample(ref_scaled, xo, yo))
            contrast = abs(outside - inside)
            grad = _bilinear_sample(grad_scaled, xb, yb)
            transition = 0.0
            if background_similarity is not None:
                transition = max(0.0, _bilinear_sample(background_similarity, xo, yo) - _bilinear_sample(background_similarity, xi, yi))
            score = 0.34 * contrast + 0.32 * grad + 0.34 * transition
            if best is None or score > best[0]:
                best = (float(score), float(rr), float(transition))
        if best is None:
            continue
        score, rr, transition = best
        points.append((cx + rr * ca, cy + rr * sa))
        weights.append(score)
        transition_values.append(transition)
    if len(points) < max(24, int(0.08 * n_angles)):
        return None
    pts = np.asarray(points, dtype=np.float32)
    wts = np.asarray(weights, dtype=np.float32)
    trs = np.asarray(transition_values, dtype=np.float32)
    if background_similarity is not None and trs.size:
        transition_floor = max(0.015, float(np.percentile(trs, 55)))
        keep_transition = trs >= transition_floor
        if int(np.sum(keep_transition)) >= max(18, int(0.05 * n_angles)):
            pts = pts[keep_transition]
            wts = wts[keep_transition]
    if pts.shape[0] < 18:
        return None
    score_floor = float(np.percentile(wts, 45))
    keep_score = wts >= score_floor
    if int(np.sum(keep_score)) >= 18:
        pts = pts[keep_score]
        wts = wts[keep_score]
    fit = _fit_circle_weighted_least_squares(pts, wts)
    if fit is None:
        return None
    fx, fy, fr = fit
    radial = np.sqrt((pts[:, 0] - fx) ** 2 + (pts[:, 1] - fy) ** 2)
    resid = np.abs(radial - fr)
    med = float(np.median(resid))
    mad = float(np.median(np.abs(resid - med)))
    keep = resid <= max(2.0, med + 2.5 * max(mad, 1e-6))
    if int(np.sum(keep)) >= 18 and int(np.sum(keep)) < pts.shape[0]:
        fit2 = _fit_circle_weighted_least_squares(pts[keep], wts[keep])
        if fit2 is not None:
            fx, fy, fr = fit2
    max_center_shift = max(2.0 * float(r_common), 0.08 * r)
    max_radius_shift = max(1.35 * float(r_common), 0.07 * r)
    if math.hypot(fx - cx, fy - cy) > max_center_shift:
        return None
    if abs(fr - r) > max_radius_shift:
        return None
    return float(fx), float(fy), float(fr)


def _fine_refine_circle_by_consensus_score(
    mean_gray: np.ndarray,
    frame0_gray: np.ndarray,
    rough_mask: np.ndarray,
    anchor_rows: list[dict[str, object]],
    circle: tuple[float, float, float],
    r_common: float,
    material_similarity: np.ndarray | None,
    background_similarity: np.ndarray | None,
    grid: tuple[np.ndarray, np.ndarray],
    rim_maps: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> tuple[float, float, float]:
    best_circle = (float(circle[0]), float(circle[1]), float(circle[2]))
    best_score = _wafer_circle_model_score(mean_gray, frame0_gray, rough_mask, anchor_rows, best_circle, r_common, material_similarity, background_similarity, grid, rim_maps)
    step = max(1.0, 0.50 * float(r_common))
    min_step = 0.75
    while step >= min_step:
        improved = False
        local_best = (best_score, best_circle)
        for dx in (-step, 0.0, step):
            for dy in (-step, 0.0, step):
                for dr in (-0.5 * step, 0.0, 0.5 * step):
                    if dx == 0.0 and dy == 0.0 and dr == 0.0:
                        continue
                    cand = (best_circle[0] + dx, best_circle[1] + dy, best_circle[2] + dr)
                    if cand[2] <= 2.0 * float(r_common):
                        continue
                    score = _wafer_circle_model_score(mean_gray, frame0_gray, rough_mask, anchor_rows, cand, r_common, material_similarity, background_similarity, grid, rim_maps)
                    if score > local_best[0] + 1e-6:
                        local_best = (score, cand)
        if local_best[0] > best_score + 1e-6:
            best_score, best_circle = local_best
            improved = True
        if not improved:
            step *= 0.5
    return best_circle


def _circle_contains_visible_frame(shape: tuple[int, int], circle: tuple[float, float, float]) -> bool:
    h, w = shape
    cx, cy, r = float(circle[0]), float(circle[1]), float(circle[2])
    for x, y in ((0.0, 0.0), (float(w - 1), 0.0), (0.0, float(h - 1)), (float(w - 1), float(h - 1))):
        if math.hypot(x - cx, y - cy) > r:
            return False
    return True


def _candidate_circle_variants(seed_circles: list[tuple[float, float, float]], anchor_rows: list[dict[str, object]], shape: tuple[int, int], r_common: float) -> list[tuple[float, float, float]]:
    h, w = shape
    if anchor_rows:
        try:
            _origin, u_ref, v_ref, _residual = _least_squares_lattice(anchor_rows)
            lattice_scale = min(float(np.linalg.norm(u_ref)), float(np.linalg.norm(v_ref)))
        except Exception:
            lattice_scale = 4.0 * float(r_common)
    else:
        lattice_scale = 4.0 * float(r_common)
    step = max(float(r_common), 0.25 * lattice_scale)
    radius_steps = [0.0, -step, step, -0.5 * step, 0.5 * step]
    center_steps = [(0.0, 0.0), (-step, 0.0), (step, 0.0), (0.0, -step), (0.0, step)]
    out: list[tuple[float, float, float]] = []
    min_r = max(2.0 * float(r_common), 0.08 * float(min(h, w)))
    max_r = 2.5 * float(max(h, w))
    for cx, cy, r in seed_circles:
        for dx, dy in center_steps:
            for dr in radius_steps:
                rr = float(r + dr)
                if rr < min_r or rr > max_r:
                    continue
                out.append((float(cx + dx), float(cy + dy), rr))
    deduped: list[tuple[float, float, float]] = []
    for cx, cy, r in out:
        if any(math.hypot(cx - ox, cy - oy) < 0.05 * max(r, orad) and abs(r - orad) < 0.06 * max(r, orad) for ox, oy, orad in deduped):
            continue
        deduped.append((cx, cy, r))
    return deduped


def _dedupe_circle_candidates(candidates: list[tuple[float, float, float]]) -> list[tuple[float, float, float]]:
    deduped: list[tuple[float, float, float]] = []
    for cx, cy, r in candidates:
        if any(math.hypot(float(cx) - ox, float(cy) - oy) < 0.04 * max(float(r), orad) and abs(float(r) - orad) < 0.05 * max(float(r), orad) for ox, oy, orad in deduped):
            continue
        deduped.append((float(cx), float(cy), float(r)))
    return deduped


def _wafer_circle_model_score(
    mean_gray: np.ndarray,
    frame0_gray: np.ndarray,
    rough_mask: np.ndarray,
    anchor_rows: list[dict[str, object]],
    circle: tuple[float, float, float],
    r_common: float,
    material_similarity: np.ndarray | None = None,
    background_similarity: np.ndarray | None = None,
    grid: tuple[np.ndarray, np.ndarray] | None = None,
    rim_maps: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
) -> float:
    shape = mean_gray.shape
    h, w = shape
    cx, cy, r = float(circle[0]), float(circle[1]), float(circle[2])
    if r <= 1.0:
        return -1.0
    if grid is None:
        yy, xx = np.indices(shape)
    else:
        yy, xx = grid
    d2 = (xx - cx) ** 2 + (yy - cy) ** 2
    wafer_circle_mask = d2 <= r ** 2
    circle_area = int(wafer_circle_mask.sum())
    if circle_area <= 0:
        return -1.0
    rough_area = int(rough_mask.sum())
    if rough_area <= 0:
        overlap_score = 0.0
    else:
        overlap_score = float(np.sum(wafer_circle_mask & rough_mask) / max(rough_area, 1))
    outer = (d2 <= (r + max(2.0, float(r_common))) ** 2) & ~wafer_circle_mask
    outside_leak = float(np.sum(outer & rough_mask) / max(int(outer.sum()), 1)) if int(outer.sum()) else 0.0
    material_score = float(np.clip(overlap_score - outside_leak, 0.0, 1.0))
    surface_score = _circle_material_exclusion_score(material_similarity, (cx, cy, r), r_common, grid)
    background_score = _circle_background_exclusion_score(background_similarity, (cx, cy, r), r_common, grid)
    background_transition_score = _circle_background_transition_score(background_similarity, (cx, cy, r), r_common)

    if anchor_rows:
        contain_vals = []
        anchor_cells: set[tuple[int, int]] = set()
        for row in anchor_rows:
            x = float(row.get('x_final', row.get('x', 0.0)))
            y = float(row.get('y_final', row.get('y', 0.0)))
            margin = r - math.hypot(x - cx, y - cy)
            contain_vals.append(float(np.clip((margin + 0.5 * float(r_common)) / max(1.5 * float(r_common), 1.0), 0.0, 1.0)))
            anchor_cells.add((int(row['lattice_i']), int(row['lattice_j'])))
        containment_score = float(np.mean(contain_vals)) if contain_vals else 0.0
        expected_cells = _lattice_expected_cells_for_circle(anchor_rows, (cx, cy, r), shape, r_common)
        if expected_cells:
            lattice_score = float(len(anchor_cells & expected_cells) / max(len(expected_cells), 1))
        else:
            lattice_score = 0.0
    else:
        containment_score = 0.0
        lattice_score = 0.0

    if rim_maps is None:
        rim_score = _rim_model_score(mean_gray, frame0_gray, (cx, cy, r), max(2.0, float(r_common)))
    else:
        rim_score = _rim_model_score_from_maps(rim_maps[0], rim_maps[1], rim_maps[2], (cx, cy, r), max(2.0, float(r_common)))
    rim_component = float(1.0 - math.exp(-5.0 * max(rim_score, 0.0)))
    frame_swallow_penalty = 0.08 if _circle_contains_visible_frame(shape, (cx, cy, r)) and rough_area < 0.92 * h * w else 0.0
    return float(
        0.20 * rim_component
        + 0.20 * containment_score
        + 0.15 * lattice_score
        + 0.16 * surface_score
        + 0.13 * background_score
        + 0.12 * background_transition_score
        + 0.04 * material_score
        - frame_swallow_penalty
    )


def _refine_support_circle_from_lattice_model(
    mean_gray: np.ndarray,
    frame0_gray: np.ndarray,
    rough_mask: np.ndarray,
    support_circle: tuple[int, int, int],
    anchor_rows: list[dict[str, object]],
    r_common: float,
    *,
    allow_downscale: bool = True,
    watchdog: _DetectorWatchdog | None = None,
) -> tuple[tuple[int, int, int], np.ndarray] | None:
    if len(anchor_rows) < 8:
        return None
    if watchdog is not None:
        watchdog.check("wafer support circle consensus start")
    h, w = mean_gray.shape
    max_dim = max(h, w)
    if allow_downscale and max_dim > 620:
        if watchdog is not None:
            watchdog.check("wafer support circle downscale")
        target_dim = 560.0
        scale = float(max_dim) / target_dim
        small_shape = (max(32, int(round(float(h) / scale))), max(32, int(round(float(w) / scale))))
        small_mean = cv2.resize(mean_gray.astype(np.float32), (small_shape[1], small_shape[0]), interpolation=cv2.INTER_AREA)
        small_frame0 = cv2.resize(frame0_gray.astype(np.float32), (small_shape[1], small_shape[0]), interpolation=cv2.INTER_AREA)
        small_mask = _resize_bool_mask(rough_mask, small_shape)
        small_circle = (
            int(round(float(support_circle[0]) / scale)),
            int(round(float(support_circle[1]) / scale)),
            max(1, int(round(float(support_circle[2]) / scale))),
        )
        small_anchors = _scale_geometry_rows(anchor_rows, scale)
        small_result = _refine_support_circle_from_lattice_model(
            small_mean,
            small_frame0,
            small_mask,
            small_circle,
            small_anchors,
            max(1.0, float(r_common) / scale),
            allow_downscale=False,
            watchdog=watchdog,
        )
        if watchdog is not None:
            watchdog.check("wafer support circle downscale complete")
        if small_result is not None:
            (scx, scy, sr), _small_refined_mask = small_result
            cx, cy, r = float(scx) * scale, float(scy) * scale, float(sr) * scale
            background_similarity = _background_similarity_map(mean_gray, frame0_gray)
            rim_mean = _robust_scale(mean_gray)
            rim_ref = _robust_scale(frame0_gray)
            rim_grad = _robust_scale(0.65 * _gradient_magnitude(rim_mean) + 0.35 * _gradient_magnitude(rim_ref))
            rim_maps = (rim_mean, rim_ref, rim_grad)
            if watchdog is not None:
                watchdog.check("wafer support circle full-resolution rim polish")
            rim_fit = _fine_refine_circle_from_visible_rim(rim_maps, background_similarity, (cx, cy, r), r_common)
            if rim_fit is not None:
                cur_transition = _circle_background_transition_score(background_similarity, (cx, cy, r), r_common)
                rim_transition = _circle_background_transition_score(background_similarity, rim_fit, r_common)
                if rim_transition >= cur_transition - 0.010:
                    cx, cy, r = rim_fit
            refined_circle = (int(round(cx)), int(round(cy)), int(round(r)))
            return refined_circle, _circle_mask(mean_gray.shape, refined_circle)
    candidates: list[tuple[float, float, float]] = [
        (float(support_circle[0]), float(support_circle[1]), float(support_circle[2])),
    ]
    candidates.extend(circle for _score, circle in _visible_boundary_circle_candidates(rough_mask))
    origin_ref, u_ref, v_ref, _ = _least_squares_lattice(anchor_rows)
    anchor_pts = np.asarray([[float(r['x_final']), float(r['y_final'])] for r in anchor_rows], dtype=np.float32)
    if len(anchor_pts) >= 3:
        enclosing = cv2.minEnclosingCircle(anchor_pts)
        (hx, hy), hr = enclosing
        lattice_scale = max(float(np.linalg.norm(u_ref)), float(np.linalg.norm(v_ref)), float(r_common))
        for margin in (float(r_common), 0.35 * lattice_scale, 0.55 * lattice_scale, 0.85 * lattice_scale, 1.10 * lattice_scale):
            candidates.append((float(hx), float(hy), float(hr + margin)))
        hull = cv2.convexHull(anchor_pts.reshape(-1, 1, 2)).reshape(-1, 2)
        if hull.shape[0] >= 3:
            hull_fit = _fit_circle_least_squares(hull.astype(np.float32))
            if hull_fit is not None:
                hcx, hcy, hr_fit = hull_fit
                for margin in (float(r_common), 0.45 * lattice_scale, 0.75 * lattice_scale):
                    candidates.append((float(hcx), float(hcy), float(hr_fit + margin)))
    if watchdog is not None:
        watchdog.check("wafer support circle material/background maps")
    material_similarity = _anchor_surface_similarity_map(mean_gray, frame0_gray, anchor_rows, r_common)
    background_similarity = _background_similarity_map(mean_gray, frame0_gray)
    candidates.extend(_background_boundary_circle_candidates(background_similarity, anchor_rows, r_common, rough_mask))
    score_grid = tuple(np.indices(mean_gray.shape))
    rim_mean = _robust_scale(mean_gray)
    rim_ref = _robust_scale(frame0_gray)
    rim_grad = _robust_scale(0.65 * _gradient_magnitude(rim_mean) + 0.35 * _gradient_magnitude(rim_ref))
    rim_maps = (rim_mean, rim_ref, rim_grad)
    candidates = _dedupe_circle_candidates(candidates)
    best: tuple[float, tuple[float, float, float]] | None = None
    scored_seeds: list[tuple[float, tuple[float, float, float]]] = []
    for circle_i, circle in enumerate(candidates):
        if watchdog is not None and circle_i % 12 == 0:
            watchdog.check("wafer support circle seed scoring")
        score = _wafer_circle_model_score(mean_gray, frame0_gray, rough_mask, anchor_rows, circle, r_common, material_similarity, background_similarity, score_grid, rim_maps)
        scored_seeds.append((score, circle))
        if best is None or score > best[0]:
            best = (score, circle)
    top_seed_circles = [circle for _score, circle in sorted(scored_seeds, key=lambda item: item[0], reverse=True)[:4]]
    for circle_i, circle in enumerate(_candidate_circle_variants(top_seed_circles, anchor_rows, mean_gray.shape, r_common)):
        if watchdog is not None and circle_i % 12 == 0:
            watchdog.check("wafer support circle variant scoring")
        score = _wafer_circle_model_score(mean_gray, frame0_gray, rough_mask, anchor_rows, circle, r_common, material_similarity, background_similarity, score_grid, rim_maps)
        if best is None or score > best[0]:
            best = (score, circle)
    if best is None:
        return None
    _score, (cx, cy, r) = best
    lattice_scale = max(float(np.linalg.norm(u_ref)), float(np.linalg.norm(v_ref)), float(r_common))
    for step in (0.35 * lattice_scale, 0.18 * lattice_scale):
        if watchdog is not None:
            watchdog.check("wafer support circle local search")
        local_best = best
        moves = (
            (0.0, 0.0, 0.0),
            (-step, 0.0, 0.0),
            (step, 0.0, 0.0),
            (0.0, -step, 0.0),
            (0.0, step, 0.0),
            (0.0, 0.0, -step),
            (0.0, 0.0, step),
        )
        for dx, dy, dr in moves:
            rr = r + dr
            if rr <= 2.0 * float(r_common):
                continue
            circle = (cx + dx, cy + dy, rr)
            score = _wafer_circle_model_score(mean_gray, frame0_gray, rough_mask, anchor_rows, circle, r_common, material_similarity, background_similarity, score_grid, rim_maps)
            if local_best is None or score > local_best[0]:
                local_best = (score, circle)
        best = local_best
        _score, (cx, cy, r) = best
    if watchdog is not None:
        watchdog.check("wafer support circle rim refinement")
    rim_fit = _fine_refine_circle_from_visible_rim(rim_maps, background_similarity, (cx, cy, r), r_common)
    if rim_fit is not None:
        rim_score = _wafer_circle_model_score(mean_gray, frame0_gray, rough_mask, anchor_rows, rim_fit, r_common, material_similarity, background_similarity, score_grid, rim_maps)
        cur_score = _wafer_circle_model_score(mean_gray, frame0_gray, rough_mask, anchor_rows, (cx, cy, r), r_common, material_similarity, background_similarity, score_grid, rim_maps)
        rim_transition = _circle_background_transition_score(background_similarity, rim_fit, r_common)
        cur_transition = _circle_background_transition_score(background_similarity, (cx, cy, r), r_common)
        if rim_score >= cur_score - 0.015 and rim_transition >= cur_transition - 0.005:
            cx, cy, r = rim_fit
    if watchdog is not None:
        watchdog.check("wafer support circle consensus polish")
    cx, cy, r = _fine_refine_circle_by_consensus_score(
        mean_gray,
        frame0_gray,
        rough_mask,
        anchor_rows,
        (cx, cy, r),
        r_common,
        material_similarity,
        background_similarity,
        score_grid,
        rim_maps,
    )
    if watchdog is not None:
        watchdog.check("wafer support circle consensus complete")
    refined_circle = (int(round(cx)), int(round(cy)), int(round(r)))
    refined_mask = _circle_mask(mean_gray.shape, refined_circle)
    return refined_circle, refined_mask


def _pattern_extended_prediction(anchor_rows: list[dict[str, object]], strong_rows: list[dict[str, object]], wafer_circle: tuple[int, int, int], r_common: int):
    xc, yc, wr = float(wafer_circle[0]), float(wafer_circle[1]), float(wafer_circle[2])
    if not anchor_rows:
        return []
    origin_ref, u_ref, v_ref, _ = _least_squares_lattice(anchor_rows)
    imin = min(int(r['lattice_i']) for r in anchor_rows) - 3
    imax = max(int(r['lattice_i']) for r in anchor_rows) + 3
    jmin = min(int(r['lattice_j']) for r in anchor_rows) - 3
    jmax = max(int(r['lattice_j']) for r in anchor_rows) + 3
    anchor_map = {(int(r['lattice_i']), int(r['lattice_j'])): r for r in anchor_rows}
    strong_map = {(int(r['lattice_i']), int(r['lattice_j'])): r for r in strong_rows}
    out = []
    for i in range(imin, imax + 1):
        for j in range(jmin, jmax + 1):
            x = float(origin_ref[0] + u_ref[0] * i + v_ref[0] * j)
            y = float(origin_ref[1] + u_ref[1] * i + v_ref[1] * j)
            d = float(math.hypot(x - xc, y - yc))
            if d > wr + r_common + 2:
                continue
            geometry_class = 'full' if d <= wr - r_common - 1 else 'partial'
            if (i, j) in anchor_map:
                rr = anchor_map[(i, j)]
                out.append({'lattice_i': i, 'lattice_j': j, 'x': float(rr['x_final']), 'y': float(rr['y_final']), 'r': float(rr['r_final']), 'kind': 'anchor', 'geometry_class': geometry_class})
            elif (i, j) in strong_map:
                rr = strong_map[(i, j)]
                out.append({'lattice_i': i, 'lattice_j': j, 'x': float(rr['x_rec']), 'y': float(rr['y_rec']), 'r': float(rr['r_rec']), 'kind': 'recovered_strong', 'geometry_class': geometry_class})
            else:
                out.append({'lattice_i': i, 'lattice_j': j, 'x': x, 'y': y, 'r': float(r_common), 'kind': 'predicted_only', 'geometry_class': geometry_class})
    return out


def _lattice_from_anchor_rows(anchor_rows: list[dict[str, object]]) -> LatticeModel:
    if not anchor_rows:
        return LatticeModel(0.0, 0.0, (1.0, 0.0), (0.0, 1.0), 0.0, 1.0, 1.0, 0.0)
    origin_ref, u_ref, v_ref, residual = _least_squares_lattice(anchor_rows)
    spacing_u = float(np.linalg.norm(u_ref))
    spacing_v = float(np.linalg.norm(v_ref))
    angle = float(math.degrees(math.atan2(float(u_ref[1]), float(u_ref[0]))))
    spacing_ref = max(min(spacing_u, spacing_v), 1e-6)
    conf = float(np.clip(1.0 / (1.0 + residual / spacing_ref), 0.0, 1.0))
    return LatticeModel(float(origin_ref[0]), float(origin_ref[1]), (float(u_ref[0]), float(u_ref[1])), (float(v_ref[0]), float(v_ref[1])), angle, spacing_u, spacing_v, conf)


def _visible_frame_support(shape: tuple[int, int]) -> tuple[tuple[int, int, int], np.ndarray]:
    h, w = shape
    cx = 0.5 * float(w - 1)
    cy = 0.5 * float(h - 1)
    corners = np.asarray(
        [[0.0, 0.0], [float(w - 1), 0.0], [0.0, float(h - 1)], [float(w - 1), float(h - 1)]],
        dtype=np.float32,
    )
    radius = int(math.ceil(float(np.max(np.linalg.norm(corners - np.array([cx, cy], dtype=np.float32), axis=1)))))
    return (int(round(cx)), int(round(cy)), max(1, radius)), np.ones((h, w), dtype=bool)


def _detect_exact_wafer_holes_with_support(
    mean_gray: np.ndarray,
    std_gray: np.ndarray,
    mad_gray: np.ndarray,
    reference_gray: np.ndarray,
    support_circle: tuple[int, int, int],
    wafer_mask: np.ndarray,
    cfg: GeometryConfig,
    *,
    mode: str,
    refine_support: bool,
    progress_callback: Callable[[int, int, str], None] | None = None,
    progress_offset: int = 0,
    progress_total: int = 14,
    watchdog: _DetectorWatchdog | None = None,
) -> StableGridDetectionResult:
    def report(local_step: int, message: str) -> None:
        if progress_callback is not None:
            progress_callback(min(progress_total, int(progress_offset) + int(local_step)), progress_total, message)

    wafer_mask = wafer_mask.astype(bool, copy=False)
    if watchdog is not None:
        watchdog.check(f"{mode} evidence maps")
    evidence = _build_evidence_maps(mean_gray, std_gray, mad_gray, reference_gray, wafer_mask, cfg)
    seeds = _extract_consensus_seeds(evidence, wafer_mask, cfg)
    report(1, f"Built hole evidence maps ({mode})")
    initial_accepted: list[dict[str, float | str]]
    rim_rows: list[dict[str, float | str]] = []
    if _support_mask_is_rectangular_content(wafer_mask):
        if watchdog is not None:
            watchdog.check(f"{mode} texture-void rim candidates")
        void_rows = _connected_texture_void_candidates(evidence, wafer_mask, cfg)
        hough_rows = _hough_round_rim_candidates(evidence, wafer_mask, cfg) if len(void_rows) < 8 else []
        if len(void_rows) >= 8 and len(hough_rows) >= 8:
            void_quality = float(len(void_rows)) - 12.0 * min(_candidate_basis_residual(void_rows), 1.0)
            hough_quality = float(len(hough_rows)) - 12.0 * min(_candidate_basis_residual(hough_rows), 1.0)
            rim_rows = void_rows if void_quality >= hough_quality else hough_rows
        elif len(void_rows) >= 8:
            rim_rows = void_rows
        else:
            rim_rows = hough_rows
    if len(rim_rows) >= 8 and _estimate_probe_basis(rim_rows) is not None:
        initial_accepted = rim_rows
        report(2, f"Selected texture-void rim candidates ({len(initial_accepted)} kept)")
        if watchdog is not None:
            watchdog.check(f"{mode} texture-void rim candidates complete")
    else:
        if watchdog is not None:
            watchdog.check(f"{mode} initial candidate fitting")
        initial_accepted, _ = _initial_fitted_candidates(evidence, seeds, cfg)
        initial_accepted = _select_common_radius_rows(initial_accepted)
        if len(rim_rows) >= 8:
            rim_residual = _candidate_basis_residual(rim_rows)
            initial_residual = _candidate_basis_residual(initial_accepted)
            if rim_residual <= initial_residual + 0.08:
                initial_accepted = rim_rows
        report(2, f"Fitted initial hole candidates ({len(initial_accepted)} kept)")
        if watchdog is not None:
            watchdog.check(f"{mode} initial candidate fitting complete")

    raw_count = int(max(len(seeds), len(rim_rows)))
    filtered_count = int(len(initial_accepted))
    if len(initial_accepted) < 8:
        accepted_candidates = [HoleCandidate(float(r['x']), float(r['y']), float(r['r']), 0.0, float(r['fit_score']), float(r['confidence'])) for r in initial_accepted]
        lattice = _lattice_from_anchor_rows([])
        lattice_indices = {i: (i, 0) for i in range(len(accepted_candidates))}
        tiers = [{'tier': 'anchor', 'cell_u': int(i), 'cell_v': 0, 'x': float(c.x), 'y': float(c.y), 'radius_px': float(c.radius_px), 'confidence': float(c.confidence)} for i, c in enumerate(accepted_candidates)]
        debug = StableGridDetectionDebug(support_circle, wafer_mask, raw_count, filtered_count, len(accepted_candidates), 0, 0, 0, len(accepted_candidates), mode, float(np.median([c.radius_px for c in accepted_candidates])) if accepted_candidates else 0.0, tiers=tiers, predicted_only=[])
        return StableGridDetectionResult(accepted_candidates, lattice, lattice_indices, debug)

    if watchdog is not None:
        watchdog.check(f"{mode} lattice basis estimation")
    basis = _estimate_consensus_probe_basis(initial_accepted, evidence, wafer_mask)
    report(3, "Estimated consensus lattice basis")
    if basis is None:
        accepted_candidates = [HoleCandidate(float(r['x']), float(r['y']), float(r['r']), 0.0, float(r['fit_score']), float(r['confidence'])) for r in initial_accepted]
        lattice = _lattice_from_anchor_rows([])
        lattice_indices = {i: (i, 0) for i in range(len(accepted_candidates))}
        tiers = [{'tier': 'anchor', 'cell_u': int(i), 'cell_v': 0, 'x': float(c.x), 'y': float(c.y), 'radius_px': float(c.radius_px), 'confidence': float(c.confidence)} for i, c in enumerate(accepted_candidates)]
        debug = StableGridDetectionDebug(support_circle, wafer_mask, raw_count, filtered_count, len(accepted_candidates), 0, 0, 0, len(accepted_candidates), mode, float(np.median([c.radius_px for c in accepted_candidates])) if accepted_candidates else 0.0, tiers=tiers, predicted_only=[])
        return StableGridDetectionResult(accepted_candidates, lattice, lattice_indices, debug)

    _, basis_u, basis_v, origin = basis
    B = np.column_stack([basis_u, basis_v]).astype(np.float32)
    inv = np.linalg.pinv(B)
    present_map: dict[tuple[int, int], dict[str, float | str]] = {}
    for row in sorted(initial_accepted, key=lambda r: float(r['confidence']), reverse=True):
        uv = (np.array([float(row['x']), float(row['y'])], dtype=np.float32) - origin) @ inv.T
        cell = (int(round(float(uv[0]))), int(round(float(uv[1]))))
        if cell not in present_map:
            present_map[cell] = row
    xc, yc, wr = float(support_circle[0]), float(support_circle[1]), float(support_circle[2])
    r_common0 = float(np.median([float(r['r']) for r in initial_accepted])) if initial_accepted else 13.0
    expected_nodes = _enumerate_expected_nodes(origin, basis_u, basis_v, set(present_map.keys()), xc, yc, wr, r_common0, pad=2)
    max_physical_nodes = _max_nonoverlapping_holes_in_mask(wafer_mask, r_common0)
    if max_physical_nodes > 0 and len(expected_nodes) > max(8, max_physical_nodes):
        accepted_candidates = [HoleCandidate(float(r['x']), float(r['y']), float(r['r']), 0.0, float(r['fit_score']), float(r['confidence'])) for r in initial_accepted]
        lattice = _lattice_from_anchor_rows([])
        lattice_indices = {i: (i, 0) for i in range(len(accepted_candidates))}
        tiers = [{'tier': 'anchor', 'cell_u': int(i), 'cell_v': 0, 'x': float(c.x), 'y': float(c.y), 'radius_px': float(c.radius_px), 'confidence': float(c.confidence)} for i, c in enumerate(accepted_candidates)]
        debug = StableGridDetectionDebug(
            support_circle,
            wafer_mask,
            raw_count,
            filtered_count,
            len(accepted_candidates),
            0,
            0,
            0,
            len(accepted_candidates),
            f'{mode}_physical_guard',
            float(r_common0),
            tiers=tiers,
            predicted_only=[],
            watchdog_events=[{
                'reason': 'expected_lattice_nodes_exceed_physical_nonoverlap_limit',
                'expected_nodes': int(len(expected_nodes)),
                'max_physical_nodes': int(max_physical_nodes),
                'hole_radius_px': float(r_common0),
            }],
        )
        return StableGridDetectionResult(accepted_candidates, lattice, lattice_indices, debug)
    for node in expected_nodes:
        present = present_map.get((int(node['lattice_i']), int(node['lattice_j'])))
        if present is not None:
            node['x_seed'] = float(present['x'])
            node['y_seed'] = float(present['y'])
    anchor_rows_pre, _ = _common_radius_anchor_stage(expected_nodes, list(present_map.values()), mean_gray, reference_gray, std_gray, mad_gray, wafer_mask, watchdog=watchdog)
    anchor_rows, r_common = _border_refine_anchors(anchor_rows_pre, mean_gray, reference_gray, wafer_mask, watchdog=watchdog)
    report(4, f"Refined lattice anchors ({len(anchor_rows)} anchors)")
    if r_common <= 0:
        r_common = int(round(r_common0))

    refined_support = None
    if refine_support and not _support_mask_is_rectangular_content(wafer_mask) and not _circle_center_outside_frame(mean_gray.shape, support_circle):
        refined_support = _refine_support_circle_from_lattice_model(mean_gray, reference_gray, wafer_mask, support_circle, anchor_rows, float(r_common), watchdog=watchdog)
        report(5, "Scored wafer support consensus circle")
    elif refine_support:
        report(5, "Using physical support without circular polish")
    if refined_support is not None:
        support_circle, wafer_mask = refined_support
        xc, yc, wr = float(support_circle[0]), float(support_circle[1]), float(support_circle[2])
        anchor_rows = [
            row for row in anchor_rows
            if math.hypot(float(row['x_final']) - xc, float(row['y_final']) - yc) <= wr + max(1.0, 0.25 * float(r_common))
        ]
        expected_nodes = _enumerate_expected_nodes(origin, basis_u, basis_v, set(present_map.keys()), xc, yc, wr, float(r_common), pad=2)
    strong_rows, weak_rows, still_rows, _, _, _, _ = _recover_strong_missing(anchor_rows, expected_nodes, mean_gray, reference_gray, support_circle, int(round(r_common)), wafer_mask, watchdog=watchdog)
    report(6, f"Recovered missing lattice holes ({len(strong_rows)} strong)")
    if _support_mask_is_rectangular_content(wafer_mask):
        report(7, "Kept texture-void rim radii for rectangular support")
    else:
        anchor_rows, strong_rows, r_common = _refine_rows_to_outer_contours(
            anchor_rows,
            strong_rows,
            mean_gray,
            reference_gray,
            wafer_mask,
            int(round(r_common)),
            watchdog=watchdog,
        )
        report(7, "Aligned hole contours to outer rims")
    pattern_rows = _pattern_extended_prediction(anchor_rows, strong_rows, support_circle, int(round(r_common)))
    predicted_only_rows = [r for r in pattern_rows if str(r['kind']) == 'predicted_only']
    report(8, "Finalized lattice support pattern")

    accepted_candidates: list[HoleCandidate] = []
    lattice_indices: dict[int, tuple[int, int]] = {}
    tiers: list[dict[str, object]] = []
    idx = 0
    for row in anchor_rows:
        cand = HoleCandidate(float(row['x_final']), float(row['y_final']), float(row['r_final']), 0.0, float(row.get('score_final', row.get('score_total', 0.0))), float(row.get('confidence', 1.0)))
        accepted_candidates.append(cand)
        lattice_indices[idx] = (int(row['lattice_i']), int(row['lattice_j']))
        tiers.append({'tier': 'anchor', 'cell_u': int(row['lattice_i']), 'cell_v': int(row['lattice_j']), 'x': float(row['x_final']), 'y': float(row['y_final']), 'radius_px': float(row['r_final']), 'confidence': float(row.get('confidence', 1.0))})
        idx += 1
    weak_thr = 0.0
    strong_thr = 1.0
    if strong_rows or weak_rows:
        scores = [float(r['score_local']) for r in strong_rows + weak_rows + still_rows]
        if scores:
            weak_thr = min(scores)
            strong_thr = max(scores)
    for row in strong_rows:
        conf = float(np.clip((float(row['score_local']) - weak_thr) / max(strong_thr - weak_thr, 1e-6), 0.0, 1.0))
        cand = HoleCandidate(float(row['x_rec']), float(row['y_rec']), float(row['r_rec']), 0.0, float(row['score_local']), conf)
        accepted_candidates.append(cand)
        lattice_indices[idx] = (int(row['lattice_i']), int(row['lattice_j']))
        tiers.append({'tier': 'recovered_strong', 'cell_u': int(row['lattice_i']), 'cell_v': int(row['lattice_j']), 'x': float(row['x_rec']), 'y': float(row['y_rec']), 'radius_px': float(row['r_rec']), 'confidence': conf})
        idx += 1

    lattice = _lattice_from_anchor_rows(anchor_rows)
    predicted_only_full = int(sum(1 for r in predicted_only_rows if str(r['geometry_class']) == 'full'))
    predicted_only_partial = int(sum(1 for r in predicted_only_rows if str(r['geometry_class']) == 'partial'))
    debug = StableGridDetectionDebug(
        support_circle,
        wafer_mask,
        raw_count,
        filtered_count,
        len(anchor_rows),
        len(strong_rows),
        predicted_only_full,
        predicted_only_partial,
        len(accepted_candidates),
        mode,
        float(r_common),
        tiers=tiers,
        predicted_only=[{'cell_u': int(r['lattice_i']), 'cell_v': int(r['lattice_j']), 'x_pred': float(r['x']), 'y_pred': float(r['y']), 'geometry_class': str(r['geometry_class']), 'status': 'predicted_only'} for r in predicted_only_rows],
    )
    return StableGridDetectionResult(accepted_candidates, lattice, lattice_indices, debug)


def _large_frame_support_fallback(
    mean_gray: np.ndarray,
    std_gray: np.ndarray,
    mad_gray: np.ndarray,
    reference_gray: np.ndarray,
    cfg: GeometryConfig,
    *,
    progress_callback: Callable[[int, int, str], None] | None,
    progress_offset: int,
    progress_total: int,
    watchdog: _DetectorWatchdog | None = None,
) -> StableGridDetectionResult:
    h, w = mean_gray.shape
    max_dim = max(h, w)
    target_dim = 900.0
    if max_dim <= target_dim:
        if watchdog is not None:
            watchdog.check("full-frame support fallback start")
        frame_circle, frame_mask = _visible_frame_support(mean_gray.shape)
        return _detect_exact_wafer_holes_with_support(
            mean_gray,
            std_gray,
            mad_gray,
            reference_gray,
            frame_circle,
            frame_mask,
            cfg,
            mode='exact_sequence_frame_support',
            refine_support=False,
            progress_callback=progress_callback,
            progress_offset=progress_offset,
            progress_total=progress_total,
            watchdog=watchdog,
        )
    if watchdog is not None:
        watchdog.check("large-frame reduced fallback resize")
    scale = float(max_dim) / float(target_dim)
    small_shape = (max(32, int(round(float(h) / scale))), max(32, int(round(float(w) / scale))))

    def resize_map(arr: np.ndarray) -> np.ndarray:
        return cv2.resize(arr.astype(np.float32), (small_shape[1], small_shape[0]), interpolation=cv2.INTER_AREA).astype(np.float32)

    small_mean = resize_map(mean_gray)
    small_std = resize_map(std_gray)
    small_mad = resize_map(mad_gray)
    small_reference = resize_map(reference_gray)
    small_circle, small_mask = _visible_frame_support(small_mean.shape)
    if watchdog is not None:
        watchdog.check("large-frame reduced fallback detector")
    small_result = _detect_exact_wafer_holes_with_support(
        small_mean,
        small_std,
        small_mad,
        small_reference,
        small_circle,
        small_mask,
        _scale_geometry_cfg(cfg, scale),
        mode='exact_sequence_frame_support_reduced',
        refine_support=False,
        progress_callback=progress_callback,
        progress_offset=progress_offset,
        progress_total=progress_total,
        watchdog=watchdog,
    )
    if watchdog is not None:
        watchdog.check("large-frame reduced fallback scale result")
    return _scale_detection_result(small_result, scale, mean_gray.shape)


def _large_frame_exact_primary(
    mean_gray: np.ndarray,
    std_gray: np.ndarray,
    mad_gray: np.ndarray,
    reference_gray: np.ndarray,
    support_circle: tuple[int, int, int],
    wafer_mask: np.ndarray,
    cfg: GeometryConfig,
    *,
    progress_callback: Callable[[int, int, str], None] | None,
    progress_offset: int,
    progress_total: int,
    watchdog: _DetectorWatchdog | None = None,
) -> StableGridDetectionResult:
    h, w = mean_gray.shape
    max_dim = max(h, w)
    target_dim = 640.0 if _support_mask_is_rectangular_content(wafer_mask) else 900.0
    if max_dim <= target_dim:
        return _detect_exact_wafer_holes_with_support(
            mean_gray,
            std_gray,
            mad_gray,
            reference_gray,
            support_circle,
            wafer_mask,
            cfg,
            mode='exact_sequence',
            refine_support=True,
            progress_callback=progress_callback,
            progress_offset=progress_offset,
            progress_total=progress_total,
            watchdog=watchdog,
        )
    if watchdog is not None:
        watchdog.check("large-frame primary exact detector resize")
    scale = float(max_dim) / float(target_dim)
    small_shape = (max(32, int(round(float(h) / scale))), max(32, int(round(float(w) / scale))))

    def resize_map(arr: np.ndarray) -> np.ndarray:
        return cv2.resize(arr.astype(np.float32), (small_shape[1], small_shape[0]), interpolation=cv2.INTER_AREA).astype(np.float32)

    small_support_circle = (
        int(round(float(support_circle[0]) / scale)),
        int(round(float(support_circle[1]) / scale)),
        max(1, int(round(float(support_circle[2]) / scale))),
    )
    small_result = _detect_exact_wafer_holes_with_support(
        resize_map(mean_gray),
        resize_map(std_gray),
        resize_map(mad_gray),
        resize_map(reference_gray),
        small_support_circle,
        _resize_bool_mask(wafer_mask, small_shape),
        _scale_geometry_cfg(cfg, scale),
        mode='exact_sequence_reduced',
        refine_support=True,
        progress_callback=progress_callback,
        progress_offset=progress_offset,
        progress_total=progress_total,
        watchdog=watchdog,
    )
    if watchdog is not None:
        watchdog.check("large-frame primary exact detector scale result")
    return _scale_detection_result(small_result, scale, mean_gray.shape)


def _detection_quality_key(result: StableGridDetectionResult) -> tuple[int, int, int, float]:
    coherent = int(result.debug.anchor_count >= 8 and result.lattice.confidence >= 0.85)
    return coherent, int(round(1000.0 * float(result.lattice.confidence))), int(result.debug.anchor_count), float(result.debug.completed_count)


def detect_exact_wafer_holes_sequence_full(
    images: Sequence[np.ndarray],
    cfg: GeometryConfig,
    reference_index: int = 0,
    progress_callback: Callable[[int, int, str], None] | None = None,
) -> StableGridDetectionResult:
    progress_total = 22

    def report(step: int, message: str) -> None:
        if progress_callback is not None:
            progress_callback(min(progress_total, int(step)), progress_total, message)

    n_source_frames = int(len(images))
    if n_source_frames == 0:
        debug = StableGridDetectionDebug(None, None, 0, 0, 0, 0, 0, 0, 0, 'empty')
        return StableGridDetectionResult([], _lattice_from_anchor_rows([]), {}, debug)

    reference_index = int(np.clip(reference_index, 0, n_source_frames - 1))
    total_watchdog = _DetectorWatchdog("exact sequence detector", float(getattr(cfg, "detector_watchdog_s", 0.0) or 0.0))
    sample_indices, sample_history = _adaptive_sequence_indices(images, reference_index, cfg, watchdog=total_watchdog)

    def branch_watchdog(label: str) -> _DetectorWatchdog:
        branch_limit = float(getattr(cfg, "detector_branch_watchdog_s", 0.0) or 0.0)
        total_limit = float(getattr(cfg, "detector_watchdog_s", 0.0) or 0.0)
        if branch_limit > 0.0 and total_limit > 0.0:
            branch_limit = min(branch_limit, total_limit)
        elif branch_limit <= 0.0:
            branch_limit = total_limit
        return _DetectorWatchdog(label, branch_limit)

    def timeout_event(exc: DetectorWatchdogTimeout) -> dict[str, object]:
        return {
            "label": str(exc.label),
            "phase": str(exc.phase),
            "elapsed_s": float(exc.elapsed_s),
            "limit_s": float(exc.limit_s),
            "reason": "watchdog_timeout",
        }

    def run_on_sample(sample_idx: list[int], emit_progress: bool) -> StableGridDetectionResult:
        total_watchdog.check("sample detector start")
        sampled_images = [images[int(i)] for i in sample_idx]
        sampled_reference_index = sample_idx.index(reference_index) if reference_index in sample_idx else 0
        gray_stack = _sequence_gray_stack(sampled_images)
        if emit_progress:
            report(1, f"Built grayscale frame stack from {len(sample_idx)}/{n_source_frames} representative frames")
        reference_gray = gray_stack[sampled_reference_index]
        support_circle, wafer_mask, mean_gray, std_gray, mad_gray = _detect_support_from_sequence(gray_stack)
        if emit_progress:
            report(2, "Detected temporal wafer support")
        if support_circle is None or not np.any(wafer_mask):
            support_circle, wafer_mask = _visible_frame_support(mean_gray.shape)
            if emit_progress:
                report(3, "Using full-frame wafer support fallback")
        elif emit_progress:
            report(3, "Using detected wafer support")

        detector_callback = progress_callback if emit_progress else None
        primary_watchdog = branch_watchdog("primary exact detector")
        primary = _large_frame_exact_primary(
            mean_gray,
            std_gray,
            mad_gray,
            reference_gray,
            support_circle,
            wafer_mask,
            cfg,
            progress_callback=detector_callback,
            progress_offset=3,
            progress_total=progress_total,
            watchdog=primary_watchdog,
        )
        if emit_progress:
            report(12, "Assessed primary wafer/lattice geometry")
        total_watchdog.check("primary exact detector complete")
        if primary.debug.completed_count >= 8 and primary.lattice.confidence >= 0.85:
            return primary
        if (
            _circle_center_outside_frame(mean_gray.shape, support_circle)
            and primary.debug.anchor_count >= 10
            and primary.debug.completed_count >= 16
            and primary.lattice.confidence >= 0.75
        ):
            return primary
        if max(mean_gray.shape) > 1400 and primary.debug.completed_count >= 8 and primary.lattice.confidence >= 0.65:
            if emit_progress:
                report(13, "Skipping large-frame fallback because primary lattice is usable")
            return primary

        if emit_progress:
            report(13, "Trying full-frame support detector fallback")
        fallback_watchdog = branch_watchdog("frame-support fallback detector")
        try:
            fallback = _large_frame_support_fallback(
                mean_gray,
                std_gray,
                mad_gray,
                reference_gray,
                cfg,
                progress_callback=detector_callback,
                progress_offset=12,
                progress_total=progress_total,
                watchdog=fallback_watchdog,
            )
        except DetectorWatchdogTimeout as exc:
            primary.debug.watchdog_events.append(timeout_event(exc))
            if emit_progress:
                report(14, f"Detector watchdog stopped fallback at {exc.phase}; using primary geometry")
            if primary.debug.completed_count >= 8 and primary.lattice.confidence >= 0.50:
                return primary
            raise
        total_watchdog.check("frame-support fallback detector complete")
        return fallback if _detection_quality_key(fallback) > _detection_quality_key(primary) else primary

    result = run_on_sample(sample_indices, emit_progress=True)
    if not _full_detection_consistent_with_sampling(result, sample_history):
        expanded_indices = _next_midpoint_expansion_indices(sample_indices, n_source_frames, reference_index, count=5)
        if expanded_indices != sample_indices:
            sample_history.append({
                'sampled_count': int(len(expanded_indices)),
                'sample_indices': [int(i) for i in expanded_indices],
                'stable': False,
                'stable_run': 0,
                'reason': 'expanded_after_full_detector_consistency_check',
                'previous_completed_count': int(result.debug.completed_count),
            })
            report(14, f"Expanded representative sample to {len(expanded_indices)}/{n_source_frames} after consistency check")
            sample_indices = expanded_indices
            result = run_on_sample(sample_indices, emit_progress=False)

    report(22, f"Reference geometry detector complete with {result.debug.mode}")
    return _annotate_sequence_sampling_debug(result, n_source_frames, sample_indices, sample_history)


def detect_exact_wafer_holes_sequence(images: Sequence[np.ndarray], cfg: GeometryConfig, reference_index: int = 0, return_debug: bool = False):
    result = detect_exact_wafer_holes_sequence_full(images, cfg, reference_index=reference_index)
    if return_debug:
        return result.accepted_candidates, result.debug
    return result.accepted_candidates
