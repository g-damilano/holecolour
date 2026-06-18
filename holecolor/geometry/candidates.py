from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence

import cv2
import numpy as np
from scipy.spatial import cKDTree

from holecolor.config.schema import GeometryConfig
from holecolor.core.types import HoleCandidate


@dataclass(slots=True)
class GridDetectionDebug:
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


def _circle_confidence(area: float, peri: float, contrast: float, ellipticity: float, fill_ratio: float) -> float:
    circ = 4 * np.pi * area / max(peri * peri, 1e-6)
    score = 0.40 * np.clip(circ, 0, 1)
    score += 0.25 * np.clip(contrast / 64.0, 0, 1)
    score += 0.20 * np.clip(fill_ratio, 0, 1)
    score += 0.15 * (1.0 - np.clip(ellipticity, 0, 1))
    return float(np.clip(score, 0, 1))


def _deduplicate(candidates: list[HoleCandidate], px_factor: float) -> list[HoleCandidate]:
    if not candidates:
        return []
    ordered = sorted(candidates, key=lambda c: (c.confidence, c.boundary_contrast), reverse=True)
    kept: list[HoleCandidate] = []
    for cand in ordered:
        min_sep = px_factor * max(cand.radius_px, 1.0)
        if any(((cand.x - k.x) ** 2 + (cand.y - k.y) ** 2) ** 0.5 < min_sep for k in kept):
            continue
        kept.append(cand)
    return kept


def _legacy_dark_blob_candidates(image: np.ndarray, cfg: GeometryConfig) -> list[HoleCandidate]:
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = np.quantile(blur, cfg.dark_threshold_quantile)
    bw = (blur <= thresh).astype(np.uint8) * 255
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    contours, _ = cv2.findContours(bw, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    out: list[HoleCandidate] = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area <= 0:
            continue
        (x, y), radius = cv2.minEnclosingCircle(cnt)
        if not (cfg.min_radius_px <= radius <= cfg.max_radius_px):
            continue
        peri = cv2.arcLength(cnt, True)
        if peri <= 0:
            continue
        mask = np.zeros_like(gray, dtype=np.uint8)
        cv2.drawContours(mask, [cnt], -1, 255, -1)
        dil = cv2.dilate(mask, np.ones((5, 5), np.uint8), iterations=1)
        ring_mask = (dil > 0) & ~(mask > 0)
        inside = gray[mask > 0]
        ring = gray[ring_mask]
        contrast = float(np.mean(ring) - np.mean(inside)) if inside.size and ring.size else 0.0
        ellipse = cv2.fitEllipse(cnt) if len(cnt) >= 5 else None
        ellipticity = 0.0
        if ellipse is not None:
            (ex, ey), (ma, mi), _ = ellipse
            x, y = float(ex), float(ey)
            a, b = max(ma, mi), min(ma, mi)
            radius = 0.25 * (a + b)
            ellipticity = float(1.0 - b / max(a, 1e-6))
        fill_ratio = float(area / max(np.pi * radius * radius, 1e-6))
        conf = _circle_confidence(area, peri, contrast, ellipticity, fill_ratio)
        out.append(HoleCandidate(float(x), float(y), float(radius), ellipticity, contrast, conf))
    out = [c for c in out if c.confidence >= cfg.min_confidence]
    return _deduplicate(out, cfg.duplicate_suppression_px)


def _detect_support_circle(gray: np.ndarray) -> tuple[int, int, int] | None:
    proc = cv2.medianBlur(gray, 9)
    circles = cv2.HoughCircles(proc, cv2.HOUGH_GRADIENT, dp=1.2, minDist=min(gray.shape) // 4, param1=100, param2=30, minRadius=max(40, min(gray.shape) // 6), maxRadius=max(60, min(gray.shape) // 2))
    if circles is None:
        return None
    h, w = gray.shape
    yy, xx = np.ogrid[:h, :w]
    center = np.array([w / 2.0, h / 2.0], dtype=np.float32)
    best = None
    best_score = float('inf')
    for x, y, r in np.round(circles[0]).astype(int):
        inside = (xx - x) ** 2 + (yy - y) ** 2 <= int(max(1, round(0.92 * r))) ** 2
        if not np.any(inside):
            continue
        mean_inside = float(gray[inside].mean())
        score = mean_inside + 0.02 * float(np.linalg.norm(np.array([x, y], dtype=np.float32) - center))
        if score < best_score:
            best_score = score
            best = (int(x), int(y), int(r))
    return best


def _support_mask_from_density(gray: np.ndarray, radii: np.ndarray, centers: np.ndarray, contrasts: np.ndarray) -> np.ndarray | None:
    if len(centers) < 12:
        return None
    h, w = gray.shape
    occ = np.zeros((h, w), dtype=np.float32)
    weights = np.clip(contrasts.astype(np.float32), 0.1, None)
    for (x, y), wt in zip(centers.astype(int), weights):
        if 0 <= x < w and 0 <= y < h:
            occ[y, x] += float(wt)
    sigma = max(12.0, float(np.median(radii)) * 3.5)
    density = cv2.GaussianBlur(occ, (0, 0), sigmaX=sigma, sigmaY=sigma)
    pos = density[density > 0]
    if pos.size == 0:
        return None
    thr = float(np.quantile(pos, 0.88))
    mask = (density >= thr).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((41, 41), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((17, 17), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if n <= 1:
        return None
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    mask = (labels == largest).astype(np.uint8)
    mask = cv2.dilate(mask, np.ones((29, 29), np.uint8), iterations=1)
    return mask.astype(bool)


def _estimate_grid_basis(centers: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float] | None:
    if len(centers) < 8:
        return None
    tree = cKDTree(centers)
    dists, idxs = tree.query(centers, k=min(7, len(centers)))
    nn = float(np.median(dists[:, 1]))
    local_keep = []
    for i, p in enumerate(centers):
        idx = tree.query_ball_point(p, r=nn * 1.35)
        idx = [j for j in idx if j != i]
        vecs = centers[idx] - p
        if len(vecs) == 0:
            local_keep.append(False)
            continue
        dist = np.linalg.norm(vecs, axis=1)
        local_keep.append(int(np.sum((dist > 0.7 * nn) & (dist < 1.3 * nn))) >= 2)
    centers = centers[np.asarray(local_keep, dtype=bool)]
    if len(centers) < 8:
        return None
    tree = cKDTree(centers)
    dists, idxs = tree.query(centers, k=min(7, len(centers)))
    nn = float(np.median(dists[:, 1]))
    vectors = []
    for i, p in enumerate(centers):
        for j in idxs[i, 1:]:
            v = centers[int(j)] - p
            d = float(np.linalg.norm(v))
            if 0.7 * nn < d < 1.3 * nn:
                vectors.append(v)
    if len(vectors) < 8:
        return None
    vectors = np.asarray(vectors, dtype=np.float32)
    ang = np.mod(np.arctan2(vectors[:, 1], vectors[:, 0]), np.pi)
    hist, edges = np.histogram(ang, bins=180, range=(0.0, np.pi))
    i1 = int(np.argmax(hist))
    theta1 = float(0.5 * (edges[i1] + edges[i1 + 1]))
    sep_mask = np.abs(((ang - theta1 + np.pi / 2.0) % np.pi) - np.pi / 2.0) > np.deg2rad(15.0)
    if np.any(sep_mask):
        hist2, edges2 = np.histogram(ang[sep_mask], bins=180, range=(0.0, np.pi))
        i2 = int(np.argmax(hist2))
        theta2 = float(0.5 * (edges2[i2] + edges2[i2 + 1]))
    else:
        theta2 = float((theta1 + np.pi / 2.0) % np.pi)

    def _basis(theta: float) -> np.ndarray:
        good = np.abs(((ang - theta + np.pi / 2.0) % np.pi) - np.pi / 2.0) < np.deg2rad(10.0)
        vs = vectors[good]
        if len(vs) == 0:
            return np.array([math.cos(theta), math.sin(theta)], dtype=np.float32) * nn
        proj = np.array([math.cos(theta), math.sin(theta)], dtype=np.float32)
        signs = np.sign(vs @ proj)
        signs[signs == 0] = 1.0
        vs = vs * signs[:, None]
        lengths = np.linalg.norm(vs, axis=1)
        dirs = vs / np.maximum(lengths[:, None], 1e-6)
        direction = np.median(dirs, axis=0)
        direction /= np.linalg.norm(direction) + 1e-6
        return direction.astype(np.float32) * float(np.median(lengths))

    u = _basis(theta1)
    v = _basis(theta2)
    b = np.column_stack([u, v]).astype(np.float32)
    inv = np.linalg.pinv(b)
    best_origin = centers[0]
    best_res = float('inf')
    for p in centers:
        uv = (centers - p) @ inv.T
        residual = float(np.mean(np.linalg.norm(uv - np.round(uv), axis=1)))
        if residual < best_res:
            best_res = residual
            best_origin = p
    return centers, u, v, best_origin.astype(np.float32), nn


def _robust_scale(x: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    vals = x[mask] if mask is not None and np.any(mask) else x.reshape(-1)
    if vals.size == 0:
        return np.zeros_like(x, dtype=np.float32)
    p1, p99 = np.percentile(vals, [1, 99])
    den = max(float(p99 - p1), 1e-6)
    return np.clip((x.astype(np.float32) - float(p1)) / den, 0.0, 1.0).astype(np.float32)


def _gradient_magnitude(im: np.ndarray) -> np.ndarray:
    gx = cv2.Sobel(im.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(im.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
    return np.sqrt(gx * gx + gy * gy)


def _circle_ring_masks(radius: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ring_outer = int(round(1.75 * radius))
    g = np.arange(-ring_outer, ring_outer + 1)
    gy, gx = np.meshgrid(g, g, indexing='ij')
    dist = np.sqrt(gx * gx + gy * gy)
    core = dist <= radius
    ring = (dist >= 1.25 * radius) & (dist <= 1.75 * radius)
    edge = (dist >= radius - 1.2) & (dist <= radius + 1.2)
    return core, ring, edge


def _polarity_circle_candidates(gray: np.ndarray, support_mask: np.ndarray, cfg: GeometryConfig, polarity: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    work = gray if polarity == 'bright' else (255 - gray)
    eq = cv2.equalizeHist(work)
    circles = cv2.HoughCircles(eq, cv2.HOUGH_GRADIENT, dp=1.1, minDist=max(10, int(round(cfg.min_radius_px * 2.0))), param1=100, param2=18, minRadius=max(3, int(round(cfg.min_radius_px))), maxRadius=max(5, int(round(min(cfg.max_radius_px, 20.0)))))
    if circles is None:
        return np.zeros((0, 2), dtype=np.float32), np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=np.float32)
    h, w = gray.shape
    yy, xx = np.ogrid[:h, :w]
    centers, radii, contrasts = [], [], []
    for x, y, r in np.round(circles[0]).astype(int):
        if not (0 <= x < w and 0 <= y < h) or not support_mask[y, x]:
            continue
        d2 = (xx - x) ** 2 + (yy - y) ** 2
        inside = d2 <= r * r
        ring = (d2 <= (r * 1.45) ** 2) & (d2 >= (r * 1.05) ** 2)
        if not np.any(inside) or not np.any(ring):
            continue
        contrast = float(gray[inside].mean() - gray[ring].mean()) if polarity == 'bright' else float(gray[ring].mean() - gray[inside].mean())
        centers.append((float(x), float(y)))
        radii.append(float(r))
        contrasts.append(contrast)
    if not centers:
        return np.zeros((0, 2), dtype=np.float32), np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=np.float32)
    return np.asarray(centers, dtype=np.float32), np.asarray(radii, dtype=np.float32), np.asarray(contrasts, dtype=np.float32)


def _raw_circle_hole_candidates(centers: np.ndarray, radii: np.ndarray, contrasts: np.ndarray, cfg: GeometryConfig) -> list[HoleCandidate]:
    provisional = []
    for (x, y), radius, contrast in zip(centers, radii, contrasts):
        conf = float(np.clip(abs(float(contrast)) / 32.0, 0.25, 0.95))
        provisional.append(HoleCandidate(float(x), float(y), float(radius), 0.0, float(contrast), conf))
    return _deduplicate(provisional, max(0.7, float(cfg.duplicate_suppression_px)))


def _score_local_candidate(x: float, y: float, radius: float, mean_scaled: np.ndarray, ref_scaled: np.ndarray, inv_std: np.ndarray, inv_mad: np.ndarray, grad_scaled: np.ndarray, support_mask: np.ndarray) -> tuple[float, float]:
    r = max(3, int(round(radius)))
    core_t, ring_t, edge_t = _circle_ring_masks(r)
    R = core_t.shape[0] // 2
    xi, yi = int(round(x)), int(round(y))
    h, w = mean_scaled.shape
    if xi < R or xi >= w - R or yi < R or yi >= h - R:
        return -1.0, 0.0
    sl_y = slice(yi - R, yi + R + 1)
    sl_x = slice(xi - R, xi + R + 1)
    local_support = support_mask[sl_y, sl_x]
    core = core_t & local_support
    ring = ring_t & local_support
    edge = edge_t & local_support
    if core.sum() < 0.65 * core_t.sum() or ring.sum() < 0.45 * ring_t.sum() or edge.sum() < 0.35 * edge_t.sum():
        return -1.0, 0.0
    p_mean = mean_scaled[sl_y, sl_x]
    p_ref = ref_scaled[sl_y, sl_x]
    p_std = inv_std[sl_y, sl_x]
    p_mad = inv_mad[sl_y, sl_x]
    p_grad = grad_scaled[sl_y, sl_x]
    mean_gap = float(p_mean[core].mean() - p_mean[ring].mean())
    ref_gap = float(p_ref[core].mean() - p_ref[ring].mean())
    std_gap = float(p_std[core].mean() - p_std[ring].mean())
    mad_gap = float(p_mad[core].mean() - p_mad[ring].mean())
    edge_grad = float(p_grad[edge].mean())
    bright = 0.26 * mean_gap + 0.16 * ref_gap + 0.20 * std_gap + 0.14 * mad_gap + 0.24 * edge_grad
    dark = 0.26 * (-mean_gap) + 0.16 * (-ref_gap) + 0.20 * std_gap + 0.14 * mad_gap + 0.24 * edge_grad
    return float(max(bright, dark)), float(abs(mean_gap))


def _fit_refined_lattice(pts: np.ndarray, basis_u: np.ndarray, basis_v: np.ndarray, origin: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    B = np.column_stack([basis_u, basis_v]).astype(np.float32)
    inv = np.linalg.pinv(B)
    uv = (pts - origin) @ inv.T
    ij = np.round(uv).astype(int)
    A = np.column_stack([np.ones(len(ij), dtype=np.float32), ij[:, 0].astype(np.float32), ij[:, 1].astype(np.float32)])
    coef_x, *_ = np.linalg.lstsq(A, pts[:, 0], rcond=None)
    coef_y, *_ = np.linalg.lstsq(A, pts[:, 1], rcond=None)
    origin_ref = np.array([coef_x[0], coef_y[0]], dtype=np.float32)
    u_ref = np.array([coef_x[1], coef_y[1]], dtype=np.float32)
    v_ref = np.array([coef_x[2], coef_y[2]], dtype=np.float32)
    return origin_ref, u_ref, v_ref, ij


def _complete_grid_from_sequence(mean_gray: np.ndarray, ref_gray: np.ndarray, support_mask: np.ndarray, seed_pts: np.ndarray, seed_scores: np.ndarray, seed_radii: np.ndarray, basis_u: np.ndarray, basis_v: np.ndarray, origin: np.ndarray, spacing: float, inv_std: np.ndarray, inv_mad: np.ndarray) -> list[HoleCandidate]:
    mean_scaled = _robust_scale(mean_gray, support_mask)
    ref_scaled = _robust_scale(ref_gray, support_mask)
    grad_scaled = _robust_scale(_gradient_magnitude(mean_scaled) * 0.7 + _gradient_magnitude(ref_scaled) * 0.3, support_mask)
    common_r = max(3, int(round(float(np.median(seed_radii)))))
    origin_ref, u_ref, v_ref, ij = _fit_refined_lattice(seed_pts.astype(np.float32), basis_u, basis_v, origin)
    B = np.column_stack([u_ref, v_ref]).astype(np.float32)
    inv = np.linalg.pinv(B)
    order = np.argsort(-seed_scores)
    cell_map: dict[tuple[int, int], tuple[np.ndarray, float, float]] = {}
    for idx in order:
        cell = (int(ij[idx, 0]), int(ij[idx, 1]))
        if cell not in cell_map:
            cell_map[cell] = (seed_pts[idx], float(seed_scores[idx]), float(seed_radii[idx]))
    ys, xs = np.where(support_mask)
    corners = np.array([[xs.min(), ys.min()], [xs.max(), ys.min()], [xs.min(), ys.max()], [xs.max(), ys.max()]], dtype=np.float32)
    uv_bounds = (corners - origin_ref) @ inv.T
    umin, umax = int(np.floor(uv_bounds[:, 0].min())) - 2, int(np.ceil(uv_bounds[:, 0].max())) + 2
    vmin, vmax = int(np.floor(uv_bounds[:, 1].min())) - 2, int(np.ceil(uv_bounds[:, 1].max())) + 2
    strong_seed_scores = np.asarray([v[1] for v in cell_map.values()], dtype=np.float32)
    strong_thr = float(np.percentile(strong_seed_scores, 12)) if len(strong_seed_scores) else 0.05
    weak_thr = float(np.percentile(strong_seed_scores, 3)) if len(strong_seed_scores) else 0.02
    h, w = mean_gray.shape
    out = []
    for iu in range(umin, umax + 1):
        for iv in range(vmin, vmax + 1):
            p = origin_ref + iu * u_ref + iv * v_ref
            x, y = float(p[0]), float(p[1])
            xi, yi = int(round(x)), int(round(y))
            if not (0 <= xi < w and 0 <= yi < h) or not support_mask[yi, xi]:
                continue
            cell = (iu, iv)
            if cell in cell_map:
                pt, sc, _rr = cell_map[cell]
                out.append(HoleCandidate(float(pt[0]), float(pt[1]), float(common_r), 0.0, float(sc), float(np.clip(sc, 0.0, 1.0))))
                continue
            best = None
            for dy in range(-4, 5):
                yy = yi + dy
                if yy < 0 or yy >= h:
                    continue
                for dx in range(-4, 5):
                    xx = xi + dx
                    if xx < 0 or xx >= w or not support_mask[yy, xx]:
                        continue
                    for rr in range(max(3, common_r - 2), common_r + 3):
                        sc, contrast = _score_local_candidate(xx, yy, rr, mean_scaled, ref_scaled, inv_std, inv_mad, grad_scaled, support_mask)
                        if sc < -0.5:
                            continue
                        total = sc - 0.02 * math.hypot(dx, dy) - 0.03 * abs(rr - common_r)
                        if best is None or total > best[0]:
                            best = (float(total), xx, yy, rr, contrast)
            if best is None:
                continue
            total, xx, yy, rr, contrast = best
            if total >= strong_thr:
                conf = float(np.clip((total - weak_thr) / max(strong_thr - weak_thr, 1e-6), 0.55, 0.95))
                out.append(HoleCandidate(float(xx), float(yy), float(rr), 0.0, float(contrast), conf))
    return _deduplicate(out, 0.7)



def _build_overflow_mask(shape: tuple[int, int], centers: np.ndarray, radius: int, support_mask: np.ndarray, margin_px: int = 8) -> np.ndarray:
    h, w = shape
    mask = np.zeros((h, w), dtype=np.uint8)
    rr = max(1, int(round(radius + margin_px)))
    for x, y in centers:
        cv2.circle(mask, (int(round(float(x))), int(round(float(y)))), rr, 1, thickness=-1)
    return (mask > 0) & support_mask


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


def _local_border_score(
    x0: float,
    y0: float,
    radius: int,
    mean_scaled: np.ndarray,
    ref_scaled: np.ndarray,
    grad_scaled: np.ndarray,
    support_mask: np.ndarray,
    overflow_mask: np.ndarray,
    templates: dict[int, dict[str, np.ndarray]],
    allow_partial: bool,
) -> dict[str, float] | None:
    tpl = templates[int(radius)]
    R = int(tpl['R'])
    xi, yi = int(round(float(x0))), int(round(float(y0)))
    h, w = mean_scaled.shape
    if xi < R or xi >= w - R or yi < R or yi >= h - R:
        return None
    sl_y = slice(yi - R, yi + R + 1)
    sl_x = slice(xi - R, xi + R + 1)
    local_support = support_mask[sl_y, sl_x]
    local_overflow = overflow_mask[sl_y, sl_x]
    valid = local_support & local_overflow
    edge = tpl['edge_band'] & valid
    inner = tpl['inner_band'] & valid
    outer = tpl['outer_band'] & valid
    edge_cov = float(edge.sum() / max(int(np.sum(tpl['edge_band'])), 1))
    inner_cov = float(inner.sum() / max(int(np.sum(tpl['inner_band'])), 1))
    outer_cov = float(outer.sum() / max(int(np.sum(tpl['outer_band'])), 1))
    if allow_partial:
        if edge_cov < 0.25 or inner_cov < 0.20 or outer_cov < 0.15:
            return None
    else:
        if edge_cov < 0.60 or inner_cov < 0.55 or outer_cov < 0.45:
            return None
    p_mean = mean_scaled[sl_y, sl_x]
    p_ref = ref_scaled[sl_y, sl_x]
    p_grad = grad_scaled[sl_y, sl_x]
    edge_grad = float(p_grad[edge].mean())
    edge_contrast_mean = float(p_mean[inner].mean() - p_mean[outer].mean())
    edge_contrast_ref = float(p_ref[inner].mean() - p_ref[outer].mean())
    return {
        'edge_grad': edge_grad,
        'edge_contrast_mean': edge_contrast_mean,
        'edge_contrast_ref': edge_contrast_ref,
        'edge_coverage': edge_cov,
        'inner_coverage': inner_cov,
        'outer_coverage': outer_cov,
    }


def _refine_anchor_consensus(
    mean_gray: np.ndarray,
    ref_gray: np.ndarray,
    support_mask: np.ndarray,
    anchor_cells: dict[tuple[int, int], tuple[np.ndarray, float, float]],
    common_radius: int,
) -> tuple[list[HoleCandidate], np.ndarray, np.ndarray, np.ndarray, float]:
    if not anchor_cells:
        return [], np.zeros((0, 2), dtype=np.float32), np.zeros((0, 2), dtype=np.int32), np.zeros((0, 2), dtype=np.float32), float(common_radius)
    mean_scaled = _robust_scale(mean_gray, support_mask)
    ref_scaled = _robust_scale(ref_gray, support_mask)
    grad_scaled = _robust_scale(0.7 * _gradient_magnitude(mean_scaled) + 0.3 * _gradient_magnitude(ref_scaled), support_mask)
    anchor_pts = np.asarray([v[0] for v in anchor_cells.values()], dtype=np.float32)
    overflow = _build_overflow_mask(mean_gray.shape, anchor_pts, common_radius, support_mask, margin_px=8)
    rad_candidates = list(range(max(3, common_radius - 2), common_radius + 3))
    templates = _build_border_templates(rad_candidates)
    search_xy = 3
    loose_rows: list[dict[str, float | int]] = []
    cell_keys = list(anchor_cells.keys())
    for cell in cell_keys:
        p, seed_score, _ = anchor_cells[cell]
        x0, y0 = float(p[0]), float(p[1])
        best = None
        for dy in range(-search_xy, search_xy + 1):
            yy = int(round(y0 + dy))
            for dx in range(-search_xy, search_xy + 1):
                xx = int(round(x0 + dx))
                for rr in rad_candidates:
                    if not (0 <= xx < mean_gray.shape[1] and 0 <= yy < mean_gray.shape[0]):
                        continue
                    res = _local_border_score(xx, yy, rr, mean_scaled, ref_scaled, grad_scaled, support_mask, overflow, templates, allow_partial=False)
                    if res is None:
                        continue
                    score = 0.62 * res['edge_grad'] + 0.23 * res['edge_contrast_mean'] + 0.15 * res['edge_contrast_ref'] - 0.018 * math.hypot(dx, dy) - 0.028 * abs(rr - common_radius)
                    if best is None or score > best['score_loose']:
                        best = {
                            'x': float(xx), 'y': float(yy), 'r': float(rr), 'seed_score': float(seed_score),
                            'score_loose': float(score),
                        }
        if best is not None:
            loose_rows.append({'cell_u': int(cell[0]), 'cell_v': int(cell[1]), **best})
    if not loose_rows:
        candidates = [HoleCandidate(float(p[0]), float(p[1]), float(common_radius), 0.0, 0.0, float(np.clip(sc, 0.0, 1.0))) for p, sc, _ in anchor_cells.values()]
        pts = np.asarray([[c.x, c.y] for c in candidates], dtype=np.float32)
        ij = np.asarray([[int(k[0]), int(k[1])] for k in anchor_cells.keys()], dtype=np.int32)
        return candidates, pts, ij, anchor_pts, float(common_radius)
    loose_df = np.asarray([[r['cell_u'], r['cell_v'], r['x'], r['y'], r['r'], r['seed_score'], r['score_loose']] for r in loose_rows], dtype=np.float32)
    refined_radius = float(np.median(loose_df[:, 4]))
    fixed_r = max(3, int(round(refined_radius)))
    if fixed_r not in templates:
        templates = _build_border_templates(rad_candidates + [fixed_r])
    final_candidates: list[HoleCandidate] = []
    final_pts = []
    final_ij = []
    for row in loose_rows:
        x0, y0 = float(row['x']), float(row['y'])
        best = None
        for dy in range(-search_xy, search_xy + 1):
            yy = int(round(y0 + dy))
            for dx in range(-search_xy, search_xy + 1):
                xx = int(round(x0 + dx))
                res = _local_border_score(xx, yy, fixed_r, mean_scaled, ref_scaled, grad_scaled, support_mask, overflow, templates, allow_partial=False)
                if res is None:
                    continue
                score = 0.68 * res['edge_grad'] + 0.20 * res['edge_contrast_mean'] + 0.12 * res['edge_contrast_ref'] - 0.020 * math.hypot(dx, dy)
                if best is None or score > best[0]:
                    best = (float(score), xx, yy)
        if best is None:
            continue
        score_final, xx, yy = best
        conf = float(np.clip(0.65 * row['seed_score'] + 0.35 * score_final, 0.55, 0.99))
        final_candidates.append(HoleCandidate(float(xx), float(yy), float(fixed_r), 0.0, float(score_final), conf))
        final_pts.append((float(xx), float(yy)))
        final_ij.append((int(row['cell_u']), int(row['cell_v'])))
    return final_candidates, np.asarray(final_pts, dtype=np.float32), np.asarray(final_ij, dtype=np.int32), anchor_pts, float(fixed_r)


def _enumerate_predicted_nodes(support_mask: np.ndarray, origin_ref: np.ndarray, u_ref: np.ndarray, v_ref: np.ndarray, common_radius: int) -> list[dict[str, object]]:
    ys, xs = np.where(support_mask)
    corners = np.array([[xs.min(), ys.min()], [xs.max(), ys.min()], [xs.min(), ys.max()], [xs.max(), ys.max()]], dtype=np.float32)
    B = np.column_stack([u_ref, v_ref]).astype(np.float32)
    inv = np.linalg.pinv(B)
    uv_bounds = (corners - origin_ref) @ inv.T
    umin, umax = int(np.floor(uv_bounds[:, 0].min())) - 3, int(np.ceil(uv_bounds[:, 0].max())) + 3
    vmin, vmax = int(np.floor(uv_bounds[:, 1].min())) - 3, int(np.ceil(uv_bounds[:, 1].max())) + 3
    h, w = support_mask.shape
    out: list[dict[str, object]] = []
    ys_mask, xs_mask = np.where(support_mask)
    if xs_mask.size == 0:
        return out
    xmid = 0.5 * (float(xs_mask.min()) + float(xs_mask.max()))
    ymid = 0.5 * (float(ys_mask.min()) + float(ys_mask.max()))
    dist_support = np.sqrt((xs_mask - xmid) ** 2 + (ys_mask - ymid) ** 2)
    support_radius = float(np.max(dist_support)) if dist_support.size else 0.0
    for iu in range(umin, umax + 1):
        for iv in range(vmin, vmax + 1):
            p = origin_ref + iu * u_ref + iv * v_ref
            x, y = float(p[0]), float(p[1])
            xi, yi = int(round(x)), int(round(y))
            if not (0 <= xi < w and 0 <= yi < h):
                continue
            d = float(math.hypot(x - xmid, y - ymid))
            if d > support_radius + common_radius + 2:
                continue
            geometry_class = 'full' if d <= support_radius - common_radius - 1 else 'partial'
            out.append({'cell_u': int(iu), 'cell_v': int(iv), 'x_pred': x, 'y_pred': y, 'geometry_class': geometry_class})
    return out


def _recover_missing_from_grid(
    mean_gray: np.ndarray,
    ref_gray: np.ndarray,
    support_mask: np.ndarray,
    origin_ref: np.ndarray,
    u_ref: np.ndarray,
    v_ref: np.ndarray,
    anchor_candidates: list[HoleCandidate],
    anchor_ij: np.ndarray,
    common_radius: int,
) -> tuple[list[HoleCandidate], list[dict[str, object]], list[dict[str, object]]]:
    mean_scaled = _robust_scale(mean_gray, support_mask)
    ref_scaled = _robust_scale(ref_gray, support_mask)
    grad_scaled = _robust_scale(0.7 * _gradient_magnitude(mean_scaled) + 0.3 * _gradient_magnitude(ref_scaled), support_mask)
    anchor_pts = np.asarray([[c.x, c.y] for c in anchor_candidates], dtype=np.float32)
    overflow = _build_overflow_mask(mean_gray.shape, anchor_pts, common_radius, support_mask, margin_px=8)
    rad_candidates = list(range(max(3, common_radius - 2), common_radius + 3))
    templates = _build_border_templates(rad_candidates)
    existing = {(int(i), int(j)) for i, j in anchor_ij.tolist()} if len(anchor_ij) else set()
    anchor_local_scores = []
    for cand in anchor_candidates:
        res_anchor = _local_border_score(cand.x, cand.y, int(round(cand.radius_px)), mean_scaled, ref_scaled, grad_scaled, support_mask, overflow, templates, allow_partial=False)
        if res_anchor is None:
            continue
        anchor_local_scores.append(0.66 * res_anchor['edge_grad'] + 0.22 * res_anchor['edge_contrast_mean'] + 0.12 * res_anchor['edge_contrast_ref'])
    anchor_scores = np.asarray(anchor_local_scores, dtype=np.float32)
    strong_thr = float(np.percentile(anchor_scores, 10)) if len(anchor_scores) else 0.22
    weak_thr = float(np.percentile(anchor_scores, 3)) if len(anchor_scores) else 0.16
    predicted_rows = _enumerate_predicted_nodes(support_mask, origin_ref, u_ref, v_ref, common_radius)
    if len(anchor_ij):
        umin = int(np.min(anchor_ij[:, 0])) - 2
        umax = int(np.max(anchor_ij[:, 0])) + 2
        vmin = int(np.min(anchor_ij[:, 1])) - 2
        vmax = int(np.max(anchor_ij[:, 1])) + 2
        predicted_rows = [row for row in predicted_rows if umin <= int(row['cell_u']) <= umax and vmin <= int(row['cell_v']) <= vmax]
    predicted_only: list[dict[str, object]] = []
    recovered_strong: list[HoleCandidate] = []
    recovered_rows: list[dict[str, object]] = []
    anchor_support = {(int(i), int(j)): True for i, j in anchor_ij.tolist()} if len(anchor_ij) else {}

    def neighbor_support(i: int, j: int) -> float:
        ncoords = [(i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1), (i - 1, j + 1), (i + 1, j - 1)]
        vals = [1.0 if anchor_support.get(c, False) else 0.0 for c in ncoords]
        return float(np.mean(vals)) if vals else 0.0

    for row in predicted_rows:
        cell = (int(row['cell_u']), int(row['cell_v']))
        if cell in existing:
            continue
        x0 = float(row['x_pred'])
        y0 = float(row['y_pred'])
        allow_partial = str(row['geometry_class']) == 'partial'
        n_support = neighbor_support(*cell)
        best = None
        for dy in range(-4, 5):
            yy = int(round(y0 + dy))
            for dx in range(-4, 5):
                xx = int(round(x0 + dx))
                for rr in rad_candidates:
                    res = _local_border_score(xx, yy, rr, mean_scaled, ref_scaled, grad_scaled, support_mask, overflow, templates, allow_partial=allow_partial)
                    if res is None:
                        continue
                    score_local = 0.66 * res['edge_grad'] + 0.22 * res['edge_contrast_mean'] + 0.12 * res['edge_contrast_ref']
                    score_total = score_local + 0.05 * n_support - 0.018 * math.hypot(dx, dy) - 0.020 * abs(rr - common_radius)
                    if best is None or score_total > best['score_total']:
                        best = {
                            'cell_u': int(cell[0]), 'cell_v': int(cell[1]), 'x_pred': x0, 'y_pred': y0,
                            'x': float(xx), 'y': float(yy), 'r': float(rr), 'geometry_class': str(row['geometry_class']),
                            'neighbor_support': float(n_support), 'score_local': float(score_local), 'score_total': float(score_total),
                        }
        if best is None:
            predicted_only.append({'cell_u': int(cell[0]), 'cell_v': int(cell[1]), 'x_pred': x0, 'y_pred': y0, 'geometry_class': str(row['geometry_class']), 'status': 'predicted_only'})
            continue
        min_neighbor = 0.34 if not allow_partial else 0.17
        if best['score_local'] >= strong_thr and n_support >= min_neighbor:
            conf = float(np.clip((best['score_local'] - weak_thr) / max(strong_thr - weak_thr, 1e-6), 0.55, 0.92))
            recovered_strong.append(HoleCandidate(best['x'], best['y'], best['r'], 0.0, best['score_local'], conf))
            recovered_rows.append({'tier': 'recovered_strong', **best, 'confidence': conf})
            existing.add(cell)
        else:
            predicted_only.append({'cell_u': int(cell[0]), 'cell_v': int(cell[1]), 'x_pred': x0, 'y_pred': y0, 'geometry_class': str(row['geometry_class']), 'status': 'predicted_only'})
    return recovered_strong, recovered_rows, predicted_only


def _sequence_gray_stack(images: Sequence[np.ndarray]) -> np.ndarray:
    return np.stack([cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32) for img in images], axis=0)


def detect_stable_grid_hole_candidates(images: Sequence[np.ndarray], cfg: GeometryConfig, reference_index: int = 0, return_debug: bool = False):
    if len(images) == 0:
        debug = GridDetectionDebug(None, None, 0, 0, 0, 0, 0, 0, 0, 'empty')
        return ([], debug) if return_debug else []

    gray_stack = _sequence_gray_stack(images)
    ref_gray = np.clip(gray_stack[int(np.clip(reference_index, 0, len(images) - 1))], 0, 255).astype(np.uint8)
    mean_gray = np.clip(gray_stack.mean(axis=0), 0, 255).astype(np.uint8)
    med_gray = np.median(gray_stack, axis=0).astype(np.float32)
    std_gray = gray_stack.std(axis=0).astype(np.float32)
    mad_gray = np.median(np.abs(gray_stack - med_gray[None, :, :]), axis=0).astype(np.float32)

    support_circle = _detect_support_circle(mean_gray)
    support_mask = None
    mode = 'sequence_density'
    if support_circle is not None:
        tmp = np.zeros_like(mean_gray, dtype=np.uint8)
        xw, yw, rw = support_circle
        cv2.circle(tmp, (xw, yw), int(round(rw * 0.95)), 255, -1)
        support_mask = tmp > 0
        mode = 'sequence_circle'
    provisional_mask = support_mask if support_mask is not None else np.ones_like(mean_gray, dtype=bool)

    candidate_sets = []
    for gray in (mean_gray, ref_gray):
        for polarity in ('bright', 'dark'):
            c, r, k = _polarity_circle_candidates(gray, provisional_mask, cfg, polarity)
            if len(c):
                candidate_sets.append((c, r, k))
    if candidate_sets:
        centers = np.concatenate([c for c, _, _ in candidate_sets], axis=0)
        radii = np.concatenate([r for _, r, _ in candidate_sets], axis=0)
        contrasts = np.concatenate([k for _, _, k in candidate_sets], axis=0)
    else:
        centers = np.zeros((0, 2), dtype=np.float32)
        radii = np.zeros((0,), dtype=np.float32)
        contrasts = np.zeros((0,), dtype=np.float32)

    if support_mask is None:
        mask = _support_mask_from_density(mean_gray, radii, centers, np.abs(contrasts)) if len(centers) else None
        support_mask = mask if mask is not None else np.ones_like(mean_gray, dtype=bool)
        mode = 'sequence_density' if mask is not None else 'sequence_fullframe'

    raw_count = int(len(centers))
    if raw_count == 0:
        debug = GridDetectionDebug(support_circle, support_mask, raw_count, 0, 0, 0, 0, 0, 0, mode)
        return ([], debug) if return_debug else []

    mean_scaled = _robust_scale(mean_gray.astype(np.float32), support_mask)
    ref_scaled = _robust_scale(ref_gray.astype(np.float32), support_mask)
    inv_std = _robust_scale(1.0 / (std_gray + 1e-6), support_mask)
    inv_mad = _robust_scale(1.0 / (mad_gray + 1e-6), support_mask)
    grad_scaled = _robust_scale(0.7 * _gradient_magnitude(mean_scaled) + 0.3 * _gradient_magnitude(ref_scaled), support_mask)

    seeds = []
    for (x, y), r, c in zip(centers, radii, contrasts):
        score, contrast = _score_local_candidate(float(x), float(y), float(r), mean_scaled, ref_scaled, inv_std, inv_mad, grad_scaled, support_mask)
        if score < -0.5:
            continue
        conf = float(np.clip(0.55 * score + 0.20 * max(c, 0.0), 0.0, 1.0))
        seeds.append(HoleCandidate(float(x), float(y), float(r), 0.0, float(contrast), conf))
    seeds = _deduplicate(seeds, 0.7)
    if len(seeds) < 8:
        debug = GridDetectionDebug(support_circle, support_mask, raw_count, len(seeds), len(seeds), 0, 0, 0, len(seeds), mode, float(np.median([s.radius_px for s in seeds])) if seeds else 0.0)
        return (seeds, debug) if return_debug else seeds

    seed_scores = np.asarray([max(s.confidence, 0.0) for s in seeds], dtype=np.float32)
    seed_pts = np.asarray([[s.x, s.y] for s in seeds], dtype=np.float32)
    seed_radii = np.asarray([s.radius_px for s in seeds], dtype=np.float32)
    score_thr = max(float(np.percentile(seed_scores, 45)), float(cfg.min_confidence) * 0.55)
    keep = seed_scores >= score_thr
    if int(np.sum(keep)) < 8:
        keep = seed_scores >= float(np.percentile(seed_scores, 30))
    filtered_pts = seed_pts[keep]
    filtered_scores = seed_scores[keep]
    filtered_radii = seed_radii[keep]
    basis = _estimate_grid_basis(filtered_pts)
    if basis is None:
        debug = GridDetectionDebug(support_circle, support_mask, raw_count, int(len(filtered_pts)), len(seeds), 0, 0, 0, len(seeds), mode, float(np.median(seed_radii)))
        return (seeds, debug) if return_debug else seeds

    centers_f, u, v, origin, _spacing = basis
    tree = cKDTree(filtered_pts)
    _, idx = tree.query(centers_f, k=1)
    _origin_ref0, _u_ref0, _v_ref0, ij = _fit_refined_lattice(centers_f.astype(np.float32), u, v, origin)

    order = np.argsort(-filtered_scores[idx])
    anchor_cells: dict[tuple[int, int], tuple[np.ndarray, float, float]] = {}
    for ord_idx in order:
        cell = (int(ij[ord_idx, 0]), int(ij[ord_idx, 1]))
        if cell not in anchor_cells:
            anchor_cells[cell] = (centers_f[ord_idx], float(filtered_scores[idx][ord_idx]), float(filtered_radii[idx][ord_idx]))

    common_radius = max(3, int(round(float(np.median([v[2] for v in anchor_cells.values()])))))
    anchor_candidates, anchor_pts_refined, anchor_ij, _anchor_pts_raw, common_radius_refined = _refine_anchor_consensus(
        mean_gray.astype(np.float32), ref_gray.astype(np.float32), support_mask, anchor_cells, common_radius
    )
    if len(anchor_candidates) < 8:
        debug = GridDetectionDebug(support_circle, support_mask, raw_count, int(len(filtered_pts)), len(anchor_candidates), 0, 0, 0, len(anchor_candidates), mode, float(common_radius_refined))
        return (anchor_candidates, debug) if return_debug else anchor_candidates

    anchor_pts_refined = np.asarray([[c.x, c.y] for c in anchor_candidates], dtype=np.float32)
    if len(anchor_ij) != len(anchor_pts_refined):
        anchor_ij = np.asarray([[int(k[0]), int(k[1])] for k in list(anchor_cells.keys())[:len(anchor_pts_refined)]], dtype=np.int32)
    A = np.column_stack([
        np.ones(len(anchor_ij), dtype=np.float32),
        anchor_ij[:, 0].astype(np.float32),
        anchor_ij[:, 1].astype(np.float32),
    ])
    coef_x, *_ = np.linalg.lstsq(A, anchor_pts_refined[:, 0], rcond=None)
    coef_y, *_ = np.linalg.lstsq(A, anchor_pts_refined[:, 1], rcond=None)
    origin_ref = np.array([coef_x[0], coef_y[0]], dtype=np.float32)
    u_ref = np.array([coef_x[1], coef_y[1]], dtype=np.float32)
    v_ref = np.array([coef_x[2], coef_y[2]], dtype=np.float32)

    recovered_strong, recovered_rows, predicted_only = _recover_missing_from_grid(
        mean_gray.astype(np.float32),
        ref_gray.astype(np.float32),
        support_mask,
        origin_ref,
        u_ref,
        v_ref,
        anchor_candidates,
        anchor_ij,
        int(round(common_radius_refined)),
    )

    final_candidates = _deduplicate(anchor_candidates + recovered_strong, 0.7)
    tiers: list[dict[str, object]] = []
    for cand, (i, j) in zip(anchor_candidates, anchor_ij.tolist()):
        tiers.append({'tier': 'anchor', 'cell_u': int(i), 'cell_v': int(j), 'x': float(cand.x), 'y': float(cand.y), 'radius_px': float(cand.radius_px), 'confidence': float(cand.confidence)})
    tiers.extend(recovered_rows)
    predicted_only_full = int(sum(1 for r in predicted_only if str(r.get('geometry_class')) == 'full'))
    predicted_only_partial = int(sum(1 for r in predicted_only if str(r.get('geometry_class')) == 'partial'))
    debug = GridDetectionDebug(
        support_circle,
        support_mask,
        raw_count,
        int(len(filtered_pts)),
        int(len(anchor_candidates)),
        int(len(recovered_strong)),
        predicted_only_full,
        predicted_only_partial,
        int(len(final_candidates)),
        mode,
        float(common_radius_refined),
        tiers=tiers,
        predicted_only=predicted_only,
    )
    return (final_candidates, debug) if return_debug else final_candidates


def detect_regular_grid_hole_candidates(image: np.ndarray, cfg: GeometryConfig, return_debug: bool = False):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    support_circle = _detect_support_circle(gray)
    support_mask = None
    mode = 'density'
    if support_circle is not None:
        tmp = np.zeros_like(gray, dtype=np.uint8)
        xw, yw, rw = support_circle
        cv2.circle(tmp, (xw, yw), int(round(rw * 0.95)), 255, -1)
        support_mask = tmp > 0
        mode = 'circle'
    provisional_mask = support_mask if support_mask is not None else np.ones_like(gray, dtype=bool)
    candidate_sets = []
    for polarity in ('bright', 'dark'):
        c, r, k = _polarity_circle_candidates(gray, provisional_mask, cfg, polarity)
        if len(c):
            candidate_sets.append((c, r, k))
    if candidate_sets:
        centers = np.concatenate([c for c, _, _ in candidate_sets], axis=0)
        radii = np.concatenate([r for _, r, _ in candidate_sets], axis=0)
        contrasts = np.concatenate([k for _, _, k in candidate_sets], axis=0)
    else:
        centers = np.zeros((0, 2), dtype=np.float32)
        radii = np.zeros((0,), dtype=np.float32)
        contrasts = np.zeros((0,), dtype=np.float32)
    if support_mask is None:
        support_mask = _support_mask_from_density(gray, radii, centers, np.abs(contrasts)) if len(centers) else None
        if support_mask is None:
            support_mask = np.ones_like(gray, dtype=bool)
            mode = 'fullframe'
    raw_count = len(centers)
    raw_candidates = _raw_circle_hole_candidates(centers, radii, contrasts, cfg)
    basis = _estimate_grid_basis(centers)
    if basis is None:
        debug = GridDetectionDebug(support_circle, support_mask, raw_count, int(len(raw_candidates)), int(len(raw_candidates)), 0, 0, 0, int(len(raw_candidates)), mode)
        return (raw_candidates, debug) if return_debug else raw_candidates
    centers_f, u, v, origin, spacing = basis
    tree = cKDTree(centers)
    _, idx = tree.query(centers_f, k=1)
    completed = _complete_grid(gray, support_mask, centers_f, radii[idx], np.abs(contrasts[idx]), u, v, origin, spacing)
    if not completed and raw_candidates:
        completed = raw_candidates
    debug = GridDetectionDebug(support_circle, support_mask, raw_count, int(len(centers_f)), int(len(completed)), 0, 0, 0, int(len(completed)), mode)
    return (completed, debug) if return_debug else completed


def _complete_grid(gray: np.ndarray, support_mask: np.ndarray, centers: np.ndarray, radii: np.ndarray, contrasts: np.ndarray, basis_u: np.ndarray, basis_v: np.ndarray, origin: np.ndarray, spacing: float) -> list[HoleCandidate]:
    b = np.column_stack([basis_u, basis_v]).astype(np.float32)
    inv = np.linalg.pinv(b)
    pts = centers.astype(np.float32)
    uv = (pts - origin) @ inv.T
    ij = np.round(uv).astype(int)
    pred = origin + ij @ b.T
    residual = np.linalg.norm(pts - pred, axis=1)
    keep = residual < 0.35 * spacing
    pts = pts[keep]
    radii = radii[keep]
    contrasts = contrasts[keep]
    ij = ij[keep]
    if len(pts) == 0:
        return []
    order = np.argsort(-contrasts)
    used, kept_pts, kept_r, kept_c = set(), [], [], []
    for idx in order:
        cell = (int(ij[idx, 0]), int(ij[idx, 1]))
        if cell in used:
            continue
        used.add(cell)
        kept_pts.append(pts[idx]); kept_r.append(float(radii[idx])); kept_c.append(float(contrasts[idx]))
    pts = np.asarray(kept_pts, dtype=np.float32)
    radii = np.asarray(kept_r, dtype=np.float32)
    contrasts = np.asarray(kept_c, dtype=np.float32)
    r_med = int(max(3, round(float(np.median(radii)))))
    size = 2 * r_med + 7
    yy, xx = np.mgrid[:size, :size]
    cy = cx = size // 2
    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    templ = np.exp(-(dist / (r_med * 0.9)) ** 4) - 0.45 * np.exp(-((dist - r_med * 1.45) / (r_med * 0.5)) ** 2)
    templ = templ.astype(np.float32); templ -= float(templ.mean())
    g = (gray.astype(np.float32) - float(gray.mean())) / (float(gray.std()) + 1e-6)
    res = cv2.matchTemplate(g.astype(np.float32), templ, cv2.TM_CCOEFF_NORMED)
    score_map = np.pad(res, ((size // 2, size - size // 2 - 1), (size // 2, size - size // 2 - 1)))
    ys, xs = np.where(support_mask)
    corners = np.array([[xs.min(), ys.min()], [xs.max(), ys.min()], [xs.min(), ys.max()], [xs.max(), ys.max()]], dtype=np.float32)
    uv_bounds = (corners - origin) @ inv.T
    umin, umax = int(np.floor(uv_bounds[:, 0].min())) - 2, int(np.ceil(uv_bounds[:, 0].max())) + 2
    vmin, vmax = int(np.floor(uv_bounds[:, 1].min())) - 2, int(np.ceil(uv_bounds[:, 1].max())) + 2
    tree = cKDTree(pts)
    out = []
    for iu in range(umin, umax + 1):
        for iv in range(vmin, vmax + 1):
            p = origin + iu * basis_u + iv * basis_v
            x, y = float(p[0]), float(p[1])
            xi, yi = int(round(x)), int(round(y))
            if not (0 <= xi < support_mask.shape[1] and 0 <= yi < support_mask.shape[0]) or not support_mask[yi, xi]:
                continue
            d, idx = tree.query([x, y], k=1)
            if float(d) < 0.35 * spacing:
                q = pts[int(idx)]
                out.append(HoleCandidate(float(q[0]), float(q[1]), float(radii[int(idx)]), 0.0, float(contrasts[int(idx)]), 0.95))
            else:
                y0, y1 = max(0, yi - 8), min(score_map.shape[0], yi + 9)
                x0, x1 = max(0, xi - 8), min(score_map.shape[1], xi + 9)
                local = score_map[y0:y1, x0:x1]
                if local.size and float(local.max()) > 0.35:
                    yy0, xx0 = np.unravel_index(int(np.argmax(local)), local.shape)
                    out.append(HoleCandidate(float(x0 + xx0), float(y0 + yy0), float(r_med), 0.0, 0.0, 0.70))
    return _deduplicate(out, 0.7)


def detect_dark_hole_candidates(image: np.ndarray, cfg: GeometryConfig) -> list[HoleCandidate]:
    try:
        grid_candidates, _debug = detect_regular_grid_hole_candidates(image, cfg, return_debug=True)
        if len(grid_candidates) >= 20:
            return grid_candidates
    except Exception:
        pass
    return _legacy_dark_blob_candidates(image, cfg)


# Exact wafer-sequence detector (probe-aligned logic)
_fallback_stable_grid_detector = detect_stable_grid_hole_candidates
from holecolor.geometry.exact_sequence import detect_exact_wafer_holes_sequence


def detect_stable_grid_hole_candidates(images: Sequence[np.ndarray], cfg: GeometryConfig, reference_index: int = 0, return_debug: bool = False):
    exact = detect_exact_wafer_holes_sequence(images, cfg, reference_index=reference_index, return_debug=return_debug)
    if return_debug:
        candidates, debug = exact
        if len(candidates) > 0:
            return candidates, debug
        return _fallback_stable_grid_detector(images, cfg, reference_index=reference_index, return_debug=True)
    candidates = exact
    if len(candidates) > 0:
        return candidates
    return _fallback_stable_grid_detector(images, cfg, reference_index=reference_index, return_debug=False)
