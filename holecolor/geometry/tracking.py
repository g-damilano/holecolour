from __future__ import annotations

import cv2
import numpy as np

from holecolor.core.types import HoleGeometry


def _crop_patch(gray: np.ndarray, hole: HoleGeometry, search_radius_px: float) -> tuple[np.ndarray, int, int]:
    r = max(float(hole.radius_outer_px) + float(search_radius_px), 4.0)
    x0 = max(int(round(hole.x - r)), 0)
    y0 = max(int(round(hole.y - r)), 0)
    x1 = min(int(round(hole.x + r + 1)), gray.shape[1])
    y1 = min(int(round(hole.y + r + 1)), gray.shape[0])
    return gray[y0:y1, x0:x1], x0, y0


def _prepare_frame_contours(gray: np.ndarray) -> list[np.ndarray]:
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, bw = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(bw, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    return contours


def _refine_single_hole(prev: HoleGeometry, contours: list[np.ndarray], shape: tuple[int, int], search_radius_px: float) -> HoleGeometry:
    h, w = shape
    r = max(float(prev.radius_outer_px) + float(search_radius_px), 4.0)
    x0 = max(int(round(prev.x - r)), 0)
    y0 = max(int(round(prev.y - r)), 0)
    x1 = min(int(round(prev.x + r + 1)), w)
    y1 = min(int(round(prev.y + r + 1)), h)
    if x1 <= x0 or y1 <= y0 or not contours:
        return HoleGeometry(prev.hole_id, prev.x, prev.y, prev.radius_inner_px, prev.radius_outer_px, prev.confidence * 0.95)

    target = np.array([prev.x, prev.y], dtype=np.float32)
    best: tuple[float, float, float, float] | None = None
    best_score = float("inf")
    for cnt in contours:
        bx, by, bw, bh = cv2.boundingRect(cnt)
        if bx >= x1 or by >= y1 or bx + bw <= x0 or by + bh <= y0:
            continue
        area = cv2.contourArea(cnt)
        if area <= 0:
            continue
        if len(cnt) >= 5:
            (cx, cy), (ma, mi), _ = cv2.fitEllipse(cnt)
            radius = 0.25 * (float(ma) + float(mi))
        else:
            (cx, cy), radius = cv2.minEnclosingCircle(cnt)
            radius = float(radius)
        if radius <= 1.0:
            continue
        center = np.array([cx, cy], dtype=np.float32)
        center_penalty = float(np.linalg.norm(center - target))
        radius_penalty = abs(radius - prev.radius_outer_px)
        score = center_penalty + 0.6 * radius_penalty
        if score < best_score:
            best_score = score
            best = (float(cx), float(cy), float(radius), float(radius))
    if best is None:
        return HoleGeometry(prev.hole_id, prev.x, prev.y, prev.radius_inner_px, prev.radius_outer_px, prev.confidence * 0.95)
    cx, cy, _r1, _r = best
    conf = float(np.clip(prev.confidence * (1.0 / (1.0 + best_score / max(prev.radius_outer_px, 1e-6))), 0.0, 1.0))
    return HoleGeometry(prev.hole_id, cx, cy, prev.radius_inner_px, prev.radius_outer_px, conf)


def propagate_geometry_to_frame(prev_holes: list[HoleGeometry], frame: np.ndarray, search_radius_px: float = 5.0) -> list[HoleGeometry]:
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    contours = _prepare_frame_contours(gray)
    refined = [_refine_single_hole(h, contours, gray.shape[:2], search_radius_px=search_radius_px) for h in prev_holes]
    return sorted(refined, key=lambda h: h.hole_id)


def smooth_hole_trajectories(holes_by_frame: dict[int, list[HoleGeometry]], window: int = 3) -> dict[int, list[HoleGeometry]]:
    if window <= 1 or not holes_by_frame:
        return holes_by_frame
    frame_ids = sorted(holes_by_frame)
    hole_ids = sorted({h.hole_id for holes in holes_by_frame.values() for h in holes})
    by_frame_hole = {(fid, h.hole_id): h for fid, holes in holes_by_frame.items() for h in holes}
    smoothed: dict[int, list[HoleGeometry]] = {fid: [] for fid in frame_ids}
    pad = window // 2
    for hole_id in hole_ids:
        xs = []
        ys = []
        rin = []
        rout = []
        conf = []
        for fid in frame_ids:
            h = by_frame_hole[(fid, hole_id)]
            xs.append(h.x)
            ys.append(h.y)
            rin.append(h.radius_inner_px)
            rout.append(h.radius_outer_px)
            conf.append(h.confidence)
        def _smooth(arr: list[float]) -> np.ndarray:
            vals = np.asarray(arr, dtype=float)
            padded = np.pad(vals, (pad, pad), mode="edge")
            ker = np.ones(window, dtype=float) / float(window)
            return np.convolve(padded, ker, mode="valid")
        xs_s = _smooth(xs)
        ys_s = _smooth(ys)
        rin_s = _smooth(rin)
        rout_s = _smooth(rout)
        conf_s = _smooth(conf)
        for idx, fid in enumerate(frame_ids):
            smoothed[fid].append(HoleGeometry(hole_id, float(xs_s[idx]), float(ys_s[idx]), float(rin_s[idx]), float(rout_s[idx]), float(conf_s[idx])))
    for fid in frame_ids:
        smoothed[fid] = sorted(smoothed[fid], key=lambda h: h.hole_id)
    return smoothed
