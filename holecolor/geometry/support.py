from __future__ import annotations

from typing import Sequence

import cv2
import numpy as np

from holecolor.geometry.exact_sequence import _detect_support_from_sequence, _sequence_gray_stack
from holecolor.geometry.models import BufferGeometry, WaferGeometry


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


def _angle_coverage_deg(points: np.ndarray, center_xy: tuple[float, float]) -> float:
    if points.shape[0] == 0:
        return 0.0
    cx, cy = center_xy
    ang = (np.degrees(np.arctan2(points[:, 1] - cy, points[:, 0] - cx)) + 360.0) % 360.0
    bins = np.unique(np.floor(ang / 12.0).astype(int))
    return float(len(bins) * 12.0)


def _candidate_geometry_score(
    *,
    cx: float,
    cy: float,
    r: float,
    pts: np.ndarray,
    mask: np.ndarray,
    wafer_mask: np.ndarray,
    wx: float,
    wy: float,
    wr: float,
    W: int,
    H: int,
) -> tuple[float, dict[str, float | bool | str]]:
    radial = np.sqrt((pts[:, 0] - cx) ** 2 + (pts[:, 1] - cy) ** 2)
    residual = float(np.median(np.abs(radial - r))) if pts.size else float("inf")
    coverage = _angle_coverage_deg(pts, (cx, cy))
    center_outside = not (0.0 <= cx < W and 0.0 <= cy < H)
    edge_touches = bool(np.any((pts[:, 0] <= 1) | (pts[:, 0] >= W - 2) | (pts[:, 1] <= 1) | (pts[:, 1] >= H - 2)))
    area_frac = float(mask.sum() / max(int(wafer_mask.sum()), 1))
    local_patch = bool(r < 0.45 * wr and area_frac < 0.20)
    center_dist = float(np.hypot(cx - wx, cy - wy))
    # prefer candidates that sit plausibly inside or around the wafer and have broad arc support
    radius_ratio = float(r / max(wr, 1e-6))
    radius_score = float(np.clip(1.0 - abs(radius_ratio - 0.65) / 0.55, 0.0, 1.0))
    residual_score = float(np.clip(1.0 - residual / max(r, 1e-6), 0.0, 1.0))
    coverage_score = float(np.clip(coverage / 360.0, 0.0, 1.0))
    area_score = float(np.clip(area_frac / 0.22, 0.0, 1.0))
    center_score = float(np.clip(1.0 - center_dist / max(1.5 * wr, 1e-6), 0.0, 1.0))
    partial_bonus = 0.08 if (center_outside or edge_touches) else 0.0
    penalties = (0.22 if local_patch else 0.0)
    score = 0.34 * coverage_score + 0.22 * residual_score + 0.16 * area_score + 0.16 * radius_score + 0.12 * center_score + partial_bonus - penalties
    return float(score), {
        "coverage": coverage,
        "residual": residual,
        "area_frac": area_frac,
        "center_outside": center_outside,
        "edge_touches": edge_touches,
        "local_patch": local_patch,
        "radius_ratio": radius_ratio,
    }


def _detect_buffer_geometry_from_sequence(gray_stack: np.ndarray, wafer_mask: np.ndarray | None) -> BufferGeometry:
    H, W = gray_stack.shape[1:]
    if wafer_mask is None or not np.any(wafer_mask):
        return BufferGeometry(
            id="buffer-0",
            state="unknown",
            center_xy_px=None,
            radius_px=None,
            confidence=0.0,
            visible_arc_intervals_deg=[],
            center_outside_frame=False,
            detection_mode="support_unavailable",
            notes="Wafer support unavailable; cannot estimate buffer border.",
        )

    mean_gray = gray_stack.mean(axis=0).astype(np.float32)
    median_gray = np.median(gray_stack, axis=0).astype(np.float32)
    std_gray = gray_stack.std(axis=0).astype(np.float32)
    mad_gray = np.median(np.abs(gray_stack - median_gray[None, ...]), axis=0).astype(np.float32)
    ptp_gray = (gray_stack.max(axis=0) - gray_stack.min(axis=0)).astype(np.float32)

    vals = np.concatenate([std_gray[wafer_mask].ravel(), mad_gray[wafer_mask].ravel(), ptp_gray[wafer_mask].ravel()])
    lo, hi = np.percentile(vals, [2.0, 98.0]) if vals.size else (0.0, 1.0)
    den = max(float(hi - lo), 1e-6)
    variability = np.clip((0.45 * std_gray + 0.30 * mad_gray + 0.25 * ptp_gray - lo) / den, 0.0, 1.0)

    yy, xx = np.indices((H, W))
    ys, xs = np.where(wafer_mask)
    if ys.size == 0 or xs.size == 0:
        return BufferGeometry(
            id="buffer-0", state="unknown", center_xy_px=None, radius_px=None, confidence=0.0,
            visible_arc_intervals_deg=[], center_outside_frame=False, detection_mode="empty_wafer_mask",
            notes="Wafer mask had no support pixels.",
        )
    wx = 0.5 * (float(xs.min()) + float(xs.max()))
    wy = 0.5 * (float(ys.min()) + float(ys.max()))
    wr = max(float(xs.max() - xs.min()), float(ys.max() - ys.min())) / 2.0
    d = np.sqrt((xx - wx) ** 2 + (yy - wy) ** 2)
    inner_search = wafer_mask & (d <= 0.92 * wr)
    raw_score = (variability * inner_search).astype(np.float32)

    candidate_payloads: list[tuple[float, tuple[float, float, float], np.ndarray, np.ndarray, dict[str, float | bool | str], str]] = []
    for sigma in (3.0, 5.0, 7.0):
        score = cv2.GaussianBlur(raw_score, (0, 0), sigma)
        positive = score[inner_search]
        if positive.size < 50:
            continue
        for pct in (82.0, 86.0, 90.0):
            thr = max(float(np.percentile(positive, pct)), 0.16)
            mask = (score >= thr) & inner_search
            mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((21, 21), np.uint8))
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8)).astype(bool)
            if not np.any(mask):
                continue
            cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            if not cnts:
                continue
            cnt = max(cnts, key=cv2.contourArea)
            pts = cnt.reshape(-1, 2).astype(np.float64)
            fit = _fit_circle_least_squares(pts)
            if fit is None:
                (xc, yc), rr = cv2.minEnclosingCircle(cnt)
                fit = (float(xc), float(yc), float(rr))
            cx, cy, r = fit
            if r <= 5.0:
                continue
            score_val, meta = _candidate_geometry_score(
                cx=cx, cy=cy, r=r, pts=pts, mask=mask, wafer_mask=wafer_mask, wx=wx, wy=wy, wr=wr, W=W, H=H
            )
            candidate_payloads.append((score_val, (cx, cy, r), pts, mask, meta, f"sequence_variability_circle_fit:sigma={sigma}:p={pct}"))

    if not candidate_payloads:
        return BufferGeometry(
            id="buffer-0", state="unknown", center_xy_px=None, radius_px=None, confidence=0.0,
            visible_arc_intervals_deg=[], center_outside_frame=False, detection_mode="threshold_empty",
            notes="Variability threshold did not yield a plausible buffer region.",
        )

    candidate_payloads.sort(key=lambda x: x[0], reverse=True)
    best_score, (cx, cy, r), pts, mask, meta, mode = candidate_payloads[0]
    coverage = float(meta["coverage"])
    residual = float(meta["residual"])
    center_outside = bool(meta["center_outside"])
    edge_touches = bool(meta["edge_touches"])
    local_patch = bool(meta["local_patch"])
    area_frac = float(meta["area_frac"])

    partial = center_outside or coverage < 300.0 or edge_touches
    state = 'partial' if partial else 'full'
    conf = float(np.clip(best_score, 0.0, 1.0))
    if local_patch and not center_outside:
        state = 'unknown'
        conf *= 0.35
    if conf < 0.12:
        return BufferGeometry(
            id="buffer-0", state="unknown", center_xy_px=None, radius_px=None, confidence=conf,
            visible_arc_intervals_deg=[], center_outside_frame=False, detection_mode="low_confidence",
            notes="Buffer circle fit confidence below threshold.",
        )
    return BufferGeometry(
        id="buffer-0",
        state=state,
        center_xy_px=(float(cx), float(cy)),
        radius_px=float(r),
        confidence=conf,
        visible_arc_intervals_deg=[(0.0, min(360.0, coverage))],
        center_outside_frame=center_outside,
        detection_mode=mode,
        notes=f"Estimated from temporal variability inside wafer support; residual={residual:.3f}, area_frac={area_frac:.3f}.",
    )


def detect_support_geometries_from_sequence(images: Sequence[np.ndarray], reference_index: int = 0) -> tuple[WaferGeometry, BufferGeometry, np.ndarray | None]:
    if not images:
        raise ValueError("images cannot be empty")
    gray_stack = _sequence_gray_stack(images)
    support_circle, wafer_mask, *_ = _detect_support_from_sequence(gray_stack)
    H, W = gray_stack.shape[1:]
    if support_circle is None:
        wafer = WaferGeometry(
            id="wafer-0",
            center_xy_px=(0.5 * W, 0.5 * H),
            radius_px=0.0,
            confidence=0.0,
            visible_arc_intervals_deg=[],
            detection_mode="sequence_support_unknown",
            notes="Support circle could not be estimated from the stabilized sequence.",
        )
        mask = None
    else:
        wafer = WaferGeometry(
            id="wafer-0",
            center_xy_px=(float(support_circle[0]), float(support_circle[1])),
            radius_px=float(support_circle[2]),
            confidence=0.85,
            visible_arc_intervals_deg=[(0.0, 360.0)],
            detection_mode="sequence_support_circle",
            notes="Estimated from stabilized sequence support evidence.",
        )
        mask = np.asarray(wafer_mask, dtype=bool)
    buffer = _detect_buffer_geometry_from_sequence(gray_stack, mask)
    return wafer, buffer, mask


def draw_support_overlay(image: np.ndarray, wafer: WaferGeometry, buffer: BufferGeometry | None = None) -> np.ndarray:
    out = image.copy()
    if wafer.radius_px > 0:
        cv2.circle(out, (int(round(wafer.center_xy_px[0])), int(round(wafer.center_xy_px[1]))), int(round(wafer.radius_px)), (255, 255, 255), 2, lineType=cv2.LINE_AA)
    if buffer is not None and buffer.center_xy_px is not None and buffer.radius_px is not None:
        cv2.circle(out, (int(round(buffer.center_xy_px[0])), int(round(buffer.center_xy_px[1]))), int(round(buffer.radius_px)), (0, 255, 255), 2, lineType=cv2.LINE_AA)
    return out
