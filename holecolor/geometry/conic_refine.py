from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from holecolor.core.types import HoleCandidate, HoleGeometry


@dataclass(slots=True)
class ConicFit:
    cx: float
    cy: float
    major_px: float
    minor_px: float
    angle_deg: float
    residual: float


def _crop_window(gray: np.ndarray, candidate: HoleCandidate, pad_factor: float = 2.0):
    r = max(float(candidate.radius_px), 2.0)
    pad = int(round(pad_factor * r))
    x0 = max(int(round(candidate.x)) - pad, 0)
    y0 = max(int(round(candidate.y)) - pad, 0)
    x1 = min(int(round(candidate.x)) + pad + 1, gray.shape[1])
    y1 = min(int(round(candidate.y)) + pad + 1, gray.shape[0])
    return gray[y0:y1, x0:x1], x0, y0


def refine_candidate_with_conic(gray: np.ndarray, candidate: HoleCandidate) -> ConicFit:
    crop, x0, y0 = _crop_window(gray, candidate)
    if crop.size == 0:
        return ConicFit(candidate.x, candidate.y, 2 * candidate.radius_px, 2 * candidate.radius_px, 0.0, 1.0)
    blur = cv2.GaussianBlur(crop, (5, 5), 0)
    _, bw = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(bw, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return ConicFit(candidate.x, candidate.y, 2 * candidate.radius_px, 2 * candidate.radius_px, 0.0, 1.0)

    target = np.array([candidate.x - x0, candidate.y - y0], dtype=np.float32)
    best = None
    best_score = float('inf')
    for cnt in contours:
        if len(cnt) < 5:
            continue
        area = cv2.contourArea(cnt)
        if area <= 0:
            continue
        ellipse = cv2.fitEllipse(cnt)
        (cx, cy), (ma, mi), angle = ellipse
        center = np.array([cx, cy], dtype=np.float32)
        center_penalty = float(np.linalg.norm(center - target))
        radius = 0.25 * (ma + mi)
        radius_penalty = abs(radius - candidate.radius_px)
        score = center_penalty + 0.5 * radius_penalty
        if score < best_score:
            best_score = score
            best = ellipse
    if best is None:
        return ConicFit(candidate.x, candidate.y, 2 * candidate.radius_px, 2 * candidate.radius_px, 0.0, 1.0)

    (cx, cy), (ma, mi), angle = best
    residual = float(best_score / max(candidate.radius_px, 1e-6))
    return ConicFit(cx + x0, cy + y0, float(ma), float(mi), float(angle), residual)


def conic_to_hole_geometry(fit: ConicFit, hole_id: int, candidate_confidence: float = 1.0) -> HoleGeometry:
    outer = max(0.25 * (fit.major_px + fit.minor_px), 1.0)
    inner = max(outer - 2.0, 1.0)
    conf = float(np.clip(candidate_confidence * (1.0 / (1.0 + fit.residual)), 0.0, 1.0))
    return HoleGeometry(hole_id, float(fit.cx), float(fit.cy), float(inner), float(outer), conf)
