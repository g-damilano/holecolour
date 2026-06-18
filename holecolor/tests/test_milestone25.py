from __future__ import annotations

import cv2
import numpy as np

from holecolor.config.schema import PipelineConfig
from holecolor.geometry.candidates import detect_regular_grid_hole_candidates


def _make_wafer_grid_image() -> np.ndarray:
    h, w = 720, 960
    img = np.full((h, w, 3), 185, dtype=np.uint8)
    # steel-like background streaks
    rng = np.random.default_rng(7)
    noise = rng.normal(0, 18, size=(h, w)).astype(np.int16)
    for ch in range(3):
        img[..., ch] = np.clip(img[..., ch].astype(np.int16) + noise, 0, 255)
    # diagonal scratches
    for y0 in range(40, h, 36):
        x0 = 0
        x1 = w - 1
        y1 = int(np.clip(y0 + 0.28 * w, 0, h - 1))
        cv2.line(img, (x0, y0), (x1, y1), (205, 205, 205), 1)

    wafer_center = (470, 380)
    wafer_r = 190
    cv2.circle(img, wafer_center, wafer_r, (88, 88, 88), -1)

    # hex-like regular grid of bright holes
    r = 10
    dx = 30
    dy = int(round(dx * np.sqrt(3) / 2))
    for j in range(-8, 9):
        for i in range(-8, 9):
            x = wafer_center[0] + i * dx + (j % 2) * (dx // 2)
            y = wafer_center[1] + j * dy
            if (x - wafer_center[0]) ** 2 + (y - wafer_center[1]) ** 2 < (wafer_r - 18) ** 2:
                cv2.circle(img, (int(x), int(y)), r, (155, 155, 155), -1)
    return img


def test_milestone25_regular_grid_detector_handles_wafer_scene():
    img = _make_wafer_grid_image()
    cfg = PipelineConfig()
    candidates, debug = detect_regular_grid_hole_candidates(img, cfg.geometry, return_debug=True)
    assert debug.mode in {"circle", "density", "fullframe"}
    assert len(candidates) >= 45
