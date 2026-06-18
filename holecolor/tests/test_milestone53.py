from __future__ import annotations

import math
import time

import cv2
import numpy as np
import pytest

from holecolor.config.schema import GeometryConfig
from holecolor.core.types import HoleCandidate
from holecolor.geometry.candidates import detect_stable_grid_hole_candidates
from holecolor.geometry.exact_sequence import (
    _background_boundary_circle_candidates,
    _background_similarity_map,
    _circle_background_transition_score,
    _circle_mask,
    _DetectorWatchdog,
    _detect_support_from_sequence,
    _fine_refine_circle_from_visible_rim,
    _gradient_magnitude,
    _homogeneous_midpoint_order,
    _next_midpoint_expansion_indices,
    _representative_sequence_indices,
    _robust_scale,
    _scale_detection_result,
    _sequence_gray_stack,
    _support_circle_has_visible_boundary,
    _wafer_circle_model_score,
    DetectorWatchdogTimeout,
    detect_exact_wafer_holes_sequence_full,
    StableGridDetectionDebug,
    StableGridDetectionResult,
)
from holecolor.core.types import LatticeModel
from holecolor.geometry.lattice_fit import assign_lattice_indices, estimate_lattice_basis
from holecolor.pipeline import _geometry_sanity_checks


def _make_sequence(shape=(420, 420), radius=8, n_frames=6):
    h, w = shape
    cx, cy, wr = w // 2, h // 2, 150
    u = np.array([34.0, 0.0], dtype=np.float32)
    v = np.array([17.0, 30.0], dtype=np.float32)
    origin = np.array([cx - 3 * u[0] - 2 * v[0], cy - 3 * u[1] - 2 * v[1]], dtype=np.float32)
    yy, xx = np.indices((h, w))
    frames = []
    for t in range(n_frames):
        img = np.full((h, w, 3), 42, dtype=np.float32)
        wafer = ((xx - cx) ** 2 + (yy - cy) ** 2) <= wr * wr
        img[wafer] = 95 + 18 * math.sin(2 * math.pi * t / max(n_frames, 2))
        for i in range(7):
            for j in range(6):
                p = origin + i * u + j * v
                x, y = int(round(float(p[0]))), int(round(float(p[1])))
                if (x - cx) ** 2 + (y - cy) ** 2 > (wr - radius - 2) ** 2:
                    continue
                cv2.circle(img, (x, y), radius, (228, 228, 228), -1)
        img += np.random.default_rng(123 + t).normal(0.0, 5.0, img.shape)
        frames.append(np.clip(img, 0, 255).astype(np.uint8))
    return frames


def _make_partial_sequence(shape=(240, 280), radius=10, n_frames=5):
    h, w = shape
    cx, cy, wr = w + 55, h // 2, 175
    u = np.array([34.0, 0.0], dtype=np.float32)
    v = np.array([17.0, 30.0], dtype=np.float32)
    origin = np.array([cx - 7 * u[0] - 4 * v[0], cy - 4 * u[1] - 4 * v[1]], dtype=np.float32)
    yy, xx = np.indices((h, w))
    frames = []
    for t in range(n_frames):
        img = np.full((h, w, 3), 210, dtype=np.float32)
        wafer = ((xx - cx) ** 2 + (yy - cy) ** 2) <= wr * wr
        img[wafer] = 95 + 18 * math.sin(2 * math.pi * t / max(n_frames, 2))
        for i in range(15):
            for j in range(10):
                p = origin + i * u + j * v
                x, y = int(round(float(p[0]))), int(round(float(p[1])))
                if not (0 <= x < w and 0 <= y < h):
                    continue
                if (x - cx) ** 2 + (y - cy) ** 2 > (wr - radius - 3) ** 2:
                    continue
                cv2.circle(img, (x, y), radius, (228, 228, 228), -1)
        img += np.random.default_rng(321 + t).normal(0.0, 4.0, img.shape)
        frames.append(np.clip(img, 0, 255).astype(np.uint8))
    return frames


def test_milestone53_representative_sequence_sampling_spans_video_and_reference():
    idx = _representative_sequence_indices(44, 19)
    assert len(idx) < 44
    assert len(idx) >= 8
    assert idx[0] == 0
    assert idx[-1] == 43
    assert 19 in idx
    assert idx == sorted(set(idx))


def test_milestone53_homogeneous_midpoint_order_starts_with_endpoints_then_populates_gaps():
    order = _homogeneous_midpoint_order(9, 4)
    assert order[:3] == [0, 8, 4]
    assert order == [0, 8, 4, 2, 6, 1, 3, 5, 7]
    assert sorted(order) == list(range(9))


def test_milestone53_midpoint_expansion_preserves_existing_sample_and_adds_gaps():
    expanded = _next_midpoint_expansion_indices([0, 8, 4], 9, 4, count=2)
    assert expanded == [0, 2, 4, 6, 8]


def test_milestone53_reduced_fallback_result_scales_to_full_resolution():
    result = StableGridDetectionResult(
        [HoleCandidate(10.0, 20.0, 3.0, 0.0, 1.0, 0.8)],
        LatticeModel(5.0, 6.0, (7.0, 0.0), (3.0, 6.0), 0.0, 7.0, 6.7, 0.9),
        {0: (0, 0)},
        StableGridDetectionDebug(
            (50, 60, 70),
            np.ones((80, 90), dtype=bool),
            1,
            1,
            1,
            0,
            0,
            0,
            1,
            "reduced",
            3.0,
            tiers=[{"x": 10.0, "y": 20.0, "radius_px": 3.0}],
            watchdog_events=[{"reason": "watchdog_timeout", "phase": "fallback"}],
        ),
    )
    scaled = _scale_detection_result(result, 2.0, (160, 180))
    assert scaled.accepted_candidates[0].x == 20.0
    assert scaled.accepted_candidates[0].radius_px == 6.0
    assert scaled.lattice.origin_x == 10.0
    assert scaled.debug.support_circle == (100, 120, 140)
    assert scaled.debug.support_mask is not None
    assert scaled.debug.support_mask.shape == (160, 180)
    assert scaled.debug.tiers[0]["x"] == 20.0
    assert scaled.debug.watchdog_events[0]["phase"] == "fallback"


def test_milestone53_detector_watchdog_raises_with_phase_name():
    watchdog = _DetectorWatchdog("test detector", 0.001)
    time.sleep(0.003)
    with pytest.raises(DetectorWatchdogTimeout) as excinfo:
        watchdog.check("synthetic long phase")
    assert "synthetic long phase" in str(excinfo.value)
    assert excinfo.value.phase == "synthetic long phase"


def test_milestone53_skew_lattice_fit_assigns_indices():
    u = np.array([40.0, 0.0], dtype=np.float32)
    v = np.array([20.0, 35.0], dtype=np.float32)
    origin = np.array([120.0, 100.0], dtype=np.float32)
    candidates = []
    for i in range(5):
        for j in range(4):
            p = origin + i * u + j * v
            candidates.append(HoleCandidate(float(p[0]), float(p[1]), 8.0, 0.0, 1.0, 0.9))
    lattice = estimate_lattice_basis(candidates)
    assert lattice.confidence >= 0.0
    idx = assign_lattice_indices(candidates, lattice)
    assert len({tuple(v) for v in idx.values()}) >= 15
    angle = abs(math.degrees(math.atan2(lattice.basis_u[0] * lattice.basis_v[1] - lattice.basis_u[1] * lattice.basis_v[0], lattice.basis_u[0] * lattice.basis_v[0] + lattice.basis_u[1] * lattice.basis_v[1])))
    assert 35.0 <= angle <= 85.0


def test_milestone53_sequence_detector_finds_regular_candidates():
    frames = _make_sequence(shape=(260, 260), radius=6, n_frames=4)
    candidates, debug = detect_stable_grid_hole_candidates(frames, GeometryConfig(), reference_index=0, return_debug=True)
    assert debug.raw_count > 0
    assert debug.filtered_count >= 8
    assert debug.anchor_count >= 12
    assert debug.completed_count == len(candidates)
    assert debug.common_radius_px > 0
    assert len(debug.tiers) >= debug.anchor_count
    assert len(candidates) >= 20
    lattice = estimate_lattice_basis(candidates)
    assert lattice.confidence >= 0.0


def test_milestone53_sequence_detector_exposes_prediction_tiers():
    frames = _make_sequence(shape=(220, 220), radius=6, n_frames=4)
    cfg = GeometryConfig(min_radius_px=5.0, max_radius_px=16.0)
    candidates, debug = detect_stable_grid_hole_candidates(frames, cfg, reference_index=0, return_debug=True)
    assert len(candidates) == debug.completed_count
    assert debug.anchor_count > 0
    assert debug.completed_count >= debug.anchor_count
    assert debug.predicted_only_full_count >= 0
    assert debug.predicted_only_partial_count >= 0
    assert isinstance(debug.tiers, list)
    assert isinstance(debug.predicted_only, list)


def test_milestone53_exact_sequence_respects_configured_small_radius():
    frames = _make_sequence(shape=(260, 260), radius=6, n_frames=4)
    cfg = GeometryConfig(min_radius_px=5.0, max_radius_px=8.0)
    result = detect_exact_wafer_holes_sequence_full(frames, cfg, reference_index=0)
    assert result.debug.mode == 'exact_sequence'
    assert result.debug.filtered_count >= 12
    assert result.debug.anchor_count >= 12
    assert result.debug.completed_count == len(result.accepted_candidates)
    assert len(result.accepted_candidates) >= 20
    assert 5.0 <= result.debug.common_radius_px <= 8.5


def test_milestone53_exact_sequence_fits_partial_support_with_center_outside_frame():
    frames = _make_partial_sequence()
    cfg = GeometryConfig(min_radius_px=8.0, max_radius_px=14.0)
    result = detect_exact_wafer_holes_sequence_full(frames, cfg, reference_index=0)
    h, w = frames[0].shape[:2]
    assert result.debug.mode == 'exact_sequence'
    assert result.debug.support_circle is not None
    sx, sy, _ = result.debug.support_circle
    assert sx < 0 or sx >= w or sy < 0 or sy >= h
    assert result.debug.anchor_count >= 10
    assert result.debug.completed_count == len(result.accepted_candidates)
    assert len(result.accepted_candidates) >= 16


def test_milestone53_support_boundary_verifier_accepts_visible_wafer_border():
    h, w = 180, 220
    cx, cy, r = 110, 90, 55
    yy, xx = np.indices((h, w))
    mean_gray = np.full((h, w), 190, dtype=np.float32)
    wafer = ((xx - cx) ** 2 + (yy - cy) ** 2) <= r * r
    mean_gray[wafer] = 72

    assert _support_circle_has_visible_boundary(mean_gray, mean_gray, (cx, cy, r))


def test_milestone53_support_detection_falls_back_when_no_visible_wafer_border():
    h, w = 160, 240
    base = np.zeros((h, w), dtype=np.uint8)
    base[20:140, :] = 80
    for x in range(w):
        base[20:140, x] = np.clip(int(base[80, x]) + (x % 17) - 8, 0, 255)
    frames = [np.repeat(base[:, :, None], 3, axis=2).astype(np.uint8) for _ in range(4)]

    support_circle, wafer_mask, *_ = _detect_support_from_sequence(_sequence_gray_stack(frames))

    assert support_circle is None
    assert bool(np.all(wafer_mask))


def test_milestone53_geometry_sanity_allows_dense_full_frame_support():
    h, w = 200, 200
    candidates = [
        HoleCandidate(float(10 + 10 * (i % 19)), float(10 + 10 * (i // 19)), 3.0, 0.0, 1.0, 0.95)
        for i in range(181)
    ]
    lattice = LatticeModel(10.0, 10.0, (10.0, 0.0), (0.0, 10.0), 0.0, 10.0, 10.0, 0.70)
    debug = StableGridDetectionDebug(
        support_circle=(100, 100, 142),
        support_mask=np.ones((h, w), dtype=bool),
        raw_count=len(candidates),
        filtered_count=len(candidates),
        anchor_count=len(candidates),
        recovered_strong_count=0,
        predicted_only_full_count=0,
        predicted_only_partial_count=0,
        completed_count=len(candidates),
        mode="exact_sequence",
        common_radius_px=3.0,
    )
    result = StableGridDetectionResult(
        candidates,
        lattice,
        {i: (i % 19, i // 19) for i in range(len(candidates))},
        debug,
    )

    sanity = _geometry_sanity_checks(result, (h, w))

    assert sanity["fail_count"] == 0
    lattice_check = next(c for c in sanity["checks"] if c["name"] == "lattice_confidence")
    assert lattice_check["passed"] is True
    assert lattice_check["value"]["minimum"] == 0.65
    full_frame_check = next(c for c in sanity["checks"] if c["name"] == "full_frame_support_hole_count")
    assert full_frame_check["severity"] == "warn"
    assert full_frame_check["passed"] is False


def test_milestone53_wafer_circle_score_rejects_background_swallowing_circle():
    h, w = 180, 220
    cx, cy, r = 110, 90, 55
    yy, xx = np.indices((h, w))
    mean_gray = np.full((h, w), 190, dtype=np.float32)
    wafer = ((xx - cx) ** 2 + (yy - cy) ** 2) <= r * r
    mean_gray[wafer] = 72
    # A nearby dark non-wafer patch should not lure the support circle outward.
    patch = ((xx - 35) ** 2 + (yy - 120) ** 2) <= 23 ** 2
    mean_gray[patch & ~wafer] = 82
    frame0_gray = mean_gray.copy()
    rough_mask = wafer | patch
    background_similarity = _background_similarity_map(mean_gray, frame0_gray)
    true_score = _wafer_circle_model_score(
        mean_gray,
        frame0_gray,
        rough_mask,
        [],
        (cx, cy, r),
        8.0,
        None,
        background_similarity,
    )
    swallowing_score = _wafer_circle_model_score(
        mean_gray,
        frame0_gray,
        rough_mask,
        [],
        (cx - 8, cy + 4, r + 34),
        8.0,
        None,
        background_similarity,
    )
    assert int(_circle_mask((h, w), (cx, cy, r)).sum()) < int(_circle_mask((h, w), (cx - 8, cy + 4, r + 34)).sum())
    assert true_score > swallowing_score + 0.03


def test_milestone53_background_boundary_candidates_target_true_wafer_circle():
    h, w = 220, 260
    cx, cy, r = 132, 108, 72
    yy, xx = np.indices((h, w))
    wafer = ((xx - cx) ** 2 + (yy - cy) ** 2) <= r * r
    mean_gray = np.full((h, w), 210, dtype=np.float32)
    mean_gray[wafer] = 78
    patch = ((xx - 42) ** 2 + (yy - 170) ** 2) <= 24 ** 2
    mean_gray[patch & ~wafer] = 83
    frame0_gray = mean_gray.copy()
    rough_mask = wafer | patch
    anchor_rows = []
    idx = 0
    for i in range(-2, 3):
        for j in range(-2, 3):
            x = cx + 24 * i + 12 * j
            y = cy + 21 * j
            if (x - cx) ** 2 + (y - cy) ** 2 <= (r - 15) ** 2:
                anchor_rows.append(
                    {
                        "lattice_i": i,
                        "lattice_j": j,
                        "x_final": float(x),
                        "y_final": float(y),
                        "r_final": 7.0,
                        "confidence": 1.0,
                    }
                )
                idx += 1
    background_similarity = _background_similarity_map(mean_gray, frame0_gray)
    candidates = _background_boundary_circle_candidates(background_similarity, anchor_rows, 7.0, rough_mask)
    assert candidates
    best = min(candidates, key=lambda c: abs(c[2] - r) + math.hypot(c[0] - cx, c[1] - cy))
    assert math.hypot(best[0] - cx, best[1] - cy) < 8.0
    assert abs(best[2] - r) < 10.0
    assert _circle_background_transition_score(background_similarity, best, 7.0) > 0.15


def test_milestone53_fine_rim_refinement_recenters_shifted_circle():
    h, w = 260, 300
    cx, cy, r = 148, 132, 82
    yy, xx = np.indices((h, w))
    wafer = ((xx - cx) ** 2 + (yy - cy) ** 2) <= r * r
    mean_gray = np.full((h, w), 214, dtype=np.float32)
    mean_gray[wafer] = 78
    frame0_gray = mean_gray.copy()
    mean_scaled = _robust_scale(mean_gray)
    ref_scaled = _robust_scale(frame0_gray)
    grad_scaled = _robust_scale(0.65 * _gradient_magnitude(mean_scaled) + 0.35 * _gradient_magnitude(ref_scaled))
    background_similarity = _background_similarity_map(mean_gray, frame0_gray)
    shifted = (cx + 7.0, cy - 5.0, float(r))
    refined = _fine_refine_circle_from_visible_rim((mean_scaled, ref_scaled, grad_scaled), background_similarity, shifted, 8.0)

    assert refined is not None
    before = math.hypot(shifted[0] - cx, shifted[1] - cy)
    after = math.hypot(refined[0] - cx, refined[1] - cy)
    assert after < before
    assert abs(refined[2] - r) < 5.0
