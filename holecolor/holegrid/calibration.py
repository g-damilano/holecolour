from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import cv2
import numpy as np

from holecolor.config.schema import GeometryConfig
from holecolor.core.types import HoleGeometry
from holecolor.geometry.candidates import detect_regular_grid_hole_candidates
from holecolor.geometry.conic_refine import conic_to_hole_geometry, refine_candidate_with_conic
from holecolor.geometry.exact_sequence import detect_exact_wafer_holes_sequence_full, warmup_exact_sequence_numba
from holecolor.geometry.lattice_fit import estimate_lattice_basis
from holecolor.holegrid.model import HoleGridBundle


@dataclass(slots=True)
class CalibrationResult:
    frame_id: int
    holes: list[HoleGeometry]
    lattice: any
    bundle: HoleGridBundle
    support_mode: str
    support_circle: tuple[int, int, int] | None
    raw_count: int
    filtered_count: int
    completed_count: int


def calibrate_hole_grid_from_frames(frames: Iterable, geometry_cfg: GeometryConfig, sample_limit: int = 8) -> CalibrationResult:
    if hasattr(frames, "__len__") and hasattr(frames, "__getitem__"):
        frame_seq = list(frames)
        if len(frame_seq) > sample_limit:
            idx = np.linspace(0, len(frame_seq) - 1, num=sample_limit, dtype=int)
            sampled = [frame_seq[int(i)] for i in idx]
        else:
            sampled = frame_seq
    else:
        sampled = []
        for i, frame in enumerate(frames):
            if i >= sample_limit:
                break
            sampled.append(frame)
    if not sampled:
        raise ValueError('no frames provided for hole-grid calibration')

    best = None
    best_score = -1
    warmup_exact_sequence_numba()
    sequence_result = detect_exact_wafer_holes_sequence_full([frame.image for frame in sampled], geometry_cfg, reference_index=0)
    if len(sequence_result.accepted_candidates) >= 8:
        frame = sampled[0]
        candidates = sequence_result.accepted_candidates
        debug = sequence_result.debug
        holes = [
            HoleGeometry(
                i,
                float(c.x),
                float(c.y),
                max(float(c.radius_px) - 2.0, 1.0),
                max(float(c.radius_px), 1.0),
                float(c.confidence),
            )
            for i, c in enumerate(candidates)
        ]
        lattice = sequence_result.lattice
    else:
        for frame in sampled:
            cands, debug = detect_regular_grid_hole_candidates(frame.image, geometry_cfg, return_debug=True)
            score = len(cands)
            if score > best_score:
                best_score = score
                best = (frame, cands, debug)
        frame, candidates, debug = best
        gray = cv2.cvtColor(frame.image, cv2.COLOR_RGB2GRAY)
        holes = [conic_to_hole_geometry(refine_candidate_with_conic(gray, c), i, c.confidence) for i, c in enumerate(candidates)]
        lattice = estimate_lattice_basis(candidates, angle_tolerance_deg=geometry_cfg.angle_tolerance_deg)
    bundle = HoleGridBundle(
        version='1.0',
        support_mode=debug.mode,
        source_frame_id=int(frame.frame_id),
        source_note='auto-calibrated from representative frame',
        support_circle=debug.support_circle,
        lattice=lattice,
        holes=holes,
    )
    return CalibrationResult(
        frame_id=int(frame.frame_id),
        holes=holes,
        lattice=lattice,
        bundle=bundle,
        support_mode=debug.mode,
        support_circle=debug.support_circle,
        raw_count=int(debug.raw_count),
        filtered_count=int(debug.filtered_count),
        completed_count=int(debug.completed_count),
    )
