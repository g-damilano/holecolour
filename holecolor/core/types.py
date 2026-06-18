from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

FrameArray = np.ndarray


@dataclass(slots=True)
class VideoMeta:
    source_path: Path
    fps: float
    n_frames: int
    width: int
    height: int
    channels: int
    pixel_size_um: Optional[float] = None


@dataclass(slots=True)
class FrameRecord:
    frame_id: int
    time_s: float
    image: FrameArray


@dataclass(slots=True)
class AuditRecord:
    frame_id: int
    blur_score: float
    sat_frac_r: float
    sat_frac_g: float
    sat_frac_b: float
    frame_jump_score: float
    accepted: bool


@dataclass(slots=True)
class PhotometryScore:
    frame_id: int
    correction_name: str
    edge_energy_ratio: float
    positive_prominence: float
    hole_drift_score: float | None
    total_score: float


@dataclass(slots=True)
class HoleCandidate:
    x: float
    y: float
    radius_px: float
    ellipticity: float
    boundary_contrast: float
    confidence: float


@dataclass(slots=True)
class HoleGeometry:
    hole_id: int
    x: float
    y: float
    radius_inner_px: float
    radius_outer_px: float
    confidence: float


@dataclass(slots=True)
class LatticeModel:
    origin_x: float
    origin_y: float
    basis_u: tuple[float, float]
    basis_v: tuple[float, float]
    angle_deg: float
    spacing_u_px: float
    spacing_v_px: float
    confidence: float


@dataclass(slots=True)
class TransformRecord:
    frame_id: int
    dx: float
    dy: float
    angle_deg: float = 0.0
    scale: float = 1.0


@dataclass(slots=True)
class HotspotTrackSummary:
    track_id: int
    start_frame: int
    end_frame: int
    duration_frames: int
    mean_area_px: float
    mean_score: float
    dominant_hole_id: int | None
    mean_dist_to_hole_px: float | None
