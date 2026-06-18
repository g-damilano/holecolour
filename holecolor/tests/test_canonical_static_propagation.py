from __future__ import annotations

import csv
from pathlib import Path

import cv2
import numpy as np

from holecolor.config.schema import PipelineConfig
from holecolor.core.types import FrameRecord, HoleGeometry
from holecolor.geometry.tracking import propagate_geometry_to_frame
from holecolor.pipeline import _propagate_holes_across_frames, run_milestone24
from holecolor.tests.test_milestone14 import _make_small_rdf_video


def test_propagation_canonical_static_copies_reference_geometry() -> None:
    frames = _make_small_rdf_video(n_frames=3)
    records = [FrameRecord(i, float(i), f) for i, f in enumerate(frames)]
    ref_holes = [
        HoleGeometry(0, 10.0, 20.0, 4.0, 6.0, 0.9),
        HoleGeometry(1, 30.0, 40.0, 5.0, 7.0, 0.8),
    ]
    cfg = PipelineConfig()
    cfg.geometry.propagation_mode = "canonical_static"
    out = _propagate_holes_across_frames(records, ref_idx=1, ref_holes=ref_holes, cfg=cfg, progress_cfg=None)
    assert sorted(out) == [0, 1, 2]
    expected = [
        (0, 10.0, 20.0, 4.0, 6.0, 0.9),
        (1, 30.0, 40.0, 5.0, 7.0, 0.8),
    ]
    for fid, holes in out.items():
        got = [(h.hole_id, h.x, h.y, h.radius_inner_px, h.radius_outer_px, h.confidence) for h in holes]
        assert got == expected


def test_run_milestone24_geometry_timeseries_static_by_default(tmp_path: Path) -> None:
    frames = _make_small_rdf_video(n_frames=3)
    records = [FrameRecord(i, float(i), f) for i, f in enumerate(frames)]
    out_dir = tmp_path / "run_static"
    cfg = PipelineConfig()
    cfg.parallel.enabled = False
    cfg.parallel.backend = "none"
    cfg.parallel.show_progress = False
    # default should already be canonical_static, but make it explicit for test clarity
    cfg.geometry.propagation_mode = "canonical_static"

    summary = run_milestone24(records, out_dir, cfg)
    assert summary["n_frames"] == 3

    geom_csv = out_dir / "geometry" / "hole_geometry_timeseries.csv"
    assert geom_csv.exists()
    by_hole = {}
    with geom_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            by_hole.setdefault(int(row["hole_id"]), []).append((
                float(row["x"]),
                float(row["y"]),
                float(row["radius_inner_px"]),
                float(row["radius_outer_px"]),
            ))
    assert by_hole
    for seq in by_hole.values():
        assert len(set(seq)) == 1


def test_dynamic_tracking_keeps_physical_hole_radius_invariant() -> None:
    frame = np.full((64, 64, 3), 220, dtype=np.uint8)
    cv2.circle(frame, (32, 32), 13, (20, 20, 20), -1)
    prev = [HoleGeometry(0, 32.0, 32.0, 5.0, 7.0, 0.9)]

    propagated = propagate_geometry_to_frame(prev, frame, search_radius_px=10.0)

    assert len(propagated) == 1
    assert propagated[0].radius_inner_px == 5.0
    assert propagated[0].radius_outer_px == 7.0
