from pathlib import Path

import pandas as pd

from holecolor.config.schema import PipelineConfig
from holecolor.core.types import FrameRecord
from holecolor.pipeline import run_milestone14
from holecolor.synth.artifacts import add_known_drift, add_photometric_artifacts
from holecolor.synth.grid import make_synthetic_grid


def _make_small_rdf_video(n_frames: int = 6):
    base, gt = make_synthetic_grid(shape=(256, 256), rows=3, cols=4, radius_px=12, spacing_px=46, rotation_deg=6.0)
    frames = []
    import numpy as np
    h, w, _ = base.shape
    yy, xx = np.indices((h, w))
    for t in range(n_frames):
        img = base.astype(np.float32).copy()
        for i, (cx, cy) in enumerate(gt["centers"]):
            rr = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
            front = gt["radius_px"] + t * 1.0 + (i % 2) * 0.2
            shell = np.exp(-((rr - front) ** 2) / (2 * (2.4 ** 2)))
            img[..., 0] += 10 * shell
            img[..., 1] += 8 * shell
            img[..., 2] += 24 * shell
        frames.append(np.clip(img, 0, 255).astype(np.uint8))
    return frames


def test_milestone14_writes_per_hole_rdf_outputs(tmp_path: Path):
    frames = _make_small_rdf_video(n_frames=6)
    frames = add_photometric_artifacts(frames, brightness_ramp=0.05, vignette_strength=0.08)
    frames, _ = add_known_drift(frames, dx_per_frame=0.2, dy_per_frame=-0.1)
    records = [FrameRecord(i, float(i), f) for i, f in enumerate(frames)]
    out_dir = tmp_path / "run_m14"
    cfg = PipelineConfig()
    summary = run_milestone14(records, out_dir, cfg)
    assert summary["n_frames"] == 6

    rdf_csv = out_dir / "radial" / "per_hole_rdf_evolution.csv"
    rdf_frame_csv = out_dir / "radial" / "per_hole_rdf_frame_summary.csv"
    rdf_vel_csv = out_dir / "radial" / "per_hole_rdf_velocity_summary.csv"
    rdf_png = out_dir / "radial" / "per_hole_rdf_evolution.png"
    vel_png = out_dir / "radial" / "per_hole_rdf_front_velocity.png"
    for p in [rdf_csv, rdf_frame_csv, rdf_vel_csv, rdf_png, vel_png]:
        assert p.exists(), p

    rdf_df = pd.read_csv(rdf_csv)
    frame_df = pd.read_csv(rdf_frame_csv)
    vel_df = pd.read_csv(rdf_vel_csv)
    assert {"hole_id", "frame_id", "annulus_id", "rdf_pdf", "rdf_cdf", "delta_descriptor_value"}.issubset(rdf_df.columns)
    assert {"hole_id", "frame_id", "rdf_front_radius_norm", "rdf_total_positive_delta"}.issubset(frame_df.columns)
    assert {"hole_id", "rdf_front_velocity_per_frame", "delta_front_radius_norm", "n_frames"}.issubset(vel_df.columns)
    assert len(rdf_df) > 0
    assert len(frame_df) > 0
    assert len(vel_df) > 0
