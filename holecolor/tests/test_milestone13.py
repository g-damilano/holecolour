from pathlib import Path

import pandas as pd

from holecolor.config.schema import PipelineConfig
from holecolor.core.types import FrameRecord
from holecolor.pipeline import run_milestone13
from holecolor.synth.artifacts import add_known_drift, add_photometric_artifacts
from holecolor.synth.grid import make_synthetic_grid


def _make_small_radial_video(n_frames: int = 6):
    base, gt = make_synthetic_grid(shape=(256, 256), rows=3, cols=4, radius_px=12, spacing_px=46, rotation_deg=6.0)
    frames = []
    import numpy as np
    h, w, _ = base.shape
    yy, xx = np.indices((h, w))
    for t in range(n_frames):
        img = base.astype(np.float32).copy()
        for i, (cx, cy) in enumerate(gt["centers"]):
            rr = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
            front = gt["radius_px"] + t * 1.1 + (i % 3) * 0.25
            ring = np.exp(-((rr - front) ** 2) / (2 * (2.6 ** 2)))
            theta = np.arctan2(yy - cy, xx - cx)
            sector_bias = 1.0 + 0.25 * np.cos(theta - 0.6 * i)
            img[..., 0] += 15 * ring * sector_bias
            img[..., 1] += 5 * ring
            img[..., 2] += 22 * ring * sector_bias
        frames.append(np.clip(img, 0, 255).astype(np.uint8))
    return frames


def test_milestone13_writes_model_sector_and_hotspot_reticulum_outputs(tmp_path: Path):
    frames = _make_small_radial_video(n_frames=6)
    frames = add_photometric_artifacts(frames, brightness_ramp=0.05, vignette_strength=0.08)
    frames, _ = add_known_drift(frames, dx_per_frame=0.2, dy_per_frame=-0.1)
    records = [FrameRecord(i, float(i), f) for i, f in enumerate(frames)]
    out_dir = tmp_path / "run_m13"
    cfg = PipelineConfig()
    summary = run_milestone13(records, out_dir, cfg)
    assert summary["n_frames"] == 6

    model_csv = out_dir / "radial" / "per_hole_radial_model_fits.csv"
    model_sum_csv = out_dir / "radial" / "per_hole_radial_model_summary.csv"
    sector_csv = out_dir / "radial" / "sector_radial_timeseries.csv"
    sector_front_csv = out_dir / "radial" / "sector_front_summary.csv"
    sector_hole_csv = out_dir / "radial" / "sector_front_hole_summary.csv"
    hotspot_cmp_csv = out_dir / "radial" / "hole_hotspot_reticulum_comparison.csv"
    hotspot_group_csv = out_dir / "radial" / "hotspot_reticulum_group_summary.csv"
    model_png = out_dir / "radial" / "radial_model_fit_quality.png"
    sector_png = out_dir / "radial" / "sector_front_summary.png"
    hotspot_png = out_dir / "radial" / "hotspot_reticulum_group_comparison.png"
    for p in [model_csv, model_sum_csv, sector_csv, sector_front_csv, sector_hole_csv, hotspot_cmp_csv, hotspot_group_csv, model_png, sector_png, hotspot_png]:
        assert p.exists(), p

    model_df = pd.read_csv(model_csv)
    sector_df = pd.read_csv(sector_csv)
    front_df = pd.read_csv(sector_front_csv)
    hotspot_df = pd.read_csv(hotspot_cmp_csv)
    assert {"frame_id", "hole_id", "linear_slope", "linear_r2"}.issubset(model_df.columns)
    assert {"frame_id", "hole_id", "annulus_id", "sector_id", "descriptor_value"}.issubset(sector_df.columns)
    assert {"frame_id", "hole_id", "sector_id", "sector_front_radius"}.issubset(front_df.columns)
    assert {"hole_id", "reticulum_zone", "hotspot_proximity_bucket", "radial_conclusion_label"}.issubset(hotspot_df.columns)
    assert len(model_df) > 0
    assert len(sector_df) > 0
    assert len(front_df) > 0
    assert len(hotspot_df) > 0
