from pathlib import Path

import pandas as pd

from holecolor.config.schema import PipelineConfig
from holecolor.core.types import FrameRecord
from holecolor.pipeline import run_milestone12
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
        for (cx, cy) in gt["centers"]:
            rr = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
            front = gt["radius_px"] + t * 1.4
            ring = np.exp(-((rr - front) ** 2) / (2 * (3.0 ** 2)))
            img[..., 0] += 18 * ring
            img[..., 1] += 4 * ring
            img[..., 2] += 26 * ring
        frames.append(np.clip(img, 0, 255).astype(np.uint8))
    return frames


def test_milestone12_writes_radial_archetypes_asymmetry_and_group_outputs(tmp_path: Path):
    frames = _make_small_radial_video(n_frames=6)
    frames = add_photometric_artifacts(frames, brightness_ramp=0.05, vignette_strength=0.08)
    frames, _ = add_known_drift(frames, dx_per_frame=0.2, dy_per_frame=-0.1)
    records = [FrameRecord(i, float(i), f) for i, f in enumerate(frames)]
    out_dir = tmp_path / "run_m12"
    cfg = PipelineConfig()
    summary = run_milestone12(records, out_dir, cfg)
    assert summary["n_frames"] == 6

    arche_csv = out_dir / "radial" / "per_hole_radial_archetypes.csv"
    asym_csv = out_dir / "radial" / "angular_asymmetry_timeseries.csv"
    asym_frame_csv = out_dir / "radial" / "angular_asymmetry_frame_summary.csv"
    group_csv = out_dir / "radial" / "reticulum_group_radial_comparison.csv"
    group_sum_csv = out_dir / "radial" / "reticulum_group_frame_summary.csv"
    arche_json = out_dir / "radial" / "radial_archetype_centroids.json"
    arche_png = out_dir / "radial" / "radial_archetype_counts.png"
    asym_png = out_dir / "radial" / "angular_asymmetry_frame_summary.png"
    group_png = out_dir / "radial" / "reticulum_group_comparison.png"
    for p in [arche_csv, asym_csv, asym_frame_csv, group_csv, group_sum_csv, arche_json, arche_png, asym_png, group_png]:
        assert p.exists(), p

    arche = pd.read_csv(arche_csv)
    asym = pd.read_csv(asym_csv)
    group = pd.read_csv(group_csv)
    assert {"hole_id", "radial_archetype_label", "delta_center_of_mass"}.issubset(arche.columns)
    assert {"frame_id", "hole_id", "angular_asymmetry", "vector_strength"}.issubset(asym.columns)
    assert {"reticulum_zone", "frame_id", "annulus_id", "mean_descriptor"}.issubset(group.columns)
    assert len(arche) > 0
    assert len(asym) > 0
    assert len(group) > 0


def test_milestone12_strict_asymmetry_gate_can_fail(tmp_path: Path):
    frames = _make_small_radial_video(n_frames=5)
    records = [FrameRecord(i, float(i), f) for i, f in enumerate(frames)]
    out_dir = tmp_path / "run_m12_fail"
    cfg = PipelineConfig()
    cfg.qc.fail_on_gate_error = True
    cfg.radial.min_asymmetry_valid_fraction = 1.1
    try:
        run_milestone12(records, out_dir, cfg)
    except RuntimeError as e:
        assert "angular_asymmetry_valid_fraction" in str(e)
    else:
        raise AssertionError("Expected strict milestone12 asymmetry QC failure")
