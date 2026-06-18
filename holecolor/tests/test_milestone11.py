from pathlib import Path

import pandas as pd

from holecolor.config.schema import PipelineConfig
from holecolor.core.types import FrameRecord
from holecolor.pipeline import run_milestone11
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


def test_milestone11_writes_radial_evolution_and_validation_sweeps(tmp_path: Path):
    frames = _make_small_radial_video(n_frames=6)
    frames = add_photometric_artifacts(frames, brightness_ramp=0.05, vignette_strength=0.08)
    frames, _ = add_known_drift(frames, dx_per_frame=0.2, dy_per_frame=-0.1)
    records = [FrameRecord(i, float(i), f) for i, f in enumerate(frames)]
    out_dir = tmp_path / "run_m11"
    cfg = PipelineConfig()
    summary = run_milestone11(records, out_dir, cfg)
    assert summary["n_frames"] == 6

    rdf_csv = out_dir / "radial" / "radial_distribution_evolution.csv"
    rdf_png = out_dir / "radial" / "radial_distribution_evolution.png"
    sweep_csv = out_dir / "qc" / "radial_perturbation_sweeps.csv"
    drift_csv = out_dir / "qc" / "radial_perturbation_drift.csv"
    consistency_json = out_dir / "qc" / "radial_conclusion_consistency.json"
    drift_png = out_dir / "qc" / "radial_perturbation_drift_map.png"
    assert rdf_csv.exists()
    assert rdf_png.exists()
    assert sweep_csv.exists()
    assert drift_csv.exists()
    assert consistency_json.exists()
    assert drift_png.exists()

    rdf = pd.read_csv(rdf_csv)
    sweep = pd.read_csv(sweep_csv)
    drift = pd.read_csv(drift_csv)
    assert {"frame_id", "annulus_id", "mean_descriptor"}.issubset(set(rdf.columns))
    assert {"sweep_id", "brightness_factor", "radius_scale", "conclusion_label", "mean_profile_correlation"}.issubset(set(sweep.columns))
    assert {"sweep_id", "frame_id", "center_of_mass_delta", "peak_delta"}.issubset(set(drift.columns))
    assert len(rdf) > 0
    assert len(sweep) >= 3
    assert len(drift) > 0


def test_milestone11_strict_radial_consistency_gate_can_fail(tmp_path: Path):
    frames = _make_small_radial_video(n_frames=5)
    records = [FrameRecord(i, float(i), f) for i, f in enumerate(frames)]
    out_dir = tmp_path / "run_m11_fail"
    cfg = PipelineConfig()
    cfg.qc.fail_on_gate_error = True
    cfg.validation.min_conclusion_agreement = 1.1
    try:
        run_milestone11(records, out_dir, cfg)
    except RuntimeError as e:
        assert "radial_conclusion_agreement" in str(e) or "radial_profile_correlation" in str(e)
    else:
        raise AssertionError("Expected strict milestone11 radial consistency QC failure")
