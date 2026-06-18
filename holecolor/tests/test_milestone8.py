from pathlib import Path

import numpy as np
import pandas as pd

from holecolor.config.schema import PipelineConfig
from holecolor.core.types import FrameRecord
from holecolor.pipeline import run_milestone8
from holecolor.synth.artifacts import add_known_drift, add_photometric_artifacts
from holecolor.synth.grid import make_synthetic_grid


def _make_small_radial_video(n_frames: int = 6):
    base, gt = make_synthetic_grid(shape=(256, 256), rows=3, cols=4, radius_px=12, spacing_px=46, rotation_deg=6.0)
    frames = []
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


def test_milestone8_writes_propagation_and_phenotypes(tmp_path: Path):
    frames = _make_small_radial_video(n_frames=6)
    frames = add_photometric_artifacts(frames, brightness_ramp=0.05, vignette_strength=0.08)
    frames, _ = add_known_drift(frames, dx_per_frame=0.2, dy_per_frame=-0.1)
    records = [FrameRecord(i, float(i), f) for i, f in enumerate(frames)]
    out_dir = tmp_path / "run_m8"
    cfg = PipelineConfig()
    summary = run_milestone8(records, out_dir, cfg)
    assert summary["n_frames"] == 6

    prop_path = out_dir / "temporal" / "annulus_propagation_summary.csv"
    phen_path = out_dir / "temporal" / "per_hole_phenotypes.csv"
    cent_path = out_dir / "temporal" / "phenotype_centroids.json"
    env_path = out_dir / "qc" / "threshold_envelope.json"
    assert prop_path.exists()
    assert phen_path.exists()
    assert cent_path.exists()
    assert env_path.exists()

    prop = pd.read_csv(prop_path)
    phen = pd.read_csv(phen_path)
    assert {"hole_id", "onset_monotonic_fraction", "peak_monotonic_fraction", "negative_lag_flag"}.issubset(set(prop.columns))
    assert {"hole_id", "cluster_id", "phenotype_label"}.issubset(set(phen.columns))
    assert len(prop) > 0
    assert len(phen) > 0


def test_milestone8_strict_temporal_gate_can_fail(tmp_path: Path):
    frames = _make_small_radial_video(n_frames=5)
    records = [FrameRecord(i, float(i), f) for i, f in enumerate(frames)]
    out_dir = tmp_path / "run_m8_fail"
    cfg = PipelineConfig()
    cfg.qc.fail_on_gate_error = True
    cfg.temporal.min_monotonic_fraction = 1.1
    try:
        run_milestone8(records, out_dir, cfg)
    except RuntimeError as e:
        assert "lag_monotonic_fraction" in str(e) or "valid_onset_annuli_fraction" in str(e) or "valid_peak_annuli_fraction" in str(e)
    else:
        raise AssertionError("Expected strict temporal QC failure")
