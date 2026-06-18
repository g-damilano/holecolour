from pathlib import Path

import pandas as pd

from holecolor.config.schema import PipelineConfig
from holecolor.core.types import FrameRecord
from holecolor.pipeline import run_milestone10
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


def test_milestone10_writes_canonicalization_archetypes_and_smoothness(tmp_path: Path):
    frames = _make_small_radial_video(n_frames=6)
    frames = add_photometric_artifacts(frames, brightness_ramp=0.05, vignette_strength=0.08)
    frames, _ = add_known_drift(frames, dx_per_frame=0.2, dy_per_frame=-0.1)
    records = [FrameRecord(i, float(i), f) for i, f in enumerate(frames)]
    out_dir = tmp_path / "run_m10"
    cfg = PipelineConfig()
    summary = run_milestone10(records, out_dir, cfg)
    assert summary["n_frames"] == 6

    canon_path = out_dir / "temporal" / "phenotype_canonicalization.json"
    rerun_path = out_dir / "temporal" / "phenotype_rerun_assignments.csv"
    smooth_path = out_dir / "temporal" / "phenotype_spatial_smoothness.csv"
    arch_csv = out_dir / "temporal" / "phenotype_archetypes.csv"
    arch_png = out_dir / "temporal" / "phenotype_archetypes.png"
    assert canon_path.exists()
    assert rerun_path.exists()
    assert smooth_path.exists()
    assert arch_csv.exists()
    assert arch_png.exists()

    rerun = pd.read_csv(rerun_path)
    smooth = pd.read_csv(smooth_path)
    arche = pd.read_csv(arch_csv)
    assert {"hole_id", "rerun_canonical_label", "matches_base_canonical"}.issubset(set(rerun.columns))
    assert {"hole_id", "spatial_smoothness", "neighbor_count"}.issubset(set(smooth.columns))
    assert {"phenotype_label", "frame_id", "annulus_id", "mean_descriptor"}.issubset(set(arche.columns))
    assert len(rerun) > 0
    assert len(smooth) > 0
    assert len(arche) > 0


def test_milestone10_strict_spatial_or_canonical_gate_can_fail(tmp_path: Path):
    frames = _make_small_radial_video(n_frames=5)
    records = [FrameRecord(i, float(i), f) for i, f in enumerate(frames)]
    out_dir = tmp_path / "run_m10_fail"
    cfg = PipelineConfig()
    cfg.qc.fail_on_gate_error = True
    cfg.temporal.min_spatial_smoothness = 1.1
    try:
        run_milestone10(records, out_dir, cfg)
    except RuntimeError as e:
        assert "phenotype_spatial_smoothness" in str(e) or "phenotype_canonical_agreement" in str(e)
    else:
        raise AssertionError("Expected strict milestone10 phenotype QC failure")
