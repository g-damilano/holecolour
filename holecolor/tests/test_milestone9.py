from pathlib import Path

import pandas as pd

from holecolor.config.schema import PipelineConfig
from holecolor.core.types import FrameRecord
from holecolor.pipeline import run_milestone9
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


def test_milestone9_writes_stability_and_spatial_maps(tmp_path: Path):
    frames = _make_small_radial_video(n_frames=6)
    frames = add_photometric_artifacts(frames, brightness_ramp=0.05, vignette_strength=0.08)
    frames, _ = add_known_drift(frames, dx_per_frame=0.2, dy_per_frame=-0.1)
    records = [FrameRecord(i, float(i), f) for i, f in enumerate(frames)]
    out_dir = tmp_path / "run_m9"
    cfg = PipelineConfig()
    summary = run_milestone9(records, out_dir, cfg)
    assert summary["n_frames"] == 6

    stab_path = out_dir / "temporal" / "phenotype_stability.csv"
    coh_path = out_dir / "temporal" / "phenotype_coherence.json"
    neigh_path = out_dir / "temporal" / "phenotype_neighbor_pairs.csv"
    spatial_path = out_dir / "temporal" / "phenotype_spatial_map.png"
    ret_path = out_dir / "temporal" / "phenotype_reticulum_map.png"
    assert stab_path.exists()
    assert coh_path.exists()
    assert neigh_path.exists()
    assert spatial_path.exists()
    assert ret_path.exists()

    stab = pd.read_csv(stab_path)
    neigh = pd.read_csv(neigh_path)
    assert {"hole_id", "stability_fraction", "dominant_rerun_label", "base_matches_dominant"}.issubset(set(stab.columns))
    assert {"hole_id_a", "hole_id_b", "same_label"}.issubset(set(neigh.columns))
    assert len(stab) > 0
    assert len(neigh) > 0


def test_milestone9_strict_phenotype_gate_can_fail(tmp_path: Path):
    frames = _make_small_radial_video(n_frames=5)
    records = [FrameRecord(i, float(i), f) for i, f in enumerate(frames)]
    out_dir = tmp_path / "run_m9_fail"
    cfg = PipelineConfig()
    cfg.qc.fail_on_gate_error = True
    cfg.temporal.min_phenotype_stability = 1.1
    try:
        run_milestone9(records, out_dir, cfg)
    except RuntimeError as e:
        assert "phenotype_stability_fraction" in str(e) or "phenotype_neighbor_coherence" in str(e)
    else:
        raise AssertionError("Expected strict phenotype QC failure")
