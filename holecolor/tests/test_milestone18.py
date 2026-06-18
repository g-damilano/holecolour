from pathlib import Path

import pandas as pd

from holecolor.config.schema import PipelineConfig
from holecolor.core.types import FrameRecord
from holecolor.pipeline import run_milestone18
from holecolor.synth.artifacts import add_known_drift, add_photometric_artifacts
from holecolor.tests.test_milestone14 import _make_small_rdf_video


def test_milestone18_writes_rdf_uncertainty_and_sector_acceleration_outputs(tmp_path: Path):
    frames = _make_small_rdf_video(n_frames=6)
    frames = add_photometric_artifacts(frames, brightness_ramp=0.05, vignette_strength=0.08)
    frames, _ = add_known_drift(frames, dx_per_frame=0.2, dy_per_frame=-0.1)
    records = [FrameRecord(i, float(i), f) for i, f in enumerate(frames)]
    out_dir = tmp_path / "run_m18"
    cfg = PipelineConfig()
    summary = run_milestone18(records, out_dir, cfg)
    assert summary["n_frames"] == 6

    expected = [
        out_dir / "radial" / "rdf_uncertainty_reticulum.csv",
        out_dir / "radial" / "rdf_uncertainty_reticulum_map.png",
        out_dir / "radial" / "rdf_uncertainty_hotspot_comparison.csv",
        out_dir / "radial" / "rdf_uncertainty_hotspot_group_summary.csv",
        out_dir / "radial" / "rdf_uncertainty_hotspot_group_comparison.png",
        out_dir / "radial" / "sector_front_acceleration.csv",
        out_dir / "radial" / "sector_front_acceleration_hole_summary.csv",
        out_dir / "radial" / "sector_front_acceleration.png",
        out_dir / "qc" / "rdf_uncertainty_summary.json",
    ]
    for pth in expected:
        assert pth.exists(), pth

    ret_df = pd.read_csv(out_dir / "radial" / "rdf_uncertainty_reticulum.csv")
    cmp_df = pd.read_csv(out_dir / "radial" / "rdf_uncertainty_hotspot_comparison.csv")
    grp_df = pd.read_csv(out_dir / "radial" / "rdf_uncertainty_hotspot_group_summary.csv")
    sec_df = pd.read_csv(out_dir / "radial" / "sector_front_acceleration.csv")
    sec_hole_df = pd.read_csv(out_dir / "radial" / "sector_front_acceleration_hole_summary.csv")

    assert {"hole_id", "rdf_archetype_canonical_label", "bootstrap_rdf_archetype_support_fraction", "rdf_uncertainty_score"}.issubset(ret_df.columns)
    assert {"hole_id", "reticulum_zone", "hotspot_proximity_bucket", "bootstrap_front_velocity_ci_width"}.issubset(cmp_df.columns)
    assert {"reticulum_zone", "hotspot_proximity_bucket", "mean_bootstrap_support_fraction"}.issubset(grp_df.columns)
    assert {"hole_id", "sector_id", "sector_front_acceleration_per_frame2", "sector_front_curvature"}.issubset(sec_df.columns)
    assert {"hole_id", "mean_sector_front_acceleration_per_frame2", "valid_sector_acceleration_fraction"}.issubset(sec_hole_df.columns)
    assert len(ret_df) > 0
    assert len(cmp_df) > 0
    assert len(sec_df) > 0
    assert len(sec_hole_df) > 0
