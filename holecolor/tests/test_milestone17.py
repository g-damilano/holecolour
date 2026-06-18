from pathlib import Path

import pandas as pd

from holecolor.config.schema import PipelineConfig
from holecolor.core.types import FrameRecord
from holecolor.pipeline import run_milestone17
from holecolor.synth.artifacts import add_known_drift, add_photometric_artifacts
from holecolor.tests.test_milestone14 import _make_small_rdf_video


def test_milestone17_writes_rdf_bootstrap_and_sector_propagation_outputs(tmp_path: Path):
    frames = _make_small_rdf_video(n_frames=6)
    frames = add_photometric_artifacts(frames, brightness_ramp=0.05, vignette_strength=0.08)
    frames, _ = add_known_drift(frames, dx_per_frame=0.2, dy_per_frame=-0.1)
    records = [FrameRecord(i, float(i), f) for i, f in enumerate(frames)]
    out_dir = tmp_path / 'run_m17'
    cfg = PipelineConfig()
    summary = run_milestone17(records, out_dir, cfg)
    assert summary['n_frames'] == 6

    expected = [
        out_dir / 'radial' / 'per_hole_rdf_bootstrap_summary.csv',
        out_dir / 'radial' / 'rdf_archetype_bootstrap_support.csv',
        out_dir / 'radial' / 'per_hole_rdf_bootstrap_ci.png',
        out_dir / 'radial' / 'rdf_archetype_bootstrap_support.png',
        out_dir / 'radial' / 'sector_front_propagation.csv',
        out_dir / 'radial' / 'sector_front_propagation_hole_summary.csv',
        out_dir / 'radial' / 'sector_front_propagation.png',
        out_dir / 'qc' / 'rdf_bootstrap_summary.json',
    ]
    for pth in expected:
        assert pth.exists(), pth

    boot_df = pd.read_csv(out_dir / 'radial' / 'per_hole_rdf_bootstrap_summary.csv')
    support_df = pd.read_csv(out_dir / 'radial' / 'rdf_archetype_bootstrap_support.csv')
    sec_df = pd.read_csv(out_dir / 'radial' / 'sector_front_propagation.csv')
    sec_hole_df = pd.read_csv(out_dir / 'radial' / 'sector_front_propagation_hole_summary.csv')
    assert {'hole_id', 'bootstrap_front_velocity_per_frame', 'bootstrap_front_velocity_ci_width'}.issubset(boot_df.columns)
    assert {'hole_id', 'bootstrap_rdf_archetype_canonical_id', 'bootstrap_rdf_archetype_support_fraction'}.issubset(support_df.columns)
    assert {'hole_id', 'sector_id', 'sector_front_velocity_per_frame', 'sector_front_onset_frame'}.issubset(sec_df.columns)
    assert {'hole_id', 'valid_sector_velocity_fraction', 'sector_front_velocity_anisotropy'}.issubset(sec_hole_df.columns)
    assert len(boot_df) > 0
    assert len(support_df) > 0
    assert len(sec_df) > 0
    assert len(sec_hole_df) > 0
