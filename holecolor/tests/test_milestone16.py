from pathlib import Path

import pandas as pd

from holecolor.config.schema import PipelineConfig
from holecolor.core.types import FrameRecord
from holecolor.pipeline import run_milestone16
from holecolor.synth.artifacts import add_known_drift, add_photometric_artifacts
from holecolor.tests.test_milestone14 import _make_small_rdf_video


def test_milestone16_writes_rdf_stability_sector_lags_and_rdf_hotspot_links(tmp_path: Path):
    frames = _make_small_rdf_video(n_frames=6)
    frames = add_photometric_artifacts(frames, brightness_ramp=0.05, vignette_strength=0.08)
    frames, _ = add_known_drift(frames, dx_per_frame=0.2, dy_per_frame=-0.1)
    records = [FrameRecord(i, float(i), f) for i, f in enumerate(frames)]
    out_dir = tmp_path / 'run_m16'
    cfg = PipelineConfig()
    summary = run_milestone16(records, out_dir, cfg)
    assert summary['n_frames'] == 6

    expected = [
        out_dir / 'radial' / 'per_hole_rdf_stability.csv',
        out_dir / 'radial' / 'per_hole_rdf_stability_summary.csv',
        out_dir / 'radial' / 'per_hole_rdf_stability.png',
        out_dir / 'radial' / 'sector_front_lag_map.csv',
        out_dir / 'radial' / 'sector_front_lag_summary.csv',
        out_dir / 'radial' / 'sector_front_lag_map.png',
        out_dir / 'radial' / 'rdf_hotspot_reticulum_comparison.csv',
        out_dir / 'radial' / 'rdf_hotspot_reticulum_group_summary.csv',
        out_dir / 'radial' / 'rdf_hotspot_reticulum_group_comparison.png',
        out_dir / 'qc' / 'rdf_archetype_stability_summary.json',
    ]
    for p in expected:
        assert p.exists(), p

    stab_df = pd.read_csv(out_dir / 'radial' / 'per_hole_rdf_stability.csv')
    lag_df = pd.read_csv(out_dir / 'radial' / 'sector_front_lag_map.csv')
    rdf_link_df = pd.read_csv(out_dir / 'radial' / 'rdf_hotspot_reticulum_comparison.csv')
    assert {'sweep_id', 'hole_id', 'rdf_archetype_match'}.issubset(stab_df.columns)
    assert {'hole_id', 'sector_id', 'sector_onset_lag', 'sector_peak_lag'}.issubset(lag_df.columns)
    assert {'hole_id', 'rdf_archetype_canonical_label', 'reticulum_zone', 'hotspot_proximity_bucket'}.issubset(rdf_link_df.columns)
    assert len(stab_df) > 0
    assert len(lag_df) > 0
    assert len(rdf_link_df) > 0
