from pathlib import Path

import pandas as pd

from holecolor.config.schema import PipelineConfig
from holecolor.core.types import FrameRecord
from holecolor.pipeline import run_milestone15
from holecolor.synth.artifacts import add_known_drift, add_photometric_artifacts
from holecolor.tests.test_milestone14 import _make_small_rdf_video


def test_milestone15_writes_rdf_archetypes_dynamics_and_sector_outputs(tmp_path: Path):
    frames = _make_small_rdf_video(n_frames=6)
    frames = add_photometric_artifacts(frames, brightness_ramp=0.05, vignette_strength=0.08)
    frames, _ = add_known_drift(frames, dx_per_frame=0.2, dy_per_frame=-0.1)
    records = [FrameRecord(i, float(i), f) for i, f in enumerate(frames)]
    out_dir = tmp_path / 'run_m15'
    cfg = PipelineConfig()
    summary = run_milestone15(records, out_dir, cfg)
    assert summary['n_frames'] == 6

    expected = [
        out_dir / 'radial' / 'per_hole_rdf_archetypes.csv',
        out_dir / 'radial' / 'rdf_archetype_centroids.json',
        out_dir / 'radial' / 'rdf_archetype_counts.png',
        out_dir / 'radial' / 'per_hole_rdf_dynamics.csv',
        out_dir / 'radial' / 'rdf_front_acceleration.png',
        out_dir / 'radial' / 'sector_rdf_evolution.csv',
        out_dir / 'radial' / 'sector_rdf_frame_summary.csv',
        out_dir / 'radial' / 'sector_rdf_evolution.png',
    ]
    for p in expected:
        assert p.exists(), p

    arch_df = pd.read_csv(out_dir / 'radial' / 'per_hole_rdf_archetypes.csv')
    dyn_df = pd.read_csv(out_dir / 'radial' / 'per_hole_rdf_dynamics.csv')
    sec_df = pd.read_csv(out_dir / 'radial' / 'sector_rdf_evolution.csv')
    sec_frame_df = pd.read_csv(out_dir / 'radial' / 'sector_rdf_frame_summary.csv')
    assert {'hole_id', 'rdf_archetype_id', 'rdf_archetype_label', 'mean_front_radius_norm'}.issubset(arch_df.columns)
    assert {'hole_id', 'rdf_front_acceleration_per_frame2', 'rdf_front_nonlinearity_gain'}.issubset(dyn_df.columns)
    assert {'hole_id', 'frame_id', 'sector_id', 'annulus_id', 'sector_rdf_pdf'}.issubset(sec_df.columns)
    assert {'hole_id', 'frame_id', 'sector_id', 'sector_rdf_front_radius_norm'}.issubset(sec_frame_df.columns)
    assert len(arch_df) > 0
    assert len(dyn_df) > 0
    assert len(sec_df) > 0
    assert len(sec_frame_df) > 0
