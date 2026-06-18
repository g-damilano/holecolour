from pathlib import Path

from holecolor.config.schema import PipelineConfig
from holecolor.core.types import FrameRecord
from holecolor.pipeline import run_milestone20
from holecolor.synth.artifacts import add_known_drift, add_photometric_artifacts
from holecolor.tests.test_milestone14 import _make_small_rdf_video


def test_milestone20_writes_notebook_and_parallel_sweep_outputs(tmp_path: Path):
    frames = _make_small_rdf_video(n_frames=6)
    frames = add_photometric_artifacts(frames, brightness_ramp=0.04, vignette_strength=0.06)
    frames, _ = add_known_drift(frames, dx_per_frame=0.15, dy_per_frame=-0.1)
    records = [FrameRecord(i, float(i), f) for i, f in enumerate(frames)]

    out_dir = tmp_path / 'run_m20'
    cfg = PipelineConfig()
    cfg.parallel.backend = 'thread'
    cfg.parallel.max_workers = 2
    cfg.parallel.min_parallel_tasks = 1
    cfg.parallel.show_progress = False

    summary = run_milestone20(records, out_dir, cfg)
    assert summary['n_frames'] == 6
    expected = [
        out_dir / 'notebooks' / 'holecolor_results_explorer.ipynb',
        out_dir / 'qc' / 'radial_perturbation_sweeps.csv',
        out_dir / 'qc' / 'radial_conclusion_consistency.json',
        out_dir / 'radial' / 'per_hole_rdf_bootstrap_summary.csv',
    ]
    for p in expected:
        assert p.exists(), p
    txt = (out_dir / 'notebooks' / 'holecolor_results_explorer.ipynb').read_text(encoding='utf-8')
    assert 'HoleColor results explorer' in txt
    assert 'per_hole_rdf_evolution.csv' in txt
