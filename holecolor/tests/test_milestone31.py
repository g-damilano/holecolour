from __future__ import annotations

from pathlib import Path

from holecolor.config.schema import PipelineConfig
from holecolor.core.types import FrameRecord
from holecolor.pipeline import run_milestone24
from holecolor.tests.test_milestone14 import _make_small_rdf_video


def test_milestone31_writes_binary_cache_sidecars_and_reuses_them(tmp_path: Path):
    frames = _make_small_rdf_video(n_frames=5)
    records = [FrameRecord(i, float(i), f) for i, f in enumerate(frames)]
    out_dir = tmp_path / 'run_m31'

    cfg = PipelineConfig()
    cfg.parallel.backend = 'thread'
    cfg.parallel.max_workers = 2
    cfg.parallel.min_parallel_tasks = 1
    cfg.parallel.show_progress = False
    cfg.validation.enabled = True
    cfg.radial.rdf_bootstrap_n = 4

    summary1 = run_milestone24(records, out_dir, cfg)
    assert summary1['n_frames'] == 5

    expected_sidecars = [
        out_dir / 'descriptors' / 'hole_compartment_timeseries.csv.pkl',
        out_dir / 'descriptors' / 'matrix_timeseries.csv.pkl',
        out_dir / 'radial' / 'hole_annulus_timeseries.csv.pkl',
        out_dir / 'hotspots' / 'hotspots.csv.pkl',
        out_dir / 'radial' / 'per_hole_rdf_bootstrap_summary.csv.pkl',
        out_dir / 'qc' / 'radial_perturbation_sweeps.csv.pkl',
    ]
    for path in expected_sidecars:
        assert path.exists(), str(path)

    summary2 = run_milestone24(records, out_dir, cfg)
    assert summary2['n_frames'] == 5

    jsonl = (out_dir / 'logs' / 'run_status.jsonl').read_text(encoding='utf-8')
    assert 'Per-frame analysis cache hit; loading cached rows' in jsonl
    assert 'Validation cache hit; loading cached outputs' in jsonl
