from __future__ import annotations

import json
from pathlib import Path

from holecolor.config.schema import PipelineConfig
from holecolor.core.types import FrameRecord
from holecolor.pipeline import run_milestone24
from holecolor.tests.test_milestone14 import _make_small_rdf_video


def test_milestone27_stage_plan_is_split_and_fast_profile_skips_validation(tmp_path: Path):
    frames = _make_small_rdf_video(n_frames=5)
    records = [FrameRecord(i, float(i), f) for i, f in enumerate(frames)]
    out_dir = tmp_path / 'run_m27'
    cfg = PipelineConfig()
    cfg.parallel.backend = 'thread'
    cfg.parallel.max_workers = 2
    cfg.parallel.min_parallel_tasks = 1
    cfg.parallel.show_progress = False
    cfg.validation.enabled = False
    summary = run_milestone24(records, out_dir, cfg)
    assert summary['n_frames'] == 5

    stage_plan = json.loads((out_dir / 'logs' / 'stage_plan.json').read_text(encoding='utf-8'))
    assert 'Radial summaries' in stage_plan['stages']
    assert 'RDF summaries' in stage_plan['stages']
    assert 'Validation sweeps' in stage_plan['stages']
    assert 'Temporal phenotypes' in stage_plan['stages']

    consistency = json.loads((out_dir / 'qc' / 'radial_conclusion_consistency.json').read_text(encoding='utf-8'))
    assert consistency['n_sweeps'] == 0

    bootstrap = json.loads((out_dir / 'qc' / 'rdf_bootstrap_summary.json').read_text(encoding='utf-8'))
    assert bootstrap['validation_enabled'] is False
