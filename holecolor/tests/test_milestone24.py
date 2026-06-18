from __future__ import annotations

import json
from pathlib import Path

from holecolor.config.schema import PipelineConfig
from holecolor.core.status import format_status_line
from holecolor.core.types import FrameRecord
from holecolor.pipeline import run_milestone24
from holecolor.tests.test_milestone14 import _make_small_rdf_video


def test_milestone24_writes_stage_plan_and_progress_summary(tmp_path: Path):
    frames = _make_small_rdf_video(n_frames=5)
    records = [FrameRecord(i, float(i), f) for i, f in enumerate(frames)]
    out_dir = tmp_path / 'run_m24'
    cfg = PipelineConfig()
    cfg.parallel.backend = 'thread'
    cfg.parallel.max_workers = 2
    cfg.parallel.min_parallel_tasks = 1
    cfg.parallel.show_progress = False

    summary = run_milestone24(records, out_dir, cfg)
    assert summary['n_frames'] == 5

    stage_plan = out_dir / 'logs' / 'stage_plan.json'
    progress_summary = out_dir / 'logs' / 'progress_summary.txt'
    current_status = out_dir / 'logs' / 'current_status.json'
    output_index = out_dir / 'outputs' / 'index.html'
    output_start = out_dir / 'outputs' / 'START_HERE.md'
    output_manifest = out_dir / 'outputs' / 'output_manifest.json'
    assert stage_plan.exists(), stage_plan
    assert progress_summary.exists(), progress_summary
    assert current_status.exists(), current_status
    assert output_index.exists(), output_index
    assert output_start.exists(), output_start
    assert output_manifest.exists(), output_manifest

    plan = json.loads(stage_plan.read_text(encoding='utf-8'))
    assert 'stages' in plan and len(plan['stages']) >= 5

    payload = json.loads(current_status.read_text(encoding='utf-8'))
    assert 'overall_eta_hms' in payload
    status_line = format_status_line(payload)
    assert 'overall=' in status_line
    assert 'overall_eta=' in status_line

    summary_text = progress_summary.read_text(encoding='utf-8')
    assert 'holecolor run status' in summary_text
    assert 'stage timings:' in summary_text
    assert 'status_line:' in summary_text
    assert 'Reference geometry' in output_start.read_text(encoding='utf-8')
