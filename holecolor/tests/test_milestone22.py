from __future__ import annotations

import json
from pathlib import Path

from holecolor.config.schema import PipelineConfig
from holecolor.core.types import FrameRecord
from holecolor.pipeline import run_milestone22
from holecolor.tests.test_milestone14 import _make_small_rdf_video


def test_milestone22_writes_status_logs_and_stage_timings(tmp_path: Path):
    frames = _make_small_rdf_video(n_frames=5)
    records = [FrameRecord(i, float(i), f) for i, f in enumerate(frames)]
    out_dir = tmp_path / 'run_m22'
    cfg = PipelineConfig()
    cfg.parallel.backend = 'thread'
    cfg.parallel.max_workers = 2
    cfg.parallel.min_parallel_tasks = 1
    cfg.parallel.show_progress = False

    summary = run_milestone22(records, out_dir, cfg)
    assert summary['n_frames'] == 5

    current_status = out_dir / 'logs' / 'current_status.json'
    run_status = out_dir / 'logs' / 'run_status.jsonl'
    stage_timings = out_dir / 'logs' / 'stage_timings.json'
    assert current_status.exists(), current_status
    assert run_status.exists(), run_status
    assert stage_timings.exists(), stage_timings

    current_payload = json.loads(current_status.read_text(encoding='utf-8'))
    assert current_payload['event'] in {'stage_completed', 'stage_failed', 'run_initialized', 'run_completed'}

    lines = [json.loads(line) for line in run_status.read_text(encoding='utf-8').splitlines() if line.strip()]
    assert any(row['event'] == 'stage_started' for row in lines)
    assert any(row['event'] == 'stage_completed' for row in lines)
    timings = json.loads(stage_timings.read_text(encoding='utf-8'))
    assert len(timings) >= 3
    assert all('stage' in row and 'elapsed_s' in row for row in timings)
