from __future__ import annotations

import json
from pathlib import Path

from holecolor.config.schema import PipelineConfig
from holecolor.core.types import FrameRecord
from holecolor.pipeline import run_milestone24
from holecolor.tests.test_milestone14 import _make_small_rdf_video


def test_milestone28_reuses_validation_cache_on_rerun(tmp_path: Path):
    frames = _make_small_rdf_video(n_frames=5)
    records = [FrameRecord(i, float(i), f) for i, f in enumerate(frames)]
    out_dir = tmp_path / 'run_m28'

    cfg = PipelineConfig()
    cfg.parallel.backend = 'thread'
    cfg.parallel.max_workers = 2
    cfg.parallel.min_parallel_tasks = 1
    cfg.parallel.show_progress = False
    cfg.validation.enabled = True
    cfg.radial.rdf_bootstrap_n = 8
    cfg.validation.brightness_factors = (0.95, 1.0)
    cfg.validation.radius_scale_factors = (0.95, 1.0)

    summary1 = run_milestone24(records, out_dir, cfg)
    assert summary1['n_frames'] == 5
    manifest_path = out_dir / 'qc' / 'validation_cache_manifest.json'
    assert manifest_path.exists()
    manifest1 = json.loads(manifest_path.read_text(encoding='utf-8'))
    assert manifest1['validation_enabled'] is True

    summary2 = run_milestone24(records, out_dir, cfg)
    assert summary2['n_frames'] == 5

    jsonl = (out_dir / 'logs' / 'run_status.jsonl').read_text(encoding='utf-8')
    assert 'Validation cache hit; loading cached outputs' in jsonl

    manifest2 = json.loads(manifest_path.read_text(encoding='utf-8'))
    assert manifest2['signature'] == manifest1['signature']
