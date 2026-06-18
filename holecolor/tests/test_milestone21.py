from pathlib import Path

import numpy as np

from holecolor.config.schema import PipelineConfig
from holecolor.core.types import FrameRecord, HoleGeometry
from holecolor.masks.terraces import make_nonoverlapping_hole_terraces
from holecolor.pipeline import run_milestone20
from holecolor.synth.grid import make_synthetic_grid
from holecolor.tests.test_milestone14 import _make_small_rdf_video


def test_memory_safe_terraces_handle_many_holes_without_full_distance_cube():
    img, _ = make_synthetic_grid(shape=(720, 1280), rows=7, cols=19, radius_px=10, spacing_px=58, rotation_deg=0.0)
    holes = [
        HoleGeometry(hole_id=i, x=float(x), y=float(y), radius_inner_px=9.0, radius_outer_px=10.0, confidence=1.0)
        for i, (x, y) in enumerate(_["centers"])
    ]
    terraces_by_hole = make_nonoverlapping_hole_terraces(img.shape[:2], holes, 6)
    assert len(terraces_by_hole) >= 100
    some = next(iter(terraces_by_hole.values()))
    assert len(some) == 6
    assert sum(t.area_px for t in some) > 0
    total_local_pixels = sum(t.mask.size for regs in terraces_by_hole.values() for t in regs)
    assert total_local_pixels < img.shape[0] * img.shape[1] * len(holes) / 10.0


def test_milestone20_pipeline_still_writes_key_outputs_after_local_terrace_refactor(tmp_path: Path):
    frames = _make_small_rdf_video(n_frames=5)
    records = [FrameRecord(i, float(i), f) for i, f in enumerate(frames)]
    out_dir = tmp_path / 'run_m21'
    cfg = PipelineConfig()
    cfg.parallel.backend = 'thread'
    cfg.parallel.max_workers = 2
    cfg.parallel.min_parallel_tasks = 1
    cfg.parallel.show_progress = False
    summary = run_milestone20(records, out_dir, cfg)
    assert summary['n_frames'] == 5
    expected = [
        out_dir / 'radial' / 'per_hole_rdf_evolution.csv',
        out_dir / 'radial' / 'sector_rdf_evolution.csv',
        out_dir / 'notebooks' / 'holecolor_results_explorer.ipynb',
    ]
    for p in expected:
        assert p.exists(), p
