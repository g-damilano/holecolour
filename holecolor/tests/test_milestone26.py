from pathlib import Path

from holecolor.color_analysis import run_color_only_analysis
from holecolor.config.schema import PipelineConfig
from holecolor.core.types import FrameRecord, HoleGeometry, LatticeModel
from holecolor.holegrid.model import HoleGridBundle, load_hole_grid_model, save_hole_grid_model
from holecolor.synth.grid import make_synthetic_grid


def test_milestone26_color_only_writes_timeseries(tmp_path: Path):
    img,_ = make_synthetic_grid(shape=(128,128), rows=3, cols=4, radius_px=6, spacing_px=24)
    frames = [FrameRecord(frame_id=i, time_s=float(i), image=img.copy()) for i in range(3)]
    cfg = PipelineConfig()
    cfg.parallel.show_progress = False
    summary = run_color_only_analysis(frames, tmp_path / 'color_only_run', cfg)
    assert (tmp_path / 'color_only_run' / 'color_only' / 'frame_color_timeseries.csv').exists()
    assert summary['n_frames'] == 3


def test_milestone26_holegrid_bundle_roundtrip(tmp_path: Path):
    lattice = LatticeModel(0.0, 0.0, (10.0, 0.0), (0.0, 10.0), 90.0, 10.0, 10.0, 0.9)
    holes = [HoleGeometry(hole_id=0, x=5.0, y=5.0, radius_inner_px=3.0, radius_outer_px=4.0, confidence=0.8)]
    bundle = HoleGridBundle('1.0', 'density', 0, 'test', None, lattice, holes)
    path = tmp_path / 'hole_grid_model.json'
    save_hole_grid_model(path, bundle)
    loaded = load_hole_grid_model(path)
    assert loaded.support_mode == 'density'
    assert len(loaded.holes) == 1
    assert loaded.lattice.spacing_u_px == 10.0
