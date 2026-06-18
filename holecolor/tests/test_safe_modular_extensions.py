
from pathlib import Path

from holecolor.config.schema import PipelineConfig
from holecolor.core.types import FrameRecord, HoleGeometry, LatticeModel
from holecolor.holegrid.model import HoleGridBundle
from holecolor.pipeline import run_milestone24
from holecolor.synth.grid import make_synthetic_grid


def test_safe_modular_extensions_with_supplied_holegrid(tmp_path: Path) -> None:
    img, gt = make_synthetic_grid(shape=(128, 128), rows=3, cols=4, radius_px=6, spacing_px=24)
    centers = gt["centers"]
    frames = [FrameRecord(frame_id=i, time_s=float(i), image=img.copy()) for i in range(3)]
    holes = [
        HoleGeometry(hole_id=i, x=float(x), y=float(y), radius_inner_px=5.0, radius_outer_px=7.0, confidence=0.9)
        for i, (x, y) in enumerate(centers)
    ]
    lattice = LatticeModel(origin_x=float(centers[0][0]), origin_y=float(centers[0][1]), basis_u=(24.0, 0.0), basis_v=(0.0, 24.0), angle_deg=90.0, spacing_u_px=24.0, spacing_v_px=24.0, confidence=0.9)
    bundle = HoleGridBundle('1.0', 'supplied', 0, 'test supplied grid', (64, 64, 58), lattice, holes)
    cfg = PipelineConfig()
    cfg.wafer_nonhole_colour.enabled = True
    cfg.parallel.enabled = False
    cfg.parallel.show_progress = False
    cfg.validation.enabled = False
    out_dir = tmp_path / 'safe_modular_run'
    summary = run_milestone24(frames, out_dir, cfg, hole_grid_bundle=bundle)
    assert summary['n_frames'] == 3
    # legacy stable artifacts still present
    assert (out_dir / 'geometry' / 'hole_geometry.csv').exists()
    assert (out_dir / 'descriptors' / 'hole_compartment_timeseries.csv').exists()
    assert (out_dir / 'descriptors' / 'matrix_timeseries.csv').exists()
    # new additive artifacts present in isolated family
    ext = out_dir / 'descriptors' / 'wafer_nonhole_colour'
    assert (ext / 'stage_status.json').exists()
    assert (ext / 'frame_region_summary.csv').exists()
    assert (ext / 'frame_cluster_summary.csv').exists()
    assert (ext / 'frame_cluster_prevalence_hard.csv').exists()
    assert (ext / 'frame_cluster_prevalence_soft.csv').exists()
    assert (ext / 'global_buffer_cluster_context.csv').exists()
    assert (ext / 'local_hole_cluster_context.csv').exists()
    avg = out_dir / 'descriptors' / 'radial_cluster_average_hole'
    assert (avg / 'radial_cluster_status.json').exists()
    assert (avg / 'average_hole_terrace_chronogram.png').exists()
    assert (avg / 'cluster_front_trajectories.png').exists()
