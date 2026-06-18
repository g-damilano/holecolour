from pathlib import Path

import numpy as np

from holecolor.core.types import FrameRecord, HoleGeometry
from holecolor.extensions.radial_cluster_average_hole import write_radial_cluster_average_hole_artifacts
from holecolor.extensions.wafer_nonhole_colour import build_wafer_nonhole_colour_bundle
from holecolor.synth.radial_front import make_synthetic_radial_front_video


def test_write_radial_cluster_average_hole_artifacts(tmp_path: Path) -> None:
    frames, _ = make_synthetic_radial_front_video(n_frames=4)
    records = [FrameRecord(i, float(i), f) for i, f in enumerate(frames)]
    h, w = frames[0].shape[:2]
    holes = [HoleGeometry(hole_id=0, x=w / 2, y=h / 2, radius_inner_px=12.0, radius_outer_px=14.0, confidence=1.0)]
    holes_by_frame = {i: holes for i in range(len(records))}
    support_mask = np.ones((h, w), dtype=bool)
    lattice_indices = {0: (0, 0)}
    bundle = build_wafer_nonhole_colour_bundle(records, holes_by_frame, support_mask, max_points_per_frame=1500, min_total_points=200, k_min=2, k_max=3, random_state=0)
    out = tmp_path / 'radial_cluster_average_hole'
    status = write_radial_cluster_average_hole_artifacts(
        out,
        records,
        holes_by_frame,
        support_mask,
        bundle.cluster_rows,
        lattice_indices,
        lattice_angle_deg=0.0,
        n_terraces=7,
        n_angle_sectors=8,
        terrace_width_mode='fixed',
        terrace_gap_basis='border_gap',
        terrace_min_width_px=1.0,
        front_threshold_fraction=0.1,
    )
    assert status['status'] == 'ok'
    assert (out / 'radial_cluster_status.json').exists()
    assert (out / 'hole_terrace_sector_cluster_tensor.csv').exists()
    assert (out / 'average_hole_terrace_cluster_fractions.csv').exists()
    assert (out / 'average_hole_terrace_summary.csv').exists()
    assert (out / 'cluster_front_metrics.csv').exists()
    assert (out / 'hole_consistency_by_terrace.csv').exists()
    assert (out / 'average_hole_terrace_chronogram.png').exists()
    assert (out / 'cluster_front_trajectories.png').exists()
    assert (out / 'hole_consistency_terrace_map.png').exists()
    assert (out / 'terrace_01_angular_chronogram.png').exists()


def test_radial_cluster_average_hole_skips_physically_impossible_hole_count(tmp_path: Path) -> None:
    image = np.full((120, 120, 3), 128, dtype=np.uint8)
    records = [FrameRecord(0, 0.0, image)]
    holes = [
        HoleGeometry(hole_id=i, x=float(i % 120), y=float(i // 120), radius_inner_px=10.0, radius_outer_px=12.0, confidence=1.0)
        for i in range(200)
    ]
    cluster_rows = [
        {'cluster_id': 0, 'center_h': 0.0, 'center_s': 0.0, 'center_l': 0.5, 'center_hx': 0.0, 'center_hy': 0.0},
        {'cluster_id': 1, 'center_h': 0.5, 'center_s': 0.1, 'center_l': 0.6, 'center_hx': -0.1, 'center_hy': 0.0},
    ]
    out = tmp_path / 'radial_cluster_average_hole'
    status = write_radial_cluster_average_hole_artifacts(
        out,
        records,
        {0: holes},
        np.ones((120, 120), dtype=bool),
        cluster_rows,
        {i: (i, 0) for i in range(len(holes))},
        lattice_angle_deg=0.0,
        n_terraces=8,
        n_angle_sectors=8,
    )
    assert status['status'] == 'skipped'
    assert status['message'] == 'hole_count_exceeds_physical_nonoverlap_limit'
    assert not (out / 'hole_terrace_sector_cluster_tensor.csv').exists()
