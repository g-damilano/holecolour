
import csv
from pathlib import Path

import numpy as np

from holecolor.core.types import FrameRecord, HoleGeometry
from holecolor.extensions.wafer_nonhole_colour import (
    _choose_gmm_candidate,
    _cluster_palette_rows,
    build_wafer_nonhole_colour_bundle,
    enrich_global_matrix_rows,
    enrich_local_compartment_rows,
    support_mask_from_debug,
    write_wafer_nonhole_cluster_videos,
)
from holecolor.synth.radial_front import make_synthetic_radial_front_video


def test_build_wafer_nonhole_colour_bundle_and_contexts() -> None:
    frames, _ = make_synthetic_radial_front_video(n_frames=3)
    records = [FrameRecord(i, float(i), f) for i, f in enumerate(frames)]
    h, w = frames[0].shape[:2]
    support_mask = np.ones((h, w), dtype=bool)
    holes_by_frame = {
        i: [HoleGeometry(hole_id=0, x=w/2, y=h/2, radius_inner_px=12.0, radius_outer_px=14.0, confidence=1.0)]
        for i in range(3)
    }
    bundle = build_wafer_nonhole_colour_bundle(records, holes_by_frame, support_mask, max_points_per_frame=1500, min_total_points=200, k_min=2, k_max=3, random_state=0)
    assert bundle.status == 'ok'
    assert bundle.selected_k in {2, 3}
    assert len(bundle.frame_cluster_summary_rows) == 3
    matrix_rows = [{'frame_id': 0, 'mean_H': 0.1, 'mean_S': 0.2, 'mean_B': 0.3}]
    comp_rows = [{'frame_id': 0, 'hole_id': 0, 'region_id': 'hole_0_interior', 'mean_H': 0.15, 'mean_S': 0.25, 'mean_B': 0.35}]
    g = enrich_global_matrix_rows(matrix_rows, bundle.frame_cluster_summary_rows)
    l = enrich_local_compartment_rows(comp_rows, bundle.frame_cluster_summary_rows)
    assert 'dominant_cluster_id' in g[0]
    assert 'dominant_cluster_id' in l[0]


def test_support_mask_from_debug_fallbacks() -> None:
    shape = (10, 12)
    mask = support_mask_from_debug(shape, None, (6, 5, 3))
    assert mask.shape == shape
    assert mask.dtype == bool


def test_write_wafer_nonhole_cluster_videos(tmp_path: Path) -> None:
    frames, _ = make_synthetic_radial_front_video(n_frames=3)
    records = [FrameRecord(i, float(i), f) for i, f in enumerate(frames)]
    h, w = frames[0].shape[:2]
    support_mask = np.ones((h, w), dtype=bool)
    holes_by_frame = {
        i: [HoleGeometry(hole_id=0, x=w/2, y=h/2, radius_inner_px=12.0, radius_outer_px=14.0, confidence=1.0)]
        for i in range(3)
    }
    bundle = build_wafer_nonhole_colour_bundle(records, holes_by_frame, support_mask, max_points_per_frame=1200, min_total_points=200, k_min=2, k_max=3, random_state=0)
    out = tmp_path / 'wnhc'
    out.mkdir()
    status = write_wafer_nonhole_cluster_videos(out, records, holes_by_frame, support_mask, bundle)
    assert status['status'] == 'ok'
    assert (out / 'video_cluster_recoloured.avi').exists()
    assert (out / 'video_cluster_side_by_side.avi').exists()
    assert (out / 'video_cluster_labels.avi').exists()
    assert (out / 'video_cluster_baseline_activity.avi').exists()
    assert (out / 'cluster_display_palette.csv').exists()
    assert (out / 'cluster_video_status.json').exists()
    assert (out / 'cluster_baseline_activity_status.json').exists()
    assert (out / 'frame_cluster_baseline_activity.csv').exists()
    assert (out / 'frame_cluster_pixel_counts.csv').exists()
    assert (out / 'frame_cluster_pixel_fractions.csv').exists()
    assert (out / 'frame_cluster_pixel_counts.png').exists()
    assert (out / 'frame_cluster_pixel_fractions.png').exists()
    with (out / 'cluster_display_palette.csv').open(newline='', encoding='utf-8') as fh:
        palette_rows = list(csv.DictReader(fh))
    assert palette_rows
    assert all(row['display_source'] == 'analysis_observed_sample_median_rgb' for row in palette_rows)
    assert {'display_r', 'display_g', 'display_b', 'centroid_r', 'centroid_g', 'centroid_b', 'analysis_median_r', 'analysis_median_g', 'analysis_median_b'}.issubset(palette_rows[0])
    assert all(int(row['analysis_n_pixels']) > 0 for row in palette_rows)
    with (out / 'frame_cluster_baseline_activity.csv').open(newline='', encoding='utf-8') as fh:
        activity_rows = list(csv.DictReader(fh))
    assert activity_rows
    assert all(float(row['active_fraction']) >= 0.0 for row in activity_rows)
    assert all(0.0 <= float(row['alpha']) <= 1.0 for row in activity_rows)


def test_write_baseline_activity_video_can_use_uncorrected_display_colours(tmp_path: Path) -> None:
    frames, _ = make_synthetic_radial_front_video(n_frames=4)
    records = [FrameRecord(i, float(i), f) for i, f in enumerate(frames)]
    display_records = [
        FrameRecord(record.frame_id, record.time_s, np.full_like(record.image, np.array([91, 73, 55], dtype=np.uint8)))
        for record in records
    ]
    h, w = frames[0].shape[:2]
    support_mask = np.ones((h, w), dtype=bool)
    holes_by_frame = {
        i: [HoleGeometry(hole_id=0, x=w/2, y=h/2, radius_inner_px=12.0, radius_outer_px=14.0, confidence=1.0)]
        for i in range(4)
    }
    bundle = build_wafer_nonhole_colour_bundle(records, holes_by_frame, support_mask, max_points_per_frame=1200, min_total_points=200, k_min=2, k_max=3, random_state=0)
    out = tmp_path / 'wnhc_raw_display'
    out.mkdir()
    status = write_wafer_nonhole_cluster_videos(
        out,
        records,
        holes_by_frame,
        support_mask,
        bundle,
        display_frames=display_records,
        write_recolour_video=False,
        write_side_by_side_video=False,
        write_labelmap_video=False,
    )
    assert status['status'] == 'ok'
    assert status['baseline_activity_video_palette'] == 'uncorrected registered observed RGB'
    assert (out / 'video_cluster_baseline_activity.avi').exists()
    with (out / 'cluster_display_palette.csv').open(newline='', encoding='utf-8') as fh:
        palette_rows = list(csv.DictReader(fh))
    assert palette_rows
    assert all(row['display_source'] == 'uncorrected_registered_observed_pixel_median_rgb' for row in palette_rows)
    assert all(row['display_colour_space'] == 'raw_registered_rgb' for row in palette_rows)
    assert all((int(row['display_r']), int(row['display_g']), int(row['display_b'])) == (91, 73, 55) for row in palette_rows)


def test_cluster_palette_prefers_observed_display_rgb() -> None:
    rows = [{
        'cluster_id': 4,
        'center_h': 0.0,
        'center_s': 0.0,
        'center_l': 0.5,
        'analysis_median_r': 12,
        'analysis_median_g': 34,
        'analysis_median_b': 56,
        'analysis_mean_r': 20,
        'analysis_mean_g': 40,
        'analysis_mean_b': 60,
        'analysis_n_pixels': 100,
        'analysis_sampled_pixels': 50,
        'display_source': 'analysis_observed_sample_median_rgb',
    }]
    palette = _cluster_palette_rows(rows)
    assert palette[0]['cluster_id'] == 4
    assert (palette[0]['display_r'], palette[0]['display_g'], palette[0]['display_b']) == (12, 34, 56)
    assert (palette[0]['centroid_r'], palette[0]['centroid_g'], palette[0]['centroid_b']) == (128, 128, 128)
    assert palette[0]['display_source'] == 'analysis_observed_sample_median_rgb'


def test_gmm_selection_keeps_eight_as_upper_bound_not_target() -> None:
    candidates = [
        {"k": 2, "icl": 100.0, "is_relevant": True},
        {"k": 3, "icl": 96.0, "is_relevant": True},
        {"k": 4, "icl": 94.0, "is_relevant": True},
        {"k": 5, "icl": 93.0, "is_relevant": False},
        {"k": 6, "icl": 92.0, "is_relevant": False},
        {"k": 7, "icl": 91.0, "is_relevant": False},
        {"k": 8, "icl": 90.0, "is_relevant": False},
    ]
    selected = _choose_gmm_candidate(candidates, k_min=2)
    assert selected is not None
    assert selected["k"] == 2


def test_gmm_selection_allows_more_clusters_when_evidence_is_strong() -> None:
    candidates = [
        {"k": 2, "icl": 130.0, "is_relevant": True},
        {"k": 3, "icl": 112.0, "is_relevant": True},
        {"k": 4, "icl": 100.0, "is_relevant": True},
    ]
    selected = _choose_gmm_candidate(candidates, k_min=2)
    assert selected is not None
    assert selected["k"] == 4
