from pathlib import Path

from holecolor.config.schema import PipelineConfig
from holecolor.core.types import FrameRecord
from holecolor.pipeline import run_milestone1, run_milestone3
from holecolor.synth.artifacts import add_photometric_artifacts
from holecolor.synth.radial_front import make_synthetic_radial_front_video


def test_milestone1_writes_core_artifacts(tmp_path: Path):
    frames, _ = make_synthetic_radial_front_video(n_frames=5)
    frames = add_photometric_artifacts(frames, brightness_ramp=0.15, vignette_strength=0.2)
    records = [FrameRecord(i, float(i), f) for i, f in enumerate(frames)]
    out_dir = tmp_path / "run"
    summary = run_milestone1(records, out_dir, PipelineConfig())

    assert summary["n_frames"] == 5
    assert (out_dir / "summary.json").exists()
    assert (out_dir / "audit" / "frame_qc.csv").exists()
    assert (out_dir / "photometry" / "candidate_scores.csv").exists()
    assert (out_dir / "geometry" / "hole_candidates.csv").exists()
    assert (out_dir / "geometry" / "overlays" / "frame0_geometry_overlay.png").exists()


def test_milestone3_writes_registration_and_radial_artifacts(tmp_path: Path):
    frames, _ = make_synthetic_radial_front_video(n_frames=6)
    frames = add_photometric_artifacts(frames, brightness_ramp=0.10, vignette_strength=0.15)
    records = [FrameRecord(i, float(i), f) for i, f in enumerate(frames)]
    out_dir = tmp_path / "run_m3"
    summary = run_milestone3(records, out_dir, PipelineConfig())

    assert summary["n_frames"] == 6
    assert summary["n_radial_rows"] > 0
    assert (out_dir / "registration" / "transforms.csv").exists()
    assert (out_dir / "geometry" / "hole_geometry.csv").exists()
    assert (out_dir / "descriptors" / "hole_compartment_timeseries.csv").exists()
    assert (out_dir / "descriptors" / "matrix_timeseries.csv").exists()
    assert (out_dir / "radial" / "hole_annulus_timeseries.csv").exists()
    assert (out_dir / "masks" / "overlays" / "frame_ref_terraces_overlay.png").exists()
    assert (out_dir / "masks" / "reference_terrace_plan.csv").exists()
    assert (out_dir / "masks" / "reference_terrace_annuli.csv").exists()
    assert (out_dir / "descriptors" / "wafer_nonhole_colour" / "stage_status.json").exists()
    assert (out_dir / "descriptors" / "wafer_nonhole_colour" / "frame_cluster_summary.csv").exists()
    assert (out_dir / "descriptors" / "wafer_nonhole_colour" / "global_buffer_cluster_context.csv").exists()
    assert (out_dir / "descriptors" / "wafer_nonhole_colour" / "local_hole_cluster_context.csv").exists()
    assert (out_dir / "descriptors" / "wafer_nonhole_colour" / "cluster_video_status.json").exists()
    assert (out_dir / "descriptors" / "wafer_nonhole_colour" / "video_cluster_labels.avi").exists()
    assert (out_dir / "descriptors" / "wafer_nonhole_colour" / "video_cluster_side_by_side.avi").exists()
    assert (out_dir / "descriptors" / "wafer_nonhole_colour" / "video_cluster_recoloured.avi").exists()
    assert (out_dir / "descriptors" / "wafer_nonhole_colour" / "video_cluster_baseline_activity.avi").exists()
    assert (out_dir / "descriptors" / "wafer_nonhole_colour" / "frame_cluster_baseline_activity.csv").exists()
    assert (out_dir / "descriptors" / "radial_cluster_average_hole" / "radial_cluster_status.json").exists()
    assert (out_dir / "descriptors" / "radial_cluster_average_hole" / "average_hole_terrace_chronogram.png").exists()
    assert (out_dir / "descriptors" / "radial_cluster_average_hole" / "cluster_front_trajectories.png").exists()
    assert (out_dir / "descriptors" / "radial_cluster_average_hole" / "hole_consistency_terrace_map.png").exists()
