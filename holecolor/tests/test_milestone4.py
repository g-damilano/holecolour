from pathlib import Path

from holecolor.config.schema import PipelineConfig
from holecolor.core.types import FrameRecord
from holecolor.pipeline import run_milestone4
from holecolor.synth.artifacts import add_known_drift, add_photometric_artifacts
from holecolor.synth.radial_front import make_synthetic_radial_front_video


def test_milestone4_writes_hotspot_temporal_and_qc_artifacts(tmp_path: Path):
    frames, _ = make_synthetic_radial_front_video(n_frames=8)
    frames = add_photometric_artifacts(frames, brightness_ramp=0.12, vignette_strength=0.18)
    frames, _ = add_known_drift(frames, dx_per_frame=0.6, dy_per_frame=-0.4)
    records = [FrameRecord(i, float(i), f) for i, f in enumerate(frames)]
    out_dir = tmp_path / "run_m4"
    summary = run_milestone4(records, out_dir, PipelineConfig())

    assert summary["n_frames"] == 8
    assert summary["n_events"] > 0
    assert (out_dir / "hotspots" / "hotspots.csv").exists()
    assert (out_dir / "hotspots" / "tracks.csv").exists()
    assert (out_dir / "temporal" / "events.csv").exists()
    assert (out_dir / "temporal" / "matrix_and_hotspot_fraction.png").exists()
    assert (out_dir / "qc" / "radial_perturbation.csv").exists()


def test_milestone5_writes_track_summary_and_threshold_envelope(tmp_path: Path):
    frames, _ = make_synthetic_radial_front_video(n_frames=8)
    frames = add_photometric_artifacts(frames, brightness_ramp=0.12, vignette_strength=0.18)
    frames, _ = add_known_drift(frames, dx_per_frame=0.4, dy_per_frame=-0.25)
    records = [FrameRecord(i, float(i), f) for i, f in enumerate(frames)]
    out_dir = tmp_path / "run_m5"
    summary = run_milestone4(records, out_dir, PipelineConfig())

    assert summary["n_frames"] == 8
    assert (out_dir / "hotspots" / "track_summary.csv").exists()
    assert (out_dir / "qc" / "threshold_envelope.json").exists()
