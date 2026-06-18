from pathlib import Path

from holecolor.config.schema import PipelineConfig
from holecolor.core.types import FrameRecord
from holecolor.pipeline import run_milestone4
from holecolor.synth.artifacts import add_known_drift, add_photometric_artifacts
from holecolor.synth.radial_front import make_synthetic_radial_front_video


def test_milestone6_writes_descriptor_selection_and_per_hole_events(tmp_path: Path):
    frames, _ = make_synthetic_radial_front_video(n_frames=8)
    frames = add_photometric_artifacts(frames, brightness_ramp=0.12, vignette_strength=0.18)
    frames, _ = add_known_drift(frames, dx_per_frame=0.4, dy_per_frame=-0.25)
    records = [FrameRecord(i, float(i), f) for i, f in enumerate(frames)]
    out_dir = tmp_path / "run_m6"
    summary = run_milestone4(records, out_dir, PipelineConfig())
    assert summary["n_frames"] == 8
    assert (out_dir / "descriptors" / "descriptor_selection.json").exists()
    assert (out_dir / "temporal" / "per_hole_events.csv").exists()


def test_strict_gate_failure_raises(tmp_path: Path):
    frames, _ = make_synthetic_radial_front_video(n_frames=6)
    records = [FrameRecord(i, float(i), f) for i, f in enumerate(frames)]
    out_dir = tmp_path / "run_strict_fail"
    cfg = PipelineConfig()
    cfg.qc.fail_on_gate_error = True
    cfg.radial.max_mae_threshold = -1.0
    try:
        run_milestone4(records, out_dir, cfg)
    except RuntimeError as exc:
        assert "QC gates failed" in str(exc)
    else:
        raise AssertionError("Expected strict QC failure to raise RuntimeError")
