from holecolor.config.schema import PhotometryConfig
from holecolor.core.types import FrameRecord
from holecolor.photometry.selector import run_photometry_selection
from holecolor.synth.artifacts import add_photometric_artifacts
from holecolor.synth.radial_front import make_synthetic_radial_front_video


def test_photometry_selection_runs():
    frames, _ = make_synthetic_radial_front_video(n_frames=5)
    frames = add_photometric_artifacts(frames)
    records = [FrameRecord(i, float(i), f) for i, f in enumerate(frames)]
    winner, scores, corrected = run_photometry_selection(records, PhotometryConfig())
    assert winner in {"none", "gain_norm", "flatfield_poly"}
    assert len(scores) == len(records) * len(PhotometryConfig().candidate_methods)
    assert len(corrected) == len(records)
