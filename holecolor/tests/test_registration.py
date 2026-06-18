from holecolor.core.types import FrameRecord
from holecolor.registration.rigid import residual_difference, stabilize_sequence
from holecolor.synth.artifacts import add_known_drift
from holecolor.synth.grid import make_synthetic_grid


def test_registration_reduces_residual():
    base, _ = make_synthetic_grid(blur_sigma=1.0, noise_sigma=1.0)
    frames = [base for _ in range(5)]
    drifted, _tfms = add_known_drift(frames, dx_per_frame=1.5, dy_per_frame=-1.0)
    records = [FrameRecord(i, float(i), f) for i, f in enumerate(drifted)]
    before = residual_difference(records[-1].image, records[0].image)
    stabilized, _ = stabilize_sequence(records, reference_idx=0, max_shift_px=20)
    after = residual_difference(stabilized[-1].image, stabilized[0].image)
    assert after < before
