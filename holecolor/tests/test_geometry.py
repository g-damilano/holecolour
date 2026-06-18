from holecolor.geometry.candidates import detect_dark_hole_candidates
from holecolor.geometry.lattice_fit import estimate_lattice_basis
from holecolor.config.schema import GeometryConfig
from holecolor.synth.grid import make_synthetic_grid


def test_detect_candidates_and_lattice():
    img, _gt = make_synthetic_grid(noise_sigma=2.0, blur_sigma=1.0)
    candidates = detect_dark_hole_candidates(img, GeometryConfig())
    assert len(candidates) > 10
    lattice = estimate_lattice_basis(candidates)
    assert lattice.confidence >= 0.0
