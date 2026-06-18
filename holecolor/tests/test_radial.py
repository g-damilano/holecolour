from holecolor.core.types import HoleGeometry
from holecolor.masks.terraces import make_hole_terraces
from holecolor.radial.curves import compute_radial_curve
from holecolor.synth.radial_front import make_synthetic_radial_front_video


def test_radial_curve_runs():
    frames, gt = make_synthetic_radial_front_video(n_frames=1)
    img = frames[0][..., 0]
    cx, cy = gt['centers'][0]
    hole = HoleGeometry(0, cx, cy, gt['radius_px'] - 2, gt['radius_px'], 1.0)
    terraces = make_hole_terraces(img.shape[:2], hole, 6, hole.radius_outer_px + 24)
    curve = compute_radial_curve(0, 0, img, terraces, 'r')
    assert len(curve.terrace_values) == 6


def test_summarize_hole_radial_evolution_tolerates_none_metrics():
    from holecolor.radial.advanced import summarize_hole_radial_evolution

    rows = [
        {
            "hole_id": 1,
            "frame_id": 0,
            "lattice_u": 0,
            "lattice_v": 0,
            "center_of_mass_annulus": None,
            "peak_annulus": None,
            "inner_minus_outer": None,
        },
        {
            "hole_id": 1,
            "frame_id": 1,
            "lattice_u": 0,
            "lattice_v": 0,
            "center_of_mass_annulus": 2.0,
            "peak_annulus": 3,
            "inner_minus_outer": -0.5,
        },
    ]

    out = summarize_hole_radial_evolution(rows)
    assert len(out) == 1
    assert out[0]["start_center_of_mass"] == 2.0
    assert out[0]["end_center_of_mass"] == 2.0
    assert out[0]["delta_center_of_mass"] == 0.0
    assert out[0]["start_peak_annulus"] == 3.0
    assert out[0]["end_peak_annulus"] == 3.0
    assert out[0]["mean_inner_minus_outer"] == -0.5
