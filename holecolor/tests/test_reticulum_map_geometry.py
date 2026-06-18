from __future__ import annotations

from holecolor.core.types import HoleGeometry
from holecolor.pipeline import _phenotype_color, _reticulum_canvas_points


def test_phenotype_color_accepts_prefixed_labels() -> None:
    assert _phenotype_color("P2_outer_first") == _phenotype_color("outer_first")
    assert _phenotype_color("P3_irregular") == _phenotype_color("irregular")


def test_reticulum_canvas_uses_detected_hole_positions_not_axis_flat_uv() -> None:
    rows = [
        {"hole_id": 0, "lattice_u": 0, "lattice_v": 0},
        {"hole_id": 1, "lattice_u": 1, "lattice_v": 0},
        {"hole_id": 2, "lattice_u": 0, "lattice_v": 1},
    ]
    holes = [
        HoleGeometry(hole_id=0, x=100.0, y=100.0, radius_inner_px=5.0, radius_outer_px=6.0, confidence=1.0),
        HoleGeometry(hole_id=1, x=80.0, y=140.0, radius_inner_px=5.0, radius_outer_px=6.0, confidence=1.0),
        HoleGeometry(hole_id=2, x=150.0, y=130.0, radius_inner_px=5.0, radius_outer_px=6.0, confidence=1.0),
    ]

    points = _reticulum_canvas_points(rows, holes, None, (240, 240))
    by_hole = {int(row["hole_id"]): (x, y) for row, x, y in points}

    assert by_hole[1][1] != by_hole[0][1]
    assert by_hole[2][0] != by_hole[0][0]
