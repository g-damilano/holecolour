from holecolor.core.types import HoleCandidate, HoleGeometry
from holecolor.geometry.completeness import filter_complete_holes_and_terraces
from holecolor.masks.terraces import build_terrace_width_plan, make_nonoverlapping_hole_terraces


def test_half_gap_terrace_width_plan_uses_border_half_gap() -> None:
    holes = [
        HoleGeometry(0, 30.0, 30.0, 8.0, 10.0, 1.0),
        HoleGeometry(1, 70.0, 30.0, 8.0, 10.0, 1.0),
    ]
    lattice_indices = {0: (0, 0), 1: (1, 0)}
    plans = build_terrace_width_plan(holes, 5, lattice_indices=lattice_indices, width_mode='half_gap', gap_basis='border_gap')
    p0 = plans[0]
    assert p0.min_center_pitch_px == 40.0
    assert p0.min_border_gap_px == 20.0
    assert p0.usable_annulus_span_px == 10.0
    assert p0.annulus_width_px == 2.0
    assert p0.terrace_outer_radius_px == 20.0


def test_make_nonoverlapping_hole_terraces_can_return_plan() -> None:
    holes = [
        HoleGeometry(0, 30.0, 30.0, 8.0, 10.0, 1.0),
        HoleGeometry(1, 70.0, 30.0, 8.0, 10.0, 1.0),
    ]
    lattice_indices = {0: (0, 0), 1: (1, 0)}
    terraces, plan = make_nonoverlapping_hole_terraces((100, 100), holes, 4, lattice_indices=lattice_indices, width_mode='half_gap', gap_basis='border_gap', return_plan=True)
    assert set(terraces) == {0, 1}
    assert set(plan) == {0, 1}
    assert len(terraces[0]) == 4
    assert plan[0].span_mode == 'half_gap'


def test_complete_geometry_filter_excludes_partial_terraces_at_frame_edge() -> None:
    holes = [
        HoleGeometry(0, 30.0, 30.0, 8.0, 10.0, 1.0),
        HoleGeometry(1, 70.0, 30.0, 8.0, 10.0, 1.0),
        HoleGeometry(2, 93.0, 30.0, 8.0, 10.0, 1.0),
    ]
    candidates = [HoleCandidate(h.x, h.y, h.radius_outer_px, 0.0, 0.0, h.confidence) for h in holes]
    result = filter_complete_holes_and_terraces(
        holes,
        candidates,
        {0: (0, 0), 1: (1, 0), 2: (2, 0)},
        (100, 100),
        n_terraces=5,
        terrace_width_mode='half_gap',
        terrace_gap_basis='border_gap',
        terrace_min_width_px=0.0,
    )
    assert [h.hole_id for h in result.holes] == [0, 1]
    assert result.summary['excluded_by_reason']['terrace_outside_frame'] == 1
    assert any(row['status'] == 'excluded' and row['original_hole_id'] == 2 for row in result.rows)


def test_complete_geometry_filter_excludes_hole_cut_by_frame_edge() -> None:
    holes = [HoleGeometry(0, 8.0, 50.0, 8.0, 10.0, 1.0)]
    candidates = [HoleCandidate(h.x, h.y, h.radius_outer_px, 0.0, 0.0, h.confidence) for h in holes]

    result = filter_complete_holes_and_terraces(
        holes,
        candidates,
        {0: (0, 0)},
        (100, 100),
        n_terraces=5,
        terrace_width_mode='fixed',
        terrace_gap_basis='border_gap',
        terrace_min_width_px=0.0,
    )

    assert result.holes == []
    excluded = next(row for row in result.rows if row['status'] == 'excluded')
    assert 'hole_outside_frame' in excluded['exclude_reasons']
    assert not excluded['hole_inside_frame']


def test_complete_geometry_filter_excludes_complete_hole_with_incomplete_outer_terrace() -> None:
    holes = [HoleGeometry(0, 80.0, 50.0, 8.0, 10.0, 1.0)]
    candidates = [HoleCandidate(h.x, h.y, h.radius_outer_px, 0.0, 0.0, h.confidence) for h in holes]

    result = filter_complete_holes_and_terraces(
        holes,
        candidates,
        {0: (0, 0)},
        (100, 100),
        n_terraces=5,
        terrace_width_mode='fixed',
        terrace_gap_basis='border_gap',
        terrace_min_width_px=0.0,
    )

    assert result.holes == []
    excluded = next(row for row in result.rows if row['status'] == 'excluded')
    assert excluded['hole_inside_frame']
    assert not excluded['terrace_inside_frame']
    assert 'terrace_outside_frame' in excluded['exclude_reasons']


def test_complete_geometry_filter_excludes_partial_terraces_outside_wafer_circle() -> None:
    holes = [
        HoleGeometry(0, 50.0, 50.0, 4.0, 5.0, 1.0),
        HoleGeometry(1, 80.0, 50.0, 4.0, 5.0, 1.0),
    ]
    candidates = [HoleCandidate(h.x, h.y, h.radius_outer_px, 0.0, 0.0, h.confidence) for h in holes]
    result = filter_complete_holes_and_terraces(
        holes,
        candidates,
        {0: (0, 0), 1: (1, 0)},
        (100, 100),
        n_terraces=5,
        terrace_width_mode='half_gap',
        terrace_gap_basis='border_gap',
        terrace_min_width_px=0.0,
        support_circle=(50.0, 50.0, 35.0),
    )
    assert len(result.holes) == 1
    assert result.holes[0].x == 50.0
    assert result.summary['excluded_by_reason']['terrace_outside_wafer'] == 1
