from holecolor.geometry.models import BufferGeometry, HoleTierRecord
from holecolor.geometry.relations import classify_holes_against_buffer, select_holes_for_analysis


def test_hole_buffer_relations_inside_intersect_outside_and_unknown() -> None:
    holes = [
        HoleTierRecord(node_id=1, center_xy_px=(50.0, 50.0), radius_px=5.0, confidence=1.0, lattice_i=0, lattice_j=0, tier=1, source_class='anchor'),
        HoleTierRecord(node_id=2, center_xy_px=(86.0, 50.0), radius_px=6.0, confidence=1.0, lattice_i=1, lattice_j=0, tier=2, source_class='recovered_strong'),
        HoleTierRecord(node_id=3, center_xy_px=(104.0, 50.0), radius_px=5.0, confidence=0.0, lattice_i=2, lattice_j=0, tier=3, source_class='predicted_only'),
    ]
    buffer = BufferGeometry(id='buffer-0', state='full', center_xy_px=(50.0, 50.0), radius_px=40.0, confidence=0.8)
    rel = classify_holes_against_buffer(holes, buffer)
    by_id = {r.node_id: r for r in rel}
    assert by_id[1].relation == 'inside_buffer'
    assert by_id[2].relation == 'intersects_buffer_border'
    assert by_id[3].relation == 'outside_buffer'

    selected, policy = select_holes_for_analysis(holes, rel, include_tiers=(1, 2, 3), exclude_border_intersections_when_known=True)
    assert [h.node_id for h in selected] == [1]
    assert policy['selected_count'] == 1

    unknown = BufferGeometry(id='buffer-0', state='unknown', center_xy_px=None, radius_px=None, confidence=0.0)
    rel2 = classify_holes_against_buffer(holes, unknown)
    assert all(r.relation == 'border_unknown' for r in rel2)
    selected2, policy2 = select_holes_for_analysis(holes, rel2, include_tiers=(1, 2, 3), exclude_border_intersections_when_known=True)
    assert len(selected2) == 3
    assert policy2['selected_count'] == 3


def test_hole_buffer_relations_partial_known_border_excludes_only_intersections() -> None:
    holes = [
        HoleTierRecord(node_id=10, center_xy_px=(30.0, 40.0), radius_px=5.0, confidence=1.0, lattice_i=0, lattice_j=0, tier=1, source_class='anchor'),
        HoleTierRecord(node_id=11, center_xy_px=(56.0, 40.0), radius_px=6.0, confidence=1.0, lattice_i=1, lattice_j=0, tier=2, source_class='recovered_strong'),
        HoleTierRecord(node_id=12, center_xy_px=(72.0, 40.0), radius_px=5.0, confidence=0.5, lattice_i=2, lattice_j=0, tier=3, source_class='predicted_only', visible_fraction=0.5),
    ]
    # center lies outside a hypothetical frame, but border geometry is still known
    buffer = BufferGeometry(id='buffer-0', state='partial', center_xy_px=(-10.0, 40.0), radius_px=60.0, confidence=0.6, center_outside_frame=True)
    rel = classify_holes_against_buffer(holes, buffer)
    by_id = {r.node_id: r for r in rel}
    assert by_id[10].relation == 'inside_buffer'
    assert by_id[11].relation == 'intersects_buffer_border'
    assert by_id[12].relation == 'outside_buffer'
    assert by_id[10].partial_visibility is True
    selected, policy = select_holes_for_analysis(holes, rel, include_tiers=(1, 2, 3), exclude_border_intersections_when_known=True)
    assert [h.node_id for h in selected] == [10]
    assert [e['node_id'] for e in policy['excluded']] == [11, 12]


def test_hole_buffer_relations_falls_back_when_known_border_would_exclude_everything() -> None:
    holes = [
        HoleTierRecord(node_id=20, center_xy_px=(10.0, 10.0), radius_px=5.0, confidence=1.0, lattice_i=0, lattice_j=0, tier=1, source_class='anchor'),
        HoleTierRecord(node_id=21, center_xy_px=(25.0, 10.0), radius_px=5.0, confidence=1.0, lattice_i=1, lattice_j=0, tier=2, source_class='recovered_strong'),
    ]
    buffer = BufferGeometry(id='buffer-0', state='full', center_xy_px=(200.0, 200.0), radius_px=20.0, confidence=0.9)
    rel = classify_holes_against_buffer(holes, buffer)
    selected, policy = select_holes_for_analysis(holes, rel, include_tiers=(1, 2), exclude_border_intersections_when_known=True)
    assert [h.node_id for h in selected] == [20, 21]
    assert policy['fallback_no_selection_guard'] is True
