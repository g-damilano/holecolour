from holecolor.radial.rdf import build_per_hole_rdf_evolution, build_per_hole_rdf_archetypes, build_per_hole_rdf_front_dynamics


def test_milestone32_rdf_grouping_paths_handle_unsorted_rows():
    rows = [
        {"hole_id": 2, "frame_id": 1, "annulus_id": 1, "descriptor_value": 2.0},
        {"hole_id": 1, "frame_id": 1, "annulus_id": 1, "descriptor_value": 1.5},
        {"hole_id": 1, "frame_id": 0, "annulus_id": 0, "descriptor_value": 1.0},
        {"hole_id": 2, "frame_id": 0, "annulus_id": 0, "descriptor_value": 1.0},
        {"hole_id": 1, "frame_id": 1, "annulus_id": 0, "descriptor_value": 2.0},
        {"hole_id": 2, "frame_id": 1, "annulus_id": 0, "descriptor_value": 3.0},
        {"hole_id": 1, "frame_id": 0, "annulus_id": 1, "descriptor_value": 1.0},
        {"hole_id": 2, "frame_id": 0, "annulus_id": 1, "descriptor_value": 1.0},
    ]
    evo, frame_rows, vel = build_per_hole_rdf_evolution(rows)
    assert len(evo) == len(rows)
    assert len(frame_rows) == 4
    assert len(vel) == 2
    arche, centroids = build_per_hole_rdf_archetypes(frame_rows, k=2)
    assert len(arche) == 2
    assert centroids
    dyn = build_per_hole_rdf_front_dynamics(frame_rows)
    assert len(dyn) == 2
