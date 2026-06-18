from holecolor.radial.modeling import fit_radial_models, summarize_sector_fronts
from holecolor.radial.rdf import (
    build_rdf_uncertainty_hotspot_comparison,
    build_sector_front_acceleration,
    build_sector_front_lag_rows,
    build_sector_rdf_evolution,
)


def test_milestone34_sector_and_uncertainty_grouping_paths_handle_unsorted_rows():
    sector_radial_rows = [
        {"hole_id": 1, "frame_id": 1, "sector_id": 1, "annulus_id": 1, "descriptor_value": 1.8, "lattice_u": 0, "lattice_v": 0},
        {"hole_id": 1, "frame_id": 0, "sector_id": 0, "annulus_id": 0, "descriptor_value": 1.0, "lattice_u": 0, "lattice_v": 0},
        {"hole_id": 1, "frame_id": 1, "sector_id": 0, "annulus_id": 0, "descriptor_value": 1.3, "lattice_u": 0, "lattice_v": 0},
        {"hole_id": 1, "frame_id": 0, "sector_id": 1, "annulus_id": 1, "descriptor_value": 1.0, "lattice_u": 0, "lattice_v": 0},
        {"hole_id": 1, "frame_id": 2, "sector_id": 0, "annulus_id": 1, "descriptor_value": 1.9, "lattice_u": 0, "lattice_v": 0},
        {"hole_id": 1, "frame_id": 0, "sector_id": 0, "annulus_id": 1, "descriptor_value": 1.0, "lattice_u": 0, "lattice_v": 0},
        {"hole_id": 1, "frame_id": 2, "sector_id": 1, "annulus_id": 0, "descriptor_value": 1.8, "lattice_u": 0, "lattice_v": 0},
        {"hole_id": 1, "frame_id": 1, "sector_id": 0, "annulus_id": 1, "descriptor_value": 1.7, "lattice_u": 0, "lattice_v": 0},
        {"hole_id": 1, "frame_id": 2, "sector_id": 0, "annulus_id": 0, "descriptor_value": 1.5, "lattice_u": 0, "lattice_v": 0},
        {"hole_id": 1, "frame_id": 0, "sector_id": 1, "annulus_id": 0, "descriptor_value": 1.0, "lattice_u": 0, "lattice_v": 0},
        {"hole_id": 1, "frame_id": 1, "sector_id": 1, "annulus_id": 0, "descriptor_value": 1.4, "lattice_u": 0, "lattice_v": 0},
        {"hole_id": 1, "frame_id": 2, "sector_id": 1, "annulus_id": 1, "descriptor_value": 2.1, "lattice_u": 0, "lattice_v": 0},
        {"hole_id": 1, "frame_id": 3, "sector_id": 0, "annulus_id": 0, "descriptor_value": 1.7, "lattice_u": 0, "lattice_v": 0},
        {"hole_id": 1, "frame_id": 3, "sector_id": 0, "annulus_id": 1, "descriptor_value": 2.2, "lattice_u": 0, "lattice_v": 0},
        {"hole_id": 1, "frame_id": 3, "sector_id": 1, "annulus_id": 0, "descriptor_value": 2.0, "lattice_u": 0, "lattice_v": 0},
        {"hole_id": 1, "frame_id": 3, "sector_id": 1, "annulus_id": 1, "descriptor_value": 2.5, "lattice_u": 0, "lattice_v": 0},
    ]
    evo, frame_rows = build_sector_rdf_evolution(sector_radial_rows)
    assert len(evo) == len(sector_radial_rows)
    assert len(frame_rows) == 8

    lag_rows, lag_summary = build_sector_front_lag_rows(frame_rows)
    assert len(lag_rows) == 2
    assert len(lag_summary) == 1

    acc_rows, acc_summary = build_sector_front_acceleration(frame_rows)
    assert len(acc_rows) == 2
    assert len(acc_summary) == 1

    sector_front_rows, sector_front_summary = summarize_sector_fronts(sector_radial_rows)
    assert len(sector_front_rows) == 8
    assert len(sector_front_summary) == 4


def test_milestone34_hotspot_uncertainty_aggregation_avoids_bucket_lists():
    hotspot_rows = [
        {"nearest_hole_id": 1, "dist_to_hole_px": 5.0},
        {"nearest_hole_id": 1, "dist_to_hole_px": 7.0},
        {"nearest_hole_id": 2, "dist_to_hole_px": 30.0},
    ]
    rdf_bootstrap_rows = [
        {"hole_id": 1, "bootstrap_front_velocity_ci_width": 0.2, "bootstrap_mean_front_radius_ci_width": 0.3, "bootstrap_delta_front_radius_ci_width": 0.4},
        {"hole_id": 2, "bootstrap_front_velocity_ci_width": 0.5, "bootstrap_mean_front_radius_ci_width": 0.6, "bootstrap_delta_front_radius_ci_width": 0.7},
    ]
    rdf_bootstrap_support_rows = [
        {"hole_id": 1, "bootstrap_rdf_archetype_support_fraction": 0.9},
        {"hole_id": 2, "bootstrap_rdf_archetype_support_fraction": 0.8},
    ]
    rdf_archetype_rows = [
        {"hole_id": 1, "lattice_u": 0, "lattice_v": 0, "rdf_archetype_canonical_id": 0, "rdf_archetype_canonical_label": "a"},
        {"hole_id": 2, "lattice_u": 1, "lattice_v": 0, "rdf_archetype_canonical_id": 1, "rdf_archetype_canonical_label": "b"},
    ]
    zone_by_hole = {1: "edge", 2: "interior"}
    out, groups = build_rdf_uncertainty_hotspot_comparison(
        hotspot_rows,
        rdf_bootstrap_rows,
        rdf_bootstrap_support_rows,
        rdf_archetype_rows,
        zone_by_hole,
    )
    assert len(out) == 2
    assert len(groups) == 2
    near = [r for r in out if r["hole_id"] == 1][0]
    assert near["hotspot_proximity_bucket"] == "near"
    assert near["mean_hotspot_distance_px"] == 6.0


def test_milestone34_fit_radial_models_unsorted_rows():
    radial_rows = [
        {"hole_id": 1, "frame_id": 1, "annulus_id": 1, "descriptor_value": 2.0, "lattice_u": 0, "lattice_v": 0},
        {"hole_id": 1, "frame_id": 0, "annulus_id": 0, "descriptor_value": 1.0, "lattice_u": 0, "lattice_v": 0},
        {"hole_id": 1, "frame_id": 1, "annulus_id": 0, "descriptor_value": 1.5, "lattice_u": 0, "lattice_v": 0},
        {"hole_id": 1, "frame_id": 0, "annulus_id": 1, "descriptor_value": 1.2, "lattice_u": 0, "lattice_v": 0},
        {"hole_id": 1, "frame_id": 2, "annulus_id": 0, "descriptor_value": 1.8, "lattice_u": 0, "lattice_v": 0},
        {"hole_id": 1, "frame_id": 2, "annulus_id": 1, "descriptor_value": 2.3, "lattice_u": 0, "lattice_v": 0},
    ]
    fit_rows, hole_summary = fit_radial_models(radial_rows)
    assert len(fit_rows) == 3
    assert len(hole_summary) == 1
