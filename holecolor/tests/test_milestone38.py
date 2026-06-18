from holecolor.radial.columnar import (
    HotspotStatsTable,
    RdfUncertaintyHoleTable,
    ValidationHoleTable,
    build_validation_summary_jsons_table,
)


def test_milestone38_validation_table_reuses_bootstrap_support_and_sector_metrics():
    rdf_bootstrap_rows = [
        {"hole_id": 1, "bootstrap_front_velocity_ci_width": 0.2, "bootstrap_mean_front_radius_ci_width": 0.3, "bootstrap_delta_front_radius_ci_width": 0.4},
        {"hole_id": 2, "bootstrap_front_velocity_ci_width": 0.6, "bootstrap_mean_front_radius_ci_width": 0.5, "bootstrap_delta_front_radius_ci_width": 0.8},
    ]
    rdf_bootstrap_support_rows = [
        {"hole_id": 1, "bootstrap_rdf_archetype_support_fraction": 0.9},
        {"hole_id": 2, "bootstrap_rdf_archetype_support_fraction": 0.7},
    ]
    sector_front_propagation_hole_rows = [
        {"hole_id": 1, "valid_sector_velocity_fraction": 1.0},
        {"hole_id": 2, "valid_sector_velocity_fraction": 0.5},
    ]
    sector_front_acceleration_hole_rows = [
        {"hole_id": 1, "valid_sector_acceleration_fraction": 0.75},
        {"hole_id": 2, "valid_sector_acceleration_fraction": 0.25},
    ]
    hotspot_rows = [
        {"nearest_hole_id": 1, "dist_to_hole_px": 5.0},
        {"nearest_hole_id": 2, "dist_to_hole_px": 15.0},
    ]
    rdf_archetype_rows = [
        {"hole_id": 1, "lattice_u": 0, "lattice_v": 0, "rdf_archetype_canonical_id": 0, "rdf_archetype_canonical_label": "a"},
        {"hole_id": 2, "lattice_u": 1, "lattice_v": 0, "rdf_archetype_canonical_id": 1, "rdf_archetype_canonical_label": "b"},
    ]
    rdf_dynamics_rows = [
        {"hole_id": 1, "rdf_front_acceleration_per_frame2": 0.02, "rdf_front_nonlinearity_gain": 0.03},
        {"hole_id": 2, "rdf_front_acceleration_per_frame2": 0.04, "rdf_front_nonlinearity_gain": 0.05},
    ]
    zone_by_hole = {1: "edge", 2: "interior"}

    validation_table = ValidationHoleTable.from_rows(
        rdf_bootstrap_rows,
        rdf_bootstrap_support_rows,
        sector_front_propagation_hole_rows,
        sector_front_acceleration_hole_rows,
    )
    assert abs(validation_table.mean_bootstrap_support_fraction() - 0.8) < 1e-6
    assert abs(validation_table.mean_bootstrap_front_velocity_ci_width() - 0.4) < 1e-6
    assert abs(validation_table.mean_sector_velocity_valid_fraction() - 0.75) < 1e-6
    assert abs(validation_table.mean_sector_acceleration_valid_fraction() - 0.5) < 1e-6

    hotspot_table = HotspotStatsTable.from_rows(hotspot_rows)
    uncertainty_table = RdfUncertaintyHoleTable.from_rows(
        hotspot_table,
        rdf_bootstrap_rows,
        rdf_bootstrap_support_rows,
        rdf_archetype_rows,
        zone_by_hole,
        rdf_dynamics_rows,
        validation_table=validation_table,
    )
    bootstrap_json, uncertainty_json = build_validation_summary_jsons_table(
        validation_table,
        uncertainty_table,
        validation_enabled=True,
    )
    assert abs(bootstrap_json["mean_bootstrap_class_support"] - 0.8) < 1e-6
    assert abs(bootstrap_json["mean_velocity_ci_width"] - 0.4) < 1e-6
    assert abs(bootstrap_json["mean_sector_propagation_valid_fraction"] - 0.75) < 1e-6
    assert abs(uncertainty_json["mean_sector_acceleration_valid_fraction"] - 0.5) < 1e-6
    assert uncertainty_json["mean_uncertainty_score"] is not None
    assert abs(uncertainty_json["mean_hotspot_linked_velocity_ci_width"] - 0.4) < 1e-6
