from holecolor.radial.columnar import (
    HotspotStatsTable,
    RdfUncertaintyHoleTable,
    build_hotspot_reticulum_comparison_table,
    build_rdf_hotspot_reticulum_comparison_table,
    build_rdf_uncertainty_hotspot_comparison_table,
    build_rdf_uncertainty_reticulum_rows_table,
)
from holecolor.radial.modeling import (
    build_hotspot_reticulum_comparison,
    build_rdf_hotspot_reticulum_comparison,
)
from holecolor.radial.rdf import (
    build_rdf_uncertainty_hotspot_comparison,
    build_rdf_uncertainty_reticulum_rows,
)


def _sort_rows(rows, keys):
    return sorted(rows, key=lambda r: tuple(r.get(k) for k in keys))


def _assert_rows_close(old_rows, new_rows, keys, atol=1e-6):
    old_rows = _sort_rows(old_rows, keys)
    new_rows = _sort_rows(new_rows, keys)
    assert len(old_rows) == len(new_rows)
    for old, new in zip(old_rows, new_rows):
        assert set(old.keys()) == set(new.keys())
        for k in old:
            ov = old[k]
            nv = new[k]
            if isinstance(ov, float) or isinstance(nv, float):
                if ov is None or nv is None:
                    assert ov == nv
                else:
                    assert abs(float(ov) - float(nv)) <= atol
            else:
                assert ov == nv


def test_milestone37_hotspot_and_uncertainty_internal_tables_match_row_outputs():
    hotspot_rows = [
        {"nearest_hole_id": 1, "dist_to_hole_px": 5.0},
        {"nearest_hole_id": 1, "dist_to_hole_px": 7.0},
        {"nearest_hole_id": 2, "dist_to_hole_px": 30.0},
    ]
    per_hole_radial_summary_rows = [
        {"hole_id": 1, "lattice_u": 0, "lattice_v": 0, "radial_conclusion_label": "outward", "delta_center_of_mass": 0.4, "mean_inner_minus_outer": 0.2, "mean_angular_asymmetry": 0.1},
        {"hole_id": 2, "lattice_u": 1, "lattice_v": 0, "radial_conclusion_label": "flat", "delta_center_of_mass": 0.1, "mean_inner_minus_outer": 0.05, "mean_angular_asymmetry": 0.03},
    ]
    radial_archetype_rows = [
        {"hole_id": 1, "radial_archetype_label": "alpha"},
        {"hole_id": 2, "radial_archetype_label": "beta"},
    ]
    rdf_archetype_rows = [
        {"hole_id": 1, "lattice_u": 0, "lattice_v": 0, "rdf_archetype_canonical_id": 0, "rdf_archetype_canonical_label": "a", "mean_front_radius_norm": 0.5, "delta_front_radius_norm": 0.2},
        {"hole_id": 2, "lattice_u": 1, "lattice_v": 0, "rdf_archetype_canonical_id": 1, "rdf_archetype_canonical_label": "b", "mean_front_radius_norm": 0.7, "delta_front_radius_norm": 0.4},
    ]
    rdf_dynamics_rows = [
        {"hole_id": 1, "rdf_front_acceleration_per_frame2": 0.01, "rdf_front_nonlinearity_gain": 0.02},
        {"hole_id": 2, "rdf_front_acceleration_per_frame2": 0.03, "rdf_front_nonlinearity_gain": 0.04},
    ]
    rdf_bootstrap_rows = [
        {"hole_id": 1, "bootstrap_front_velocity_ci_width": 0.2, "bootstrap_mean_front_radius_ci_width": 0.3, "bootstrap_delta_front_radius_ci_width": 0.4},
        {"hole_id": 2, "bootstrap_front_velocity_ci_width": 0.5, "bootstrap_mean_front_radius_ci_width": 0.6, "bootstrap_delta_front_radius_ci_width": 0.7},
    ]
    rdf_bootstrap_support_rows = [
        {"hole_id": 1, "bootstrap_rdf_archetype_support_fraction": 0.9},
        {"hole_id": 2, "bootstrap_rdf_archetype_support_fraction": 0.8},
    ]
    zone_by_hole = {1: "edge", 2: "interior"}

    hotspot_table = HotspotStatsTable.from_rows(hotspot_rows)

    old_hr, old_hrg = build_hotspot_reticulum_comparison(
        hotspot_rows,
        per_hole_radial_summary_rows,
        radial_archetype_rows,
        zone_by_hole,
    )
    new_hr, new_hrg = build_hotspot_reticulum_comparison_table(
        hotspot_table,
        per_hole_radial_summary_rows,
        radial_archetype_rows,
        zone_by_hole,
    )
    _assert_rows_close(old_hr, new_hr, ["hole_id"])
    _assert_rows_close(old_hrg, new_hrg, ["reticulum_zone", "hotspot_proximity_bucket"])

    old_rh, old_rhg = build_rdf_hotspot_reticulum_comparison(
        hotspot_rows,
        rdf_archetype_rows,
        rdf_dynamics_rows,
        zone_by_hole,
    )
    new_rh, new_rhg = build_rdf_hotspot_reticulum_comparison_table(
        hotspot_table,
        rdf_archetype_rows,
        rdf_dynamics_rows,
        zone_by_hole,
    )
    _assert_rows_close(old_rh, new_rh, ["hole_id"])
    _assert_rows_close(old_rhg, new_rhg, ["reticulum_zone", "hotspot_proximity_bucket", "rdf_archetype_canonical_label"])

    uncertainty_table = RdfUncertaintyHoleTable.from_rows(
        hotspot_table,
        rdf_bootstrap_rows,
        rdf_bootstrap_support_rows,
        rdf_archetype_rows,
        zone_by_hole,
        rdf_dynamics_rows,
    )
    old_ur = build_rdf_uncertainty_reticulum_rows(
        rdf_archetype_rows,
        rdf_bootstrap_rows,
        rdf_bootstrap_support_rows,
        rdf_dynamics_rows,
    )
    new_ur = build_rdf_uncertainty_reticulum_rows_table(uncertainty_table)
    _assert_rows_close(old_ur, new_ur, ["hole_id"])

    old_uh, old_uhg = build_rdf_uncertainty_hotspot_comparison(
        hotspot_rows,
        rdf_bootstrap_rows,
        rdf_bootstrap_support_rows,
        rdf_archetype_rows,
        zone_by_hole,
    )
    new_uh, new_uhg = build_rdf_uncertainty_hotspot_comparison_table(uncertainty_table)
    _assert_rows_close(old_uh, new_uh, ["hole_id"])
    _assert_rows_close(old_uhg, new_uhg, ["reticulum_zone", "hotspot_proximity_bucket"])
