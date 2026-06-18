from pathlib import Path

import pandas as pd

from holecolor.qc.reports import write_table_columns
from holecolor.radial.columnar import (
    HotspotStatsTable,
    RdfUncertaintyHoleTable,
    ValidationHoleTable,
    build_hotspot_reticulum_columns_table,
    build_hotspot_reticulum_comparison_table,
    build_rdf_hotspot_reticulum_columns_table,
    build_rdf_hotspot_reticulum_comparison_table,
)


def test_milestone44_direct_columns_match_row_outputs(tmp_path: Path) -> None:
    hotspot_rows = [
        {"nearest_hole_id": 1, "dist_to_hole_px": 8.0},
        {"nearest_hole_id": 1, "dist_to_hole_px": 10.0},
        {"nearest_hole_id": 2, "dist_to_hole_px": 30.0},
    ]
    radial_summary_rows = [
        {"hole_id": 1, "lattice_u": 0, "lattice_v": 0, "radial_conclusion_label": "outward", "delta_center_of_mass": 0.5, "mean_inner_minus_outer": 0.2, "mean_angular_asymmetry": 0.1},
        {"hole_id": 2, "lattice_u": 1, "lattice_v": 0, "radial_conclusion_label": "flat", "delta_center_of_mass": 0.1, "mean_inner_minus_outer": 0.05, "mean_angular_asymmetry": 0.03},
    ]
    radial_archetype_rows = [
        {"hole_id": 1, "radial_archetype_label": "A"},
        {"hole_id": 2, "radial_archetype_label": "B"},
    ]
    rdf_archetype_rows = [
        {"hole_id": 1, "lattice_u": 0, "lattice_v": 0, "rdf_archetype_canonical_id": 0, "rdf_archetype_canonical_label": "rdf_A", "mean_front_radius_norm": 0.3, "delta_front_radius_norm": 0.2},
        {"hole_id": 2, "lattice_u": 1, "lattice_v": 0, "rdf_archetype_canonical_id": 1, "rdf_archetype_canonical_label": "rdf_B", "mean_front_radius_norm": 0.4, "delta_front_radius_norm": 0.25},
    ]
    rdf_dynamics_rows = [
        {"hole_id": 1, "rdf_front_acceleration_per_frame2": 0.01, "rdf_front_nonlinearity_gain": 1.1},
        {"hole_id": 2, "rdf_front_acceleration_per_frame2": 0.02, "rdf_front_nonlinearity_gain": 1.2},
    ]
    zone_by_hole = {1: "corner", 2: "edge"}

    hotspot_table = HotspotStatsTable.from_rows(hotspot_rows)

    row_rows, row_groups = build_hotspot_reticulum_comparison_table(hotspot_table, radial_summary_rows, radial_archetype_rows, zone_by_hole)
    col_rows, col_groups = build_hotspot_reticulum_columns_table(hotspot_table, radial_summary_rows, radial_archetype_rows, zone_by_hole)
    rr = tmp_path / "hr_rows.csv"
    rc = tmp_path / "hr_cols.csv"
    write_table_columns(rc, col_rows)
    pd.DataFrame(row_rows).to_csv(rr, index=False)
    assert pd.read_csv(rr).fillna("NA").to_dict("list") == pd.read_csv(rc).fillna("NA").to_dict("list")

    rg = tmp_path / "hr_groups_rows.csv"
    cg = tmp_path / "hr_groups_cols.csv"
    pd.DataFrame(row_groups).to_csv(rg, index=False)
    write_table_columns(cg, col_groups)
    assert pd.read_csv(rg).fillna("NA").to_dict("list") == pd.read_csv(cg).fillna("NA").to_dict("list")

    row_rows2, row_groups2 = build_rdf_hotspot_reticulum_comparison_table(hotspot_table, rdf_archetype_rows, rdf_dynamics_rows, zone_by_hole)
    col_rows2, col_groups2 = build_rdf_hotspot_reticulum_columns_table(hotspot_table, rdf_archetype_rows, rdf_dynamics_rows, zone_by_hole)
    rr2 = tmp_path / "rh_rows.csv"
    rc2 = tmp_path / "rh_cols.csv"
    pd.DataFrame(row_rows2).to_csv(rr2, index=False)
    write_table_columns(rc2, col_rows2)
    assert pd.read_csv(rr2).fillna("NA").to_dict("list") == pd.read_csv(rc2).fillna("NA").to_dict("list")

    rg2 = tmp_path / "rh_groups_rows.csv"
    cg2 = tmp_path / "rh_groups_cols.csv"
    pd.DataFrame(row_groups2).to_csv(rg2, index=False)
    write_table_columns(cg2, col_groups2)
    assert pd.read_csv(rg2).fillna("NA").to_dict("list") == pd.read_csv(cg2).fillna("NA").to_dict("list")

    validation_table = ValidationHoleTable.from_rows(
        rdf_bootstrap_rows=[
            {"hole_id": 1, "bootstrap_front_velocity_ci_width": 0.1, "bootstrap_mean_front_radius_ci_width": 0.2, "bootstrap_delta_front_radius_ci_width": 0.3},
            {"hole_id": 2, "bootstrap_front_velocity_ci_width": 0.2, "bootstrap_mean_front_radius_ci_width": 0.4, "bootstrap_delta_front_radius_ci_width": 0.5},
        ],
        rdf_bootstrap_support_rows=[
            {"hole_id": 1, "bootstrap_rdf_archetype_support_fraction": 0.8},
            {"hole_id": 2, "bootstrap_rdf_archetype_support_fraction": 0.7},
        ],
    )
    uncertainty = RdfUncertaintyHoleTable.from_rows(hotspot_table, [], [], rdf_archetype_rows, zone_by_hole, rdf_dynamics_rows, validation_table=validation_table)
    ret_cols = uncertainty.reticulum_columns()
    hot_cols, hot_group_cols = uncertainty.hotspot_comparison_columns()
    assert len(ret_cols["hole_id"]) == 2
    assert len(hot_cols["hole_id"]) == 2
    assert len(hot_group_cols["reticulum_zone"]) >= 1
