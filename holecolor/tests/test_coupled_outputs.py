from holecolor.geometry.models import HoleTierRecord
from holecolor.pipeline import _coupled_interpretation_rows, _coupled_local_global_rows


def test_coupled_outputs_propagate_tiers_and_build_interpretation_rows() -> None:
    tier_records = [
        HoleTierRecord(
            node_id=0,
            center_xy_px=(10.0, 10.0),
            radius_px=5.0,
            confidence=0.9,
            lattice_i=1,
            lattice_j=2,
            tier=1,
            source_class="anchor",
            visible_fraction=1.0,
            border_refined=True,
            notes="",
        )
    ]
    compartment_rows = [
        {
            "frame_id": 0,
            "hole_id": 0,
            "lattice_u": 1,
            "lattice_v": 2,
            "region_id": "hole_0_interior",
            "mean_H": 0.40,
            "mean_S": 0.50,
            "mean_B": 0.60,
            "area_px": 10,
        },
        {
            "frame_id": 1,
            "hole_id": 0,
            "lattice_u": 1,
            "lattice_v": 2,
            "region_id": "hole_0_interior",
            "mean_H": 0.45,
            "mean_S": 0.55,
            "mean_B": 0.80,
            "area_px": 10,
        },
    ]
    global_rows = [
        {
            "frame_id": 0,
            "region_policy": "buffer_known",
            "descriptor": "b",
            "mean_H": 0.20,
            "mean_S": 0.25,
            "mean_B": 0.30,
            "primary_descriptor_mean": 0.30,
            "primary_descriptor_median": 0.28,
            "hotspot_fraction_of_buffer_area": 0.10,
        },
        {
            "frame_id": 1,
            "region_policy": "buffer_known",
            "descriptor": "b",
            "mean_H": 0.22,
            "mean_S": 0.26,
            "mean_B": 0.40,
            "primary_descriptor_mean": 0.40,
            "primary_descriptor_median": 0.38,
            "hotspot_fraction_of_buffer_area": 0.20,
        },
    ]

    coupled = _coupled_local_global_rows(compartment_rows, global_rows, tier_records=tier_records)
    assert len(coupled) == 2
    assert coupled[0]["tier"] == 1
    assert coupled[0]["source_class"] == "anchor"
    assert coupled[0]["region_class"] == "interior"
    assert abs(coupled[0]["local_to_global_primary_ratio"] - 2.0) < 1e-9

    interp = _coupled_interpretation_rows(coupled)
    assert len(interp) == 1
    row = interp[0]
    assert row["hole_id"] == 0
    assert row["region_class"] == "interior"
    assert row["tier"] == 1
    assert row["source_class"] == "anchor"
    assert abs(row["mean_local_minus_global_primary_descriptor"] - 0.35) < 1e-9
    assert abs(row["late_minus_early_primary_delta"] - 0.10) < 1e-9

from holecolor.pipeline import _coupled_position_class_summary_rows, _coupled_scientific_synthesis_rows

def test_coupled_position_and_scientific_synthesis_rows() -> None:
    rows = [
        {
            "frame_id": 0,
            "hole_id": 1,
            "region_class": "interior",
            "tier": 1,
            "source_class": "anchor",
            "visible_fraction": 1.0,
            "local_minus_global_primary_descriptor": 0.30,
            "global_hotspot_fraction_of_buffer_area": 0.25,
        },
        {
            "frame_id": 1,
            "hole_id": 1,
            "region_class": "interior",
            "tier": 1,
            "source_class": "anchor",
            "visible_fraction": 1.0,
            "local_minus_global_primary_descriptor": 0.10,
            "global_hotspot_fraction_of_buffer_area": 0.15,
        },
        {
            "frame_id": 1,
            "hole_id": 2,
            "region_class": "rim",
            "tier": 2,
            "source_class": "recovered_strong",
            "visible_fraction": 0.8,
            "local_minus_global_primary_descriptor": -0.25,
            "global_hotspot_fraction_of_buffer_area": 0.01,
        },
    ]
    pos = _coupled_position_class_summary_rows(rows)
    syn = _coupled_scientific_synthesis_rows(rows)
    assert pos
    assert syn
    assert {"region_class", "tier", "source_class", "n_rows"} <= set(pos[0].keys())
    assert {"region_class", "source_class", "response_class", "hotspot_context_class", "narrative_label"} <= set(syn[0].keys())
