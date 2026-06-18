from holecolor.radial.advanced import build_reticulum_group_rows, per_hole_radial_frame_summary
from holecolor.radial.columnar import (
    RadialRowTable,
    build_per_hole_rdf_evolution_table,
    build_reticulum_group_rows_table,
    fit_radial_models_table,
    per_hole_radial_frame_summary_table,
)
from holecolor.radial.modeling import fit_radial_models
from holecolor.radial.rdf import build_per_hole_rdf_evolution


def _sort_rows(rows, keys):
    return sorted(rows, key=lambda r: tuple(r.get(k) for k in keys))


def _assert_rows_close(old_rows, new_rows, keys, atol=1e-5):
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


def test_milestone35_columnar_radial_paths_match_row_based_outputs():
    radial_rows = [
        {"hole_id": 2, "frame_id": 1, "annulus_id": 1, "descriptor_value": 2.2, "lattice_u": 1, "lattice_v": 0},
        {"hole_id": 1, "frame_id": 0, "annulus_id": 1, "descriptor_value": 1.1, "lattice_u": 0, "lattice_v": 0},
        {"hole_id": 1, "frame_id": 1, "annulus_id": 0, "descriptor_value": 1.4, "lattice_u": 0, "lattice_v": 0},
        {"hole_id": 2, "frame_id": 0, "annulus_id": 0, "descriptor_value": 1.7, "lattice_u": 1, "lattice_v": 0},
        {"hole_id": 1, "frame_id": 0, "annulus_id": 0, "descriptor_value": 1.0, "lattice_u": 0, "lattice_v": 0},
        {"hole_id": 2, "frame_id": 0, "annulus_id": 1, "descriptor_value": 1.9, "lattice_u": 1, "lattice_v": 0},
        {"hole_id": 1, "frame_id": 1, "annulus_id": 1, "descriptor_value": 1.8, "lattice_u": 0, "lattice_v": 0},
        {"hole_id": 2, "frame_id": 1, "annulus_id": 0, "descriptor_value": 1.8, "lattice_u": 1, "lattice_v": 0},
    ]
    zone_by_hole = {1: "edge", 2: "interior"}
    table = RadialRowTable.from_rows(radial_rows)

    _assert_rows_close(
        per_hole_radial_frame_summary(radial_rows),
        per_hole_radial_frame_summary_table(table),
        ["hole_id", "frame_id"],
    )

    old_rdf = build_per_hole_rdf_evolution(radial_rows)
    new_rdf = build_per_hole_rdf_evolution_table(table)
    for old_rows, new_rows, keys in zip(
        old_rdf,
        new_rdf,
        [
            ["hole_id", "frame_id", "annulus_id"],
            ["hole_id", "frame_id"],
            ["hole_id"],
        ],
    ):
        _assert_rows_close(old_rows, new_rows, keys)

    old_fit = fit_radial_models(radial_rows)
    new_fit = fit_radial_models_table(table)
    _assert_rows_close(old_fit[0], new_fit[0], ["hole_id", "frame_id"])
    _assert_rows_close(old_fit[1], new_fit[1], ["hole_id"])

    old_group = build_reticulum_group_rows(radial_rows, zone_by_hole)
    new_group = build_reticulum_group_rows_table(table, zone_by_hole)
    _assert_rows_close(old_group[0], new_group[0], ["reticulum_zone", "frame_id", "annulus_id"])
    _assert_rows_close(old_group[1], new_group[1], ["reticulum_zone", "frame_id"])
