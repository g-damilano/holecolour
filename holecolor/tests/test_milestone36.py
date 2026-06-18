from holecolor.radial.columnar import (
    SectorRadialTable,
    SectorRdfFrameTable,
    build_sector_front_acceleration_table,
    build_sector_front_lag_rows_table,
    build_sector_front_propagation_table,
    build_sector_rdf_evolution_table,
    summarize_sector_fronts_table,
)
from holecolor.radial.modeling import summarize_sector_fronts
from holecolor.radial.rdf import (
    build_sector_front_acceleration,
    build_sector_front_lag_rows,
    build_sector_front_propagation,
    build_sector_rdf_evolution,
)


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


def test_milestone36_sector_columnar_paths_match_row_based_outputs():
    sector_rows = []
    for hole_id, uv in [(1, (0, 0)), (2, (1, 0))]:
        for frame_id, frame_gain in [(0, 0.0), (1, 0.2), (2, 0.5)]:
            for sector_id, sector_gain in [(0, 0.0), (1, 0.1)]:
                for annulus_id, base in [(0, 1.0), (1, 1.2), (2, 1.4)]:
                    sector_rows.append(
                        {
                            "hole_id": hole_id,
                            "frame_id": frame_id,
                            "sector_id": sector_id,
                            "annulus_id": annulus_id,
                            "descriptor_value": base + frame_gain + sector_gain + 0.05 * annulus_id,
                            "lattice_u": uv[0],
                            "lattice_v": uv[1],
                        }
                    )
    stable_table = SectorRadialTable.from_rows(sector_rows)

    old_front = summarize_sector_fronts(sector_rows)
    new_front = summarize_sector_fronts_table(stable_table)
    _assert_rows_close(old_front[0], new_front[0], ["hole_id", "frame_id", "sector_id"])
    _assert_rows_close(old_front[1], new_front[1], ["hole_id", "frame_id"])

    old_rdf = build_sector_rdf_evolution(sector_rows)
    new_rdf = build_sector_rdf_evolution_table(stable_table)
    _assert_rows_close(old_rdf[0], new_rdf[0], ["hole_id", "frame_id", "sector_id", "annulus_id"])
    _assert_rows_close(old_rdf[1], new_rdf[1], ["hole_id", "frame_id", "sector_id"])

    frame_table = SectorRdfFrameTable.from_rows(old_rdf[1])
    old_lag = build_sector_front_lag_rows(old_rdf[1], onset_threshold=0.01)
    new_lag = build_sector_front_lag_rows_table(frame_table, onset_threshold=0.01)
    _assert_rows_close(old_lag[0], new_lag[0], ["hole_id", "sector_id"])
    _assert_rows_close(old_lag[1], new_lag[1], ["hole_id"])

    old_prop = build_sector_front_propagation(old_rdf[1], onset_threshold=0.01)
    new_prop = build_sector_front_propagation_table(frame_table, onset_threshold=0.01)
    _assert_rows_close(old_prop[0], new_prop[0], ["hole_id", "sector_id"])
    _assert_rows_close(old_prop[1], new_prop[1], ["hole_id"])

    old_acc = build_sector_front_acceleration(old_rdf[1])
    new_acc = build_sector_front_acceleration_table(frame_table)
    _assert_rows_close(old_acc[0], new_acc[0], ["hole_id", "sector_id"])
    _assert_rows_close(old_acc[1], new_acc[1], ["hole_id"])
