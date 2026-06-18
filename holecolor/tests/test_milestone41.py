from __future__ import annotations

from pathlib import Path

from holecolor.qc.reports import write_table_columns
from holecolor.radial.columnar import RadialRowTable, SectorRadialTable


def test_milestone41_column_writers_roundtrip(tmp_path: Path) -> None:
    radial_rows = [
        {"hole_id": 1, "frame_id": 0, "annulus_id": 0, "descriptor_value": 0.1, "lattice_u": 0, "lattice_v": 0},
        {"hole_id": 1, "frame_id": 1, "annulus_id": 1, "descriptor_value": 0.2, "lattice_u": 0, "lattice_v": 0},
    ]
    sector_rows = [
        {"hole_id": 1, "frame_id": 0, "sector_id": 0, "annulus_id": 0, "descriptor_value": 0.3, "lattice_u": 0, "lattice_v": 0},
        {"hole_id": 1, "frame_id": 1, "sector_id": 1, "annulus_id": 1, "descriptor_value": 0.4, "lattice_u": 0, "lattice_v": 0},
    ]
    rtab = RadialRowTable.from_rows(radial_rows)
    stab = SectorRadialTable.from_rows(sector_rows)
    out1 = tmp_path / 'radial.csv'
    out2 = tmp_path / 'sector.csv'
    write_table_columns(out1, rtab.to_columns())
    write_table_columns(out2, stab.to_columns())
    t1 = out1.read_text()
    t2 = out2.read_text()
    assert 'hole_id,frame_id,annulus_id,descriptor_value,lattice_u,lattice_v' in t1
    assert 'hole_id,frame_id,sector_id,annulus_id,descriptor_value,lattice_u,lattice_v' in t2
    x_vals, series = rtab.first_hole_series(1)
    assert x_vals == [0, 1]
    assert len(series) == 2
