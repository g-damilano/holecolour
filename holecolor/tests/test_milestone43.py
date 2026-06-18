from __future__ import annotations

from pathlib import Path

from holecolor.plotting.prepare import line_series_from_columns
from holecolor.qc.reports import write_table_columns
from holecolor.radial.columnar import SectorRadialTable, build_sector_rdf_evolution_columns


def test_milestone43_sector_rdf_columns_drive_writers_and_plots(tmp_path: Path) -> None:
    sector_rows = [
        {"hole_id": 1, "frame_id": 0, "sector_id": 0, "annulus_id": 0, "descriptor_value": 0.1, "lattice_u": 0, "lattice_v": 0},
        {"hole_id": 1, "frame_id": 0, "sector_id": 0, "annulus_id": 1, "descriptor_value": 0.2, "lattice_u": 0, "lattice_v": 0},
        {"hole_id": 1, "frame_id": 1, "sector_id": 0, "annulus_id": 0, "descriptor_value": 0.3, "lattice_u": 0, "lattice_v": 0},
        {"hole_id": 1, "frame_id": 1, "sector_id": 0, "annulus_id": 1, "descriptor_value": 0.8, "lattice_u": 0, "lattice_v": 0},
        {"hole_id": 1, "frame_id": 0, "sector_id": 1, "annulus_id": 0, "descriptor_value": 0.05, "lattice_u": 0, "lattice_v": 0},
        {"hole_id": 1, "frame_id": 0, "sector_id": 1, "annulus_id": 1, "descriptor_value": 0.10, "lattice_u": 0, "lattice_v": 0},
        {"hole_id": 1, "frame_id": 1, "sector_id": 1, "annulus_id": 0, "descriptor_value": 0.2, "lattice_u": 0, "lattice_v": 0},
        {"hole_id": 1, "frame_id": 1, "sector_id": 1, "annulus_id": 1, "descriptor_value": 0.3, "lattice_u": 0, "lattice_v": 0},
    ]
    table = SectorRadialTable.from_rows(sector_rows)
    cols = build_sector_rdf_evolution_columns(table)
    evo_path = tmp_path / 'sector_rdf_evolution.csv'
    frame_path = tmp_path / 'sector_rdf_frame_summary.csv'
    write_table_columns(evo_path, cols.evolution)
    write_table_columns(frame_path, cols.frame_summary)
    assert 'hole_id,frame_id,sector_id,annulus_id' in evo_path.read_text()
    assert 'hole_id,frame_id,sector_id,sector_rdf_front_radius_norm' in frame_path.read_text()
    x, series = line_series_from_columns(
        cols.frame_summary,
        group_field='sector_id',
        x_field='frame_id',
        y_field='sector_rdf_front_radius_norm',
        filter_field='hole_id',
        filter_value=1,
        group_values=[0, 1],
        group_labeler=lambda g: f'sector_{g}',
    )
    assert x == [0, 1]
    assert [label for label, _ in series] == ['sector_0', 'sector_1']
    frame_rows = cols.frame_rows()
    assert len(frame_rows) == 4
    assert {r['sector_id'] for r in frame_rows} == {0, 1}
