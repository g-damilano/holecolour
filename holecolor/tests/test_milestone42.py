from __future__ import annotations

from pathlib import Path

from holecolor.plotting.prepare import bar_from_columns, heatmap_from_columns
from holecolor.qc.reports import write_table_columns
from holecolor.radial.columnar import RadialRowTable, build_per_hole_rdf_evolution_columns


def test_milestone42_rdf_columns_drive_writers_and_plots(tmp_path: Path) -> None:
    radial_rows = [
        {"hole_id": 1, "frame_id": 0, "annulus_id": 0, "descriptor_value": 0.1, "lattice_u": 0, "lattice_v": 0},
        {"hole_id": 1, "frame_id": 0, "annulus_id": 1, "descriptor_value": 0.2, "lattice_u": 0, "lattice_v": 0},
        {"hole_id": 1, "frame_id": 1, "annulus_id": 0, "descriptor_value": 0.4, "lattice_u": 0, "lattice_v": 0},
        {"hole_id": 1, "frame_id": 1, "annulus_id": 1, "descriptor_value": 0.8, "lattice_u": 0, "lattice_v": 0},
    ]
    table = RadialRowTable.from_rows(radial_rows)
    cols = build_per_hole_rdf_evolution_columns(table)
    evo_path = tmp_path / 'rdf_evolution.csv'
    write_table_columns(evo_path, cols.evolution)
    text = evo_path.read_text()
    assert 'hole_id,frame_id,annulus_id,normalized_radius' in text
    matrix, annuli, frames = heatmap_from_columns(cols.evolution, row_field='annulus_id', col_field='frame_id', value_field='rdf_pdf', filter_field='hole_id', filter_value=1)
    assert matrix.shape == (2, 2)
    assert annuli == [0, 1]
    assert frames == [0, 1]
    labels, vals = bar_from_columns(cols.velocity_summary, label_field='hole_id', value_field='rdf_front_velocity_per_frame', limit=20)
    assert labels == ['1']
    assert len(vals) == 1
    frame_rows = cols.frame_rows()
    assert len(frame_rows) == 2
    assert frame_rows[0]['hole_id'] == 1
