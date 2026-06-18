from holecolor.pipeline import _write_sector_front_lag_map, _write_sector_acceleration_plot, _write_sector_propagation_plot
from holecolor.radial.columnar import sector_front_lag_columns, sector_hole_summary_plot_columns


def test_milestone48_sector_plot_helpers_accept_column_payloads(tmp_path):
    lag_cols = sector_front_lag_columns([
        {"hole_id": 1, "sector_id": 0, "sector_onset_lag": 0},
        {"hole_id": 1, "sector_id": 1, "sector_onset_lag": 2},
        {"hole_id": 2, "sector_id": 0, "sector_onset_lag": 1},
    ])
    acc_cols = sector_hole_summary_plot_columns([
        {"hole_id": 1, "mean_sector_front_acceleration_per_frame2": 0.12},
        {"hole_id": 2, "mean_sector_front_acceleration_per_frame2": 0.05},
    ], value_field="mean_sector_front_acceleration_per_frame2")
    prop_cols = sector_hole_summary_plot_columns([
        {"hole_id": 1, "sector_front_velocity_anisotropy": 0.25},
        {"hole_id": 2, "sector_front_velocity_anisotropy": 0.15},
    ], value_field="sector_front_velocity_anisotropy")
    _write_sector_front_lag_map(tmp_path, lag_cols)
    _write_sector_acceleration_plot(tmp_path, acc_cols)
    _write_sector_propagation_plot(tmp_path, prop_cols)
    assert (tmp_path / "sector_front_lag_map.png").exists()
    assert (tmp_path / "sector_front_acceleration.png").exists()
    assert (tmp_path / "sector_front_propagation.png").exists()
