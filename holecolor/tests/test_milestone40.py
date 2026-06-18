import numpy as np

from holecolor.plotting.prepare import count_by_label, heatmap_from_rows, line_series_from_rows


def test_milestone40_plot_helpers_match_expected_payloads():
    rows = [
        {"group": "A", "x": 0, "y": 1.0},
        {"group": "A", "x": 2, "y": 3.0},
        {"group": "B", "x": 1, "y": 4.0},
        {"group": "B", "x": 2, "y": None},
    ]
    x_vals, series = line_series_from_rows(rows, group_field="group", x_field="x", y_field="y")
    assert x_vals == [0, 1, 2]
    assert series[0][0] == "A"
    assert series[0][1][0] == 1.0 and np.isnan(series[0][1][1]) and series[0][1][2] == 3.0
    assert series[1][0] == "B"
    assert np.isnan(series[1][1][0]) and series[1][1][1] == 4.0

    labels, counts = count_by_label([
        {"label": "k2"},
        {"label": "k1"},
        {"label": "k2"},
    ], "label")
    assert labels == ["k1", "k2"]
    assert counts == [1, 2]

    hm_rows = [
        {"row": 1, "col": 0, "val": 2.0},
        {"row": 1, "col": 1, "val": None},
        {"row": 2, "col": 0, "val": 5.0},
    ]
    matrix, rows_u, cols_u = heatmap_from_rows(hm_rows, row_field="row", col_field="col", value_field="val")
    assert rows_u == [1, 2]
    assert cols_u == [0, 1]
    assert matrix.shape == (2, 2)
    assert matrix[0, 0] == 2.0
    assert np.isnan(matrix[0, 1])
    assert matrix[1, 0] == 5.0
