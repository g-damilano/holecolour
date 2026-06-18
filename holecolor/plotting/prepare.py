from __future__ import annotations

from typing import Any, Callable

import numpy as np


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _safe_float(value: Any, default: float = np.nan) -> float:
    try:
        return float(value) if value is not None else float(default)
    except Exception:
        return float(default)




def columns_from_records(rows: list[dict[str, Any]], fields: list[str] | None = None) -> dict[str, list[Any]]:
    if not rows:
        return {f: [] for f in (fields or [])}
    if fields is None:
        fields = sorted({str(k) for row in rows for k in row.keys()})
    return {str(field): [row.get(field) for row in rows] for field in fields}

def count_by_label(rows: list[dict[str, Any]], field: str) -> tuple[list[str], list[int]]:
    if not rows:
        return [], []
    labels = np.asarray([str(r.get(field, "unknown")) for r in rows], dtype="U128")
    uniq, counts = np.unique(labels, return_counts=True)
    return uniq.tolist(), counts.astype(int).tolist()


def line_series_from_rows(
    rows: list[dict[str, Any]],
    *,
    group_field: str,
    x_field: str,
    y_field: str,
    group_values: list[Any] | None = None,
    limit: int | None = None,
    group_labeler: Callable[[Any], str] | None = None,
) -> tuple[list[int], list[tuple[str, list[float]]]]:
    if not rows:
        return [], []
    g = np.asarray([str(r.get(group_field, "")) for r in rows], dtype="U128")
    x = np.asarray([_safe_int(r.get(x_field, 0)) for r in rows], dtype=np.int32)
    y = np.asarray([_safe_float(r.get(y_field, np.nan)) for r in rows], dtype=np.float32)
    x_vals = np.unique(x)
    x_vals.sort()
    x_index = {int(v): i for i, v in enumerate(x_vals.tolist())}
    if group_values is None:
        groups = np.unique(g).tolist()
        groups.sort()
    else:
        groups = [str(v) for v in group_values]
    if limit is not None:
        groups = groups[: int(limit)]
    series: list[tuple[str, list[float]]] = []
    for group in groups:
        mask = g == group
        if not np.any(mask):
            continue
        xs = x[mask]
        ys = y[mask]
        order = np.argsort(xs, kind="mergesort")
        xs = xs[order]
        ys = ys[order]
        vals = np.full((x_vals.size,), np.nan, dtype=np.float32)
        for xv, yv in zip(xs.tolist(), ys.tolist()):
            vals[x_index[int(xv)]] = yv
        label = group_labeler(group) if group_labeler is not None else str(group)
        series.append((label, vals.astype(float).tolist()))
    return x_vals.astype(int).tolist(), series


def heatmap_from_rows(
    rows: list[dict[str, Any]],
    *,
    row_field: str,
    col_field: str,
    value_field: str,
    filter_field: str | None = None,
    filter_value: Any | None = None,
) -> tuple[np.ndarray, list[int], list[int]]:
    if not rows:
        return np.zeros((0, 0), dtype=float), [], []
    if filter_field is not None:
        rows = [r for r in rows if r.get(filter_field) == filter_value]
        if not rows:
            return np.zeros((0, 0), dtype=float), [], []
    row_vals = np.asarray([_safe_int(r.get(row_field, 0)) for r in rows], dtype=np.int32)
    col_vals = np.asarray([_safe_int(r.get(col_field, 0)) for r in rows], dtype=np.int32)
    data_vals = np.asarray([_safe_float(r.get(value_field, np.nan)) for r in rows], dtype=np.float32)
    uniq_rows = np.unique(row_vals)
    uniq_cols = np.unique(col_vals)
    uniq_rows.sort()
    uniq_cols.sort()
    row_index = {int(v): i for i, v in enumerate(uniq_rows.tolist())}
    col_index = {int(v): i for i, v in enumerate(uniq_cols.tolist())}
    matrix = np.full((uniq_rows.size, uniq_cols.size), np.nan, dtype=np.float32)
    for r, c, v in zip(row_vals.tolist(), col_vals.tolist(), data_vals.tolist()):
        if np.isfinite(v):
            matrix[row_index[int(r)], col_index[int(c)]] = float(v)
    return matrix.astype(float), uniq_rows.astype(int).tolist(), uniq_cols.astype(int).tolist()



def line_series_from_columns(
    columns: dict[str, list[Any]],
    *,
    group_field: str,
    x_field: str,
    y_field: str,
    filter_field: str | None = None,
    filter_value: Any | None = None,
    group_values: list[Any] | None = None,
    limit: int | None = None,
    group_labeler: Callable[[Any], str] | None = None,
) -> tuple[list[int], list[tuple[str, list[float]]]]:
    if not columns or group_field not in columns or x_field not in columns or y_field not in columns:
        return [], []
    n = len(columns[group_field])
    if n == 0:
        return [], []
    g = np.asarray([str(v) for v in columns[group_field]], dtype="U128")
    x = np.asarray([_safe_int(v, 0) for v in columns[x_field]], dtype=np.int32)
    y = np.asarray([_safe_float(v, np.nan) for v in columns[y_field]], dtype=np.float32)
    if filter_field is not None and filter_field in columns:
        filt = np.asarray(columns[filter_field])
        mask = filt == filter_value
        if not np.any(mask):
            return [], []
        g = g[mask]
        x = x[mask]
        y = y[mask]
    x_vals = np.unique(x)
    x_vals.sort()
    x_index = {int(v): i for i, v in enumerate(x_vals.tolist())}
    if group_values is None:
        groups = np.unique(g).tolist()
        groups.sort()
    else:
        groups = [str(v) for v in group_values]
    if limit is not None:
        groups = groups[: int(limit)]
    series: list[tuple[str, list[float]]] = []
    for group in groups:
        mask = g == group
        if not np.any(mask):
            continue
        xs = x[mask]
        ys = y[mask]
        order = np.argsort(xs, kind="mergesort")
        xs = xs[order]
        ys = ys[order]
        vals = np.full((x_vals.size,), np.nan, dtype=np.float32)
        for xv, yv in zip(xs.tolist(), ys.tolist()):
            vals[x_index[int(xv)]] = yv
        label = group_labeler(group) if group_labeler is not None else str(group)
        series.append((label, vals.astype(float).tolist()))
    return x_vals.astype(int).tolist(), series


def heatmap_from_columns(
    columns: dict[str, list[Any]],
    *,
    row_field: str,
    col_field: str,
    value_field: str,
    filter_field: str | None = None,
    filter_value: Any | None = None,
) -> tuple[np.ndarray, list[int], list[int]]:
    if not columns or row_field not in columns or col_field not in columns or value_field not in columns:
        return np.zeros((0, 0), dtype=float), [], []
    n = len(columns[row_field])
    if n == 0:
        return np.zeros((0, 0), dtype=float), [], []
    row_vals = np.asarray([_safe_int(v, 0) for v in columns[row_field]], dtype=np.int32)
    col_vals = np.asarray([_safe_int(v, 0) for v in columns[col_field]], dtype=np.int32)
    data_vals = np.asarray([_safe_float(v, np.nan) for v in columns[value_field]], dtype=np.float32)
    if filter_field is not None and filter_field in columns:
        filt = np.asarray(columns[filter_field])
        mask = filt == filter_value
        if not np.any(mask):
            return np.zeros((0, 0), dtype=float), [], []
        row_vals = row_vals[mask]
        col_vals = col_vals[mask]
        data_vals = data_vals[mask]
    uniq_rows = np.unique(row_vals)
    uniq_cols = np.unique(col_vals)
    uniq_rows.sort(); uniq_cols.sort()
    row_index = {int(v): i for i, v in enumerate(uniq_rows.tolist())}
    col_index = {int(v): i for i, v in enumerate(uniq_cols.tolist())}
    matrix = np.full((uniq_rows.size, uniq_cols.size), np.nan, dtype=np.float32)
    for r, c, v in zip(row_vals.tolist(), col_vals.tolist(), data_vals.tolist()):
        if np.isfinite(v):
            matrix[row_index[int(r)], col_index[int(c)]] = float(v)
    return matrix.astype(float), uniq_rows.astype(int).tolist(), uniq_cols.astype(int).tolist()


def bar_from_columns(columns: dict[str, list[Any]], *, label_field: str, value_field: str, limit: int | None = None) -> tuple[list[str], list[float]]:
    if not columns or label_field not in columns or value_field not in columns:
        return [], []
    labels = [str(v) for v in columns[label_field]]
    vals = [float(v or 0.0) if v is not None else 0.0 for v in columns[value_field]]
    if limit is not None:
        labels = labels[:int(limit)]
        vals = vals[:int(limit)]
    return labels, vals
