from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np


def _safe_float(value: Any, default: float = np.nan) -> float:
    try:
        return float(value) if value is not None else float(default)
    except Exception:
        return float(default)


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)




def _acc_init(fields: tuple[str, ...]) -> dict[str, Any]:
    return {"n": 0, **{f"sum__{f}": 0.0 for f in fields}, **{f"count__{f}": 0 for f in fields}}


def _acc_add(acc: dict[str, Any], field: str, value: Any) -> None:
    v = _safe_float(value)
    if np.isfinite(v):
        acc[f"sum__{field}"] += float(v)
        acc[f"count__{field}"] += 1


def _acc_mean(acc: dict[str, Any], field: str) -> float | None:
    n = int(acc.get(f"count__{field}", 0))
    if n <= 0:
        return None
    return float(acc[f"sum__{field}"] / n)

def _group_bounds(*arrays: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if not arrays or arrays[0].size == 0:
        return np.zeros((0,), dtype=int), np.zeros((0,), dtype=int)
    n = int(arrays[0].size)
    change = np.ones((n,), dtype=bool)
    if n > 1:
        same = np.ones((n - 1,), dtype=bool)
        for arr in arrays:
            same &= arr[1:] == arr[:-1]
        change[1:] = ~same
    starts = np.flatnonzero(change)
    ends = np.r_[starts[1:], n]
    return starts.astype(int), ends.astype(int)


@dataclass(slots=True)
class RadialRowTable:
    hole_id: np.ndarray
    frame_id: np.ndarray
    annulus_id: np.ndarray
    descriptor_value: np.ndarray
    lattice_u: np.ndarray
    lattice_v: np.ndarray

    order_hfa: np.ndarray
    order_haf: np.ndarray
    order_hf_starts: np.ndarray
    order_hf_ends: np.ndarray

    @classmethod
    def from_rows(cls, rows: list[dict[str, Any]]) -> 'RadialRowTable':
        n = len(rows)
        hole_id = np.empty((n,), dtype=np.int32)
        frame_id = np.empty((n,), dtype=np.int32)
        annulus_id = np.empty((n,), dtype=np.int16)
        descriptor_value = np.empty((n,), dtype=np.float32)
        lattice_u = np.empty((n,), dtype=np.float32)
        lattice_v = np.empty((n,), dtype=np.float32)
        lattice_u.fill(np.nan)
        lattice_v.fill(np.nan)
        for i, row in enumerate(rows):
            hole_id[i] = _safe_int(row.get('hole_id', 0))
            frame_id[i] = _safe_int(row.get('frame_id', 0))
            annulus_id[i] = _safe_int(row.get('annulus_id', 0))
            descriptor_value[i] = np.float32(_safe_float(row.get('descriptor_value', np.nan)))
            lattice_u[i] = np.float32(_safe_float(row.get('lattice_u', np.nan)))
            lattice_v[i] = np.float32(_safe_float(row.get('lattice_v', np.nan)))
        order_hfa = np.lexsort((annulus_id, frame_id, hole_id)).astype(np.int32)
        order_haf = np.lexsort((frame_id, annulus_id, hole_id)).astype(np.int32)
        hid_hfa = hole_id[order_hfa]
        fid_hfa = frame_id[order_hfa]
        starts, ends = _group_bounds(hid_hfa, fid_hfa)
        return cls(
            hole_id=hole_id,
            frame_id=frame_id,
            annulus_id=annulus_id,
            descriptor_value=descriptor_value,
            lattice_u=lattice_u,
            lattice_v=lattice_v,
            order_hfa=order_hfa,
            order_haf=order_haf,
            order_hf_starts=starts,
            order_hf_ends=ends,
        )

    def uv_meta(self, idx: int) -> tuple[Any, Any]:
        u = self.lattice_u[idx]
        v = self.lattice_v[idx]
        out_u = None if not np.isfinite(u) else int(round(float(u)))
        out_v = None if not np.isfinite(v) else int(round(float(v)))
        return out_u, out_v

    def first_hole_series(self, hole_id: int) -> tuple[list[int], list[tuple[str, list[float]]]]:
        mask = self.hole_id == int(hole_id)
        if not np.any(mask):
            return [], []
        frames = self.frame_id[mask]
        ann = self.annulus_id[mask]
        vals = self.descriptor_value[mask].astype(float, copy=False)
        order = np.lexsort((ann, frames))
        frames = frames[order]
        ann = ann[order]
        vals = vals[order]
        x_vals = sorted({int(v) for v in frames.tolist()})
        series: list[tuple[str, list[float]]] = []
        for annulus in sorted({int(v) for v in ann.tolist()}):
            amask = ann == annulus
            a_frames = frames[amask]
            a_vals = vals[amask]
            lookup = {int(f): (None if not np.isfinite(v) else float(v)) for f, v in zip(a_frames.tolist(), a_vals.tolist())}
            series.append((f'annulus_{annulus}', [lookup.get(fid) for fid in x_vals]))
        return x_vals, series

    def iter_hole_frame_groups(self) -> Iterable[tuple[int, int, np.ndarray]]:
        order = self.order_hfa
        hole = self.hole_id[order]
        frame = self.frame_id[order]
        for s, e in zip(self.order_hf_starts, self.order_hf_ends):
            idx = order[s:e]
            yield int(hole[s]), int(frame[s]), idx

    def baseline_by_hole_annulus(self) -> dict[tuple[int, int], float]:
        order = self.order_haf
        hole = self.hole_id[order]
        ann = self.annulus_id[order]
        vals = self.descriptor_value[order]
        starts, _ends = _group_bounds(hole, ann)
        out: dict[tuple[int, int], float] = {}
        for s in starts:
            out[(int(hole[s]), int(ann[s]))] = float(vals[s]) if np.isfinite(vals[s]) else np.nan
        return out

    def baseline_frame_by_hole(self) -> dict[int, int]:
        order = self.order_haf
        hole = self.hole_id[order]
        frame = self.frame_id[order]
        starts, _ends = _group_bounds(hole)
        out: dict[int, int] = {}
        for s in starts:
            out[int(hole[s])] = int(frame[s])
        return out

    def to_columns(self) -> dict[str, list[Any]]:
        return {
            'hole_id': self.hole_id.astype(int).tolist(),
            'frame_id': self.frame_id.astype(int).tolist(),
            'annulus_id': self.annulus_id.astype(int).tolist(),
            'descriptor_value': self.descriptor_value.astype(float).tolist(),
            'lattice_u': [None if not np.isfinite(v) else int(round(float(v))) for v in self.lattice_u],
            'lattice_v': [None if not np.isfinite(v) else int(round(float(v))) for v in self.lattice_v],
        }


def per_hole_radial_frame_summary_table(table: RadialRowTable) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for hole_id, frame_id, idx in table.iter_hole_frame_groups():
        ann = table.annulus_id[idx].astype(float, copy=False)
        vals = table.descriptor_value[idx].astype(float, copy=False)
        finite = np.isfinite(vals)
        if np.any(finite):
            ann = ann[finite]
            vals = vals[finite]
            shifted = vals - float(np.nanmin(vals)) + 1e-6
            denom = float(np.sum(shifted))
            com = float(np.sum(ann * shifted) / denom) if denom > 0 else float(np.nanmean(ann))
            peak_annulus = int(ann[int(np.nanargmax(vals))]) if vals.size else None
            inner_minus_outer = float(vals[0] - vals[-1]) if vals.size >= 2 else None
            mean_descriptor = float(np.nanmean(vals)) if vals.size else None
            std_descriptor = float(np.nanstd(vals)) if vals.size else None
            n_valid = int(vals.size)
        else:
            com = None
            peak_annulus = None
            inner_minus_outer = None
            mean_descriptor = None
            std_descriptor = None
            n_valid = 0
        uv = table.uv_meta(int(idx[0]))
        out.append({
            'hole_id': int(hole_id),
            'frame_id': int(frame_id),
            'lattice_u': uv[0],
            'lattice_v': uv[1],
            'center_of_mass_annulus': com,
            'peak_annulus': peak_annulus,
            'inner_minus_outer': inner_minus_outer,
            'mean_descriptor': mean_descriptor,
            'std_descriptor': std_descriptor,
            'n_valid_annuli': n_valid,
        })
    return out




@dataclass(slots=True)
class RdfEvolutionColumns:
    evolution: dict[str, list[Any]]
    frame_summary: dict[str, list[Any]]
    velocity_summary: dict[str, list[Any]]

    def frame_rows(self) -> list[dict[str, Any]]:
        cols = self.frame_summary
        keys = list(cols.keys())
        n = len(cols[keys[0]]) if keys else 0
        return [{k: cols[k][i] for k in keys} for i in range(n)]


def build_per_hole_rdf_evolution_columns(table: RadialRowTable) -> RdfEvolutionColumns:
    baseline = table.baseline_by_hole_annulus()
    baseline_frame = table.baseline_frame_by_hole()
    evo = {
        'hole_id': [], 'frame_id': [], 'annulus_id': [], 'normalized_radius': [],
        'descriptor_value': [], 'baseline_descriptor_value': [], 'delta_descriptor_value': [],
        'positive_delta_descriptor': [], 'rdf_pdf': [], 'rdf_cdf': [],
    }
    frame = {
        'hole_id': [], 'frame_id': [], 'baseline_frame_id': [], 'rdf_front_radius_norm': [],
        'rdf_spread_norm': [], 'rdf_total_positive_delta': [], 'rdf_peak_annulus': [],
        'rdf_inner_delta': [], 'rdf_outer_delta': [], 'rdf_inner_minus_outer_delta': [],
    }
    vel = {
        'hole_id': [], 'rdf_front_velocity_per_frame': [], 'rdf_front_intercept': [],
        'rdf_front_velocity_r2': [], 'start_front_radius_norm': [], 'end_front_radius_norm': [],
        'delta_front_radius_norm': [], 'n_frames': [],
    }
    front_by_hole: dict[int, list[tuple[int, float]]] = {}
    for hole_id, frame_id, idx in table.iter_hole_frame_groups():
        ann = table.annulus_id[idx].astype(float, copy=False)
        vals = table.descriptor_value[idx].astype(float, copy=False)
        baselines = np.asarray([baseline.get((hole_id, int(a)), np.nan) for a in ann], dtype=float)
        delta = vals - baselines
        positive = np.clip(delta, 0.0, None)
        total = float(np.nansum(positive))
        max_ann = max(float(np.nanmax(ann)) if ann.size else 0.0, 1.0)
        norm_r = ann / max_ann if ann.size else ann
        pdf = positive / total if total > 0 else np.full_like(positive, np.nan, dtype=float)
        cdf = np.cumsum(np.nan_to_num(pdf, nan=0.0)) if pdf.size else pdf
        if total > 0 and np.isfinite(pdf).any():
            front_r = float(np.nansum(norm_r * pdf))
            second_m = float(np.nansum(((norm_r - front_r) ** 2) * pdf))
            spread = float(np.sqrt(max(second_m, 0.0)))
            peak_annulus = int(ann[int(np.nanargmax(pdf))])
        else:
            front_r = np.nan
            spread = np.nan
            peak_annulus = None
        inner_delta = float(delta[0]) if delta.size else np.nan
        outer_delta = float(delta[-1]) if delta.size else np.nan
        frame['hole_id'].append(int(hole_id))
        frame['frame_id'].append(int(frame_id))
        frame['baseline_frame_id'].append(int(baseline_frame.get(hole_id, frame_id)))
        frame['rdf_front_radius_norm'].append(None if not np.isfinite(front_r) else float(front_r))
        frame['rdf_spread_norm'].append(None if not np.isfinite(spread) else float(spread))
        frame['rdf_total_positive_delta'].append(total)
        frame['rdf_peak_annulus'].append(peak_annulus)
        frame['rdf_inner_delta'].append(None if not np.isfinite(inner_delta) else float(inner_delta))
        frame['rdf_outer_delta'].append(None if not np.isfinite(outer_delta) else float(outer_delta))
        frame['rdf_inner_minus_outer_delta'].append(None if not (np.isfinite(inner_delta) and np.isfinite(outer_delta)) else float(inner_delta - outer_delta))
        if np.isfinite(front_r):
            front_by_hole.setdefault(int(hole_id), []).append((int(frame_id), float(front_r)))
        for i, ridx in enumerate(idx):
            evo['hole_id'].append(int(hole_id))
            evo['frame_id'].append(int(frame_id))
            evo['annulus_id'].append(int(table.annulus_id[ridx]))
            evo['normalized_radius'].append(None if not (norm_r.size and np.isfinite(norm_r[i])) else float(norm_r[i]))
            evo['descriptor_value'].append(None if not np.isfinite(vals[i]) else float(vals[i]))
            evo['baseline_descriptor_value'].append(None if not np.isfinite(baselines[i]) else float(baselines[i]))
            evo['delta_descriptor_value'].append(None if not np.isfinite(delta[i]) else float(delta[i]))
            evo['positive_delta_descriptor'].append(None if not np.isfinite(positive[i]) else float(positive[i]))
            evo['rdf_pdf'].append(None if not (pdf.size and np.isfinite(pdf[i])) else float(pdf[i]))
            evo['rdf_cdf'].append(None if not (cdf.size and np.isfinite(cdf[i])) else float(cdf[i]))
    for hole_id, series in sorted(front_by_hole.items()):
        series = sorted(series)
        frames = np.asarray([s[0] for s in series], dtype=float)
        radii = np.asarray([s[1] for s in series], dtype=float)
        if radii.size >= 2 and np.ptp(frames) > 0:
            slope, intercept = np.polyfit(frames, radii, 1)
            pred = slope * frames + intercept
            ss_res = float(np.sum((radii - pred) ** 2))
            ss_tot = float(np.sum((radii - np.mean(radii)) ** 2))
            r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
        elif radii.size == 1:
            slope, intercept, r2 = 0.0, float(radii[0]), 1.0
        else:
            slope, intercept, r2 = np.nan, np.nan, np.nan
        vel['hole_id'].append(int(hole_id))
        vel['rdf_front_velocity_per_frame'].append(None if not np.isfinite(slope) else float(slope))
        vel['rdf_front_intercept'].append(None if not np.isfinite(intercept) else float(intercept))
        vel['rdf_front_velocity_r2'].append(None if not np.isfinite(r2) else float(r2))
        vel['start_front_radius_norm'].append(None if radii.size == 0 else float(radii[0]))
        vel['end_front_radius_norm'].append(None if radii.size == 0 else float(radii[-1]))
        vel['delta_front_radius_norm'].append(None if radii.size == 0 else float(radii[-1] - radii[0]))
        vel['n_frames'].append(int(radii.size))
    return RdfEvolutionColumns(evolution=evo, frame_summary=frame, velocity_summary=vel)


def build_per_hole_rdf_evolution_table(table: RadialRowTable) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    cols = build_per_hole_rdf_evolution_columns(table)
    evo_keys = list(cols.evolution.keys())
    vel_keys = list(cols.velocity_summary.keys())
    evolution_rows = [{k: cols.evolution[k][i] for k in evo_keys} for i in range(len(cols.evolution[evo_keys[0]]) if evo_keys else 0)]
    velocity_rows = [{k: cols.velocity_summary[k][i] for k in vel_keys} for i in range(len(cols.velocity_summary[vel_keys[0]]) if vel_keys else 0)]
    return evolution_rows, cols.frame_rows(), velocity_rows


def fit_radial_models_table(table: RadialRowTable) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    fit_rows: list[dict[str, Any]] = []
    hole_acc: dict[int, dict[str, Any]] = {}
    for hole_id, frame_id, idx in table.iter_hole_frame_groups():
        x = table.annulus_id[idx].astype(float, copy=False)
        y = table.descriptor_value[idx].astype(float, copy=False)
        valid = np.isfinite(x) & np.isfinite(y)
        if int(valid.sum()) < 2:
            continue
        x = x[valid]
        y = y[valid]
        lin = np.polyfit(x, y, deg=1)
        y_lin = np.polyval(lin, x)
        ss_res = float(np.sum((y - y_lin) ** 2))
        ss_tot = float(np.sum((y - np.mean(y)) ** 2))
        lin_r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 1.0
        quad_a = quad_b = quad_c = quad_r2 = front_radius = None
        if x.size >= 3:
            quad = np.polyfit(x, y, deg=2)
            quad_a, quad_b, quad_c = [float(v) for v in quad]
            y_quad = np.polyval(quad, x)
            qres = float(np.sum((y - y_quad) ** 2))
            quad_r2 = 1.0 - qres / ss_tot if ss_tot > 1e-12 else 1.0
            if abs(quad_a) > 1e-8:
                fr = -quad_b / (2.0 * quad_a)
                if np.isfinite(fr):
                    front_radius = float(fr)
        uv = table.uv_meta(int(idx[0]))
        row = {
            'hole_id': int(hole_id),
            'frame_id': int(frame_id),
            'lattice_u': uv[0],
            'lattice_v': uv[1],
            'linear_slope': float(lin[0]),
            'linear_intercept': float(lin[1]),
            'linear_r2': float(lin_r2),
            'quadratic_a': quad_a,
            'quadratic_b': quad_b,
            'quadratic_c': quad_c,
            'quadratic_r2': quad_r2,
            'front_radius_quadratic': front_radius,
            'n_valid_annuli': int(x.size),
        }
        fit_rows.append(row)
        acc = hole_acc.setdefault(int(hole_id), {
            'lattice_u': uv[0], 'lattice_v': uv[1], 'n': 0,
            'sum_linear_slope': 0.0, 'count_linear_slope': 0,
            'sum_linear_r2': 0.0, 'count_linear_r2': 0,
            'sum_quad_r2': 0.0, 'count_quad_r2': 0,
            'sum_front_r': 0.0, 'count_front_r': 0,
            'min_front_r': np.inf, 'max_front_r': -np.inf,
        })
        acc['n'] += 1
        acc['sum_linear_slope'] += float(lin[0]); acc['count_linear_slope'] += 1
        acc['sum_linear_r2'] += float(lin_r2); acc['count_linear_r2'] += 1
        if quad_r2 is not None and np.isfinite(quad_r2):
            acc['sum_quad_r2'] += float(quad_r2); acc['count_quad_r2'] += 1
        if front_radius is not None and np.isfinite(front_radius):
            fr = float(front_radius)
            acc['sum_front_r'] += fr; acc['count_front_r'] += 1
            acc['min_front_r'] = min(float(acc['min_front_r']), fr)
            acc['max_front_r'] = max(float(acc['max_front_r']), fr)
    hole_summary: list[dict[str, Any]] = []
    for hole_id, acc in sorted(hole_acc.items()):
        cfr = int(acc['count_front_r'])
        hole_summary.append({
            'hole_id': int(hole_id),
            'lattice_u': acc['lattice_u'],
            'lattice_v': acc['lattice_v'],
            'mean_linear_slope': float(acc['sum_linear_slope'] / max(int(acc['count_linear_slope']), 1)),
            'mean_linear_r2': float(acc['sum_linear_r2'] / max(int(acc['count_linear_r2']), 1)),
            'mean_quadratic_r2': None if int(acc['count_quad_r2']) <= 0 else float(acc['sum_quad_r2'] / int(acc['count_quad_r2'])),
            'mean_front_radius_quadratic': None if cfr <= 0 else float(acc['sum_front_r'] / cfr),
            'front_radius_range': None if cfr <= 0 else float(acc['max_front_r'] - acc['min_front_r']),
            'n_frames': int(acc['n']),
        })
    return fit_rows, hole_summary


def build_reticulum_group_rows_table(table: RadialRowTable, zone_by_hole: dict[int, str]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if table.hole_id.size == 0:
        return [], []
    zones = np.asarray([str(zone_by_hole.get(int(h), 'all')) for h in table.hole_id], dtype=object)
    keys = sorted({str(z) for z in zones.tolist()})
    zone_code_map = {z: i for i, z in enumerate(keys)}
    zone_codes = np.asarray([zone_code_map[str(z)] for z in zones], dtype=np.int16)
    order = np.lexsort((table.annulus_id, table.frame_id, zone_codes))
    z = zone_codes[order]
    f = table.frame_id[order]
    a = table.annulus_id[order]
    v = table.descriptor_value[order].astype(float, copy=False)
    starts, ends = _group_bounds(z, f, a)
    group_rows: list[dict[str, Any]] = []
    for s, e in zip(starts, ends):
        arr = v[s:e]
        arr = arr[np.isfinite(arr)]
        group_rows.append({
            'reticulum_zone': keys[int(z[s])],
            'frame_id': int(f[s]),
            'annulus_id': int(a[s]),
            'mean_descriptor': float(arr.mean()) if arr.size else None,
            'std_descriptor': float(arr.std()) if arr.size else None,
            'n_holes': int(arr.size),
        })
    summary_rows: list[dict[str, Any]] = []
    ordered_group = sorted(group_rows, key=lambda r: (str(r['reticulum_zone']), int(r['frame_id']), int(r['annulus_id'])))
    from itertools import groupby
    for zone, zgrp in groupby(ordered_group, key=lambda r: str(r['reticulum_zone'])):
        for frame_id, fgrp in groupby(list(zgrp), key=lambda r: int(r['frame_id'])):
            frows = list(fgrp)
            ann = np.asarray([int(r['annulus_id']) for r in frows], dtype=float)
            vals = np.asarray([float(r['mean_descriptor']) if r.get('mean_descriptor') is not None else np.nan for r in frows], dtype=float)
            finite = np.isfinite(vals)
            if not np.any(finite):
                continue
            ann = ann[finite]
            vals = vals[finite]
            shifted = vals - float(np.nanmin(vals)) + 1e-6
            denom = float(np.sum(shifted))
            com = float(np.sum(ann * shifted) / denom) if denom > 0 else float(np.nanmean(ann))
            summary_rows.append({'reticulum_zone': zone, 'frame_id': int(frame_id), 'center_of_mass_annulus': com})
    return group_rows, summary_rows


@dataclass(slots=True)
class SectorRadialTable:
    hole_id: np.ndarray
    frame_id: np.ndarray
    sector_id: np.ndarray
    annulus_id: np.ndarray
    descriptor_value: np.ndarray
    lattice_u: np.ndarray
    lattice_v: np.ndarray

    order_hfsa: np.ndarray
    order_hsfa: np.ndarray
    order_hf_starts: np.ndarray
    order_hf_ends: np.ndarray
    order_hfs_starts: np.ndarray
    order_hfs_ends: np.ndarray
    order_hs_starts: np.ndarray
    order_hs_ends: np.ndarray

    @classmethod
    def from_rows(cls, rows: list[dict[str, Any]]) -> 'SectorRadialTable':
        n = len(rows)
        hole_id = np.empty((n,), dtype=np.int32)
        frame_id = np.empty((n,), dtype=np.int32)
        sector_id = np.empty((n,), dtype=np.int16)
        annulus_id = np.empty((n,), dtype=np.int16)
        descriptor_value = np.empty((n,), dtype=np.float32)
        lattice_u = np.empty((n,), dtype=np.float32)
        lattice_v = np.empty((n,), dtype=np.float32)
        lattice_u.fill(np.nan)
        lattice_v.fill(np.nan)
        for i, row in enumerate(rows):
            hole_id[i] = _safe_int(row.get('hole_id', 0))
            frame_id[i] = _safe_int(row.get('frame_id', 0))
            sector_id[i] = _safe_int(row.get('sector_id', 0))
            annulus_id[i] = _safe_int(row.get('annulus_id', 0))
            descriptor_value[i] = np.float32(_safe_float(row.get('descriptor_value', np.nan)))
            lattice_u[i] = np.float32(_safe_float(row.get('lattice_u', np.nan)))
            lattice_v[i] = np.float32(_safe_float(row.get('lattice_v', np.nan)))
        order_hfsa = np.lexsort((annulus_id, sector_id, frame_id, hole_id)).astype(np.int32)
        order_hsfa = np.lexsort((annulus_id, frame_id, sector_id, hole_id)).astype(np.int32)
        hid_hf = hole_id[order_hfsa]
        fid_hf = frame_id[order_hfsa]
        sid_hf = sector_id[order_hfsa]
        hf_starts, hf_ends = _group_bounds(hid_hf, fid_hf)
        hfs_starts, hfs_ends = _group_bounds(hid_hf, fid_hf, sid_hf)
        hid_hs = hole_id[order_hsfa]
        sid_hs = sector_id[order_hsfa]
        hs_starts, hs_ends = _group_bounds(hid_hs, sid_hs)
        return cls(
            hole_id=hole_id,
            frame_id=frame_id,
            sector_id=sector_id,
            annulus_id=annulus_id,
            descriptor_value=descriptor_value,
            lattice_u=lattice_u,
            lattice_v=lattice_v,
            order_hfsa=order_hfsa,
            order_hsfa=order_hsfa,
            order_hf_starts=hf_starts,
            order_hf_ends=hf_ends,
            order_hfs_starts=hfs_starts,
            order_hfs_ends=hfs_ends,
            order_hs_starts=hs_starts,
            order_hs_ends=hs_ends,
        )

    def uv_meta(self, idx: int) -> tuple[Any, Any]:
        u = self.lattice_u[idx]
        v = self.lattice_v[idx]
        out_u = None if not np.isfinite(u) else int(round(float(u)))
        out_v = None if not np.isfinite(v) else int(round(float(v)))
        return out_u, out_v

    def to_columns(self) -> dict[str, list[Any]]:
        return {
            "hole_id": self.hole_id.astype(int).tolist(),
            "frame_id": self.frame_id.astype(int).tolist(),
            "sector_id": self.sector_id.astype(int).tolist(),
            "annulus_id": self.annulus_id.astype(int).tolist(),
            "descriptor_value": self.descriptor_value.astype(float).tolist(),
            "lattice_u": [None if not np.isfinite(v) else int(round(float(v))) for v in self.lattice_u],
            "lattice_v": [None if not np.isfinite(v) else int(round(float(v))) for v in self.lattice_v],
        }

    def first_hole_series(self, hole_id: int) -> tuple[list[int], list[tuple[str, list[float]]]]:
        mask = self.hole_id == int(hole_id)
        if not np.any(mask):
            return [], []
        frames = self.frame_id[mask]
        ann = self.annulus_id[mask]
        vals = self.descriptor_value[mask].astype(float, copy=False)
        order = np.lexsort((ann, frames))
        frames = frames[order]
        ann = ann[order]
        vals = vals[order]
        x_vals = sorted({int(v) for v in frames.tolist()})
        series: list[tuple[str, list[float]]] = []
        for annulus in sorted({int(v) for v in ann.tolist()}):
            amask = ann == annulus
            a_frames = frames[amask]
            a_vals = vals[amask]
            lookup = {int(f): (None if not np.isfinite(v) else float(v)) for f, v in zip(a_frames.tolist(), a_vals.tolist())}
            series.append((f'annulus_{annulus}', [lookup.get(fid) for fid in x_vals]))
        return x_vals, series

    def iter_hole_frame_sector_groups(self):
        order = self.order_hfsa
        hole = self.hole_id[order]
        frame = self.frame_id[order]
        sector = self.sector_id[order]
        for s, e in zip(self.order_hfs_starts, self.order_hfs_ends):
            idx = order[s:e]
            yield int(hole[s]), int(frame[s]), int(sector[s]), idx

    def iter_hole_frame_groups(self):
        order = self.order_hfsa
        hole = self.hole_id[order]
        frame = self.frame_id[order]
        for s, e in zip(self.order_hf_starts, self.order_hf_ends):
            idx = order[s:e]
            yield int(hole[s]), int(frame[s]), idx

    def iter_hole_sector_groups(self):
        order = self.order_hsfa
        hole = self.hole_id[order]
        sector = self.sector_id[order]
        for s, e in zip(self.order_hs_starts, self.order_hs_ends):
            idx = order[s:e]
            yield int(hole[s]), int(sector[s]), idx

    def baseline_by_hole_sector_annulus(self) -> dict[tuple[int, int, int], float]:
        order = np.lexsort((self.frame_id, self.annulus_id, self.sector_id, self.hole_id)).astype(np.int32)
        hole = self.hole_id[order]
        sector = self.sector_id[order]
        ann = self.annulus_id[order]
        vals = self.descriptor_value[order]
        starts, _ = _group_bounds(hole, sector, ann)
        out: dict[tuple[int, int, int], float] = {}
        for s in starts:
            out[(int(hole[s]), int(sector[s]), int(ann[s]))] = float(vals[s]) if np.isfinite(vals[s]) else np.nan
        return out


@dataclass(slots=True)
class SectorRdfFrameTable:
    hole_id: np.ndarray
    frame_id: np.ndarray
    sector_id: np.ndarray
    front_radius: np.ndarray
    total_positive_delta: np.ndarray
    lattice_u: np.ndarray
    lattice_v: np.ndarray

    order_hs: np.ndarray
    hs_starts: np.ndarray
    hs_ends: np.ndarray

    @classmethod
    def from_rows(cls, rows: list[dict[str, Any]]) -> 'SectorRdfFrameTable':
        n = len(rows)
        hole_id = np.empty((n,), dtype=np.int32)
        frame_id = np.empty((n,), dtype=np.int32)
        sector_id = np.empty((n,), dtype=np.int16)
        front_radius = np.empty((n,), dtype=np.float32)
        total_positive_delta = np.empty((n,), dtype=np.float32)
        lattice_u = np.empty((n,), dtype=np.float32)
        lattice_v = np.empty((n,), dtype=np.float32)
        front_radius.fill(np.nan)
        total_positive_delta.fill(np.nan)
        lattice_u.fill(np.nan)
        lattice_v.fill(np.nan)
        for i, row in enumerate(rows):
            hole_id[i] = _safe_int(row.get('hole_id', 0))
            frame_id[i] = _safe_int(row.get('frame_id', 0))
            sector_id[i] = _safe_int(row.get('sector_id', 0))
            front_radius[i] = np.float32(_safe_float(row.get('sector_rdf_front_radius_norm', np.nan)))
            total_positive_delta[i] = np.float32(_safe_float(row.get('sector_rdf_total_positive_delta', np.nan)))
            lattice_u[i] = np.float32(_safe_float(row.get('lattice_u', np.nan)))
            lattice_v[i] = np.float32(_safe_float(row.get('lattice_v', np.nan)))
        order_hs = np.lexsort((frame_id, sector_id, hole_id)).astype(np.int32)
        hid = hole_id[order_hs]
        sid = sector_id[order_hs]
        starts, ends = _group_bounds(hid, sid)
        return cls(hole_id, frame_id, sector_id, front_radius, total_positive_delta, lattice_u, lattice_v, order_hs, starts, ends)

    def uv_meta(self, idx: int) -> tuple[Any, Any]:
        u = self.lattice_u[idx]
        v = self.lattice_v[idx]
        out_u = None if not np.isfinite(u) else int(round(float(u)))
        out_v = None if not np.isfinite(v) else int(round(float(v)))
        return out_u, out_v

    def first_hole_series(self, hole_id: int) -> tuple[list[int], list[tuple[str, list[float]]]]:
        mask = self.hole_id == int(hole_id)
        if not np.any(mask):
            return [], []
        frames = self.frame_id[mask]
        ann = self.annulus_id[mask]
        vals = self.descriptor_value[mask].astype(float, copy=False)
        order = np.lexsort((ann, frames))
        frames = frames[order]
        ann = ann[order]
        vals = vals[order]
        x_vals = sorted({int(v) for v in frames.tolist()})
        series: list[tuple[str, list[float]]] = []
        for annulus in sorted({int(v) for v in ann.tolist()}):
            amask = ann == annulus
            a_frames = frames[amask]
            a_vals = vals[amask]
            lookup = {int(f): (None if not np.isfinite(v) else float(v)) for f, v in zip(a_frames.tolist(), a_vals.tolist())}
            series.append((f'annulus_{annulus}', [lookup.get(fid) for fid in x_vals]))
        return x_vals, series

    def iter_hole_sector_groups(self):
        order = self.order_hs
        hole = self.hole_id[order]
        sector = self.sector_id[order]
        for s, e in zip(self.hs_starts, self.hs_ends):
            idx = order[s:e]
            yield int(hole[s]), int(sector[s]), idx


@dataclass(slots=True)
class SectorRdfColumns:
    evolution: dict[str, list[Any]]
    frame_summary: dict[str, list[Any]]

    def frame_rows(self) -> list[dict[str, Any]]:
        cols = self.frame_summary
        keys = list(cols.keys())
        n = len(cols[keys[0]]) if keys else 0
        return [{k: cols[k][i] for k in keys} for i in range(n)]


def build_sector_rdf_evolution_columns(table: SectorRadialTable) -> SectorRdfColumns:
    baseline = table.baseline_by_hole_sector_annulus()
    evo = {
        'hole_id': [], 'frame_id': [], 'sector_id': [], 'annulus_id': [],
        'normalized_radius': [], 'descriptor_value': [], 'baseline_descriptor_value': [],
        'delta_descriptor_value': [], 'positive_delta_descriptor': [],
        'sector_rdf_pdf': [], 'sector_rdf_cdf': [], 'lattice_u': [], 'lattice_v': [],
    }
    frame = {
        'hole_id': [], 'frame_id': [], 'sector_id': [],
        'sector_rdf_front_radius_norm': [], 'sector_rdf_total_positive_delta': [],
        'lattice_u': [], 'lattice_v': [],
    }
    for hole_id, frame_id, sector_id, idx in table.iter_hole_frame_sector_groups():
        ann = table.annulus_id[idx].astype(float, copy=False)
        vals = table.descriptor_value[idx].astype(float, copy=False)
        baselines = np.asarray([baseline.get((hole_id, sector_id, int(a)), np.nan) for a in ann], dtype=float)
        delta = vals - baselines
        positive = np.clip(delta, 0.0, None)
        total = float(np.nansum(positive))
        max_ann = max(float(np.nanmax(ann)) if ann.size else 0.0, 1.0)
        norm_r = ann / max_ann if ann.size else ann
        pdf = positive / total if total > 0 else np.full_like(positive, np.nan, dtype=float)
        cdf = np.cumsum(np.nan_to_num(pdf, nan=0.0)) if pdf.size else pdf
        front = float(np.nansum(norm_r * pdf)) if total > 0 and np.isfinite(pdf).any() else np.nan
        uv = table.uv_meta(int(idx[0]))
        frame['hole_id'].append(int(hole_id))
        frame['frame_id'].append(int(frame_id))
        frame['sector_id'].append(int(sector_id))
        frame['sector_rdf_front_radius_norm'].append(None if not np.isfinite(front) else float(front))
        frame['sector_rdf_total_positive_delta'].append(total)
        frame['lattice_u'].append(uv[0])
        frame['lattice_v'].append(uv[1])
        for i, ridx in enumerate(idx):
            evo['hole_id'].append(int(hole_id))
            evo['frame_id'].append(int(frame_id))
            evo['sector_id'].append(int(sector_id))
            evo['annulus_id'].append(int(table.annulus_id[ridx]))
            evo['normalized_radius'].append(None if not (norm_r.size and np.isfinite(norm_r[i])) else float(norm_r[i]))
            evo['descriptor_value'].append(None if not np.isfinite(vals[i]) else float(vals[i]))
            evo['baseline_descriptor_value'].append(None if not np.isfinite(baselines[i]) else float(baselines[i]))
            evo['delta_descriptor_value'].append(None if not np.isfinite(delta[i]) else float(delta[i]))
            evo['positive_delta_descriptor'].append(None if not np.isfinite(positive[i]) else float(positive[i]))
            evo['sector_rdf_pdf'].append(None if not (pdf.size and np.isfinite(pdf[i])) else float(pdf[i]))
            evo['sector_rdf_cdf'].append(None if not (cdf.size and np.isfinite(cdf[i])) else float(cdf[i]))
            evo['lattice_u'].append(uv[0])
            evo['lattice_v'].append(uv[1])
    return SectorRdfColumns(evo, frame)


def build_sector_rdf_evolution_table(table: SectorRadialTable) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    cols = build_sector_rdf_evolution_columns(table)
    evo_keys = list(cols.evolution.keys())
    evo_n = len(cols.evolution[evo_keys[0]]) if evo_keys else 0
    evo_rows = [{k: cols.evolution[k][i] for k in evo_keys} for i in range(evo_n)]
    return evo_rows, cols.frame_rows()


def summarize_sector_fronts_table(table: SectorRadialTable) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    sector_front_rows: list[dict[str, Any]] = []
    frame_summary_rows: list[dict[str, Any]] = []
    for hole_id, frame_id, sector_idx in table.iter_hole_frame_groups():
        uv = table.uv_meta(int(sector_idx[0]))
        fronts = []
        for ridx in sector_idx:
            pass
        # regroup within this hole/frame by sector
        sub_order = np.argsort(table.sector_id[sector_idx], kind='mergesort')
        idx = sector_idx[sub_order]
        sectors = table.sector_id[idx]
        starts, ends = _group_bounds(sectors)
        per_front = []
        for s, e in zip(starts, ends):
            gidx = idx[s:e]
            ann = table.annulus_id[gidx].astype(float, copy=False)
            vals = table.descriptor_value[gidx].astype(float, copy=False)
            valid = np.isfinite(ann) & np.isfinite(vals)
            if int(valid.sum()) == 0:
                continue
            ann = ann[valid]
            vals = vals[valid]
            shifted = vals - float(np.nanmin(vals)) + 1e-6
            denom = float(np.sum(shifted))
            com = float(np.sum(ann * shifted) / denom) if denom > 0 else float(np.nanmean(ann))
            peak_annulus = int(ann[int(np.nanargmax(vals))]) if vals.size else None
            row = {
                'hole_id': int(hole_id),
                'frame_id': int(frame_id),
                'sector_id': int(sectors[s]),
                'lattice_u': uv[0],
                'lattice_v': uv[1],
                'sector_front_radius': com,
                'sector_peak_annulus': peak_annulus,
                'n_valid_annuli': int(len(vals)),
            }
            sector_front_rows.append(row)
            per_front.append(com)
        if per_front:
            valid = np.asarray(per_front, dtype=float)
            mean_front = float(np.mean(valid))
            std_front = float(np.std(valid))
            frame_summary_rows.append({
                'hole_id': int(hole_id),
                'frame_id': int(frame_id),
                'lattice_u': uv[0],
                'lattice_v': uv[1],
                'mean_sector_front_radius': mean_front,
                'std_sector_front_radius': std_front,
                'max_sector_front_radius': float(np.max(valid)),
                'min_sector_front_radius': float(np.min(valid)),
                'front_anisotropy_ratio': float(std_front / (abs(mean_front) + 1e-6)),
                'n_valid_sectors': int(valid.size),
            })
    return sector_front_rows, frame_summary_rows


def build_sector_front_lag_rows_table(frame_table: SectorRdfFrameTable, onset_threshold: float = 0.01) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    lag_rows: list[dict[str, Any]] = []
    by_hole: dict[int, list[dict[str, Any]]] = {}
    for hole_id, sector_id, idx in frame_table.iter_hole_sector_groups():
        frames = frame_table.frame_id[idx].astype(float, copy=False)
        delta = frame_table.total_positive_delta[idx].astype(float, copy=False)
        front = frame_table.front_radius[idx].astype(float, copy=False)
        finite_delta = np.isfinite(delta)
        finite_front = np.isfinite(front)
        onset_frame = None
        if np.any(finite_delta):
            hit = np.where((delta > float(onset_threshold)) & finite_delta)[0]
            if hit.size:
                onset_frame = int(frames[hit[0]])
        peak_frame = int(frames[np.nanargmax(front)]) if np.any(finite_front) else None
        uv = frame_table.uv_meta(int(idx[0]))
        row = {
            'hole_id': int(hole_id),
            'sector_id': int(sector_id),
            'lattice_u': uv[0],
            'lattice_v': uv[1],
            'sector_onset_frame': onset_frame,
            'sector_peak_frame': peak_frame,
            'sector_peak_front_radius_norm': None if not np.any(finite_front) else float(np.nanmax(front)),
            'sector_mean_front_radius_norm': None if not np.any(finite_front) else float(np.nanmean(front)),
            'valid_frames': int(np.sum(finite_front)),
        }
        lag_rows.append(row)
        by_hole.setdefault(int(hole_id), []).append(row)
    summary_rows: list[dict[str, Any]] = []
    for hole_id, rows in sorted(by_hole.items()):
        onset_vals = [r['sector_onset_frame'] for r in rows if r.get('sector_onset_frame') is not None]
        peak_vals = [r['sector_peak_frame'] for r in rows if r.get('sector_peak_frame') is not None]
        min_on = min(onset_vals) if onset_vals else None
        min_pk = min(peak_vals) if peak_vals else None
        onset_lags = []
        peak_lags = []
        for r in rows:
            r['sector_onset_lag'] = None if min_on is None or r.get('sector_onset_frame') is None else int(r['sector_onset_frame'] - min_on)
            r['sector_peak_lag'] = None if min_pk is None or r.get('sector_peak_frame') is None else int(r['sector_peak_frame'] - min_pk)
            if r['sector_onset_lag'] is not None:
                onset_lags.append(r['sector_onset_lag'])
            if r['sector_peak_lag'] is not None:
                peak_lags.append(r['sector_peak_lag'])
        summary_rows.append({
            'hole_id': int(hole_id),
            'lattice_u': rows[0].get('lattice_u'),
            'lattice_v': rows[0].get('lattice_v'),
            'min_sector_onset_frame': min_on,
            'min_sector_peak_frame': min_pk,
            'max_sector_onset_lag': None if not onset_lags else int(max(onset_lags)),
            'max_sector_peak_lag': None if not peak_lags else int(max(peak_lags)),
            'mean_sector_onset_lag': None if not onset_lags else float(np.mean(onset_lags)),
            'mean_sector_peak_lag': None if not peak_lags else float(np.mean(peak_lags)),
            'valid_onset_sector_fraction': float(len(onset_lags) / len(rows)) if rows else 0.0,
            'valid_peak_sector_fraction': float(len(peak_lags) / len(rows)) if rows else 0.0,
        })
    return lag_rows, summary_rows


def build_sector_front_propagation_table(frame_table: SectorRdfFrameTable, onset_threshold: float = 0.01) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    sector_rows: list[dict[str, Any]] = []
    by_hole: dict[int, list[dict[str, Any]]] = {}
    for hole_id, sector_id, idx in frame_table.iter_hole_sector_groups():
        frames = frame_table.frame_id[idx].astype(float, copy=False)
        front = frame_table.front_radius[idx].astype(float, copy=False)
        total = frame_table.total_positive_delta[idx].astype(float, copy=False)
        valid = np.isfinite(frames) & np.isfinite(front)
        slope = r2 = np.nan
        if int(np.sum(valid)) >= 2 and np.ptp(frames[valid]) > 0:
            coeff = np.polyfit(frames[valid], front[valid], 1)
            pred = np.polyval(coeff, frames[valid])
            ss_res = float(np.sum((front[valid] - pred) ** 2))
            ss_tot = float(np.sum((front[valid] - np.mean(front[valid])) ** 2))
            r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
            slope = float(coeff[0])
        onset = None
        tv = np.isfinite(total)
        if np.any(tv):
            hit = np.where((total > float(onset_threshold)) & tv)[0]
            if hit.size:
                onset = int(frames[hit[0]])
        uv = frame_table.uv_meta(int(idx[0]))
        out = {
            'hole_id': int(hole_id),
            'sector_id': int(sector_id),
            'lattice_u': uv[0],
            'lattice_v': uv[1],
            'sector_front_velocity_per_frame': None if not np.isfinite(slope) else float(slope),
            'sector_front_velocity_r2': None if not np.isfinite(r2) else float(r2),
            'sector_front_onset_frame': onset,
            'sector_front_start_radius_norm': None if not np.any(valid) else float(front[valid][0]),
            'sector_front_end_radius_norm': None if not np.any(valid) else float(front[valid][-1]),
            'sector_front_delta_radius_norm': None if not np.any(valid) else float(front[valid][-1] - front[valid][0]),
            'valid_frames': int(np.sum(valid)),
        }
        sector_rows.append(out)
        by_hole.setdefault(int(hole_id), []).append(out)
    hole_rows: list[dict[str, Any]] = []
    for hole_id, rows in sorted(by_hole.items()):
        vel = np.asarray([_safe_float(r.get('sector_front_velocity_per_frame', np.nan)) for r in rows], dtype=float)
        velv = vel[np.isfinite(vel)]
        onsets = [int(r['sector_front_onset_frame']) for r in rows if r.get('sector_front_onset_frame') is not None]
        lags = [o - min(onsets) for o in onsets] if onsets else []
        hole_rows.append({
            'hole_id': int(hole_id),
            'lattice_u': rows[0].get('lattice_u'),
            'lattice_v': rows[0].get('lattice_v'),
            'mean_sector_front_velocity_per_frame': None if velv.size == 0 else float(np.mean(velv)),
            'std_sector_front_velocity_per_frame': None if velv.size == 0 else float(np.std(velv)),
            'sector_front_velocity_anisotropy': None if velv.size == 0 else float(np.std(velv) / max(abs(np.mean(velv)), 1e-6)),
            'valid_sector_velocity_fraction': float(velv.size / max(len(rows), 1)),
            'max_sector_onset_lag': None if not lags else int(max(lags)),
            'mean_sector_onset_lag': None if not lags else float(np.mean(lags)),
        })
    return sector_rows, hole_rows


def build_sector_front_acceleration_table(frame_table: SectorRdfFrameTable) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    sector_rows: list[dict[str, Any]] = []
    by_hole: dict[int, list[dict[str, Any]]] = {}
    for hole_id, sector_id, idx in frame_table.iter_hole_sector_groups():
        frames = frame_table.frame_id[idx].astype(float, copy=False)
        front = frame_table.front_radius[idx].astype(float, copy=False)
        valid = np.isfinite(frames) & np.isfinite(front)
        if int(np.sum(valid)) < 3 or np.ptp(frames[valid]) <= 0:
            continue
        xf = frames[valid]
        yf = front[valid]
        lin = np.polyfit(xf, yf, 1)
        ylin = np.polyval(lin, xf)
        ss_res = float(np.sum((yf - ylin) ** 2))
        ss_tot = float(np.sum((yf - np.mean(yf)) ** 2))
        lin_r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
        quad = np.polyfit(xf, yf, 2)
        yquad = np.polyval(quad, xf)
        ss_res_q = float(np.sum((yf - yquad) ** 2))
        quad_r2 = 1.0 - ss_res_q / ss_tot if ss_tot > 0 else 1.0
        a, b, c = [float(v) for v in quad]
        uv = frame_table.uv_meta(int(idx[0]))
        out = {
            'hole_id': int(hole_id),
            'sector_id': int(sector_id),
            'lattice_u': uv[0],
            'lattice_v': uv[1],
            'sector_front_linear_velocity_per_frame': float(lin[0]),
            'sector_front_linear_r2': float(lin_r2),
            'sector_front_quadratic_a': a,
            'sector_front_quadratic_b': b,
            'sector_front_quadratic_c': c,
            'sector_front_quadratic_r2': float(quad_r2),
            'sector_front_acceleration_per_frame2': float(2.0 * a),
            'sector_front_curvature': float(a),
            'sector_front_nonlinearity_gain': float(quad_r2 - lin_r2),
            'valid_frames': int(np.sum(valid)),
        }
        sector_rows.append(out)
        by_hole.setdefault(int(hole_id), []).append(out)
    summary_rows: list[dict[str, Any]] = []
    for hole_id, rows in sorted(by_hole.items()):
        acc = np.asarray([_safe_float(r.get('sector_front_acceleration_per_frame2', np.nan)) for r in rows], dtype=float)
        curv = np.asarray([_safe_float(r.get('sector_front_curvature', np.nan)) for r in rows], dtype=float)
        nlg = np.asarray([_safe_float(r.get('sector_front_nonlinearity_gain', np.nan)) for r in rows], dtype=float)
        summary_rows.append({
            'hole_id': int(hole_id),
            'lattice_u': rows[0].get('lattice_u'),
            'lattice_v': rows[0].get('lattice_v'),
            'mean_sector_front_acceleration_per_frame2': None if not np.isfinite(acc).any() else float(np.nanmean(acc)),
            'sector_front_acceleration_spread': None if not np.isfinite(acc).any() else float(np.nanstd(acc)),
            'mean_abs_sector_front_curvature': None if not np.isfinite(curv).any() else float(np.nanmean(np.abs(curv))),
            'mean_sector_front_nonlinearity_gain': None if not np.isfinite(nlg).any() else float(np.nanmean(nlg)),
            'valid_sector_acceleration_fraction': float(np.mean(np.isfinite(acc))) if acc.size else 0.0,
            'n_valid_sectors': int(np.sum(np.isfinite(acc))),
        })
    return sector_rows, summary_rows



def sector_hole_summary_plot_columns(rows: list[dict[str, Any]], *, value_field: str) -> dict[str, list[Any]]:
    if not rows:
        return {'label': [], value_field: []}
    cols = {'label': [], value_field: []}
    for row in rows[:12]:
        cols['label'].append(f"hole_{_safe_int(row.get('hole_id', 0))}")
        val = row.get(value_field)
        cols[value_field].append(None if val is None else float(_safe_float(val, np.nan)))
    return cols


def sector_front_lag_columns(rows: list[dict[str, Any]]) -> dict[str, list[Any]]:
    if not rows:
        return {'hole_id': [], 'sector_id': [], 'sector_onset_lag': []}
    return {
        'hole_id': [_safe_int(r.get('hole_id', 0)) for r in rows],
        'sector_id': [_safe_int(r.get('sector_id', 0)) for r in rows],
        'sector_onset_lag': [None if r.get('sector_onset_lag') is None else int(_safe_int(r.get('sector_onset_lag', 0))) for r in rows],
    }


@dataclass(slots=True)
class HotspotStatsTable:
    hole_id: np.ndarray
    n_hotspots: np.ndarray
    mean_hotspot_distance_px: np.ndarray
    min_hotspot_distance_px: np.ndarray

    @classmethod
    def from_rows(cls, rows: list[dict[str, Any]]) -> 'HotspotStatsTable':
        if not rows:
            z_i = np.zeros((0,), dtype=np.int32)
            z_f = np.zeros((0,), dtype=np.float32)
            return cls(z_i, z_i.copy(), z_f, z_f.copy())
        acc: dict[int, list[float]] = {}
        for row in rows:
            hid = row.get('nearest_hole_id')
            if hid is None:
                continue
            hole_id = _safe_int(hid, default=-1)
            if hole_id < 0:
                continue
            bucket = acc.setdefault(hole_id, [0.0, 0.0, np.inf, 0.0])
            bucket[0] += 1.0
            dist = _safe_float(row.get('dist_to_hole_px', np.nan))
            if np.isfinite(dist):
                bucket[1] += float(dist)
                bucket[3] += 1.0
                if dist < bucket[2]:
                    bucket[2] = float(dist)
        hole_ids = np.asarray(sorted(acc), dtype=np.int32)
        n_hotspots = np.empty(hole_ids.shape, dtype=np.int32)
        mean_dist = np.empty(hole_ids.shape, dtype=np.float32)
        min_dist = np.empty(hole_ids.shape, dtype=np.float32)
        mean_dist.fill(np.nan)
        min_dist.fill(np.nan)
        for i, hole_id in enumerate(hole_ids):
            n, sum_dist, min_d, count_dist = acc[int(hole_id)]
            n_hotspots[i] = int(n)
            if count_dist > 0:
                mean_dist[i] = np.float32(sum_dist / count_dist)
            if np.isfinite(min_d):
                min_dist[i] = np.float32(min_d)
        return cls(hole_ids, n_hotspots, mean_dist, min_dist)

    def lookup(self, hole_id: int) -> tuple[int, float | None, float | None]:
        if self.hole_id.size == 0:
            return 0, None, None
        pos = int(np.searchsorted(self.hole_id, int(hole_id)))
        if pos >= int(self.hole_id.size) or int(self.hole_id[pos]) != int(hole_id):
            return 0, None, None
        mean_dist = float(self.mean_hotspot_distance_px[pos]) if np.isfinite(self.mean_hotspot_distance_px[pos]) else None
        min_dist = float(self.min_hotspot_distance_px[pos]) if np.isfinite(self.min_hotspot_distance_px[pos]) else None
        return int(self.n_hotspots[pos]), mean_dist, min_dist


def _proximity_bucket(min_dist: float | None) -> str:
    if min_dist is None:
        return 'none'
    if min_dist <= 10:
        return 'near'
    if min_dist <= 24:
        return 'mid'
    return 'far'


def build_hotspot_reticulum_comparison_table(
    hotspot_table: HotspotStatsTable,
    per_hole_radial_summary_rows: list[dict[str, Any]],
    radial_archetype_rows: list[dict[str, Any]],
    zone_by_hole: dict[int, str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not per_hole_radial_summary_rows:
        return [], []
    arche_by_hole = {int(r.get('hole_id', 0)): r for r in radial_archetype_rows}
    out: list[dict[str, Any]] = []
    group_acc: dict[tuple[str, str], dict[str, Any]] = {}
    for row in per_hole_radial_summary_rows:
        hole_id = int(row.get('hole_id', 0))
        n_hotspots, mean_dist, min_dist = hotspot_table.lookup(hole_id)
        bucket = _proximity_bucket(min_dist)
        arche = arche_by_hole.get(hole_id, {})
        out_row = {
            'hole_id': hole_id,
            'lattice_u': row.get('lattice_u'),
            'lattice_v': row.get('lattice_v'),
            'reticulum_zone': zone_by_hole.get(hole_id, 'unknown'),
            'radial_conclusion_label': row.get('radial_conclusion_label'),
            'radial_archetype_label': arche.get('radial_archetype_label'),
            'delta_center_of_mass': row.get('delta_center_of_mass'),
            'mean_inner_minus_outer': row.get('mean_inner_minus_outer'),
            'mean_angular_asymmetry': row.get('mean_angular_asymmetry'),
            'n_hotspots': n_hotspots,
            'mean_hotspot_distance_px': mean_dist,
            'min_hotspot_distance_px': min_dist,
            'hotspot_proximity_bucket': bucket,
        }
        out.append(out_row)
        key = (str(out_row.get('reticulum_zone', 'unknown')), str(bucket))
        acc = group_acc.setdefault(key, _acc_init(('delta_center_of_mass', 'mean_inner_minus_outer', 'mean_angular_asymmetry', 'mean_hotspot_distance_px')))
        acc['n'] += 1
        _acc_add(acc, 'delta_center_of_mass', out_row.get('delta_center_of_mass'))
        _acc_add(acc, 'mean_inner_minus_outer', out_row.get('mean_inner_minus_outer'))
        _acc_add(acc, 'mean_angular_asymmetry', out_row.get('mean_angular_asymmetry'))
        _acc_add(acc, 'mean_hotspot_distance_px', out_row.get('mean_hotspot_distance_px'))
    group_rows: list[dict[str, Any]] = []
    for (zone, bucket), acc in sorted(group_acc.items()):
        group_rows.append({
            'reticulum_zone': zone,
            'hotspot_proximity_bucket': bucket,
            'n_holes': int(acc.get('n', 0)),
            'mean_delta_center_of_mass': _acc_mean(acc, 'delta_center_of_mass'),
            'mean_inner_minus_outer': _acc_mean(acc, 'mean_inner_minus_outer'),
            'mean_angular_asymmetry': _acc_mean(acc, 'mean_angular_asymmetry'),
            'mean_hotspot_distance_px': _acc_mean(acc, 'mean_hotspot_distance_px'),
        })
    return out, group_rows


def build_rdf_hotspot_reticulum_comparison_table(
    hotspot_table: HotspotStatsTable,
    rdf_archetype_rows: list[dict[str, Any]],
    rdf_dynamics_rows: list[dict[str, Any]],
    zone_by_hole: dict[int, str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not rdf_archetype_rows:
        return [], []
    dyn_by_hole = {int(r.get('hole_id', 0)): r for r in rdf_dynamics_rows}
    out: list[dict[str, Any]] = []
    group_acc: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in rdf_archetype_rows:
        hole_id = int(row.get('hole_id', 0))
        n_hotspots, mean_dist, min_dist = hotspot_table.lookup(hole_id)
        bucket = _proximity_bucket(min_dist)
        dyn = dyn_by_hole.get(hole_id, {})
        out_row = {
            'hole_id': hole_id,
            'lattice_u': row.get('lattice_u'),
            'lattice_v': row.get('lattice_v'),
            'reticulum_zone': zone_by_hole.get(hole_id, 'unknown'),
            'rdf_archetype_canonical_id': row.get('rdf_archetype_canonical_id'),
            'rdf_archetype_canonical_label': row.get('rdf_archetype_canonical_label'),
            'mean_front_radius_norm': row.get('mean_front_radius_norm'),
            'delta_front_radius_norm': row.get('delta_front_radius_norm'),
            'rdf_front_acceleration_per_frame2': dyn.get('rdf_front_acceleration_per_frame2'),
            'rdf_front_nonlinearity_gain': dyn.get('rdf_front_nonlinearity_gain'),
            'n_hotspots': n_hotspots,
            'mean_hotspot_distance_px': mean_dist,
            'min_hotspot_distance_px': min_dist,
            'hotspot_proximity_bucket': bucket,
        }
        out.append(out_row)
        key = (str(out_row.get('reticulum_zone', 'unknown')), str(bucket), str(out_row.get('rdf_archetype_canonical_label', 'unknown')))
        acc = group_acc.setdefault(key, _acc_init(('mean_front_radius_norm', 'delta_front_radius_norm', 'rdf_front_acceleration_per_frame2', 'mean_hotspot_distance_px')))
        acc['n'] += 1
        _acc_add(acc, 'mean_front_radius_norm', out_row.get('mean_front_radius_norm'))
        _acc_add(acc, 'delta_front_radius_norm', out_row.get('delta_front_radius_norm'))
        _acc_add(acc, 'rdf_front_acceleration_per_frame2', out_row.get('rdf_front_acceleration_per_frame2'))
        _acc_add(acc, 'mean_hotspot_distance_px', out_row.get('mean_hotspot_distance_px'))
    group_rows: list[dict[str, Any]] = []
    for (zone, bucket, arche), acc in sorted(group_acc.items()):
        group_rows.append({
            'reticulum_zone': zone,
            'hotspot_proximity_bucket': bucket,
            'rdf_archetype_canonical_label': arche,
            'n_holes': int(acc.get('n', 0)),
            'mean_front_radius_norm': _acc_mean(acc, 'mean_front_radius_norm'),
            'mean_delta_front_radius_norm': _acc_mean(acc, 'delta_front_radius_norm'),
            'mean_rdf_front_acceleration_per_frame2': _acc_mean(acc, 'rdf_front_acceleration_per_frame2'),
            'mean_hotspot_distance_px': _acc_mean(acc, 'mean_hotspot_distance_px'),
        })
    return out, group_rows


def _columns_from_records(records: list[dict[str, Any]], field_order: list[str] | None = None) -> dict[str, list[Any]]:
    if not records:
        return {field: [] for field in (field_order or [])}
    if field_order is None:
        field_order = list(records[0].keys())
    cols: dict[str, list[Any]] = {field: [] for field in field_order}
    for rec in records:
        for field in field_order:
            cols[field].append(rec.get(field))
    return cols


def build_hotspot_reticulum_columns_table(
    hotspot_table: HotspotStatsTable,
    per_hole_radial_summary_rows: list[dict[str, Any]],
    radial_archetype_rows: list[dict[str, Any]],
    zone_by_hole: dict[int, str],
) -> tuple[dict[str, list[Any]], dict[str, list[Any]]]:
    row_fields = [
        'hole_id', 'lattice_u', 'lattice_v', 'reticulum_zone', 'radial_conclusion_label',
        'radial_archetype_label', 'delta_center_of_mass', 'mean_inner_minus_outer',
        'mean_angular_asymmetry', 'n_hotspots', 'mean_hotspot_distance_px',
        'min_hotspot_distance_px', 'hotspot_proximity_bucket'
    ]
    group_fields = [
        'reticulum_zone', 'hotspot_proximity_bucket', 'n_holes',
        'mean_delta_center_of_mass', 'mean_inner_minus_outer',
        'mean_angular_asymmetry', 'mean_hotspot_distance_px'
    ]
    if not per_hole_radial_summary_rows:
        return {f: [] for f in row_fields}, {f: [] for f in group_fields}
    arche_by_hole = {int(r.get('hole_id', 0)): r for r in radial_archetype_rows}
    cols = {f: [] for f in row_fields}
    group_acc: dict[tuple[str, str], dict[str, Any]] = {}
    for row in per_hole_radial_summary_rows:
        hole_id = int(row.get('hole_id', 0))
        n_hotspots, mean_dist, min_dist = hotspot_table.lookup(hole_id)
        bucket = _proximity_bucket(min_dist)
        arche = arche_by_hole.get(hole_id, {})
        cols['hole_id'].append(hole_id)
        cols['lattice_u'].append(row.get('lattice_u'))
        cols['lattice_v'].append(row.get('lattice_v'))
        zone = zone_by_hole.get(hole_id, 'unknown')
        cols['reticulum_zone'].append(zone)
        cols['radial_conclusion_label'].append(row.get('radial_conclusion_label'))
        cols['radial_archetype_label'].append(arche.get('radial_archetype_label'))
        cols['delta_center_of_mass'].append(row.get('delta_center_of_mass'))
        cols['mean_inner_minus_outer'].append(row.get('mean_inner_minus_outer'))
        cols['mean_angular_asymmetry'].append(row.get('mean_angular_asymmetry'))
        cols['n_hotspots'].append(n_hotspots)
        cols['mean_hotspot_distance_px'].append(mean_dist)
        cols['min_hotspot_distance_px'].append(min_dist)
        cols['hotspot_proximity_bucket'].append(bucket)
        key = (str(zone), str(bucket))
        acc = group_acc.setdefault(key, _acc_init(('delta_center_of_mass', 'mean_inner_minus_outer', 'mean_angular_asymmetry', 'mean_hotspot_distance_px')))
        acc['n'] += 1
        _acc_add(acc, 'delta_center_of_mass', row.get('delta_center_of_mass'))
        _acc_add(acc, 'mean_inner_minus_outer', row.get('mean_inner_minus_outer'))
        _acc_add(acc, 'mean_angular_asymmetry', row.get('mean_angular_asymmetry'))
        _acc_add(acc, 'mean_hotspot_distance_px', mean_dist)
    group_rows: list[dict[str, Any]] = []
    for (zone, bucket), acc in sorted(group_acc.items()):
        group_rows.append({
            'reticulum_zone': zone,
            'hotspot_proximity_bucket': bucket,
            'n_holes': int(acc.get('n', 0)),
            'mean_delta_center_of_mass': _acc_mean(acc, 'delta_center_of_mass'),
            'mean_inner_minus_outer': _acc_mean(acc, 'mean_inner_minus_outer'),
            'mean_angular_asymmetry': _acc_mean(acc, 'mean_angular_asymmetry'),
            'mean_hotspot_distance_px': _acc_mean(acc, 'mean_hotspot_distance_px'),
        })
    return cols, _columns_from_records(group_rows, group_fields)


def build_rdf_hotspot_reticulum_columns_table(
    hotspot_table: HotspotStatsTable,
    rdf_archetype_rows: list[dict[str, Any]],
    rdf_dynamics_rows: list[dict[str, Any]],
    zone_by_hole: dict[int, str],
) -> tuple[dict[str, list[Any]], dict[str, list[Any]]]:
    row_fields = [
        'hole_id', 'lattice_u', 'lattice_v', 'reticulum_zone',
        'rdf_archetype_canonical_id', 'rdf_archetype_canonical_label',
        'mean_front_radius_norm', 'delta_front_radius_norm',
        'rdf_front_acceleration_per_frame2', 'rdf_front_nonlinearity_gain',
        'n_hotspots', 'mean_hotspot_distance_px', 'min_hotspot_distance_px',
        'hotspot_proximity_bucket'
    ]
    group_fields = [
        'reticulum_zone', 'hotspot_proximity_bucket', 'rdf_archetype_canonical_label', 'n_holes',
        'mean_front_radius_norm', 'mean_delta_front_radius_norm',
        'mean_rdf_front_acceleration_per_frame2', 'mean_hotspot_distance_px'
    ]
    if not rdf_archetype_rows:
        return {f: [] for f in row_fields}, {f: [] for f in group_fields}
    dyn_by_hole = {int(r.get('hole_id', 0)): r for r in rdf_dynamics_rows}
    cols = {f: [] for f in row_fields}
    group_acc: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in rdf_archetype_rows:
        hole_id = int(row.get('hole_id', 0))
        n_hotspots, mean_dist, min_dist = hotspot_table.lookup(hole_id)
        bucket = _proximity_bucket(min_dist)
        dyn = dyn_by_hole.get(hole_id, {})
        zone = zone_by_hole.get(hole_id, 'unknown')
        arche = row.get('rdf_archetype_canonical_label')
        cols['hole_id'].append(hole_id)
        cols['lattice_u'].append(row.get('lattice_u'))
        cols['lattice_v'].append(row.get('lattice_v'))
        cols['reticulum_zone'].append(zone)
        cols['rdf_archetype_canonical_id'].append(row.get('rdf_archetype_canonical_id'))
        cols['rdf_archetype_canonical_label'].append(arche)
        cols['mean_front_radius_norm'].append(row.get('mean_front_radius_norm'))
        cols['delta_front_radius_norm'].append(row.get('delta_front_radius_norm'))
        cols['rdf_front_acceleration_per_frame2'].append(dyn.get('rdf_front_acceleration_per_frame2'))
        cols['rdf_front_nonlinearity_gain'].append(dyn.get('rdf_front_nonlinearity_gain'))
        cols['n_hotspots'].append(n_hotspots)
        cols['mean_hotspot_distance_px'].append(mean_dist)
        cols['min_hotspot_distance_px'].append(min_dist)
        cols['hotspot_proximity_bucket'].append(bucket)
        key = (str(zone), str(bucket), str(arche))
        acc = group_acc.setdefault(key, _acc_init(('mean_front_radius_norm', 'delta_front_radius_norm', 'rdf_front_acceleration_per_frame2', 'mean_hotspot_distance_px')))
        acc['n'] += 1
        _acc_add(acc, 'mean_front_radius_norm', row.get('mean_front_radius_norm'))
        _acc_add(acc, 'delta_front_radius_norm', row.get('delta_front_radius_norm'))
        _acc_add(acc, 'rdf_front_acceleration_per_frame2', dyn.get('rdf_front_acceleration_per_frame2'))
        _acc_add(acc, 'mean_hotspot_distance_px', mean_dist)
    group_rows: list[dict[str, Any]] = []
    for (zone, bucket, arche), acc in sorted(group_acc.items()):
        group_rows.append({
            'reticulum_zone': zone,
            'hotspot_proximity_bucket': bucket,
            'rdf_archetype_canonical_label': arche,
            'n_holes': int(acc.get('n', 0)),
            'mean_front_radius_norm': _acc_mean(acc, 'mean_front_radius_norm'),
            'mean_delta_front_radius_norm': _acc_mean(acc, 'delta_front_radius_norm'),
            'mean_rdf_front_acceleration_per_frame2': _acc_mean(acc, 'rdf_front_acceleration_per_frame2'),
            'mean_hotspot_distance_px': _acc_mean(acc, 'mean_hotspot_distance_px'),
        })
    return cols, _columns_from_records(group_rows, group_fields)

@dataclass(slots=True)
class ValidationHoleTable:
    hole_id: np.ndarray
    bootstrap_support_fraction: np.ndarray
    bootstrap_front_velocity_ci_width: np.ndarray
    bootstrap_mean_front_radius_ci_width: np.ndarray
    bootstrap_delta_front_radius_ci_width: np.ndarray
    valid_sector_velocity_fraction: np.ndarray
    valid_sector_acceleration_fraction: np.ndarray

    @classmethod
    def from_rows(
        cls,
        rdf_bootstrap_rows: list[dict[str, Any]] | None = None,
        rdf_bootstrap_support_rows: list[dict[str, Any]] | None = None,
        sector_front_propagation_hole_rows: list[dict[str, Any]] | None = None,
        sector_front_acceleration_hole_rows: list[dict[str, Any]] | None = None,
    ) -> 'ValidationHoleTable':
        rdf_bootstrap_rows = rdf_bootstrap_rows or []
        rdf_bootstrap_support_rows = rdf_bootstrap_support_rows or []
        sector_front_propagation_hole_rows = sector_front_propagation_hole_rows or []
        sector_front_acceleration_hole_rows = sector_front_acceleration_hole_rows or []
        hole_ids = sorted({
            _safe_int(r.get('hole_id', 0)) for r in rdf_bootstrap_rows
        } | {
            _safe_int(r.get('hole_id', 0)) for r in rdf_bootstrap_support_rows
        } | {
            _safe_int(r.get('hole_id', 0)) for r in sector_front_propagation_hole_rows
        } | {
            _safe_int(r.get('hole_id', 0)) for r in sector_front_acceleration_hole_rows
        })
        n = len(hole_ids)
        hid = np.asarray(hole_ids, dtype=np.int32)
        support = np.full((n,), np.nan, dtype=np.float32)
        vel_ci = np.full((n,), np.nan, dtype=np.float32)
        front_ci = np.full((n,), np.nan, dtype=np.float32)
        delta_ci = np.full((n,), np.nan, dtype=np.float32)
        valid_vel = np.full((n,), np.nan, dtype=np.float32)
        valid_acc = np.full((n,), np.nan, dtype=np.float32)
        index = {int(h): i for i, h in enumerate(hole_ids)}
        for row in rdf_bootstrap_rows:
            i = index.get(_safe_int(row.get('hole_id', 0)))
            if i is None:
                continue
            vel_ci[i] = np.float32(_safe_float(row.get('bootstrap_front_velocity_ci_width', np.nan)))
            front_ci[i] = np.float32(_safe_float(row.get('bootstrap_mean_front_radius_ci_width', np.nan)))
            delta_ci[i] = np.float32(_safe_float(row.get('bootstrap_delta_front_radius_ci_width', np.nan)))
        for row in rdf_bootstrap_support_rows:
            i = index.get(_safe_int(row.get('hole_id', 0)))
            if i is None:
                continue
            support[i] = np.float32(_safe_float(row.get('bootstrap_rdf_archetype_support_fraction', np.nan)))
        for row in sector_front_propagation_hole_rows:
            i = index.get(_safe_int(row.get('hole_id', 0)))
            if i is None:
                continue
            valid_vel[i] = np.float32(_safe_float(row.get('valid_sector_velocity_fraction', np.nan)))
        for row in sector_front_acceleration_hole_rows:
            i = index.get(_safe_int(row.get('hole_id', 0)))
            if i is None:
                continue
            valid_acc[i] = np.float32(_safe_float(row.get('valid_sector_acceleration_fraction', np.nan)))
        return cls(hid, support, vel_ci, front_ci, delta_ci, valid_vel, valid_acc)

    def lookup(self, hole_id: int) -> tuple[float, float, float, float, float, float]:
        if self.hole_id.size == 0:
            return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
        pos = np.searchsorted(self.hole_id, int(hole_id))
        if pos >= int(self.hole_id.size) or int(self.hole_id[pos]) != int(hole_id):
            return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
        return (
            float(self.bootstrap_support_fraction[pos]),
            float(self.bootstrap_front_velocity_ci_width[pos]),
            float(self.bootstrap_mean_front_radius_ci_width[pos]),
            float(self.bootstrap_delta_front_radius_ci_width[pos]),
            float(self.valid_sector_velocity_fraction[pos]),
            float(self.valid_sector_acceleration_fraction[pos]),
        )

    def mean_bootstrap_support_fraction(self) -> float | None:
        vals = self.bootstrap_support_fraction.astype(float, copy=False)
        return None if vals.size == 0 or not np.isfinite(vals).any() else float(np.nanmean(vals))

    def mean_bootstrap_front_velocity_ci_width(self) -> float | None:
        vals = self.bootstrap_front_velocity_ci_width.astype(float, copy=False)
        return None if vals.size == 0 or not np.isfinite(vals).any() else float(np.nanmean(vals))

    def mean_bootstrap_front_radius_ci_width(self) -> float | None:
        vals = self.bootstrap_mean_front_radius_ci_width.astype(float, copy=False)
        return None if vals.size == 0 or not np.isfinite(vals).any() else float(np.nanmean(vals))

    def mean_sector_velocity_valid_fraction(self) -> float | None:
        vals = self.valid_sector_velocity_fraction.astype(float, copy=False)
        return None if vals.size == 0 or not np.isfinite(vals).any() else float(np.nanmean(vals))

    def mean_sector_acceleration_valid_fraction(self) -> float | None:
        vals = self.valid_sector_acceleration_fraction.astype(float, copy=False)
        return None if vals.size == 0 or not np.isfinite(vals).any() else float(np.nanmean(vals))


    def bootstrap_ci_plot_columns(self, limit: int | None = None) -> dict[str, list[Any]]:
        n = int(self.hole_id.size) if limit is None else min(int(self.hole_id.size), max(int(limit), 0))
        return {
            'label': [f"hole_{int(h)}" for h in self.hole_id[:n].astype(int).tolist()],
            'bootstrap_front_velocity_ci_width': [None if not np.isfinite(v) else float(v) for v in self.bootstrap_front_velocity_ci_width[:n].tolist()],
        }

    def bootstrap_support_plot_columns(self, limit: int | None = None) -> dict[str, list[Any]]:
        n = int(self.hole_id.size) if limit is None else min(int(self.hole_id.size), max(int(limit), 0))
        return {
            'label': [f"hole_{int(h)}" for h in self.hole_id[:n].astype(int).tolist()],
            'bootstrap_rdf_archetype_support_fraction': [None if not np.isfinite(v) else float(v) for v in self.bootstrap_support_fraction[:n].tolist()],
        }


def build_validation_summary_jsons_table(
    validation_table: ValidationHoleTable,
    rdf_uncertainty_table: RdfUncertaintyHoleTable | None = None,
    validation_enabled: bool = True,
) -> tuple[dict[str, Any], dict[str, Any]]:
    bootstrap_summary = {
        'mean_bootstrap_class_support': validation_table.mean_bootstrap_support_fraction(),
        'mean_velocity_ci_width': validation_table.mean_bootstrap_front_velocity_ci_width(),
        'mean_sector_propagation_valid_fraction': validation_table.mean_sector_velocity_valid_fraction(),
        'validation_enabled': bool(validation_enabled),
    }
    uncertainty_summary = {
        'mean_uncertainty_score': None,
        'mean_hotspot_linked_velocity_ci_width': None,
        'mean_sector_acceleration_valid_fraction': validation_table.mean_sector_acceleration_valid_fraction(),
        'validation_enabled': bool(validation_enabled),
    }
    if rdf_uncertainty_table is not None and int(rdf_uncertainty_table.hole_id.size) > 0:
        u = rdf_uncertainty_table.rdf_uncertainty_score.astype(float, copy=False)
        v = rdf_uncertainty_table.bootstrap_front_velocity_ci_width.astype(float, copy=False)
        uncertainty_summary['mean_uncertainty_score'] = None if not np.isfinite(u).any() else float(np.nanmean(u))
        uncertainty_summary['mean_hotspot_linked_velocity_ci_width'] = None if not np.isfinite(v).any() else float(np.nanmean(v))
    return bootstrap_summary, uncertainty_summary

@dataclass(slots=True)
class RdfUncertaintyHoleTable:
    hole_id: np.ndarray
    lattice_u: np.ndarray
    lattice_v: np.ndarray
    reticulum_zone: np.ndarray
    rdf_archetype_canonical_id: np.ndarray
    rdf_archetype_canonical_label: np.ndarray
    bootstrap_support_fraction: np.ndarray
    bootstrap_front_velocity_ci_width: np.ndarray
    bootstrap_mean_front_radius_ci_width: np.ndarray
    bootstrap_delta_front_radius_ci_width: np.ndarray
    rdf_front_acceleration_per_frame2: np.ndarray
    rdf_front_nonlinearity_gain: np.ndarray
    rdf_uncertainty_score: np.ndarray
    n_hotspots: np.ndarray
    mean_hotspot_distance_px: np.ndarray
    min_hotspot_distance_px: np.ndarray
    hotspot_proximity_bucket: np.ndarray

    @classmethod
    def from_rows(
        cls,
        hotspot_table: HotspotStatsTable,
        rdf_bootstrap_rows: list[dict[str, Any]],
        rdf_bootstrap_support_rows: list[dict[str, Any]],
        rdf_archetype_rows: list[dict[str, Any]],
        zone_by_hole: dict[int, str],
        rdf_dynamics_rows: list[dict[str, Any]] | None = None,
        validation_table: ValidationHoleTable | None = None,
    ) -> 'RdfUncertaintyHoleTable':
        if not rdf_archetype_rows:
            z_i = np.zeros((0,), dtype=np.int32)
            z_f = np.zeros((0,), dtype=np.float32)
            z_s = np.zeros((0,), dtype='U1')
            return cls(z_i, z_f, z_f.copy(), z_s, z_i.copy(), z_s.copy(), z_f.copy(), z_f.copy(), z_f.copy(), z_f.copy(), z_f.copy(), z_f.copy(), z_f.copy(), z_i.copy(), z_f.copy(), z_f.copy(), z_s.copy())
        if validation_table is None:
            validation_table = ValidationHoleTable.from_rows(rdf_bootstrap_rows, rdf_bootstrap_support_rows)
        dyn_by_hole = {int(r.get('hole_id', 0)): r for r in (rdf_dynamics_rows or [])}
        rows = sorted(rdf_archetype_rows, key=lambda r: int(r.get('hole_id', 0)))
        n = len(rows)
        hole_id = np.empty((n,), dtype=np.int32)
        lattice_u = np.empty((n,), dtype=np.float32); lattice_u.fill(np.nan)
        lattice_v = np.empty((n,), dtype=np.float32); lattice_v.fill(np.nan)
        reticulum_zone = np.empty((n,), dtype=object)
        arche_id = np.empty((n,), dtype=np.int32)
        arche_label = np.empty((n,), dtype=object)
        support = np.empty((n,), dtype=np.float32); support.fill(np.nan)
        vel_ci = np.empty((n,), dtype=np.float32); vel_ci.fill(np.nan)
        front_ci = np.empty((n,), dtype=np.float32); front_ci.fill(np.nan)
        delta_ci = np.empty((n,), dtype=np.float32); delta_ci.fill(np.nan)
        accel = np.empty((n,), dtype=np.float32); accel.fill(np.nan)
        nonlin = np.empty((n,), dtype=np.float32); nonlin.fill(np.nan)
        uncert = np.empty((n,), dtype=np.float32); uncert.fill(np.nan)
        n_hotspots = np.empty((n,), dtype=np.int32)
        mean_dist = np.empty((n,), dtype=np.float32); mean_dist.fill(np.nan)
        min_dist = np.empty((n,), dtype=np.float32); min_dist.fill(np.nan)
        bucket = np.empty((n,), dtype=object)
        for i, row in enumerate(rows):
            hid = int(row.get('hole_id', 0))
            hole_id[i] = hid
            lattice_u[i] = np.float32(_safe_float(row.get('lattice_u', np.nan)))
            lattice_v[i] = np.float32(_safe_float(row.get('lattice_v', np.nan)))
            reticulum_zone[i] = str(zone_by_hole.get(hid, 'unknown'))
            arche_id[i] = _safe_int(row.get('rdf_archetype_canonical_id', 0))
            arche_label[i] = str(row.get('rdf_archetype_canonical_label', 'unknown'))
            dyn = dyn_by_hole.get(hid, {})
            support_i, vel_i, front_i, delta_i, _valid_vel, _valid_acc = validation_table.lookup(hid)
            support[i] = np.float32(support_i)
            vel_ci[i] = np.float32(vel_i)
            front_ci[i] = np.float32(front_i)
            delta_ci[i] = np.float32(delta_i)
            accel[i] = np.float32(_safe_float(dyn.get('rdf_front_acceleration_per_frame2', np.nan)))
            nonlin[i] = np.float32(_safe_float(dyn.get('rdf_front_nonlinearity_gain', np.nan)))
            nh, md, mind = hotspot_table.lookup(hid)
            n_hotspots[i] = nh
            if md is not None:
                mean_dist[i] = np.float32(md)
            if mind is not None:
                min_dist[i] = np.float32(mind)
            bucket[i] = _proximity_bucket(mind)
            pieces = []
            if np.isfinite(vel_ci[i]): pieces.append(abs(float(vel_ci[i])))
            if np.isfinite(front_ci[i]): pieces.append(abs(float(front_ci[i])))
            if np.isfinite(delta_ci[i]): pieces.append(abs(float(delta_ci[i])))
            if np.isfinite(support[i]): pieces.append(max(0.0, 1.0 - float(support[i])))
            if pieces:
                uncert[i] = np.float32(float(np.mean(np.asarray(pieces, dtype=float))))
        return cls(hole_id, lattice_u, lattice_v, reticulum_zone.astype('U32'), arche_id, arche_label.astype('U64'), support, vel_ci, front_ci, delta_ci, accel, nonlin, uncert, n_hotspots, mean_dist, min_dist, bucket.astype('U8'))


    def reticulum_columns(self) -> dict[str, list[Any]]:
        return {
            'hole_id': self.hole_id.astype(int).tolist(),
            'lattice_u': [None if not np.isfinite(v) else int(round(float(v))) for v in self.lattice_u.tolist()],
            'lattice_v': [None if not np.isfinite(v) else int(round(float(v))) for v in self.lattice_v.tolist()],
            'rdf_archetype_canonical_id': self.rdf_archetype_canonical_id.astype(int).tolist(),
            'rdf_archetype_canonical_label': self.rdf_archetype_canonical_label.astype(str).tolist(),
            'bootstrap_rdf_archetype_support_fraction': [None if not np.isfinite(v) else float(v) for v in self.bootstrap_support_fraction.tolist()],
            'bootstrap_mean_front_radius_ci_width': [None if not np.isfinite(v) else float(v) for v in self.bootstrap_mean_front_radius_ci_width.tolist()],
            'bootstrap_delta_front_radius_ci_width': [None if not np.isfinite(v) else float(v) for v in self.bootstrap_delta_front_radius_ci_width.tolist()],
            'bootstrap_front_velocity_ci_width': [None if not np.isfinite(v) else float(v) for v in self.bootstrap_front_velocity_ci_width.tolist()],
            'rdf_front_acceleration_per_frame2': [None if not np.isfinite(v) else float(v) for v in self.rdf_front_acceleration_per_frame2.tolist()],
            'rdf_front_nonlinearity_gain': [None if not np.isfinite(v) else float(v) for v in self.rdf_front_nonlinearity_gain.tolist()],
            'rdf_uncertainty_score': [None if not np.isfinite(v) else float(v) for v in self.rdf_uncertainty_score.tolist()],
        }

    def hotspot_comparison_columns(self) -> tuple[dict[str, list[Any]], dict[str, list[Any]]]:
        n = int(self.hole_id.size)
        out = {
            'hole_id': self.hole_id.astype(int).tolist(),
            'lattice_u': [None if not np.isfinite(v) else int(round(float(v))) for v in self.lattice_u.tolist()],
            'lattice_v': [None if not np.isfinite(v) else int(round(float(v))) for v in self.lattice_v.tolist()],
            'reticulum_zone': self.reticulum_zone.astype(str).tolist(),
            'rdf_archetype_canonical_id': self.rdf_archetype_canonical_id.astype(int).tolist(),
            'rdf_archetype_canonical_label': self.rdf_archetype_canonical_label.astype(str).tolist(),
            'n_hotspots': self.n_hotspots.astype(int).tolist(),
            'mean_hotspot_distance_px': [None if not np.isfinite(v) else float(v) for v in self.mean_hotspot_distance_px.tolist()],
            'min_hotspot_distance_px': [None if not np.isfinite(v) else float(v) for v in self.min_hotspot_distance_px.tolist()],
            'hotspot_proximity_bucket': self.hotspot_proximity_bucket.astype(str).tolist(),
            'bootstrap_rdf_archetype_support_fraction': [None if not np.isfinite(v) else float(v) for v in self.bootstrap_support_fraction.tolist()],
            'bootstrap_front_velocity_ci_width': [None if not np.isfinite(v) else float(v) for v in self.bootstrap_front_velocity_ci_width.tolist()],
            'bootstrap_mean_front_radius_ci_width': [None if not np.isfinite(v) else float(v) for v in self.bootstrap_mean_front_radius_ci_width.tolist()],
            'bootstrap_delta_front_radius_ci_width': [None if not np.isfinite(v) else float(v) for v in self.bootstrap_delta_front_radius_ci_width.tolist()],
        }
        acc: dict[tuple[str, str], dict[str, Any]] = {}
        for i in range(n):
            key = (str(self.reticulum_zone[i]), str(self.hotspot_proximity_bucket[i]))
            a = acc.setdefault(key, _acc_init(('bootstrap_rdf_archetype_support_fraction', 'bootstrap_front_velocity_ci_width', 'bootstrap_mean_front_radius_ci_width')))
            a['n'] += 1
            _acc_add(a, 'bootstrap_rdf_archetype_support_fraction', out['bootstrap_rdf_archetype_support_fraction'][i])
            _acc_add(a, 'bootstrap_front_velocity_ci_width', out['bootstrap_front_velocity_ci_width'][i])
            _acc_add(a, 'bootstrap_mean_front_radius_ci_width', out['bootstrap_mean_front_radius_ci_width'][i])
        group_rows = []
        for (zone, bucket), a in sorted(acc.items()):
            group_rows.append({
                'reticulum_zone': zone,
                'hotspot_proximity_bucket': bucket,
                'n_holes': int(a.get('n', 0)),
                'mean_bootstrap_support_fraction': _acc_mean(a, 'bootstrap_rdf_archetype_support_fraction'),
                'mean_bootstrap_front_velocity_ci_width': _acc_mean(a, 'bootstrap_front_velocity_ci_width'),
                'mean_bootstrap_front_radius_ci_width': _acc_mean(a, 'bootstrap_mean_front_radius_ci_width'),
            })
        group_cols = _columns_from_records(group_rows, [
            'reticulum_zone', 'hotspot_proximity_bucket', 'n_holes',
            'mean_bootstrap_support_fraction', 'mean_bootstrap_front_velocity_ci_width', 'mean_bootstrap_front_radius_ci_width'
        ])
        return out, group_cols


def build_rdf_uncertainty_reticulum_rows_table(table: RdfUncertaintyHoleTable) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for i in range(int(table.hole_id.size)):
        out.append({
            'hole_id': int(table.hole_id[i]),
            'lattice_u': None if not np.isfinite(table.lattice_u[i]) else int(round(float(table.lattice_u[i]))),
            'lattice_v': None if not np.isfinite(table.lattice_v[i]) else int(round(float(table.lattice_v[i]))),
            'rdf_archetype_canonical_id': int(table.rdf_archetype_canonical_id[i]),
            'rdf_archetype_canonical_label': str(table.rdf_archetype_canonical_label[i]),
            'bootstrap_rdf_archetype_support_fraction': None if not np.isfinite(table.bootstrap_support_fraction[i]) else float(table.bootstrap_support_fraction[i]),
            'bootstrap_mean_front_radius_ci_width': None if not np.isfinite(table.bootstrap_mean_front_radius_ci_width[i]) else float(table.bootstrap_mean_front_radius_ci_width[i]),
            'bootstrap_delta_front_radius_ci_width': None if not np.isfinite(table.bootstrap_delta_front_radius_ci_width[i]) else float(table.bootstrap_delta_front_radius_ci_width[i]),
            'bootstrap_front_velocity_ci_width': None if not np.isfinite(table.bootstrap_front_velocity_ci_width[i]) else float(table.bootstrap_front_velocity_ci_width[i]),
            'rdf_front_acceleration_per_frame2': None if not np.isfinite(table.rdf_front_acceleration_per_frame2[i]) else float(table.rdf_front_acceleration_per_frame2[i]),
            'rdf_front_nonlinearity_gain': None if not np.isfinite(table.rdf_front_nonlinearity_gain[i]) else float(table.rdf_front_nonlinearity_gain[i]),
            'rdf_uncertainty_score': None if not np.isfinite(table.rdf_uncertainty_score[i]) else float(table.rdf_uncertainty_score[i]),
        })
    return out


def build_rdf_uncertainty_hotspot_comparison_table(table: RdfUncertaintyHoleTable) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    out: list[dict[str, Any]] = []
    group_acc: dict[tuple[str, str], dict[str, Any]] = {}
    for i in range(int(table.hole_id.size)):
        out_row = {
            'hole_id': int(table.hole_id[i]),
            'lattice_u': None if not np.isfinite(table.lattice_u[i]) else int(round(float(table.lattice_u[i]))),
            'lattice_v': None if not np.isfinite(table.lattice_v[i]) else int(round(float(table.lattice_v[i]))),
            'reticulum_zone': str(table.reticulum_zone[i]),
            'rdf_archetype_canonical_id': int(table.rdf_archetype_canonical_id[i]),
            'rdf_archetype_canonical_label': str(table.rdf_archetype_canonical_label[i]),
            'n_hotspots': int(table.n_hotspots[i]),
            'mean_hotspot_distance_px': None if not np.isfinite(table.mean_hotspot_distance_px[i]) else float(table.mean_hotspot_distance_px[i]),
            'min_hotspot_distance_px': None if not np.isfinite(table.min_hotspot_distance_px[i]) else float(table.min_hotspot_distance_px[i]),
            'hotspot_proximity_bucket': str(table.hotspot_proximity_bucket[i]),
            'bootstrap_rdf_archetype_support_fraction': None if not np.isfinite(table.bootstrap_support_fraction[i]) else float(table.bootstrap_support_fraction[i]),
            'bootstrap_front_velocity_ci_width': None if not np.isfinite(table.bootstrap_front_velocity_ci_width[i]) else float(table.bootstrap_front_velocity_ci_width[i]),
            'bootstrap_mean_front_radius_ci_width': None if not np.isfinite(table.bootstrap_mean_front_radius_ci_width[i]) else float(table.bootstrap_mean_front_radius_ci_width[i]),
            'bootstrap_delta_front_radius_ci_width': None if not np.isfinite(table.bootstrap_delta_front_radius_ci_width[i]) else float(table.bootstrap_delta_front_radius_ci_width[i]),
        }
        out.append(out_row)
        key = (str(table.reticulum_zone[i]), str(table.hotspot_proximity_bucket[i]))
        acc = group_acc.setdefault(key, _acc_init(('bootstrap_rdf_archetype_support_fraction', 'bootstrap_front_velocity_ci_width', 'bootstrap_mean_front_radius_ci_width')))
        acc['n'] += 1
        _acc_add(acc, 'bootstrap_rdf_archetype_support_fraction', out_row.get('bootstrap_rdf_archetype_support_fraction'))
        _acc_add(acc, 'bootstrap_front_velocity_ci_width', out_row.get('bootstrap_front_velocity_ci_width'))
        _acc_add(acc, 'bootstrap_mean_front_radius_ci_width', out_row.get('bootstrap_mean_front_radius_ci_width'))
    group_rows: list[dict[str, Any]] = []
    for (zone, bucket), acc in sorted(group_acc.items()):
        group_rows.append({
            'reticulum_zone': zone,
            'hotspot_proximity_bucket': bucket,
            'n_holes': int(acc.get('n', 0)),
            'mean_bootstrap_support_fraction': _acc_mean(acc, 'bootstrap_rdf_archetype_support_fraction'),
            'mean_bootstrap_front_velocity_ci_width': _acc_mean(acc, 'bootstrap_front_velocity_ci_width'),
            'mean_bootstrap_front_radius_ci_width': _acc_mean(acc, 'bootstrap_mean_front_radius_ci_width'),
        })
    return out, group_rows
