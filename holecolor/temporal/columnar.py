from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from holecolor.radial.columnar import RadialRowTable, _group_bounds


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


@dataclass(slots=True)
class PhenotypeTable:
    hole_id: np.ndarray
    lattice_u: np.ndarray
    lattice_v: np.ndarray
    phenotype_label: np.ndarray
    semantic_label: np.ndarray
    canonical_id: np.ndarray
    canonical_label: np.ndarray

    @classmethod
    def from_rows(cls, rows: list[dict[str, Any]]) -> 'PhenotypeTable':
        n = len(rows)
        hole_id = np.empty((n,), dtype=np.int32)
        lattice_u = np.empty((n,), dtype=np.float32); lattice_u.fill(np.nan)
        lattice_v = np.empty((n,), dtype=np.float32); lattice_v.fill(np.nan)
        phenotype_label = np.empty((n,), dtype=object)
        semantic_label = np.empty((n,), dtype=object)
        canonical_id = np.empty((n,), dtype=np.int32)
        canonical_label = np.empty((n,), dtype=object)
        for i, row in enumerate(rows):
            hole_id[i] = _safe_int(row.get('hole_id', 0))
            lattice_u[i] = np.float32(_safe_float(row.get('lattice_u', np.nan)))
            lattice_v[i] = np.float32(_safe_float(row.get('lattice_v', np.nan)))
            phenotype_label[i] = str(row.get('phenotype_label', 'unknown'))
            semantic_label[i] = str(row.get('semantic_label', phenotype_label[i]))
            canonical_id[i] = _safe_int(row.get('canonical_id', -1), -1)
            canonical_label[i] = str(row.get('canonical_label', phenotype_label[i]))
        return cls(
            hole_id=hole_id,
            lattice_u=lattice_u,
            lattice_v=lattice_v,
            phenotype_label=phenotype_label.astype('U64'),
            semantic_label=semantic_label.astype('U64'),
            canonical_id=canonical_id,
            canonical_label=canonical_label.astype('U64'),
        )

    def label_by_hole(self) -> dict[int, str]:
        return {int(self.hole_id[i]): str(self.phenotype_label[i]) for i in range(int(self.hole_id.size))}

    def count_by_label(self) -> dict[str, int]:
        if self.hole_id.size == 0:
            return {}
        labels, counts = np.unique(self.phenotype_label, return_counts=True)
        return {str(lbl): int(cnt) for lbl, cnt in zip(labels.tolist(), counts.tolist())}


def build_phenotype_neighbor_and_smoothness_table(
    table: PhenotypeTable,
    radius: int = 2,
) -> tuple[float | None, list[dict[str, Any]], float | None, list[dict[str, Any]], dict[str, Any]]:
    if table.hole_id.size == 0:
        return None, [], None, [], {"mean_spatial_smoothness": None, "radius": int(radius), "mean_neighbor_count": 0.0}
    by_uv: dict[tuple[int, int], int] = {}
    for i in range(int(table.hole_id.size)):
        u = table.lattice_u[i]
        v = table.lattice_v[i]
        if np.isfinite(u) and np.isfinite(v):
            by_uv[(int(round(float(u))), int(round(float(v))))] = i
    if not by_uv:
        return None, [], None, [], {"mean_spatial_smoothness": None, "radius": int(radius), "mean_neighbor_count": 0.0}

    neighbor_rows: list[dict[str, Any]] = []
    smooth_rows: list[dict[str, Any]] = []
    seen: set[tuple[tuple[int, int], tuple[int, int]]] = set()
    pair_same: list[bool] = []
    smoothness_vals: list[float] = []
    neighbor_counts: list[int] = []

    for (u, v), idx in by_uv.items():
        label = str(table.phenotype_label[idx])
        weights: list[float] = []
        same_weights: list[float] = []
        local_neighbor_count = 0
        for du in range(-radius, radius + 1):
            for dv in range(-radius, radius + 1):
                if du == 0 and dv == 0:
                    continue
                dist = abs(du) + abs(dv)
                if dist == 0 or dist > radius:
                    continue
                nb = (u + du, v + dv)
                nb_idx = by_uv.get(nb)
                if nb_idx is None:
                    continue
                local_neighbor_count += 1
                same = str(table.phenotype_label[nb_idx]) == label
                w = 1.0 / float(dist)
                weights.append(w)
                same_weights.append(w if same else 0.0)
                # only cardinal neighbors contribute to pair table, matching old function
                if dist == 1 and ((abs(du) == 1 and dv == 0) or (abs(dv) == 1 and du == 0)):
                    key = tuple(sorted(((u, v), nb)))
                    if key not in seen:
                        seen.add(key)
                        neighbor_rows.append({
                            'hole_id_a': int(table.hole_id[idx]),
                            'hole_id_b': int(table.hole_id[nb_idx]),
                            'u_a': int(u),
                            'v_a': int(v),
                            'u_b': int(nb[0]),
                            'v_b': int(nb[1]),
                            'same_label': bool(same),
                            'label_a': label,
                            'label_b': str(table.phenotype_label[nb_idx]),
                        })
                        pair_same.append(bool(same))
        total_w = float(np.sum(weights)) if weights else 0.0
        smooth = float(np.sum(same_weights) / total_w) if total_w > 0 else float('nan')
        neighbor_counts.append(local_neighbor_count)
        if np.isfinite(smooth):
            smoothness_vals.append(smooth)
        smooth_rows.append({
            'hole_id': int(table.hole_id[idx]),
            'lattice_u': int(u),
            'lattice_v': int(v),
            'phenotype_label': label,
            'spatial_smoothness': None if not np.isfinite(smooth) else smooth,
            'neighbor_count': int(local_neighbor_count),
            'radius': int(radius),
        })
    coherence_fraction = float(np.mean(pair_same)) if pair_same else None
    spatial_smoothness = float(np.mean(smoothness_vals)) if smoothness_vals else None
    summary = {
        'mean_spatial_smoothness': spatial_smoothness,
        'min_spatial_smoothness': float(np.min(smoothness_vals)) if smoothness_vals else None,
        'radius': int(radius),
        'mean_neighbor_count': float(np.mean(neighbor_counts)) if neighbor_counts else 0.0,
    }
    return coherence_fraction, neighbor_rows, spatial_smoothness, smooth_rows, summary


def build_phenotype_archetype_rows_table(
    radial_table: RadialRowTable,
    phenotype_table: PhenotypeTable,
) -> list[dict[str, Any]]:
    if radial_table.hole_id.size == 0 or phenotype_table.hole_id.size == 0:
        return []
    label_by_hole = phenotype_table.label_by_hole()
    labels = np.empty((radial_table.hole_id.size,), dtype=object)
    keep = np.zeros((radial_table.hole_id.size,), dtype=bool)
    for i in range(int(radial_table.hole_id.size)):
        hid = int(radial_table.hole_id[i])
        lbl = label_by_hole.get(hid)
        if lbl is not None:
            labels[i] = lbl
            keep[i] = True
        else:
            labels[i] = ''
    if not np.any(keep):
        return []
    idx = np.flatnonzero(keep)
    labels_keep = labels[idx].astype('U64')
    frames = radial_table.frame_id[idx]
    annuli = radial_table.annulus_id[idx]
    vals = radial_table.descriptor_value[idx].astype(float, copy=False)
    order = np.lexsort((annuli, frames, labels_keep)).astype(np.int32)
    labels_o = labels_keep[order]
    frames_o = frames[order]
    annuli_o = annuli[order]
    vals_o = vals[order]
    starts, ends = _group_bounds(labels_o, frames_o, annuli_o)
    out: list[dict[str, Any]] = []
    for s, e in zip(starts, ends):
        group_vals = vals_o[s:e]
        finite = group_vals[np.isfinite(group_vals)]
        out.append({
            'phenotype_label': str(labels_o[s]),
            'frame_id': int(frames_o[s]),
            'annulus_id': int(annuli_o[s]),
            'mean_descriptor': float(finite.mean()) if finite.size else None,
            'std_descriptor': float(finite.std()) if finite.size else None,
            'n_holes': int(finite.size),
        })
    return out


@dataclass(slots=True)
class TemporalValidationSummary:
    valid_onset_fraction: float
    valid_peak_fraction: float
    lag_monotonic_fraction: float
    negative_lag_fraction: float
    phenotype_stability_fraction: float
    phenotype_canonical_agreement: float
    phenotype_neighbor_fraction: float
    phenotype_spatial_fraction: float

    @classmethod
    def from_rows(
        cls,
        per_hole_summary_rows: list[dict[str, Any]],
        propagation_rows: list[dict[str, Any]],
        stability_summary: dict[str, Any] | None,
        coherence_fraction: float | None,
        spatial_smoothness: float | None,
        min_valid_annuli_onset: int,
        min_valid_annuli_peak: int,
    ) -> 'TemporalValidationSummary':
        if per_hole_summary_rows:
            onset_ok = np.asarray([int(r.get('n_valid_annuli_onset', 0)) >= int(min_valid_annuli_onset) for r in per_hole_summary_rows], dtype=bool)
            peak_ok = np.asarray([int(r.get('n_valid_annuli_peak', 0)) >= int(min_valid_annuli_peak) for r in per_hole_summary_rows], dtype=bool)
            mono_ok = np.asarray([(bool(r.get('monotonic_onset')) and bool(r.get('monotonic_peak'))) for r in per_hole_summary_rows], dtype=bool)
            valid_onset_fraction = float(np.mean(onset_ok))
            valid_peak_fraction = float(np.mean(peak_ok))
            lag_monotonic_fraction = float(np.mean(mono_ok))
        else:
            valid_onset_fraction = 0.0
            valid_peak_fraction = 0.0
            lag_monotonic_fraction = 0.0
        if propagation_rows:
            neg = np.asarray([bool(r.get('negative_lag_flag')) for r in propagation_rows], dtype=bool)
            negative_lag_fraction = float(np.mean(neg))
        else:
            negative_lag_fraction = 0.0
        stability_summary = dict(stability_summary or {})
        phenotype_stability_fraction = float(stability_summary.get('mean_stability_fraction')) if stability_summary.get('mean_stability_fraction') is not None else 0.0
        phenotype_canonical_agreement = float(stability_summary.get('canonical_agreement_fraction')) if stability_summary.get('canonical_agreement_fraction') is not None else 0.0
        phenotype_neighbor_fraction = float(coherence_fraction) if coherence_fraction is not None else 0.0
        phenotype_spatial_fraction = float(spatial_smoothness) if spatial_smoothness is not None else 0.0
        return cls(
            valid_onset_fraction=valid_onset_fraction,
            valid_peak_fraction=valid_peak_fraction,
            lag_monotonic_fraction=lag_monotonic_fraction,
            negative_lag_fraction=negative_lag_fraction,
            phenotype_stability_fraction=phenotype_stability_fraction,
            phenotype_canonical_agreement=phenotype_canonical_agreement,
            phenotype_neighbor_fraction=phenotype_neighbor_fraction,
            phenotype_spatial_fraction=phenotype_spatial_fraction,
        )
