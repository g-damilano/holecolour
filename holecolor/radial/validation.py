from __future__ import annotations

from dataclasses import asdict
from typing import Any

import numpy as np

from holecolor.config.schema import ParallelConfig, PipelineConfig
from holecolor.core.parallel import iter_with_progress, parallel_map
from holecolor.core.types import FrameRecord, HoleGeometry
from holecolor.descriptors.color_spaces import descriptor_image, rgb_to_hsv
from holecolor.masks.terraces import make_nonoverlapping_hole_terraces
from holecolor.radial.curves import compute_all_radial_curves
from holecolor.radial.rdf import build_per_hole_rdf_archetypes, build_per_hole_rdf_evolution, canonicalize_rdf_archetypes




def _safe_float(value: Any, default: float = np.nan) -> float:
    try:
        return float(value) if value is not None else float(default)
    except Exception:
        return float(default)


def _build_validation_parallel_cfg(cfg: PipelineConfig) -> ParallelConfig:
    base = cfg.parallel
    if not base.enabled or base.backend == 'none':
        return ParallelConfig(
            enabled=False,
            show_progress=base.show_progress,
            progress_leave=base.progress_leave,
            progress_mininterval_s=base.progress_mininterval_s,
            status_heartbeat_interval_s=base.status_heartbeat_interval_s,
        )
    backend = 'thread' if base.backend in ('auto', 'process') else base.backend
    return ParallelConfig(
        enabled=True,
        backend=backend,
        max_workers=base.max_workers,
        min_parallel_tasks=1,
        chunksize=base.chunksize,
        show_progress=base.show_progress,
        progress_leave=base.progress_leave,
        progress_mininterval_s=base.progress_mininterval_s,
        status_heartbeat_interval_s=base.status_heartbeat_interval_s,
        opencv_threads_per_worker=base.opencv_threads_per_worker,
    )


def _run_single_radial_sweep(task: dict[str, Any]) -> dict[str, Any]:
    stabilized = task['stabilized']
    holes_by_frame = task['holes_by_frame']
    chosen_descriptor = task['chosen_descriptor']
    cfg_data = task['cfg'] if isinstance(task['cfg'], dict) else asdict(task['cfg'])
    brightness = float(task['brightness'])
    radius_scale = float(task['radius_scale'])
    shape = stabilized[0].image.shape[:2]
    radial_rows: list[dict[str, Any]] = []
    for frame in stabilized:
        holes = _scale_holes(holes_by_frame[frame.frame_id], radius_scale)
        terraces_by_hole = make_nonoverlapping_hole_terraces(shape, holes, int(cfg_data['masks']['n_terraces']))
        img = _brightness_scale(frame.image, brightness)
        hsv = rgb_to_hsv(img)
        desc_img = descriptor_image(img, hsv, chosen_descriptor)
        curves = compute_all_radial_curves(frame.frame_id, desc_img, terraces_by_hole, chosen_descriptor)
        for curve in curves:
            for annulus_id, value in enumerate(curve.terrace_values):
                radial_rows.append(
                    {
                        'frame_id': int(frame.frame_id),
                        'hole_id': int(curve.hole_id),
                        'annulus_id': int(annulus_id),
                        'descriptor_value': float(value),
                    }
                )
    agg_rows = aggregate_radial_rows(radial_rows)
    frame_rows, summary = summarize_radial_evolution(agg_rows)
    return {
        'brightness_factor': brightness,
        'radius_scale': radius_scale,
        'aggregate_rows': agg_rows,
        'frame_rows': frame_rows,
        'summary': summary,
    }


def _run_single_rdf_sweep(task: dict[str, Any]) -> dict[str, Any]:
    stabilized = task['stabilized']
    holes_by_frame = task['holes_by_frame']
    chosen_descriptor = task['chosen_descriptor']
    cfg_data = task['cfg'] if isinstance(task['cfg'], dict) else asdict(task['cfg'])
    brightness = float(task['brightness'])
    radius_scale = float(task['radius_scale'])
    shape = stabilized[0].image.shape[:2]
    radial_rows: list[dict[str, Any]] = []
    for frame in stabilized:
        holes = _scale_holes(holes_by_frame[frame.frame_id], radius_scale)
        terraces_by_hole = make_nonoverlapping_hole_terraces(shape, holes, int(cfg_data['masks']['n_terraces']))
        img = _brightness_scale(frame.image, brightness)
        hsv = rgb_to_hsv(img)
        desc_img = descriptor_image(img, hsv, chosen_descriptor)
        curves = compute_all_radial_curves(frame.frame_id, desc_img, terraces_by_hole, chosen_descriptor)
        for curve in curves:
            for annulus_id, value in enumerate(curve.terrace_values):
                radial_rows.append({
                    'frame_id': int(frame.frame_id),
                    'hole_id': int(curve.hole_id),
                    'annulus_id': int(annulus_id),
                    'descriptor_value': float(value),
                })
    _rdf_rows, rdf_frame_rows, _vel = build_per_hole_rdf_evolution(radial_rows)
    sweep_arche, sweep_centroids = build_per_hole_rdf_archetypes(rdf_frame_rows, k=int(cfg_data['radial']['archetype_k']))
    sweep_arche, _ = canonicalize_rdf_archetypes(sweep_arche, sweep_centroids)
    return {
        'brightness_factor': brightness,
        'radius_scale': radius_scale,
        'sweep_arche': sweep_arche,
    }

def aggregate_radial_rows(radial_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not radial_rows:
        return []
    buckets: dict[tuple[int, int], list[float]] = {}
    for row in radial_rows:
        key = (int(row.get("frame_id", 0)), int(row.get("annulus_id", 0)))
        buckets.setdefault(key, []).append(_safe_float(row.get("descriptor_value", np.nan)))
    out: list[dict[str, Any]] = []
    for (frame_id, annulus_id), vals in sorted(buckets.items()):
        arr = np.asarray(vals, dtype=float)
        finite = arr[np.isfinite(arr)]
        out.append(
            {
                "frame_id": int(frame_id),
                "annulus_id": int(annulus_id),
                "mean_descriptor": float(finite.mean()) if finite.size else None,
                "std_descriptor": float(finite.std()) if finite.size else None,
                "n_holes": int(finite.size),
            }
        )
    return out


def summarize_radial_evolution(aggregate_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not aggregate_rows:
        return [], {
            "start_center_of_mass": None,
            "end_center_of_mass": None,
            "delta_center_of_mass": None,
            "start_peak_annulus": None,
            "end_peak_annulus": None,
            "conclusion_label": "unknown",
            "n_frames": 0,
        }
    by_frame: dict[int, list[dict[str, Any]]] = {}
    for row in aggregate_rows:
        by_frame.setdefault(int(row["frame_id"]), []).append(row)
    frame_rows: list[dict[str, Any]] = []
    for frame_id in sorted(by_frame):
        rows = sorted(by_frame[frame_id], key=lambda r: int(r["annulus_id"]))
        ann = np.asarray([int(r["annulus_id"]) for r in rows], dtype=float)
        vals = np.asarray([float(r["mean_descriptor"]) if r.get("mean_descriptor") is not None else np.nan for r in rows], dtype=float)
        finite = np.isfinite(vals)
        if not np.any(finite):
            frame_rows.append({"frame_id": int(frame_id), "center_of_mass_annulus": None, "peak_annulus": None, "inner_minus_outer": None, "n_valid_annuli": 0})
            continue
        ann = ann[finite]
        vals = vals[finite]
        shifted = vals - float(np.nanmin(vals)) + 1e-6
        com = float(np.sum(ann * shifted) / np.sum(shifted)) if np.sum(shifted) > 0 else float(np.nanmean(ann))
        peak_idx = int(ann[int(np.nanargmax(vals))]) if vals.size else None
        inner_minus_outer = float(vals[0] - vals[-1]) if vals.size >= 2 else None
        frame_rows.append(
            {
                "frame_id": int(frame_id),
                "center_of_mass_annulus": com,
                "peak_annulus": peak_idx,
                "inner_minus_outer": inner_minus_outer,
                "n_valid_annuli": int(vals.size),
            }
        )
    finite_com_rows = [r for r in frame_rows if r.get("center_of_mass_annulus") is not None]
    if not finite_com_rows:
        return frame_rows, {
            "start_center_of_mass": None,
            "end_center_of_mass": None,
            "delta_center_of_mass": None,
            "start_peak_annulus": None,
            "end_peak_annulus": None,
            "conclusion_label": "unknown",
            "n_frames": len(frame_rows),
        }
    start = float(finite_com_rows[0]["center_of_mass_annulus"])
    end = float(finite_com_rows[-1]["center_of_mass_annulus"])
    delta = end - start
    thresh = 0.35
    if delta > thresh:
        conclusion = "outward_shift"
    elif delta < -thresh:
        conclusion = "inward_shift"
    else:
        conclusion = "stable"
    summary = {
        "start_center_of_mass": start,
        "end_center_of_mass": end,
        "delta_center_of_mass": float(delta),
        "start_peak_annulus": finite_com_rows[0].get("peak_annulus"),
        "end_peak_annulus": finite_com_rows[-1].get("peak_annulus"),
        "conclusion_label": conclusion,
        "n_frames": int(len(frame_rows)),
    }
    return frame_rows, summary


def _brightness_scale(image: np.ndarray, factor: float) -> np.ndarray:
    out = np.clip(image.astype(np.float32) * float(factor), 0, 255)
    return out.astype(np.uint8)


def _scale_holes(holes: list[HoleGeometry], radius_scale: float) -> list[HoleGeometry]:
    out: list[HoleGeometry] = []
    for hole in holes:
        out.append(
            HoleGeometry(
                hole_id=hole.hole_id,
                x=hole.x,
                y=hole.y,
                radius_inner_px=max(1.0, float(hole.radius_inner_px) * float(radius_scale)),
                radius_outer_px=max(1.0, float(hole.radius_outer_px) * float(radius_scale)),
                confidence=hole.confidence,
            )
        )
    return out


def _profile_correlation(base_rows: list[dict[str, Any]], sweep_rows: list[dict[str, Any]]) -> float | None:
    base = {(int(r["frame_id"]), int(r["annulus_id"])): r for r in base_rows}
    sweep = {(int(r["frame_id"]), int(r["annulus_id"])): r for r in sweep_rows}
    keys = sorted(set(base) & set(sweep))
    if not keys:
        return None
    frame_corrs: list[float] = []
    by_frame = sorted({k[0] for k in keys})
    for frame_id in by_frame:
        fkeys = [k for k in keys if k[0] == frame_id]
        x = np.asarray([float(base[k].get("mean_descriptor", np.nan)) for k in fkeys], dtype=float)
        y = np.asarray([float(sweep[k].get("mean_descriptor", np.nan)) for k in fkeys], dtype=float)
        finite = np.isfinite(x) & np.isfinite(y)
        if np.sum(finite) < 2:
            continue
        x = x[finite]
        y = y[finite]
        if np.allclose(np.std(x), 0.0) or np.allclose(np.std(y), 0.0):
            corr = 1.0 if np.allclose(x, y) else 0.0
        else:
            corr = float(np.corrcoef(x, y)[0, 1])
        if np.isfinite(corr):
            frame_corrs.append(corr)
    return float(np.mean(frame_corrs)) if frame_corrs else None


def run_radial_perturbation_sweeps(
    stabilized: list[FrameRecord],
    holes_by_frame: dict[int, list[HoleGeometry]],
    chosen_descriptor: str,
    cfg: PipelineConfig,
    base_aggregate_rows: list[dict[str, Any]],
    base_frame_rows: list[dict[str, Any]],
    base_summary: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    if not cfg.validation.enabled or not stabilized:
        return [], [], {"base_conclusion_label": base_summary.get("conclusion_label"), "n_sweeps": 0, "conclusion_agreement_fraction": None, "mean_profile_correlation": None}
    sweep_rows: list[dict[str, Any]] = []
    drift_rows: list[dict[str, Any]] = []
    agreement_flags: list[bool] = []
    corr_vals: list[float] = []
    base_com = {int(r["frame_id"]): r.get("center_of_mass_annulus") for r in base_frame_rows}
    base_peak = {int(r["frame_id"]): r.get("peak_annulus") for r in base_frame_rows}
    base_label = str(base_summary.get("conclusion_label", "unknown"))
    sweep_params = [(float(b), float(r)) for b in cfg.validation.brightness_factors for r in cfg.validation.radius_scale_factors]
    tasks = [
        {
            'stabilized': stabilized,
            'holes_by_frame': holes_by_frame,
            'chosen_descriptor': chosen_descriptor,
            'cfg': asdict(cfg),
            'brightness': brightness,
            'radius_scale': radius_scale,
        }
        for brightness, radius_scale in sweep_params
    ]
    parallel_cfg = _build_validation_parallel_cfg(cfg)
    sweep_out = parallel_map(_run_single_radial_sweep, tasks, parallel_cfg, desc='Radial perturbation sweeps')
    for sweep_id, sweep_result in enumerate(sweep_out):
        brightness = float(sweep_result['brightness_factor'])
        radius_scale = float(sweep_result['radius_scale'])
        agg_rows = sweep_result['aggregate_rows']
        frame_rows = sweep_result['frame_rows']
        summary = sweep_result['summary']
        profile_corr = _profile_correlation(base_aggregate_rows, agg_rows)
        conclusion_match = str(summary.get('conclusion_label', 'unknown')) == base_label
        if profile_corr is not None:
            corr_vals.append(float(profile_corr))
        if not (float(brightness) == 1.0 and float(radius_scale) == 1.0):
            agreement_flags.append(bool(conclusion_match))
        sweep_rows.append(
            {
                'sweep_id': int(sweep_id),
                'brightness_factor': float(brightness),
                'radius_scale': float(radius_scale),
                'is_base': bool(float(brightness) == 1.0 and float(radius_scale) == 1.0),
                'conclusion_label': summary.get('conclusion_label'),
                'matches_base_conclusion': bool(conclusion_match),
                'mean_profile_correlation': profile_corr,
                'start_center_of_mass': summary.get('start_center_of_mass'),
                'end_center_of_mass': summary.get('end_center_of_mass'),
                'delta_center_of_mass': summary.get('delta_center_of_mass'),
                'start_peak_annulus': summary.get('start_peak_annulus'),
                'end_peak_annulus': summary.get('end_peak_annulus'),
            }
        )
        for row in frame_rows:
            fid = int(row['frame_id'])
            drift_rows.append(
                {
                    'sweep_id': int(sweep_id),
                    'brightness_factor': float(brightness),
                    'radius_scale': float(radius_scale),
                    'frame_id': fid,
                    'base_center_of_mass_annulus': base_com.get(fid),
                    'sweep_center_of_mass_annulus': row.get('center_of_mass_annulus'),
                    'center_of_mass_delta': None if base_com.get(fid) is None or row.get('center_of_mass_annulus') is None else float(row.get('center_of_mass_annulus') - base_com.get(fid)),
                    'base_peak_annulus': base_peak.get(fid),
                    'sweep_peak_annulus': row.get('peak_annulus'),
                    'peak_delta': None if base_peak.get(fid) is None or row.get('peak_annulus') is None else int(row.get('peak_annulus') - base_peak.get(fid)),
                }
            )
    consistency = {
        'base_conclusion_label': base_label,
        'n_sweeps': int(len(sweep_rows)),
        'n_nonbase_sweeps': int(sum(not bool(r['is_base']) for r in sweep_rows)),
        'conclusion_agreement_fraction': float(np.mean(agreement_flags)) if agreement_flags else 1.0,
        'mean_profile_correlation': float(np.mean(corr_vals)) if corr_vals else None,
    }
    return sweep_rows, drift_rows, consistency


def run_rdf_archetype_perturbation_sweeps(
    stabilized: list[FrameRecord],
    holes_by_frame: dict[int, list[HoleGeometry]],
    chosen_descriptor: str,
    cfg: PipelineConfig,
    base_rdf_archetype_rows: list[dict[str, Any]],
    base_centroids: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    if not cfg.validation.enabled or not stabilized or not base_rdf_archetype_rows:
        return [], [], {"mean_rdf_archetype_stability": None, "n_sweeps": 0}
    base_rows, _ = canonicalize_rdf_archetypes(base_rdf_archetype_rows, base_centroids)
    base_by_hole = {int(r.get('hole_id', 0)): int(r.get('rdf_archetype_canonical_id', 0)) for r in base_rows}
    hole_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    stability_vals: list[float] = []
    sweep_params = [(float(b), float(r)) for b in cfg.validation.brightness_factors for r in cfg.validation.radius_scale_factors]
    tasks = [
        {
            'stabilized': stabilized,
            'holes_by_frame': holes_by_frame,
            'chosen_descriptor': chosen_descriptor,
            'cfg': asdict(cfg),
            'brightness': brightness,
            'radius_scale': radius_scale,
        }
        for brightness, radius_scale in sweep_params
    ]
    parallel_cfg = _build_validation_parallel_cfg(cfg)
    sweep_out = parallel_map(_run_single_rdf_sweep, tasks, parallel_cfg, desc='RDF perturbation sweeps')
    for sweep_id, sweep_result in enumerate(sweep_out):
        brightness = float(sweep_result['brightness_factor'])
        radius_scale = float(sweep_result['radius_scale'])
        sweep_arche = sweep_result['sweep_arche']
        sweep_by_hole = {int(r.get('hole_id', 0)): int(r.get('rdf_archetype_canonical_id', 0)) for r in sweep_arche}
        matches: list[float] = []
        for hole_id in sorted(set(base_by_hole) | set(sweep_by_hole)):
            base_label = base_by_hole.get(hole_id)
            sweep_label = sweep_by_hole.get(hole_id)
            match = base_label is not None and sweep_label is not None and int(base_label) == int(sweep_label)
            if base_label is not None and sweep_label is not None:
                matches.append(float(match))
            hole_rows.append({
                'sweep_id': int(sweep_id),
                'brightness_factor': float(brightness),
                'radius_scale': float(radius_scale),
                'hole_id': int(hole_id),
                'base_rdf_archetype_canonical_id': base_label,
                'sweep_rdf_archetype_canonical_id': sweep_label,
                'rdf_archetype_match': None if base_label is None or sweep_label is None else bool(match),
            })
        stab = float(np.mean(matches)) if matches else np.nan
        if np.isfinite(stab):
            stability_vals.append(stab)
        summary_rows.append({
            'sweep_id': int(sweep_id),
            'brightness_factor': float(brightness),
            'radius_scale': float(radius_scale),
            'rdf_archetype_stability_fraction': None if not np.isfinite(stab) else stab,
            'n_holes_compared': int(len(matches)),
            'is_base': bool(float(brightness) == 1.0 and float(radius_scale) == 1.0),
        })
    meta = {
        'mean_rdf_archetype_stability': float(np.nanmean(stability_vals)) if stability_vals else None,
        'n_sweeps': int(len(summary_rows)),
    }
    return hole_rows, summary_rows, meta
