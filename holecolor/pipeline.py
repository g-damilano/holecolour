from __future__ import annotations

from dataclasses import asdict, replace
import csv
import hashlib
import html
import json
import pickle
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from holecolor.audit.frame_audit import audit_sequence
from holecolor.config.schema import PipelineConfig, ParallelConfig
from holecolor.core.parallel import iter_with_progress, parallel_map, prefer_thread_for_image_tasks
from holecolor.core.paths import ensure_dir, ensure_subdirs
from holecolor.core.status import PipelineProgress
from holecolor.core.types import AuditRecord, FrameRecord, HoleCandidate, HoleGeometry, LatticeModel, PhotometryScore, TransformRecord
from holecolor.descriptors.color_spaces import descriptor_image, rgb_to_hsv
from holecolor.extensions.wafer_nonhole_colour import (
    WaferNonholeColourBundle,
    build_wafer_nonhole_colour_bundle,
    build_wafer_nonhole_colour_bundle_from_samples,
    enrich_global_matrix_rows,
    enrich_local_compartment_rows,
    support_mask_from_debug,
    extract_wafer_nonhole_region,
    write_wafer_nonhole_colour_artifacts,
    write_wafer_nonhole_cluster_videos,
)
from holecolor.extensions.radial_cluster_average_hole import warmup_radial_cluster_numba, write_radial_cluster_average_hole_artifacts
from holecolor.descriptors.region_stats import compute_region_stats, compute_region_stats_from_coords, compute_region_stats_from_region, mean_from_region
from holecolor.descriptors.separability import rank_descriptors
from holecolor.geometry.candidates import detect_dark_hole_candidates, detect_stable_grid_hole_candidates
from holecolor.geometry.completeness import filter_complete_holes_and_terraces
from holecolor.geometry.exact_sequence import detect_exact_wafer_holes_sequence_full, warmup_exact_sequence_numba
from holecolor.geometry.conic_refine import conic_to_hole_geometry, refine_candidate_with_conic
from holecolor.geometry.lattice_fit import assign_lattice_indices, estimate_lattice_basis
from holecolor.holegrid.model import HoleGridBundle
from holecolor.geometry.overlays import draw_candidates
from holecolor.geometry.tracking import propagate_geometry_to_frame, smooth_hole_trajectories
from holecolor.hotspots.detect import detect_hotspots
from holecolor.hotspots.score import hotspot_score_map
from holecolor.hotspots.track import link_hotspots, summarize_tracks
from holecolor.io.video import save_frame
from holecolor.notebook import write_results_notebook
from holecolor.masks.matrix import make_global_hole_union, make_matrix_bulk_mask
from holecolor.masks.terraces import make_hole_interior_mask, make_hole_rim_mask, make_nonoverlapping_hole_terraces, make_hole_interior_region, make_hole_rim_region, TerraceWidthPlan
from holecolor.photometry.selector import run_photometry_selection
from holecolor.photometry.corrections import apply_correction_stack
from holecolor.qc.gates import assert_gate_results, require
from holecolor.qc.reports import save_bar_plot, save_heatmap_plot, save_image, save_line_plot, write_gate_report, write_json, write_table, write_table_columns
from holecolor.registration.rigid import apply_transform
from holecolor.plotting.prepare import bar_from_columns, columns_from_records, count_by_label, heatmap_from_columns, heatmap_from_rows, line_series_from_rows, line_series_from_columns
from holecolor.radial.curves import compute_all_radial_curves
from holecolor.radial.advanced import (
    aggregate_angular_asymmetry_rows,
    assign_radial_archetypes,
    compute_frame_angular_asymmetry,
    merge_hole_radial_and_asymmetry,
    reticulum_zone_by_hole,
    summarize_hole_radial_evolution,
)
from holecolor.radial.columnar import (
    HotspotStatsTable,
    RadialRowTable,
    RdfUncertaintyHoleTable,
    ValidationHoleTable,
    SectorRadialTable,
    SectorRdfFrameTable,
    sector_front_lag_columns,
    sector_hole_summary_plot_columns,
    build_hotspot_reticulum_comparison_table,
    build_hotspot_reticulum_columns_table,
    build_validation_summary_jsons_table,
    build_per_hole_rdf_evolution_columns,
    build_per_hole_rdf_evolution_table,
    build_rdf_hotspot_reticulum_comparison_table,
    build_rdf_hotspot_reticulum_columns_table,
    build_rdf_uncertainty_hotspot_comparison_table,
    build_rdf_uncertainty_reticulum_rows_table,
    build_reticulum_group_rows_table,
    build_sector_front_acceleration_table,
    build_sector_front_lag_rows_table,
    build_sector_front_propagation_table,
    build_sector_rdf_evolution_columns,
    build_sector_rdf_evolution_table,
    fit_radial_models_table,
    per_hole_radial_frame_summary_table,
    summarize_sector_fronts_table,
)
from holecolor.radial.modeling import (
    compute_sector_radial_timeseries,
    fit_radial_models,
    summarize_sector_fronts,
)
from holecolor.radial.validation import aggregate_radial_rows, run_radial_perturbation_sweeps, run_rdf_archetype_perturbation_sweeps, summarize_radial_evolution
from holecolor.radial.rdf import (
    build_per_hole_rdf_archetypes,
    build_per_hole_rdf_bootstrap_summary,
    build_per_hole_rdf_front_dynamics,
    build_rdf_archetype_bootstrap_support,
    build_sector_front_acceleration,
    build_sector_front_lag_rows,
    build_sector_front_propagation,
    build_sector_rdf_evolution,
    canonicalize_rdf_archetypes,
)
from holecolor.radial.perturbation import radial_curve_stability
from holecolor.registration.rigid import residual_difference, select_reference_frame, stabilize_sequence
from holecolor.temporal.events import summarize_curve_events
from holecolor.temporal.phenotypes import (
    assign_hole_phenotypes,
    build_propagation_feature_rows,
    phenotype_stability_across_reruns,
    write_centroids_json,
)
from holecolor.temporal.columnar import (
    PhenotypeTable,
    TemporalValidationSummary,
    build_phenotype_archetype_rows_table,
    build_phenotype_neighbor_and_smoothness_table,
)
from holecolor.visualisation.cluster_rdf_views import run_cluster_rdf_visualisations


def _quiet_progress_cfg(cfg: ParallelConfig) -> ParallelConfig:
    return replace(cfg, show_progress=False)


def _hash_payload(payload: Any) -> str:
    blob = json.dumps(_stable_payload(payload), sort_keys=True, separators=(",", ":"))
    return hashlib.md5(blob.encode("utf-8")).hexdigest()


def _frames_content_digest(frames: list[FrameRecord], sample_target: int = 96) -> str:
    hsh = hashlib.md5()
    hsh.update(str(len(frames)).encode("utf-8"))
    for frame in frames:
        arr = np.asarray(frame.image)
        hsh.update(str(int(frame.frame_id)).encode("utf-8"))
        hsh.update(repr(tuple(int(v) for v in arr.shape)).encode("utf-8"))
        hsh.update(str(arr.dtype).encode("utf-8"))
        if arr.ndim >= 2:
            sy = max(1, int(arr.shape[0]) // max(1, int(sample_target)))
            sx = max(1, int(arr.shape[1]) // max(1, int(sample_target)))
            sample = np.ascontiguousarray(arr[::sy, ::sx])
        else:
            sample = np.ascontiguousarray(arr)
        hsh.update(sample.tobytes())
    return hsh.hexdigest()


def _array_content_digest(arr: np.ndarray | None) -> str | None:
    if arr is None:
        return None
    data = np.ascontiguousarray(np.asarray(arr))
    hsh = hashlib.md5()
    hsh.update(repr(tuple(int(v) for v in data.shape)).encode("utf-8"))
    hsh.update(str(data.dtype).encode("utf-8"))
    hsh.update(data.tobytes())
    return hsh.hexdigest()


def _checkpoint_reuse_enabled(cfg: PipelineConfig) -> bool:
    cp = getattr(cfg, "checkpoint", None)
    return bool(cp is not None and cp.enabled and cp.reuse)


def _checkpoint_write_enabled(cfg: PipelineConfig) -> bool:
    cp = getattr(cfg, "checkpoint", None)
    return bool(cp is not None and cp.enabled and cp.write)


def _checkpoint_manifest_path(qc_dir: Path, name: str) -> Path:
    safe = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in name.lower())
    return qc_dir / f"{safe}_checkpoint.json"


def _checkpoint_required_ready(required: list[Path] | tuple[Path, ...]) -> bool:
    return all(path.exists() for path in required)


def _checkpoint_matches(cfg: PipelineConfig, qc_dir: Path, name: str, signature: dict[str, Any], required: list[Path] | tuple[Path, ...]) -> bool:
    if not _checkpoint_reuse_enabled(cfg):
        return False
    manifest = _read_json(_checkpoint_manifest_path(qc_dir, name))
    if not manifest or manifest.get("status") != "complete":
        return False
    return manifest.get("signature") == signature and _checkpoint_required_ready(required)


def _write_checkpoint(
    cfg: PipelineConfig,
    out_dir: Path,
    qc_dir: Path,
    name: str,
    signature: dict[str, Any],
    outputs: dict[str, str | Path],
    *,
    reused: bool = False,
) -> None:
    if not _checkpoint_write_enabled(cfg):
        return
    payload = {
        "schema_version": 1,
        "stage": name,
        "status": "complete",
        "reused": bool(reused),
        "signature": signature,
        "outputs": {str(k): str(v) for k, v in outputs.items()},
    }
    write_json(_checkpoint_manifest_path(qc_dir, name), payload)
    index_path = out_dir / "logs" / "stage_checkpoints.json"
    index = _read_json(index_path, {"schema_version": 1, "stages": {}})
    stages = index.setdefault("stages", {})
    stages[str(name)] = {
        "status": "complete",
        "reused": bool(reused),
        "manifest": str(_checkpoint_manifest_path(qc_dir, name)),
        "outputs": payload["outputs"],
    }
    write_json(index_path, index)


def _signature_for_stage(name: str, **items: Any) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "stage": str(name),
        "items": _stable_payload(items),
        "digest": _hash_payload(items),
    }


def _audit_records_from_rows(rows: list[dict[str, Any]]) -> list[AuditRecord]:
    records: list[AuditRecord] = []
    for row in rows:
        records.append(
            AuditRecord(
                frame_id=int(row["frame_id"]),
                blur_score=float(row["blur_score"]),
                sat_frac_r=float(row["sat_frac_r"]),
                sat_frac_g=float(row["sat_frac_g"]),
                sat_frac_b=float(row["sat_frac_b"]),
                frame_jump_score=float(row.get("frame_jump_score", 0.0) or 0.0),
                accepted=bool(row.get("accepted", False)),
            )
        )
    return records


def _photometry_scores_from_rows(rows: list[dict[str, Any]]) -> list[PhotometryScore]:
    scores: list[PhotometryScore] = []
    for row in rows:
        scores.append(
            PhotometryScore(
                frame_id=int(row["frame_id"]),
                correction_name=str(row["correction_name"]),
                edge_energy_ratio=float(row["edge_energy_ratio"]),
                positive_prominence=float(row["positive_prominence"]),
                hole_drift_score=None if row.get("hole_drift_score") is None else float(row["hole_drift_score"]),
                total_score=float(row["total_score"]),
            )
        )
    return scores


def _transforms_from_rows(rows: list[dict[str, Any]]) -> list[TransformRecord]:
    out: list[TransformRecord] = []
    for row in rows:
        out.append(
            TransformRecord(
                frame_id=int(row["frame_id"]),
                dx=float(row.get("dx", 0.0) or 0.0),
                dy=float(row.get("dy", 0.0) or 0.0),
                angle_deg=float(row.get("angle_deg", 0.0) or 0.0),
                scale=float(row.get("scale", 1.0) or 1.0),
            )
        )
    return out


def _holes_from_rows(rows: list[dict[str, Any]]) -> list[HoleGeometry]:
    holes: list[HoleGeometry] = []
    for row in sorted(rows, key=lambda r: int(r.get("hole_id", 0) or 0)):
        holes.append(
            HoleGeometry(
                hole_id=int(row["hole_id"]),
                x=float(row["x"]),
                y=float(row["y"]),
                radius_inner_px=float(row["radius_inner_px"]),
                radius_outer_px=float(row["radius_outer_px"]),
                confidence=float(row.get("confidence", 1.0) or 0.0),
            )
        )
    return holes


def _candidates_from_rows(rows: list[dict[str, Any]]) -> list[HoleCandidate]:
    candidates: list[HoleCandidate] = []
    for row in sorted(rows, key=lambda r: int(r.get("candidate_id", len(candidates)) or 0)):
        candidates.append(
            HoleCandidate(
                x=float(row["x"]),
                y=float(row["y"]),
                radius_px=float(row.get("radius_px", row.get("radius_outer_px", 1.0)) or 1.0),
                ellipticity=float(row.get("ellipticity", 0.0) or 0.0),
                boundary_contrast=float(row.get("boundary_contrast", 0.0) or 0.0),
                confidence=float(row.get("confidence", 1.0) or 0.0),
            )
        )
    return candidates


def _lattice_from_json(payload: dict[str, Any]) -> LatticeModel:
    return LatticeModel(
        origin_x=float(payload["origin_x"]),
        origin_y=float(payload["origin_y"]),
        basis_u=tuple(float(v) for v in payload["basis_u"]),  # type: ignore[arg-type]
        basis_v=tuple(float(v) for v in payload["basis_v"]),  # type: ignore[arg-type]
        angle_deg=float(payload["angle_deg"]),
        spacing_u_px=float(payload["spacing_u_px"]),
        spacing_v_px=float(payload["spacing_v_px"]),
        confidence=float(payload["confidence"]),
    )


def _lattice_indices_from_rows(rows: list[dict[str, Any]]) -> dict[int, tuple[int, int]]:
    out: dict[int, tuple[int, int]] = {}
    for row in rows:
        if row.get("lattice_u") is None or row.get("lattice_v") is None:
            continue
        out[int(row["hole_id"])] = (int(row["lattice_u"]), int(row["lattice_v"]))
    return out


def _holes_by_frame_from_rows(rows: list[dict[str, Any]]) -> dict[int, list[HoleGeometry]]:
    out: dict[int, list[HoleGeometry]] = {}
    for row in rows:
        fid = int(row["frame_id"])
        out.setdefault(fid, []).append(
            HoleGeometry(
                hole_id=int(row["hole_id"]),
                x=float(row["x"]),
                y=float(row["y"]),
                radius_inner_px=float(row["radius_inner_px"]),
                radius_outer_px=float(row["radius_outer_px"]),
                confidence=float(row.get("confidence", 1.0) or 0.0),
            )
        )
    for fid in list(out):
        out[fid] = sorted(out[fid], key=lambda h: h.hole_id)
    return out


def _load_wafer_nonhole_bundle_from_artifacts(base_dir: Path, support_mask: np.ndarray | None) -> WaferNonholeColourBundle | None:
    status = _read_json(base_dir / "stage_status.json")
    if not status:
        return None
    cluster_rows = _read_table_rows(base_dir / "cluster_model_summary.csv")
    frame_rows = _read_table_rows(base_dir / "frame_region_summary.csv")
    frame_cluster_summary_rows = _read_table_rows(base_dir / "frame_cluster_summary.csv")
    hard_rows = _read_table_rows(base_dir / "frame_cluster_prevalence_hard.csv")
    soft_rows = _read_table_rows(base_dir / "frame_cluster_prevalence_soft.csv")
    model_selection_rows = _read_table_rows(base_dir / "cluster_model_selection.csv")
    return WaferNonholeColourBundle(
        str(status.get("status", "skipped")),
        support_mask,
        frame_rows,
        cluster_rows,
        frame_cluster_summary_rows,
        hard_rows,
        soft_rows,
        _read_table_rows(base_dir / "global_buffer_cluster_context.csv"),
        _read_table_rows(base_dir / "local_hole_cluster_context.csv"),
        None if status.get("selected_k") is None else int(status["selected_k"]),
        message=None if status.get("message") is None else str(status.get("message")),
        model_selection_rows=model_selection_rows,
    )


def _relpath(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def _artifact_record(root: Path, key: str, path: Path, label: str, role: str, group: str, description: str) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return {
        "key": key,
        "label": label,
        "role": role,
        "group": group,
        "path": _relpath(path, root),
        "description": description,
        "size_bytes": int(path.stat().st_size),
    }


def _collect_curated_artifacts(out_dir: Path) -> list[dict[str, Any]]:
    specs = [
        ("summary", out_dir / "summary.json", "Run summary", "primary_json", "run", "Compact scientific and QC summary for the run."),
        ("qc_gates", out_dir / "qc_gates.json", "QC gates", "primary_json", "run", "Pass/fail checks and warning details."),
        ("checkpoints", out_dir / "logs" / "stage_checkpoints.json", "Stage checkpoints", "support_json", "run", "Checkpoint overview showing which resumable stages were written or reused."),
        ("geometry_overlay", out_dir / "geometry" / "overlays" / "frame_ref_geometry_overlay.png", "Reference geometry", "primary_image", "geometry", "Wafer contour, hole centres, and lattice basis on the reference frame."),
        ("geometry_sanity", out_dir / "geometry" / "geometry_sanity_checks.json", "Geometry sanity checks", "primary_json", "geometry", "Sanity checks for wafer support, lattice, and completeness."),
        ("hole_exclusion", out_dir / "geometry" / "hole_terrace_exclusion_summary.json", "Hole/terrace completeness", "primary_json", "geometry", "How many holes were retained after excluding partial holes or terraces."),
        ("terrace_overlay", out_dir / "masks" / "overlays" / "frame_ref_terraces_overlay.png", "Reference terraces", "support_image", "geometry", "Reference terrace mask overlay."),
        ("cluster_legend", out_dir / "descriptors" / "radial_cluster_average_hole" / "cluster_colour_legend.png", "Colour cluster legend", "primary_image", "colour", "Cluster IDs using observed cluster colours."),
        ("cluster_model", out_dir / "descriptors" / "wafer_nonhole_colour" / "cluster_model_summary.csv", "Colour cluster model", "primary_table", "colour", "Global colour cluster centroids and observed display colours."),
        ("cluster_selection", out_dir / "descriptors" / "wafer_nonhole_colour" / "cluster_model_selection.csv", "Cluster-count selection", "support_table", "colour", "Model selection table used to decide how many colour clusters are relevant."),
        ("frame_cluster_summary", out_dir / "descriptors" / "wafer_nonhole_colour" / "frame_cluster_summary.csv", "Frame colour cluster summary", "primary_table", "colour", "Global cluster prevalence per frame."),
        ("baseline_activity_video", out_dir / "descriptors" / "wafer_nonhole_colour" / "video_cluster_baseline_activity.avi", "Baseline activity recolour video", "primary_video", "colour", "Neutral background with alpha proportional to baseline-corrected cluster activity."),
        ("hard_label_video", out_dir / "descriptors" / "wafer_nonhole_colour" / "video_cluster_labels.avi", "Diagnostic hard-label video", "diagnostic_video", "colour", "Hard cluster labels, useful mainly to catch masking or geometry failures."),
        ("rdf_montage", out_dir / "descriptors" / "radial_cluster_average_hole" / "00_holecolor_visualisation_montage.png", "Cluster RDF montage", "primary_image", "cluster_rdf", "Combined preferred cluster RDF visualisation panel."),
        ("active_rdf", out_dir / "descriptors" / "radial_cluster_average_hole" / "option_02_baseline_corrected_active_rdf.png", "Baseline-corrected cluster RDF", "primary_image", "cluster_rdf", "Cluster activity through time and terraces after baseline correction."),
        ("dominant_cluster_map", out_dir / "descriptors" / "radial_cluster_average_hole" / "option_03_dominant_cluster_phase_map.png", "Dominant cluster phase map", "primary_image", "cluster_rdf", "Dominant cluster identity by time and terrace."),
        ("sector_fan", out_dir / "descriptors" / "radial_cluster_average_hole" / "option_06_sector_fan_active_cluster.png", "Sector fan activity", "primary_image", "cluster_rdf", "Sector-resolved active cluster behaviour."),
        ("front_trajectory", out_dir / "descriptors" / "radial_cluster_average_hole" / "option_09_activity_weighted_front_trajectory.png", "Activity-weighted front trajectory", "primary_image", "cluster_rdf", "Activity-weighted terrace position through time."),
        ("radial_activity_table", out_dir / "descriptors" / "radial_cluster_average_hole" / "radial_cluster_rdf_activity.csv", "Cluster RDF activity table", "primary_table", "cluster_rdf", "Baseline-corrected activity table behind the preferred cluster RDF views."),
        ("radial_tensor", out_dir / "descriptors" / "radial_cluster_average_hole" / "hole_terrace_sector_cluster_tensor.csv", "Hole/terrace/sector cluster tensor", "source_table", "cluster_rdf", "Canonical tensor used by the new visualisation layer."),
        ("per_hole_rdf", out_dir / "radial" / "per_hole_rdf_evolution.png", "Per-hole RDF evolution", "primary_image", "spatial", "Hole-level radial colour evolution."),
        ("rdf_evolution", out_dir / "radial" / "radial_distribution_evolution.png", "Global RDF evolution", "primary_image", "global", "Global radial distribution evolution."),
        ("phenotype_map", out_dir / "temporal" / "phenotype_spatial_map.png", "Phenotype spatial map", "primary_image", "spatial", "Spatial layout of temporal response phenotypes."),
        ("phenotypes", out_dir / "temporal" / "per_hole_phenotypes.csv", "Per-hole phenotypes", "primary_table", "spatial", "Per-hole temporal response classes."),
        ("notebook", out_dir / "notebooks" / "holecolor_results_explorer.ipynb", "Analysis notebook", "support_notebook", "run", "Notebook entry point for deeper exploration."),
    ]
    artifacts: list[dict[str, Any]] = []
    for spec in specs:
        record = _artifact_record(out_dir, *spec)
        if record is not None:
            artifacts.append(record)
    return artifacts


def _write_curated_output_index(out_dir: Path, summary: dict[str, Any]) -> None:
    output_dir = ensure_dir(out_dir / "outputs")
    artifacts = _collect_curated_artifacts(out_dir)
    primary = [a for a in artifacts if str(a["role"]).startswith("primary")]
    diagnostics = [a for a in artifacts if not str(a["role"]).startswith("primary")]
    write_json(output_dir / "output_manifest.json", {"schema_version": 1, "summary": summary, "artifacts": artifacts})

    def link_for(record: dict[str, Any]) -> str:
        href = "../" + str(record["path"])
        return f"[{record['label']}]({href})"

    md_lines = [
        "# Holecolor Outputs",
        "",
        "Start here. This folder is the curated view of the run; the original technical outputs remain in their stage folders for traceability.",
        "",
        "## Primary Outputs",
        "",
    ]
    for record in primary:
        md_lines.append(f"- {link_for(record)} - {record['description']}")
    if diagnostics:
        md_lines.extend(["", "## Diagnostics And Source Tables", ""])
        for record in diagnostics:
            md_lines.append(f"- {link_for(record)} - {record['description']}")
    md_lines.extend([
        "",
        "## Suggested Reading Order",
        "",
        "1. Geometry overlay and geometry sanity checks.",
        "2. Colour cluster legend, cluster model, and baseline activity recolour video.",
        "3. Cluster RDF montage, active RDF, sector fan, and front trajectory.",
        "4. Per-hole phenotypes and phenotype spatial map.",
    ])
    (output_dir / "START_HERE.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    image_cards = []
    for record in primary:
        path = str(record["path"])
        if not path.lower().endswith((".png", ".jpg", ".jpeg")):
            continue
        href = "../" + path
        image_cards.append(
            "<figure>"
            f"<a href=\"{html.escape(href)}\"><img src=\"{html.escape(href)}\" alt=\"{html.escape(str(record['label']))}\"></a>"
            f"<figcaption><strong>{html.escape(str(record['label']))}</strong><br>{html.escape(str(record['description']))}</figcaption>"
            "</figure>"
        )
    table_rows = "\n".join(
        "<tr>"
        f"<td>{html.escape(str(record['group']))}</td>"
        f"<td>{html.escape(str(record['role']))}</td>"
        f"<td><a href=\"../{html.escape(str(record['path']))}\">{html.escape(str(record['label']))}</a></td>"
        f"<td>{html.escape(str(record['description']))}</td>"
        "</tr>"
        for record in artifacts
    )
    html_text = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Holecolor Outputs</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; color: #1f2933; background: #fafafa; }}
    h1, h2 {{ margin-bottom: 0.35rem; }}
    .summary {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 12px; margin: 18px 0; }}
    .metric {{ background: white; border: 1px solid #dde3ea; border-radius: 6px; padding: 12px; }}
    .metric span {{ display: block; font-size: 12px; color: #52606d; }}
    .metric strong {{ font-size: 20px; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 18px; }}
    figure {{ margin: 0; background: white; border: 1px solid #dde3ea; border-radius: 6px; padding: 10px; }}
    img {{ width: 100%; height: auto; display: block; }}
    figcaption {{ font-size: 13px; color: #3e4c59; margin-top: 8px; line-height: 1.35; }}
    table {{ width: 100%; border-collapse: collapse; background: white; margin-top: 12px; }}
    th, td {{ border: 1px solid #dde3ea; padding: 8px; text-align: left; vertical-align: top; font-size: 13px; }}
    th {{ background: #eef2f6; }}
  </style>
</head>
<body>
  <h1>Holecolor Outputs</h1>
  <p>This is the curated navigation layer for the run. Technical outputs remain in their original folders.</p>
  <section class="summary">
    <div class="metric"><span>frames</span><strong>{html.escape(str(summary.get('n_frames', 'n/a')))}</strong></div>
    <div class="metric"><span>holes</span><strong>{html.escape(str(summary.get('n_candidates', 'n/a')))}</strong></div>
    <div class="metric"><span>correction</span><strong>{html.escape(str(summary.get('winner_correction', 'n/a')))}</strong></div>
    <div class="metric"><span>lattice confidence</span><strong>{html.escape(str(round(float(summary.get('lattice_confidence', float('nan'))), 3) if summary.get('lattice_confidence') is not None else 'n/a'))}</strong></div>
  </section>
  <h2>Primary Figures</h2>
  <section class="grid">
    {''.join(image_cards)}
  </section>
  <h2>Artifact Index</h2>
  <table>
    <thead><tr><th>Group</th><th>Role</th><th>Artifact</th><th>Description</th></tr></thead>
    <tbody>{table_rows}</tbody>
  </table>
</body>
</html>
"""
    (output_dir / "index.html").write_text(html_text, encoding="utf-8")



def _row_cache_path(path: Path) -> Path:
    return path.with_suffix(path.suffix + ".pkl")


def _npz_cache_path(path: Path) -> Path:
    return path.with_suffix(path.suffix + ".npz")


def _table_cache_exists(path: Path) -> bool:
    return path.exists() or _npz_cache_path(path).exists() or _row_cache_path(path).exists()


def _column_keys(rows: list[dict[str, Any]]) -> list[str]:
    keys: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            skey = str(key)
            if skey not in seen:
                seen.add(skey)
                keys.append(skey)
    return keys


def _infer_npz_column_kind(values: list[Any]) -> str | None:
    non_null = [v for v in values if v is not None]
    if not non_null:
        return 'empty'
    if all(isinstance(v, (bool, np.bool_)) for v in non_null):
        return 'bool'
    if all(isinstance(v, (int, np.integer)) and not isinstance(v, (bool, np.bool_)) for v in non_null):
        return 'int'
    if all(isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, (bool, np.bool_)) for v in non_null):
        return 'float'
    if all(isinstance(v, str) for v in non_null):
        return 'str'
    return None


def _write_npz_table(path: Path, rows: list[dict[str, Any]]) -> bool:
    npz_path = _npz_cache_path(path)
    columns = _column_keys(rows)
    arrays: dict[str, Any] = {}
    meta: dict[str, Any] = {"columns": columns, "kinds": {}}
    for col in columns:
        values = [row.get(col) for row in rows]
        kind = _infer_npz_column_kind(values)
        if kind is None:
            return False
        meta['kinds'][col] = kind
        mask = np.asarray([v is not None for v in values], dtype=np.bool_)
        arrays[f'{col}__mask'] = mask
        if kind == 'empty':
            arrays[f'{col}__data'] = np.zeros(len(values), dtype=np.float64)
        elif kind == 'bool':
            arrays[f'{col}__data'] = np.asarray([bool(v) if v is not None else False for v in values], dtype=np.bool_)
        elif kind == 'int':
            arrays[f'{col}__data'] = np.asarray([int(v) if v is not None else 0 for v in values], dtype=np.int64)
        elif kind == 'float':
            arrays[f'{col}__data'] = np.asarray([float(v) if v is not None else np.nan for v in values], dtype=np.float64)
        elif kind == 'str':
            arrays[f'{col}__data'] = np.asarray([str(v) if v is not None else '' for v in values], dtype=np.str_)
    arrays['__meta_json__'] = np.asarray(json.dumps(meta), dtype=np.str_)
    np.savez_compressed(npz_path, **arrays)
    return True


def _read_npz_table(path: Path) -> list[dict[str, Any]] | None:
    npz_path = _npz_cache_path(path)
    if not npz_path.exists():
        return None
    try:
        with np.load(npz_path, allow_pickle=False) as data:
            meta = json.loads(str(data['__meta_json__']))
            columns = [str(c) for c in meta.get('columns', [])]
            kinds = {str(k): str(v) for k, v in meta.get('kinds', {}).items()}
            if not columns:
                return []
            n = len(data[f'{columns[0]}__mask'])
            rows: list[dict[str, Any]] = []
            for i in range(n):
                row: dict[str, Any] = {}
                for col in columns:
                    mask = data[f'{col}__mask']
                    if not bool(mask[i]):
                        row[col] = None
                        continue
                    raw = data[f'{col}__data'][i]
                    kind = kinds.get(col, 'float')
                    if kind == 'bool':
                        row[col] = bool(raw)
                    elif kind == 'int':
                        row[col] = int(raw)
                    elif kind == 'float' or kind == 'empty':
                        row[col] = float(raw)
                    elif kind == 'str':
                        row[col] = str(raw)
                    else:
                        row[col] = raw.item() if hasattr(raw, 'item') else raw
                rows.append(row)
            return rows
    except Exception:
        return None


def _normalize_cache_rows(records: list[dict[str, Any]] | list[Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for record in records:
        if isinstance(record, dict):
            out.append(record)
        else:
            try:
                out.append(asdict(record))
            except Exception:
                out.append(vars(record))
    return out


def _write_cached_table(path: Path, records: list[dict[str, Any]] | list[Any]) -> None:
    rows = _normalize_cache_rows(records)
    write_table(path, rows)
    _write_npz_table(path, rows)
    with _row_cache_path(path).open("wb") as f:
        pickle.dump(rows, f, protocol=pickle.HIGHEST_PROTOCOL)

def _coerce_csv_value(value: str | None) -> Any:
    if value is None or value == "":
        return None
    low = value.lower()
    if low == "true":
        return True
    if low == "false":
        return False
    try:
        if value.isdigit() or (value.startswith("-") and value[1:].isdigit()):
            return int(value)
    except Exception:
        pass
    try:
        return float(value)
    except Exception:
        return value


def _read_table_rows(path: Path) -> list[dict[str, Any]]:
    cache_path = _row_cache_path(path)
    if cache_path.exists():
        try:
            with cache_path.open("rb") as f:
                rows = pickle.load(f)
            if isinstance(rows, list):
                return rows
        except Exception:
            pass
    npz_rows = _read_npz_table(path)
    if npz_rows is not None:
        return npz_rows
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows: list[dict[str, Any]] = []
        for row in reader:
            rows.append({k: _coerce_csv_value(v) for k, v in row.items()})
        return rows


def _read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def _stable_payload(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _stable_payload(value[k]) for k in sorted(value)}
    if isinstance(value, (list, tuple)):
        return [_stable_payload(v) for v in value]
    if isinstance(value, float):
        if np.isnan(value):
            return "NaN"
        if np.isposinf(value):
            return "Inf"
        if np.isneginf(value):
            return "-Inf"
        return round(value, 8)
    return value


def _rows_digest(rows: list[dict[str, Any]], keys: list[str] | tuple[str, ...] | None = None, limit: int | None = None) -> str:
    use_rows = rows if limit is None else rows[: max(int(limit), 0)]
    payload: list[dict[str, Any]] = []
    for row in use_rows:
        if keys is None:
            item = {str(k): _stable_payload(v) for k, v in sorted(row.items(), key=lambda kv: str(kv[0]))}
        else:
            item = {str(k): _stable_payload(row.get(k)) for k in keys}
        payload.append(item)
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.md5(blob.encode("utf-8")).hexdigest()




def _frame_analysis_cache_signature(
    cfg: PipelineConfig,
    chosen_descriptor: str,
    stabilized: list[FrameRecord],
    geometry_rows: list[dict[str, Any]],
    lattice_indices: dict[int, tuple[int, int]],
) -> dict[str, Any]:
    frame_rows: list[dict[str, Any]] = []
    for frame in stabilized:
        img = frame.image.astype(np.float32, copy=False)
        sample = img[::32, ::32] if img.ndim >= 2 else img
        row = {
            "frame_id": int(frame.frame_id),
            "shape": list(map(int, img.shape)),
            "mean_r": float(np.mean(sample[..., 0])) if sample.ndim == 3 else float(np.mean(sample)),
            "mean_g": float(np.mean(sample[..., 1])) if sample.ndim == 3 and sample.shape[-1] > 1 else None,
            "mean_b": float(np.mean(sample[..., 2])) if sample.ndim == 3 and sample.shape[-1] > 2 else None,
            "std_r": float(np.std(sample[..., 0])) if sample.ndim == 3 else float(np.std(sample)),
            "std_g": float(np.std(sample[..., 1])) if sample.ndim == 3 and sample.shape[-1] > 1 else None,
            "std_b": float(np.std(sample[..., 2])) if sample.ndim == 3 and sample.shape[-1] > 2 else None,
        }
        frame_rows.append(row)
    return {
        "schema_version": 2,
        "chosen_descriptor": str(chosen_descriptor),
        "n_terraces": int(cfg.masks.n_terraces),
        "n_sectors": int(cfg.radial.angular_n_sectors),
        "interior_shrink_px": int(cfg.masks.interior_shrink_px),
        "rim_width_px": int(cfg.masks.rim_width_px),
        "hotspot_threshold_mode": str(cfg.hotspots.threshold_mode),
        "hotspot_threshold_value": float(cfg.hotspots.threshold_value),
        "hotspot_min_area_px": int(cfg.hotspots.min_area_px),
        "hotspot_link_max_dist_px": float(cfg.hotspots.link_max_dist_px),
        "hotspot_max_area_ratio": float(cfg.hotspots.max_area_ratio),
        "frame_count": len(stabilized),
        "geometry_count": len(geometry_rows),
        "lattice_index_count": len(lattice_indices),
        "frame_digest": _rows_digest(frame_rows, keys=("frame_id", "shape", "mean_r", "mean_g", "mean_b", "std_r", "std_g", "std_b")),
        "geometry_digest": _rows_digest(geometry_rows, keys=("frame_id", "hole_id", "x", "y", "radius_inner_px", "radius_outer_px", "confidence")),
        "lattice_digest": hashlib.md5(json.dumps({str(k): lattice_indices[k] for k in sorted(lattice_indices)}, sort_keys=True).encode("utf-8")).hexdigest(),
    }


def _frame_analysis_cache_paths(descriptors_dir: Path, radial_dir: Path, hotspots_dir: Path, qc_dir: Path) -> dict[str, Path]:
    return {
        "manifest": qc_dir / "frame_analysis_cache_manifest.json",
        "compartments": descriptors_dir / "hole_compartment_timeseries.csv",
        "matrix": descriptors_dir / "matrix_timeseries.csv",
        "radial": radial_dir / "hole_annulus_timeseries.csv",
        "angular": radial_dir / "angular_asymmetry_timeseries.csv",
        "sector": radial_dir / "sector_radial_timeseries.csv",
        "hotspots": hotspots_dir / "hotspots.csv",
        "tracks": hotspots_dir / "tracks.csv",
    }


def _frame_analysis_cache_available(descriptors_dir: Path, radial_dir: Path, hotspots_dir: Path, qc_dir: Path, expected_signature: dict[str, Any]) -> bool:
    paths = _frame_analysis_cache_paths(descriptors_dir, radial_dir, hotspots_dir, qc_dir)
    manifest = _read_json(paths["manifest"])
    if not manifest or manifest.get("signature") != expected_signature:
        return False
    required = ["compartments", "matrix", "radial", "angular", "sector", "hotspots", "tracks"]
    return all(_table_cache_exists(paths[name]) for name in required)


def _load_frame_analysis_cache(descriptors_dir: Path, radial_dir: Path, hotspots_dir: Path, qc_dir: Path, expected_signature: dict[str, Any]) -> dict[str, Any] | None:
    paths = _frame_analysis_cache_paths(descriptors_dir, radial_dir, hotspots_dir, qc_dir)
    manifest = _read_json(paths["manifest"])
    if not manifest or manifest.get("signature") != expected_signature:
        return None
    required = ["compartments", "matrix", "radial", "angular", "sector", "hotspots", "tracks"]
    if not all(_table_cache_exists(paths[name]) for name in required):
        return None
    return {
        "compartment_rows": _read_table_rows(paths["compartments"]),
        "matrix_rows": _read_table_rows(paths["matrix"]),
        "radial_rows": _read_table_rows(paths["radial"]),
        "angular_rows": _read_table_rows(paths["angular"]),
        "sector_radial_rows": _read_table_rows(paths["sector"]),
        "hotspot_rows": _read_table_rows(paths["hotspots"]),
        "hotspot_track_rows": _read_table_rows(paths["tracks"]),
    }


def _write_frame_analysis_cache(descriptors_dir: Path, radial_dir: Path, hotspots_dir: Path, qc_dir: Path, signature: dict[str, Any], *, compartment_rows: list[dict[str, Any]], matrix_rows: list[dict[str, Any]], radial_rows: list[dict[str, Any]], angular_rows: list[dict[str, Any]], sector_radial_rows: list[dict[str, Any]], hotspot_rows: list[dict[str, Any]], hotspot_track_rows: list[dict[str, Any]]) -> None:
    paths = _frame_analysis_cache_paths(descriptors_dir, radial_dir, hotspots_dir, qc_dir)
    _write_cached_table(paths["compartments"], compartment_rows)
    _write_cached_table(paths["matrix"], matrix_rows)
    _write_cached_table(paths["radial"], radial_rows)
    _write_cached_table(paths["angular"], angular_rows)
    _write_cached_table(paths["sector"], sector_radial_rows)
    _write_cached_table(paths["hotspots"], hotspot_rows)
    _write_cached_table(paths["tracks"], hotspot_track_rows)
    write_json(paths["manifest"], {"signature": signature})
def _validation_cache_signature(
    cfg: PipelineConfig,
    chosen_descriptor: str,
    radial_conclusion_summary: dict[str, Any],
    per_hole_rdf_frame_rows: list[dict[str, Any]],
    sector_rdf_frame_rows: list[dict[str, Any]],
    hotspot_rows: list[dict[str, Any]],
    per_hole_rdf_archetype_rows: list[dict[str, Any]],
    per_hole_rdf_dynamics_rows: list[dict[str, Any]],
    zone_by_hole: dict[int, str],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "validation_enabled": bool(cfg.validation.enabled),
        "chosen_descriptor": str(chosen_descriptor),
        "rdf_bootstrap_n": int(cfg.radial.rdf_bootstrap_n),
        "brightness_factors": [float(x) for x in cfg.validation.brightness_factors],
        "radius_scale_factors": [float(x) for x in cfg.validation.radius_scale_factors],
        "conclusion_label": radial_conclusion_summary.get("conclusion_label"),
        "rdf_frame_count": len(per_hole_rdf_frame_rows),
        "sector_frame_count": len(sector_rdf_frame_rows),
        "hotspot_count": len(hotspot_rows),
        "archetype_count": len(per_hole_rdf_archetype_rows),
        "dynamics_count": len(per_hole_rdf_dynamics_rows),
        "zone_count": len(zone_by_hole),
        "rdf_frame_digest": _rows_digest(per_hole_rdf_frame_rows, keys=("hole_id", "frame_id", "annulus_id", "rdf_pdf", "front_radius_norm", "delta_descriptor")),
        "sector_frame_digest": _rows_digest(sector_rdf_frame_rows, keys=("hole_id", "frame_id", "sector_id", "front_radius_norm", "delta_descriptor")),
        "hotspot_digest": _rows_digest(hotspot_rows, keys=("frame_id", "hotspot_id", "nearest_hole_id", "dist_to_hole_px", "score")),
        "archetype_digest": _rows_digest(per_hole_rdf_archetype_rows, keys=("hole_id", "rdf_archetype_canonical_id", "rdf_archetype_label")),
        "dynamics_digest": _rows_digest(per_hole_rdf_dynamics_rows, keys=("hole_id", "front_velocity_per_frame", "front_acceleration", "nonlinearity_gain")),
        "zone_digest": hashlib.md5(json.dumps({str(k): zone_by_hole[k] for k in sorted(zone_by_hole)}, sort_keys=True).encode("utf-8")).hexdigest(),
    }


def _validation_cache_paths(radial_dir: Path, qc_dir: Path) -> dict[str, Path]:
    return {
        "manifest": qc_dir / "validation_cache_manifest.json",
        "rdf_stability": radial_dir / "per_hole_rdf_stability.csv",
        "rdf_stability_summary": radial_dir / "per_hole_rdf_stability_summary.csv",
        "rdf_bootstrap": radial_dir / "per_hole_rdf_bootstrap_summary.csv",
        "rdf_bootstrap_support": radial_dir / "rdf_archetype_bootstrap_support.csv",
        "sector_front_propagation": radial_dir / "sector_front_propagation.csv",
        "sector_front_propagation_hole": radial_dir / "sector_front_propagation_hole_summary.csv",
        "rdf_uncertainty_reticulum": radial_dir / "rdf_uncertainty_reticulum.csv",
        "rdf_uncertainty_hotspot": radial_dir / "rdf_uncertainty_hotspot_comparison.csv",
        "rdf_uncertainty_hotspot_group": radial_dir / "rdf_uncertainty_hotspot_group_summary.csv",
        "sector_front_acceleration": radial_dir / "sector_front_acceleration.csv",
        "sector_front_acceleration_hole": radial_dir / "sector_front_acceleration_hole_summary.csv",
        "sweeps": qc_dir / "radial_perturbation_sweeps.csv",
        "sweep_drift": qc_dir / "radial_perturbation_drift.csv",
        "radial_consistency": qc_dir / "radial_conclusion_consistency.json",
        "rdf_stability_summary_json": qc_dir / "rdf_archetype_stability_summary.json",
    }


def _load_validation_cache(radial_dir: Path, qc_dir: Path, expected_signature: dict[str, Any]) -> dict[str, Any] | None:
    paths = _validation_cache_paths(radial_dir, qc_dir)
    manifest = _read_json(paths["manifest"])
    if not manifest or manifest.get("signature") != expected_signature:
        return None
    required = [
        "rdf_stability", "rdf_stability_summary", "rdf_bootstrap", "rdf_bootstrap_support",
        "sector_front_propagation", "sector_front_propagation_hole", "rdf_uncertainty_reticulum",
        "rdf_uncertainty_hotspot", "rdf_uncertainty_hotspot_group", "sector_front_acceleration",
        "sector_front_acceleration_hole", "sweeps", "sweep_drift", "radial_consistency",
        "rdf_stability_summary_json",
    ]
    if not all((paths[name].exists() if paths[name].suffix == ".json" else _table_cache_exists(paths[name])) for name in required):
        return None
    return {
        "rdf_stability_rows": _read_table_rows(paths["rdf_stability"]),
        "rdf_stability_summary_rows": _read_table_rows(paths["rdf_stability_summary"]),
        "rdf_stability_summary": _read_json(paths["rdf_stability_summary_json"], {"mean_rdf_archetype_stability": None, "n_sweeps": 0}),
        "rdf_bootstrap_rows": _read_table_rows(paths["rdf_bootstrap"]),
        "rdf_bootstrap_support_rows": _read_table_rows(paths["rdf_bootstrap_support"]),
        "sector_front_propagation_rows": _read_table_rows(paths["sector_front_propagation"]),
        "sector_front_propagation_hole_rows": _read_table_rows(paths["sector_front_propagation_hole"]),
        "rdf_uncertainty_reticulum_rows": _read_table_rows(paths["rdf_uncertainty_reticulum"]),
        "rdf_uncertainty_hotspot_rows": _read_table_rows(paths["rdf_uncertainty_hotspot"]),
        "rdf_uncertainty_hotspot_group_rows": _read_table_rows(paths["rdf_uncertainty_hotspot_group"]),
        "sector_front_acceleration_rows": _read_table_rows(paths["sector_front_acceleration"]),
        "sector_front_acceleration_hole_rows": _read_table_rows(paths["sector_front_acceleration_hole"]),
        "sweep_rows": _read_table_rows(paths["sweeps"]),
        "sweep_drift_rows": _read_table_rows(paths["sweep_drift"]),
        "radial_consistency_summary": _read_json(paths["radial_consistency"], {}),
    }


def _write_validation_cache_manifest(radial_dir: Path, qc_dir: Path, signature: dict[str, Any], validation_enabled: bool) -> None:
    payload = {
        "signature": signature,
        "validation_enabled": bool(validation_enabled),
    }
    write_json(_validation_cache_paths(radial_dir, qc_dir)["manifest"], payload)


def _candidate_table(candidates: list[HoleCandidate], lattice_indices: dict[int, tuple[int, int]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for i, cand in enumerate(candidates):
        row = asdict(cand)
        row["candidate_id"] = i
        uv = lattice_indices.get(i)
        row["lattice_u"] = None if uv is None else uv[0]
        row["lattice_v"] = None if uv is None else uv[1]
        rows.append(row)
    return rows


def _support_fraction_from_debug(grid_debug: Any) -> float:
    support_mask = getattr(grid_debug, "support_mask", None)
    if support_mask is None:
        return 1.0
    arr = np.asarray(support_mask, dtype=bool)
    if arr.size == 0:
        return 1.0
    return float(np.count_nonzero(arr) / max(int(arr.size), 1))


def _geometry_sanity_checks(result: Any, image_shape: tuple[int, int]) -> dict[str, Any]:
    debug = getattr(result, "debug", None)
    lattice = getattr(result, "lattice", None)
    candidates = list(getattr(result, "accepted_candidates", []) or [])
    h, w = int(image_shape[0]), int(image_shape[1])
    support_fraction = _support_fraction_from_debug(debug) if debug is not None else 1.0
    support_circle = getattr(debug, "support_circle", None) if debug is not None else None
    common_radius = float(getattr(debug, "common_radius_px", 0.0) or 0.0) if debug is not None else 0.0
    confidence = float(getattr(lattice, "confidence", 0.0) or 0.0) if lattice is not None else 0.0
    spacing_u = float(getattr(lattice, "spacing_u_px", 0.0) or 0.0) if lattice is not None else 0.0
    spacing_v = float(getattr(lattice, "spacing_v_px", 0.0) or 0.0) if lattice is not None else 0.0
    spacing = min(spacing_u, spacing_v) if spacing_u > 0 and spacing_v > 0 else max(spacing_u, spacing_v)
    radius_spacing_ratio = common_radius / spacing if spacing > 0 else float("nan")
    n_candidates = int(len(candidates))
    frame_area = max(h * w, 1)
    support_area = int(np.count_nonzero(getattr(debug, "support_mask", np.ones((h, w), dtype=bool)))) if debug is not None and getattr(debug, "support_mask", None) is not None else frame_area
    physical_max_holes = 0
    if common_radius > 0.0 and support_area > 0:
        physical_max_holes = max(1, int(np.floor(float(support_area) / (np.pi * common_radius * common_radius))))
    full_frame_support = bool(support_fraction > 0.985 or str(getattr(debug, "mode", "")).endswith("frame_support")) if debug is not None else True
    lattice_confidence_min = 0.65 if full_frame_support and n_candidates >= 12 else 0.75
    checks: list[dict[str, Any]] = []

    def add(name: str, passed: bool, severity: str, detail: str, value: Any = None) -> None:
        checks.append({
            "name": str(name),
            "passed": bool(passed),
            "severity": str(severity),
            "detail": str(detail),
            "value": value,
        })

    add(
        "minimum_hole_count",
        n_candidates >= 8,
        "fail",
        f"Detected {n_candidates} holes; expected at least 8 for lattice analysis.",
        n_candidates,
    )
    add(
        "lattice_confidence",
        confidence >= lattice_confidence_min,
        "fail",
        f"Lattice confidence is {confidence:.3f}; required >= {lattice_confidence_min:.2f}.",
        {"confidence": confidence, "minimum": lattice_confidence_min},
    )
    add(
        "radius_spacing_ratio",
        bool(np.isfinite(radius_spacing_ratio) and 0.08 <= radius_spacing_ratio <= 0.42),
        "warn",
        f"Hole radius / lattice spacing is {radius_spacing_ratio:.3f}; expected a moderate fraction of pitch.",
        None if not np.isfinite(radius_spacing_ratio) else float(radius_spacing_ratio),
    )
    add(
        "physical_nonoverlap_hole_count",
        not (physical_max_holes > 0 and n_candidates > physical_max_holes),
        "fail",
        f"Detected {n_candidates} holes but at most {physical_max_holes} non-overlapping radius-{common_radius:.1f}px holes fit in the wafer support.",
        {
            "n_candidates": int(n_candidates),
            "physical_max_holes": int(physical_max_holes),
            "support_area_px": int(support_area),
            "common_radius_px": float(common_radius),
        },
    )
    add(
        "support_not_empty",
        support_area > 0,
        "fail",
        f"Support mask area is {support_area} px.",
        support_area,
    )
    add(
        "support_is_object_specific",
        not full_frame_support,
        "warn",
        f"Wafer support covers {support_fraction:.3f} of the frame; this may be valid for full-wafer crops, but should be inspected.",
        support_fraction,
    )
    add(
        "full_frame_support_hole_count",
        not (full_frame_support and n_candidates > 180),
        "warn",
        f"Full-frame support with {n_candidates} holes should be inspected when a wafer border is expected.",
        n_candidates,
    )
    if support_circle is not None:
        cx, cy, r = [float(v) for v in support_circle]
        diag = float(np.hypot(h, w))
        add(
            "support_circle_scale",
            not (r > 0.75 * diag and n_candidates > 180),
            "warn" if full_frame_support else "fail",
            f"Support radius is {r:.1f}px for frame diagonal {diag:.1f}px.",
            {"radius_px": r, "frame_diagonal_px": diag},
        )
        outside_centers = 0
        for cand in candidates:
            if np.hypot(float(cand.x) - cx, float(cand.y) - cy) > r + 1.0:
                outside_centers += 1
        add(
            "hole_centers_inside_support",
            outside_centers == 0,
            "fail",
            f"{outside_centers} accepted hole centers lie outside support.",
            outside_centers,
        )
    failed = [c for c in checks if not c["passed"] and c["severity"] == "fail"]
    warned = [c for c in checks if not c["passed"] and c["severity"] == "warn"]
    return {
        "passed": len(failed) == 0,
        "status": "ok" if not failed and not warned else ("fail" if failed else "warn"),
        "fail_count": int(len(failed)),
        "warning_count": int(len(warned)),
        "support_fraction": float(support_fraction),
        "support_area_px": int(support_area),
        "frame_area_px": int(frame_area),
        "n_candidates": int(n_candidates),
        "lattice_confidence": float(confidence),
        "common_radius_px": float(common_radius),
        "spacing_u_px": float(spacing_u),
        "spacing_v_px": float(spacing_v),
        "physical_max_holes": int(physical_max_holes),
        "checks": checks,
    }


def _prefer_raw_geometry_result(corrected_result: Any, raw_result: Any) -> bool:
    corrected_debug = getattr(corrected_result, "debug", None)
    raw_debug = getattr(raw_result, "debug", None)
    if corrected_debug is None or raw_debug is None:
        return False
    corrected_n = len(getattr(corrected_result, "accepted_candidates", []))
    raw_n = len(getattr(raw_result, "accepted_candidates", []))
    if raw_n < 8:
        return False
    corrected_support = _support_fraction_from_debug(corrected_debug)
    raw_support = _support_fraction_from_debug(raw_debug)
    corrected_conf = float(getattr(getattr(corrected_result, "lattice", None), "confidence", 0.0))
    raw_conf = float(getattr(getattr(raw_result, "lattice", None), "confidence", 0.0))
    corrected_frame_support = corrected_support > 0.985 or str(getattr(corrected_debug, "mode", "")).endswith("frame_support")
    raw_has_object_support = raw_support < 0.95 and not str(getattr(raw_debug, "mode", "")).endswith("frame_support")
    if corrected_frame_support and raw_has_object_support and raw_conf >= max(0.75, corrected_conf - 0.08):
        return True
    if corrected_frame_support and raw_n < 0.45 * max(corrected_n, 1) and raw_conf >= max(0.75, corrected_conf - 0.12):
        return True
    return False


def _geometry_result_needs_raw_check(result: Any) -> bool:
    debug = getattr(result, "debug", None)
    if debug is None:
        return False
    support_fraction = _support_fraction_from_debug(debug)
    return support_fraction > 0.985 or str(getattr(debug, "mode", "")).endswith("frame_support")




def _terrace_plan_rows(terrace_plan: dict[int, TerraceWidthPlan], lattice_indices: dict[int, tuple[int, int]] | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for hole_id in sorted(terrace_plan):
        plan = terrace_plan[hole_id]
        row = asdict(plan)
        if lattice_indices is not None:
            uv = lattice_indices.get(int(hole_id))
            row['lattice_u'] = None if uv is None else int(uv[0])
            row['lattice_v'] = None if uv is None else int(uv[1])
        rows.append(row)
    return rows


def _terrace_annulus_rows(terrace_plan: dict[int, TerraceWidthPlan]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for hole_id in sorted(terrace_plan):
        rows.extend(terrace_plan[hole_id].annulus_rows())
    return rows
def _refined_holes_from_candidates(gray: np.ndarray, candidates: list[HoleCandidate]) -> list[HoleGeometry]:
    holes: list[HoleGeometry] = []
    for i, cand in enumerate(candidates):
        fit = refine_candidate_with_conic(gray, cand)
        holes.append(conic_to_hole_geometry(fit, i, candidate_confidence=cand.confidence))
    return holes


def _hole_geometry_from_exact_candidates(candidates: list[HoleCandidate]) -> list[HoleGeometry]:
    holes: list[HoleGeometry] = []
    for i, cand in enumerate(candidates):
        outer = max(float(cand.radius_px), 1.0)
        inner = max(outer - 2.0, 1.0)
        holes.append(HoleGeometry(i, float(cand.x), float(cand.y), inner, outer, float(cand.confidence)))
    return holes


def _process_frame_analysis_task(task: dict[str, Any]) -> dict[str, Any]:
    frame: FrameRecord = task["frame"]
    frame_holes: list[HoleGeometry] = task["frame_holes"]
    lattice_indices: dict[int, tuple[int, int]] = task["lattice_indices"]
    shape = frame.image.shape[:2]
    n_terraces = int(task["n_terraces"])
    n_sectors = int(task["n_sectors"])
    chosen_descriptor = str(task["chosen_descriptor"])
    baseline_descriptor = task["baseline_descriptor"]
    interior_shrink_px = int(task["interior_shrink_px"])
    rim_width_px = int(task["rim_width_px"])
    hotspot_cfg = task["hotspot_cfg"]

    frame_hole_union = make_global_hole_union(shape, frame_holes)
    frame_roi_mask = np.ones(shape, dtype=bool)
    matrix_mask = make_matrix_bulk_mask(frame_roi_mask, frame_hole_union)
    terraces_by_hole = make_nonoverlapping_hole_terraces(
        shape,
        frame_holes,
        n_terraces,
        lattice_indices=lattice_indices,
        width_mode=str(task.get("terrace_width_mode", "fixed")),
        gap_basis=str(task.get("terrace_gap_basis", "border_gap")),
        min_width_px=float(task.get("terrace_min_width_px", 0.0)),
    )

    image_hsv = rgb_to_hsv(frame.image)
    chosen_descriptor_image = descriptor_image(frame.image, image_hsv, chosen_descriptor)
    angular_rows = compute_frame_angular_asymmetry(
        frame.frame_id,
        chosen_descriptor_image,
        frame_holes,
        terraces_by_hole,
        n_sectors=n_sectors,
        lattice_indices=lattice_indices,
    )
    sector_radial_rows = compute_sector_radial_timeseries(
        frame.frame_id,
        chosen_descriptor_image,
        frame_holes,
        terraces_by_hole,
        n_sectors=n_sectors,
        lattice_indices=lattice_indices,
    )

    matrix_row = _matrix_row(frame, image_hsv, matrix_mask)
    matrix_row["primary_descriptor"] = chosen_descriptor
    matrix_row["primary_descriptor_mean"] = float(np.nanmean(chosen_descriptor_image[matrix_mask])) if np.any(matrix_mask) else float("nan")

    compartment_rows: list[dict[str, Any]] = []
    radial_rows: list[dict[str, Any]] = []
    for hole in frame_holes:
        uv = lattice_indices.get(hole.hole_id)
        interior_region = make_hole_interior_region(shape, hole, interior_shrink_px)
        rim_region = make_hole_rim_region(shape, hole, rim_width_px)
        s_inner = compute_region_stats_from_region(frame.frame_id, frame.image, image_hsv, interior_region, f"hole_{hole.hole_id}_interior")
        s_rim = compute_region_stats_from_region(frame.frame_id, frame.image, image_hsv, rim_region, f"hole_{hole.hole_id}_rim")
        compartment_rows.extend(
            [
                {
                    "frame_id": s_inner.frame_id,
                    "hole_id": hole.hole_id,
                    "lattice_u": None if uv is None else uv[0],
                    "lattice_v": None if uv is None else uv[1],
                    "region_id": s_inner.region_id,
                    "mean_R": s_inner.mean_r,
                    "mean_G": s_inner.mean_g,
                    "mean_B": s_inner.mean_b,
                    "mean_H": s_inner.mean_h,
                    "mean_S": s_inner.mean_s,
                    "area_px": s_inner.area_px,
                },
                {
                    "frame_id": s_rim.frame_id,
                    "hole_id": hole.hole_id,
                    "lattice_u": None if uv is None else uv[0],
                    "lattice_v": None if uv is None else uv[1],
                    "region_id": s_rim.region_id,
                    "mean_R": s_rim.mean_r,
                    "mean_G": s_rim.mean_g,
                    "mean_B": s_rim.mean_b,
                    "mean_H": s_rim.mean_h,
                    "mean_S": s_rim.mean_s,
                    "area_px": s_rim.area_px,
                },
            ]
        )

        hole_terraces = terraces_by_hole.get(hole.hole_id, [])
        for terrace_idx, terrace_region in enumerate(hole_terraces):
            stats = compute_region_stats_from_region(
                frame.frame_id,
                frame.image,
                image_hsv,
                terrace_region,
                f"hole_{hole.hole_id}_annulus_{terrace_idx}",
            )
            value = mean_from_region(chosen_descriptor_image, terrace_region)
            row = {
                "frame_id": frame.frame_id,
                "hole_id": hole.hole_id,
                "lattice_u": None if uv is None else uv[0],
                "lattice_v": None if uv is None else uv[1],
                "annulus_id": terrace_idx,
                "descriptor_name": chosen_descriptor,
                "descriptor_value": value,
                "primary_descriptor": chosen_descriptor,
                "mean_R": stats.mean_r,
                "mean_G": stats.mean_g,
                "mean_B": stats.mean_b,
                "mean_H": stats.mean_h,
                "mean_S": stats.mean_s,
                "area_px": stats.area_px,
            }
            radial_rows.append(row)
            compartment_rows.append(
                {
                    "frame_id": frame.frame_id,
                    "hole_id": hole.hole_id,
                    "lattice_u": None if uv is None else uv[0],
                    "lattice_v": None if uv is None else uv[1],
                    "region_id": f"hole_{hole.hole_id}_annulus_{terrace_idx}",
                    "mean_R": stats.mean_r,
                    "mean_G": stats.mean_g,
                    "mean_B": stats.mean_b,
                    "mean_H": stats.mean_h,
                    "mean_S": stats.mean_s,
                    "area_px": stats.area_px,
                }
            )

    score_map = hotspot_score_map(chosen_descriptor_image, baseline_descriptor)
    hotspots = detect_hotspots(frame.frame_id, score_map, matrix_mask, hotspot_cfg, frame_holes, image_rgb=frame.image, image_hsv=image_hsv)
    matrix_row["hotspot_fraction"] = float(sum(h.area_px for h in hotspots) / max(int(matrix_mask.sum()), 1))
    matrix_row["hotspot_count"] = int(len(hotspots))
    return {
        "frame_id": frame.frame_id,
        "compartment_rows": compartment_rows,
        "radial_rows": radial_rows,
        "angular_rows": angular_rows,
        "sector_radial_rows": sector_radial_rows,
        "matrix_row": matrix_row,
        "hotspots": hotspots,
    }


def _summary_payload(
    frames: list[FrameRecord],
    audit_records,
    winner: str,
    candidates: list[HoleCandidate],
    lattice,
    radial_rows: int = 0,
    reference_idx: int = 0,
    registration_mean_residual: float = 0.0,
    n_hotspots: int = 0,
    n_events: int = 0,
    n_geometry_rows: int = 0,
    geometry_sanity: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "n_frames": len(frames),
        "accepted_frames": int(sum(r.accepted for r in audit_records)),
        "rejected_frames": int(sum(not r.accepted for r in audit_records)),
        "winner_correction": winner,
        "n_candidates": len(candidates),
        "lattice_confidence": float(lattice.confidence),
        "spacing_u_px": float(lattice.spacing_u_px),
        "spacing_v_px": float(lattice.spacing_v_px),
        "angle_deg": float(lattice.angle_deg),
        "n_radial_rows": int(radial_rows),
        "reference_frame_id": int(reference_idx),
        "registration_mean_residual": float(registration_mean_residual),
        "n_hotspots": int(n_hotspots),
        "n_events": int(n_events),
        "n_geometry_rows": int(n_geometry_rows),
        "geometry_sanity": geometry_sanity or {},
    }


def _terrace_overlay(image: np.ndarray, holes: list[HoleGeometry], terraces_by_hole, matrix_mask: np.ndarray) -> np.ndarray:
    overlay = image.copy()
    color_cycle = np.array(
        [
            [255, 80, 80],
            [255, 180, 80],
            [255, 255, 80],
            [80, 255, 80],
            [80, 220, 255],
            [80, 120, 255],
            [200, 80, 255],
            [255, 80, 180],
        ],
        dtype=np.uint8,
    )
    overlay[~matrix_mask] = np.clip(0.75 * overlay[~matrix_mask], 0, 255).astype(np.uint8)
    for hole in holes:
        for i, terrace in enumerate(terraces_by_hole.get(hole.hole_id, [])):
            color = color_cycle[i % len(color_cycle)]
            terrace.paint(overlay, color, alpha=0.45)
        cv2.circle(overlay, (int(round(hole.x)), int(round(hole.y))), int(round(hole.radius_outer_px)), (255, 255, 255), 1)
    return overlay


def _hotspot_overlay(image: np.ndarray, hotspots, holes: list[HoleGeometry]) -> np.ndarray:
    overlay = image.copy()
    for hole in holes:
        cv2.circle(overlay, (int(round(hole.x)), int(round(hole.y))), int(round(hole.radius_outer_px)), (255, 255, 255), 1)
    for hs in hotspots:
        cv2.circle(overlay, (int(round(hs.cx)), int(round(hs.cy))), 6, (255, 0, 255), 2)
        cv2.putText(overlay, f"{hs.hotspot_id}", (int(round(hs.cx)) + 4, int(round(hs.cy)) - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)
    return overlay


def _phenotype_color(label: str) -> tuple[int, int, int]:
    palette = {
        "outward_propagating": (80, 220, 255),
        "near_synchronous": (80, 255, 120),
        "outer_first": (255, 180, 80),
        "irregular": (255, 90, 180),
    }
    text = str(label)
    for key, color in palette.items():
        if text == key or text.endswith(f"_{key}"):
            return color
    return palette.get(text, (220, 220, 220))


def _row_phenotype_label(row: dict[str, Any]) -> str:
    return str(row.get("semantic_label") or row.get("phenotype_label") or "unknown")


def _phenotype_overlay(image: np.ndarray, holes: list[HoleGeometry], phenotype_rows: list[dict[str, Any]]) -> np.ndarray:
    overlay = image.copy()
    label_by_hole = {int(r["hole_id"]): _row_phenotype_label(r) for r in phenotype_rows}
    for hole in holes:
        color = _phenotype_color(label_by_hole.get(hole.hole_id, "unknown"))
        cx, cy = int(round(hole.x)), int(round(hole.y))
        rr = max(3, int(round(hole.radius_outer_px)))
        cv2.circle(overlay, (cx, cy), rr, color, -1)
        cv2.circle(overlay, (cx, cy), rr, (255, 255, 255), 1)
    overlay = np.clip(0.45 * image + 0.55 * overlay, 0, 255).astype(np.uint8)
    return overlay


def _reticulum_point(row: dict[str, Any], holes_by_id: dict[int, HoleGeometry], lattice: LatticeModel | None) -> tuple[float, float] | None:
    if row.get("x") is not None and row.get("y") is not None:
        return float(row["x"]), float(row["y"])
    try:
        hole_id = int(row.get("hole_id"))
    except Exception:
        hole_id = -1
    hole = holes_by_id.get(hole_id)
    if hole is not None:
        return float(hole.x), float(hole.y)
    if lattice is not None and row.get("lattice_u") is not None and row.get("lattice_v") is not None:
        u = float(row["lattice_u"])
        v = float(row["lattice_v"])
        return (
            float(lattice.origin_x + u * lattice.basis_u[0] + v * lattice.basis_v[0]),
            float(lattice.origin_y + u * lattice.basis_u[1] + v * lattice.basis_v[1]),
        )
    if row.get("lattice_u") is not None and row.get("lattice_v") is not None:
        return float(row["lattice_u"]), float(row["lattice_v"])
    return None


def _reticulum_canvas_points(
    rows: list[dict[str, Any]],
    holes: list[HoleGeometry] | None,
    lattice: LatticeModel | None,
    shape: tuple[int, int],
) -> list[tuple[dict[str, Any], int, int]]:
    holes_by_id = {int(h.hole_id): h for h in (holes or [])}
    pts: list[tuple[dict[str, Any], float, float]] = []
    for row in rows:
        point = _reticulum_point(row, holes_by_id, lattice)
        if point is not None and np.isfinite(point[0]) and np.isfinite(point[1]):
            pts.append((row, float(point[0]), float(point[1])))
    if not pts:
        return []
    xy = np.asarray([[x, y] for _, x, y in pts], dtype=float)
    min_xy = np.nanmin(xy, axis=0)
    max_xy = np.nanmax(xy, axis=0)
    span = np.maximum(max_xy - min_xy, 1.0)
    margin = 40.0
    usable_w = max(1.0, float(shape[1]) - 2.0 * margin)
    usable_h = max(1.0, float(shape[0]) - 2.0 * margin)
    scale = min(usable_w / span[0], usable_h / span[1])
    drawn_w = span[0] * scale
    drawn_h = span[1] * scale
    offset_x = 0.5 * (float(shape[1]) - drawn_w) - min_xy[0] * scale
    offset_y = 0.5 * (float(shape[0]) - drawn_h) - min_xy[1] * scale
    out: list[tuple[dict[str, Any], int, int]] = []
    for row, x, y in pts:
        out.append((row, int(round(x * scale + offset_x)), int(round(y * scale + offset_y))))
    return out


def _phenotype_reticulum_map(
    phenotype_rows: list[dict[str, Any]],
    holes: list[HoleGeometry] | None = None,
    lattice: LatticeModel | None = None,
    shape: tuple[int, int] = (420, 420),
) -> np.ndarray:
    canvas = np.full((shape[0], shape[1], 3), 255, dtype=np.uint8)
    rows = [r for r in phenotype_rows if r.get("lattice_u") is not None and r.get("lattice_v") is not None]
    if not rows:
        return canvas
    for row, x, y in _reticulum_canvas_points(rows, holes, lattice, shape):
        color = _phenotype_color(_row_phenotype_label(row))
        cv2.circle(canvas, (x, y), 12, color, -1)
        cv2.circle(canvas, (x, y), 12, (60, 60, 60), 1)
        cv2.putText(canvas, str(row.get("hole_id", "?")), (x - 10, y - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (40, 40, 40), 1, cv2.LINE_AA)
    return canvas


def _rdf_uncertainty_reticulum_map(
    rows: list[dict[str, Any]],
    holes: list[HoleGeometry] | None = None,
    lattice: LatticeModel | None = None,
    shape: tuple[int, int] = (420, 420),
) -> np.ndarray:
    canvas = np.full((shape[0], shape[1], 3), 255, dtype=np.uint8)
    pts = [r for r in rows if r.get("lattice_u") is not None and r.get("lattice_v") is not None]
    if not pts:
        return canvas
    vals = np.asarray([float(r.get("rdf_uncertainty_score", np.nan)) for r in pts], dtype=float)
    finite = vals[np.isfinite(vals)]
    vmin = float(np.nanmin(finite)) if finite.size else 0.0
    vmax = float(np.nanmax(finite)) if finite.size else 1.0
    denom = max(vmax - vmin, 1e-6)
    for row, x, y in _reticulum_canvas_points(pts, holes, lattice, shape):
        val = float(row.get("rdf_uncertainty_score", np.nan))
        frac = 0.5 if not np.isfinite(val) else float(np.clip((val - vmin) / denom, 0.0, 1.0))
        color = (int(255 * frac), int(80 + 120 * (1.0 - frac)), int(255 * (1.0 - frac)))
        cv2.circle(canvas, (x, y), 12, color, -1)
        cv2.circle(canvas, (x, y), 12, (60, 60, 60), 1)
        cv2.putText(canvas, str(row.get("hole_id", "?")), (x - 10, y - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (40, 40, 40), 1, cv2.LINE_AA)
    return canvas


def _write_phenotype_archetype_plot(temporal_dir: Path, archetype_rows: list[dict[str, Any]]) -> None:
    if not archetype_rows:
        return
    annuli = sorted({int(r.get("annulus_id", 0)) for r in archetype_rows})
    if not annuli:
        return
    inner_ann = annuli[0]
    outer_ann = annuli[-1]
    labels = sorted({str(r.get("phenotype_label", "unknown")) for r in archetype_rows})
    x_vals, inner_series = line_series_from_rows(
        [r for r in archetype_rows if int(r.get("annulus_id", -1)) == inner_ann],
        group_field="phenotype_label",
        x_field="frame_id",
        y_field="mean_descriptor",
        group_values=labels,
        group_labeler=lambda g: f"{g}_inner",
    )
    _, outer_series = line_series_from_rows(
        [r for r in archetype_rows if int(r.get("annulus_id", -1)) == outer_ann],
        group_field="phenotype_label",
        x_field="frame_id",
        y_field="mean_descriptor",
        group_values=labels,
        group_labeler=lambda g: f"{g}_outer",
    )
    series = inner_series + outer_series
    if series:
        save_line_plot(
            temporal_dir / "phenotype_archetypes.png",
            x_vals,
            series,
            title="Phenotype archetype trajectories",
            ylabel="mean descriptor",
        )




def _write_radial_archetype_plot(radial_dir: Path, archetype_rows: list[dict[str, Any]]) -> None:
    if not archetype_rows:
        return
    labels, counts = count_by_label(archetype_rows, "radial_archetype_label")
    if labels:
        save_bar_plot(
            radial_dir / "radial_archetype_counts.png",
            labels,
            counts,
            title="Per-hole radial archetype counts",
            ylabel="count",
        )


def _write_angular_asymmetry_plot(radial_dir: Path, frame_rows: list[dict[str, Any]]) -> None:
    if not frame_rows:
        return
    frame_rows = sorted(frame_rows, key=lambda r: int(r.get("frame_id", 0)))
    x = [int(r.get("frame_id", 0)) for r in frame_rows]
    save_line_plot(
        radial_dir / "angular_asymmetry_frame_summary.png",
        x,
        [
            ("mean_angular_asymmetry", [r.get("mean_angular_asymmetry", np.nan) for r in frame_rows]),
            ("mean_vector_strength", [r.get("mean_vector_strength", np.nan) for r in frame_rows]),
        ],
        title="Angular asymmetry around holes",
        ylabel="value",
    )


def _write_sector_front_plot(radial_dir: Path, frame_summary_rows: list[dict[str, Any]]) -> None:
    if not frame_summary_rows:
        return
    frame_ids, series = line_series_from_rows(
        frame_summary_rows,
        group_field="hole_id",
        x_field="frame_id",
        y_field="mean_sector_front_radius",
        group_values=sorted({int(r.get("hole_id", 0)) for r in frame_summary_rows})[:6],
        group_labeler=lambda g: f"hole_{g}",
    )
    if series:
        save_line_plot(
            radial_dir / "sector_front_summary.png",
            frame_ids,
            series,
            title="Sector front radius evolution",
            ylabel="mean sector front radius",
        )


def _write_model_fit_quality_plot(radial_dir: Path, model_summary_rows: list[dict[str, Any]]) -> None:
    if not model_summary_rows:
        return
    labels = [f"hole_{int(r.get('hole_id', 0))}" for r in model_summary_rows[:10]]
    vals = [float(r.get("mean_quadratic_r2") if r.get("mean_quadratic_r2") is not None else r.get("mean_linear_r2", 0.0)) for r in model_summary_rows[:10]]
    save_bar_plot(
        radial_dir / "radial_model_fit_quality.png",
        labels,
        vals,
        title="Per-hole radial model fit quality",
        ylabel="R^2",
    )


def _write_hotspot_reticulum_plot(radial_dir: Path, group_payload: list[dict[str, Any]] | dict[str, list[Any]]) -> None:
    if not group_payload:
        return
    if isinstance(group_payload, dict):
        labels = [f"{z}:{b}" for z, b in zip(group_payload.get('reticulum_zone', []), group_payload.get('hotspot_proximity_bucket', []))]
        vals = [float(v or 0.0) if v is not None else 0.0 for v in group_payload.get('mean_delta_center_of_mass', [])]
    else:
        labels = [f"{r.get('reticulum_zone')}:{r.get('hotspot_proximity_bucket')}" for r in group_payload]
        vals = [float(r.get("mean_delta_center_of_mass", 0.0)) for r in group_payload]
    save_bar_plot(
        radial_dir / "hotspot_reticulum_group_comparison.png",
        labels,
        vals,
        title="Radial COM shift by hotspot proximity and reticulum zone",
        ylabel="mean delta center of mass",
    )


def _write_reticulum_group_plot(radial_dir: Path, group_summary_rows: list[dict[str, Any]]) -> None:
    if not group_summary_rows:
        return
    frame_ids, series = line_series_from_rows(
        group_summary_rows,
        group_field="reticulum_zone",
        x_field="frame_id",
        y_field="center_of_mass_annulus",
        group_values=sorted({str(r.get("reticulum_zone", "all")) for r in group_summary_rows}),
    )
    if series:
        save_line_plot(
            radial_dir / "reticulum_group_comparison.png",
            frame_ids,
            series,
            title="Reticulum-group radial COM evolution",
            ylabel="center of mass annulus",
        )


def _write_radial_evolution_artifacts(radial_dir: Path, aggregate_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    frame_rows, summary = summarize_radial_evolution(aggregate_rows)
    write_table(radial_dir / "radial_distribution_evolution.csv", aggregate_rows)
    write_table(radial_dir / "radial_distribution_frame_summary.csv", frame_rows)
    if aggregate_rows:
        matrix, annuli, frame_ids = heatmap_from_rows(
            aggregate_rows,
            row_field="annulus_id",
            col_field="frame_id",
            value_field="mean_descriptor",
        )
        save_heatmap_plot(
            radial_dir / "radial_distribution_evolution.png",
            matrix,
            title="Hole radial distribution evolution",
            xlabel="frame",
            ylabel="annulus",
            xticklabels=[str(x) for x in frame_ids],
            yticklabels=[str(y) for y in annuli],
        )
    return frame_rows, summary


def _write_per_hole_rdf_artifacts(radial_dir: Path, rdf_columns, rdf_frame_rows: list[dict[str, Any]]) -> None:
    write_table_columns(radial_dir / "per_hole_rdf_evolution.csv", rdf_columns.evolution)
    write_table_columns(radial_dir / "per_hole_rdf_frame_summary.csv", rdf_columns.frame_summary)
    write_table_columns(radial_dir / "per_hole_rdf_velocity_summary.csv", rdf_columns.velocity_summary)
    if rdf_columns.evolution.get("hole_id"):
        first_hole = sorted({int(v) for v in rdf_columns.evolution.get("hole_id", [])})[0]
        matrix, annuli, frame_ids = heatmap_from_columns(
            rdf_columns.evolution,
            row_field="annulus_id",
            col_field="frame_id",
            value_field="rdf_pdf",
            filter_field="hole_id",
            filter_value=first_hole,
        )
        save_heatmap_plot(
            radial_dir / "per_hole_rdf_evolution.png",
            matrix,
            title=f"Hole {first_hole} normalized radial distribution evolution",
            xlabel="frame",
            ylabel="annulus",
            xticklabels=[str(x) for x in frame_ids],
            yticklabels=[str(y) for y in annuli],
        )
    labels, vals = bar_from_columns(rdf_columns.velocity_summary, label_field="hole_id", value_field="rdf_front_velocity_per_frame", limit=20)
    if labels:
        save_bar_plot(
            radial_dir / "per_hole_rdf_front_velocity.png",
            labels,
            vals,
            title="Per-hole RDF front velocity",
            ylabel="normalized radius / frame",
        )


def _write_per_hole_rdf_archetype_plot(radial_dir: Path, rdf_archetype_rows: list[dict[str, Any]]) -> None:
    if not rdf_archetype_rows:
        return
    labels, counts = count_by_label(rdf_archetype_rows, "rdf_archetype_label")
    save_bar_plot(
        radial_dir / "rdf_archetype_counts.png",
        labels,
        counts,
        title="Per-hole RDF archetype counts",
        ylabel="count",
    )


def _write_rdf_front_dynamics_plot(radial_dir: Path, dynamics_rows: list[dict[str, Any]]) -> None:
    if not dynamics_rows:
        return
    labels = [f"hole_{int(r.get('hole_id', 0))}" for r in dynamics_rows[:10]]
    vals = [float(r.get("rdf_front_acceleration_per_frame2", 0.0) or 0.0) for r in dynamics_rows[:10]]
    save_bar_plot(
        radial_dir / "rdf_front_acceleration.png",
        labels,
        vals,
        title="Per-hole RDF front acceleration",
        ylabel="acceleration / frame²",
    )


def _write_sector_rdf_plot(radial_dir: Path, sector_frame_payload: list[dict[str, Any]] | dict[str, list[Any]]) -> None:
    if not sector_frame_payload:
        return
    if isinstance(sector_frame_payload, dict):
        hole_vals = sector_frame_payload.get("hole_id", [])
        if not hole_vals:
            return
        first_hole = sorted({int(v) for v in hole_vals})[0]
        sector_vals = [int(v) for v, h in zip(sector_frame_payload.get("sector_id", []), hole_vals) if int(h) == first_hole]
        frame_ids, series = line_series_from_columns(
            sector_frame_payload,
            group_field="sector_id",
            x_field="frame_id",
            y_field="sector_rdf_front_radius_norm",
            filter_field="hole_id",
            filter_value=first_hole,
            group_values=sorted(set(sector_vals))[:8],
            group_labeler=lambda g: f"sector_{g}",
        )
    else:
        if not sector_frame_payload:
            return
        first_hole = sorted({int(r.get("hole_id", 0)) for r in sector_frame_payload})[0]
        rows = [r for r in sector_frame_payload if int(r.get("hole_id", -1)) == first_hole]
        frame_ids, series = line_series_from_rows(
            rows,
            group_field="sector_id",
            x_field="frame_id",
            y_field="sector_rdf_front_radius_norm",
            group_values=sorted({int(r.get("sector_id", 0)) for r in rows})[:8],
            group_labeler=lambda g: f"sector_{g}",
        )
    if series:
        save_line_plot(
            radial_dir / "sector_rdf_evolution.png",
            frame_ids,
            series,
            title=f"Hole {first_hole} sector RDF evolution",
            ylabel="sector RDF front radius",
        )


def _write_rdf_stability_plot(radial_dir: Path, stability_summary_payload: list[dict[str, Any]] | dict[str, list[Any]]) -> None:
    if not stability_summary_payload:
        return
    if isinstance(stability_summary_payload, dict):
        columns = stability_summary_payload
    else:
        columns = columns_from_records(stability_summary_payload, [
            'brightness_factor', 'radius_scale', 'rdf_archetype_stability_fraction', 'is_base',
        ])
    if not columns or not columns.get('rdf_archetype_stability_fraction'):
        return
    labels_all = [f"b{float(b or 1.0):.2f}|r{float(r or 1.0):.2f}" for b, r in zip(columns.get('brightness_factor', []), columns.get('radius_scale', []))]
    vals_all = [float(v or 0.0) if v is not None else 0.0 for v in columns.get('rdf_archetype_stability_fraction', [])]
    base_mask = [bool(v) for v in columns.get('is_base', [False] * len(labels_all))]
    labels = [lab for lab, is_base in zip(labels_all, base_mask) if not is_base]
    vals = [val for val, is_base in zip(vals_all, base_mask) if not is_base]
    if not labels:
        labels, vals = labels_all, vals_all
    save_bar_plot(radial_dir / "per_hole_rdf_stability.png", labels, vals, title="Per-hole RDF archetype stability across sweeps", ylabel="stability fraction")


def _write_sector_front_lag_map(radial_dir: Path, lag_payload: list[dict[str, Any]] | dict[str, list[Any]]) -> None:
    if not lag_payload:
        return
    if isinstance(lag_payload, dict):
        matrix, hole_ids, sectors = heatmap_from_columns(
            lag_payload,
            row_field="hole_id",
            col_field="sector_id",
            value_field="sector_onset_lag",
        )
    else:
        matrix, hole_ids, sectors = heatmap_from_rows(
            lag_payload,
            row_field="hole_id",
            col_field="sector_id",
            value_field="sector_onset_lag",
        )
    save_heatmap_plot(radial_dir / "sector_front_lag_map.png", matrix, title="Sector-front onset lag map", xlabel="sector", ylabel="hole", xticklabels=[str(s) for s in sectors], yticklabels=[str(h) for h in hole_ids])


def _write_rdf_hotspot_reticulum_plot(radial_dir: Path, group_payload: list[dict[str, Any]] | dict[str, list[Any]]) -> None:
    if not group_payload:
        return
    if isinstance(group_payload, dict):
        labels = [f"{a}|{z}|{b}" for a, z, b in zip(group_payload.get('rdf_archetype_canonical_label', []), group_payload.get('reticulum_zone', []), group_payload.get('hotspot_proximity_bucket', []))]
        vals = [float(v or 0.0) if v is not None else 0.0 for v in group_payload.get('mean_delta_front_radius_norm', [])]
    else:
        labels = [f"{r.get('rdf_archetype_canonical_label')}|{r.get('reticulum_zone')}|{r.get('hotspot_proximity_bucket')}" for r in group_payload]
        vals = [float(r.get("mean_delta_front_radius_norm", 0.0) or 0.0) for r in group_payload]
    save_bar_plot(radial_dir / "rdf_hotspot_reticulum_group_comparison.png", labels, vals, title="RDF class by hotspot proximity and reticulum zone", ylabel="mean Δ front radius")

def _write_rdf_bootstrap_plot(radial_dir: Path, bootstrap_payload: list[dict[str, Any]] | dict[str, list[Any]]) -> None:
    if not bootstrap_payload:
        return
    if isinstance(bootstrap_payload, dict):
        labels, vals = bar_from_columns(bootstrap_payload, label_field='label', value_field='bootstrap_front_velocity_ci_width', limit=12)
    else:
        labels = [f"hole_{int(r.get('hole_id', 0))}" for r in bootstrap_payload[:12]]
        vals = [float(r.get('bootstrap_front_velocity_ci_width', 0.0) or 0.0) for r in bootstrap_payload[:12]]
    if not labels:
        return
    save_bar_plot(radial_dir / "per_hole_rdf_bootstrap_ci.png", labels, vals, title="Per-hole RDF front velocity CI width", ylabel="95% CI width")


def _write_rdf_bootstrap_support_plot(radial_dir: Path, support_payload: list[dict[str, Any]] | dict[str, list[Any]]) -> None:
    if not support_payload:
        return
    if isinstance(support_payload, dict):
        labels, vals = bar_from_columns(support_payload, label_field='label', value_field='bootstrap_rdf_archetype_support_fraction', limit=12)
    else:
        labels = [f"hole_{int(r.get('hole_id', 0))}" for r in support_payload[:12]]
        vals = [float(r.get('bootstrap_rdf_archetype_support_fraction', 0.0) or 0.0) for r in support_payload[:12]]
    if not labels:
        return
    save_bar_plot(radial_dir / "rdf_archetype_bootstrap_support.png", labels, vals, title="RDF archetype bootstrap support", ylabel="support fraction")


def _write_rdf_uncertainty_hotspot_plot(radial_dir: Path, group_payload: list[dict[str, Any]] | dict[str, list[Any]]) -> None:
    if not group_payload:
        return
    if isinstance(group_payload, dict):
        labels = [f"{z}:{b}" for z, b in zip(group_payload.get('reticulum_zone', []), group_payload.get('hotspot_proximity_bucket', []))]
        vals = [float(v or 0.0) if v is not None else 0.0 for v in group_payload.get('mean_bootstrap_front_velocity_ci_width', [])]
    else:
        labels = [f"{r.get('reticulum_zone')}:{r.get('hotspot_proximity_bucket')}" for r in group_payload]
        vals = [float(r.get('mean_bootstrap_front_velocity_ci_width', 0.0) or 0.0) for r in group_payload]
    save_bar_plot(
        radial_dir / "rdf_uncertainty_hotspot_group_comparison.png",
        labels,
        vals,
        title="Bootstrap RDF uncertainty vs hotspot proximity",
        ylabel="mean velocity CI width",
    )


def _write_sector_acceleration_plot(radial_dir: Path, rows: list[dict[str, Any]] | dict[str, list[Any]]) -> None:
    if not rows:
        return
    if isinstance(rows, dict):
        labels, vals = bar_from_columns(rows, label_field='label', value_field='mean_sector_front_acceleration_per_frame2', limit=12)
    else:
        labels = [f"hole_{int(r.get('hole_id', 0))}" for r in rows[:12]]
        vals = [float(r.get('mean_sector_front_acceleration_per_frame2', 0.0) or 0.0) for r in rows[:12]]
    if not labels:
        return
    save_bar_plot(
        radial_dir / "sector_front_acceleration.png",
        labels,
        vals,
        title="Mean sector front acceleration by hole",
        ylabel="acceleration / frame²",
    )


def _write_sector_propagation_plot(radial_dir: Path, sector_hole_rows: list[dict[str, Any]] | dict[str, list[Any]]) -> None:
    if not sector_hole_rows:
        return
    if isinstance(sector_hole_rows, dict):
        labels, vals = bar_from_columns(sector_hole_rows, label_field='label', value_field='sector_front_velocity_anisotropy', limit=12)
    else:
        labels = [f"hole_{int(r.get('hole_id', 0))}" for r in sector_hole_rows[:12]]
        vals = [float(r.get('sector_front_velocity_anisotropy', 0.0) or 0.0) for r in sector_hole_rows[:12]]
    if not labels:
        return
    save_bar_plot(radial_dir / "sector_front_propagation.png", labels, vals, title="Sector front propagation anisotropy", ylabel="velocity anisotropy")


def _write_radial_drift_map(qc_dir: Path, drift_rows: list[dict[str, Any]]) -> None:
    if not drift_rows:
        return
    matrix, sweep_ids, frame_ids = heatmap_from_rows(
        drift_rows,
        row_field="sweep_id",
        col_field="frame_id",
        value_field="center_of_mass_delta",
    )
    save_heatmap_plot(
        qc_dir / "radial_perturbation_drift_map.png",
        matrix,
        title="Radial perturbation drift map",
        xlabel="frame",
        ylabel="sweep",
        xticklabels=[str(x) for x in frame_ids],
        yticklabels=[str(y) for y in sweep_ids],
    )


def _matrix_row(frame: FrameRecord, image_hsv: np.ndarray, matrix_mask: np.ndarray) -> dict[str, Any]:
    stats = compute_region_stats(frame.frame_id, frame.image, image_hsv, matrix_mask, "matrix_bulk")
    return {
        "frame_id": frame.frame_id,
        "mean_R": stats.mean_r,
        "mean_G": stats.mean_g,
        "mean_B": stats.mean_b,
        "mean_H": stats.mean_h,
        "mean_S": stats.mean_s,
        "area_px": stats.area_px,
    }


def _write_audit_artifacts(out_dir: Path, audit_records) -> None:
    write_table(out_dir / "frame_qc.csv", audit_records)
    x = [r.frame_id for r in audit_records]
    save_line_plot(
        out_dir / "plots" / "focus_plot.png",
        x,
        [("blur_score", [r.blur_score for r in audit_records])],
        title="Focus / blur score by frame",
        ylabel="blur score",
    )
    save_line_plot(
        out_dir / "plots" / "saturation_plot.png",
        x,
        [
            ("sat_r", [r.sat_frac_r for r in audit_records]),
            ("sat_g", [r.sat_frac_g for r in audit_records]),
            ("sat_b", [r.sat_frac_b for r in audit_records]),
        ],
        title="Channel saturation fraction",
        ylabel="fraction",
    )


def _write_photometry_artifacts(out_dir: Path, frames: list[FrameRecord], winner: str, photo_scores) -> None:
    write_table(out_dir / "candidate_scores.csv", photo_scores)
    write_json(out_dir / "winner.json", {"winner": winner})
    save_frame(out_dir / "previews" / "frame0_raw.png", frames[0].image)
    by_method: dict[str, list[float]] = {}
    for score in photo_scores:
        by_method.setdefault(score.correction_name, []).append(score.total_score)
    save_bar_plot(
        out_dir / "candidate_score_means.png",
        list(by_method.keys()),
        [float(np.mean(v)) for v in by_method.values()],
        title="Mean photometry score by correction",
        ylabel="mean total score",
    )


def _write_registration_artifacts(out_dir: Path, reference_idx: int, corrected: list[FrameRecord], stabilized: list[FrameRecord], transforms) -> float:
    rows = []
    residuals = []
    ref = stabilized[reference_idx].image
    for frame, tfm, st in zip(corrected, transforms, stabilized):
        rows.append(TransformRecord(frame.frame_id, float(tfm.dx), float(tfm.dy), float(tfm.angle_deg), float(tfm.scale)))
        residuals.append(residual_difference(st.image, ref))
    write_table(out_dir / "transforms.csv", rows)
    write_json(out_dir / "reference.json", {"reference_frame_id": int(corrected[reference_idx].frame_id)})
    save_frame(out_dir / "previews" / "frame_ref_corrected.png", corrected[reference_idx].image)
    save_frame(out_dir / "previews" / "frame_ref_stabilized.png", stabilized[reference_idx].image)
    save_frame(out_dir / "previews" / "frame_last_stabilized.png", stabilized[-1].image)
    return float(np.mean(residuals))


def _pick_primary_descriptor(ref_frame: FrameRecord, holes: list[HoleGeometry], matrix_mask: np.ndarray, cfg: PipelineConfig) -> tuple[str, list[dict[str, float]]]:
    image_hsv = rgb_to_hsv(ref_frame.image)
    if cfg.descriptors.primary_descriptor != "auto":
        return cfg.descriptors.primary_descriptor, [{"descriptor": cfg.descriptors.primary_descriptor, "score": float("nan")}]
    if not holes:
        return "s", [{"descriptor": "s", "score": float("nan")}]
    interior_union = np.zeros(ref_frame.image.shape[:2], dtype=bool)
    for hole in holes:
        region = make_hole_interior_region(ref_frame.image.shape[:2], hole, cfg.masks.interior_shrink_px)
        if region.mask.size:
            interior_union[region.y0:region.y1, region.x0:region.x1] |= region.mask
    named = {}
    for name in ["r", "g", "b", "h", "s"]:
        arr = descriptor_image(ref_frame.image, image_hsv, name)
        named[name] = (arr[interior_union], arr[matrix_mask])
    ranked = rank_descriptors(named)
    return ranked[0][1], [{"descriptor": name, "score": float(score)} for score, name in ranked]


def _maybe_assert_gates(cfg: PipelineConfig, gates) -> None:
    if getattr(cfg.qc, "fail_on_gate_error", False):
        assert_gate_results(gates)


def _make_hole_lattice_rows(holes: list[HoleGeometry], lattice_indices: dict[int, tuple[int, int]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for hole in holes:
        uv = lattice_indices.get(hole.hole_id)
        rows.append(
            {
                "hole_id": hole.hole_id,
                "lattice_u": None if uv is None else uv[0],
                "lattice_v": None if uv is None else uv[1],
                "x_ref": hole.x,
                "y_ref": hole.y,
                "radius_inner_ref_px": hole.radius_inner_px,
                "radius_outer_ref_px": hole.radius_outer_px,
                "confidence_ref": hole.confidence,
            }
        )
    return rows


def _propagate_holes_across_frames(
    stabilized: list[FrameRecord],
    ref_idx: int,
    ref_holes: list[HoleGeometry],
    cfg: PipelineConfig,
    progress_cfg=None,
    progress_callback=None,
) -> dict[int, list[HoleGeometry]]:
    propagation_mode = str(getattr(cfg.geometry, "propagation_mode", "canonical_static"))
    if propagation_mode == "canonical_static":
        ref_sorted = sorted(ref_holes, key=lambda h: h.hole_id)
        holes_by_frame = {
            frame.frame_id: [
                HoleGeometry(
                    hole_id=h.hole_id,
                    x=float(h.x),
                    y=float(h.y),
                    radius_inner_px=float(h.radius_inner_px),
                    radius_outer_px=float(h.radius_outer_px),
                    confidence=float(h.confidence),
                )
                for h in ref_sorted
            ]
            for frame in stabilized
        }
        if progress_callback is not None:
            progress_callback(max(1, len(stabilized) - 1), max(1, len(stabilized) - 1))
        return holes_by_frame

    holes_by_index: dict[int, list[HoleGeometry]] = {ref_idx: sorted(ref_holes, key=lambda h: h.hole_id)}
    forward_indices = list(range(ref_idx + 1, len(stabilized)))
    backward_indices = list(range(ref_idx - 1, -1, -1))
    total = len(forward_indices) + len(backward_indices)
    done = 0
    for idx in iter_with_progress(forward_indices, total=len(forward_indices), cfg=progress_cfg, desc="Geometry propagate fwd"):
        holes_by_index[idx] = propagate_geometry_to_frame(holes_by_index[idx - 1], stabilized[idx].image, search_radius_px=cfg.geometry.propagation_search_px)
        done += 1
        if progress_callback is not None:
            progress_callback(done, total)
    for idx in iter_with_progress(backward_indices, total=len(backward_indices), cfg=progress_cfg, desc="Geometry propagate back"):
        holes_by_index[idx] = propagate_geometry_to_frame(holes_by_index[idx + 1], stabilized[idx].image, search_radius_px=cfg.geometry.propagation_search_px)
        done += 1
        if progress_callback is not None:
            progress_callback(done, total)
    holes_by_frame = {stabilized[idx].frame_id: holes for idx, holes in holes_by_index.items()}
    return smooth_hole_trajectories(holes_by_frame, window=cfg.geometry.smoothing_window)


def _geometry_timeseries_rows(
    stabilized: list[FrameRecord],
    holes_by_frame: dict[int, list[HoleGeometry]],
    lattice_indices: dict[int, tuple[int, int]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for frame in stabilized:
        for hole in holes_by_frame[frame.frame_id]:
            uv = lattice_indices.get(hole.hole_id)
            rows.append(
                {
                    "frame_id": frame.frame_id,
                    "hole_id": hole.hole_id,
                    "lattice_u": None if uv is None else uv[0],
                    "lattice_v": None if uv is None else uv[1],
                    "x": hole.x,
                    "y": hole.y,
                    "radius_inner_px": hole.radius_inner_px,
                    "radius_outer_px": hole.radius_outer_px,
                    "confidence": hole.confidence,
                }
            )
    return rows


def _radial_plot_for_first_hole(radial_dir: Path, radial_rows: list[dict[str, Any]] | RadialRowTable, first_hole_id: int, chosen_descriptor: str, n_terraces: int) -> None:
    if isinstance(radial_rows, RadialRowTable):
        x_vals, raw_series = radial_rows.first_hole_series(first_hole_id)
        if not x_vals or not raw_series:
            return
        series = []
        present = {name: values for name, values in raw_series}
        for ann in range(n_terraces):
            name = f"annulus_{ann}"
            values = present.get(name)
            if values is None:
                values = [None for _ in x_vals]
            series.append((name, values))
    else:
        first_rows = [r for r in radial_rows if r["hole_id"] == first_hole_id]
        if not first_rows:
            return
        x_vals = sorted(set(r["frame_id"] for r in first_rows))
        series = []
        for ann in range(n_terraces):
            ann_rows = sorted([r for r in first_rows if r["annulus_id"] == ann], key=lambda r: r["frame_id"])
            series.append((f"annulus_{ann}", [r["descriptor_value"] for r in ann_rows]))
    save_line_plot(
        radial_dir / f"hole_{first_hole_id}_{chosen_descriptor}_by_annulus.png",
        x_vals,
        series,
        title=f"Hole {first_hole_id} {chosen_descriptor} by annulus",
        ylabel=f"mean {chosen_descriptor}",
    )


def _lag_summary_from_radial_rows(radial_rows: list[dict[str, Any]], holes: list[HoleGeometry], chosen_descriptor: str, n_terraces: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    temporal_validation_summary = TemporalValidationSummary(
        valid_onset_fraction=0.0,
        valid_peak_fraction=0.0,
        lag_monotonic_fraction=0.0,
        negative_lag_fraction=0.0,
        phenotype_stability_fraction=0.0,
        phenotype_canonical_agreement=0.0,
        phenotype_neighbor_fraction=0.0,
        phenotype_spatial_fraction=0.0,
    )

    event_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    detail_rows: list[dict[str, Any]] = []
    if not radial_rows or not holes:
        return event_rows, summary_rows, detail_rows
    for hole in holes:
        hole_rows = [r for r in radial_rows if r["hole_id"] == hole.hole_id]
        if not hole_rows:
            continue
        annulus_events: list[dict[str, Any]] = []
        for ann in range(n_terraces):
            vals = [r["descriptor_value"] for r in sorted(hole_rows, key=lambda r: (r["frame_id"], r["annulus_id"])) if r["annulus_id"] == ann]
            if not vals:
                continue
            ev = summarize_curve_events(f"hole_{hole.hole_id}_annulus_{ann}_{chosen_descriptor}", np.asarray(vals, dtype=float))
            ev_row = asdict(ev)
            ev_row["hole_id"] = hole.hole_id
            ev_row["annulus_id"] = ann
            ev_row["descriptor"] = chosen_descriptor
            annulus_events.append(ev_row)
            detail_rows.append(ev_row)
            event_rows.append(ev_row)
        if not annulus_events:
            continue
        annulus_events = sorted(annulus_events, key=lambda r: r["annulus_id"])
        valid_onsets = [r["onset_frame"] for r in annulus_events if r["onset_frame"] is not None]
        valid_peaks = [r["peak_frame"] for r in annulus_events if r["peak_frame"] is not None]
        inner = annulus_events[0]
        outer = annulus_events[-1]
        onset_lag = None if inner["onset_frame"] is None or outer["onset_frame"] is None else int(outer["onset_frame"] - inner["onset_frame"])
        peak_lag = None if inner["peak_frame"] is None or outer["peak_frame"] is None else int(outer["peak_frame"] - inner["peak_frame"])
        monotonic_onset = bool(all(a <= b for a, b in zip(valid_onsets, valid_onsets[1:]))) if len(valid_onsets) >= 2 else True
        monotonic_peak = bool(all(a <= b for a, b in zip(valid_peaks, valid_peaks[1:]))) if len(valid_peaks) >= 2 else True
        summary_rows.append(
            {
                "hole_id": hole.hole_id,
                "descriptor": chosen_descriptor,
                "inner_onset_frame": inner["onset_frame"],
                "outer_onset_frame": outer["onset_frame"],
                "onset_lag_frames": onset_lag,
                "inner_peak_frame": inner["peak_frame"],
                "outer_peak_frame": outer["peak_frame"],
                "peak_lag_frames": peak_lag,
                "n_valid_annuli_onset": len(valid_onsets),
                "n_valid_annuli_peak": len(valid_peaks),
                "monotonic_onset": monotonic_onset,
                "monotonic_peak": monotonic_peak,
                "peak_value_outer": outer["peak_value"],
                "baseline_value_inner": inner["baseline_value"],
            }
        )
    return event_rows, summary_rows, detail_rows


def run_milestone1(frames: list[FrameRecord], out_dir: Path, cfg: PipelineConfig | None = None) -> dict[str, Any]:
    if not frames:
        raise ValueError("frames cannot be empty")
    cfg = cfg or PipelineConfig()
    out_dir = ensure_dir(out_dir)
    audit_dir, photometry_dir, geometry_dir = ensure_subdirs(out_dir, "audit", "photometry", "geometry")
    descriptors_dir = ensure_subdirs(out_dir, "descriptors")[0]
    ensure_subdirs(audit_dir, "plots")
    ensure_subdirs(photometry_dir, "previews")
    ensure_subdirs(geometry_dir, "overlays")

    audit_records = audit_sequence(frames, cfg.audit)
    _write_audit_artifacts(audit_dir, audit_records)

    winner, photo_scores, corrected = run_photometry_selection(frames, cfg.photometry)
    _write_photometry_artifacts(photometry_dir, frames, winner, photo_scores)
    save_frame(photometry_dir / "previews" / f"frame0_{winner}.png", corrected[0].image)

    candidates = detect_dark_hole_candidates(corrected[0].image, cfg.geometry)
    gray = cv2.cvtColor(corrected[0].image, cv2.COLOR_RGB2GRAY)
    holes = _refined_holes_from_candidates(gray, candidates)
    lattice = estimate_lattice_basis(candidates, angle_tolerance_deg=cfg.geometry.angle_tolerance_deg)
    lattice_indices = assign_lattice_indices(candidates, lattice)
    matrix_mask = make_matrix_bulk_mask(np.ones(corrected[0].image.shape[:2], dtype=bool), make_global_hole_union(corrected[0].image.shape[:2], holes)) if holes else np.ones(corrected[0].image.shape[:2], dtype=bool)
    chosen_descriptor, descriptor_ranking = _pick_primary_descriptor(corrected[0], holes, matrix_mask, cfg)
    write_json(descriptors_dir / "descriptor_selection.json", {"chosen_descriptor": chosen_descriptor, "ranking": descriptor_ranking})
    write_table(geometry_dir / "hole_candidates.csv", _candidate_table(candidates, lattice_indices))
    write_table(geometry_dir / "hole_geometry.csv", holes)
    write_json(geometry_dir / "lattice_model.json", asdict(lattice))
    overlay = draw_candidates(corrected[0].image, candidates, lattice=lattice, lattice_indices=lattice_indices)
    save_image(geometry_dir / "overlays" / "frame0_geometry_overlay.png", overlay)

    gates = [
        require(any(r.accepted for r in audit_records), "audit_has_accepted_frames", "At least one frame passed audit."),
        require(chosen_descriptor in {"r", "g", "b", "h", "s"}, "descriptor_selected", f"Chosen descriptor={chosen_descriptor}."),
        require(len(descriptor_ranking) > 0, "descriptor_ranking_written", f"Descriptor ranking rows={len(descriptor_ranking)}."),
        require(winner in cfg.photometry.candidate_methods, "photometry_winner_valid", f"Winner '{winner}' is configured."),
        require(len(candidates) >= 4, "geometry_min_candidates", f"Detected {len(candidates)} candidates."),
        require(lattice.confidence >= 0.0, "lattice_fitted", f"Lattice confidence={lattice.confidence:.3f}."),
    ]
    write_gate_report(out_dir / "qc_gates.json", gates)

    summary = _summary_payload(frames, audit_records, winner, candidates, lattice)
    write_json(out_dir / "summary.json", summary)
    write_json(out_dir / "run_manifest.json", {"pipeline_config": asdict(cfg), "summary": summary})
    return summary

def run_milestone2(frames: list[FrameRecord], out_dir: Path, cfg: PipelineConfig | None = None) -> dict[str, Any]:
    return run_milestone16(frames, out_dir, cfg)


def run_milestone3(frames: list[FrameRecord], out_dir: Path, cfg: PipelineConfig | None = None) -> dict[str, Any]:
    return run_milestone16(frames, out_dir, cfg)


def run_milestone4(frames: list[FrameRecord], out_dir: Path, cfg: PipelineConfig | None = None) -> dict[str, Any]:
    return run_milestone16(frames, out_dir, cfg)


def run_milestone7(frames: list[FrameRecord], out_dir: Path, cfg: PipelineConfig | None = None) -> dict[str, Any]:
    return run_milestone16(frames, out_dir, cfg)


def run_milestone8(frames: list[FrameRecord], out_dir: Path, cfg: PipelineConfig | None = None) -> dict[str, Any]:
    return run_milestone16(frames, out_dir, cfg)


def run_milestone9(frames: list[FrameRecord], out_dir: Path, cfg: PipelineConfig | None = None) -> dict[str, Any]:
    return run_milestone16(frames, out_dir, cfg)


def run_milestone10(frames: list[FrameRecord], out_dir: Path, cfg: PipelineConfig | None = None) -> dict[str, Any]:
    return run_milestone16(frames, out_dir, cfg)


def run_milestone11(frames: list[FrameRecord], out_dir: Path, cfg: PipelineConfig | None = None) -> dict[str, Any]:
    return run_milestone16(frames, out_dir, cfg)


def run_milestone12(frames: list[FrameRecord], out_dir: Path, cfg: PipelineConfig | None = None) -> dict[str, Any]:
    return run_milestone16(frames, out_dir, cfg)


def run_milestone13(frames: list[FrameRecord], out_dir: Path, cfg: PipelineConfig | None = None) -> dict[str, Any]:
    return run_milestone16(frames, out_dir, cfg)


def run_milestone14(frames: list[FrameRecord], out_dir: Path, cfg: PipelineConfig | None = None) -> dict[str, Any]:
    return run_milestone16(frames, out_dir, cfg)


def run_milestone15(frames: list[FrameRecord], out_dir: Path, cfg: PipelineConfig | None = None) -> dict[str, Any]:
    return run_milestone16(frames, out_dir, cfg)


def run_milestone16(frames: list[FrameRecord], out_dir: Path, cfg: PipelineConfig | None = None) -> dict[str, Any]:
    return run_milestone17(frames, out_dir, cfg)


def run_milestone17(frames: list[FrameRecord], out_dir: Path, cfg: PipelineConfig | None = None) -> dict[str, Any]:
    return run_milestone18(frames, out_dir, cfg)


def run_milestone18(frames: list[FrameRecord], out_dir: Path, cfg: PipelineConfig | None = None, hole_grid_bundle: HoleGridBundle | None = None) -> dict[str, Any]:

    if not frames:
        raise ValueError("frames cannot be empty")
    cfg = cfg or PipelineConfig()
    out_dir = ensure_dir(out_dir)
    audit_dir, photometry_dir, registration_dir, geometry_dir, masks_dir, descriptors_dir, radial_dir, hotspots_dir, temporal_dir, qc_dir = ensure_subdirs(
        out_dir, "audit", "photometry", "registration", "geometry", "masks", "descriptors", "radial", "hotspots", "temporal", "qc"
    )
    ensure_subdirs(audit_dir, "plots")
    ensure_subdirs(photometry_dir, "previews")
    ensure_subdirs(registration_dir, "previews")
    ensure_subdirs(geometry_dir, "overlays")
    ensure_subdirs(masks_dir, "overlays")
    ensure_subdirs(hotspots_dir, "overlays")

    stage_names = [
        "Audit",
        "Photometry",
        "Registration",
        "Reference geometry",
        "Descriptor selection",
        "Geometry propagation",
    ]
    if cfg.wafer_nonhole_colour.enabled:
        stage_names.extend(["Wafer non-hole region", "Wafer non-hole colour", "Wafer non-hole recolour video", "Wafer non-hole context"])
    stage_names.extend([
        "Reference terraces",
        *(["Radial cluster average hole"] if cfg.radial_cluster_average_hole.enabled else []),
        "Per-frame analysis",
        "Hotspot linking",
        "Radial summaries",
        "RDF summaries",
        "Validation sweeps",
        "Hotspot and perturbation outputs",
        "Temporal phenotypes",
        *(["Cluster RDF visualisations"] if cfg.visualisation.cluster_rdf.enabled else []),
        "QC and manifest",
    ])
    pipeline_progress = PipelineProgress(
        out_dir,
        stages=stage_names,
        enabled=cfg.parallel.show_progress,
        heartbeat_interval_s=cfg.parallel.status_heartbeat_interval_s,
        progress_mininterval_s=cfg.parallel.progress_mininterval_s,
    )

    quiet_progress = _quiet_progress_cfg(cfg.parallel)
    source_frames_digest = _frames_content_digest(frames)

    with pipeline_progress.stage("Audit", total=len(frames), message="Quality control over raw frames") as tracker:
        audit_signature = _signature_for_stage("Audit", frames_digest=source_frames_digest, audit=asdict(cfg.audit))
        audit_required = [audit_dir / "frame_qc.csv"]
        if _checkpoint_matches(cfg, qc_dir, "audit", audit_signature, audit_required):
            tracker.progress(1, max(1, len(frames)), message="Audit checkpoint hit; loading frame QC table")
            audit_records = _audit_records_from_rows(_read_table_rows(audit_dir / "frame_qc.csv"))
            tracker.progress(max(1, len(frames)), max(1, len(frames)), message="Audit checkpoint loaded")
            _write_checkpoint(cfg, out_dir, qc_dir, "audit", audit_signature, {"frame_qc": audit_dir / "frame_qc.csv"}, reused=True)
        else:
            audit_records = audit_sequence(frames, cfg.audit, progress_cfg=quiet_progress, progress_callback=lambda c, t: tracker.progress(c, t))
            _write_audit_artifacts(audit_dir, audit_records)
            _write_checkpoint(cfg, out_dir, qc_dir, "audit", audit_signature, {"frame_qc": audit_dir / "frame_qc.csv"})

    with pipeline_progress.stage("Photometry", total=len(cfg.photometry.candidate_methods), message="Selecting the photometric correction") as tracker:
        photometry_signature = _signature_for_stage("Photometry", frames_digest=source_frames_digest, photometry=asdict(cfg.photometry))
        photometry_required = [photometry_dir / "winner.json", photometry_dir / "candidate_scores.csv"]
        if _checkpoint_matches(cfg, qc_dir, "photometry", photometry_signature, photometry_required):
            tracker.progress(1, max(2, len(cfg.photometry.candidate_methods)), message="Photometry checkpoint hit; loading selected correction")
            winner_payload = _read_json(photometry_dir / "winner.json", {})
            winner = str(winner_payload.get("winner", "none"))
            photo_scores = _photometry_scores_from_rows(_read_table_rows(photometry_dir / "candidate_scores.csv"))
            corrected = apply_correction_stack(frames, winner, parallel_cfg=quiet_progress)
            if corrected:
                save_frame(photometry_dir / "previews" / f"frame0_{winner}.png", corrected[0].image)
            tracker.progress(max(2, len(cfg.photometry.candidate_methods)), max(2, len(cfg.photometry.candidate_methods)), message="Photometry checkpoint loaded")
            _write_checkpoint(cfg, out_dir, qc_dir, "photometry", photometry_signature, {"winner": photometry_dir / "winner.json", "scores": photometry_dir / "candidate_scores.csv"}, reused=True)
        else:
            winner, photo_scores, corrected = run_photometry_selection(
                frames,
                cfg.photometry,
                progress_cfg=quiet_progress,
                progress_callback=lambda c, t, message=None: tracker.progress(c, t, message=message),
            )
            _write_photometry_artifacts(photometry_dir, frames, winner, photo_scores)
            save_frame(photometry_dir / "previews" / f"frame0_{winner}.png", corrected[0].image)
            _write_checkpoint(cfg, out_dir, qc_dir, "photometry", photometry_signature, {"winner": photometry_dir / "winner.json", "scores": photometry_dir / "candidate_scores.csv"})

    with pipeline_progress.stage("Registration", total=len(corrected), message="Stabilizing frames against the reference frame") as tracker:
        corrected_digest = _frames_content_digest(corrected)
        registration_signature = _signature_for_stage(
            "Registration",
            corrected_digest=corrected_digest,
            audit_digest=_rows_digest([asdict(r) for r in audit_records]),
            registration=asdict(cfg.registration),
            winner=winner,
        )
        registration_required = [registration_dir / "transforms.csv", registration_dir / "reference.json"]
        if _checkpoint_matches(cfg, qc_dir, "registration", registration_signature, registration_required):
            tracker.progress(1, max(1, len(corrected)), message="Registration checkpoint hit; applying cached transforms")
            ref_payload = _read_json(registration_dir / "reference.json", {})
            ref_frame_id = int(ref_payload.get("reference_frame_id", corrected[0].frame_id))
            ref_idx = next((i for i, frame in enumerate(corrected) if int(frame.frame_id) == ref_frame_id), 0)
            transforms = _transforms_from_rows(_read_table_rows(registration_dir / "transforms.csv"))
            if len(transforms) != len(corrected):
                raise ValueError("registration checkpoint transform count does not match frame count")
            stabilized = [FrameRecord(frame.frame_id, frame.time_s, apply_transform(frame.image, tfm)) for frame, tfm in zip(corrected, transforms)]
            ref = stabilized[ref_idx].image
            registration_mean_residual = float(np.mean([residual_difference(st.image, ref) for st in stabilized])) if stabilized else float("nan")
            tracker.progress(max(1, len(corrected)), max(1, len(corrected)), message="Registration checkpoint loaded")
            _write_checkpoint(cfg, out_dir, qc_dir, "registration", registration_signature, {"transforms": registration_dir / "transforms.csv", "reference": registration_dir / "reference.json"}, reused=True)
        else:
            ref_idx = select_reference_frame(corrected, audit_records, mode=cfg.registration.reference_frame)
            stabilized, transforms = stabilize_sequence(
                corrected,
                reference_idx=ref_idx,
                max_shift_px=cfg.registration.max_shift_px,
                progress_cfg=quiet_progress,
                progress_callback=lambda c, t: tracker.progress(c, t),
            )
            registration_mean_residual = _write_registration_artifacts(registration_dir, ref_idx, corrected, stabilized, transforms)
            _write_checkpoint(cfg, out_dir, qc_dir, "registration", registration_signature, {"transforms": registration_dir / "transforms.csv", "reference": registration_dir / "reference.json"})

    grid_debug = None
    support_mask_runtime = None
    support_circle_runtime = None
    geometry_sanity: dict[str, Any] = {}
    geometry_progress_total = 34
    with pipeline_progress.stage("Reference geometry", total=geometry_progress_total, message="Detecting and indexing holes on the reference frame") as tracker:
        ref_frame = stabilized[ref_idx]

        def detector_progress_mapper(start: int, end: int, label: str):
            span = max(1, int(end) - int(start))

            def _callback(current: int, total: int, message: str) -> None:
                frac = float(current) / max(float(total), 1.0)
                step = int(start) + int(round(span * frac))
                tracker.progress(min(int(end), max(int(start), step)), geometry_progress_total, message=f"{label}: {message}")

            return _callback

        if hole_grid_bundle is None:
            tracker.progress(1, geometry_progress_total, message="Warming geometry kernels")
            warmup_exact_sequence_numba()
            tracker.progress(1, geometry_progress_total, message="Detecting candidates")
            candidate_images = [fr.image for fr in frames]
            stable_result = detect_exact_wafer_holes_sequence_full(
                candidate_images,
                cfg.geometry,
                reference_index=ref_idx,
                progress_callback=detector_progress_mapper(1, 22, "raw-frame detector"),
            )
            if bool(getattr(cfg.geometry, "use_photometry_fallback_for_geometry", False)) and (len(stable_result.accepted_candidates) < 8 or float(stable_result.lattice.confidence) < 0.50):
                try:
                    corrected_result = detect_exact_wafer_holes_sequence_full(
                        [fr.image for fr in stabilized],
                        cfg.geometry,
                        reference_index=ref_idx,
                        progress_callback=detector_progress_mapper(23, 28, "corrected-frame detector"),
                    )
                except Exception:
                    corrected_result = None
                if corrected_result is not None and len(corrected_result.accepted_candidates) >= 8 and float(corrected_result.lattice.confidence) > float(stable_result.lattice.confidence):
                    stable_result = corrected_result
            candidates = stable_result.accepted_candidates
            grid_debug = stable_result.debug
            support_mask_runtime = getattr(grid_debug, "support_mask", None)
            support_circle_runtime = getattr(grid_debug, "support_circle", None)
            tracker.progress(29, geometry_progress_total, message="Using exact detector geometry")
            ref_holes = _hole_geometry_from_exact_candidates(candidates)
            tracker.progress(30, geometry_progress_total, message="Using exact detector lattice")
            lattice = stable_result.lattice
            lattice_indices = dict(stable_result.lattice_indices)
            write_json(geometry_dir / "grid_detection_debug.json", {
                "support_circle": None if grid_debug.support_circle is None else {"x": int(grid_debug.support_circle[0]), "y": int(grid_debug.support_circle[1]), "r": int(grid_debug.support_circle[2])},
                "raw_count": int(grid_debug.raw_count),
                "filtered_count": int(grid_debug.filtered_count),
                "anchor_count": int(grid_debug.anchor_count),
                "recovered_strong_count": int(grid_debug.recovered_strong_count),
                "predicted_only_full_count": int(grid_debug.predicted_only_full_count),
                "predicted_only_partial_count": int(grid_debug.predicted_only_partial_count),
                "completed_count": int(grid_debug.completed_count),
                "mode": grid_debug.mode,
                "common_radius_px": float(grid_debug.common_radius_px),
                "sequence_frame_count": int(getattr(grid_debug, "sequence_frame_count", 0) or 0),
                "sequence_sampled_count": int(getattr(grid_debug, "sequence_sampled_count", 0) or 0),
                "sequence_sample_indices": [int(i) for i in getattr(grid_debug, "sequence_sample_indices", [])],
                "sequence_sampling_history": list(getattr(grid_debug, "sequence_sampling_history", [])),
                "watchdog_events": list(getattr(grid_debug, "watchdog_events", [])),
            })
            geometry_sanity = _geometry_sanity_checks(stable_result, ref_frame.image.shape[:2])
            write_json(geometry_dir / "geometry_sanity_checks.json", geometry_sanity)
            if getattr(grid_debug, "tiers", None):
                write_table(geometry_dir / "grid_detection_tiers.csv", list(grid_debug.tiers))
            if getattr(grid_debug, "predicted_only", None):
                write_table(geometry_dir / "grid_detection_predicted_only.csv", list(grid_debug.predicted_only))
        else:
            tracker.progress(1, geometry_progress_total, message="Using supplied hole-grid model")
            ref_holes = hole_grid_bundle.holes
            lattice = hole_grid_bundle.lattice
            from holecolor.core.types import HoleCandidate
            candidates = [HoleCandidate(float(h.x), float(h.y), float(max(1.0, 0.5*(h.radius_inner_px+h.radius_outer_px))), 0.0, 0.0, float(h.confidence)) for h in ref_holes]
            tracker.progress(29, geometry_progress_total, message="Assigning lattice indices")
            lattice_indices = assign_lattice_indices(candidates, lattice)
            geometry_sanity = {
                "passed": True,
                "status": "supplied_model",
                "fail_count": 0,
                "warning_count": 0,
                "n_candidates": int(len(candidates)),
                "lattice_confidence": float(lattice.confidence),
                "checks": [
                    {
                        "name": "supplied_hole_grid_model",
                        "passed": True,
                        "severity": "info",
                        "detail": "Automatic geometry sanity checks are informational because a supplied hole-grid model was used.",
                        "value": None,
                    }
                ],
            }
            write_json(geometry_dir / "geometry_sanity_checks.json", geometry_sanity)
            tracker.progress(30, geometry_progress_total, message="Skipping automatic geometry detection")
        detected_candidates = list(candidates)
        detected_ref_holes = list(ref_holes)
        detected_lattice_indices = dict(lattice_indices)
        write_table(geometry_dir / "hole_candidates_detected.csv", _candidate_table(detected_candidates, detected_lattice_indices))
        write_table(geometry_dir / "hole_geometry_detected.csv", detected_ref_holes)
        write_table(geometry_dir / "hole_lattice_index_detected.csv", _make_hole_lattice_rows(detected_ref_holes, detected_lattice_indices))
        write_json(geometry_dir / "lattice_model.json", asdict(lattice))
        detected_overlay = draw_candidates(ref_frame.image, detected_candidates, lattice=lattice, lattice_indices=detected_lattice_indices, support_circle=support_circle_runtime)
        save_image(geometry_dir / "overlays" / "frame_ref_geometry_overlay.png", detected_overlay)
        save_image(geometry_dir / "overlays" / "frame_ref_detected_geometry_overlay.png", detected_overlay)
        tracker.progress(31, geometry_progress_total, message="Excluding partial holes and terraces")
        reference_support_mask = support_mask_from_debug(ref_frame.image.shape[:2], support_mask_runtime, support_circle_runtime)
        complete_filter = filter_complete_holes_and_terraces(
            detected_ref_holes,
            detected_candidates,
            detected_lattice_indices,
            ref_frame.image.shape[:2],
            n_terraces=cfg.masks.n_terraces,
            terrace_width_mode=cfg.masks.terrace_width_mode,
            terrace_gap_basis=cfg.masks.terrace_gap_basis,
            terrace_min_width_px=cfg.masks.terrace_min_width_px,
            support_mask=reference_support_mask,
            support_circle=support_circle_runtime,
        )
        candidates = complete_filter.candidates
        ref_holes = complete_filter.holes
        lattice_indices = complete_filter.lattice_indices
        geometry_sanity["complete_geometry_filter"] = complete_filter.summary
        geometry_sanity["n_candidates_after_complete_filter"] = int(len(candidates))
        complete_check = {
            "name": "complete_holes_after_filter",
            "passed": bool(len(ref_holes) > 0),
            "severity": "fail",
            "detail": f"{len(ref_holes)} holes remain after excluding partial holes and terraces.",
            "value": int(len(ref_holes)),
        }
        geometry_sanity.setdefault("checks", []).append(complete_check)
        if not complete_check["passed"]:
            geometry_sanity["passed"] = False
            geometry_sanity["status"] = "fail"
            geometry_sanity["fail_count"] = int(geometry_sanity.get("fail_count", 0) or 0) + 1
        write_table(geometry_dir / "hole_terrace_exclusion.csv", complete_filter.rows)
        write_json(geometry_dir / "hole_terrace_exclusion_summary.json", complete_filter.summary)
        write_json(geometry_dir / "geometry_sanity_checks.json", geometry_sanity)
        if bool(getattr(cfg.geometry, "fail_on_sanity_failure", True)) and int(geometry_sanity.get("fail_count", 0) or 0) > 0:
            raise RuntimeError(
                "Reference geometry failed sanity checks; stopping before downstream colour/radial analysis. "
                f"See {geometry_dir / 'geometry_sanity_checks.json'}."
            )
        if not ref_holes:
            raise ValueError(
                "No complete holes remain after requiring hole and terrace disks to stay inside the wafer and video frame. "
                f"See {geometry_dir / 'hole_terrace_exclusion.csv'}."
            )
        tracker.progress(32, geometry_progress_total, message="Writing selected geometry artifacts")
        write_table(geometry_dir / "hole_candidates.csv", _candidate_table(candidates, lattice_indices))
        write_table(geometry_dir / "hole_candidates_selected.csv", _candidate_table(candidates, lattice_indices))
        write_table(geometry_dir / "hole_geometry.csv", ref_holes)
        write_table(geometry_dir / "hole_geometry_selected.csv", ref_holes)
        write_table(geometry_dir / "hole_lattice_index.csv", _make_hole_lattice_rows(ref_holes, lattice_indices))
        write_table(geometry_dir / "hole_lattice_index_selected.csv", _make_hole_lattice_rows(ref_holes, lattice_indices))
        selected_overlay = draw_candidates(ref_frame.image, candidates, lattice=lattice, lattice_indices=lattice_indices, support_circle=support_circle_runtime)
        save_image(geometry_dir / "overlays" / "frame_ref_selected_geometry_overlay.png", selected_overlay)
        tracker.progress(34, geometry_progress_total, message="Reference geometry complete")

    with pipeline_progress.stage("Descriptor selection", total=3, message="Selecting the primary descriptor for the run") as tracker:
        shape = ref_frame.image.shape[:2]
        ref_hole_union = make_global_hole_union(shape, ref_holes)
        ref_roi_mask = np.ones(shape, dtype=bool)
        ref_matrix_mask = make_matrix_bulk_mask(ref_roi_mask, ref_hole_union)
        tracker.progress(1, 3, message="Scoring candidate descriptors")
        chosen_descriptor, descriptor_ranking = _pick_primary_descriptor(ref_frame, ref_holes, ref_matrix_mask, cfg)
        tracker.progress(2, 3, message="Writing descriptor selection")
        write_json(descriptors_dir / "descriptor_selection.json", {"chosen_descriptor": chosen_descriptor, "ranking": descriptor_ranking})
        tracker.progress(3, 3, message=f"Primary descriptor: {chosen_descriptor}")

    with pipeline_progress.stage("Geometry propagation", total=max(1, len(stabilized) - 1), message="Propagating hole geometry through the time series") as tracker:
        holes_by_frame = _propagate_holes_across_frames(
            stabilized, ref_idx, ref_holes, cfg, progress_cfg=quiet_progress, progress_callback=lambda c, t: tracker.progress(c, max(1, t))
        )
        geometry_rows = _geometry_timeseries_rows(stabilized, holes_by_frame, lattice_indices)
        write_table(geometry_dir / "hole_geometry_timeseries.csv", geometry_rows)

    runtime_support_mask = support_mask_from_debug(shape, support_mask_runtime, support_circle_runtime)

    wafer_nonhole_dir = descriptors_dir / "wafer_nonhole_colour"
    wafer_nonhole_bundle = WaferNonholeColourBundle('skipped', None, [], [], [], [], [], [], [], None, message='not_run')
    wafer_nonhole_samples = None
    wafer_nonhole_overlay = None
    if cfg.wafer_nonhole_colour.enabled:
        wafer_nonhole_dir.mkdir(parents=True, exist_ok=True)
        with pipeline_progress.stage("Wafer non-hole region", total=3, message="Extracting wafer-minus-holes colour region") as tracker:
            try:
                runtime_support_mask = support_mask_from_debug(shape, support_mask_runtime, support_circle_runtime)
                tracker.progress(1, 3, message="Resolved support mask")
                region_rows, wafer_nonhole_samples, wafer_nonhole_overlay = extract_wafer_nonhole_region(
                    stabilized, holes_by_frame, runtime_support_mask,
                    max_points_per_frame=cfg.wafer_nonhole_colour.max_points_per_frame,
                    rng_seed=cfg.wafer_nonhole_colour.random_state,
                )
                tracker.progress(2, 3, message="Sampled wafer non-hole HSL points")
                write_json(wafer_nonhole_dir / "frame_region_status.json", {"status": "ok", "n_frames": len(region_rows)})
                write_table(wafer_nonhole_dir / "frame_region_summary.csv", region_rows)
                if wafer_nonhole_overlay is not None:
                    save_image(wafer_nonhole_dir / "frame0_region_overlay.png", wafer_nonhole_overlay)
                tracker.progress(3, 3, message="Wafer non-hole region extracted")
            except Exception as exc:
                write_json(wafer_nonhole_dir / "frame_region_status.json", {"status": "error", "message": str(exc)})
                if not cfg.wafer_nonhole_colour.fail_open:
                    raise
                tracker.progress(3, 3, message=f"Wafer non-hole region skipped after error: {exc}")
        with pipeline_progress.stage("Wafer non-hole colour", total=3, message="Fitting pooled wafer non-hole colour model") as tracker:
            try:
                if wafer_nonhole_samples is not None:
                    runtime_support_mask = support_mask_from_debug(shape, support_mask_runtime, support_circle_runtime)
                    wafer_nonhole_bundle = build_wafer_nonhole_colour_bundle_from_samples(
                        region_rows, wafer_nonhole_samples, runtime_support_mask,
                        max_total_fit_points=cfg.wafer_nonhole_colour.max_total_fit_points,
                        min_total_points=cfg.wafer_nonhole_colour.min_total_points,
                        k_min=cfg.wafer_nonhole_colour.gmm_k_min,
                        k_max=cfg.wafer_nonhole_colour.gmm_k_max,
                        random_state=cfg.wafer_nonhole_colour.random_state,
                    )
                    tracker.progress(1, 3, message="Built colour bundle")
                    if wafer_nonhole_bundle.frame_cluster_summary_rows:
                        wafer_nonhole_bundle.global_context_rows = enrich_global_matrix_rows([], wafer_nonhole_bundle.frame_cluster_summary_rows)
                    tracker.progress(2, 3, message="Prepared frame cluster summaries")
                    write_wafer_nonhole_colour_artifacts(wafer_nonhole_dir, wafer_nonhole_bundle, overlay=wafer_nonhole_overlay, sampled_points=wafer_nonhole_samples)
                else:
                    write_json(wafer_nonhole_dir / "stage_status.json", {"status": "skipped", "message": "region stage unavailable"})
                    tracker.progress(2, 3, message="Skipped colour model because region stage did not complete")
                tracker.progress(3, 3, message="Wafer non-hole colour stage complete")
            except Exception as exc:
                write_json(wafer_nonhole_dir / "stage_status.json", {"status": "error", "message": str(exc)})
                if not cfg.wafer_nonhole_colour.fail_open:
                    raise
                tracker.progress(3, 3, message=f"Wafer non-hole colour skipped after error: {exc}")
        recolour_passes = 1 + int(bool(cfg.wafer_nonhole_colour.write_baseline_activity_video))
        recolour_progress_total = max(3, len(stabilized) * recolour_passes + 2)
        with pipeline_progress.stage("Wafer non-hole recolour video", total=recolour_progress_total, message="Rendering cluster recoloured wafer non-hole videos") as tracker:
            try:
                if wafer_nonhole_bundle.status == 'ok' and wafer_nonhole_bundle.cluster_rows:
                    runtime_support_mask = support_mask_from_debug(shape, support_mask_runtime, support_circle_runtime)
                    tracker.progress(1, recolour_progress_total, message="Prepared recolour inputs")
                    raw_registered = [
                        FrameRecord(frame.frame_id, frame.time_s, apply_transform(frame.image, tfm))
                        for frame, tfm in zip(frames, transforms)
                    ]
                    write_wafer_nonhole_cluster_videos(
                        wafer_nonhole_dir,
                        stabilized,
                        holes_by_frame,
                        runtime_support_mask,
                        wafer_nonhole_bundle,
                        display_frames=raw_registered,
                        write_recolour_video=cfg.wafer_nonhole_colour.write_recolour_video,
                        write_side_by_side_video=cfg.wafer_nonhole_colour.write_side_by_side_video,
                        write_labelmap_video=cfg.wafer_nonhole_colour.write_labelmap_video,
                        write_baseline_activity_video=cfg.wafer_nonhole_colour.write_baseline_activity_video,
                        baseline_frames=cfg.visualisation.cluster_rdf.baseline_frames,
                        progress_callback=lambda c, t, message: tracker.progress(c, t, message=message),
                    )
                    tracker.progress(recolour_progress_total, recolour_progress_total, message="Rendered recolour videos")
                else:
                    write_json(wafer_nonhole_dir / 'cluster_video_status.json', {'status': 'skipped', 'message': 'cluster_bundle_unavailable'})
                    tracker.progress(recolour_progress_total - 1, recolour_progress_total, message="Skipped recolour videos because colour bundle unavailable")
                tracker.progress(recolour_progress_total, recolour_progress_total, message="Wafer non-hole recolour stage complete")
            except Exception as exc:
                write_json(wafer_nonhole_dir / 'cluster_video_status.json', {'status': 'error', 'message': str(exc)})
                if not cfg.wafer_nonhole_colour.fail_open:
                    raise
                tracker.progress(recolour_progress_total, recolour_progress_total, message=f"Wafer non-hole recolour skipped after error: {exc}")

    with pipeline_progress.stage("Frame-analysis cache check", total=2, message="Computing the per-frame cache signature and checking cache availability") as tracker:
        tracker.progress(1, 2, message="Computing frame-analysis cache signature")
        frame_analysis_signature = _frame_analysis_cache_signature(cfg, chosen_descriptor, stabilized, geometry_rows, lattice_indices)
        tracker.progress(2, 2, message="Checking per-frame cache availability")
        frame_analysis_cache_ready = _frame_analysis_cache_available(descriptors_dir, radial_dir, hotspots_dir, qc_dir, frame_analysis_signature)
    cached_frame_analysis = None

    with pipeline_progress.stage("Reference terraces", total=5, message="Building reference terraces and matrix masks") as tracker:
        tracker.progress(1, 5, message="Building terrace ownership")
        ref_terraces, ref_terrace_plan = make_nonoverlapping_hole_terraces(
            shape,
            holes_by_frame[ref_frame.frame_id],
            cfg.masks.n_terraces,
            lattice_indices=lattice_indices,
            width_mode=cfg.masks.terrace_width_mode,
            gap_basis=cfg.masks.terrace_gap_basis,
            min_width_px=cfg.masks.terrace_min_width_px,
            return_plan=True,
        )
        tracker.progress(2, 5, message="Writing terrace plan")
        write_table(masks_dir / "reference_terrace_plan.csv", _terrace_plan_rows(ref_terrace_plan, lattice_indices))
        write_table(masks_dir / "reference_terrace_annuli.csv", _terrace_annulus_rows(ref_terrace_plan))
        write_json(masks_dir / "reference_terrace_plan.json", {
            "span_mode": cfg.masks.terrace_width_mode,
            "gap_basis": cfg.masks.terrace_gap_basis,
            "n_terraces": int(cfg.masks.n_terraces),
            "min_width_px": float(cfg.masks.terrace_min_width_px),
        })
        tracker.progress(3, 5, message="Building matrix mask")
        ref_matrix_mask = make_matrix_bulk_mask(np.ones(shape, dtype=bool), make_global_hole_union(shape, holes_by_frame[ref_frame.frame_id]))
        tracker.progress(4, 5, message="Rendering terrace overlay")
        terrace_overlay = _terrace_overlay(ref_frame.image, holes_by_frame[ref_frame.frame_id], ref_terraces, ref_matrix_mask)
        save_image(masks_dir / "overlays" / "frame_ref_terraces_overlay.png", terrace_overlay)
        tracker.progress(5, 5, message="Reference terraces complete")

    if cfg.radial_cluster_average_hole.enabled:
        radial_cluster_dir = descriptors_dir / "radial_cluster_average_hole"
        radial_cluster_total = max(5, len(stabilized) + 5)
        with pipeline_progress.stage("Radial cluster average hole", total=radial_cluster_total, message="Building average-hole cluster chronograms and consistency outputs") as tracker:
            try:
                radial_cluster_dir.mkdir(parents=True, exist_ok=True)
                tracker.progress(1, radial_cluster_total, message="Resolved cluster and terrace inputs")
                if wafer_nonhole_bundle.status == 'ok' and wafer_nonhole_bundle.cluster_rows:
                    warmup_radial_cluster_numba()
                    write_radial_cluster_average_hole_artifacts(
                        radial_cluster_dir,
                        stabilized,
                        holes_by_frame,
                        runtime_support_mask,
                        wafer_nonhole_bundle.cluster_rows,
                        lattice_indices,
                        lattice.angle_deg,
                        n_terraces=cfg.masks.n_terraces,
                        n_angle_sectors=cfg.radial_cluster_average_hole.n_angle_sectors,
                        terrace_width_mode=cfg.masks.terrace_width_mode,
                        terrace_gap_basis=cfg.masks.terrace_gap_basis,
                        terrace_min_width_px=cfg.masks.terrace_min_width_px,
                        front_threshold_fraction=cfg.radial_cluster_average_hole.front_threshold_fraction,
                        parallel_cfg=cfg.parallel,
                        progress_callback=lambda c, t, message: tracker.progress(
                            min(int(c) + 1, radial_cluster_total),
                            radial_cluster_total,
                            message=message,
                        ),
                    )
                    tracker.progress(radial_cluster_total - 1, radial_cluster_total, message="Wrote average-hole radial cluster outputs")
                else:
                    write_json(radial_cluster_dir / "radial_cluster_status.json", {"status": "skipped", "message": "cluster_bundle_unavailable"})
                    tracker.progress(radial_cluster_total - 1, radial_cluster_total, message="Skipped average-hole branch because cluster bundle unavailable")
                tracker.progress(radial_cluster_total, radial_cluster_total, message="Radial cluster average-hole stage complete")
            except Exception as exc:
                write_json(radial_cluster_dir / "radial_cluster_status.json", {"status": "error", "message": str(exc)})
                if not cfg.radial_cluster_average_hole.fail_open:
                    raise
                tracker.progress(radial_cluster_total, radial_cluster_total, message=f"Radial cluster average-hole skipped after error: {exc}")

    compartment_rows: list[dict[str, Any]] = []
    radial_rows: list[dict[str, Any]] = []
    angular_rows: list[dict[str, Any]] = []
    matrix_rows: list[dict[str, Any]] = []
    hotspot_rows: list[dict[str, Any]] = []
    hotspot_track_rows: list[dict[str, Any]] = []
    sector_radial_rows: list[dict[str, Any]] = []
    perturb_rows: list[dict[str, Any]] = []

    ref_hsv = rgb_to_hsv(ref_frame.image)
    baseline_descriptor = descriptor_image(ref_frame.image, ref_hsv, chosen_descriptor)

    prev_hotspots = []
    next_track_id = 0
    prev_track_map: dict[int, int] = {}

    frame_tasks = [
        {
            "frame": frame,
            "frame_holes": holes_by_frame[frame.frame_id],
            "lattice_indices": lattice_indices,
            "n_terraces": cfg.masks.n_terraces,
            "n_sectors": cfg.radial.angular_n_sectors,
            "chosen_descriptor": chosen_descriptor,
            "baseline_descriptor": baseline_descriptor,
            "interior_shrink_px": cfg.masks.interior_shrink_px,
            "rim_width_px": cfg.masks.rim_width_px,
            "terrace_width_mode": cfg.masks.terrace_width_mode,
            "terrace_gap_basis": cfg.masks.terrace_gap_basis,
            "terrace_min_width_px": cfg.masks.terrace_min_width_px,
            "hotspot_cfg": cfg.hotspots,
        }
        for frame in stabilized
    ]
    frame_results: list[dict[str, Any]] = []
    if frame_analysis_cache_ready:
        with pipeline_progress.stage("Per-frame analysis", total=2, message="Loading cached per-frame radial, angular, and hotspot measurements") as tracker:
            tracker.progress(1, 2, message="Per-frame analysis cache hit; loading cached rows")
            cached_frame_analysis = _load_frame_analysis_cache(descriptors_dir, radial_dir, hotspots_dir, qc_dir, frame_analysis_signature)
            if cached_frame_analysis is None:
                tracker.progress(2, 2, message="Cache became unavailable; recomputing per-frame analysis")
                frame_analysis_cache_ready = False
            else:
                compartment_rows = cached_frame_analysis["compartment_rows"]
                radial_rows = cached_frame_analysis["radial_rows"]
                angular_rows = cached_frame_analysis["angular_rows"]
                matrix_rows = cached_frame_analysis["matrix_rows"]
                sector_radial_rows = cached_frame_analysis["sector_radial_rows"]
                hotspot_rows = cached_frame_analysis["hotspot_rows"]
                hotspot_track_rows = cached_frame_analysis["hotspot_track_rows"]
                tracker.progress(2, 2, message="Per-frame analysis cache loaded")
    if frame_analysis_cache_ready:
        with pipeline_progress.stage("Hotspot linking", total=1, message="Using cached hotspot linking outputs") as tracker:
            tracker.progress(1, 1, message="Hotspot linking cache hit; using cached tracks and hotspot tables")
    else:
        with pipeline_progress.stage("Per-frame analysis", total=len(frame_tasks), message="Extracting per-frame radial, angular, and hotspot measurements") as tracker:
            frame_results = parallel_map(
                _process_frame_analysis_task,
                frame_tasks,
                prefer_thread_for_image_tasks(cfg.parallel),
                desc="Per-frame analysis",
                progress_callback=lambda c, t: tracker.progress(c, t),
            )
        frame_results = sorted(frame_results, key=lambda r: int(r["frame_id"]))
        keyframe_ids = {stabilized[0].frame_id, ref_frame.frame_id, stabilized[-1].frame_id}
        track_iter = list(zip(stabilized, frame_results))
        with pipeline_progress.stage("Hotspot linking", total=len(track_iter), message="Linking hotspots and collecting per-frame outputs") as tracker:
            for idx_pair, (frame, frame_result) in enumerate(track_iter, start=1):
                tracker.progress(idx_pair, len(track_iter), message="Linking hotspots and collecting per-frame outputs")
                frame_holes = holes_by_frame[frame.frame_id]
                compartment_rows.extend(frame_result["compartment_rows"])
                radial_rows.extend(frame_result["radial_rows"])
                angular_rows.extend(frame_result["angular_rows"])
                sector_radial_rows.extend(frame_result["sector_radial_rows"])
                matrix_rows.append(frame_result["matrix_row"])
                hotspots = frame_result["hotspots"]

                link_dist = max(6.0, min(cfg.hotspots.link_max_dist_px, 0.75 * np.mean([h.radius_outer_px for h in frame_holes]) if frame_holes else cfg.hotspots.link_max_dist_px))
                links = link_hotspots(prev_hotspots, hotspots, max_dist_px=link_dist, max_area_ratio=cfg.hotspots.max_area_ratio) if prev_hotspots else []
                cur_track_map: dict[int, int] = {}
                linked_prev_to_cur = {cur_id: prev_id for prev_id, cur_id in links}
                for hs in hotspots:
                    if hs.hotspot_id in linked_prev_to_cur and linked_prev_to_cur[hs.hotspot_id] in prev_track_map:
                        track_id = prev_track_map[linked_prev_to_cur[hs.hotspot_id]]
                    else:
                        track_id = next_track_id
                        next_track_id += 1
                    cur_track_map[hs.hotspot_id] = track_id
                    hotspot_rows.append(
                        {
                            "frame_id": hs.frame_id,
                            "hotspot_id": hs.hotspot_id,
                            "track_id": track_id,
                            "cx": hs.cx,
                            "cy": hs.cy,
                            "area_px": hs.area_px,
                            "score": hs.score,
                            "nearest_hole_id": hs.nearest_hole_id,
                            "dist_to_hole_px": hs.dist_to_hole_px,
                            "mean_R": hs.mean_r,
                            "mean_G": hs.mean_g,
                            "mean_B": hs.mean_b,
                            "mean_H": hs.mean_h,
                            "mean_S": hs.mean_s,
                            "bbox_x": hs.bbox_x,
                            "bbox_y": hs.bbox_y,
                            "bbox_w": hs.bbox_w,
                            "bbox_h": hs.bbox_h,
                        }
                    )
                for prev_id, cur_id in links:
                    hotspot_track_rows.append({"frame_prev": frame.frame_id - 1, "frame_cur": frame.frame_id, "prev_hotspot_id": prev_id, "cur_hotspot_id": cur_id, "track_id": cur_track_map.get(cur_id)})
                if frame.frame_id in keyframe_ids:
                    save_image(hotspots_dir / "overlays" / f"frame_{frame.frame_id:04d}_hotspots.png", _hotspot_overlay(frame.image, hotspots, frame_holes))

                prev_hotspots = hotspots
                prev_track_map = cur_track_map
        _write_frame_analysis_cache(
            descriptors_dir,
            radial_dir,
            hotspots_dir,
            qc_dir,
            frame_analysis_signature,
            compartment_rows=compartment_rows,
            matrix_rows=matrix_rows,
            radial_rows=radial_rows,
            angular_rows=angular_rows,
            sector_radial_rows=sector_radial_rows,
            hotspot_rows=hotspot_rows,
            hotspot_track_rows=hotspot_track_rows,
        )

    if cfg.wafer_nonhole_colour.enabled:
        with pipeline_progress.stage("Wafer non-hole context", total=3, message="Linking wafer non-hole colour intel to existing outputs") as tracker:
            try:
                if wafer_nonhole_bundle.frame_cluster_summary_rows:
                    wafer_nonhole_bundle.global_context_rows = enrich_global_matrix_rows(matrix_rows, wafer_nonhole_bundle.frame_cluster_summary_rows)
                    wafer_nonhole_bundle.local_context_rows = enrich_local_compartment_rows(compartment_rows, wafer_nonhole_bundle.frame_cluster_summary_rows)
                    write_table(wafer_nonhole_dir / "global_buffer_cluster_context.csv", wafer_nonhole_bundle.global_context_rows)
                    tracker.progress(1, 3, message="Wrote global buffer context")
                    write_table(wafer_nonhole_dir / "local_hole_cluster_context.csv", wafer_nonhole_bundle.local_context_rows)
                    tracker.progress(2, 3, message="Wrote local hole context")
                    write_wafer_nonhole_colour_artifacts(wafer_nonhole_dir, wafer_nonhole_bundle, overlay=wafer_nonhole_overlay, sampled_points=wafer_nonhole_samples)
                else:
                    write_json(wafer_nonhole_dir / "context_status.json", {"status": "skipped", "message": "no frame cluster summary rows"})
                    tracker.progress(2, 3, message="No cluster summaries available for context linking")
                tracker.progress(3, 3, message="Wafer non-hole context complete")
            except Exception as exc:
                write_json(wafer_nonhole_dir / "context_status.json", {"status": "error", "message": str(exc)})
                if not cfg.wafer_nonhole_colour.fail_open:
                    raise
                tracker.progress(3, 3, message=f"Wafer non-hole context skipped after error: {exc}")

    radial_table = RadialRowTable.from_rows(radial_rows) if radial_rows else RadialRowTable.from_rows([])
    sector_table = SectorRadialTable.from_rows(sector_radial_rows) if sector_radial_rows else SectorRadialTable.from_rows([])
    hotspot_table = HotspotStatsTable.from_rows(hotspot_rows) if hotspot_rows else HotspotStatsTable.from_rows([])

    with pipeline_progress.stage("Radial summaries", total=7, message="Writing radial tables and first-order summaries") as tracker:
        tracker.progress(1, 7, message="Writing descriptor and annulus tables")
        write_table(descriptors_dir / "hole_compartment_timeseries.csv", compartment_rows)
        write_table(descriptors_dir / "matrix_timeseries.csv", matrix_rows)
        write_table_columns(radial_dir / "hole_annulus_timeseries.csv", radial_table.to_columns())
        aggregate_radial_rows_base = aggregate_radial_rows(radial_rows)

        tracker.progress(2, 7, message="Building aggregate radial evolution")
        radial_frame_rows, radial_conclusion_summary = _write_radial_evolution_artifacts(radial_dir, aggregate_radial_rows_base)
        per_hole_frame_rows = per_hole_radial_frame_summary_table(radial_table)
        angular_frame_rows, angular_hole_rows = aggregate_angular_asymmetry_rows(angular_rows)
        per_hole_radial_summary_rows = summarize_hole_radial_evolution(per_hole_frame_rows)
        per_hole_radial_summary_rows = merge_hole_radial_and_asymmetry(per_hole_radial_summary_rows, angular_hole_rows)

        tracker.progress(3, 7, message="Assigning radial archetypes and reticulum zones")
        radial_archetype_rows, radial_archetype_centroids = assign_radial_archetypes(per_hole_radial_summary_rows, k=cfg.radial.archetype_k)
        zone_by_hole = reticulum_zone_by_hole(per_hole_radial_summary_rows)
        reticulum_group_rows, reticulum_group_summary_rows = build_reticulum_group_rows_table(radial_table, zone_by_hole)

        tracker.progress(4, 7, message="Fitting radial and sector models")
        radial_model_fit_rows, radial_model_summary_rows = fit_radial_models_table(radial_table)
        sector_front_rows, sector_front_summary_rows = summarize_sector_fronts_table(sector_table)
        hotspot_reticulum_columns, hotspot_reticulum_group_columns = build_hotspot_reticulum_columns_table(
            hotspot_table,
            per_hole_radial_summary_rows,
            radial_archetype_rows,
            zone_by_hole,
        )

        tracker.progress(5, 7, message="Writing radial summary tables")
        write_table(radial_dir / "per_hole_radial_frame_summary.csv", per_hole_frame_rows)
        write_table(radial_dir / "per_hole_radial_summary.csv", per_hole_radial_summary_rows)
        write_table(radial_dir / "per_hole_radial_archetypes.csv", radial_archetype_rows)
        write_json(radial_dir / "radial_archetype_centroids.json", radial_archetype_centroids)
        write_table(radial_dir / "angular_asymmetry_timeseries.csv", angular_rows)
        write_table(radial_dir / "angular_asymmetry_frame_summary.csv", angular_frame_rows)
        write_table(radial_dir / "angular_asymmetry_hole_summary.csv", angular_hole_rows)
        write_table(radial_dir / "reticulum_group_radial_comparison.csv", reticulum_group_rows)
        write_table(radial_dir / "reticulum_group_frame_summary.csv", reticulum_group_summary_rows)
        write_table(radial_dir / "per_hole_radial_model_fits.csv", radial_model_fit_rows)
        write_table(radial_dir / "per_hole_radial_model_summary.csv", radial_model_summary_rows)
        write_table_columns(radial_dir / "sector_radial_timeseries.csv", sector_table.to_columns())
        write_table(radial_dir / "sector_front_summary.csv", sector_front_rows)
        write_table(radial_dir / "sector_front_hole_summary.csv", sector_front_summary_rows)
        write_table_columns(radial_dir / "hole_hotspot_reticulum_comparison.csv", hotspot_reticulum_columns)
        write_table_columns(radial_dir / "hotspot_reticulum_group_summary.csv", hotspot_reticulum_group_columns)

        tracker.progress(6, 7, message="Writing radial summary plots")
        _write_radial_archetype_plot(radial_dir, radial_archetype_rows)
        _write_angular_asymmetry_plot(radial_dir, angular_frame_rows)
        _write_reticulum_group_plot(radial_dir, reticulum_group_summary_rows)
        _write_sector_front_plot(radial_dir, sector_front_summary_rows)
        _write_model_fit_quality_plot(radial_dir, radial_model_summary_rows)
        _write_hotspot_reticulum_plot(radial_dir, hotspot_reticulum_group_columns)

        tracker.progress(7, 7, message="Radial summaries complete")

    with pipeline_progress.stage("RDF summaries", total=6, message="Building RDF evolution, archetypes, and sector summaries") as tracker:
        tracker.progress(1, 6, message="Building per-hole RDF evolution")
        per_hole_rdf_columns = build_per_hole_rdf_evolution_columns(radial_table)
        per_hole_rdf_frame_rows = per_hole_rdf_columns.frame_rows()
        per_hole_rdf_velocity_rows = [{k: per_hole_rdf_columns.velocity_summary[k][i] for k in per_hole_rdf_columns.velocity_summary} for i in range(len(per_hole_rdf_columns.velocity_summary['hole_id']))]
        _write_per_hole_rdf_artifacts(radial_dir, per_hole_rdf_columns, per_hole_rdf_frame_rows)

        tracker.progress(2, 6, message="Assigning RDF archetypes and dynamics")
        per_hole_rdf_archetype_rows, per_hole_rdf_archetype_centroids = build_per_hole_rdf_archetypes(per_hole_rdf_frame_rows, k=cfg.radial.archetype_k)
        per_hole_rdf_dynamics_rows = build_per_hole_rdf_front_dynamics(per_hole_rdf_frame_rows)
        per_hole_rdf_archetype_rows, per_hole_rdf_archetype_centroids = canonicalize_rdf_archetypes(per_hole_rdf_archetype_rows, per_hole_rdf_archetype_centroids)

        tracker.progress(3, 6, message="Building sector RDF and lag summaries")
        sector_rdf_columns = build_sector_rdf_evolution_columns(sector_table)
        sector_rdf_frame_rows = sector_rdf_columns.frame_rows()
        sector_frame_table = SectorRdfFrameTable.from_rows(sector_rdf_frame_rows) if sector_rdf_frame_rows else SectorRdfFrameTable.from_rows([])
        sector_front_lag_rows, sector_front_lag_summary_rows = build_sector_front_lag_rows_table(sector_frame_table, onset_threshold=cfg.radial.sector_lag_onset_threshold)

        tracker.progress(4, 6, message="Building RDF hotspot/reticulum comparisons")
        rdf_hotspot_reticulum_columns, rdf_hotspot_reticulum_group_columns = build_rdf_hotspot_reticulum_columns_table(
            hotspot_table,
            per_hole_rdf_archetype_rows,
            per_hole_rdf_dynamics_rows,
            zone_by_hole,
        )

        tracker.progress(5, 6, message="Writing RDF tables and plots")
        write_table(radial_dir / "per_hole_rdf_archetypes.csv", per_hole_rdf_archetype_rows)
        write_json(radial_dir / "rdf_archetype_centroids.json", per_hole_rdf_archetype_centroids)
        write_table(radial_dir / "per_hole_rdf_dynamics.csv", per_hole_rdf_dynamics_rows)
        write_table_columns(radial_dir / "sector_rdf_evolution.csv", sector_rdf_columns.evolution)
        write_table_columns(radial_dir / "sector_rdf_frame_summary.csv", sector_rdf_columns.frame_summary)
        write_table(radial_dir / "sector_front_lag_map.csv", sector_front_lag_rows)
        write_table(radial_dir / "sector_front_lag_summary.csv", sector_front_lag_summary_rows)
        write_table_columns(radial_dir / "rdf_hotspot_reticulum_comparison.csv", rdf_hotspot_reticulum_columns)
        write_table_columns(radial_dir / "rdf_hotspot_reticulum_group_summary.csv", rdf_hotspot_reticulum_group_columns)
        _write_per_hole_rdf_archetype_plot(radial_dir, per_hole_rdf_archetype_rows)
        _write_rdf_front_dynamics_plot(radial_dir, per_hole_rdf_dynamics_rows)
        _write_sector_rdf_plot(radial_dir, sector_rdf_columns.frame_summary)
        _write_sector_front_lag_map(radial_dir, sector_front_lag_columns(sector_front_lag_rows))
        _write_rdf_hotspot_reticulum_plot(radial_dir, rdf_hotspot_reticulum_group_columns)

        tracker.progress(6, 6, message="RDF summaries complete")

    rdf_stability_rows: list[dict[str, Any]] = []
    rdf_stability_summary_rows: list[dict[str, Any]] = []
    rdf_stability_summary: dict[str, Any] = {"mean_rdf_archetype_stability": None, "n_sweeps": 0}
    rdf_bootstrap_rows: list[dict[str, Any]] = []
    rdf_bootstrap_support_rows: list[dict[str, Any]] = []
    sector_front_propagation_rows: list[dict[str, Any]] = []
    sector_front_propagation_hole_rows: list[dict[str, Any]] = []
    validation_hole_table = ValidationHoleTable.from_rows()
    rdf_uncertainty_table = None
    rdf_uncertainty_reticulum_rows: list[dict[str, Any]] = []
    rdf_uncertainty_hotspot_rows: list[dict[str, Any]] = []
    rdf_uncertainty_hotspot_group_rows: list[dict[str, Any]] = []
    sector_front_acceleration_rows: list[dict[str, Any]] = []
    sector_front_acceleration_hole_rows: list[dict[str, Any]] = []
    sweep_rows: list[dict[str, Any]] = []
    sweep_drift_rows: list[dict[str, Any]] = []
    radial_consistency_summary: dict[str, Any] = {
        "base_conclusion_label": radial_conclusion_summary.get("conclusion_label"),
        "n_sweeps": 0,
        "conclusion_agreement_fraction": None,
        "mean_profile_correlation": None,
    }

    validation_signature = _validation_cache_signature(
        cfg,
        chosen_descriptor,
        radial_conclusion_summary,
        per_hole_rdf_frame_rows,
        sector_rdf_frame_rows,
        hotspot_rows,
        per_hole_rdf_archetype_rows,
        per_hole_rdf_dynamics_rows,
        zone_by_hole,
    )
    cached_validation = _load_validation_cache(radial_dir, qc_dir, validation_signature) if cfg.validation.enabled else None
    validation_stage_total = 3 if cached_validation is not None else (7 if cfg.validation.enabled else 2)
    if cached_validation is not None:
        validation_msg = "Loading cached perturbation, bootstrap, and uncertainty outputs"
    else:
        validation_msg = "Running perturbation sweeps, bootstrap summaries, and uncertainty outputs" if cfg.validation.enabled else "Validation profile disabled; writing lightweight placeholders"
    with pipeline_progress.stage("Validation sweeps", total=validation_stage_total, message=validation_msg) as tracker:
        if cached_validation is not None:
            tracker.progress(1, validation_stage_total, message="Validation cache hit; loading cached outputs")
            rdf_stability_rows = cached_validation["rdf_stability_rows"]
            rdf_stability_summary_rows = cached_validation["rdf_stability_summary_rows"]
            rdf_stability_summary = cached_validation["rdf_stability_summary"]
            rdf_bootstrap_rows = cached_validation["rdf_bootstrap_rows"]
            rdf_bootstrap_support_rows = cached_validation["rdf_bootstrap_support_rows"]
            sector_front_propagation_rows = cached_validation["sector_front_propagation_rows"]
            sector_front_propagation_hole_rows = cached_validation["sector_front_propagation_hole_rows"]
            rdf_uncertainty_reticulum_rows = cached_validation["rdf_uncertainty_reticulum_rows"]
            rdf_uncertainty_hotspot_rows = cached_validation["rdf_uncertainty_hotspot_rows"]
            rdf_uncertainty_hotspot_group_rows = cached_validation["rdf_uncertainty_hotspot_group_rows"]
            sector_front_acceleration_rows = cached_validation["sector_front_acceleration_rows"]
            sector_front_acceleration_hole_rows = cached_validation["sector_front_acceleration_hole_rows"]
            sweep_rows = cached_validation["sweep_rows"]
            sweep_drift_rows = cached_validation["sweep_drift_rows"]
            radial_consistency_summary = cached_validation["radial_consistency_summary"]
            tracker.progress(2, validation_stage_total, message="Cached validation outputs loaded")
            tracker.progress(3, validation_stage_total, message="Validation cache complete")
        elif cfg.validation.enabled:
            tracker.progress(1, validation_stage_total, message="Running RDF archetype perturbation sweeps")
            rdf_stability_rows, rdf_stability_summary_rows, rdf_stability_summary = run_rdf_archetype_perturbation_sweeps(
                stabilized,
                holes_by_frame,
                chosen_descriptor,
                cfg,
                per_hole_rdf_archetype_rows,
                per_hole_rdf_archetype_centroids,
            )
            tracker.progress(2, validation_stage_total, message="Computing per-hole RDF bootstrap summaries")
            bootstrap_parallel_cfg = {**asdict(cfg.parallel), 'min_parallel_tasks': max(int(cfg.parallel.min_parallel_tasks), 24)}
            rdf_bootstrap_rows = build_per_hole_rdf_bootstrap_summary(per_hole_rdf_frame_rows, n_boot=cfg.radial.rdf_bootstrap_n, parallel_cfg=bootstrap_parallel_cfg)
            tracker.progress(3, validation_stage_total, message="Computing RDF bootstrap class support")
            rdf_bootstrap_support_rows = build_rdf_archetype_bootstrap_support(per_hole_rdf_frame_rows, per_hole_rdf_archetype_centroids, n_boot=cfg.radial.rdf_bootstrap_n, parallel_cfg=bootstrap_parallel_cfg)
            tracker.progress(4, validation_stage_total, message="Building sector propagation summaries")
            sector_front_propagation_rows, sector_front_propagation_hole_rows = build_sector_front_propagation_table(sector_frame_table, onset_threshold=cfg.radial.sector_lag_onset_threshold)
            tracker.progress(5, validation_stage_total, message="Building uncertainty summaries")
            validation_hole_table = ValidationHoleTable.from_rows(
                rdf_bootstrap_rows,
                rdf_bootstrap_support_rows,
                sector_front_propagation_hole_rows,
                sector_front_acceleration_hole_rows,
            )
            rdf_uncertainty_table = RdfUncertaintyHoleTable.from_rows(
                hotspot_table,
                rdf_bootstrap_rows,
                rdf_bootstrap_support_rows,
                per_hole_rdf_archetype_rows,
                zone_by_hole,
                per_hole_rdf_dynamics_rows,
                validation_table=validation_hole_table,
            )
            rdf_uncertainty_reticulum_rows = build_rdf_uncertainty_reticulum_rows_table(rdf_uncertainty_table)
            rdf_uncertainty_hotspot_rows, rdf_uncertainty_hotspot_group_rows = build_rdf_uncertainty_hotspot_comparison_table(rdf_uncertainty_table)
            sector_front_acceleration_rows, sector_front_acceleration_hole_rows = build_sector_front_acceleration_table(sector_frame_table)
            tracker.progress(6, validation_stage_total, message="Running radial perturbation sweeps")
            sweep_rows, sweep_drift_rows, radial_consistency_summary = run_radial_perturbation_sweeps(
                stabilized,
                holes_by_frame,
                chosen_descriptor,
                cfg,
                aggregate_radial_rows_base,
                radial_frame_rows,
                radial_conclusion_summary,
            )
            _write_validation_cache_manifest(radial_dir, qc_dir, validation_signature, validation_enabled=True)
            tracker.progress(7, validation_stage_total, message="Validation sweeps complete")
        else:
            tracker.progress(1, validation_stage_total, message="Skipping perturbation sweeps and bootstrap summaries")
            _write_validation_cache_manifest(radial_dir, qc_dir, validation_signature, validation_enabled=False)
            tracker.progress(2, validation_stage_total, message="Validation profile complete")

    validation_hole_table = ValidationHoleTable.from_rows(
        rdf_bootstrap_rows,
        rdf_bootstrap_support_rows,
        sector_front_propagation_hole_rows,
        sector_front_acceleration_hole_rows,
    )
    if cfg.validation.enabled and rdf_uncertainty_table is None and per_hole_rdf_archetype_rows:
        rdf_uncertainty_table = RdfUncertaintyHoleTable.from_rows(
            hotspot_table,
            rdf_bootstrap_rows,
            rdf_bootstrap_support_rows,
            per_hole_rdf_archetype_rows,
            zone_by_hole,
            per_hole_rdf_dynamics_rows,
            validation_table=validation_hole_table,
        )

    with pipeline_progress.stage("Hotspot and perturbation outputs", total=6, message="Writing hotspot tables, validation files, and lightweight perturbation checks") as tracker:
        tracker.progress(1, 6, message="Writing hotspot and RDF validation tables")
        rdf_stability_summary_columns = columns_from_records(rdf_stability_summary_rows, [
            'sweep_id', 'brightness_factor', 'radius_scale', 'rdf_archetype_stability_fraction', 'n_holes_compared', 'is_base',
        ])
        rdf_bootstrap_plot_columns = validation_hole_table.bootstrap_ci_plot_columns(limit=12)
        rdf_bootstrap_support_plot_columns = validation_hole_table.bootstrap_support_plot_columns(limit=12)
        _write_cached_table(radial_dir / "per_hole_rdf_stability.csv", rdf_stability_rows)
        _write_cached_table(radial_dir / "per_hole_rdf_stability_summary.csv", rdf_stability_summary_rows)
        _write_cached_table(radial_dir / "per_hole_rdf_bootstrap_summary.csv", rdf_bootstrap_rows)
        _write_cached_table(radial_dir / "rdf_archetype_bootstrap_support.csv", rdf_bootstrap_support_rows)
        _write_cached_table(radial_dir / "sector_front_propagation.csv", sector_front_propagation_rows)
        _write_cached_table(radial_dir / "sector_front_propagation_hole_summary.csv", sector_front_propagation_hole_rows)
        if rdf_uncertainty_table is not None:
            rdf_uncertainty_reticulum_columns = rdf_uncertainty_table.reticulum_columns()
            rdf_uncertainty_hotspot_columns, rdf_uncertainty_hotspot_group_columns = rdf_uncertainty_table.hotspot_comparison_columns()
            write_table_columns(radial_dir / "rdf_uncertainty_reticulum.csv", rdf_uncertainty_reticulum_columns)
            write_table_columns(radial_dir / "rdf_uncertainty_hotspot_comparison.csv", rdf_uncertainty_hotspot_columns)
            write_table_columns(radial_dir / "rdf_uncertainty_hotspot_group_summary.csv", rdf_uncertainty_hotspot_group_columns)
        else:
            _write_cached_table(radial_dir / "rdf_uncertainty_reticulum.csv", rdf_uncertainty_reticulum_rows)
            _write_cached_table(radial_dir / "rdf_uncertainty_hotspot_comparison.csv", rdf_uncertainty_hotspot_rows)
            _write_cached_table(radial_dir / "rdf_uncertainty_hotspot_group_summary.csv", rdf_uncertainty_hotspot_group_rows)
        _write_cached_table(radial_dir / "sector_front_acceleration.csv", sector_front_acceleration_rows)
        _write_cached_table(radial_dir / "sector_front_acceleration_hole_summary.csv", sector_front_acceleration_hole_rows)
        _write_cached_table(qc_dir / "radial_perturbation_sweeps.csv", sweep_rows)
        _write_cached_table(qc_dir / "radial_perturbation_drift.csv", sweep_drift_rows)
        write_json(qc_dir / "radial_conclusion_consistency.json", radial_consistency_summary)
        write_json(qc_dir / "rdf_archetype_stability_summary.json", rdf_stability_summary)

        tracker.progress(2, 6, message="Writing uncertainty and validation summary json")
        rdf_bootstrap_summary_json, rdf_uncertainty_summary_json = build_validation_summary_jsons_table(
            validation_hole_table,
            rdf_uncertainty_table if cfg.validation.enabled else None,
            validation_enabled=bool(cfg.validation.enabled),
        )
        write_json(qc_dir / "rdf_bootstrap_summary.json", rdf_bootstrap_summary_json)
        write_json(qc_dir / "rdf_uncertainty_summary.json", rdf_uncertainty_summary_json)
        if cfg.validation.enabled:
            _write_validation_cache_manifest(radial_dir, qc_dir, validation_signature, validation_enabled=True)
        _write_radial_drift_map(qc_dir, sweep_drift_rows)

        tracker.progress(3, 6, message="Writing hotspot tracks and summaries")
        write_table(hotspots_dir / "hotspots.csv", hotspot_rows)
        write_table(hotspots_dir / "tracks.csv", hotspot_track_rows)
        track_summaries = summarize_tracks(hotspot_rows)
        write_table(hotspots_dir / "track_summary.csv", track_summaries)

        tracker.progress(4, 6, message="Running lightweight perturbation checks")
        if ref_holes:
            _radial_plot_for_first_hole(radial_dir, radial_table, ref_holes[0].hole_id, chosen_descriptor, cfg.masks.n_terraces)
        descriptor_ref = baseline_descriptor
        for hole in ref_holes[: min(5, len(ref_holes))]:
            stab = radial_curve_stability(descriptor_ref, hole, cfg.radial)
            perturb_rows.append({"hole_id": hole.hole_id, **stab})
        write_table(qc_dir / "radial_perturbation.csv", perturb_rows)

        tracker.progress(5, 6, message="Writing validation plots")
        _write_rdf_stability_plot(radial_dir, rdf_stability_summary_columns)
        _write_rdf_bootstrap_plot(radial_dir, rdf_bootstrap_plot_columns)
        _write_rdf_bootstrap_support_plot(radial_dir, rdf_bootstrap_support_plot_columns)
        _write_sector_propagation_plot(radial_dir, sector_hole_summary_plot_columns(sector_front_propagation_hole_rows, value_field='sector_front_velocity_anisotropy'))
        save_image(radial_dir / "rdf_uncertainty_reticulum_map.png", _rdf_uncertainty_reticulum_map(rdf_uncertainty_reticulum_rows, ref_holes, lattice))
        _write_rdf_uncertainty_hotspot_plot(radial_dir, rdf_uncertainty_table.hotspot_comparison_columns()[1] if rdf_uncertainty_table is not None else rdf_uncertainty_hotspot_group_rows)
        _write_sector_acceleration_plot(radial_dir, sector_hole_summary_plot_columns(sector_front_acceleration_hole_rows, value_field='mean_sector_front_acceleration_per_frame2'))

        tracker.progress(6, 6, message="Hotspot and perturbation outputs complete")

    event_rows: list[dict[str, Any]] = []
    if matrix_rows:
        vals = np.asarray([r["mean_S"] for r in matrix_rows], dtype=float)
        event_rows.append(asdict(summarize_curve_events("matrix_mean_S", vals)))
        vals = np.asarray([r.get("hotspot_fraction", 0.0) for r in matrix_rows], dtype=float)
        event_rows.append(asdict(summarize_curve_events("hotspot_fraction", vals)))
        vals = np.asarray([r.get("primary_descriptor_mean", np.nan) for r in matrix_rows], dtype=float)
        event_rows.append(asdict(summarize_curve_events(f"matrix_primary_{chosen_descriptor}", vals)))
    annulus_event_rows, per_hole_summary_rows, per_hole_detail_rows = _lag_summary_from_radial_rows(radial_rows, ref_holes, chosen_descriptor, cfg.masks.n_terraces)
    event_rows.extend(annulus_event_rows)

    with pipeline_progress.stage("Temporal phenotypes", total=5, message="Building temporal events and phenotype summaries") as tracker:
        tracker.progress(1, 5, message="Writing temporal event tables")
        write_table(temporal_dir / "events.csv", event_rows)
        write_table(temporal_dir / "per_hole_events.csv", per_hole_summary_rows)
        write_table(temporal_dir / "per_hole_annulus_events.csv", per_hole_detail_rows)

        tracker.progress(2, 5, message="Building propagation features and phenotypes")
        propagation_rows = build_propagation_feature_rows(per_hole_summary_rows, per_hole_detail_rows)
        lattice_uv_by_hole = {int(row["hole_id"]): (row.get("lattice_u"), row.get("lattice_v")) for row in _make_hole_lattice_rows(ref_holes, lattice_indices)}
        for row in propagation_rows:
            uv = lattice_uv_by_hole.get(int(row["hole_id"]), (None, None))
            row["lattice_u"] = uv[0]
            row["lattice_v"] = uv[1]
        phenotype_rows, centroid_rows = assign_hole_phenotypes(propagation_rows, k=cfg.temporal.cluster_k)
        stability_rows, stability_summary, rerun_assignment_rows = phenotype_stability_across_reruns(
            propagation_rows,
            phenotype_rows,
            base_centroid_rows=centroid_rows,
            k=cfg.temporal.cluster_k,
            n_reruns=cfg.temporal.stability_reruns,
            jitter_scale=cfg.temporal.stability_jitter_scale,
        )
        phenotype_table = PhenotypeTable.from_rows(phenotype_rows)
        coherence_fraction, neighbor_rows, spatial_smoothness, smoothness_rows, smoothness_summary = build_phenotype_neighbor_and_smoothness_table(
            phenotype_table,
            radius=cfg.temporal.spatial_neighbor_radius,
        )
        archetype_rows = build_phenotype_archetype_rows_table(radial_table, phenotype_table)

        tracker.progress(3, 5, message="Writing phenotype tables")
        write_table(temporal_dir / "annulus_propagation_summary.csv", propagation_rows)
        write_table(temporal_dir / "per_hole_phenotypes.csv", phenotype_rows)
        write_table(temporal_dir / "phenotype_stability.csv", stability_rows)
        write_table(temporal_dir / "phenotype_rerun_assignments.csv", rerun_assignment_rows)
        write_table(temporal_dir / "phenotype_neighbor_pairs.csv", neighbor_rows)
        write_table(temporal_dir / "phenotype_spatial_smoothness.csv", smoothness_rows)
        write_table(temporal_dir / "phenotype_archetypes.csv", archetype_rows)

        tracker.progress(4, 5, message="Writing phenotype summaries and plots")
        write_json(
            temporal_dir / "phenotype_coherence.json",
            {
                "neighbor_coherence_fraction": coherence_fraction,
                "spatial_smoothness_fraction": spatial_smoothness,
                **stability_summary,
                **smoothness_summary,
            },
        )
        write_json(
            temporal_dir / "phenotype_canonicalization.json",
            {
                "canonical_labels": [
                    {
                        "canonical_id": row.get("canonical_id"),
                        "canonical_label": row.get("canonical_label"),
                        "semantic_label": row.get("label"),
                        "cluster_id": row.get("cluster_id"),
                    }
                    for row in centroid_rows
                ]
            },
        )
        write_centroids_json(temporal_dir / "phenotype_centroids.json", centroid_rows)
        if phenotype_rows:
            phenotype_counts = phenotype_table.count_by_label()
            save_bar_plot(
                temporal_dir / "phenotype_counts.png",
                list(phenotype_counts.keys()),
                list(phenotype_counts.values()),
                title="Hole phenotype counts",
                ylabel="count",
            )
            save_image(temporal_dir / "phenotype_spatial_map.png", _phenotype_overlay(ref_frame.image, ref_holes, phenotype_rows))
            save_image(temporal_dir / "phenotype_reticulum_map.png", _phenotype_reticulum_map(phenotype_rows, ref_holes, lattice))
            _write_phenotype_archetype_plot(temporal_dir, archetype_rows)
        if matrix_rows:
            x = [r["frame_id"] for r in matrix_rows]
            save_line_plot(
                temporal_dir / "matrix_and_hotspot_fraction.png",
                x,
                [
                    ("mean_S", [r["mean_S"] for r in matrix_rows]),
                    ("hotspot_fraction", [r.get("hotspot_fraction", 0.0) for r in matrix_rows]),
                    (f"primary_{chosen_descriptor}", [r.get("primary_descriptor_mean", np.nan) for r in matrix_rows]),
                ],
                title="Matrix signal and hotspot fraction",
                ylabel="value",
            )

        tracker.progress(5, 5, message="Temporal phenotypes complete")

        temporal_validation_summary = TemporalValidationSummary.from_rows(
            per_hole_summary_rows,
            propagation_rows,
            stability_summary,
            coherence_fraction,
            spatial_smoothness,
            cfg.temporal.min_valid_annuli_onset,
            cfg.temporal.min_valid_annuli_peak,
        )

        perturb_ok = (not perturb_rows) or (
            max(float(r.get("mae", 0.0)) for r in perturb_rows) <= cfg.radial.max_mae_threshold
            and max(float(r.get("max_abs", 0.0)) for r in perturb_rows) <= cfg.radial.max_max_abs_threshold
        )
        geometry_rows_ok = len(geometry_rows) == len(stabilized) * len(ref_holes)
        lag_rows_ok = len(per_hole_summary_rows) > 0
        propagation_rows_ok = len(propagation_rows) > 0
        radial_archetype_rows_ok = len(radial_archetype_rows) > 0
        angular_rows_ok = len(angular_rows) > 0
        reticulum_group_rows_ok = len(reticulum_group_rows) > 0
        radial_model_rows_ok = len(radial_model_fit_rows) > 0
        per_hole_rdf_rows_ok = len(per_hole_rdf_columns.evolution.get("hole_id", [])) > 0
        per_hole_rdf_velocity_ok = len(per_hole_rdf_velocity_rows) > 0
        sector_front_rows_ok = len(sector_front_rows) > 0
        hotspot_reticulum_row_count = len(hotspot_reticulum_columns.get('hole_id', [])) if 'hotspot_reticulum_columns' in locals() else len(hotspot_reticulum_rows)
        hotspot_reticulum_rows_ok = hotspot_reticulum_row_count > 0
        sector_front_lag_rows_ok = len(sector_front_lag_rows) > 0
        rdf_hotspot_reticulum_row_count = len(rdf_hotspot_reticulum_columns.get('hole_id', [])) if 'rdf_hotspot_reticulum_columns' in locals() else len(rdf_hotspot_reticulum_rows)
        rdf_hotspot_reticulum_rows_ok = rdf_hotspot_reticulum_row_count > 0
        rdf_bootstrap_rows_ok = len(rdf_bootstrap_rows) > 0
        rdf_bootstrap_support_ok = len(rdf_bootstrap_support_rows) > 0
        sector_front_propagation_ok = len(sector_front_propagation_rows) > 0
        rdf_uncertainty_reticulum_ok = len(rdf_uncertainty_reticulum_rows) > 0
        rdf_uncertainty_hotspot_ok = len(rdf_uncertainty_hotspot_rows) > 0
        sector_front_acceleration_ok = len(sector_front_acceleration_rows) > 0
        rdf_archetype_stability_fraction = float(rdf_stability_summary.get("mean_rdf_archetype_stability")) if rdf_stability_summary.get("mean_rdf_archetype_stability") is not None else 0.0
        rdf_bootstrap_class_support = float(validation_hole_table.mean_bootstrap_support_fraction() or 0.0)
        sector_propagation_valid_fraction = float(validation_hole_table.mean_sector_velocity_valid_fraction() or 0.0)
        sector_acceleration_valid_fraction = float(validation_hole_table.mean_sector_acceleration_valid_fraction() or 0.0)
        asymmetry_valid_fraction = float(np.mean([np.isfinite(float(r.get("angular_asymmetry", np.nan))) for r in angular_rows])) if angular_rows else 0.0
        valid_onset_fraction = float(temporal_validation_summary.valid_onset_fraction)
        valid_peak_fraction = float(temporal_validation_summary.valid_peak_fraction)
        lag_monotonic_fraction = float(temporal_validation_summary.lag_monotonic_fraction)
        negative_lag_fraction = float(temporal_validation_summary.negative_lag_fraction)
        phenotype_stability_fraction = float(temporal_validation_summary.phenotype_stability_fraction)
        phenotype_canonical_agreement = float(temporal_validation_summary.phenotype_canonical_agreement)
        phenotype_neighbor_fraction = float(temporal_validation_summary.phenotype_neighbor_fraction)
        phenotype_spatial_fraction = float(temporal_validation_summary.phenotype_spatial_fraction)
        radial_conclusion_agreement = float(radial_consistency_summary.get("conclusion_agreement_fraction")) if radial_consistency_summary.get("conclusion_agreement_fraction") is not None else 0.0
        radial_profile_consistency = float(radial_consistency_summary.get("mean_profile_correlation")) if radial_consistency_summary.get("mean_profile_correlation") is not None else 0.0
    cluster_rdf_visualisation_status: dict[str, Any] = {"status": "not_run"}
    if cfg.visualisation.cluster_rdf.enabled:
        cluster_vis_cfg = cfg.visualisation.cluster_rdf
        cluster_vis_total = max(1, 2 + sum(
            int(bool(getattr(cluster_vis_cfg, flag)))
            for flag in [
                "generate_raw_rdf",
                "generate_active_rdf",
                "generate_dominant_chronogram",
                "generate_radial_snapshots",
                "generate_average_hole_rings",
                "generate_sector_fan",
                "generate_hole_barcode",
                "generate_front_trajectory",
                "generate_cluster_legend",
            ]
        ) + int(bool(cluster_vis_cfg.generate_montage)))
        with pipeline_progress.stage("Cluster RDF visualisations", total=cluster_vis_total, message="Writing cluster RDF visualisation option figures") as tracker:
            tracker.progress(1, cluster_vis_total, message="Loading cluster tensor and building visual summaries")
            cluster_rdf_visualisation_status = run_cluster_rdf_visualisations(
                out_dir,
                cfg,
                progress_callback=lambda c, t, message: tracker.progress(c, t, message=message),
            )
            tracker.progress(cluster_vis_total, cluster_vis_total, message=f"Cluster RDF visualisations {cluster_rdf_visualisation_status.get('status', 'unknown')}")
    with pipeline_progress.stage("QC and manifest", total=4, message="Evaluating gates and finalizing run artifacts") as tracker:
        tracker.progress(1, 4, message="Assembling QC gates")
        gates = [
            require(any(r.accepted for r in audit_records), "audit_has_accepted_frames", "At least one frame passed audit."),
            require(chosen_descriptor in {"r", "g", "b", "h", "s"}, "descriptor_selected", f"Chosen descriptor={chosen_descriptor}."),
            require(len(descriptor_ranking) > 0, "descriptor_ranking_written", f"Descriptor ranking rows={len(descriptor_ranking)}."),
            require(winner in cfg.photometry.candidate_methods, "photometry_winner_valid", f"Winner '{winner}' is configured."),
            require(len(candidates) >= 4, "geometry_min_candidates", f"Detected {len(candidates)} candidates."),
            require(lattice.confidence >= 0.0, "lattice_fitted", f"Lattice confidence={lattice.confidence:.3f}."),
            require(int(geometry_sanity.get("fail_count", 0)) == 0, "geometry_sanity_passed", f"Geometry sanity status={geometry_sanity.get('status', 'unknown')} warnings={geometry_sanity.get('warning_count', 0)} failures={geometry_sanity.get('fail_count', 0)}."),
            require(len(radial_rows) > 0, "radial_rows_written", f"Wrote {len(radial_rows)} radial rows."),
            require(any(r["area_px"] > 0 for r in matrix_rows), "matrix_mask_nonempty", "Matrix mask has non-empty support."),
            require(np.isfinite(registration_mean_residual), "registration_residual_finite", f"Registration residual={registration_mean_residual:.3f}."),
            require(registration_mean_residual <= max(8.0, 0.75 * float(np.mean([h.radius_outer_px for h in ref_holes])) if ref_holes else 8.0), "registration_residual_envelope", f"Registration residual={registration_mean_residual:.3f}."),
            require(len(event_rows) > 0, "temporal_events_written", f"Wrote {len(event_rows)} temporal events."),
            require(len(per_hole_summary_rows) > 0, "per_hole_events_written", f"Wrote {len(per_hole_summary_rows)} per-hole summaries."),
            require(len(track_summaries) >= 0, "hotspot_track_summary_written", f"Wrote {len(track_summaries)} hotspot track summaries."),
            require(perturb_ok, "radial_perturbation_stable", f"Wrote {len(perturb_rows)} perturbation rows."),
            require(geometry_rows_ok, "geometry_timeseries_complete", f"Wrote {len(geometry_rows)} geometry rows for {len(ref_holes)} holes across {len(stabilized)} frames."),
            require(lag_rows_ok, "per_hole_lag_summary_written", f"Wrote {len(per_hole_summary_rows)} lag summary rows."),
            require(propagation_rows_ok, "annulus_propagation_summary_written", f"Wrote {len(propagation_rows)} propagation rows."),
            require(radial_archetype_rows_ok, "radial_archetypes_written", f"Wrote {len(radial_archetype_rows)} per-hole radial archetype rows."),
            require(radial_model_rows_ok, "radial_model_fits_written", f"Wrote {len(radial_model_fit_rows)} per-hole radial model rows."),
            require(per_hole_rdf_rows_ok, "per_hole_rdf_written", f"Wrote {len(per_hole_rdf_columns.evolution.get('hole_id', []))} per-hole RDF rows."),
            require(per_hole_rdf_velocity_ok, "per_hole_rdf_velocity_written", f"Wrote {len(per_hole_rdf_velocity_rows)} per-hole RDF velocity rows."),
            require(sector_front_rows_ok, "sector_front_rows_written", f"Wrote {len(sector_front_rows)} sector front rows."),
            require(hotspot_reticulum_rows_ok, "hotspot_reticulum_rows_written", f"Wrote {hotspot_reticulum_row_count} hotspot-reticulum comparison rows."),
            require(sector_front_lag_rows_ok, "sector_front_lag_rows_written", f"Wrote {len(sector_front_lag_rows)} sector-front lag rows."),
            require(rdf_hotspot_reticulum_rows_ok, "rdf_hotspot_reticulum_rows_written", f"Wrote {rdf_hotspot_reticulum_row_count} RDF hotspot-reticulum comparison rows."),
            require(rdf_bootstrap_rows_ok, "rdf_bootstrap_rows_written", f"Wrote {len(rdf_bootstrap_rows)} RDF bootstrap summary rows."),
            require(rdf_bootstrap_support_ok, "rdf_bootstrap_support_written", f"Wrote {len(rdf_bootstrap_support_rows)} RDF bootstrap support rows."),
            require(sector_front_propagation_ok, "sector_front_propagation_written", f"Wrote {len(sector_front_propagation_rows)} sector propagation rows."),
            require(rdf_uncertainty_reticulum_ok, "rdf_uncertainty_reticulum_written", f"Wrote {len(rdf_uncertainty_reticulum_rows)} RDF uncertainty reticulum rows."),
            require(rdf_uncertainty_hotspot_ok, "rdf_uncertainty_hotspot_written", f"Wrote {len(rdf_uncertainty_hotspot_rows)} RDF uncertainty-hotspot rows."),
            require(sector_front_acceleration_ok, "sector_front_acceleration_written", f"Wrote {len(sector_front_acceleration_rows)} sector acceleration rows."),
            require(angular_rows_ok, "angular_asymmetry_written", f"Wrote {len(angular_rows)} angular asymmetry rows."),
            require(reticulum_group_rows_ok, "reticulum_group_rows_written", f"Wrote {len(reticulum_group_rows)} reticulum-group radial rows."),
            require(asymmetry_valid_fraction >= cfg.radial.min_asymmetry_valid_fraction, "angular_asymmetry_valid_fraction", f"Angular asymmetry valid fraction={asymmetry_valid_fraction:.3f}."),
            require(rdf_archetype_stability_fraction >= cfg.radial.min_rdf_archetype_stability, "rdf_archetype_stability_fraction", f"RDF archetype stability fraction={rdf_archetype_stability_fraction:.3f}."),
            require(rdf_bootstrap_class_support >= cfg.radial.min_rdf_bootstrap_class_support, "rdf_bootstrap_class_support", f"RDF bootstrap class support={rdf_bootstrap_class_support:.3f}."),
            require(sector_propagation_valid_fraction >= cfg.radial.min_sector_propagation_valid_fraction, "sector_propagation_valid_fraction", f"Sector propagation valid fraction={sector_propagation_valid_fraction:.3f}."),
            require(sector_acceleration_valid_fraction >= cfg.radial.min_sector_acceleration_valid_fraction, "sector_acceleration_valid_fraction", f"Sector acceleration valid fraction={sector_acceleration_valid_fraction:.3f}."),
            require(valid_onset_fraction >= cfg.temporal.min_monotonic_fraction, "valid_onset_annuli_fraction", f"Fraction with >= {cfg.temporal.min_valid_annuli_onset} onset annuli: {valid_onset_fraction:.3f}."),
            require(valid_peak_fraction >= cfg.temporal.min_monotonic_fraction, "valid_peak_annuli_fraction", f"Fraction with >= {cfg.temporal.min_valid_annuli_peak} peak annuli: {valid_peak_fraction:.3f}."),
            require(lag_monotonic_fraction >= cfg.temporal.min_monotonic_fraction, "lag_monotonic_fraction", f"Monotonic lag fraction={lag_monotonic_fraction:.3f}."),
            require(negative_lag_fraction <= cfg.temporal.max_negative_lag_fraction, "negative_lag_fraction", f"Negative lag fraction={negative_lag_fraction:.3f}."),
            require(phenotype_stability_fraction >= cfg.temporal.min_phenotype_stability, "phenotype_stability_fraction", f"Mean phenotype stability={phenotype_stability_fraction:.3f}."),
            require(phenotype_canonical_agreement >= cfg.temporal.min_canonical_agreement, "phenotype_canonical_agreement", f"Canonical agreement={phenotype_canonical_agreement:.3f}."),
            require(phenotype_neighbor_fraction >= cfg.temporal.min_neighbor_coherence, "phenotype_neighbor_coherence", f"Neighbor phenotype coherence={phenotype_neighbor_fraction:.3f}."),
            require(phenotype_spatial_fraction >= cfg.temporal.min_spatial_smoothness, "phenotype_spatial_smoothness", f"Spatial smoothness={phenotype_spatial_fraction:.3f}."),
            require(len(archetype_rows) > 0, "phenotype_archetypes_written", f"Wrote {len(archetype_rows)} archetype rows."),
            require(radial_conclusion_agreement >= cfg.validation.min_conclusion_agreement, "radial_conclusion_agreement", f"Radial conclusion agreement={radial_conclusion_agreement:.3f}."),
            require(radial_profile_consistency >= cfg.validation.min_mean_profile_correlation, "radial_profile_correlation", f"Mean radial profile correlation={radial_profile_consistency:.3f}."),
        ]
        write_json(
            qc_dir / "threshold_envelope.json",
            {
                "registration_max_residual": max(8.0, 0.75 * float(np.mean([h.radius_outer_px for h in ref_holes])) if ref_holes else 8.0),
                "radial_max_mae": cfg.radial.max_mae_threshold,
                "radial_max_abs": cfg.radial.max_max_abs_threshold,
                "radial_archetype_k": cfg.radial.archetype_k,
                "radial_angular_n_sectors": cfg.radial.angular_n_sectors,
                "radial_min_asymmetry_valid_fraction": cfg.radial.min_asymmetry_valid_fraction,
                "radial_min_rdf_archetype_stability": cfg.radial.min_rdf_archetype_stability,
                "radial_sector_lag_onset_threshold": cfg.radial.sector_lag_onset_threshold,
                "radial_rdf_bootstrap_n": cfg.radial.rdf_bootstrap_n,
                "radial_min_rdf_bootstrap_class_support": cfg.radial.min_rdf_bootstrap_class_support,
                "radial_min_sector_propagation_valid_fraction": cfg.radial.min_sector_propagation_valid_fraction,
                "radial_min_sector_acceleration_valid_fraction": cfg.radial.min_sector_acceleration_valid_fraction,
                "radial_per_hole_rdf_rows": len(per_hole_rdf_columns.evolution.get("hole_id", [])),
                "radial_per_hole_rdf_velocity_rows": len(per_hole_rdf_velocity_rows),
                "radial_sector_front_lag_rows": len(sector_front_lag_rows),
                "radial_rdf_hotspot_reticulum_rows": rdf_hotspot_reticulum_row_count,
                "radial_modeling_enabled": True,
                "radial_sector_front_enabled": True,
                "radial_sector_front_acceleration_enabled": True,
                "radial_hotspot_reticulum_enabled": True,
                "validation_enabled": bool(cfg.validation.enabled),
                "hotspot_link_max_dist_px": cfg.hotspots.link_max_dist_px,
                "hotspot_max_area_ratio": cfg.hotspots.max_area_ratio,
                "chosen_descriptor": chosen_descriptor,
                "geometry_propagation_search_px": cfg.geometry.propagation_search_px,
                "geometry_smoothing_window": cfg.geometry.smoothing_window,
                "temporal_cluster_k": cfg.temporal.cluster_k,
                "temporal_min_valid_annuli_onset": cfg.temporal.min_valid_annuli_onset,
                "temporal_min_valid_annuli_peak": cfg.temporal.min_valid_annuli_peak,
                "temporal_min_monotonic_fraction": cfg.temporal.min_monotonic_fraction,
                "temporal_max_negative_lag_fraction": cfg.temporal.max_negative_lag_fraction,
                "temporal_stability_reruns": cfg.temporal.stability_reruns,
                "temporal_stability_jitter_scale": cfg.temporal.stability_jitter_scale,
                "temporal_min_phenotype_stability": cfg.temporal.min_phenotype_stability,
                "temporal_min_neighbor_coherence": cfg.temporal.min_neighbor_coherence,
                "temporal_min_canonical_agreement": cfg.temporal.min_canonical_agreement,
                "temporal_spatial_neighbor_radius": cfg.temporal.spatial_neighbor_radius,
                "temporal_min_spatial_smoothness": cfg.temporal.min_spatial_smoothness,
                "validation_brightness_factors": list(cfg.validation.brightness_factors),
                "validation_radius_scale_factors": list(cfg.validation.radius_scale_factors),
                "validation_min_conclusion_agreement": cfg.validation.min_conclusion_agreement,
                "validation_min_mean_profile_correlation": cfg.validation.min_mean_profile_correlation,
            },
        )
        write_gate_report(out_dir / "qc_gates.json", gates)
        tracker.progress(2, 4, message="Writing QC gate report")
        _maybe_assert_gates(cfg, gates)

        summary = _summary_payload(
            frames,
            audit_records,
            winner,
            candidates,
            lattice,
            radial_rows=len(radial_rows),
            reference_idx=ref_idx,
            registration_mean_residual=registration_mean_residual,
            n_hotspots=len(hotspot_rows),
            n_events=len(event_rows),
            n_geometry_rows=len(geometry_rows),
            geometry_sanity=geometry_sanity,
        )
        write_json(out_dir / "summary.json", summary)
        tracker.progress(3, 4, message="Writing summary and manifest")
        write_json(out_dir / "run_manifest.json", {"pipeline_config": asdict(cfg), "summary": summary})
        write_results_notebook(out_dir)
        _write_curated_output_index(out_dir, summary)
        tracker.progress(4, 4, message="QC and manifest complete")
    pipeline_progress.close()
    return summary

def run_milestone19(frames: list[FrameRecord], out_dir: Path, cfg: PipelineConfig | None = None, hole_grid_bundle: HoleGridBundle | None = None) -> dict[str, Any]:
    return run_milestone18(frames, out_dir, cfg, hole_grid_bundle=hole_grid_bundle)


def run_milestone20(frames: list[FrameRecord], out_dir: Path, cfg: PipelineConfig | None = None, hole_grid_bundle: HoleGridBundle | None = None) -> dict[str, Any]:
    return run_milestone18(frames, out_dir, cfg, hole_grid_bundle=hole_grid_bundle)


def run_milestone21(frames: list[FrameRecord], out_dir: Path, cfg: PipelineConfig | None = None, hole_grid_bundle: HoleGridBundle | None = None) -> dict[str, Any]:
    return run_milestone18(frames, out_dir, cfg, hole_grid_bundle=hole_grid_bundle)


def run_milestone22(frames: list[FrameRecord], out_dir: Path, cfg: PipelineConfig | None = None, hole_grid_bundle: HoleGridBundle | None = None) -> dict[str, Any]:
    return run_milestone18(frames, out_dir, cfg, hole_grid_bundle=hole_grid_bundle)


def run_milestone23(frames: list[FrameRecord], out_dir: Path, cfg: PipelineConfig | None = None, hole_grid_bundle: HoleGridBundle | None = None) -> dict[str, Any]:
    return run_milestone18(frames, out_dir, cfg, hole_grid_bundle=hole_grid_bundle)


def run_milestone24(frames: list[FrameRecord], out_dir: Path, cfg: PipelineConfig | None = None, hole_grid_bundle: HoleGridBundle | None = None) -> dict[str, Any]:
    return run_milestone18(frames, out_dir, cfg, hole_grid_bundle=hole_grid_bundle)
