from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import sys

from holecolor.audit.frame_audit import audit_sequence
from holecolor.config.schema import PipelineConfig
from holecolor.core.logging import get_logger
from holecolor.core.paths import ensure_dir
from holecolor.core.types import HoleCandidate
from holecolor.geometry.candidates import detect_dark_hole_candidates, detect_regular_grid_hole_candidates
from holecolor.geometry.conic_refine import conic_to_hole_geometry, refine_candidate_with_conic
from holecolor.holegrid.calibration import calibrate_hole_grid_from_frames
from holecolor.holegrid.model import load_hole_grid_model, save_hole_grid_model
from holecolor.geometry.lattice_fit import assign_lattice_indices, estimate_lattice_basis
from holecolor.geometry.overlays import draw_candidates
from holecolor.io.trim import FrameTrimSelection, select_frame_range_visual, suggest_histogram_stabilization_trim
from holecolor.io.video import iter_video_frames, probe_video, save_frame, strip_black_bands_from_frames
from holecolor.notebook import write_results_notebook
from holecolor.color_analysis import run_color_only_analysis
from holecolor.photometry.selector import run_photometry_selection
from holecolor.pipeline import run_milestone1, run_milestone24
from holecolor.core.status import format_status_line
from holecolor.qc.reports import write_json, write_table

LOG = get_logger(__name__)


def _read_trim_selection(path: Path) -> tuple[int, int, dict]:
    import json

    payload = json.loads(path.read_text(encoding="utf-8"))
    if "start_frame" not in payload or "end_frame" not in payload:
        raise ValueError(f"trim selection missing start_frame/end_frame: {path}")
    return int(payload["start_frame"]), int(payload["end_frame"]), dict(payload)


def _trim_payload(
    input_path: Path,
    meta,
    start_frame: int,
    end_frame: int,
    source: str,
    every_n: int,
    loaded_frame_count: int | None = None,
    extra: dict | None = None,
) -> dict:
    raw_count = int(max(0, int(end_frame) - int(start_frame) + 1))
    payload = {
        "source": str(source),
        "source_path": str(input_path),
        "total_frames": int(meta.n_frames),
        "fps": float(meta.fps),
        "start_frame": int(start_frame),
        "end_frame": int(end_frame),
        "raw_selected_frame_count": int(raw_count),
        "every_n": int(every_n),
        "loaded_frame_count": None if loaded_frame_count is None else int(loaded_frame_count),
        "start_time_s": float(start_frame) / max(float(meta.fps), 1e-6),
        "end_time_s": float(end_frame) / max(float(meta.fps), 1e-6),
    }
    if extra:
        payload["selection_file_payload"] = extra
    return payload


def _resolve_frame_trim(args: argparse.Namespace, input_path: Path, out: Path | None, meta) -> tuple[int | None, int | None]:
    start = getattr(args, "start_frame", None)
    end = getattr(args, "end_frame", None)
    source = None
    extra = None
    selection_path = getattr(args, "trim_selection", None)
    if selection_path:
        start, end, extra = _read_trim_selection(Path(selection_path))
        source = f"selection_file:{selection_path}"
    if bool(getattr(args, "trim_ui", False)):
        initial_start = 0 if start is None else int(start)
        initial_end = None if end is None else int(end)
        suggestion = None
        if not bool(getattr(args, "no_trim_autosuggest", False)):
            try:
                suggestion = suggest_histogram_stabilization_trim(input_path, meta=meta)
                if suggestion is not None:
                    LOG.info(
                        "trim autosuggest: cursor=%d first_stable=%d confidence=%.3f stable_run=%d",
                        int(suggestion.cursor_frame),
                        int(suggestion.first_stable_frame),
                        float(suggestion.confidence),
                        int(suggestion.stable_run_frames),
                    )
            except Exception as exc:
                LOG.warning("trim autosuggest unavailable: %s", exc)
                suggestion = None
        selection: FrameTrimSelection = select_frame_range_visual(input_path, initial_start=initial_start, initial_end=initial_end, meta=meta, suggestion=suggestion)
        start = int(selection.start_frame)
        end = int(selection.end_frame)
        source = "visual_ui"
        extra = selection.to_jsonable()
        if suggestion is not None:
            extra["histogram_stabilization_suggestion"] = suggestion.to_jsonable()
    elif start is not None or end is not None:
        n = int(meta.n_frames)
        start = 0 if start is None else max(0, int(start))
        end = (n - 1 if n > 0 else int(start)) if end is None else max(int(start), int(end))
        if n > 0:
            end = min(int(end), n - 1)
        source = source or "cli_flags"
    if start is None and end is None:
        return None, None
    if start is None:
        start = 0
    if end is None:
        end = int(meta.n_frames) - 1 if int(meta.n_frames) > 0 else int(start)
    if out is not None:
        write_json(out / "logs" / "frame_trim_selection.json", _trim_payload(input_path, meta, int(start), int(end), str(source or "unknown"), int(args.every_n), extra=extra))
    LOG.info(
        "frame trim: source=%s start=%d end=%d raw_frames=%d",
        source or "unknown",
        int(start),
        int(end),
        int(max(0, int(end) - int(start) + 1)),
    )
    return int(start), int(end)


def _load_command_frames(args: argparse.Namespace, out: Path | None = None, meta=None) -> list:
    input_path = Path(args.input)
    meta = meta or probe_video(input_path)
    start, end = _resolve_frame_trim(args, input_path, out, meta)
    frames = iter_video_frames(
        input_path,
        every_n=args.every_n,
        show_progress=not getattr(args, "no_progress", False),
        start_frame=start,
        end_frame=end,
    )
    if not bool(getattr(args, "no_black_band_crop", False)):
        frames, black_band_crop = strip_black_bands_from_frames(frames)
        if out is not None:
            write_json(out / "logs" / "black_band_crop.json", black_band_crop.to_jsonable())
        if black_band_crop.applied:
            LOG.info(
                "black-band crop: top=%d bottom=%d size=%dx%d -> %dx%d",
                int(black_band_crop.top),
                int(black_band_crop.bottom),
                int(black_band_crop.original_width),
                int(black_band_crop.original_height),
                int(black_band_crop.cropped_width),
                int(black_band_crop.cropped_height),
            )
        else:
            LOG.info("black-band crop: not applied (%s)", black_band_crop.reason)
    if start is not None and end is not None and out is not None:
        payload_path = out / "logs" / "frame_trim_selection.json"
        try:
            import json

            payload = json.loads(payload_path.read_text(encoding="utf-8"))
        except Exception:
            payload = _trim_payload(input_path, meta, int(start), int(end), "unknown", int(args.every_n))
        payload["loaded_frame_count"] = int(len(frames))
        payload["loaded_frame_ids"] = [int(frame.frame_id) for frame in frames]
        write_json(payload_path, payload)
    return frames


def _existing_acceptance_gates(out: Path) -> list[dict[str, str]]:
    candidates = [
        (
            "Reference geometry overlay",
            out / "geometry" / "overlays" / "frame_ref_geometry_overlay.png",
            "Does the detected wafer/lattice geometry match the video frame?",
        ),
        (
            "Hard cluster label video",
            out / "descriptors" / "wafer_nonhole_colour" / "video_cluster_labels.avi",
            "Are labels restricted to wafer non-hole pixels, with holes excluded?",
        ),
        (
            "Baseline activity recolour video",
            out / "descriptors" / "wafer_nonhole_colour" / "video_cluster_baseline_activity.avi",
            "Does opacity match baseline-corrected cluster activity against a neutral background?",
        ),
        (
            "Cluster RDF montage",
            out / "descriptors" / "radial_cluster_average_hole" / "00_holecolor_visualisation_montage.png",
            "Are the cluster RDF visualisations acceptable for interpretation?",
        ),
    ]
    gates = []
    for name, path, question in candidates:
        if path.exists():
            gates.append({"name": name, "path": str(path), "question": question})
    return gates


def _prompt_acceptance_gates(out: Path) -> int:
    gates = _existing_acceptance_gates(out)
    if not gates:
        LOG.warning("acceptance gates requested but no gate artifacts were found under %s", out)
        return 0

    def emit(message: str = "") -> None:
        print(message, file=sys.stderr, flush=True)

    pending_payload = {
        "timestamp_local": datetime.now().isoformat(timespec="seconds"),
        "status": "waiting_for_user",
        "accepted": None,
        "gates": [{**gate, "accepted": None, "answer": None} for gate in gates],
    }
    write_json(out / "logs" / "acceptance_gates.json", pending_payload)

    emit("")
    emit("=" * 78)
    emit("HOLECOLOR ACCEPTANCE GATES")
    emit("=" * 78)
    emit("The pipeline is complete. Open each artifact path, inspect it, then")
    emit("answer Y or N in this terminal. The process will block here until")
    emit("you answer each gate.")
    emit(f"Results file: {out / 'logs' / 'acceptance_gates.json'}")
    emit("=" * 78)

    results = []
    for idx, gate in enumerate(gates, start=1):
        emit("")
        emit(f"[{idx}/{len(gates)}] {gate['name']}")
        emit(f"Artifact: {gate['path']}")
        emit(f"Question: {gate['question']}")
        while True:
            emit("Answer [y/n], then press Enter:")
            raw = sys.stdin.readline()
            if raw == "":
                emit("No terminal input available; treating this gate as rejected.")
                accepted = False
                break
            answer = raw.strip().lower()
            if answer in {"y", "yes"}:
                accepted = True
                break
            if answer in {"n", "no"}:
                accepted = False
                break
            emit("Please answer y or n.")
        results.append({**gate, "accepted": accepted, "answer": "y" if accepted else "n"})
    payload = {
        "timestamp_local": datetime.now().isoformat(timespec="seconds"),
        "status": "accepted" if all(r["accepted"] for r in results) else "rejected",
        "accepted": bool(all(r["accepted"] for r in results)),
        "gates": results,
    }
    write_json(out / "logs" / "acceptance_gates.json", payload)
    if payload["accepted"]:
        LOG.info("acceptance gates: accepted | details=%s", out / "logs" / "acceptance_gates.json")
        return 0
    LOG.warning("acceptance gates: rejected | details=%s", out / "logs" / "acceptance_gates.json")
    return 2


def _log_geometry_sanity(summary: dict, out: Path) -> None:
    sanity = summary.get("geometry_sanity") or {}
    status = str(sanity.get("status", "unknown"))
    fail_count = int(sanity.get("fail_count", 0) or 0)
    warning_count = int(sanity.get("warning_count", 0) or 0)
    path = out / "geometry" / "geometry_sanity_checks.json"
    if fail_count > 0:
        LOG.warning("geometry sanity: FAILED | failures=%d warnings=%d | details=%s", fail_count, warning_count, path)
    elif warning_count > 0:
        LOG.warning("geometry sanity: WARN | warnings=%d | details=%s", warning_count, path)
    else:
        LOG.info("geometry sanity: %s | details=%s", status, path)
    for check in sanity.get("checks", []):
        if check.get("passed", True):
            continue
        level = str(check.get("severity", "warn"))
        detail = str(check.get("detail", ""))
        name = str(check.get("name", "geometry_check"))
        if level == "fail":
            LOG.warning("geometry sanity failure: %s | %s", name, detail)
        else:
            LOG.warning("geometry sanity warning: %s | %s", name, detail)
    complete_filter = sanity.get("complete_geometry_filter") or {}
    if complete_filter:
        LOG.info(
            "complete geometry filter: kept=%d excluded=%d | details=%s",
            int(complete_filter.get("kept_holes", 0) or 0),
            int(complete_filter.get("excluded_holes", 0) or 0),
            out / "geometry" / "hole_terrace_exclusion.csv",
        )


def _apply_runtime_profile(cfg: PipelineConfig, args: argparse.Namespace) -> str:
    profile = "standard"
    if bool(getattr(args, "fast", False)):
        profile = "fast"
        cfg.validation.enabled = False
        cfg.radial.rdf_bootstrap_n = min(int(cfg.radial.rdf_bootstrap_n), 32)
        cfg.temporal.stability_reruns = min(int(cfg.temporal.stability_reruns), 2)
        cfg.parallel.backend = "thread" if getattr(args, "parallel_backend", "auto") == "auto" else cfg.parallel.backend
    return profile


def cmd_audit(args: argparse.Namespace) -> int:
    out = ensure_dir(Path(args.out)) if args.out else None
    frames = _load_command_frames(args, out=out)
    cfg = PipelineConfig()
    cfg.parallel.show_progress = not getattr(args, "no_progress", False)
    records = audit_sequence(frames, cfg.audit, progress_cfg=cfg.parallel)
    accepted = sum(r.accepted for r in records)
    if out is not None:
        write_table(out / "frame_qc.csv", records)
        write_json(out / "summary.json", {"n_frames": len(records), "accepted": int(accepted)})
    LOG.info("audited %d frames, accepted=%d", len(records), accepted)
    return 0


def cmd_photometry(args: argparse.Namespace) -> int:
    out = ensure_dir(Path(args.out)) if args.out else None
    frames = _load_command_frames(args, out=out)
    cfg = PipelineConfig()
    cfg.parallel.show_progress = not getattr(args, "no_progress", False)
    if hasattr(args, "workers") and args.workers is not None:
        cfg.parallel.max_workers = int(args.workers)
    if hasattr(args, "parallel_backend") and args.parallel_backend is not None:
        cfg.parallel.backend = str(args.parallel_backend)
    winner, scores, corrected = run_photometry_selection(frames, cfg.photometry, progress_cfg=cfg.parallel)
    if out is not None:
        write_table(out / "candidate_scores.csv", scores)
        write_json(out / "winner.json", {"winner": winner})
        save_frame(out / f"frame0_{winner}.png", corrected[0].image)
    LOG.info("best correction: %s", winner)
    return 0


def cmd_geometry(args: argparse.Namespace) -> int:
    out = ensure_dir(Path(args.out)) if args.out else ensure_dir(Path("holegrid_calibration"))
    frames = _load_command_frames(args, out=out)
    cfg = PipelineConfig()
    cfg.parallel.show_progress = not getattr(args, "no_progress", False)
    if hasattr(args, "workers") and args.workers is not None:
        cfg.parallel.max_workers = int(args.workers)
    if hasattr(args, "parallel_backend") and args.parallel_backend is not None:
        cfg.parallel.backend = str(args.parallel_backend)
    calib = calibrate_hole_grid_from_frames(frames, cfg.geometry, sample_limit=max(1, int(getattr(args, "sample_frames", 8))))
    candidates = [
        HoleCandidate(
            float(h.x),
            float(h.y),
            float(max(1.0, h.radius_outer_px)),
            0.0,
            0.0,
            float(h.confidence),
        )
        for h in calib.holes
    ]
    lattice_indices = assign_lattice_indices(candidates, calib.lattice)
    overlay = draw_candidates(
        frames[calib.frame_id].image,
        candidates,
        lattice=calib.lattice,
        lattice_indices=lattice_indices,
        support_circle=calib.support_circle,
    )
    write_table(out / "hole_candidates.csv", [
        {
            "candidate_id": i,
            "x": c.x,
            "y": c.y,
            "radius_px": c.radius_px,
            "ellipticity": c.ellipticity,
            "boundary_contrast": c.boundary_contrast,
            "confidence": c.confidence,
            "lattice_u": None if i not in lattice_indices else lattice_indices[i][0],
            "lattice_v": None if i not in lattice_indices else lattice_indices[i][1],
        }
        for i, c in enumerate(candidates)
    ])
    write_table(out / "hole_geometry.csv", calib.holes)
    write_table(out / "hole_lattice_index.csv", [
        {"hole_id": i, "lattice_u": lattice_indices.get(i, (None, None))[0], "lattice_v": lattice_indices.get(i, (None, None))[1]}
        for i in range(len(candidates))
    ])
    write_json(out / "lattice_model.json", {
        "origin_x": calib.lattice.origin_x,
        "origin_y": calib.lattice.origin_y,
        "basis_u": calib.lattice.basis_u,
        "basis_v": calib.lattice.basis_v,
        "angle_deg": calib.lattice.angle_deg,
        "spacing_u_px": calib.lattice.spacing_u_px,
        "spacing_v_px": calib.lattice.spacing_v_px,
        "confidence": calib.lattice.confidence,
    })
    save_hole_grid_model(out / "hole_grid_model.json", calib.bundle)
    save_frame(out / f"frame_{calib.frame_id:03d}_holegrid_overlay.png", overlay)
    write_json(out / "hole_grid_summary.json", {
        "source_frame_id": calib.frame_id,
        "support_mode": calib.support_mode,
        "support_circle": calib.support_circle,
        "raw_count": calib.raw_count,
        "filtered_count": calib.filtered_count,
        "completed_count": calib.completed_count,
        "n_holes": len(calib.holes),
    })
    LOG.info("holegrid: frame=%d holes=%d support=%s lattice confidence=%.3f", calib.frame_id, len(calib.holes), calib.support_mode, calib.lattice.confidence)
    return 0


def cmd_color(args: argparse.Namespace) -> int:
    out = ensure_dir(Path(args.out)) if args.out else ensure_dir(Path("holecolor_color_only"))
    frames = _load_command_frames(args, out=out)
    cfg = PipelineConfig()
    cfg.parallel.show_progress = not getattr(args, "no_progress", False)
    if hasattr(args, "workers") and args.workers is not None:
        cfg.parallel.max_workers = int(args.workers)
    if hasattr(args, "parallel_backend") and args.parallel_backend is not None:
        cfg.parallel.backend = str(args.parallel_backend)
    summary = run_color_only_analysis(frames, out, cfg)
    LOG.info("color-only summary | accepted=%d/%d | correction=%s", summary["accepted_frames"], summary["n_frames"], summary["winner_correction"])
    return 0


def cmd_radial(args: argparse.Namespace) -> int:
    meta = probe_video(Path(args.input))
    LOG.info("source=%s frames=%d size=%dx%d fps=%.3f", meta.source_path, meta.n_frames, meta.width, meta.height, meta.fps)
    out = ensure_dir(Path(args.out)) if args.out else ensure_dir(Path("holecolor_radial_run"))
    frames = _load_command_frames(args, out=out, meta=meta)
    cfg = PipelineConfig()
    cfg.parallel.show_progress = not getattr(args, "no_progress", False)
    if hasattr(args, "workers") and args.workers is not None:
        cfg.parallel.max_workers = int(args.workers)
    if hasattr(args, "parallel_backend") and args.parallel_backend is not None:
        cfg.parallel.backend = str(args.parallel_backend)
    cfg.visualisation.cluster_rdf.enabled = not bool(getattr(args, "no_cluster_rdf_visualisations", False))
    cfg.qc.fail_on_gate_error = bool(getattr(args, "strict_gates", False))
    profile = _apply_runtime_profile(cfg, args)
    LOG.info("profile=%s | status -> %s | events -> %s | summary -> %s | plan -> %s", profile, out / "logs" / "current_status.json", out / "logs" / "run_status.jsonl", out / "logs" / "progress_summary.txt", out / "logs" / "stage_plan.json")
    bundle = load_hole_grid_model(Path(args.hole_grid_model)) if getattr(args, "hole_grid_model", None) else None
    summary = run_milestone24(frames, out, cfg, hole_grid_bundle=bundle)
    _log_geometry_sanity(summary, out)
    LOG.info(
        "radial summary | accepted=%d/%d | correction=%s | candidates=%d | radial_rows=%d",
        summary["accepted_frames"],
        summary["n_frames"],
        summary["winner_correction"],
        summary["n_candidates"],
        summary["n_radial_rows"],
    )
    if bool(getattr(args, "acceptance_gates", False)):
        return _prompt_acceptance_gates(out)
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    meta = probe_video(Path(args.input))
    LOG.info("source=%s frames=%d size=%dx%d fps=%.3f", meta.source_path, meta.n_frames, meta.width, meta.height, meta.fps)
    out = ensure_dir(Path(args.out)) if args.out else ensure_dir(Path("holecolor_run"))
    frames = _load_command_frames(args, out=out, meta=meta)
    cfg = PipelineConfig()
    cfg.parallel.show_progress = not getattr(args, "no_progress", False)
    if hasattr(args, "workers") and args.workers is not None:
        cfg.parallel.max_workers = int(args.workers)
    if hasattr(args, "parallel_backend") and args.parallel_backend is not None:
        cfg.parallel.backend = str(args.parallel_backend)
    cfg.visualisation.cluster_rdf.enabled = not bool(getattr(args, "no_cluster_rdf_visualisations", False))
    cfg.qc.fail_on_gate_error = bool(getattr(args, "strict_gates", False))
    profile = _apply_runtime_profile(cfg, args)
    LOG.info("profile=%s | status -> %s | events -> %s | summary -> %s | plan -> %s", profile, out / "logs" / "current_status.json", out / "logs" / "run_status.jsonl", out / "logs" / "progress_summary.txt", out / "logs" / "stage_plan.json")
    bundle = load_hole_grid_model(Path(args.hole_grid_model)) if getattr(args, "hole_grid_model", None) else None
    summary = run_milestone24(frames, out, cfg, hole_grid_bundle=bundle)
    _log_geometry_sanity(summary, out)
    LOG.info(
        "run summary | accepted=%d/%d | correction=%s | candidates=%d | radial_rows=%d | lattice_conf=%.3f",
        summary["accepted_frames"],
        summary["n_frames"],
        summary["winner_correction"],
        summary["n_candidates"],
        summary["n_radial_rows"],
        summary["lattice_confidence"],
    )
    if bool(getattr(args, "acceptance_gates", False)):
        return _prompt_acceptance_gates(out)
    return 0



def cmd_status(args: argparse.Namespace) -> int:
    run_dir = Path(args.input)
    current = run_dir / "logs" / "current_status.json"
    if not current.exists():
        raise SystemExit(f"status file not found: {current}")
    def read_payload() -> dict:
        import json
        return json.loads(current.read_text(encoding="utf-8"))
    if not getattr(args, "watch", False):
        payload = read_payload()
        print(format_status_line(payload))
        return 0
    import time
    interval = max(0.25, float(getattr(args, "interval", 1.0)))
    last_text = None
    try:
        while True:
            payload = read_payload()
            text = format_status_line(payload)
            if text != last_text:
                print(text, flush=True)
                last_text = text
            if payload.get("event") == "run_completed":
                return 0
            time.sleep(interval)
    except KeyboardInterrupt:
        return 130


def cmd_notebook(args: argparse.Namespace) -> int:
    run_dir = Path(args.input)
    out_path = write_results_notebook(run_dir)
    LOG.info("wrote notebook: %s", out_path)
    return 0


def cmd_accept(args: argparse.Namespace) -> int:
    return _prompt_acceptance_gates(Path(args.input))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="holecolor")
    sub = parser.add_subparsers(dest="command", required=True)

    def add_common(p: argparse.ArgumentParser) -> None:
        p.add_argument("input")
        p.add_argument("--every-n", type=int, default=1)
        p.add_argument("--out", type=str, default=None)
        p.add_argument("--strict-gates", action="store_true")
        p.add_argument("--workers", type=int, default=0)
        p.add_argument("--parallel-backend", choices=["auto", "process", "thread", "none"], default="auto")
        p.add_argument("--no-progress", action="store_true")
        p.add_argument("--hole-grid-model", type=str, default=None)
        p.add_argument("--fast", action="store_true", help="Skip the heaviest validation sweeps and use a lighter runtime profile")
        p.add_argument("--no-cluster-rdf-visualisations", action="store_true", help="Skip the late cluster RDF figure-rendering stage")
        p.add_argument("--trim-ui", action="store_true", help="Open a visual start/end frame selector before loading frames")
        p.add_argument("--no-trim-autosuggest", action="store_true", help="Do not initialise the trim UI from histogram/CDF stabilisation")
        p.add_argument("--start-frame", type=int, default=None, help="Inclusive first source frame to process")
        p.add_argument("--end-frame", type=int, default=None, help="Inclusive last source frame to process")
        p.add_argument("--trim-selection", type=str, default=None, help="Reuse a saved frame_trim_selection.json start/end range")
        p.add_argument("--no-black-band-crop", action="store_true", help="Disable automatic top/bottom black-band cropping before analysis")

    p = sub.add_parser("run")
    add_common(p)
    p.add_argument("--acceptance-gates", action="store_true", help="After completion, block for Y/N acceptance prompts on key visual artifacts")
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("audit")
    add_common(p)
    p.set_defaults(func=cmd_audit)

    p = sub.add_parser("photometry")
    add_common(p)
    p.set_defaults(func=cmd_photometry)

    p = sub.add_parser("geometry")
    add_common(p)
    p.set_defaults(func=cmd_geometry)

    p = sub.add_parser("holegrid")
    add_common(p)
    p.add_argument("--sample-frames", type=int, default=8)
    p.set_defaults(func=cmd_geometry)

    p = sub.add_parser("color")
    add_common(p)
    p.set_defaults(func=cmd_color)

    p = sub.add_parser("radial")
    add_common(p)
    p.add_argument("--acceptance-gates", action="store_true", help="After completion, block for Y/N acceptance prompts on key visual artifacts")
    p.set_defaults(func=cmd_radial)

    p = sub.add_parser("status")
    p.add_argument("input")
    p.add_argument("--watch", action="store_true")
    p.add_argument("--interval", type=float, default=1.0)
    p.set_defaults(func=cmd_status)

    p = sub.add_parser("notebook")
    p.add_argument("input")
    p.set_defaults(func=cmd_notebook)

    p = sub.add_parser("accept")
    p.add_argument("input", help="Existing run directory whose key artifacts should be reviewed with blocking Y/N prompts")
    p.set_defaults(func=cmd_accept)
    return parser



def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
