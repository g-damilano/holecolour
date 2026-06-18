from __future__ import annotations

from collections import defaultdict

import numpy as np

from holecolor.config.schema import ParallelConfig, PhotometryConfig
from holecolor.core.parallel import iter_with_progress, parallel_map, prefer_thread_for_image_tasks
from holecolor.core.types import FrameRecord, PhotometryScore
from holecolor.photometry.corrections import apply_correction_stack
from holecolor.photometry.metrics import edge_energy_ratio, hole_control_drift, positive_prominence


def _score_corrected_frame_task(task: tuple[FrameRecord, str, PhotometryConfig, np.ndarray | None, np.ndarray | None]) -> PhotometryScore:
    frame, correction_name, cfg, provisional_hole_mask, baseline = task
    eer = edge_energy_ratio(frame.image, cfg.border_width_px)
    pp = positive_prominence(frame.image)
    drift = hole_control_drift(frame.image, provisional_hole_mask, baseline)
    total = eer + 0.25 * pp + cfg.hole_control_weight * float(drift or 0.0) / 255.0
    return PhotometryScore(frame.frame_id, correction_name, eer, pp, drift, total)


def score_corrected_stack(
    frames: list[FrameRecord],
    correction_name: str,
    cfg: PhotometryConfig,
    provisional_hole_mask: np.ndarray | None = None,
    progress_cfg: ParallelConfig | None = None,
    desc: str | None = None,
) -> list[PhotometryScore]:
    baseline = frames[0].image if frames else None
    if not frames:
        return []
    task_cfg = prefer_thread_for_image_tasks(progress_cfg)
    tasks = [(frame, correction_name, cfg, provisional_hole_mask, baseline) for frame in frames]
    return parallel_map(_score_corrected_frame_task, tasks, task_cfg, desc=desc)


def choose_best_correction(scores: list[PhotometryScore]) -> str:
    grouped = defaultdict(list)
    for s in scores:
        grouped[s.correction_name].append(s.total_score)
    ranked = sorted((float(np.mean(v)), k) for k, v in grouped.items())
    return ranked[0][1]


def run_photometry_selection(
    frames: list[FrameRecord],
    cfg: PhotometryConfig,
    progress_cfg: ParallelConfig | None = None,
    progress_callback=None,
):
    all_scores: list[PhotometryScore] = []
    corrected_by_method: dict[str, list[FrameRecord]] = {}
    methods = list(cfg.candidate_methods)
    iterable = iter_with_progress(methods, total=len(methods), cfg=progress_cfg, desc="Photometry methods")
    total = len(methods)
    for i, method in enumerate(iterable, start=1):
        corrected = apply_correction_stack(frames, method, parallel_cfg=progress_cfg)
        corrected_by_method[method] = corrected
        all_scores.extend(score_corrected_stack(corrected, method, cfg, progress_cfg=progress_cfg, desc=f"Score {method}"))
        if progress_callback is not None:
            progress_callback(i, total, message=f'method={method}')
    winner = choose_best_correction(all_scores)
    return winner, all_scores, corrected_by_method[winner]
