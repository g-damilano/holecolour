from __future__ import annotations

import cv2

from holecolor.audit.blur import classify_blur, laplacian_variance_score, wavelet_blur_score
from holecolor.audit.frame_change import histogram_jump_score
from holecolor.audit.saturation import saturation_fraction
from holecolor.config.schema import AuditConfig, ParallelConfig
from holecolor.core.parallel import parallel_map, prefer_thread_for_image_tasks
from holecolor.core.types import AuditRecord, FrameRecord


def _compute_audit_without_jump(frame: FrameRecord, cfg: AuditConfig) -> AuditRecord:
    gray = cv2.cvtColor(frame.image, cv2.COLOR_RGB2GRAY)
    blur_score = wavelet_blur_score(gray) if cfg.blur_method == "wavelet" else laplacian_variance_score(gray)
    sat_r = saturation_fraction(frame.image[..., 0], cfg.saturation_threshold)
    sat_g = saturation_fraction(frame.image[..., 1], cfg.saturation_threshold)
    sat_b = saturation_fraction(frame.image[..., 2], cfg.saturation_threshold)
    accepted = classify_blur(blur_score, cfg.blur_threshold) and max(sat_r, sat_g, sat_b) < 0.05
    return AuditRecord(frame.frame_id, blur_score, sat_r, sat_g, sat_b, 0.0, accepted)


def _compute_audit_without_jump_task(task: tuple[FrameRecord, AuditConfig]) -> AuditRecord:
    frame, cfg = task
    return _compute_audit_without_jump(frame, cfg)


def compute_audit_record(frame: FrameRecord, prev_frame: FrameRecord | None, cfg: AuditConfig) -> AuditRecord:
    base = _compute_audit_without_jump(frame, cfg)
    jump = 0.0 if prev_frame is None else histogram_jump_score(prev_frame.image, frame.image, bins=cfg.frame_jump_hist_bins)
    return AuditRecord(base.frame_id, base.blur_score, base.sat_frac_r, base.sat_frac_g, base.sat_frac_b, jump, base.accepted)


def audit_sequence(
    frames: list[FrameRecord],
    cfg: AuditConfig,
    progress_cfg: ParallelConfig | None = None,
    progress_callback=None,
) -> list[AuditRecord]:
    if not frames:
        return []
    task_cfg = prefer_thread_for_image_tasks(progress_cfg)
    tasks = [(frame, cfg) for frame in frames]
    base_records = parallel_map(
        _compute_audit_without_jump_task,
        tasks,
        task_cfg,
        desc="Audit frames",
        progress_callback=progress_callback,
    )
    out: list[AuditRecord] = []
    for i, base in enumerate(base_records):
        jump = 0.0 if i == 0 else histogram_jump_score(frames[i - 1].image, frames[i].image, bins=cfg.frame_jump_hist_bins)
        out.append(AuditRecord(base.frame_id, base.blur_score, base.sat_frac_r, base.sat_frac_g, base.sat_frac_b, jump, base.accepted))
    return out
