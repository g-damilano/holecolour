from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from holecolor.config.schema import ParallelConfig
from holecolor.core.parallel import parallel_map, prefer_thread_for_image_tasks
from holecolor.core.types import AuditRecord, FrameRecord


@dataclass(slots=True)
class Transform2D:
    dx: float
    dy: float
    angle_deg: float = 0.0
    scale: float = 1.0


def _to_gray(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32)


def select_reference_frame(frames: list[FrameRecord], audit_records: list[AuditRecord], mode: str = "best_qc") -> int:
    if not frames:
        raise ValueError("frames cannot be empty")
    if mode == "first":
        return 0
    accepted = [r for r in audit_records if r.accepted]
    if not accepted:
        return 0
    best = max(accepted, key=lambda r: r.blur_score)
    return int(best.frame_id)


def estimate_rigid_shift(moving: np.ndarray, fixed: np.ndarray, max_shift_px: int | None = None) -> Transform2D:
    mg = _to_gray(moving)
    fg = _to_gray(fixed)
    (dx, dy), _ = cv2.phaseCorrelate(fg, mg)
    if max_shift_px is not None:
        dx = float(np.clip(dx, -max_shift_px, max_shift_px))
        dy = float(np.clip(dy, -max_shift_px, max_shift_px))
    return Transform2D(dx=-dx, dy=-dy)


def apply_transform(image: np.ndarray, tfm: Transform2D) -> np.ndarray:
    m = np.array([[1.0, 0.0, tfm.dx], [0.0, 1.0, tfm.dy]], dtype=np.float32)
    return cv2.warpAffine(
        image,
        m,
        (image.shape[1], image.shape[0]),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT,
    )


def _stabilize_frame_task(task: tuple[FrameRecord, np.ndarray, int | None]) -> tuple[FrameRecord, Transform2D]:
    frame, ref, max_shift_px = task
    tfm = estimate_rigid_shift(frame.image, ref, max_shift_px=max_shift_px)
    return FrameRecord(frame.frame_id, frame.time_s, apply_transform(frame.image, tfm)), tfm


def stabilize_sequence(
    frames: list[FrameRecord],
    reference_idx: int = 0,
    max_shift_px: int | None = None,
    progress_cfg: ParallelConfig | None = None,
    progress_callback=None,
):
    ref = frames[reference_idx].image
    task_cfg = prefer_thread_for_image_tasks(progress_cfg)
    results = parallel_map(
        _stabilize_frame_task,
        [(frame, ref, max_shift_px) for frame in frames],
        task_cfg,
        desc="Registration frames",
        progress_callback=progress_callback,
    )
    out = [row[0] for row in results]
    tfms = [row[1] for row in results]
    return out, tfms


def residual_difference(a: np.ndarray, b: np.ndarray) -> float:
    aa = _to_gray(a)
    bb = _to_gray(b)
    return float(np.mean(np.abs(aa - bb)))
