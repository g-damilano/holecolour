from __future__ import annotations

import cv2
import numpy as np

from holecolor.config.schema import ParallelConfig
from holecolor.core.parallel import parallel_map, prefer_thread_for_image_tasks
from holecolor.core.types import FrameRecord


def no_correction(image: np.ndarray) -> np.ndarray:
    return image.copy()


def frame_gain_normalize(image: np.ndarray) -> np.ndarray:
    arr = image.astype(np.float32)
    gain = arr.mean(axis=(0, 1), keepdims=True)
    target = gain.mean()
    out = arr * (target / np.maximum(gain, 1e-6))
    return np.clip(out, 0, 255).astype(np.uint8)


def polynomial_flatfield(image: np.ndarray, degree: int = 2) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32)
    bg = cv2.GaussianBlur(gray, (0, 0), sigmaX=32)
    bg = np.maximum(bg, 1.0)
    out = image.astype(np.float32) / bg[..., None] * bg.mean()
    return np.clip(out, 0, 255).astype(np.uint8)


def apply_correction(image: np.ndarray, method: str) -> np.ndarray:
    if method == "none":
        return no_correction(image)
    if method == "gain_norm":
        return frame_gain_normalize(image)
    if method == "flatfield_poly":
        return polynomial_flatfield(image)
    raise ValueError(f"unknown correction method: {method}")


def _apply_correction_task(task: tuple[FrameRecord, str]) -> FrameRecord:
    frame, method = task
    return FrameRecord(frame.frame_id, frame.time_s, apply_correction(frame.image, method))


def apply_correction_stack(frames: list[FrameRecord], method: str, parallel_cfg: ParallelConfig | None = None) -> list[FrameRecord]:
    if not frames:
        return []
    task_cfg = prefer_thread_for_image_tasks(parallel_cfg)
    return parallel_map(
        _apply_correction_task,
        [(frame, method) for frame in frames],
        task_cfg,
        desc=f"Apply {method}",
    )
