from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import cv2
import imageio.v3 as iio
import numpy as np

from holecolor.config.schema import ParallelConfig
from holecolor.core.parallel import iter_with_progress, parallel_map
from holecolor.core.types import FrameRecord, VideoMeta


@dataclass(slots=True)
class BlackBandCrop:
    applied: bool
    top: int
    bottom: int
    original_height: int
    original_width: int
    cropped_height: int
    cropped_width: int
    sample_count: int
    sample_frame_ids: tuple[int, ...]
    top_detected_fraction: float
    bottom_detected_fraction: float
    top_band_counts: tuple[int, ...]
    bottom_band_counts: tuple[int, ...]
    reason: str
    dark_luma_threshold: float = 8.0
    row_mean_threshold: float = 5.0
    row_std_threshold: float = 3.5
    dark_fraction_threshold: float = 0.995
    min_band_px: int = 2
    min_detect_fraction: float = 0.8
    crop_count_percentile: float = 10.0
    min_content_fraction: float = 0.25

    def to_jsonable(self) -> dict[str, object]:
        payload = asdict(self)
        payload["crop_window"] = {
            "x_start": 0,
            "x_stop_exclusive": int(self.original_width),
            "y_start": int(self.top if self.applied else 0),
            "y_stop_exclusive": int(self.original_height - self.bottom if self.applied else self.original_height),
        }
        return payload


def ensure_rgb_uint8(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image)
    if arr.ndim == 2:
        arr = np.repeat(arr[..., None], 3, axis=2)
    elif arr.ndim == 3 and arr.shape[2] == 1:
        arr = np.repeat(arr, 3, axis=2)
    elif arr.ndim == 3 and arr.shape[2] > 3:
        arr = arr[..., :3]
    if arr.dtype == np.uint8:
        return np.ascontiguousarray(arr)
    if np.issubdtype(arr.dtype, np.unsignedinteger):
        info = np.iinfo(arr.dtype)
        scaled = arr.astype(np.float32) * (255.0 / max(float(info.max), 1.0))
        return np.clip(np.round(scaled), 0, 255).astype(np.uint8)
    if np.issubdtype(arr.dtype, np.signedinteger):
        info = np.iinfo(arr.dtype)
        scaled = (arr.astype(np.float32) - float(info.min)) * (255.0 / max(float(info.max - info.min), 1.0))
        return np.clip(np.round(scaled), 0, 255).astype(np.uint8)
    data = arr.astype(np.float32, copy=False)
    finite = np.isfinite(data)
    if not np.any(finite):
        return np.zeros(arr.shape, dtype=np.uint8)
    valid = data[finite]
    if float(valid.min()) >= 0.0 and float(valid.max()) <= 1.0:
        data = data * 255.0
    return np.clip(np.round(np.nan_to_num(data, nan=0.0, posinf=255.0, neginf=0.0)), 0, 255).astype(np.uint8)


def _empty_black_band_crop(frames: Sequence[FrameRecord], reason: str) -> BlackBandCrop:
    if frames:
        h, w = frames[0].image.shape[:2]
    else:
        h, w = 0, 0
    return BlackBandCrop(
        applied=False,
        top=0,
        bottom=0,
        original_height=int(h),
        original_width=int(w),
        cropped_height=int(h),
        cropped_width=int(w),
        sample_count=0,
        sample_frame_ids=(),
        top_detected_fraction=0.0,
        bottom_detected_fraction=0.0,
        top_band_counts=(),
        bottom_band_counts=(),
        reason=reason,
    )


def detect_frame_black_bands(
    image: np.ndarray,
    *,
    dark_luma_threshold: float = 8.0,
    row_mean_threshold: float = 5.0,
    row_std_threshold: float = 3.5,
    dark_fraction_threshold: float = 0.995,
) -> tuple[int, int]:
    """Return contiguous top and bottom black-band row counts for one frame."""
    arr = ensure_rgb_uint8(image)
    if arr.ndim == 2:
        gray = arr.astype(np.float32, copy=False)
    else:
        rgb = arr[..., :3].astype(np.float32, copy=False)
        gray = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
    dark_fraction = np.mean(gray <= float(dark_luma_threshold), axis=1)
    row_mean = np.mean(gray, axis=1)
    row_std = np.std(gray, axis=1)
    black_rows = (
        (dark_fraction >= float(dark_fraction_threshold))
        & (row_mean <= float(row_mean_threshold))
        & (row_std <= float(row_std_threshold))
    )
    top = 0
    for is_black in black_rows:
        if not bool(is_black):
            break
        top += 1
    bottom = 0
    for is_black in black_rows[::-1]:
        if not bool(is_black):
            break
        bottom += 1
    return int(top), int(bottom)


def _sample_indices(n_items: int, sample_limit: int) -> list[int]:
    n = int(n_items)
    limit = max(1, int(sample_limit))
    if n <= 0:
        return []
    if n <= limit:
        return list(range(n))
    return sorted({int(round(v)) for v in np.linspace(0, n - 1, limit)})


def detect_black_band_crop(
    frames: Sequence[FrameRecord],
    *,
    dark_luma_threshold: float = 8.0,
    row_mean_threshold: float = 5.0,
    row_std_threshold: float = 3.5,
    dark_fraction_threshold: float = 0.995,
    min_band_px: int = 2,
    min_detect_fraction: float = 0.8,
    crop_count_percentile: float = 10.0,
    min_content_fraction: float = 0.25,
    sample_limit: int = 64,
) -> BlackBandCrop:
    if not frames:
        return _empty_black_band_crop(frames, "no_frames")
    first_h, first_w = frames[0].image.shape[:2]
    if any(frame.image.shape[:2] != (first_h, first_w) for frame in frames):
        return _empty_black_band_crop(frames, "inconsistent_frame_shapes")
    sample_ids = _sample_indices(len(frames), sample_limit)
    if not sample_ids:
        return _empty_black_band_crop(frames, "no_sampled_frames")
    top_counts: list[int] = []
    bottom_counts: list[int] = []
    frame_ids: list[int] = []
    for idx in sample_ids:
        frame = frames[int(idx)]
        top, bottom = detect_frame_black_bands(
            frame.image,
            dark_luma_threshold=dark_luma_threshold,
            row_mean_threshold=row_mean_threshold,
            row_std_threshold=row_std_threshold,
            dark_fraction_threshold=dark_fraction_threshold,
        )
        top_counts.append(int(top))
        bottom_counts.append(int(bottom))
        frame_ids.append(int(frame.frame_id))
    top_arr = np.asarray(top_counts, dtype=np.int32)
    bottom_arr = np.asarray(bottom_counts, dtype=np.int32)
    min_band = max(1, int(min_band_px))
    sample_count = int(len(sample_ids))
    top_pos = top_arr[top_arr >= min_band]
    bottom_pos = bottom_arr[bottom_arr >= min_band]
    top_fraction = float(top_pos.size) / max(float(sample_count), 1.0)
    bottom_fraction = float(bottom_pos.size) / max(float(sample_count), 1.0)

    def crop_count(values: np.ndarray, detected_fraction: float) -> int:
        if values.size == 0 or detected_fraction < float(min_detect_fraction):
            return 0
        count = int(np.floor(np.percentile(values, float(crop_count_percentile))))
        return count if count >= min_band else 0

    top_crop = crop_count(top_pos, top_fraction)
    bottom_crop = crop_count(bottom_pos, bottom_fraction)
    cropped_height = int(first_h) - int(top_crop) - int(bottom_crop)
    cropped_width = int(first_w)
    applied = bool(top_crop > 0 or bottom_crop > 0)
    reason = "detected_consistent_black_bands" if applied else "no_consistent_black_bands"
    if applied and cropped_height <= 0:
        top_crop = 0
        bottom_crop = 0
        cropped_height = int(first_h)
        applied = False
        reason = "unsafe_crop_empty_content"
    if applied and cropped_height < max(1, int(round(float(first_h) * float(min_content_fraction)))):
        top_crop = 0
        bottom_crop = 0
        cropped_height = int(first_h)
        applied = False
        reason = "unsafe_crop_too_little_content"

    return BlackBandCrop(
        applied=bool(applied),
        top=int(top_crop),
        bottom=int(bottom_crop),
        original_height=int(first_h),
        original_width=int(first_w),
        cropped_height=int(cropped_height),
        cropped_width=int(cropped_width),
        sample_count=int(sample_count),
        sample_frame_ids=tuple(frame_ids),
        top_detected_fraction=float(top_fraction),
        bottom_detected_fraction=float(bottom_fraction),
        top_band_counts=tuple(int(v) for v in top_counts),
        bottom_band_counts=tuple(int(v) for v in bottom_counts),
        reason=reason,
        dark_luma_threshold=float(dark_luma_threshold),
        row_mean_threshold=float(row_mean_threshold),
        row_std_threshold=float(row_std_threshold),
        dark_fraction_threshold=float(dark_fraction_threshold),
        min_band_px=int(min_band_px),
        min_detect_fraction=float(min_detect_fraction),
        crop_count_percentile=float(crop_count_percentile),
        min_content_fraction=float(min_content_fraction),
    )


def strip_black_bands_from_frames(frames: list[FrameRecord], *, sample_limit: int = 64) -> tuple[list[FrameRecord], BlackBandCrop]:
    crop = detect_black_band_crop(frames, sample_limit=sample_limit)
    if not crop.applied:
        return frames, crop
    y0 = int(crop.top)
    y1 = int(crop.original_height - crop.bottom)
    for frame in frames:
        frame.image = np.array(frame.image[y0:y1, ...], copy=True, order="C")
    return frames, crop


def probe_video(path: Path) -> VideoMeta:
    if path.is_dir():
        files = sorted(p for p in path.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"})
        if not files:
            raise FileNotFoundError(f"no images found in directory: {path}")
        first = np.asarray(iio.imread(files[0]))
        h, w = first.shape[:2]
        channels = 1 if first.ndim == 2 else first.shape[2]
        return VideoMeta(path, fps=1.0, n_frames=len(files), width=w, height=h, channels=channels)
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(path)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0) or 1.0
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    cap.release()
    return VideoMeta(path, fps=fps, n_frames=n_frames, width=width, height=height, channels=3)


def _normalize_frame_window(n_frames: int, start_frame: int | None, end_frame: int | None) -> tuple[int, int | None]:
    start = max(0, int(start_frame or 0))
    if end_frame is None:
        end = int(n_frames) - 1 if int(n_frames) > 0 else None
    else:
        end = max(start, int(end_frame))
        if int(n_frames) > 0:
            end = min(end, int(n_frames) - 1)
    return start, end


def iter_video_frames(
    path: Path,
    every_n: int = 1,
    show_progress: bool = False,
    start_frame: int | None = None,
    end_frame: int | None = None,
) -> list[FrameRecord]:
    frames: list[FrameRecord] = []
    meta = probe_video(path)
    start, end = _normalize_frame_window(meta.n_frames, start_frame, end_frame)
    every_n = max(1, int(every_n))
    if path.is_dir():
        files = sorted(p for p in path.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"})
        end_idx = len(files) - 1 if end is None else min(int(end), len(files) - 1)
        selected = [(i, str(files[i]), float(meta.fps)) for i in range(start, end_idx + 1) if (i - start) % every_n == 0]
        if len(selected) >= 8:
            load_cfg = ParallelConfig(enabled=True, backend="thread", max_workers=0, min_parallel_tasks=1, show_progress=show_progress, progress_leave=False)
            return parallel_map(_load_image_frame_task, selected, load_cfg, desc="Loading frames")
        progress_cfg = ParallelConfig(show_progress=show_progress, enabled=False)
        selected_files = [(i, files[i]) for i in range(start, end_idx + 1) if (i - start) % every_n == 0]
        for i, file in iter_with_progress(selected_files, total=len(selected_files), cfg=progress_cfg, desc="Loading frames"):
            if (i - start) % every_n != 0:
                continue
            arr = np.asarray(iio.imread(file))
            frames.append(FrameRecord(frame_id=i, time_s=i / max(meta.fps, 1e-6), image=ensure_rgb_uint8(arr)))
        return frames
    progress_cfg = ParallelConfig(show_progress=show_progress, enabled=False)
    total = meta.n_frames if meta.n_frames > 0 else None
    cv2_frames = _load_video_frames_cv2(path, meta, every_n, progress_cfg, start_frame=start, end_frame=end)
    if cv2_frames:
        return cv2_frames
    try:
        iterable = iio.imiter(path)
        for i, frame in enumerate(iter_with_progress(iterable, total=total, cfg=progress_cfg, desc="Loading frames")):
            if i < start:
                continue
            if end is not None and i > end:
                break
            if (i - start) % every_n != 0:
                continue
            arr = np.asarray(frame)
            frames.append(FrameRecord(frame_id=i, time_s=i / max(meta.fps, 1e-6), image=ensure_rgb_uint8(arr)))
        if frames:
            return frames
        return _load_video_frames_cv2(path, meta, every_n, progress_cfg, start_frame=start, end_frame=end)
    except Exception:
        return _load_video_frames_cv2(path, meta, every_n, progress_cfg, start_frame=start, end_frame=end)




def _load_image_frame_task(task: tuple[int, str, float]) -> FrameRecord:
    i, file_str, fps = task
    arr = np.asarray(iio.imread(Path(file_str)))
    return FrameRecord(frame_id=int(i), time_s=float(i) / max(float(fps), 1e-6), image=ensure_rgb_uint8(arr))


def _load_video_frames_cv2(
    path: Path,
    meta: VideoMeta,
    every_n: int,
    progress_cfg: ParallelConfig,
    start_frame: int = 0,
    end_frame: int | None = None,
) -> list[FrameRecord]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(path)
    frames: list[FrameRecord] = []
    total = int(meta.n_frames)
    start = max(0, int(start_frame))
    end = (total - 1 if total > 0 else None) if end_frame is None else int(end_frame)
    every_n = max(1, int(every_n))
    if total > 0:
        end = min(int(end), total - 1)
        if start > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, float(start))
        span = max(0, int(end) - start + 1)
        iterator = iter_with_progress(range(start, int(end) + 1), total=span, cfg=progress_cfg, desc="Loading frames")
        for i in iterator:
            ok, frame = cap.read()
            if not ok:
                break
            if (i - start) % every_n != 0:
                continue
            arr = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(FrameRecord(frame_id=int(i), time_s=float(i) / max(float(meta.fps), 1e-6), image=ensure_rgb_uint8(arr)))
    else:
        i = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if i < start:
                i += 1
                continue
            if end is not None and i > end:
                break
            if (i - start) % every_n == 0:
                arr = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(FrameRecord(frame_id=int(i), time_s=float(i) / max(float(meta.fps), 1e-6), image=ensure_rgb_uint8(arr)))
            i += 1
    cap.release()
    return frames


def save_frame(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    iio.imwrite(path, image)
