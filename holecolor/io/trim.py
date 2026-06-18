from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from pathlib import Path

import cv2
import imageio.v3 as iio
import numpy as np

from holecolor.core.types import VideoMeta
from holecolor.io.video import ensure_rgb_uint8, probe_video


@dataclass(slots=True)
class FrameTrimSelection:
    start_frame: int
    end_frame: int
    total_frames: int
    fps: float
    source_path: str
    selected_frame_count: int

    def to_jsonable(self) -> dict[str, object]:
        return asdict(self)


@dataclass(slots=True)
class HistogramStabilizationSuggestion:
    cursor_frame: int
    first_stable_frame: int
    confidence: float
    jump_score: float
    post_stability_score: float
    post_constancy_score: float
    stable_run_frames: int
    stable_window_frames: int
    reason: str

    def to_jsonable(self) -> dict[str, object]:
        return asdict(self)


def _image_files(path: Path) -> list[Path]:
    return sorted(p for p in path.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"})


def _read_preview_frame(path: Path, meta: VideoMeta, frame_index: int, cap: cv2.VideoCapture | None) -> np.ndarray:
    idx = int(np.clip(frame_index, 0, max(0, int(meta.n_frames) - 1)))
    if path.is_dir():
        files = _image_files(path)
        if not files:
            raise FileNotFoundError(f"no images found in directory: {path}")
        return ensure_rgb_uint8(np.asarray(iio.imread(files[idx])))
    if cap is None or not cap.isOpened():
        raise RuntimeError(f"could not open video for trim preview: {path}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, float(idx))
    ok, frame = cap.read()
    if not ok:
        raise RuntimeError(f"could not read preview frame {idx} from {path}")
    return ensure_rgb_uint8(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))


def _fit_preview(image: np.ndarray, width: int, height: int) -> np.ndarray:
    arr = ensure_rgb_uint8(image)
    h, w = arr.shape[:2]
    scale = min(float(width) / max(float(w), 1.0), float(height) / max(float(h), 1.0))
    out_w = max(1, int(round(float(w) * scale)))
    out_h = max(1, int(round(float(h) * scale)))
    resized = cv2.resize(arr, (out_w, out_h), interpolation=cv2.INTER_AREA)
    canvas = np.full((height, width, 3), 24, dtype=np.uint8)
    y0 = (height - out_h) // 2
    x0 = (width - out_w) // 2
    canvas[y0:y0 + out_h, x0:x0 + out_w] = resized
    return canvas


def _put_text(img: np.ndarray, text: str, x: int, y: int, scale: float = 0.58, color: tuple[int, int, int] = (245, 245, 245)) -> None:
    cv2.putText(img, text, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, float(scale), (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, text, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, float(scale), color, 1, cv2.LINE_AA)


def _diagnostic_gray(image: np.ndarray, max_dim: int = 360) -> np.ndarray:
    arr = ensure_rgb_uint8(image)
    if arr.ndim == 3:
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    else:
        gray = arr
    h, w = gray.shape[:2]
    scale = min(1.0, float(max_dim) / max(float(h), float(w), 1.0))
    if scale < 1.0:
        gray = cv2.resize(gray, (max(1, int(round(w * scale))), max(1, int(round(h * scale)))), interpolation=cv2.INTER_AREA)
    return np.ascontiguousarray(gray, dtype=np.uint8)


def _gray_histogram_and_cdf(gray: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).reshape(-1).astype(np.float64)
    total = float(hist.sum())
    if total <= 0.0:
        hist[:] = 0.0
    else:
        hist /= total
    cdf = np.cumsum(hist)
    return hist, cdf, float(np.mean(gray) / 255.0)


def _read_diagnostic_frame(path: Path, meta: VideoMeta, frame_index: int, cap: cv2.VideoCapture | None) -> np.ndarray:
    return _diagnostic_gray(_read_preview_frame(path, meta, frame_index, cap))


def _histogram_diagnostics(path: Path, meta: VideoMeta) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_frames = int(meta.n_frames)
    if n_frames <= 1:
        return np.zeros((0, 256), dtype=np.float64), np.zeros((0, 256), dtype=np.float64), np.zeros(0, dtype=np.float64)
    cap = None if path.is_dir() else cv2.VideoCapture(str(path))
    hists: list[np.ndarray] = []
    cdfs: list[np.ndarray] = []
    means: list[float] = []
    try:
        for frame_idx in range(n_frames):
            gray = _read_diagnostic_frame(path, meta, frame_idx, cap)
            hist, cdf, mean = _gray_histogram_and_cdf(gray)
            hists.append(hist)
            cdfs.append(cdf)
            means.append(mean)
    finally:
        if cap is not None:
            cap.release()
    return np.vstack(hists), np.vstack(cdfs), np.asarray(means, dtype=np.float64)


def _transition_scores(hists: np.ndarray, cdfs: np.ndarray, means: np.ndarray) -> np.ndarray:
    if hists.shape[0] < 2:
        return np.zeros(0, dtype=np.float64)
    hist_change = 0.5 * np.sum(np.abs(np.diff(hists, axis=0)), axis=1)
    cdf_change = np.mean(np.abs(np.diff(cdfs, axis=0)), axis=1)
    mean_change = np.abs(np.diff(means))
    return (hist_change + 2.0 * cdf_change + 0.5 * mean_change).astype(np.float64)


def _profile_constancy_scores(
    hists: np.ndarray,
    cdfs: np.ndarray,
    means: np.ndarray,
    reference_slice: slice,
    frame_indices: np.ndarray,
) -> np.ndarray:
    ref_hists = hists[reference_slice]
    ref_cdfs = cdfs[reference_slice]
    ref_means = means[reference_slice]
    if ref_hists.size == 0 or frame_indices.size == 0:
        return np.zeros(0, dtype=np.float64)
    ref_hist = np.median(ref_hists, axis=0)
    ref_cdf = np.median(ref_cdfs, axis=0)
    ref_mean = float(np.median(ref_means))
    frame_hists = hists[frame_indices]
    frame_cdfs = cdfs[frame_indices]
    frame_means = means[frame_indices]
    hist_dist = 0.5 * np.sum(np.abs(frame_hists - ref_hist[None, :]), axis=1)
    cdf_dist = np.mean(np.abs(frame_cdfs - ref_cdf[None, :]), axis=1)
    mean_dist = np.abs(frame_means - ref_mean)
    return (hist_dist + 2.0 * cdf_dist + 0.5 * mean_dist).astype(np.float64)


def suggest_histogram_stabilization_trim(
    path: Path,
    *,
    meta: VideoMeta | None = None,
    stable_window_frames: int | None = None,
) -> HistogramStabilizationSuggestion | None:
    meta = meta or probe_video(path)
    n_frames = int(meta.n_frames)
    if n_frames < 6:
        return None
    stable_window = int(stable_window_frames or max(5, min(18, round(0.06 * n_frames))))
    stable_window = max(3, min(stable_window, max(3, n_frames // 2)))
    hists, cdfs, means = _histogram_diagnostics(path, meta)
    scores = _transition_scores(hists, cdfs, means)
    if scores.size < stable_window + 1:
        return None
    median_score = float(np.median(scores))
    mad = float(np.median(np.abs(scores - median_score)))
    stable_change_threshold = median_score + 3.0 * max(mad, 1e-6)
    best: tuple[float, int, float, float, float, int] | None = None
    for first_stable in range(1, n_frames - stable_window + 1):
        jump = float(scores[first_stable - 1])
        post = scores[first_stable:min(scores.size, first_stable + stable_window - 1)]
        if post.size < max(2, stable_window - 1):
            continue
        post_p90 = float(np.percentile(post, 90))
        post_median = float(np.median(post))
        reference_slice = slice(first_stable, first_stable + stable_window)
        window_indices = np.arange(first_stable, first_stable + stable_window, dtype=np.int64)
        constancy = _profile_constancy_scores(hists, cdfs, means, reference_slice, window_indices)
        if constancy.size == 0:
            continue
        constancy_p90 = float(np.percentile(constancy, 90))
        stability_threshold = max(stable_change_threshold, 0.25 * jump)
        constancy_threshold = max(stable_change_threshold, 0.20 * jump)
        jump_is_large = jump >= max(median_score + 6.0 * max(mad, 1e-6), 3.0 * max(post_p90, constancy_p90, 1e-6))
        post_is_stable = post_p90 <= stability_threshold
        post_is_constant = constancy_p90 <= constancy_threshold
        if not jump_is_large or not post_is_stable or not post_is_constant:
            continue
        stable_run = 1
        for frame_idx in range(first_stable + 1, n_frames):
            transition_score = float(scores[frame_idx - 1])
            profile_score = float(_profile_constancy_scores(hists, cdfs, means, reference_slice, np.asarray([frame_idx], dtype=np.int64))[0])
            if transition_score <= stability_threshold and profile_score <= constancy_threshold:
                stable_run += 1
            else:
                break
        tail_bonus = stable_run / max(n_frames - first_stable, 1)
        contrast = jump / max(post_p90, post_median, constancy_p90, 1e-6)
        merit = float(jump * contrast * (1.0 + tail_bonus) * math.sqrt(float(stable_run)))
        if best is None or merit > best[0]:
            best = (merit, int(first_stable), jump, post_p90, constancy_p90, int(stable_run))
    if best is None:
        return None
    _merit, first_stable, jump, post_p90, constancy_p90, stable_run = best
    confidence = float(np.clip((jump / max(post_p90, constancy_p90, stable_change_threshold, 1e-6)) / 8.0, 0.0, 1.0))
    return HistogramStabilizationSuggestion(
        cursor_frame=int(max(0, first_stable - 1)),
        first_stable_frame=int(first_stable),
        confidence=confidence,
        jump_score=float(jump),
        post_stability_score=float(post_p90),
        post_constancy_score=float(constancy_p90),
        stable_run_frames=int(stable_run),
        stable_window_frames=int(stable_window),
        reason="last frame before a large histogram/CDF discontinuity followed by a sustained constant post-event histogram/CDF regime",
    )


def _compose_trim_canvas(
    start_img: np.ndarray,
    preview_img: np.ndarray,
    end_img: np.ndarray,
    start_frame: int,
    preview_frame: int,
    end_frame: int,
    meta: VideoMeta,
    suggestion: HistogramStabilizationSuggestion | None = None,
) -> np.ndarray:
    panel_w = 420
    panel_h = 360
    top_h = 96
    bottom_h = 54
    gap = 8
    width = 3 * panel_w + 4 * gap
    height = top_h + panel_h + bottom_h
    canvas = np.full((height, width, 3), 32, dtype=np.uint8)
    duration_s = max(0.0, float(end_frame - start_frame) / max(float(meta.fps), 1e-6))
    _put_text(canvas, "Select processing frame range", 16, 30, scale=0.75)
    _put_text(
        canvas,
        f"range: {start_frame} to {end_frame} | selected frames: {end_frame - start_frame + 1} | duration: {duration_s:.2f}s",
        16,
        62,
        scale=0.55,
    )
    if suggestion is None:
        _put_text(canvas, "Keys: s=set start, e=set end, arrows/+- move preview, Enter/Space accept, Esc/q cancel", 16, 88, scale=0.50)
    else:
        _put_text(
            canvas,
            f"Auto suggestion: end at {suggestion.cursor_frame}; stable regime starts at {suggestion.first_stable_frame}; confidence {suggestion.confidence:.2f}",
            16,
            88,
            scale=0.50,
            color=(175, 225, 255),
        )
    panels = (
        ("START", start_img, start_frame),
        ("PREVIEW", preview_img, preview_frame),
        ("END", end_img, end_frame),
    )
    for idx, (label, img, frame_idx) in enumerate(panels):
        x0 = gap + idx * (panel_w + gap)
        y0 = top_h
        panel = _fit_preview(img, panel_w, panel_h)
        canvas[y0:y0 + panel_h, x0:x0 + panel_w] = panel
        _put_text(canvas, f"{label} frame {frame_idx}", x0 + 12, y0 + 28, scale=0.58)
        _put_text(canvas, f"t={float(frame_idx) / max(float(meta.fps), 1e-6):.3f}s", x0 + 12, y0 + 54, scale=0.50)
    _put_text(canvas, "Use the sliders for coarse/fine positioning. The saved JSON lets you replay the same trim without opening the UI.", 16, height - 18, scale=0.50)
    return cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)


def select_frame_range_visual(
    path: Path,
    *,
    initial_start: int = 0,
    initial_end: int | None = None,
    meta: VideoMeta | None = None,
    suggestion: HistogramStabilizationSuggestion | None = None,
) -> FrameTrimSelection:
    meta = meta or probe_video(path)
    n_frames = int(meta.n_frames)
    if n_frames <= 0:
        raise RuntimeError("visual trim requires a known positive frame count")
    start = int(np.clip(initial_start, 0, n_frames - 1))
    suggested_end = None if suggestion is None else int(suggestion.cursor_frame)
    end_seed = initial_end if initial_end is not None else (suggested_end if suggested_end is not None else n_frames - 1)
    end = int(np.clip(end_seed, start, n_frames - 1))
    preview_seed = suggested_end if suggested_end is not None else (start + end) // 2
    preview = int(np.clip(preview_seed, start, end))
    cap = None if path.is_dir() else cv2.VideoCapture(str(path))
    cache: dict[int, np.ndarray] = {}
    window = "holecolor trim selector"

    def get_frame(idx: int) -> np.ndarray:
        idx = int(np.clip(idx, 0, n_frames - 1))
        if idx not in cache:
            if len(cache) > 24:
                cache.clear()
            cache[idx] = _read_preview_frame(path, meta, idx, cap)
        return cache[idx]

    try:
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window, 1320, 540)
        cv2.createTrackbar("start", window, start, n_frames - 1, lambda _v: None)
        cv2.createTrackbar("preview", window, preview, n_frames - 1, lambda _v: None)
        cv2.createTrackbar("end", window, end, n_frames - 1, lambda _v: None)
    except Exception as exc:
        if cap is not None:
            cap.release()
        raise RuntimeError("OpenCV visual trim window could not be created") from exc

    last_state: tuple[int, int, int] | None = None
    try:
        while True:
            slider_start = int(cv2.getTrackbarPos("start", window))
            slider_preview = int(cv2.getTrackbarPos("preview", window))
            slider_end = int(cv2.getTrackbarPos("end", window))
            start = int(np.clip(slider_start, 0, n_frames - 1))
            end = int(np.clip(slider_end, start, n_frames - 1))
            preview = int(np.clip(slider_preview, start, end))
            if (start, preview, end) != (slider_start, slider_preview, slider_end):
                cv2.setTrackbarPos("start", window, start)
                cv2.setTrackbarPos("preview", window, preview)
                cv2.setTrackbarPos("end", window, end)
            state = (start, preview, end)
            if state != last_state:
                canvas = _compose_trim_canvas(get_frame(start), get_frame(preview), get_frame(end), start, preview, end, meta, suggestion=suggestion)
                cv2.imshow(window, canvas)
                last_state = state
            key = cv2.waitKeyEx(60)
            if key < 0:
                continue
            if key in (13, 10, 32):
                break
            if key in (27, ord('q'), ord('Q')):
                raise RuntimeError("frame trim selection cancelled")
            if key in (ord('s'), ord('S')):
                start = min(preview, end)
                cv2.setTrackbarPos("start", window, start)
                continue
            if key in (ord('e'), ord('E')):
                end = max(preview, start)
                cv2.setTrackbarPos("end", window, end)
                continue
            step = 1
            if key in (ord('+'), ord('='), 2555904, ord('l'), ord('L')):
                cv2.setTrackbarPos("preview", window, min(end, preview + step))
            elif key in (ord('-'), ord('_'), 2424832, ord('h'), ord('H')):
                cv2.setTrackbarPos("preview", window, max(start, preview - step))
            elif key in (2228224, ord(']')):
                cv2.setTrackbarPos("preview", window, min(end, preview + max(1, n_frames // 100)))
            elif key in (2162688, ord('[')):
                cv2.setTrackbarPos("preview", window, max(start, preview - max(1, n_frames // 100)))
    finally:
        if cap is not None:
            cap.release()
        try:
            cv2.destroyWindow(window)
        except Exception:
            pass

    return FrameTrimSelection(
        start_frame=int(start),
        end_frame=int(end),
        total_frames=int(n_frames),
        fps=float(meta.fps),
        source_path=str(path),
        selected_frame_count=int(end - start + 1),
    )
