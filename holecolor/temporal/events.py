from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class TimeSeriesEvent:
    entity_id: str
    onset_frame: int | None
    peak_frame: int | None
    halfmax_frame: int | None
    peak_value: float
    duration_frames: int | None
    baseline_value: float | None


def smooth_values(values: np.ndarray, window: int = 3) -> np.ndarray:
    vals = np.asarray(values, dtype=float)
    if vals.size < 3 or window <= 1:
        return vals
    pad = window // 2
    padded = np.pad(vals, (pad, pad), mode='edge')
    ker = np.ones(window, dtype=float) / float(window)
    return np.convolve(padded, ker, mode='valid')


def detect_onset(values: np.ndarray, z_thresh: float = 2.0, sustain: int = 2) -> int | None:
    vals = smooth_values(values, window=3)
    if vals.size == 0 or not np.any(np.isfinite(vals)):
        return None
    base = vals[: max(3, len(vals) // 5)]
    finite_base = base[np.isfinite(base)]
    if finite_base.size == 0:
        return None
    mu = float(np.median(finite_base))
    mad = float(np.median(np.abs(finite_base - mu)))
    denom = max(1.4826 * mad, 1e-6)
    above = np.isfinite(vals) & (vals > mu + z_thresh * denom)
    run = 0
    for i, ok in enumerate(above):
        run = run + 1 if ok else 0
        if run >= sustain:
            return i - sustain + 1
    return None


def detect_peak(values: np.ndarray) -> int | None:
    vals = smooth_values(values, window=3)
    if vals.size == 0 or not np.any(np.isfinite(vals)):
        return None
    finite_vals = np.where(np.isfinite(vals), vals, -np.inf)
    peak = int(np.argmax(finite_vals))
    return None if not np.isfinite(finite_vals[peak]) else peak


def summarize_curve_events(entity_id: str, values: np.ndarray) -> TimeSeriesEvent:
    vals = np.asarray(values, dtype=float)
    sm = smooth_values(vals, window=3)
    onset = detect_onset(sm)
    peak = detect_peak(sm)
    base = sm[: max(3, len(sm) // 5)] if sm.size else np.array([], dtype=float)
    baseline = float(np.median(base[np.isfinite(base)])) if np.any(np.isfinite(base)) else None
    peak_val = float(sm[peak]) if peak is not None and np.isfinite(sm[peak]) else float('nan')
    halfmax = None
    duration = None
    if peak is not None and np.isfinite(peak_val):
        hm = peak_val / 2.0 if baseline is None else baseline + 0.5 * (peak_val - baseline)
        idx = np.where(np.isfinite(sm) & (sm >= hm))[0]
        halfmax = int(idx[0]) if idx.size else None
        if onset is not None:
            post = np.where(np.isfinite(sm[onset:]) & (sm[onset:] >= hm))[0]
            duration = int(post[-1] - post[0] + 1) if post.size else None
    return TimeSeriesEvent(entity_id, onset, peak, halfmax, peak_val, duration, baseline)
