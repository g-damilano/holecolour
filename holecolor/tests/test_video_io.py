import numpy as np
import imageio.v3 as iio

from holecolor.io.trim import suggest_histogram_stabilization_trim
from holecolor.io.video import detect_black_band_crop, ensure_rgb_uint8, iter_video_frames, strip_black_bands_from_frames
from holecolor.core.types import FrameRecord


def test_ensure_rgb_uint8_scales_high_bit_depth_integer_frames():
    frame = np.zeros((2, 2, 3), dtype=np.uint16)
    frame[0, 0] = [0, 32768, 65535]
    out = ensure_rgb_uint8(frame)

    assert out.dtype == np.uint8
    assert tuple(out[0, 0]) == (0, 128, 255)


def test_ensure_rgb_uint8_scales_unit_float_frames():
    frame = np.array([[[0.0, 0.5, 1.0]]], dtype=np.float32)
    out = ensure_rgb_uint8(frame)

    assert out.dtype == np.uint8
    assert tuple(out[0, 0]) == (0, 128, 255)


def test_iter_video_frames_applies_inclusive_trim_before_stride(tmp_path):
    for i in range(6):
        frame = np.full((3, 4, 3), i, dtype=np.uint8)
        iio.imwrite(tmp_path / f"frame_{i:03d}.png", frame)

    frames = iter_video_frames(tmp_path, every_n=2, start_frame=1, end_frame=5)

    assert [frame.frame_id for frame in frames] == [1, 3, 5]
    assert [int(frame.image[0, 0, 0]) for frame in frames] == [1, 3, 5]


def test_detect_black_band_crop_finds_consistent_top_and_bottom_bands():
    rng = np.random.default_rng(12)
    records = []
    for i in range(5):
        frame = np.full((14, 9, 3), 0, dtype=np.uint8)
        frame[3:12] = rng.integers(45, 180, size=(9, 9, 3), dtype=np.uint8)
        records.append(FrameRecord(i, float(i), frame))

    crop = detect_black_band_crop(records)

    assert crop.applied
    assert crop.top == 3
    assert crop.bottom == 2
    assert crop.cropped_height == 9


def test_strip_black_bands_crops_frames_in_place_with_fixed_window():
    records = []
    for i in range(3):
        frame = np.zeros((10, 6, 3), dtype=np.uint8)
        frame[2:7] = 90 + i
        records.append(FrameRecord(i, float(i), frame))

    stripped, crop = strip_black_bands_from_frames(records)

    assert crop.applied
    assert (crop.top, crop.bottom) == (2, 3)
    assert [record.image.shape[:2] for record in stripped] == [(5, 6), (5, 6), (5, 6)]
    assert int(stripped[0].image[0, 0, 0]) == 90


def test_black_band_crop_refuses_all_dark_frames():
    records = [FrameRecord(i, float(i), np.zeros((12, 8, 3), dtype=np.uint8)) for i in range(4)]

    crop = detect_black_band_crop(records)

    assert not crop.applied
    assert crop.reason == "unsafe_crop_empty_content"
    assert crop.cropped_height == 12


def test_histogram_stabilization_suggestion_marks_last_frame_before_constant_regime(tmp_path):
    rng = np.random.default_rng(4)
    for i in range(130):
        if i < 115:
            base = 120 + (20 if i < 60 else 65)
            frame = np.clip(rng.normal(base, 18, (64, 64)), 0, 255).astype(np.uint8)
        else:
            frame = np.full((64, 64), 82, dtype=np.uint8)
            frame += rng.integers(0, 2, frame.shape, dtype=np.uint8)
        iio.imwrite(tmp_path / f"frame_{i:03d}.png", frame)

    suggestion = suggest_histogram_stabilization_trim(tmp_path)

    assert suggestion is not None
    assert suggestion.first_stable_frame == 115
    assert suggestion.cursor_frame == 114
    assert suggestion.stable_run_frames >= 8
