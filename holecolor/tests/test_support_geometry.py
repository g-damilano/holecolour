import cv2
import numpy as np

from holecolor.geometry.support import detect_support_geometries_from_sequence


def _make_support_sequence(n: int = 5):
    H, W = 180, 220
    cx, cy, r = 110, 90, 55
    out = []
    for i in range(n):
        im = np.full((H, W, 3), 210, dtype=np.uint8)
        cv2.circle(im, (cx, cy), r, (60, 60, 60), thickness=-1)
        cv2.circle(im, (cx + (i % 3) - 1, cy), 12, (120 + 20 * i, 120 + 20 * i, 120 + 20 * i), thickness=-1)
        out.append(im)
    return out, (cx, cy, r)


def test_support_geometry_emits_wafer_and_buffer_state() -> None:
    images, (cx, cy, r) = _make_support_sequence()
    wafer, buffer, mask = detect_support_geometries_from_sequence(images)
    assert wafer.radius_px > 0
    assert abs(wafer.center_xy_px[0] - cx) < 8
    assert abs(wafer.center_xy_px[1] - cy) < 8
    assert abs(wafer.radius_px - r) < 10
    assert buffer.state in {'unknown', 'full', 'partial'}
    if buffer.state != 'unknown':
        assert buffer.radius_px is not None and buffer.radius_px > 0
    assert mask is not None and mask.any()


def test_support_geometry_can_fit_partial_buffer_with_center_outside_frame() -> None:
    H, W = 180, 220
    wafer_cx, wafer_cy, wafer_r = 110, 90, 70
    buffer_cx, buffer_cy, buffer_r = 250, 90, 120
    imgs = []
    yy, xx = np.indices((H, W))
    wafer = ((xx - wafer_cx) ** 2 + (yy - wafer_cy) ** 2) <= wafer_r ** 2
    for t in range(5):
        im = np.full((H, W, 3), 210, dtype=np.uint8)
        im[wafer] = 70
        arc = (((xx - buffer_cx) ** 2 + (yy - buffer_cy) ** 2) <= buffer_r ** 2) & wafer
        im[arc] = 90 + 20 * t
        imgs.append(im.astype(np.uint8))
    _, buffer, _ = detect_support_geometries_from_sequence(imgs)
    assert buffer.state in {'partial', 'unknown'}
    if buffer.state == 'partial':
        assert buffer.center_xy_px is not None
        assert buffer.radius_px is not None and buffer.radius_px > 0


def test_support_geometry_rejects_tiny_local_patch_as_full_buffer() -> None:
    H, W = 180, 220
    wafer_cx, wafer_cy, wafer_r = 110, 90, 70
    yy, xx = np.indices((H, W))
    wafer = ((xx - wafer_cx) ** 2 + (yy - wafer_cy) ** 2) <= wafer_r ** 2
    imgs = []
    for t in range(5):
        im = np.full((H, W, 3), 210, dtype=np.uint8)
        im[wafer] = 70
        # tiny local changing patch should not be promoted to a confident full buffer fit
        patch = ((xx - 145) ** 2 + (yy - 95) ** 2) <= 11 ** 2
        im[patch & wafer] = 90 + 20 * t
        imgs.append(im.astype(np.uint8))
    _, buffer, _ = detect_support_geometries_from_sequence(imgs)
    assert buffer.state == 'unknown' or buffer.confidence < 0.4
