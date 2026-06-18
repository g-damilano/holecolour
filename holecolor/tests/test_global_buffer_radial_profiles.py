import numpy as np

from holecolor.core.types import FrameRecord
from holecolor.descriptors.color_spaces import descriptor_image, rgb_to_hsv
from holecolor.geometry.models import BufferGeometry, WaferGeometry
from holecolor.pipeline import _buffer_distance_maps, _global_buffer_radial_rows


def test_global_buffer_radial_profiles_emit_center_and_border_rows_when_buffer_known() -> None:
    h = w = 80
    yy, xx = np.indices((h, w))
    image = np.zeros((h, w, 3), dtype=np.uint8)
    dist = np.sqrt((xx - 40.0) ** 2 + (yy - 40.0) ** 2)
    image[..., 2] = np.clip(200 - 3 * dist, 0, 255).astype(np.uint8)
    frame = FrameRecord(0, 0.0, image)
    wafer = WaferGeometry(id='wafer-0', center_xy_px=(40.0, 40.0), radius_px=32.0, confidence=0.9)
    buffer = BufferGeometry(id='buffer-0', state='full', center_xy_px=(40.0, 40.0), radius_px=20.0, confidence=0.8)
    maps = _buffer_distance_maps(image.shape[:2], wafer, buffer, None)
    hsv = rgb_to_hsv(image)
    desc = descriptor_image(image, hsv, 'b')
    rows = _global_buffer_radial_rows(frame, hsv, 'b', desc, maps)
    assert rows
    axes = {r['axis'] for r in rows}
    assert axes == {'center', 'border'}
    border_rows = [r for r in rows if r['axis'] == 'border']
    assert border_rows
    assert all(r['axis_policy'] == 'buffer_border_full' for r in border_rows)


def test_global_buffer_radial_profiles_skip_border_when_buffer_unknown() -> None:
    image = np.full((50, 60, 3), 120, dtype=np.uint8)
    frame = FrameRecord(0, 0.0, image)
    wafer = WaferGeometry(id='wafer-0', center_xy_px=(30.0, 25.0), radius_px=20.0, confidence=0.9)
    buffer = BufferGeometry(id='buffer-0', state='unknown', center_xy_px=None, radius_px=None, confidence=0.0)
    support_mask = np.zeros((50, 60), dtype=bool)
    support_mask[10:40, 15:45] = True
    maps = _buffer_distance_maps(image.shape[:2], wafer, buffer, support_mask)
    hsv = rgb_to_hsv(image)
    desc = descriptor_image(image, hsv, 'b')
    rows = _global_buffer_radial_rows(frame, hsv, 'b', desc, maps)
    assert rows
    assert {r['axis'] for r in rows} == {'center'}
    assert all(r['axis_policy'] == 'wafer_center_proxy' for r in rows)
