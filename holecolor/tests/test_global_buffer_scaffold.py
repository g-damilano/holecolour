import numpy as np

from holecolor.core.types import FrameRecord
from holecolor.descriptors.color_spaces import descriptor_image, rgb_to_hsv
from holecolor.geometry.models import BufferGeometry, WaferGeometry
from holecolor.pipeline import _buffer_region_mask, _buffer_vector_masks, _global_buffer_row, _global_buffer_band_rows


def test_global_buffer_scaffold_uses_support_proxy_and_emits_stats() -> None:
    image = np.full((40, 50, 3), 100, dtype=np.uint8)
    image[10:30, 15:35, :] = 160
    frame = FrameRecord(0, 0.0, image)
    wafer = WaferGeometry(id='wafer-0', center_xy_px=(25.0, 20.0), radius_px=14.0, confidence=0.9)
    buffer = BufferGeometry(id='buffer-0', state='unknown', center_xy_px=None, radius_px=None, confidence=0.0)
    support_mask = np.zeros((40, 50), dtype=bool)
    support_mask[8:32, 12:38] = True
    mask, policy = _buffer_region_mask(image.shape[:2], wafer, buffer, support_mask)
    hsv = rgb_to_hsv(image)
    desc = descriptor_image(image, hsv, 'b')
    row = _global_buffer_row(frame, hsv, mask, policy, 'b', desc)
    assert policy == 'wafer_support_proxy'
    assert row['area_px'] == int(mask.sum())
    assert row['descriptor'] == 'b'
    assert row['primary_descriptor_mean'] > 0


def test_global_buffer_scaffold_uses_buffer_circle_when_available() -> None:
    image = np.full((60, 60, 3), 90, dtype=np.uint8)
    frame = FrameRecord(1, 1.0, image)
    wafer = WaferGeometry(id='wafer-0', center_xy_px=(30.0, 30.0), radius_px=25.0, confidence=0.9)
    buffer = BufferGeometry(id='buffer-0', state='partial', center_xy_px=(45.0, 30.0), radius_px=20.0, confidence=0.5)
    mask, policy = _buffer_region_mask(image.shape[:2], wafer, buffer, None)
    hsv = rgb_to_hsv(image)
    desc = descriptor_image(image, hsv, 'b')
    row = _global_buffer_row(frame, hsv, mask, policy, 'b', desc)
    assert policy == 'buffer_partial'
    assert row['area_px'] > 0



def test_global_buffer_vector_masks_emit_border_and_band_rows_when_buffer_known() -> None:
    image = np.full((80, 80, 3), 120, dtype=np.uint8)
    frame = FrameRecord(2, 2.0, image)
    wafer = WaferGeometry(id='wafer-0', center_xy_px=(40.0, 40.0), radius_px=30.0, confidence=0.9)
    buffer = BufferGeometry(id='buffer-0', state='full', center_xy_px=(40.0, 40.0), radius_px=20.0, confidence=0.8)
    vec = _buffer_vector_masks(image.shape[:2], wafer, buffer, None)
    hsv = rgb_to_hsv(image)
    desc = descriptor_image(image, hsv, 'b')
    rows = _global_buffer_band_rows(frame, hsv, 'b', desc, vec)
    assert vec['border_mask'].sum() > 0
    assert rows
    assert {r['band_id'] for r in rows} == {'center_core', 'mid_band', 'border_band'}


def test_global_buffer_row_can_record_hotspot_fraction_of_buffer_area() -> None:
    image = np.full((40, 50, 3), 100, dtype=np.uint8)
    frame = FrameRecord(3, 3.0, image)
    wafer = WaferGeometry(id='wafer-0', center_xy_px=(25.0, 20.0), radius_px=14.0, confidence=0.9)
    buffer = BufferGeometry(id='buffer-0', state='unknown', center_xy_px=None, radius_px=None, confidence=0.0)
    support_mask = np.zeros((40, 50), dtype=bool)
    support_mask[8:32, 12:38] = True
    mask, policy = _buffer_region_mask(image.shape[:2], wafer, buffer, support_mask)
    hsv = rgb_to_hsv(image)
    desc = descriptor_image(image, hsv, 'b')
    row = _global_buffer_row(frame, hsv, mask, policy, 'b', desc, hotspot_fraction_of_buffer_area=0.25)
    assert row['hotspot_fraction_of_buffer_area'] == 0.25
