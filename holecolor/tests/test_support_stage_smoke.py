import cv2

from holecolor.geometry.support import detect_support_geometries_from_sequence


def test_support_stage_smoke_on_real_video_sample() -> None:
    cap = cv2.VideoCapture('/mnt/data/JAO25.avi')
    images = []
    idx = 0
    while len(images) < 5:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % 10 == 0:
            images.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        idx += 1
    cap.release()
    assert images
    wafer, buffer, mask = detect_support_geometries_from_sequence(images)
    assert wafer.radius_px > 0
    assert buffer.state in {'unknown', 'full', 'partial'}
    if buffer.state != 'unknown':
        assert buffer.radius_px is not None and buffer.radius_px > 0
    assert mask is not None and mask.any()
