import numpy as np

from holecolor.masks.terraces import TerraceRegion


def test_terrace_paint_rgb_image() -> None:
    image = np.zeros((10, 10, 3), dtype=np.uint8)
    mask = np.zeros((4, 4), dtype=bool)
    mask[1:3, 1:3] = True
    region = TerraceRegion(2, 6, 3, 7, mask)
    region.paint(image, np.array([255, 80, 80], dtype=np.uint8), alpha=0.5)
    painted = image[3:5, 4:6]
    assert painted.shape == (2, 2, 3)
    assert np.any(painted[..., 0] > 0)


def test_terrace_paint_rgba_image_preserves_extra_channels() -> None:
    image = np.zeros((8, 8, 4), dtype=np.uint8)
    image[..., 3] = 123
    mask = np.zeros((3, 3), dtype=bool)
    mask[1, :] = True
    region = TerraceRegion(2, 5, 2, 5, mask)
    region.paint(image, np.array([255, 0, 0], dtype=np.uint8), alpha=0.5)
    painted = image[3, 2:5]
    assert painted.shape == (3, 4)
    assert np.all(painted[:, 3] == 123)
    assert np.any(painted[:, 0] > 0)


def test_terrace_paint_grayscale_image() -> None:
    image = np.zeros((8, 8), dtype=np.uint8)
    mask = np.zeros((2, 2), dtype=bool)
    mask[:, :] = True
    region = TerraceRegion(1, 3, 1, 3, mask)
    region.paint(image, np.array([255, 255, 255], dtype=np.uint8), alpha=0.5)
    assert image[1:3, 1:3].mean() > 0
