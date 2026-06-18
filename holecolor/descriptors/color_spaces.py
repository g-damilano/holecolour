from __future__ import annotations

import cv2
import numpy as np


def rgb_to_hsv(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.float32) / np.array([179.0, 255.0, 255.0], dtype=np.float32)


def rgb_to_hsl(image: np.ndarray) -> np.ndarray:
    rgb = image.astype(np.float32) / 255.0
    mx = rgb.max(axis=2)
    mn = rgb.min(axis=2)
    c = mx - mn
    l = 0.5 * (mx + mn)
    s = np.where(c == 0, 0, c / (1 - np.abs(2 * l - 1) + 1e-6))
    h = np.zeros_like(l)
    mask = c > 0
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    idx = mask & (mx == r)
    h[idx] = ((g[idx] - b[idx]) / c[idx]) % 6
    idx = mask & (mx == g)
    h[idx] = (b[idx] - r[idx]) / c[idx] + 2
    idx = mask & (mx == b)
    h[idx] = (r[idx] - g[idx]) / c[idx] + 4
    h /= 6.0
    return np.stack([h, s, l], axis=2)


def chromatic_distance(image: np.ndarray, baseline: np.ndarray) -> np.ndarray:
    diff = image.astype(np.float32) - baseline.astype(np.float32)
    return np.sqrt(np.sum(diff * diff, axis=2))


def mask_low_saturation_hue(hsv_or_hsl: np.ndarray, min_sat: float) -> np.ndarray:
    return hsv_or_hsl[..., 1] >= min_sat


def descriptor_image(image_rgb: np.ndarray, image_hsv: np.ndarray, name: str) -> np.ndarray:
    name=name.lower()
    if name=="r": return image_rgb[...,0].astype(np.float32)/255.0
    if name=="g": return image_rgb[...,1].astype(np.float32)/255.0
    if name=="b": return image_rgb[...,2].astype(np.float32)/255.0
    if name=="h": return image_hsv[...,0].astype(np.float32)
    if name=="s": return image_hsv[...,1].astype(np.float32)
    raise ValueError(f"Unsupported descriptor {name!r}")
