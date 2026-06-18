from __future__ import annotations

import math

import numpy as np

from holecolor.core.types import HoleCandidate, LatticeModel


def _centers(candidates: list[HoleCandidate]) -> np.ndarray:
    return np.array([[c.x, c.y] for c in candidates], dtype=np.float32)


def _nearest_neighbor_distances(pts: np.ndarray) -> np.ndarray:
    diffs = pts[None, :, :] - pts[:, None, :]
    dist = np.linalg.norm(diffs, axis=2)
    dist[dist == 0] = np.inf
    return np.min(dist, axis=1)


def _angle_diff(a: np.ndarray, b: float) -> np.ndarray:
    d = np.abs(a - b)
    return np.minimum(d, np.pi - d)


def _dominant_theta(vectors: np.ndarray) -> float:
    ang = np.mod(np.arctan2(vectors[:, 1], vectors[:, 0]), np.pi)
    hist, edges = np.histogram(ang, bins=90, range=(0.0, np.pi))
    idx = int(np.argmax(hist))
    return float(0.5 * (edges[idx] + edges[idx + 1]))


def _basis_from_vectors(vectors: np.ndarray, spacing_est: float, theta_u: float, theta_v: float, angle_tol_deg: float):
    tol = np.deg2rad(angle_tol_deg)
    ang = np.mod(np.arctan2(vectors[:, 1], vectors[:, 0]), np.pi)

    def _pick(theta: float) -> np.ndarray:
        sel = vectors[_angle_diff(ang, theta) < tol]
        if len(sel) == 0:
            return np.array([spacing_est * np.cos(theta), spacing_est * np.sin(theta)], dtype=np.float32)
        axis = np.array([np.cos(theta), np.sin(theta)], dtype=np.float32)
        signs = np.sign(sel @ axis)
        signs[signs == 0] = 1.0
        sel = sel * signs[:, None]
        lengths = np.linalg.norm(sel, axis=1)
        dirs = sel / np.maximum(lengths[:, None], 1e-6)
        direction = np.median(dirs, axis=0)
        direction /= np.linalg.norm(direction) + 1e-6
        return (direction * np.median(lengths)).astype(np.float32)

    return _pick(theta_u), _pick(theta_v)


def _choose_origin(pts: np.ndarray, basis_u: np.ndarray, basis_v: np.ndarray) -> np.ndarray:
    B = np.column_stack([basis_u, basis_v]).astype(np.float32)
    inv = np.linalg.pinv(B)
    best_origin = pts[0]
    best_score = float('inf')
    for origin in pts:
        uv = (pts - origin) @ inv.T
        residual = np.mean(np.linalg.norm(uv - np.round(uv), axis=1))
        if residual < best_score:
            best_score = float(residual)
            best_origin = origin
    return best_origin.astype(np.float32)


def estimate_lattice_basis(candidates: list[HoleCandidate], angle_tolerance_deg: float = 18.0) -> LatticeModel:
    pts = _centers(candidates)
    if len(pts) < 4:
        return LatticeModel(0.0, 0.0, (1.0, 0.0), (0.0, 1.0), 0.0, 0.0, 0.0, 0.0)

    nnd = _nearest_neighbor_distances(pts)
    spacing_est = float(np.median(nnd[np.isfinite(nnd)]))
    diffs = pts[None, :, :] - pts[:, None, :]
    vectors = diffs.reshape(-1, 2)
    dist = np.linalg.norm(vectors, axis=1)
    keep = (dist > 0.6 * spacing_est) & (dist < 1.6 * spacing_est)
    vectors = vectors[keep]
    if len(vectors) < 4:
        origin = pts.mean(axis=0)
        return LatticeModel(float(origin[0]), float(origin[1]), (spacing_est, 0.0), (0.0, spacing_est), 0.0, spacing_est, spacing_est, 0.1)

    ang = np.mod(np.arctan2(vectors[:, 1], vectors[:, 0]), np.pi)
    hist, edges = np.histogram(ang, bins=180, range=(0.0, np.pi))
    i1 = int(np.argmax(hist))
    theta_u = float(0.5 * (edges[i1] + edges[i1 + 1]))
    sep_mask = np.abs(((ang - theta_u + np.pi / 2.0) % np.pi) - np.pi / 2.0) > np.deg2rad(15.0)
    if np.any(sep_mask):
        hist2, edges2 = np.histogram(ang[sep_mask], bins=180, range=(0.0, np.pi))
        i2 = int(np.argmax(hist2))
        theta_v = float(0.5 * (edges2[i2] + edges2[i2 + 1]))
    else:
        theta_v = float((theta_u + np.pi / 2.0) % np.pi)
    basis_u, basis_v = _basis_from_vectors(vectors, spacing_est, theta_u, theta_v, angle_tolerance_deg)
    cross_z = float(basis_u[0] * basis_v[1] - basis_u[1] * basis_v[0])
    if cross_z < 0:
        basis_v = -basis_v

    origin = _choose_origin(pts, basis_u, basis_v)
    B = np.column_stack([basis_u, basis_v]).astype(np.float32)
    inv = np.linalg.pinv(B)
    uv = (pts - origin) @ inv.T
    residual = np.linalg.norm(uv - np.round(uv), axis=1)
    spacing_u = float(np.linalg.norm(basis_u))
    spacing_v = float(np.linalg.norm(basis_v))
    spacing_cv = float(np.std(nnd) / max(np.mean(nnd), 1e-6))
    conf = float(np.clip((1.0 / (1.0 + np.mean(residual))) * (1.0 / (1.0 + spacing_cv)), 0.0, 1.0))
    angle = float(math.degrees(math.atan2(float(basis_u[1]), float(basis_u[0]))))
    return LatticeModel(float(origin[0]), float(origin[1]), (float(basis_u[0]), float(basis_u[1])), (float(basis_v[0]), float(basis_v[1])), angle, spacing_u, spacing_v, conf)


def assign_lattice_indices(candidates: list[HoleCandidate], lattice: LatticeModel) -> dict[int, tuple[int, int]]:
    if lattice.spacing_u_px <= 0 or lattice.spacing_v_px <= 0:
        return {}
    b = np.array([lattice.basis_u, lattice.basis_v], dtype=np.float32).T
    inv = np.linalg.pinv(b)
    origin = np.array([lattice.origin_x, lattice.origin_y], dtype=np.float32)
    out = {}
    for i, c in enumerate(candidates):
        uv = inv @ (np.array([c.x, c.y], dtype=np.float32) - origin)
        out[i] = (int(round(float(uv[0]))), int(round(float(uv[1]))))
    return out
