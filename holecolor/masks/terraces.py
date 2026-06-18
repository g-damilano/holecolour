from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any

import cv2
import numpy as np

from holecolor.core.types import HoleGeometry


@dataclass(slots=True)
class TerraceRegion:
    y0: int
    y1: int
    x0: int
    x1: int
    mask: np.ndarray

    @property
    def area_px(self) -> int:
        return int(np.count_nonzero(self.mask))

    def local_coords(self) -> tuple[np.ndarray, np.ndarray]:
        return np.nonzero(self.mask)

    def global_coords(self) -> tuple[np.ndarray, np.ndarray]:
        yy, xx = self.local_coords()
        return yy + int(self.y0), xx + int(self.x0)

    def paint(self, image: np.ndarray, color: np.ndarray, alpha: float = 0.45) -> None:
        yy, xx = self.local_coords()
        if yy.size == 0:
            return
        gy = yy + int(self.y0)
        gx = xx + int(self.x0)
        image[gy, gx] = np.clip((1.0 - alpha) * image[gy, gx] + alpha * color, 0, 255).astype(np.uint8)


@dataclass(slots=True)
class TerraceWidthPlan:
    hole_id: int
    lattice_u: int | None
    lattice_v: int | None
    neighbor_count: int
    min_center_pitch_px: float | None
    min_border_gap_px: float | None
    usable_annulus_span_px: float
    annulus_count: int
    annulus_width_px: float
    span_mode: str
    gap_basis: str
    fallback_mode: str
    radius_outer_px: float
    terrace_outer_radius_px: float

    def annulus_rows(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for annulus_index in range(int(self.annulus_count)):
            inner = float(self.radius_outer_px + annulus_index * self.annulus_width_px)
            outer = float(self.radius_outer_px + (annulus_index + 1) * self.annulus_width_px)
            rows.append({
                'hole_id': int(self.hole_id),
                'annulus_index': int(annulus_index),
                'inner_radius_px': inner,
                'outer_radius_px': outer,
                'annulus_width_px': float(self.annulus_width_px),
                'span_mode': str(self.span_mode),
                'gap_basis': str(self.gap_basis),
            })
        return rows




def make_hole_interior_region(shape: tuple[int, int], hole: HoleGeometry, shrink_px: int) -> TerraceRegion:
    r = max(float(hole.radius_inner_px - shrink_px), 1.0)
    y0, y1, x0, x1 = _local_bbox(shape, hole, r)
    if y1 <= y0 or x1 <= x0:
        return TerraceRegion(0, 0, 0, 0, np.zeros((0, 0), dtype=bool))
    yy, xx = np.mgrid[y0:y1, x0:x1]
    rr = np.sqrt((xx.astype(np.float32, copy=False) - np.float32(hole.x)) ** 2 + (yy.astype(np.float32, copy=False) - np.float32(hole.y)) ** 2, dtype=np.float32)
    mask = rr <= np.float32(r)
    return TerraceRegion(y0, y1, x0, x1, mask.astype(bool, copy=False))


def make_hole_rim_region(shape: tuple[int, int], hole: HoleGeometry, rim_width_px: int) -> TerraceRegion:
    r_outer = max(float(hole.radius_outer_px), 1.0)
    r_inner = max(float(hole.radius_outer_px - rim_width_px), 1.0)
    y0, y1, x0, x1 = _local_bbox(shape, hole, r_outer)
    if y1 <= y0 or x1 <= x0:
        return TerraceRegion(0, 0, 0, 0, np.zeros((0, 0), dtype=bool))
    yy, xx = np.mgrid[y0:y1, x0:x1]
    rr = np.sqrt((xx.astype(np.float32, copy=False) - np.float32(hole.x)) ** 2 + (yy.astype(np.float32, copy=False) - np.float32(hole.y)) ** 2, dtype=np.float32)
    mask = (rr <= np.float32(r_outer)) & (rr > np.float32(r_inner))
    return TerraceRegion(y0, y1, x0, x1, mask.astype(bool, copy=False))

def make_hole_interior_mask(shape: tuple[int, int], hole: HoleGeometry, shrink_px: int) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    r = max(int(round(hole.radius_inner_px - shrink_px)), 1)
    cv2.circle(mask, (int(round(hole.x)), int(round(hole.y))), r, 1, -1)
    return mask.astype(bool)


def make_hole_rim_mask(shape: tuple[int, int], hole: HoleGeometry, rim_width_px: int) -> np.ndarray:
    outer = np.zeros(shape, dtype=np.uint8)
    inner = np.zeros(shape, dtype=np.uint8)
    c = (int(round(hole.x)), int(round(hole.y)))
    cv2.circle(outer, c, int(round(hole.radius_outer_px)), 1, -1)
    cv2.circle(inner, c, max(int(round(hole.radius_outer_px - rim_width_px)), 1), 1, -1)
    return outer.astype(bool) & ~inner.astype(bool)


def make_distance_transform_terraces(roi_mask: np.ndarray, n_terraces: int) -> list[np.ndarray]:
    dist = cv2.distanceTransform(roi_mask.astype(np.uint8), cv2.DIST_L2, 3)
    if dist.max() <= 0:
        return [np.zeros_like(roi_mask, dtype=bool) for _ in range(n_terraces)]
    bins = np.linspace(0, dist.max(), n_terraces + 1)
    terraces = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        terraces.append(((dist >= lo) & (dist < hi) & roi_mask).astype(bool))
    return terraces


def make_hole_terraces(shape: tuple[int, int], hole: HoleGeometry, n_terraces: int, max_radius_px: float) -> list[np.ndarray]:
    yy, xx = np.indices(shape)
    rr = np.sqrt((xx - hole.x) ** 2 + (yy - hole.y) ** 2)
    outer = rr <= max_radius_px
    inner = rr <= hole.radius_outer_px
    roi = outer & ~inner
    bins = np.linspace(hole.radius_outer_px, max_radius_px, n_terraces + 1)
    terraces = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        terraces.append(((rr >= lo) & (rr < hi) & roi).astype(bool))
    return terraces


def _local_bbox(shape: tuple[int, int], hole: HoleGeometry, outer_radius: float) -> tuple[int, int, int, int]:
    h, w = shape
    pad = int(np.ceil(float(outer_radius)))
    cx = int(round(float(hole.x)))
    cy = int(round(float(hole.y)))
    x0 = max(0, cx - pad)
    x1 = min(w, cx + pad + 1)
    y0 = max(0, cy - pad)
    y1 = min(h, cy + pad + 1)
    return y0, y1, x0, x1




def _hole_radius_for_gap(hole: HoleGeometry) -> float:
    return float(max(hole.radius_outer_px, 0.0))


def _global_min_gap(holes: list[HoleGeometry], gap_basis: str) -> tuple[float | None, float | None]:
    min_center = None
    min_gap = None
    for i, h1 in enumerate(holes):
        for j in range(i + 1, len(holes)):
            h2 = holes[j]
            center = float(np.hypot(float(h1.x) - float(h2.x), float(h1.y) - float(h2.y)))
            gap = float(center - _hole_radius_for_gap(h1) - _hole_radius_for_gap(h2))
            if min_center is None or center < min_center:
                min_center = center
            if gap > 0 and (min_gap is None or gap < min_gap):
                min_gap = gap
    return min_center, min_gap


def build_terrace_width_plan(
    holes: list[HoleGeometry],
    n_terraces: int,
    lattice_indices: dict[int, tuple[int, int]] | None = None,
    width_mode: str = 'fixed',
    gap_basis: str = 'border_gap',
    terrace_outer_radius_px: float | None = None,
    min_width_px: float = 0.0,
) -> dict[int, TerraceWidthPlan]:
    if not holes:
        return {}
    by_hole = {int(h.hole_id): h for h in holes}
    lattice_by_hole = {} if lattice_indices is None else {int(k): (int(v[0]), int(v[1])) for k, v in lattice_indices.items() if v is not None}
    hole_by_lattice = {uv: hid for hid, uv in lattice_by_hole.items()}
    global_min_center, global_min_gap = _global_min_gap(holes, gap_basis)
    plans: dict[int, TerraceWidthPlan] = {}
    for hole in holes:
        hid = int(hole.hole_id)
        radius_outer = float(hole.radius_outer_px)
        uv = lattice_by_hole.get(hid)
        neighbor_ids: list[int] = []
        if uv is not None:
            u, v = uv
            for nuv in ((u - 1, v), (u + 1, v), (u, v - 1), (u, v + 1)):
                nhid = hole_by_lattice.get(nuv)
                if nhid is not None and nhid != hid:
                    neighbor_ids.append(int(nhid))
        center_pitches: list[float] = []
        border_gaps: list[float] = []
        for nhid in neighbor_ids:
            nh = by_hole[nhid]
            center = float(np.hypot(float(hole.x) - float(nh.x), float(hole.y) - float(nh.y)))
            gap = float(center - _hole_radius_for_gap(hole) - _hole_radius_for_gap(nh))
            center_pitches.append(center)
            if gap > 0:
                border_gaps.append(gap)
        min_center = min(center_pitches) if center_pitches else global_min_center
        min_gap = min(border_gaps) if border_gaps else global_min_gap
        fallback_mode = 'neighbors'
        if not neighbor_ids:
            fallback_mode = 'global_min_gap'
        elif gap_basis == 'border_gap' and not border_gaps:
            fallback_mode = 'global_min_gap'
        elif gap_basis == 'center_pitch' and not center_pitches:
            fallback_mode = 'global_min_pitch'
        if width_mode == 'fixed':
            outer_radius = float(terrace_outer_radius_px) if terrace_outer_radius_px is not None else float(radius_outer + 2.5 * max(radius_outer, 8.0))
            usable_span = max(float(outer_radius - radius_outer), 0.0)
        else:
            if gap_basis == 'center_pitch':
                base_span = float(min_center) if min_center is not None else 0.0
            else:
                base_span = float(min_gap) if min_gap is not None else 0.0
            usable_span = base_span if width_mode == 'full_gap' else 0.5 * base_span
            usable_span = max(float(usable_span), 0.0)
            if usable_span <= 0 and terrace_outer_radius_px is not None:
                usable_span = max(float(terrace_outer_radius_px - radius_outer), 0.0)
                fallback_mode = 'fixed_outer_radius'
            outer_radius = float(radius_outer + usable_span)
        annulus_width = max(float(usable_span / max(int(n_terraces), 1)), float(min_width_px))
        usable_span = float(annulus_width * max(int(n_terraces), 1))
        outer_radius = float(radius_outer + usable_span)
        plans[hid] = TerraceWidthPlan(
            hole_id=hid,
            lattice_u=None if uv is None else int(uv[0]),
            lattice_v=None if uv is None else int(uv[1]),
            neighbor_count=int(len(neighbor_ids)),
            min_center_pitch_px=None if min_center is None else float(min_center),
            min_border_gap_px=None if min_gap is None else float(min_gap),
            usable_annulus_span_px=float(usable_span),
            annulus_count=int(n_terraces),
            annulus_width_px=float(annulus_width),
            span_mode=str(width_mode),
            gap_basis=str(gap_basis),
            fallback_mode=str(fallback_mode),
            radius_outer_px=float(radius_outer),
            terrace_outer_radius_px=float(outer_radius),
        )
    return plans

def make_nonoverlapping_hole_terraces(
    shape: tuple[int, int],
    holes: list[HoleGeometry],
    n_terraces: int,
    terrace_outer_radius_px: float | None = None,
    lattice_indices: dict[int, tuple[int, int]] | None = None,
    width_mode: str = "fixed",
    gap_basis: str = "border_gap",
    min_width_px: float = 0.0,
    return_plan: bool = False,
) -> dict[int, list[TerraceRegion]] | tuple[dict[int, list[TerraceRegion]], dict[int, TerraceWidthPlan]]:
    """Memory-safe terrace assignment using local crops per hole.

    Instead of materializing an H x W x N_holes distance cube, each hole is
    processed inside a local crop and compared only against nearby competitors
    that could actually own pixels in that crop.
    """
    if not holes:
        return {}

    centers = np.asarray([[float(h.x), float(h.y)] for h in holes], dtype=np.float32)
    terrace_plan = build_terrace_width_plan(
        holes,
        n_terraces,
        lattice_indices=lattice_indices,
        width_mode=width_mode,
        gap_basis=gap_basis,
        terrace_outer_radius_px=terrace_outer_radius_px,
        min_width_px=min_width_px,
    )
    out: dict[int, list[TerraceRegion]] = {}
    for idx, hole in enumerate(holes):
        plan = terrace_plan.get(int(hole.hole_id))
        outer_radius = float(plan.terrace_outer_radius_px) if plan is not None else float(terrace_outer_radius_px) if terrace_outer_radius_px is not None else float(
            hole.radius_outer_px + 2.5 * max(hole.radius_outer_px, 8.0)
        )
        y0, y1, x0, x1 = _local_bbox(shape, hole, outer_radius)
        if y1 <= y0 or x1 <= x0:
            out[hole.hole_id] = [TerraceRegion(0, 0, 0, 0, np.zeros((0, 0), dtype=bool)) for _ in range(n_terraces)]
            continue

        yy, xx = np.mgrid[y0:y1, x0:x1]
        xx = xx.astype(np.float32, copy=False)
        yy = yy.astype(np.float32, copy=False)
        rr_current = np.sqrt((xx - np.float32(hole.x)) ** 2 + (yy - np.float32(hole.y)) ** 2, dtype=np.float32)
        owner = rr_current <= np.float32(outer_radius)

        center_dists = np.sqrt(np.sum((centers - centers[idx]) ** 2, axis=1), dtype=np.float32)
        competitor_ids = np.where((center_dists < np.float32(2.0 * outer_radius)) & (np.arange(len(holes)) != idx))[0]
        for j in competitor_ids.tolist():
            comp = holes[int(j)]
            rr_comp = np.sqrt((xx - np.float32(comp.x)) ** 2 + (yy - np.float32(comp.y)) ** 2, dtype=np.float32)
            owner &= rr_current <= rr_comp

        roi = owner & (rr_current > np.float32(hole.radius_outer_px)) & (rr_current <= np.float32(outer_radius))
        bins = np.linspace(float(hole.radius_outer_px), float(outer_radius), int(n_terraces) + 1, dtype=np.float32)
        terraces: list[TerraceRegion] = []
        for lo, hi in zip(bins[:-1], bins[1:]):
            local_mask = roi & (rr_current >= lo) & (rr_current < hi)
            terraces.append(TerraceRegion(y0, y1, x0, x1, local_mask.astype(bool, copy=False)))
        out[hole.hole_id] = terraces
    if return_plan:
        return out, terrace_plan
    return out
