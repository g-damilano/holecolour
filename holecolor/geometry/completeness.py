from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from holecolor.core.types import HoleCandidate, HoleGeometry
from holecolor.masks.terraces import build_terrace_width_plan


@dataclass(slots=True)
class CompleteGeometryFilterResult:
    candidates: list[HoleCandidate]
    holes: list[HoleGeometry]
    lattice_indices: dict[int, tuple[int, int]]
    rows: list[dict[str, Any]]
    summary: dict[str, Any]


def _disk_inside_frame(shape: tuple[int, int], x: float, y: float, radius: float) -> bool:
    h, w = int(shape[0]), int(shape[1])
    r = float(max(radius, 0.0))
    return bool(float(x) - r >= 0.0 and float(y) - r >= 0.0 and float(x) + r <= float(w - 1) and float(y) + r <= float(h - 1))


def _disk_inside_circle(x: float, y: float, radius: float, circle: tuple[float, float, float] | None) -> bool:
    if circle is None:
        return True
    cx, cy, cr = circle
    return bool(float(np.hypot(float(x) - float(cx), float(y) - float(cy))) + float(max(radius, 0.0)) <= float(cr) + 1e-6)


def _disk_inside_mask(mask: np.ndarray | None, x: float, y: float, radius: float) -> bool:
    if mask is None:
        return True
    support = np.asarray(mask, dtype=bool)
    h, w = support.shape[:2]
    r = float(max(radius, 0.0))
    if not _disk_inside_frame((h, w), x, y, r):
        return False
    x0 = int(np.floor(float(x) - r))
    x1 = int(np.ceil(float(x) + r)) + 1
    y0 = int(np.floor(float(y) - r))
    y1 = int(np.ceil(float(y) + r)) + 1
    if x0 < 0 or y0 < 0 or x1 > w or y1 > h:
        return False
    yy, xx = np.mgrid[y0:y1, x0:x1]
    disk = ((xx.astype(np.float32, copy=False) - np.float32(x)) ** 2 + (yy.astype(np.float32, copy=False) - np.float32(y)) ** 2) <= np.float32(r * r)
    if not np.any(disk):
        return False
    return bool(np.all(support[y0:y1, x0:x1][disk]))


def _support_source(support_mask: np.ndarray | None, support_circle: tuple[float, float, float] | None) -> str:
    if support_circle is not None:
        return "circle"
    if support_mask is None:
        return "frame_only"
    mask = np.asarray(support_mask, dtype=bool)
    if mask.size == 0 or bool(np.all(mask)):
        return "frame_only"
    return "mask"


def _uv_for_hole(lattice_indices: dict[int, tuple[int, int]], candidate_id: int, hole_id: int) -> tuple[int, int] | None:
    uv = lattice_indices.get(int(hole_id))
    if uv is None:
        uv = lattice_indices.get(int(candidate_id))
    if uv is None:
        return None
    return int(uv[0]), int(uv[1])


def _containment_flags(
    shape: tuple[int, int],
    support_mask: np.ndarray | None,
    support_circle: tuple[float, float, float] | None,
    hole: HoleGeometry,
    terrace_outer_radius_px: float,
) -> dict[str, bool]:
    hole_r = float(max(hole.radius_outer_px, 0.0))
    terrace_r = float(max(terrace_outer_radius_px, hole_r))
    use_mask = support_circle is None
    return {
        "hole_inside_frame": _disk_inside_frame(shape, hole.x, hole.y, hole_r),
        "terrace_inside_frame": _disk_inside_frame(shape, hole.x, hole.y, terrace_r),
        "hole_inside_wafer": _disk_inside_circle(hole.x, hole.y, hole_r, support_circle)
        and (not use_mask or _disk_inside_mask(support_mask, hole.x, hole.y, hole_r)),
        "terrace_inside_wafer": _disk_inside_circle(hole.x, hole.y, terrace_r, support_circle)
        and (not use_mask or _disk_inside_mask(support_mask, hole.x, hole.y, terrace_r)),
    }


def _reasons_from_flags(flags: dict[str, bool]) -> list[str]:
    reasons: list[str] = []
    if not flags["hole_inside_frame"]:
        reasons.append("hole_outside_frame")
    if not flags["terrace_inside_frame"]:
        reasons.append("terrace_outside_frame")
    if not flags["hole_inside_wafer"]:
        reasons.append("hole_outside_wafer")
    if not flags["terrace_inside_wafer"]:
        reasons.append("terrace_outside_wafer")
    return reasons


def filter_complete_holes_and_terraces(
    holes: list[HoleGeometry],
    candidates: list[HoleCandidate],
    lattice_indices: dict[int, tuple[int, int]],
    image_shape: tuple[int, int],
    *,
    n_terraces: int,
    terrace_width_mode: str,
    terrace_gap_basis: str,
    terrace_min_width_px: float,
    support_mask: np.ndarray | None = None,
    support_circle: tuple[float, float, float] | None = None,
) -> CompleteGeometryFilterResult:
    """Keep only holes whose complete hole and planned terrace disks are visible.

    The terrace radius is recomputed after each exclusion pass. That makes the
    final accepted set a fixed point: every retained hole satisfies the rule
    under the same terrace plan that downstream analysis will use.
    """
    paired = [
        {
            "candidate_id": int(i),
            "hole": hole,
            "candidate": candidates[i],
            "uv": _uv_for_hole(lattice_indices, i, int(hole.hole_id)),
        }
        for i, hole in enumerate(holes[: len(candidates)])
    ]
    active = {int(item["candidate_id"]) for item in paired}
    excluded: dict[int, dict[str, Any]] = {}
    final_plan = {}
    iterations = 0

    while active:
        iterations += 1
        active_items = [item for item in paired if int(item["candidate_id"]) in active]
        active_holes = [item["hole"] for item in active_items]
        active_lattice = {
            int(item["hole"].hole_id): item["uv"]
            for item in active_items
            if item["uv"] is not None
        }
        plan = build_terrace_width_plan(
            active_holes,
            n_terraces,
            lattice_indices=active_lattice,
            width_mode=str(terrace_width_mode),
            gap_basis=str(terrace_gap_basis),
            min_width_px=float(terrace_min_width_px),
        )
        rejected_this_pass: list[int] = []
        for item in active_items:
            hole = item["hole"]
            candidate_id = int(item["candidate_id"])
            terrace_plan = plan.get(int(hole.hole_id))
            terrace_outer = float(terrace_plan.terrace_outer_radius_px) if terrace_plan is not None else float(hole.radius_outer_px)
            flags = _containment_flags(image_shape, support_mask, support_circle, hole, terrace_outer)
            reasons = _reasons_from_flags(flags)
            if not reasons:
                continue
            uv = item["uv"]
            excluded[candidate_id] = {
                "original_candidate_id": candidate_id,
                "original_hole_id": int(hole.hole_id),
                "new_hole_id": None,
                "status": "excluded",
                "exclude_reasons": ";".join(reasons),
                "exclusion_iteration": int(iterations),
                "x": float(hole.x),
                "y": float(hole.y),
                "radius_outer_px": float(hole.radius_outer_px),
                "terrace_outer_radius_px": float(terrace_outer),
                "lattice_u": None if uv is None else int(uv[0]),
                "lattice_v": None if uv is None else int(uv[1]),
                **flags,
            }
            rejected_this_pass.append(candidate_id)
        if not rejected_this_pass:
            final_plan = plan
            break
        for candidate_id in rejected_this_pass:
            active.discard(candidate_id)

    if not active:
        final_plan = {}

    kept_items = [item for item in paired if int(item["candidate_id"]) in active]
    kept_candidates: list[HoleCandidate] = []
    kept_holes: list[HoleGeometry] = []
    kept_lattice: dict[int, tuple[int, int]] = {}
    rows: list[dict[str, Any]] = []
    for new_id, item in enumerate(kept_items):
        hole = item["hole"]
        uv = item["uv"]
        terrace_plan = final_plan.get(int(hole.hole_id))
        terrace_outer = float(terrace_plan.terrace_outer_radius_px) if terrace_plan is not None else float(hole.radius_outer_px)
        flags = _containment_flags(image_shape, support_mask, support_circle, hole, terrace_outer)
        kept_candidates.append(item["candidate"])
        kept_holes.append(
            HoleGeometry(
                int(new_id),
                float(hole.x),
                float(hole.y),
                float(hole.radius_inner_px),
                float(hole.radius_outer_px),
                float(hole.confidence),
            )
        )
        if uv is not None:
            kept_lattice[int(new_id)] = (int(uv[0]), int(uv[1]))
        rows.append(
            {
                "original_candidate_id": int(item["candidate_id"]),
                "original_hole_id": int(hole.hole_id),
                "new_hole_id": int(new_id),
                "status": "kept",
                "exclude_reasons": "",
                "exclusion_iteration": None,
                "x": float(hole.x),
                "y": float(hole.y),
                "radius_outer_px": float(hole.radius_outer_px),
                "terrace_outer_radius_px": float(terrace_outer),
                "lattice_u": None if uv is None else int(uv[0]),
                "lattice_v": None if uv is None else int(uv[1]),
                **flags,
            }
        )

    rows.extend(excluded[candidate_id] for candidate_id in sorted(excluded))
    reason_counts: dict[str, int] = {}
    for row in excluded.values():
        for reason in str(row["exclude_reasons"]).split(";"):
            if reason:
                reason_counts[reason] = int(reason_counts.get(reason, 0) + 1)

    summary = {
        "criterion": "complete hole disk and complete planned terrace disk must be inside both video frame and wafer support",
        "input_holes": int(len(paired)),
        "kept_holes": int(len(kept_holes)),
        "excluded_holes": int(len(excluded)),
        "iterations": int(iterations),
        "support_source": _support_source(support_mask, support_circle),
        "excluded_by_reason": reason_counts,
        "n_terraces": int(n_terraces),
        "terrace_width_mode": str(terrace_width_mode),
        "terrace_gap_basis": str(terrace_gap_basis),
    }
    return CompleteGeometryFilterResult(kept_candidates, kept_holes, kept_lattice, rows, summary)
