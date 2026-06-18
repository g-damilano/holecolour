from __future__ import annotations

from collections import Counter

import numpy as np

from holecolor.core.types import HotspotTrackSummary
from holecolor.hotspots.detect import Hotspot


def _link_cost(a: Hotspot, b: Hotspot) -> float:
    d = float(((a.cx - b.cx) ** 2 + (a.cy - b.cy) ** 2) ** 0.5)
    area_ratio = max(a.area_px, b.area_px) / max(min(a.area_px, b.area_px), 1)
    hole_penalty = 0.0 if (a.nearest_hole_id == b.nearest_hole_id or a.nearest_hole_id is None or b.nearest_hole_id is None) else 1e6
    score_penalty = abs(float(a.score) - float(b.score))
    return d + 0.5 * area_ratio + 0.25 * score_penalty + hole_penalty


def link_hotspots(
    regions_t: list[Hotspot],
    regions_t1: list[Hotspot],
    max_dist_px: float = 10.0,
    max_area_ratio: float = 3.0,
) -> list[tuple[int, int]]:
    candidates: list[tuple[float, int, int]] = []
    for a in regions_t:
        for b in regions_t1:
            d = float(((a.cx - b.cx) ** 2 + (a.cy - b.cy) ** 2) ** 0.5)
            if d > max_dist_px:
                continue
            area_ratio = max(a.area_px, b.area_px) / max(min(a.area_px, b.area_px), 1)
            if area_ratio > max_area_ratio:
                continue
            if a.nearest_hole_id is not None and b.nearest_hole_id is not None and a.nearest_hole_id != b.nearest_hole_id:
                continue
            candidates.append((_link_cost(a, b), a.hotspot_id, b.hotspot_id))
    links: list[tuple[int, int]] = []
    used_a: set[int] = set()
    used_b: set[int] = set()
    for _, aid, bid in sorted(candidates, key=lambda x: x[0]):
        if aid in used_a or bid in used_b:
            continue
        used_a.add(aid)
        used_b.add(bid)
        links.append((aid, bid))
    return links


def summarize_tracks(hotspot_rows: list[dict]) -> list[HotspotTrackSummary]:
    if not hotspot_rows:
        return []
    by_track: dict[int, list[dict]] = {}
    for row in hotspot_rows:
        by_track.setdefault(int(row['track_id']), []).append(row)
    out: list[HotspotTrackSummary] = []
    for track_id, rows in sorted(by_track.items()):
        rows = sorted(rows, key=lambda r: (int(r['frame_id']), int(r['hotspot_id'])))
        holes = [r.get('nearest_hole_id') for r in rows if r.get('nearest_hole_id') is not None]
        dists = [float(r['dist_to_hole_px']) for r in rows if r.get('dist_to_hole_px') is not None]
        dominant = Counter(holes).most_common(1)[0][0] if holes else None
        out.append(HotspotTrackSummary(
            track_id=track_id,
            start_frame=int(rows[0]['frame_id']),
            end_frame=int(rows[-1]['frame_id']),
            duration_frames=int(rows[-1]['frame_id']) - int(rows[0]['frame_id']) + 1,
            mean_area_px=float(np.mean([float(r['area_px']) for r in rows])),
            mean_score=float(np.mean([float(r['score']) for r in rows])),
            dominant_hole_id=None if dominant is None else int(dominant),
            mean_dist_to_hole_px=float(np.mean(dists)) if dists else None,
        ))
    return out
