from __future__ import annotations

from holecolor.hotspots.detect import Hotspot


def link_regions_across_frames(regions_t: list[Hotspot], regions_t1: list[Hotspot], max_dist_px: float) -> list[tuple[int, int]]:
    links: list[tuple[int, int]] = []
    for a in regions_t:
        best = None
        best_d = max_dist_px
        for b in regions_t1:
            d = ((a.cx - b.cx) ** 2 + (a.cy - b.cy) ** 2) ** 0.5
            if d < best_d:
                best = b.hotspot_id
                best_d = d
        if best is not None:
            links.append((a.hotspot_id, best))
    return links
