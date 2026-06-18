from __future__ import annotations

from typing import Iterable

from holecolor.geometry.models import BufferGeometry, HoleBufferRelation, HoleTierRecord


def classify_hole_against_buffer(hole: HoleTierRecord, buffer: BufferGeometry, border_tolerance_px: float = 0.5) -> HoleBufferRelation:
    if buffer.state == 'unknown' or buffer.center_xy_px is None or buffer.radius_px is None or buffer.radius_px <= 0:
        return HoleBufferRelation(
            node_id=int(hole.node_id),
            relation='border_unknown',
            partial_visibility=hole.visible_fraction < 0.999,
            exclusion_reason='',
            notes='Buffer border unavailable; no border-based exclusion applied.',
        )
    hx, hy = float(hole.center_xy_px[0]), float(hole.center_xy_px[1])
    bx, by = float(buffer.center_xy_px[0]), float(buffer.center_xy_px[1])
    hr = float(hole.radius_px)
    br = float(buffer.radius_px)
    d = ((hx - bx) ** 2 + (hy - by) ** 2) ** 0.5
    if d + hr <= br - border_tolerance_px:
        rel = 'inside_buffer'
        reason = ''
    elif d - hr >= br + border_tolerance_px:
        rel = 'outside_buffer'
        reason = 'outside_buffer'
    else:
        rel = 'intersects_buffer_border'
        reason = 'intersects_buffer_border'
    return HoleBufferRelation(
        node_id=int(hole.node_id),
        relation=rel,
        partial_visibility=hole.visible_fraction < 0.999 or buffer.state == 'partial',
        exclusion_reason=reason,
        notes=f'distance_to_buffer_center_px={d:.3f}',
    )


def classify_holes_against_buffer(holes: Iterable[HoleTierRecord], buffer: BufferGeometry, border_tolerance_px: float = 0.5) -> list[HoleBufferRelation]:
    return [classify_hole_against_buffer(h, buffer, border_tolerance_px=border_tolerance_px) for h in holes]


def select_holes_for_analysis(
    holes: Iterable[HoleTierRecord],
    relations: Iterable[HoleBufferRelation],
    include_tiers: tuple[int, ...] = (1, 2, 3),
    exclude_border_intersections_when_known: bool = True,
) -> tuple[list[HoleTierRecord], dict[str, object]]:
    hole_list = list(holes)
    relation_map = {int(r.node_id): r for r in relations}
    selected: list[HoleTierRecord] = []
    excluded: list[dict[str, object]] = []
    for hole in hole_list:
        if int(hole.tier) not in include_tiers:
            excluded.append({'node_id': int(hole.node_id), 'reason': 'tier_excluded'})
            continue
        rel = relation_map.get(int(hole.node_id))
        if rel is None:
            selected.append(hole)
            continue
        if exclude_border_intersections_when_known and rel.relation in {'intersects_buffer_border', 'outside_buffer'}:
            excluded.append({'node_id': int(hole.node_id), 'reason': rel.exclusion_reason or rel.relation})
            continue
        selected.append(hole)
    fallback_no_selection = False
    if not selected and hole_list:
        # honest guard: if border geometry would exclude every hole, preserve the hole set and flag the fallback
        fallback_no_selection = True
        excluded = []
        selected = [h for h in hole_list if int(h.tier) in include_tiers]
    policy = {
        'include_tiers': list(include_tiers),
        'exclude_border_intersections_when_known': bool(exclude_border_intersections_when_known),
        'selected_count': int(len(selected)),
        'excluded': excluded,
        'fallback_no_selection_guard': bool(fallback_no_selection),
    }
    return selected, policy
