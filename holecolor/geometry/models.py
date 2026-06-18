from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class WaferGeometry:
    id: str
    center_xy_px: tuple[float, float]
    radius_px: float
    confidence: float
    visible_arc_intervals_deg: list[tuple[float, float]] = field(default_factory=list)
    detection_mode: str = "unknown"
    notes: str = ""

    def to_json(self) -> dict[str, Any]:
        data = asdict(self)
        data["center_xy_px"] = [float(self.center_xy_px[0]), float(self.center_xy_px[1])]
        data["visible_arc_intervals_deg"] = [
            [float(a0), float(a1)] for a0, a1 in self.visible_arc_intervals_deg
        ]
        return data


@dataclass(slots=True)
class BufferGeometry:
    id: str
    state: str
    center_xy_px: tuple[float, float] | None
    radius_px: float | None
    confidence: float
    visible_arc_intervals_deg: list[tuple[float, float]] = field(default_factory=list)
    center_outside_frame: bool = False
    detection_mode: str = "unknown"
    notes: str = ""

    def to_json(self) -> dict[str, Any]:
        data = asdict(self)
        data["center_xy_px"] = None if self.center_xy_px is None else [float(self.center_xy_px[0]), float(self.center_xy_px[1])]
        data["visible_arc_intervals_deg"] = [
            [float(a0), float(a1)] for a0, a1 in self.visible_arc_intervals_deg
        ]
        return data


@dataclass(slots=True)
class HoleTierRecord:
    node_id: int
    center_xy_px: tuple[float, float]
    radius_px: float
    confidence: float
    lattice_i: int | None
    lattice_j: int | None
    tier: int
    source_class: str
    visible_fraction: float = 1.0
    border_refined: bool = False
    notes: str = ""

    def to_json(self) -> dict[str, Any]:
        data = asdict(self)
        data["center_xy_px"] = [float(self.center_xy_px[0]), float(self.center_xy_px[1])]
        return data


@dataclass(slots=True)
class HoleBufferRelation:
    node_id: int
    relation: str
    partial_visibility: bool = False
    exclusion_reason: str = ""
    notes: str = ""

    def to_json(self) -> dict[str, Any]:
        return asdict(self)
