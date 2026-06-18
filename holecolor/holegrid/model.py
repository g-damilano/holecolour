from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
import json

from holecolor.core.types import HoleGeometry, LatticeModel


@dataclass(slots=True)
class HoleGridBundle:
    version: str
    support_mode: str
    source_frame_id: int
    source_note: str
    support_circle: tuple[int, int, int] | None
    lattice: LatticeModel
    holes: list[HoleGeometry]


def hole_grid_bundle_to_dict(bundle: HoleGridBundle) -> dict[str, Any]:
    return {
        'version': bundle.version,
        'support_mode': bundle.support_mode,
        'source_frame_id': bundle.source_frame_id,
        'source_note': bundle.source_note,
        'support_circle': bundle.support_circle,
        'lattice': asdict(bundle.lattice),
        'holes': [asdict(h) for h in bundle.holes],
    }


def hole_grid_bundle_from_dict(data: dict[str, Any]) -> HoleGridBundle:
    lattice = LatticeModel(**data['lattice'])
    holes = [HoleGeometry(**row) for row in data['holes']]
    return HoleGridBundle(
        version=str(data.get('version', '1.0')),
        support_mode=str(data.get('support_mode', 'unknown')),
        source_frame_id=int(data.get('source_frame_id', 0)),
        source_note=str(data.get('source_note', '')),
        support_circle=tuple(data['support_circle']) if data.get('support_circle') else None,
        lattice=lattice,
        holes=holes,
    )


def save_hole_grid_model(path: str | Path, bundle: HoleGridBundle) -> None:
    Path(path).write_text(json.dumps(hole_grid_bundle_to_dict(bundle), indent=2), encoding='utf-8')


def load_hole_grid_model(path: str | Path) -> HoleGridBundle:
    data = json.loads(Path(path).read_text(encoding='utf-8'))
    return hole_grid_bundle_from_dict(data)
