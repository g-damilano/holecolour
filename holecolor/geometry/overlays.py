from __future__ import annotations

import cv2
import numpy as np

from holecolor.core.types import HoleCandidate, LatticeModel


def draw_candidates(
    image: np.ndarray,
    candidates: list[HoleCandidate],
    lattice: LatticeModel | None = None,
    lattice_indices: dict[int, tuple[int, int]] | None = None,
    support_circle: tuple[float, float, float] | None = None,
) -> np.ndarray:
    overlay = image.copy()
    if support_circle is not None:
        xw, yw, rw = support_circle
        if rw > 0:
            cv2.circle(
                overlay,
                (int(round(xw)), int(round(yw))),
                int(round(rw)),
                (255, 0, 255),
                2,
            )
    for i, cand in enumerate(candidates):
        center = (int(round(cand.x)), int(round(cand.y)))
        radius = max(1, int(round(cand.radius_px)))
        cv2.circle(overlay, center, radius, (255, 200, 0), 2)
        cv2.circle(overlay, center, 2, (255, 0, 0), -1)
        label = f"{i}"
        if lattice_indices and i in lattice_indices:
            u, v = lattice_indices[i]
            label = f"{i}:{u},{v}"
        cv2.putText(overlay, label, (center[0] + 4, center[1] - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1, cv2.LINE_AA)
    if lattice is not None and lattice.spacing_u_px > 0 and lattice.spacing_v_px > 0:
        origin = np.array([lattice.origin_x, lattice.origin_y], dtype=np.float32)
        u = np.array(lattice.basis_u, dtype=np.float32)
        v = np.array(lattice.basis_v, dtype=np.float32)
        p_u = tuple(np.round(origin + u).astype(int))
        p_v = tuple(np.round(origin + v).astype(int))
        p0 = tuple(np.round(origin).astype(int))
        cv2.arrowedLine(overlay, p0, p_u, (0, 255, 255), 2, tipLength=0.12)
        cv2.arrowedLine(overlay, p0, p_v, (255, 0, 255), 2, tipLength=0.12)
        cv2.putText(overlay, "u", (p_u[0] + 4, p_u[1] + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(overlay, "v", (p_v[0] + 4, p_v[1] + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 255), 1, cv2.LINE_AA)
    return overlay
