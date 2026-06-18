from __future__ import annotations

import pandas as pd


def build_entity_trajectories(table: pd.DataFrame, entity_col: str, value_col: str) -> dict[str, list[float]]:
    out = {}
    for entity, sub in table.groupby(entity_col):
        out[str(entity)] = sub.sort_values('frame_id')[value_col].tolist()
    return out
