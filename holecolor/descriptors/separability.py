from __future__ import annotations
import numpy as np

def separability_score(values_a: np.ndarray, values_b: np.ndarray) -> float:
    ma, mb = np.nanmean(values_a), np.nanmean(values_b)
    va, vb = np.nanvar(values_a), np.nanvar(values_b)
    return float((ma - mb) ** 2 / max(va + vb, 1e-6))

def rank_descriptors(named_arrays: dict[str, tuple[np.ndarray, np.ndarray]]) -> list[tuple[float, str]]:
    ranked=[]
    for name,(a,b) in named_arrays.items():
        ranked.append((separability_score(np.asarray(a,float), np.asarray(b,float)), name))
    ranked.sort(reverse=True)
    return ranked

def choose_primary_descriptor(named_arrays: dict[str, tuple[np.ndarray, np.ndarray]]) -> str:
    ranked=rank_descriptors(named_arrays)
    if not ranked: raise ValueError("named_arrays cannot be empty")
    return ranked[0][1]
