from __future__ import annotations

from pathlib import Path


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def ensure_subdirs(root: Path, *parts: str) -> list[Path]:
    out: list[Path] = []
    for part in parts:
        out.append(ensure_dir(root / part))
    return out
