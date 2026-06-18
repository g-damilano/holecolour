from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import imageio.v3 as iio
import matplotlib.pyplot as plt
import pandas as pd

from holecolor.core.paths import ensure_dir
from holecolor.qc.gates import GateResult


def _normalize_records(records: Sequence[Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for record in records:
        if is_dataclass(record):
            out.append(asdict(record))
        elif isinstance(record, dict):
            out.append(record)
        else:
            out.append(vars(record))
    return out


def write_table(path: Path, records: Sequence[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(_normalize_records(records))
    df.to_csv(path, index=False)


def write_table_columns(path: Path, columns: dict[str, Sequence[Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(columns)
    df.to_csv(path, index=False)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_gate_report(path: Path, gates: list[GateResult]) -> None:
    payload = [{"name": g.name, "passed": g.passed, "detail": g.detail} for g in gates]
    write_json(path, payload)


def save_image(path: Path, image) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    iio.imwrite(path, image)


def save_line_plot(
    path: Path,
    x: Sequence[float],
    series: Sequence[tuple[str, Sequence[float]]],
    title: str,
    ylabel: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    for label, values in series:
        ax.plot(x, values, label=label)
    ax.set_title(title)
    ax.set_xlabel("frame")
    ax.set_ylabel(ylabel)
    if len(series) > 1:
        ax.legend(loc="best")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_bar_plot(path: Path, labels: Sequence[str], values: Sequence[float], title: str, ylabel: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    ax.bar(labels, values)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)



def save_heatmap_plot(path: Path, matrix, title: str, xlabel: str, ylabel: str, xticklabels=None, yticklabels=None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    im = ax.imshow(matrix, aspect="auto", interpolation="nearest")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if xticklabels is not None:
        ax.set_xticks(range(len(xticklabels)))
        ax.set_xticklabels(xticklabels)
    if yticklabels is not None:
        ax.set_yticks(range(len(yticklabels)))
        ax.set_yticklabels(yticklabels)
    fig.colorbar(im, ax=ax, shrink=0.85)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
