from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, Callable

import imageio.v3 as iio
import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle, Wedge
import numpy as np
import pandas as pd

from holecolor.core.logging import get_logger
from holecolor.qc.reports import write_json

LOG = get_logger(__name__)


OPTION_OUTPUTS = {
    "raw": "option_01_cluster_rdf_raw_fraction_stack.png",
    "active": "option_02_baseline_corrected_active_rdf.png",
    "dominant": "option_03_dominant_cluster_phase_map.png",
    "profiles": "option_04_radial_profile_snapshots.png",
    "rings": "option_05_average_hole_ring_snapshots.png",
    "sector": "option_06_sector_fan_active_cluster.png",
    "barcode": "option_08_hole_response_barcode.png",
    "front": "option_09_activity_weighted_front_trajectory.png",
    "legend": "cluster_colour_legend.png",
    "montage": "00_holecolor_visualisation_montage.png",
}


@dataclass(slots=True)
class ClusterRdfVisualisationData:
    run_dir: Path
    out_dir: Path
    tensor_df: pd.DataFrame
    activity_df: pd.DataFrame
    sector_df: pd.DataFrame
    hole_response_df: pd.DataFrame
    front_df: pd.DataFrame
    cluster_summary_df: pd.DataFrame
    measured_palette_df: pd.DataFrame
    phenotype_df: pd.DataFrame
    event_df: pd.DataFrame
    baseline_frames: list[int]


def _cluster_cfg(config: Any) -> Any:
    cfg = config
    if cfg is not None and hasattr(cfg, "visualisation"):
        cfg = getattr(cfg, "visualisation")
    if cfg is not None and hasattr(cfg, "cluster_rdf"):
        cfg = getattr(cfg, "cluster_rdf")
    return cfg


def _cfg_value(config: Any, name: str, default: Any) -> Any:
    cfg = _cluster_cfg(config)
    if isinstance(cfg, dict):
        return cfg.get(name, default)
    if cfg is not None and hasattr(cfg, name):
        return getattr(cfg, name)
    return default


def _resolve_output_dir(run_dir: str | Path) -> tuple[Path, Path]:
    root = Path(run_dir)
    direct_tensor = root / "hole_terrace_sector_cluster_tensor.csv"
    if direct_tensor.exists():
        return root, root
    return root, root / "descriptors" / "radial_cluster_average_hole"


def _read_csv_if_exists(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _normalise_tensor_df(tensor_df: pd.DataFrame) -> pd.DataFrame:
    df = tensor_df.copy()
    if "frame" not in df.columns and "frame_id" in df.columns:
        df["frame"] = df["frame_id"]
    if "time_s" not in df.columns:
        df["time_s"] = df["frame"]
    if "pixel_count" not in df.columns:
        if {"pixel_fraction", "n_valid_pixels"}.issubset(df.columns):
            df["pixel_count"] = pd.to_numeric(df["pixel_fraction"], errors="coerce").fillna(0.0) * pd.to_numeric(
                df["n_valid_pixels"], errors="coerce"
            ).fillna(0.0)
        elif "cluster_count" in df.columns:
            df["pixel_count"] = df["cluster_count"]
        else:
            df["pixel_count"] = 1.0
    required = {"frame", "terrace_index", "cluster_id", "pixel_count"}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"cluster RDF tensor is missing required columns: {missing}")
    for col in ("frame", "terrace_index", "cluster_id"):
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
    df = df.dropna(subset=["frame", "terrace_index", "cluster_id"]).copy()
    for col in ("frame", "terrace_index", "cluster_id"):
        df[col] = df[col].astype(int)
    df["pixel_count"] = pd.to_numeric(df["pixel_count"], errors="coerce").fillna(0.0).clip(lower=0.0)
    df["time_s"] = pd.to_numeric(df["time_s"], errors="coerce")
    if "hole_id" in df.columns:
        df["hole_id"] = pd.to_numeric(df["hole_id"], errors="coerce").astype("Int64")
    if "sector_id" in df.columns:
        df["sector_id"] = pd.to_numeric(df["sector_id"], errors="coerce").astype("Int64")
    return df


def _baseline_frame_ids(frames: list[int], baseline_frames: int) -> list[int]:
    if not frames:
        return []
    n = max(1, min(int(baseline_frames), len(frames)))
    return [int(v) for v in frames[:n]]


def _complete_grid(frames: list[int], terraces: list[int], clusters: list[int], times: dict[int, float]) -> pd.DataFrame:
    grid = pd.MultiIndex.from_product([frames, terraces, clusters], names=["frame", "terrace_index", "cluster_id"]).to_frame(index=False)
    grid["time_s"] = grid["frame"].map(times)
    return grid


def build_radial_cluster_activity_table(
    tensor_df: pd.DataFrame,
    baseline_frames: int = 3,
    terrace_display_base: int = 1,
) -> pd.DataFrame:
    df = _normalise_tensor_df(tensor_df)
    if df.empty:
        return pd.DataFrame()
    frames = sorted(int(v) for v in df["frame"].unique())
    terraces = sorted(int(v) for v in df["terrace_index"].unique())
    clusters = sorted(int(v) for v in df["cluster_id"].unique())
    time_rows = df[["frame", "time_s"]].dropna(subset=["time_s"]).drop_duplicates("frame")
    times = {int(r.frame): float(r.time_s) for r in time_rows.itertuples()}
    for frame in frames:
        times.setdefault(int(frame), float(frame))

    group_keys = ["frame", "terrace_index", "cluster_id"]
    counts = df.groupby(group_keys, as_index=False)["pixel_count"].sum().rename(columns={"pixel_count": "cluster_count"})
    totals = counts.groupby(["frame", "terrace_index"], as_index=False)["cluster_count"].sum().rename(columns={"cluster_count": "n_pixels"})
    if "hole_id" in df.columns:
        holes = df.groupby(["frame", "terrace_index"], as_index=False)["hole_id"].nunique().rename(columns={"hole_id": "n_holes"})
    else:
        holes = totals[["frame", "terrace_index"]].copy()
        holes["n_holes"] = np.nan

    out = _complete_grid(frames, terraces, clusters, times)
    out = out.merge(counts, on=group_keys, how="left")
    out = out.merge(totals, on=["frame", "terrace_index"], how="left")
    out = out.merge(holes, on=["frame", "terrace_index"], how="left")
    out["cluster_count"] = out["cluster_count"].fillna(0.0)
    out["n_pixels"] = out["n_pixels"].fillna(0.0)
    out["n_samples"] = out["n_pixels"]
    out["raw_fraction"] = np.where(out["n_pixels"] > 0, out["cluster_count"] / out["n_pixels"], np.nan)
    base_ids = _baseline_frame_ids(frames, baseline_frames)
    baseline = (
        out[out["frame"].isin(base_ids)]
        .groupby(["terrace_index", "cluster_id"], as_index=False)["raw_fraction"]
        .mean()
        .rename(columns={"raw_fraction": "baseline_fraction"})
    )
    out = out.merge(baseline, on=["terrace_index", "cluster_id"], how="left")
    out["active_fraction"] = np.maximum(0.0, out["raw_fraction"] - out["baseline_fraction"])
    active_total = out.groupby(["frame", "cluster_id"], as_index=False)["active_fraction"].sum(min_count=1).rename(
        columns={"active_fraction": "active_cluster_total"}
    )
    out = out.merge(active_total, on=["frame", "cluster_id"], how="left")
    out["normalized_active_fraction"] = np.where(
        out["active_cluster_total"] > 0,
        out["active_fraction"] / out["active_cluster_total"],
        np.nan,
    )
    out["terrace_label"] = out["terrace_index"].map(lambda v: f"T{int(v) + int(terrace_display_base)}")
    return out[
        [
            "frame",
            "time_s",
            "terrace_index",
            "terrace_label",
            "cluster_id",
            "raw_fraction",
            "baseline_fraction",
            "active_fraction",
            "normalized_active_fraction",
            "n_samples",
            "n_holes",
            "n_pixels",
            "cluster_count",
        ]
    ].sort_values(["frame", "terrace_index", "cluster_id"]).reset_index(drop=True)


def build_sector_cluster_activity_table(tensor_df: pd.DataFrame, baseline_frames: int = 3, terrace_display_base: int = 1) -> pd.DataFrame:
    df = _normalise_tensor_df(tensor_df)
    if df.empty or "sector_id" not in df.columns:
        return pd.DataFrame()
    df = df.dropna(subset=["sector_id"]).copy()
    if df.empty:
        return pd.DataFrame()
    df["sector_id"] = df["sector_id"].astype(int)
    frames = sorted(int(v) for v in df["frame"].unique())
    sectors = sorted(int(v) for v in df["sector_id"].unique())
    clusters = sorted(int(v) for v in df["cluster_id"].unique())
    times = {int(r.frame): float(r.time_s) for r in df[["frame", "time_s"]].drop_duplicates("frame").itertuples()}
    for frame in frames:
        times.setdefault(int(frame), float(frame))
    terraces = sorted(int(v) for v in df["terrace_index"].unique())
    grid = pd.MultiIndex.from_product([frames, terraces, sectors, clusters], names=["frame", "terrace_index", "sector_id", "cluster_id"]).to_frame(index=False)
    grid["time_s"] = grid["frame"].map(times)
    counts = df.groupby(["frame", "terrace_index", "sector_id", "cluster_id"], as_index=False)["pixel_count"].sum().rename(
        columns={"pixel_count": "cluster_count"}
    )
    totals = counts.groupby(["frame", "terrace_index", "sector_id"], as_index=False)["cluster_count"].sum().rename(columns={"cluster_count": "n_pixels"})
    out = grid.merge(counts, on=["frame", "terrace_index", "sector_id", "cluster_id"], how="left").merge(totals, on=["frame", "terrace_index", "sector_id"], how="left")
    out["cluster_count"] = out["cluster_count"].fillna(0.0)
    out["n_pixels"] = out["n_pixels"].fillna(0.0)
    out["raw_fraction"] = np.where(out["n_pixels"] > 0, out["cluster_count"] / out["n_pixels"], np.nan)
    base_ids = _baseline_frame_ids(frames, baseline_frames)
    baseline = (
        out[out["frame"].isin(base_ids)]
        .groupby(["terrace_index", "sector_id", "cluster_id"], as_index=False)["raw_fraction"]
        .mean()
        .rename(columns={"raw_fraction": "baseline_fraction"})
    )
    out = out.merge(baseline, on=["terrace_index", "sector_id", "cluster_id"], how="left")
    out["active_fraction"] = np.maximum(0.0, out["raw_fraction"] - out["baseline_fraction"])
    out["terrace_label"] = out["terrace_index"].map(lambda v: f"T{int(v) + int(terrace_display_base)}")
    return out.sort_values(["frame", "sector_id", "cluster_id"]).reset_index(drop=True)


def build_hole_response_table(tensor_df: pd.DataFrame, baseline_frames: int = 3) -> pd.DataFrame:
    df = _normalise_tensor_df(tensor_df)
    if df.empty or "hole_id" not in df.columns:
        return pd.DataFrame()
    df = df.dropna(subset=["hole_id"]).copy()
    if df.empty:
        return pd.DataFrame()
    df["hole_id"] = df["hole_id"].astype(int)
    frames = sorted(int(v) for v in df["frame"].unique())
    base_ids = _baseline_frame_ids(frames, baseline_frames)
    group_keys = ["frame", "hole_id", "terrace_index", "cluster_id"]
    counts = df.groupby(group_keys, as_index=False)["pixel_count"].sum().rename(columns={"pixel_count": "cluster_count"})
    totals = counts.groupby(["frame", "hole_id", "terrace_index"], as_index=False)["cluster_count"].sum().rename(
        columns={"cluster_count": "n_pixels"}
    )
    out = counts.merge(totals, on=["frame", "hole_id", "terrace_index"], how="left")
    out["raw_fraction"] = np.where(out["n_pixels"] > 0, out["cluster_count"] / out["n_pixels"], np.nan)
    baseline = (
        out[out["frame"].isin(base_ids)]
        .groupby(["hole_id", "terrace_index", "cluster_id"], as_index=False)["raw_fraction"]
        .mean()
        .rename(columns={"raw_fraction": "baseline_fraction"})
    )
    out = out.merge(baseline, on=["hole_id", "terrace_index", "cluster_id"], how="left")
    out["active_fraction"] = np.maximum(0.0, out["raw_fraction"] - out["baseline_fraction"])
    response = out.groupby(["frame", "hole_id"], as_index=False)["active_fraction"].sum(min_count=1).rename(
        columns={"active_fraction": "response"}
    )
    times = df[["frame", "time_s"]].drop_duplicates("frame")
    response = response.merge(times, on="frame", how="left")
    return response.sort_values(["hole_id", "frame"]).reset_index(drop=True)


def compute_activity_weighted_front(activity_df: pd.DataFrame, terrace_display_base: int = 1) -> pd.DataFrame:
    if activity_df.empty:
        return pd.DataFrame(columns=["frame", "time_s", "cluster_id", "front_terrace", "spread_terrace", "total_activity"])
    rows: list[dict[str, Any]] = []
    for (frame, cluster_id), grp in activity_df.groupby(["frame", "cluster_id"], sort=True):
        weights = pd.to_numeric(grp["active_fraction"], errors="coerce").to_numpy(dtype=float)
        terrace_pos = (pd.to_numeric(grp["terrace_index"], errors="coerce").to_numpy(dtype=float) + float(terrace_display_base))
        finite = np.isfinite(weights) & np.isfinite(terrace_pos)
        weights = weights[finite]
        terrace_pos = terrace_pos[finite]
        total = float(np.sum(weights)) if weights.size else 0.0
        if total > 1e-12:
            front = float(np.sum(terrace_pos * weights) / total)
            spread = float(np.sqrt(np.sum(weights * (terrace_pos - front) ** 2) / total))
        else:
            front = float("nan")
            spread = float("nan")
        time_vals = grp["time_s"].dropna()
        rows.append(
            {
                "frame": int(frame),
                "time_s": None if time_vals.empty else float(time_vals.iloc[0]),
                "cluster_id": int(cluster_id),
                "front_terrace": front,
                "spread_terrace": spread,
                "total_activity": total,
            }
        )
    return pd.DataFrame(rows).sort_values(["cluster_id", "frame"]).reset_index(drop=True)


def _cluster_ids(activity_df: pd.DataFrame, cluster_summary_df: pd.DataFrame | None = None) -> list[int]:
    ids: set[int] = set()
    if not activity_df.empty and "cluster_id" in activity_df.columns:
        ids.update(int(v) for v in activity_df["cluster_id"].dropna().unique())
    if cluster_summary_df is not None and not cluster_summary_df.empty and "cluster_id" in cluster_summary_df.columns:
        ids.update(int(v) for v in cluster_summary_df["cluster_id"].dropna().unique())
    return sorted(ids)


def _stable_cluster_palette(cluster_ids: list[int]) -> dict[int, tuple[float, float, float, float]]:
    cmap = plt.get_cmap("tab10" if len(cluster_ids) <= 10 else "tab20")
    return {int(cid): cmap(i % cmap.N) for i, cid in enumerate(cluster_ids)}


def _rgba(rgb: tuple[float, float, float] | tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    if len(rgb) == 4:
        return float(rgb[0]), float(rgb[1]), float(rgb[2]), float(rgb[3])
    return float(rgb[0]), float(rgb[1]), float(rgb[2]), 1.0


def _cluster_intensity_cmap(cluster_id: int, palette: dict[int, Any] | None) -> LinearSegmentedColormap:
    color = _rgba((palette or {}).get(int(cluster_id), (0.25, 0.25, 0.25, 1.0)))
    return LinearSegmentedColormap.from_list(f"cluster_{int(cluster_id)}_intensity", [(1.0, 1.0, 1.0, 1.0), color])


_MARKERS = ("o", "s", "^", "D", "v", "P", "X", "*", "h", "<", ">")
_LINESTYLES = ("-", "--", "-.", ":")


def _cluster_marker_map(cluster_ids: list[int]) -> dict[int, str]:
    return {int(cid): _MARKERS[i % len(_MARKERS)] for i, cid in enumerate(cluster_ids)}


def _cluster_linestyle_map(cluster_ids: list[int]) -> dict[int, str]:
    return {int(cid): _LINESTYLES[(i // len(_MARKERS)) % len(_LINESTYLES)] for i, cid in enumerate(cluster_ids)}


def _add_cluster_legend(
    fig,
    cluster_ids: list[int],
    palette: dict[int, Any],
    title: str = "global colour clusters",
    marker_map: dict[int, str] | None = None,
    linestyle_map: dict[int, str] | None = None,
) -> None:
    if not cluster_ids:
        return
    if marker_map is not None:
        handles = [
            Line2D(
                [0],
                [0],
                color=palette.get(cid, "#999999"),
                marker=marker_map.get(cid, "o"),
                linestyle=(linestyle_map or {}).get(cid, "-"),
                linewidth=1.6,
                markersize=5.2,
                label=f"cluster {cid}",
            )
            for cid in cluster_ids
        ]
    else:
        handles = [Patch(facecolor=palette.get(cid, "#999999"), edgecolor="black", label=f"cluster {cid}") for cid in cluster_ids]
    fig.legend(handles=handles, title=title, loc="center left", bbox_to_anchor=(1.005, 0.5), frameon=False)


def _hsl_to_rgb_tuple(h: float, s: float, l: float) -> tuple[float, float, float]:
    h = float(h) % 1.0
    s = float(np.clip(s, 0.0, 1.0))
    l = float(np.clip(l, 0.0, 1.0))
    c = (1.0 - abs(2.0 * l - 1.0)) * s
    hp = h * 6.0
    x = c * (1.0 - abs((hp % 2.0) - 1.0))
    if 0 <= hp < 1:
        r1, g1, b1 = c, x, 0.0
    elif 1 <= hp < 2:
        r1, g1, b1 = x, c, 0.0
    elif 2 <= hp < 3:
        r1, g1, b1 = 0.0, c, x
    elif 3 <= hp < 4:
        r1, g1, b1 = 0.0, x, c
    elif 4 <= hp < 5:
        r1, g1, b1 = x, 0.0, c
    else:
        r1, g1, b1 = c, 0.0, x
    m = l - 0.5 * c
    return float(r1 + m), float(g1 + m), float(b1 + m)


def _measured_cluster_colours(cluster_summary_df: pd.DataFrame, measured_palette_df: pd.DataFrame) -> dict[int, tuple[float, float, float]]:
    out: dict[int, tuple[float, float, float]] = {}
    if not measured_palette_df.empty and {"cluster_id", "display_r", "display_g", "display_b"}.issubset(measured_palette_df.columns):
        for row in measured_palette_df.itertuples(index=False):
            cid = int(getattr(row, "cluster_id"))
            out[cid] = (
                float(getattr(row, "display_r")) / 255.0,
                float(getattr(row, "display_g")) / 255.0,
                float(getattr(row, "display_b")) / 255.0,
            )
    if not cluster_summary_df.empty and {"cluster_id", "center_h", "center_s", "center_l"}.issubset(cluster_summary_df.columns):
        for row in cluster_summary_df.itertuples(index=False):
            cid = int(getattr(row, "cluster_id"))
            out.setdefault(
                cid,
                _hsl_to_rgb_tuple(float(getattr(row, "center_h")), float(getattr(row, "center_s")), float(getattr(row, "center_l"))),
            )
    return out


def _cluster_colour_palette(
    cluster_ids: list[int],
    cluster_summary_df: pd.DataFrame,
    measured_palette_df: pd.DataFrame,
) -> dict[int, tuple[float, float, float, float]]:
    fallback = _stable_cluster_palette(cluster_ids)
    measured = _measured_cluster_colours(cluster_summary_df, measured_palette_df)
    out: dict[int, tuple[float, float, float, float]] = {}
    for cid in cluster_ids:
        out[int(cid)] = _rgba(measured.get(int(cid), fallback[int(cid)]))
    return out


def _time_axis_mode(df: pd.DataFrame, style: Any = None) -> str:
    mode = str(_cfg_value(style, "time_axis", "time_s"))
    if mode not in {"frame", "time_s"}:
        mode = "time_s"
    if mode == "time_s" and "time_s" not in df.columns:
        mode = "frame"
    return mode


def _frame_axis(activity_df: pd.DataFrame, style: Any = None) -> tuple[list[int], list[float], str]:
    frames = sorted(int(v) for v in activity_df["frame"].dropna().unique())
    if not frames:
        return [], [], "frame"
    mode = _time_axis_mode(activity_df, style)
    time_lookup = activity_df[["frame", "time_s"]].dropna().drop_duplicates("frame") if "time_s" in activity_df.columns else pd.DataFrame()
    times = {int(r.frame): float(r.time_s) for r in time_lookup.itertuples()}
    use_time = mode == "time_s" and len(times) == len(frames)
    values = [times.get(frame, float(frame)) if use_time else float(frame) for frame in frames]
    return frames, values, "time (s)" if use_time else "frame"


def _set_time_ticks(ax, values: list[float]) -> None:
    if not values:
        return
    n = len(values)
    idx = np.linspace(0, n - 1, min(6, n), dtype=int)
    ax.set_xticks(idx)
    ax.set_xticklabels([f"{values[i]:.1f}" if abs(values[i] - round(values[i])) > 1e-6 else str(int(round(values[i]))) for i in idx], rotation=0)


def _frame_title(df: pd.DataFrame, frame: int, style: Any = None) -> str:
    if _time_axis_mode(df, style) == "time_s" and "time_s" in df.columns:
        vals = df.loc[df["frame"] == int(frame), "time_s"].dropna()
        if not vals.empty:
            return f"t={float(vals.iloc[0]):.1f}s"
    return f"frame {int(frame)}"


def _terraces(activity_df: pd.DataFrame) -> tuple[list[int], list[str]]:
    terr = sorted(int(v) for v in activity_df["terrace_index"].dropna().unique())
    labels = []
    for t in terr:
        sub = activity_df[activity_df["terrace_index"] == t]
        labels.append(str(sub["terrace_label"].iloc[0]) if "terrace_label" in sub.columns and not sub.empty else f"T{t + 1}")
    return terr, labels


def _heatmap_matrix(activity_df: pd.DataFrame, cluster_id: int, value_col: str, frames: list[int], terraces: list[int]) -> np.ndarray:
    sub = activity_df[activity_df["cluster_id"] == int(cluster_id)]
    pivot = sub.pivot_table(index="terrace_index", columns="frame", values=value_col, aggfunc="mean")
    pivot = pivot.reindex(index=terraces, columns=frames)
    return pivot.to_numpy(dtype=float)


def _placeholder_figure(out_path: Path, title: str, message: str) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.axis("off")
    ax.text(0.5, 0.58, title, ha="center", va="center", fontsize=14, weight="bold")
    ax.text(0.5, 0.42, message, ha="center", va="center", fontsize=10)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_cluster_rdf_raw_fraction(
    activity_df: pd.DataFrame,
    out_path: Path,
    palette: dict[int, Any] | None = None,
    style: Any = None,
) -> Path:
    if activity_df.empty:
        return _placeholder_figure(out_path, "Cluster RDF: raw terrace fraction", "No cluster RDF activity rows available.")
    cluster_ids = _cluster_ids(activity_df)
    palette = palette or _stable_cluster_palette(cluster_ids)
    frames, x_values, xlabel = _frame_axis(activity_df, style)
    terraces, terrace_labels = _terraces(activity_df)
    ncols = min(3, max(1, len(cluster_ids)))
    nrows = int(math.ceil(len(cluster_ids) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 2.8 * nrows + 0.7), squeeze=False)
    im = None
    for ax, cid in zip(axes.ravel(), cluster_ids):
        mat = _heatmap_matrix(activity_df, cid, "raw_fraction", frames, terraces)
        im = ax.imshow(mat, aspect="auto", origin="lower", vmin=0.0, vmax=1.0, cmap=_cluster_intensity_cmap(cid, palette))
        ax.set_title(f"cluster {cid}")
        ax.set_yticks(range(len(terrace_labels)))
        ax.set_yticklabels(terrace_labels)
        _set_time_ticks(ax, x_values)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("terrace")
    for ax in axes.ravel()[len(cluster_ids) :]:
        ax.axis("off")
    if im is not None:
        fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.82, label="raw fraction")
    _add_cluster_legend(fig, cluster_ids, palette)
    fig.suptitle("Cluster RDF: raw terrace fraction", y=0.995)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_cluster_rdf_active(
    activity_df: pd.DataFrame,
    out_path: Path,
    palette: dict[int, Any] | None = None,
    style: Any = None,
) -> Path:
    if activity_df.empty:
        return _placeholder_figure(out_path, "Baseline-corrected cluster activity RDF", "No cluster RDF activity rows available.")
    cluster_ids = _cluster_ids(activity_df)
    palette = palette or _stable_cluster_palette(cluster_ids)
    frames, x_values, xlabel = _frame_axis(activity_df, style)
    terraces, terrace_labels = _terraces(activity_df)
    vmax = float(np.nanmax(activity_df["active_fraction"].to_numpy(dtype=float))) if np.isfinite(activity_df["active_fraction"]).any() else 1.0
    vmax = max(vmax, 1e-6)
    ncols = min(3, max(1, len(cluster_ids)))
    nrows = int(math.ceil(len(cluster_ids) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 2.8 * nrows + 0.9), squeeze=False)
    im = None
    for ax, cid in zip(axes.ravel(), cluster_ids):
        mat = _heatmap_matrix(activity_df, cid, "active_fraction", frames, terraces)
        im = ax.imshow(mat, aspect="auto", origin="lower", vmin=0.0, vmax=vmax, cmap=_cluster_intensity_cmap(cid, palette))
        ax.set_title(f"cluster {cid}")
        ax.set_yticks(range(len(terrace_labels)))
        ax.set_yticklabels(terrace_labels)
        _set_time_ticks(ax, x_values)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("terrace")
    for ax in axes.ravel()[len(cluster_ids) :]:
        ax.axis("off")
    if im is not None:
        fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.82, label="active fraction above baseline")
    _add_cluster_legend(fig, cluster_ids, palette)
    baseline_frames = sorted(int(v) for v in activity_df.loc[activity_df["baseline_fraction"].notna(), "frame"].unique())[:3]
    subtitle = f"Baseline-corrected cluster activity RDF | baseline: first {len(baseline_frames) or 3} valid frames"
    fig.suptitle(subtitle, y=0.995)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_dominant_cluster_chronogram(activity_df: pd.DataFrame, out_path: Path, palette: dict[int, Any] | None = None, style: Any = None) -> Path:
    if activity_df.empty:
        return _placeholder_figure(out_path, "Dominant cluster chronogram", "No cluster RDF activity rows available.")
    cluster_ids = _cluster_ids(activity_df)
    palette = palette or _stable_cluster_palette(cluster_ids)
    frames, x_values, xlabel = _frame_axis(activity_df, style)
    terraces, terrace_labels = _terraces(activity_df)
    cid_to_idx = {cid: i for i, cid in enumerate(cluster_ids)}
    mat = np.full((len(terraces), len(frames)), np.nan, dtype=float)
    for (frame, terrace), grp in activity_df.groupby(["frame", "terrace_index"], sort=True):
        vals = grp[["cluster_id", "raw_fraction"]].dropna()
        if vals.empty:
            continue
        cid = int(vals.sort_values("raw_fraction", ascending=False).iloc[0]["cluster_id"])
        mat[terraces.index(int(terrace)), frames.index(int(frame))] = float(cid_to_idx[cid])
    cmap = ListedColormap([palette[cid] for cid in cluster_ids])
    cmap.set_bad("#dddddd")
    fig, ax = plt.subplots(figsize=(10, 4.8))
    im = ax.imshow(np.ma.masked_invalid(mat), aspect="auto", origin="lower", cmap=cmap, vmin=-0.5, vmax=len(cluster_ids) - 0.5)
    ax.set_title("Dominant cluster chronogram")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("terrace")
    ax.set_yticks(range(len(terrace_labels)))
    ax.set_yticklabels(terrace_labels)
    _set_time_ticks(ax, x_values)
    _add_cluster_legend(fig, cluster_ids, palette)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _selected_snapshot_frames(activity_df: pd.DataFrame, baseline_frames: int = 3) -> list[int]:
    frames = sorted(int(v) for v in activity_df["frame"].dropna().unique())
    return frames


def plot_radial_profile_snapshots(
    activity_df: pd.DataFrame,
    out_path: Path,
    palette: dict[int, Any] | None = None,
    style: Any = None,
) -> Path:
    if activity_df.empty:
        return _placeholder_figure(out_path, "Radial cluster profiles at selected times", "No cluster RDF activity rows available.")
    cluster_ids = _cluster_ids(activity_df)
    palette = palette or _stable_cluster_palette(cluster_ids)
    marker_map = _cluster_marker_map(cluster_ids)
    linestyle_map = _cluster_linestyle_map(cluster_ids)
    selected = _selected_snapshot_frames(activity_df, int(_cfg_value(style, "baseline_frames", 3)))
    terraces, terrace_labels = _terraces(activity_df)
    ncols = min(4 if len(selected) > 12 else 5, len(selected))
    nrows = int(math.ceil(len(selected) / ncols))
    ymax = float(np.nanmax(pd.to_numeric(activity_df["active_fraction"], errors="coerce").to_numpy(dtype=float))) if np.isfinite(activity_df["active_fraction"]).any() else 1.0
    ymax = max(0.05, ymax * 1.08)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.35 * ncols + 1.8, 2.25 * nrows + 0.9), squeeze=False, sharex=True, sharey=True)
    for ax, frame in zip(axes.ravel(), selected):
        sub = activity_df[activity_df["frame"] == int(frame)]
        for cid in cluster_ids:
            csub = sub[sub["cluster_id"] == cid].set_index("terrace_index").reindex(terraces)
            ax.plot(
                range(len(terraces)),
                csub["active_fraction"].to_numpy(dtype=float),
                marker=marker_map[cid],
                linestyle=linestyle_map[cid],
                color=palette[cid],
                linewidth=1.15,
                markersize=3.2,
            )
        ax.set_title(_frame_title(activity_df, frame, style))
        ax.set_xticks(range(len(terraces)))
        ax.set_xticklabels(terrace_labels)
        ax.set_ylim(0, ymax)
        ax.tick_params(axis="both", labelsize=7)
        ax.label_outer()
    for ax in axes.ravel()[len(selected) :]:
        ax.axis("off")
    fig.suptitle("Radial cluster profiles at each timestep", y=0.995, fontsize=12)
    fig.supxlabel("terrace", fontsize=10)
    fig.supylabel("active fraction", fontsize=10)
    _add_cluster_legend(fig, cluster_ids, palette, marker_map=marker_map, linestyle_map=linestyle_map)
    fig.tight_layout(rect=[0.035, 0.035, 0.86, 0.975], h_pad=0.85, w_pad=0.45)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_average_hole_ring_snapshots(
    activity_df: pd.DataFrame,
    out_path: Path,
    palette: dict[int, Any] | None = None,
    style: Any = None,
    sector_df: pd.DataFrame | None = None,
) -> Path:
    if activity_df.empty:
        return _placeholder_figure(out_path, "Average-hole annular cluster snapshots", "No cluster RDF activity rows available.")
    cluster_ids = _cluster_ids(activity_df)
    palette = palette or _stable_cluster_palette(cluster_ids)
    selected = _selected_snapshot_frames(activity_df, int(_cfg_value(style, "baseline_frames", 3)))
    terraces, _labels = _terraces(activity_df)
    sector_df = pd.DataFrame() if sector_df is None else sector_df
    has_sectors = not sector_df.empty and {"frame", "terrace_index", "sector_id", "cluster_id", "active_fraction", "raw_fraction"}.issubset(sector_df.columns)
    sectors = sorted(int(v) for v in sector_df["sector_id"].dropna().unique()) if has_sectors else []
    ncols = min(6, len(selected))
    nrows = int(math.ceil(len(selected) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.45 * ncols + 1.7, 2.65 * nrows + 0.5), squeeze=False)
    for ax, frame in zip(axes.ravel(), selected):
        sub = activity_df[activity_df["frame"] == int(frame)]
        ax.set_aspect("equal")
        ax.axis("off")
        ax.add_patch(Wedge((0, 0), 1.0, 0, 360, facecolor="#1b1b1b", edgecolor="white", linewidth=0.8))
        if has_sectors and sectors:
            fsector = sector_df[sector_df["frame"] == int(frame)]
            angle_width = 360.0 / float(len(sectors))
            for i, terrace in enumerate(terraces):
                for j, sector in enumerate(sectors):
                    tsub = fsector[(fsector["terrace_index"] == terrace) & (fsector["sector_id"] == sector)]
                    if tsub.empty:
                        color = "#dddddd"
                        alpha = 0.25
                    elif float(pd.to_numeric(tsub["active_fraction"], errors="coerce").sum(skipna=True)) > 1e-12:
                        cid = int(tsub.sort_values("active_fraction", ascending=False).iloc[0]["cluster_id"])
                        color = palette[cid]
                        alpha = 0.95
                    else:
                        cid = int(tsub.sort_values("raw_fraction", ascending=False).iloc[0]["cluster_id"])
                        color = palette[cid]
                        alpha = 0.35
                    ax.add_patch(
                        Wedge(
                            (0, 0),
                            1.0 + i + 1.0,
                            j * angle_width,
                            (j + 1) * angle_width,
                            width=1.0,
                            facecolor=color,
                            edgecolor="white",
                            alpha=alpha,
                            linewidth=0.35,
                        )
                    )
        else:
            for i, terrace in enumerate(terraces):
                tsub = sub[sub["terrace_index"] == terrace]
                if tsub.empty:
                    color = "#dddddd"
                    alpha = 0.35
                elif float(tsub["active_fraction"].sum()) > 0:
                    cid = int(tsub.sort_values("active_fraction", ascending=False).iloc[0]["cluster_id"])
                    color = palette[cid]
                    alpha = 0.95
                else:
                    cid = int(tsub.sort_values("raw_fraction", ascending=False).iloc[0]["cluster_id"])
                    color = palette[cid]
                    alpha = 0.35
                ax.add_patch(Wedge((0, 0), 1.0 + i + 1.0, 0, 360, width=1.0, facecolor=color, edgecolor="white", alpha=alpha, linewidth=0.7))
        lim = len(terraces) + 1.3
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_title(_frame_title(activity_df, frame, style))
    for ax in axes.ravel()[len(selected) :]:
        ax.axis("off")
    _add_cluster_legend(fig, cluster_ids, palette)
    title = "Average-hole sector ring snapshots at each timestep" if has_sectors else "Average-hole annular cluster snapshots at each timestep"
    fig.suptitle(title, y=0.995)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_sector_fan_active_cluster(
    sector_df: pd.DataFrame,
    out_path: Path,
    palette: dict[int, Any] | None = None,
    style: Any = None,
) -> Path:
    if sector_df.empty:
        return _placeholder_figure(out_path, "Sector fan active-cluster visualisation", "No sector-resolved cluster activity rows available.")
    cluster_ids = sorted(int(v) for v in sector_df["cluster_id"].dropna().unique())
    palette = palette or _stable_cluster_palette(cluster_ids)
    fan_source = sector_df.copy()
    if "n_pixels" in fan_source.columns:
        fan_source["_active_weight"] = pd.to_numeric(fan_source["active_fraction"], errors="coerce").fillna(0.0) * pd.to_numeric(
            fan_source["n_pixels"], errors="coerce"
        ).fillna(0.0)
        fan_df = (
            fan_source.groupby(["frame", "time_s", "sector_id", "cluster_id"], as_index=False)
            .agg(active_weight=("_active_weight", "sum"), n_pixels=("n_pixels", "sum"))
            .sort_values(["frame", "sector_id", "cluster_id"])
        )
        fan_df["active_fraction"] = np.where(fan_df["n_pixels"] > 0, fan_df["active_weight"] / fan_df["n_pixels"], np.nan)
    else:
        fan_df = (
            fan_source.groupby(["frame", "time_s", "sector_id", "cluster_id"], as_index=False)["active_fraction"]
            .mean()
            .sort_values(["frame", "sector_id", "cluster_id"])
        )
    frames, x_values, xlabel = _frame_axis(fan_df, style)
    sectors = sorted(int(v) for v in sector_df["sector_id"].dropna().unique())
    ncols = min(3, len(cluster_ids))
    nrows = int(math.ceil(len(cluster_ids) / ncols))
    vmax = float(np.nanmax(fan_df["active_fraction"].to_numpy(dtype=float))) if np.isfinite(fan_df["active_fraction"]).any() else 1.0
    vmax = max(vmax, 1e-6)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.4 * ncols, 2.8 * nrows + 0.7), squeeze=False)
    im = None
    for ax, cid in zip(axes.ravel(), cluster_ids):
        sub = fan_df[fan_df["cluster_id"] == cid]
        mat = sub.pivot_table(index="sector_id", columns="frame", values="active_fraction", aggfunc="mean").reindex(index=sectors, columns=frames).to_numpy(dtype=float)
        im = ax.imshow(mat, aspect="auto", origin="lower", vmin=0.0, vmax=vmax, cmap=_cluster_intensity_cmap(cid, palette))
        ax.set_title(f"cluster {cid}")
        ax.set_yticks(range(len(sectors)))
        ax.set_yticklabels([str(s) for s in sectors])
        ax.set_ylabel("sector")
        ax.set_xlabel(xlabel)
        _set_time_ticks(ax, x_values)
    for ax in axes.ravel()[len(cluster_ids) :]:
        ax.axis("off")
    if im is not None:
        fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.82, label="active fraction")
    _add_cluster_legend(fig, cluster_ids, palette)
    fig.suptitle("Sector fan active-cluster visualisation", y=0.995)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _sort_holes(response_df: pd.DataFrame, phenotype_df: pd.DataFrame, event_df: pd.DataFrame) -> list[int]:
    holes = sorted(int(v) for v in response_df["hole_id"].dropna().unique())
    rows: list[dict[str, Any]] = []
    phenotype_map: dict[int, Any] = {}
    if not phenotype_df.empty and "hole_id" in phenotype_df.columns:
        label_col = "phenotype_label" if "phenotype_label" in phenotype_df.columns else None
        if label_col is not None:
            phenotype_map = {int(r.hole_id): str(getattr(r, label_col)) for r in phenotype_df[["hole_id", label_col]].dropna().itertuples(index=False)}
    onset_map: dict[int, float] = {}
    if not event_df.empty and "hole_id" in event_df.columns:
        onset_cols = [c for c in event_df.columns if "onset" in c and "frame" in c]
        if onset_cols:
            for hole_id, grp in event_df.groupby("hole_id"):
                vals = pd.to_numeric(grp[onset_cols].stack(), errors="coerce").dropna()
                if not vals.empty:
                    onset_map[int(hole_id)] = float(vals.min())
    for hole_id in holes:
        sub = response_df[response_df["hole_id"] == hole_id]
        responses = pd.to_numeric(sub["response"], errors="coerce")
        peak_frame = int(sub.iloc[int(responses.fillna(-np.inf).to_numpy().argmax())]["frame"]) if len(sub) else math.inf
        rows.append(
            {
                "hole_id": hole_id,
                "phenotype": phenotype_map.get(hole_id, ""),
                "onset": onset_map.get(hole_id, math.inf),
                "peak": peak_frame,
                "integrated": -float(responses.sum(skipna=True)),
            }
        )
    return [int(r["hole_id"]) for r in sorted(rows, key=lambda r: (str(r["phenotype"]), float(r["onset"]), float(r["peak"]), float(r["integrated"]), int(r["hole_id"])))]


def plot_hole_response_barcode(
    response_df: pd.DataFrame,
    out_path: Path,
    palette: dict[int, Any] | None = None,
    style: Any = None,
    phenotype_df: pd.DataFrame | None = None,
    event_df: pd.DataFrame | None = None,
) -> Path:
    if response_df.empty:
        return _placeholder_figure(out_path, "Hole response barcode", "No per-hole response rows available.")
    phenotype_df = pd.DataFrame() if phenotype_df is None else phenotype_df
    event_df = pd.DataFrame() if event_df is None else event_df
    holes = _sort_holes(response_df, phenotype_df, event_df)
    frames, x_values, xlabel = _frame_axis(response_df, style)
    pivot = response_df.pivot_table(index="hole_id", columns="frame", values="response", aggfunc="mean").reindex(index=holes, columns=frames)
    fig, ax = plt.subplots(figsize=(10, max(4.0, 0.22 * len(holes) + 2.2)))
    im = ax.imshow(pivot.to_numpy(dtype=float), aspect="auto", origin="lower", cmap="magma")
    ax.set_title("Hole response barcode")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("hole")
    ax.set_yticks(range(len(holes)))
    ax.set_yticklabels([str(h) for h in holes])
    _set_time_ticks(ax, x_values)
    fig.colorbar(im, ax=ax, label="summed active fraction")
    if palette:
        _add_cluster_legend(fig, sorted(int(v) for v in palette), palette)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_activity_weighted_front_trajectory(
    front_df: pd.DataFrame,
    out_path: Path,
    palette: dict[int, Any] | None = None,
    style: Any = None,
) -> Path:
    if front_df.empty:
        return _placeholder_figure(out_path, "Activity-weighted radial front trajectory", "No front trajectory rows available.")
    cluster_ids = sorted(int(v) for v in front_df["cluster_id"].dropna().unique())
    palette = palette or _stable_cluster_palette(cluster_ids)
    marker_map = _cluster_marker_map(cluster_ids)
    linestyle_map = _cluster_linestyle_map(cluster_ids)
    axis_frames, axis_values, xlabel = _frame_axis(front_df, style)
    x_lookup = {int(frame): float(value) for frame, value in zip(axis_frames, axis_values)}
    total_activity = front_df.groupby("cluster_id")["total_activity"].sum(min_count=1).to_dict() if "total_activity" in front_df.columns else {}
    draw_order = sorted(cluster_ids, key=lambda cid: float(total_activity.get(cid, 0.0)))
    fig, ax = plt.subplots(figsize=(10.8, 5.0))
    for cid in draw_order:
        sub = front_df[front_df["cluster_id"] == cid].sort_values("frame")
        x = np.asarray([x_lookup.get(int(frame), float(frame)) for frame in sub["frame"].to_numpy(dtype=int)], dtype=float)
        y = sub["front_terrace"].to_numpy(dtype=float)
        spread = sub["spread_terrace"].to_numpy(dtype=float)
        ax.plot(
            x,
            y,
            marker=marker_map[cid],
            linestyle=linestyle_map[cid],
            color=palette[cid],
            linewidth=1.8,
            markersize=5.0,
            markeredgewidth=0.7,
            markeredgecolor="white",
            label=f"cluster {cid}",
        )
        finite = np.isfinite(y) & np.isfinite(spread)
        if np.any(finite):
            ax.fill_between(x[finite], y[finite] - spread[finite], y[finite] + spread[finite], color=palette[cid], alpha=0.055, linewidth=0)
    ax.set_title("Activity-weighted radial front trajectory")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("terrace position")
    finite_y = pd.to_numeric(front_df["front_terrace"], errors="coerce").dropna()
    if not finite_y.empty:
        ax.set_ylim(max(0.0, float(finite_y.min()) - 0.7), float(finite_y.max()) + 0.7)
    _add_cluster_legend(fig, cluster_ids, palette, marker_map=marker_map, linestyle_map=linestyle_map)
    ax.grid(True, alpha=0.25)
    ax.margins(x=0.02)
    fig.tight_layout(rect=[0.04, 0.05, 0.84, 0.96])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_cluster_colour_legend(
    cluster_summary_df: pd.DataFrame,
    out_path: Path,
    palette: dict[int, Any],
    style: Any = None,
    activity_df: pd.DataFrame | None = None,
    measured_palette_df: pd.DataFrame | None = None,
) -> Path:
    activity_df = pd.DataFrame() if activity_df is None else activity_df
    measured_palette_df = pd.DataFrame() if measured_palette_df is None else measured_palette_df
    cluster_ids = _cluster_ids(activity_df, cluster_summary_df)
    if not cluster_ids:
        cluster_ids = sorted(int(v) for v in palette)
    measured = _measured_cluster_colours(cluster_summary_df, measured_palette_df)
    activity_scores = {}
    if not activity_df.empty:
        activity_scores = activity_df.groupby("cluster_id")["active_fraction"].sum(min_count=1).to_dict()
    fig_h = max(2.4, 0.55 * len(cluster_ids) + 1.2)
    fig, ax = plt.subplots(figsize=(8.8, fig_h))
    ax.axis("off")
    ax.text(0.02, 0.96, "Cluster colour legend", fontsize=14, weight="bold", transform=ax.transAxes)
    headers = [("cluster", 0.02), ("cluster colour", 0.20), ("centroid / activity", 0.42)]
    for text, x in headers:
        ax.text(x, 0.86, text, fontsize=10, weight="bold", transform=ax.transAxes)
    y0 = 0.78
    dy = 0.72 / max(len(cluster_ids), 1)
    summary_by_id = cluster_summary_df.set_index("cluster_id") if not cluster_summary_df.empty and "cluster_id" in cluster_summary_df.columns else pd.DataFrame()
    for i, cid in enumerate(cluster_ids):
        y = y0 - i * dy
        ax.text(0.02, y, f"cluster {cid}", va="center", fontsize=10, transform=ax.transAxes)
        ax.add_patch(Rectangle((0.20, y - 0.035), 0.13, 0.07, transform=ax.transAxes, facecolor=palette.get(cid, "#999999"), edgecolor="black"))
        pieces = []
        if not summary_by_id.empty and cid in summary_by_id.index:
            row = summary_by_id.loc[cid]
            for name in ("center_h", "center_s", "center_l"):
                if name in row and pd.notna(row[name]):
                    pieces.append(f"{name[7:]}={float(row[name]):.3f}")
        if cid in measured:
            rgb = measured[cid]
            pieces.append(f"rgb={int(round(rgb[0] * 255))},{int(round(rgb[1] * 255))},{int(round(rgb[2] * 255))}")
        if cid in activity_scores and pd.notna(activity_scores[cid]):
            pieces.append(f"activity={float(activity_scores[cid]):.3f}")
        ax.text(0.42, y, " | ".join(pieces), va="center", fontsize=9, transform=ax.transAxes)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out_path


def build_visualisation_montage(image_paths: list[Path], out_path: Path) -> Path:
    existing = [Path(p) for p in image_paths if Path(p).exists() and Path(p).stat().st_size > 0]
    if not existing:
        return _placeholder_figure(out_path, "HoleColor visualisation montage", "No option images were available for montage.")
    ncols = 3
    nrows = int(math.ceil(len(existing) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, 3.8 * nrows), squeeze=False)
    for ax, path in zip(axes.ravel(), existing):
        try:
            image = iio.imread(path)
            ax.imshow(image)
        except Exception:
            ax.text(0.5, 0.5, f"Could not read\n{path.name}", ha="center", va="center")
        ax.set_title(path.stem, fontsize=9)
        ax.axis("off")
    for ax in axes.ravel()[len(existing) :]:
        ax.axis("off")
    fig.suptitle("HoleColor cluster RDF visualisation montage", y=0.995)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def load_cluster_radial_visualisation_data(run_dir: str | Path, config: Any = None) -> ClusterRdfVisualisationData:
    root, out_dir = _resolve_output_dir(run_dir)
    tensor_path = out_dir / "hole_terrace_sector_cluster_tensor.csv"
    if not tensor_path.exists():
        raise FileNotFoundError(f"required radial cluster tensor not found: {tensor_path}")
    tensor_df = _read_csv_if_exists(tensor_path)
    baseline_n = int(_cfg_value(config, "baseline_frames", 3))
    terrace_base = int(_cfg_value(config, "terrace_display_base", 1))
    activity_df = build_radial_cluster_activity_table(tensor_df, baseline_frames=baseline_n, terrace_display_base=terrace_base)
    sector_df = build_sector_cluster_activity_table(tensor_df, baseline_frames=baseline_n, terrace_display_base=terrace_base)
    hole_response_df = build_hole_response_table(tensor_df, baseline_frames=baseline_n)
    front_df = compute_activity_weighted_front(activity_df, terrace_display_base=terrace_base)
    cluster_summary_df = _read_csv_if_exists(root / "descriptors" / "wafer_nonhole_colour" / "cluster_model_summary.csv")
    measured_palette_df = _read_csv_if_exists(out_dir / "radial_cluster_palette.csv")
    phenotype_df = _read_csv_if_exists(root / "temporal" / "per_hole_phenotypes.csv")
    event_df = _read_csv_if_exists(root / "temporal" / "per_hole_events.csv")
    frames = sorted(int(v) for v in activity_df["frame"].dropna().unique()) if not activity_df.empty else []
    return ClusterRdfVisualisationData(
        run_dir=root,
        out_dir=out_dir,
        tensor_df=tensor_df,
        activity_df=activity_df,
        sector_df=sector_df,
        hole_response_df=hole_response_df,
        front_df=front_df,
        cluster_summary_df=cluster_summary_df,
        measured_palette_df=measured_palette_df,
        phenotype_df=phenotype_df,
        event_df=event_df,
        baseline_frames=_baseline_frame_ids(frames, baseline_n),
    )


def run_cluster_rdf_visualisations(
    run_dir: str | Path,
    config: Any = None,
    progress_callback: Callable[[int, int, str], None] | None = None,
) -> dict[str, Any]:
    root, out_dir = _resolve_output_dir(run_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    status_path = out_dir / "cluster_rdf_visualisation_status.json"
    if not bool(_cfg_value(config, "enabled", True)):
        status = {"status": "skipped", "message": "cluster_rdf_visualisation_disabled", "outputs": []}
        write_json(status_path, status)
        return status
    fail_open = bool(_cfg_value(config, "fail_open", True))
    try:
        progress_step = 0

        def report(message: str, total: int, final: bool = False) -> None:
            nonlocal progress_step
            progress_step = int(total) if final else min(int(total), progress_step + 1)
            if progress_callback is not None:
                progress_callback(progress_step, int(total), message)

        data = load_cluster_radial_visualisation_data(root, config)
        if data.activity_df.empty:
            raise ValueError("radial cluster activity table is empty")
        data.activity_df.to_csv(data.out_dir / "radial_cluster_rdf_activity.csv", index=False)
        data.front_df.to_csv(data.out_dir / "activity_weighted_front_trajectory.csv", index=False)
        cluster_ids = _cluster_ids(data.activity_df, data.cluster_summary_df)
        palette = _cluster_colour_palette(cluster_ids, data.cluster_summary_df, data.measured_palette_df)
        outputs: list[Path] = []
        plot_jobs = [
            ("generate_raw_rdf", "raw", "raw cluster RDF", lambda path: plot_cluster_rdf_raw_fraction(data.activity_df, path, palette, config)),
            ("generate_active_rdf", "active", "active cluster RDF", lambda path: plot_cluster_rdf_active(data.activity_df, path, palette, config)),
            ("generate_dominant_chronogram", "dominant", "dominant cluster map", lambda path: plot_dominant_cluster_chronogram(data.activity_df, path, palette, config)),
            ("generate_radial_snapshots", "profiles", "radial profile snapshots", lambda path: plot_radial_profile_snapshots(data.activity_df, path, palette, config)),
            ("generate_average_hole_rings", "rings", "average-hole rings", lambda path: plot_average_hole_ring_snapshots(data.activity_df, path, palette, config, data.sector_df)),
            ("generate_sector_fan", "sector", "sector fan", lambda path: plot_sector_fan_active_cluster(data.sector_df, path, palette, config)),
            (
                "generate_hole_barcode",
                "barcode",
                "hole response barcode",
                lambda path: plot_hole_response_barcode(data.hole_response_df, path, palette, config, data.phenotype_df, data.event_df),
            ),
            ("generate_front_trajectory", "front", "front trajectory", lambda path: plot_activity_weighted_front_trajectory(data.front_df, path, palette, config)),
            (
                "generate_cluster_legend",
                "legend",
                "cluster legend",
                lambda path: plot_cluster_colour_legend(data.cluster_summary_df, path, palette, config, data.activity_df, data.measured_palette_df),
            ),
        ]
        enabled_jobs = [job for job in plot_jobs if bool(_cfg_value(config, job[0], True))]
        progress_total = max(1, 2 + len(enabled_jobs) + int(bool(_cfg_value(config, "generate_montage", True))))
        report("Loaded cluster tensor and wrote activity tables", progress_total)

        for _flag, key, label, fn in enabled_jobs:
            path = data.out_dir / OPTION_OUTPUTS[key]
            outputs.append(fn(path))
            report(f"Wrote {label}", progress_total)

        if bool(_cfg_value(config, "generate_montage", True)):
            montage_inputs = [p for p in outputs if p.name != OPTION_OUTPUTS["montage"]]
            outputs.append(build_visualisation_montage(montage_inputs, data.out_dir / OPTION_OUTPUTS["montage"]))
            report("Wrote visualisation montage", progress_total)
        status = {
            "status": "ok",
            "message": "ok",
            "baseline_frames": data.baseline_frames,
            "time_axis": _cfg_value(config, "time_axis", "time_s"),
            "screened_frames": int(data.activity_df["frame"].nunique()),
            "visualised_frames": _selected_snapshot_frames(data.activity_df, int(_cfg_value(config, "baseline_frames", 3))),
            "n_activity_rows": int(len(data.activity_df)),
            "n_clusters": int(len(cluster_ids)),
            "outputs": [p.name for p in outputs],
            "option_07_generated": False,
        }
        write_json(status_path, status)
        report("Cluster RDF visualisations complete", progress_total, final=True)
        return status
    except FileNotFoundError as exc:
        status = {"status": "skipped", "message": str(exc), "outputs": [], "option_07_generated": False}
        write_json(status_path, status)
        LOG.warning("cluster RDF visualisations skipped: %s", exc)
        return status
    except Exception as exc:
        status = {"status": "error", "message": str(exc), "outputs": [], "option_07_generated": False}
        write_json(status_path, status)
        if not fail_open:
            raise
        LOG.warning("cluster RDF visualisations skipped after error: %s", exc)
        return status
