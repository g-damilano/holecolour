from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from holecolor.config.schema import PipelineConfig
from holecolor.visualisation.cluster_rdf_views import OPTION_OUTPUTS, run_cluster_rdf_visualisations


def _write_synthetic_cluster_run(root: Path) -> Path:
    out = root / "descriptors" / "radial_cluster_average_hole"
    out.mkdir(parents=True, exist_ok=True)
    wafer = root / "descriptors" / "wafer_nonhole_colour"
    wafer.mkdir(parents=True, exist_ok=True)
    temporal = root / "temporal"
    temporal.mkdir(parents=True, exist_ok=True)

    cluster_ids = [2, 5, 9, 12]
    rows = []
    for frame in range(5):
        for hole_id in range(3):
            for terrace in range(4):
                for sector in range(8):
                    base = np.array([70, 10, 10, 10], dtype=int)
                    if frame == 3:
                        base = np.array([50, 10, 10, 30], dtype=int) if terrace >= 2 else np.array([50, 30, 10, 10], dtype=int)
                    elif frame == 4:
                        base = np.array([35, 15, 35, 15], dtype=int) if terrace >= 2 else np.array([35, 40, 15, 10], dtype=int)
                    if sector >= 4 and frame >= 3:
                        base[1], base[2] = base[2], base[1]
                    for cid, count in zip(cluster_ids, base.tolist()):
                        rows.append(
                            {
                                "frame_id": frame,
                                "time_s": frame * 0.5,
                                "hole_id": hole_id,
                                "lattice_u": hole_id,
                                "lattice_v": 0,
                                "terrace_index": terrace,
                                "sector_id": sector,
                                "theta_center_deg": (sector + 0.5) * 45.0,
                                "cluster_id": cid,
                                "pixel_count": count,
                                "pixel_fraction": count / 100.0,
                                "n_valid_pixels": 100,
                            }
                        )
    pd.DataFrame(rows).to_csv(out / "hole_terrace_sector_cluster_tensor.csv", index=False)
    pd.DataFrame(
        [
            {
                "cluster_id": cid,
                "center_h": (i + 1) / 7.0,
                "center_s": 0.35 + 0.1 * i,
                "center_l": 0.35 + 0.08 * i,
                "center_hx": 0.0,
                "center_hy": 0.0,
            }
            for i, cid in enumerate(cluster_ids)
        ]
    ).to_csv(wafer / "cluster_model_summary.csv", index=False)
    pd.DataFrame(
        [
            {"hole_id": 0, "phenotype_label": "early"},
            {"hole_id": 1, "phenotype_label": "middle"},
            {"hole_id": 2, "phenotype_label": "late"},
        ]
    ).to_csv(temporal / "per_hole_phenotypes.csv", index=False)
    pd.DataFrame(
        [
            {"hole_id": 0, "inner_onset_frame": 3},
            {"hole_id": 1, "inner_onset_frame": 4},
            {"hole_id": 2, "inner_onset_frame": 4},
        ]
    ).to_csv(temporal / "per_hole_events.csv", index=False)
    return out


def test_cluster_rdf_visualisation_outputs_and_invariants(tmp_path: Path) -> None:
    out = _write_synthetic_cluster_run(tmp_path)
    cfg = PipelineConfig()
    assert cfg.wafer_nonhole_colour.gmm_k_max == 8
    assert cfg.visualisation.cluster_rdf.time_axis == "time_s"
    status = run_cluster_rdf_visualisations(tmp_path, cfg)

    assert status["status"] == "ok"
    assert status["screened_frames"] == 5
    assert status["visualised_frames"] == [0, 1, 2, 3, 4]
    expected = [
        OPTION_OUTPUTS["raw"],
        OPTION_OUTPUTS["active"],
        OPTION_OUTPUTS["dominant"],
        OPTION_OUTPUTS["profiles"],
        OPTION_OUTPUTS["rings"],
        OPTION_OUTPUTS["sector"],
        OPTION_OUTPUTS["barcode"],
        OPTION_OUTPUTS["front"],
        OPTION_OUTPUTS["legend"],
        OPTION_OUTPUTS["montage"],
    ]
    for name in expected:
        path = out / name
        assert path.exists(), name
        assert path.stat().st_size > 0, name

    activity = pd.read_csv(out / "radial_cluster_rdf_activity.csv")
    assert set(activity["cluster_id"].unique()) == {2, 5, 9, 12}
    assert (activity["active_fraction"].dropna() >= 0.0).all()
    assert set(activity["terrace_label"].unique()) == {"T1", "T2", "T3", "T4"}

    front = pd.read_csv(out / "activity_weighted_front_trajectory.csv")
    assert front.loc[front["frame"] == 0, "front_terrace"].isna().all()
    assert not (out / "option_07_wafer_ring_glyph.png").exists()
