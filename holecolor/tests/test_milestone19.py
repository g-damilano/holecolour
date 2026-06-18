from pathlib import Path

import pandas as pd

from holecolor.config.schema import PipelineConfig
from holecolor.core.types import FrameRecord
from holecolor.pipeline import run_milestone18
from holecolor.synth.artifacts import add_known_drift, add_photometric_artifacts
from holecolor.tests.test_milestone14 import _make_small_rdf_video


def test_milestone19_parallel_thread_run_writes_core_outputs(tmp_path: Path):
    frames = _make_small_rdf_video(n_frames=6)
    frames = add_photometric_artifacts(frames, brightness_ramp=0.04, vignette_strength=0.06)
    frames, _ = add_known_drift(frames, dx_per_frame=0.15, dy_per_frame=-0.1)
    records = [FrameRecord(i, float(i), f) for i, f in enumerate(frames)]

    out_dir = tmp_path / "run_m19"
    cfg = PipelineConfig()
    cfg.parallel.backend = "thread"
    cfg.parallel.max_workers = 2
    cfg.parallel.min_parallel_tasks = 1
    cfg.parallel.show_progress = False

    summary = run_milestone18(records, out_dir, cfg)
    assert summary["n_frames"] == 6

    expected = [
        out_dir / "descriptors" / "descriptor_selection.json",
        out_dir / "radial" / "per_hole_rdf_evolution.csv",
        out_dir / "radial" / "per_hole_rdf_bootstrap_summary.csv",
        out_dir / "qc" / "radial_conclusion_consistency.json",
    ]
    for pth in expected:
        assert pth.exists(), pth

    rdf_df = pd.read_csv(out_dir / "radial" / "per_hole_rdf_evolution.csv")
    boot_df = pd.read_csv(out_dir / "radial" / "per_hole_rdf_bootstrap_summary.csv")
    assert len(rdf_df) > 0
    assert len(boot_df) > 0
    assert {"hole_id", "rdf_pdf", "rdf_cdf"}.issubset(rdf_df.columns)
    assert {"hole_id", "bootstrap_front_velocity_ci_width"}.issubset(boot_df.columns)
