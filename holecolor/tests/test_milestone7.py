from pathlib import Path

import pandas as pd

from holecolor.config.schema import PipelineConfig
from holecolor.core.types import FrameRecord
from holecolor.pipeline import run_milestone7
from holecolor.synth.artifacts import add_known_drift, add_photometric_artifacts
from holecolor.synth.radial_front import make_synthetic_radial_front_video


def test_milestone7_writes_geometry_timeseries_and_lag_summary(tmp_path: Path):
    frames, _ = make_synthetic_radial_front_video(n_frames=6)
    frames = add_photometric_artifacts(frames, brightness_ramp=0.08, vignette_strength=0.12)
    frames, _ = add_known_drift(frames, dx_per_frame=0.35, dy_per_frame=-0.2)
    records = [FrameRecord(i, float(i), f) for i, f in enumerate(frames)]
    out_dir = tmp_path / "run_m7"
    summary = run_milestone7(records, out_dir, PipelineConfig())
    assert summary["n_frames"] == 6
    geo_path = out_dir / "geometry" / "hole_geometry_timeseries.csv"
    lag_path = out_dir / "temporal" / "per_hole_events.csv"
    ann_path = out_dir / "temporal" / "per_hole_annulus_events.csv"
    assert geo_path.exists()
    assert lag_path.exists()
    assert ann_path.exists()

    geo = pd.read_csv(geo_path)
    lag = pd.read_csv(lag_path)
    assert {"frame_id", "hole_id", "lattice_u", "lattice_v", "x", "y"}.issubset(set(geo.columns))
    assert {"hole_id", "onset_lag_frames", "peak_lag_frames", "monotonic_onset", "monotonic_peak"}.issubset(set(lag.columns))
    assert len(geo) > 0
    assert len(lag) > 0
