# Radial cluster average-hole patch

## What this patch adds

This patch adds a new additive branch under:

- `descriptors/radial_cluster_average_hole/`

The new branch is driven by the already-settled upstream objects:

- canonical/static hole geometry
- existing terrace plan
- existing cluster identities and colours from `cluster_model_summary`

It does **not** re-cluster colours and does **not** alter legacy radial outputs.

## New runtime stage

A new pipeline stage is added after **Reference terraces**:

- `Radial cluster average hole`

The stage is additive and fail-open.

## New primary outputs

The branch now writes the four target output families:

1. `average_hole_terrace_chronogram.png`
2. `terrace_XX_angular_chronogram.png`
3. `cluster_front_trajectories.png`
4. `hole_consistency_terrace_map.png`

It also writes the supporting tables:

- `hole_terrace_sector_cluster_tensor.csv`
- `average_hole_terrace_cluster_fractions.csv`
- `average_hole_terrace_summary.csv`
- `terrace_angle_cluster_fractions.csv`
- `terrace_angle_cluster_pooled.csv`
- `cluster_front_metrics.csv`
- `hole_consistency_by_terrace.csv`
- `radial_cluster_status.json`

## Files changed

- `holecolor/config/schema.py`
- `holecolor/pipeline.py`
- `holecolor/extensions/radial_cluster_average_hole.py` (new)
- `holecolor/tests/test_radial_cluster_average_hole.py` (new)
- `holecolor/tests/test_safe_modular_extensions.py`
- `holecolor/tests/test_pipeline_artifacts.py`

## Verification completed

Focused verification completed on:

- `holecolor/tests/test_radial_cluster_average_hole.py`
- `holecolor/tests/test_wafer_nonhole_colour.py`
- `holecolor/tests/test_safe_modular_extensions.py`
- `holecolor/tests/test_pipeline_artifacts.py`
- `holecolor/tests/test_milestone24.py`

And one additional synthetic pipeline smoke verified emission of:

- `radial_cluster_status.json`
- `hole_terrace_sector_cluster_tensor.csv`
- `average_hole_terrace_summary.csv`
- `terrace_01_angular_chronogram.png`
- `cluster_front_trajectories.png`
- `hole_consistency_terrace_map.png`

## Important note

This patch is additive. Legacy geometry, terrace, radial, and wafer-nonhole outputs remain in place.
