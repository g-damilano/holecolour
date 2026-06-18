# Radial cluster visuals patch

Added an additive radial-cluster analysis branch that consumes the existing wafer non-hole cluster model and the existing annulus geometry without altering the legacy radial descriptor path.

## New runtime artifacts

Written under `descriptors/radial_cluster/`:

- `radial_cluster_status.json`
- `radial_cluster_palette.csv`
- `radial_cluster_counts.csv`
- `radial_cluster_pooled_fractions.csv`
- `radial_cluster_front_metrics.csv`
- `radial_cluster_dominant_map.csv`
- `radial_cluster_front_trajectories.png`
- `radial_cluster_dominant_map.png`
- `radial_cluster_heatmap_cluster_<k>.png`

## What the new branch does

- assigns wafer non-hole pixels to the fitted colour clusters
- intersects those assignments with the existing non-overlapping annulus regions
- builds per-hole and pooled radial cluster prevalence tables
- derives cluster front metrics over time
- renders pooled radial heatmaps per cluster
- renders a dominant-cluster radial map and front-trajectory plot

## Integration policy

- additive only
- legacy radial tables remain untouched
- legacy hole/global outputs remain untouched
- the branch runs only when the wafer non-hole cluster bundle is available
- fail-open behaviour is preserved through the existing wafer non-hole extension logic

## Focused verification completed

- `holecolor/tests/test_wafer_nonhole_colour.py`
- `holecolor/tests/test_pipeline_artifacts.py`

These checks verified both the direct radial-cluster writer and artifact emission through the milestone-3 pipeline path.

## Important note

This patch focuses on the first radial depiction layer: pooled radius vs time vs cluster. It does **not** yet add the angle-residual branch.
