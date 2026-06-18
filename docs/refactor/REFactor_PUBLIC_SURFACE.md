# Refactor Public Surface Declaration

Timestamp: `2026-03-18T17:32:07Z`

## Canonical runtime entrypoints
- `run_milestone1`
- `run_milestone16`
- `run_milestone18`

## Preserved legacy aliases

### Forwarding to milestone16
- `run_milestone2`
- `run_milestone3`
- `run_milestone4`
- `run_milestone7`
- `run_milestone8`
- `run_milestone9`
- `run_milestone10`
- `run_milestone11`
- `run_milestone12`
- `run_milestone13`
- `run_milestone14`
- `run_milestone15`

### Forwarding to milestone18
- `run_milestone2`
- `run_milestone3`
- `run_milestone4`
- `run_milestone7`
- `run_milestone8`
- `run_milestone9`
- `run_milestone10`
- `run_milestone11`
- `run_milestone12`
- `run_milestone13`
- `run_milestone14`
- `run_milestone15`
- `run_milestone17`
- `run_milestone19`
- `run_milestone20`
- `run_milestone21`
- `run_milestone22`
- `run_milestone23`
- `run_milestone24`

## Compatibility helpers
- `_run_legacy_alias_to_milestone16(...)`
- `_run_legacy_alias_to_milestone18(...)`

## Verified stable runtime artifacts
- `geometry/wafer_geometry.json`
- `geometry/buffer_geometry.json`
- `geometry/hole_tier_records.csv`
- `geometry/hole_tier_records.json`
- `geometry/hole_buffer_relations.csv`
- `geometry/hole_buffer_relations.json`
- `geometry/hole_selection_policy.json`
- `descriptors/global_buffer_timeseries.csv`
- `descriptors/global_buffer_region.json`
- `descriptors/global_buffer_radial_profiles.csv`
- `descriptors/global_buffer_band_profiles.csv`
- `descriptors/coupled_hole_buffer_timeseries.csv`
- `descriptors/coupled_hole_buffer_interpretation.csv`
- `descriptors/coupled_hole_buffer_position_summary.csv`
- `descriptors/coupled_hole_buffer_scientific_synthesis.csv`

## Policy
- public milestone names are preserved
- alias removal has **not** been performed
- canonical entrypoints are now explicitly documented
- cross-video validation is **not** part of this public-surface declaration
