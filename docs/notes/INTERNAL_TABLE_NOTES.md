Milestone 35 introduces an internal array-backed radial table used inside the aggregate stage.

The external contract is unchanged:
- CSV artifacts are still written as row-oriented tables
- downstream public functions can still accept list[dict] rows

The efficiency change is internal:
- radial annulus rows are converted once into a compact columnar table
- several hot first-run aggregate paths operate on that shared table
- conversion back to row dicts happens only at the artifact boundary

This reduces repeated:
- row sorting
- Python dict lookups
- per-function list bucketing
- array reconstruction from the same radial rows
