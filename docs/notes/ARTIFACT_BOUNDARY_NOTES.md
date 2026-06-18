Artifact-boundary efficiency notes

This milestone reduces row-dict churn at the writing boundary for the largest annulus/sector timeseries artifacts.

Changes:
- hole_annulus_timeseries.csv is written directly from RadialRowTable columns
- sector_radial_timeseries.csv is written directly from SectorRadialTable columns
- the first-hole annulus plot is prepared directly from RadialRowTable instead of re-slicing row dicts

Readable CSV outputs are preserved. The change is internal and targets the largest row-oriented payloads already represented in array-backed tables.
