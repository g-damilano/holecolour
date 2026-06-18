Milestone 32 aggregate efficiency notes

This build reduces first-run aggregate overhead by replacing several bucket-building paths with sorted/grouped iteration and by pre-aggregating hotspot distance statistics per hole.

Key changes:
- RDF evolution and per-hole RDF summaries iterate over sorted rows rather than building duplicate defaultdict(list) bucket structures.
- Angular asymmetry aggregation now uses incremental accumulators.
- Hotspot/reticulum comparison paths precompute per-hole hotspot stats once and reuse them.

These changes target first-run runtime and memory churn in the aggregate/RDF stage rather than only cache-hit speed.
