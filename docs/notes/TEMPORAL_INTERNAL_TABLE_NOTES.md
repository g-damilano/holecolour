# Temporal internal-table notes

Milestone 39 moves the hottest temporal/phenotype summary reuse onto an internal table path.

What changed:
- phenotype lattice adjacency and spatial smoothness are now computed from one shared `PhenotypeTable`
- phenotype archetype aggregation now reuses the shared `RadialRowTable` plus `PhenotypeTable`
- QC temporal validation fractions are assembled once into `TemporalValidationSummary`

What stays the same:
- CSV/JSON outputs remain row-oriented and human-readable
- public artifact names are unchanged
- old row-based helpers remain available for compatibility
