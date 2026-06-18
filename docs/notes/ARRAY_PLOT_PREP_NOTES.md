# Plot-preparation efficiency notes (Milestone 40)

This milestone keeps the user-facing plot outputs unchanged, but makes the hot plot-preparation paths lighter.

## What changed

The pipeline used to rebuild plot payloads with repeated row-list slicing and per-group dict lookups for several plots:

- radial evolution heatmaps
- per-hole RDF heatmaps
- sector RDF line plots
- sector lag heatmaps
- reticulum-group line plots
- sector front line plots
- archetype count plots
- phenotype archetype line plots
- perturbation drift heatmaps

This milestone introduces `holecolor.plotting.prepare` with array-backed helpers:

- `count_by_label(...)`
- `line_series_from_rows(...)`
- `heatmap_from_rows(...)`

These helpers convert the relevant columns once and then build the plot payload from arrays instead of repeatedly filtering and regrouping Python row dicts.
