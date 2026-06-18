This milestone moves the hotspot and uncertainty comparison layer onto lighter internal table paths.

What changed
- hotspot rows are summarized once into a reusable HotspotStatsTable
- RDF uncertainty joins are materialized once into a reusable RdfUncertaintyHoleTable
- pipeline reuses these internal tables across:
  - hotspot/reticulum comparisons
  - RDF hotspot/reticulum comparisons
  - RDF uncertainty reticulum summaries
  - RDF uncertainty hotspot comparisons

Why this helps
- avoids recomputing per-hole hotspot distance summaries multiple times
- avoids rebuilding the same bootstrap/support/archetype/dynamics join repeatedly
- keeps the external row/CSV artifacts unchanged while reducing repeated row-dict regrouping in the hot comparison layer
