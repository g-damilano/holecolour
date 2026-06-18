# Comparison Boundary Notes

Milestone 44 extends direct artifact emission to the hotspot/reticulum and RDF uncertainty comparison layer.

The run now writes several comparison CSVs directly from internal aligned tables or direct column payloads instead of materializing large row lists purely for writing/plotting.

Affected artifacts:
- radial/hole_hotspot_reticulum_comparison.csv
- radial/hotspot_reticulum_group_summary.csv
- radial/rdf_hotspot_reticulum_comparison.csv
- radial/rdf_hotspot_reticulum_group_summary.csv
- radial/rdf_uncertainty_reticulum.csv
- radial/rdf_uncertainty_hotspot_comparison.csv
- radial/rdf_uncertainty_hotspot_group_summary.csv

Plots consuming the grouped comparison payloads were updated to accept column dictionaries directly.
