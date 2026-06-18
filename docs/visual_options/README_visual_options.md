# holecolor visual options

Generated from `/mnt/data/JAO25_avi.zip`.

Dataset used for the main visuals:
- 44 frames, time range 0.000–5.375 s
- 8 outside-hole terraces, T1 nearest the hole edge through T8 outermost
- 4 learned colour clusters C0–C3
- black hole interiors are deliberately excluded from the RDF / terrace visualisations

Most useful candidate for the next app default:
1. `01_cluster_activity_heatmaps.png` for literal cluster activity by radius and time.
2. `03_dominant_cluster_chronogram.png` for a compact human-readable story.
3. `07_wafer_ring_glyph_map_final.png` for explaining where those radial patterns occur on the wafer.

Two RDF interpretations included:
- `p(cluster | distance, time)`: how active a cluster is at a given terrace.
- `p(distance | cluster, time)`: where a cluster's own pixels are distributed radially.
