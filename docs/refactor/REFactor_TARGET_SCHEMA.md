# Refactor Target Schema

## Geometry layer

### WaferGeometry
- id
- center_xy_px
- radius_px
- confidence
- visible_arc_intervals_deg
- detection_mode
- notes

### BufferGeometry
- id
- state: `full | partial | unknown`
- center_xy_px
- radius_px
- confidence
- visible_arc_intervals_deg
- center_outside_frame
- notes

### LatticeModel
- id
- basis_v1_xy_px
- basis_v2_xy_px
- origin_xy_px
- fit_confidence
- index_bounds
- notes

### HoleRecord
- node_id
- center_xy_px
- radius_px
- confidence
- lattice_i
- lattice_j
- tier
- source_class
- visible_fraction
- border_refined
- notes

### HoleBufferRelation
- node_id
- relation: `inside_buffer | intersects_buffer_border | outside_buffer | border_unknown`
- partial_visibility
- exclusion_reason
- notes

## Analysis layer

### LocalHoleMetrics
- node_id
- interior stats
- rim stats
- annulus stats
- sector stats
- radial trajectory summary
- angular asymmetry summary

### GlobalBufferMetrics
- frame_index
- valid_buffer_area_px
- bulk color stats
- histogram stats
- heterogeneity stats
- hotspot fraction
- radial-from-center metrics
- radial-from-border metrics
- sector metrics

### CoupledMetrics
- frame_index
- node_id or grouped tier
- local-vs-global deltas
- lag measures
- border-distance correlation

## Archive-level management layer

### RefactorState
- current_milestone
- overall_progress_percent
- last_verified_action_id
- stable_path
- next_action
- blockers
- rollback_point
- baselines

### LedgerEntry
- action_id
- timestamp
- intent
- scope
- files_changed
- expected_effect
- observed_effect
- verification_run
- pass_fail
- regression_summary
- rollback_note
