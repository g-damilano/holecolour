# Canonical Static Geometry Patch

Base archive:
- `holecolor_milestone53_v0_53_safe_modular_cluster_recolour_video_patch.zip`

Purpose:
- freeze hole geometry to the canonical reference-consensus set for all frames
- remove per-frame hole inference by default
- keep the rest of the pipeline unchanged

Implemented behavior:
- new config field: `geometry.propagation_mode`
- supported values:
  - `canonical_static` (default)
  - `free`
- under `canonical_static`, geometry propagation copies the reference hole bundle to every frame unchanged:
  - same hole ids
  - same centers
  - same radii
  - same confidences
- under `free`, the prior contour-based propagation path is preserved

Files changed:
- `holecolor/config/schema.py`
- `holecolor/pipeline.py`
- `holecolor/tests/test_canonical_static_propagation.py`

Verification completed:
- `pytest -q holecolor/tests/test_canonical_static_propagation.py` -> passed
- `pytest -q holecolor/tests/test_safe_modular_default_off.py holecolor/tests/test_safe_modular_extensions.py holecolor/tests/test_wafer_nonhole_colour.py holecolor/tests/test_milestone24.py` -> passed
- `pytest -q holecolor/tests/test_pipeline_artifacts.py` -> passed
- real-video 2-frame smoke on `JAO25.avi` completed and emitted:
  - `geometry/hole_geometry_timeseries.csv`
  - `descriptors/wafer_nonhole_colour/*`

Smoke verification result:
- every hole in `geometry/hole_geometry_timeseries.csv` had exactly one unique `(x, y, radius_inner_px, radius_outer_px)` tuple across frames

Important note:
- this patch changes geometry propagation default behavior to `canonical_static`
- it does not remove the old contour-tracking path; that remains available through `geometry.propagation_mode = "free"`
