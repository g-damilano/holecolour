# Directive Pack — Wafer Non-Hole HSL Cloud + Dominant Colour Cluster Time Analysis

## Purpose

Integrate a **new global wafer colour-cloud analysis layer** into the current refactor archive (`holecolor_milestone53_v0_53_refactor_roundF.zip`) so the pipeline can:

1. extract **all pixel datapoints** from the **wafer area with holes excluded**,
2. store those datapoints as **HSL-valued observations**,
3. fit a small set of **dominant colour clusters** over the pooled video dataset,
4. quantify each cluster by **center + variability + mass**, and
5. produce **time-series of cluster prevalence** using both:
   - hard frequency,
   - distance-weighted / soft frequency.

This must land as an **M5 extension** in the current refactor, without disturbing the exact-sequence hole logic or current global-buffer outputs.

---

## Architectural positioning

This is **not** a hole-local layer and **not** a buffer-only layer.

It is a new **global wafer non-hole branch**.

It must sit **after**:
- photometry,
- registration,
- reference/support geometry,
- hole tier detection,
- hole propagation / reference hole geometry selection,
- hole-selection policy resolution.

It must sit **before**:
- interpretation-level coupled local/global synthesis,
- final temporal interpretation packaging.

So the effective milestone-3 ordering becomes:

1. audit
2. photometry
3. registration
4. wafer geometry
5. buffer geometry
6. tiered hole geometry
7. hole-buffer relations / selection policy
8. descriptor selection
9. local hole analysis
10. existing global buffer analysis
11. **new global wafer non-hole HSL cloud extraction**
12. **new pooled colour clustering**
13. **new global wafer non-hole cluster time-series**
14. coupled local/global interpretation
15. QC / manifest

---

## Non-negotiable region policy

### Canonical region

For every analysed frame, the new layer must operate on:

`wafer_nonhole_mask_t = wafer_support_mask_t AND NOT(hole_union_mask_t)`

### Hole exclusion policy

Default exclusion must use the same hole set that the main analysis treats as accepted for geometry-aware analysis.

Required policy field:
- `hole_exclusion_policy`

Allowed values:
- `selected` → use currently selected holes under `hole_selection_policy.json`
- `tiers_1_2` → exclude Tier 1 + Tier 2 only
- `tiers_1_2_3` → exclude Tier 1 + Tier 2 + Tier 3

**Default: `selected`**

Rationale:
- this preserves consistency with the current archive’s formal hole policy,
- does not silently reintroduce Tier 3 predicted-only positions into the exclusion mask unless explicitly requested.

### Border rule

This new layer is defined on the **wafer minus holes**, not on the buffer interior alone.
So it must not fail when `BufferGeometry` is unknown.

### Required honesty rule

If the final wafer-nonhole mask is empty or too small, the stage must:
- emit an explicit status row / JSON note,
- skip clustering cleanly,
- never fabricate cluster outputs.

---

## Why HSL must not be clustered naïvely

The archive must **store HSL**, but **must not fit clusters on raw H, S, L directly**, because hue is circular.

Example:
- H = 0.99 and H = 0.01 are nearby colours,
- but raw scalar distance treats them as far apart.

### Required clustering embedding

For each pixel with HSL `(h, s, l)` in `[0,1]`, derive:

- `hx = s * cos(2πh)`
- `hy = s * sin(2πh)`
- `l = l`

Optional fourth feature:
- `s`

### Canonical clustering feature set

Required default feature space:
- `[hx, hy, l]`

Optional enhanced mode:
- `[hx, hy, s, l]`

### Canonical stored values

For interpretability, the raw pixel table must still carry:
- `H`
- `S`
- `L`

---

## New runtime artifact family

All outputs must be placed under:

`descriptors/wafer_nonhole_colour/`

This avoids polluting the existing root of `descriptors/` and keeps the new layer clearly separate from the current scaffold outputs.

### Required artifacts

#### 1. Region definition
- `descriptors/wafer_nonhole_colour/region_definition.json`
- `descriptors/wafer_nonhole_colour/frame_ref_region_overlay.png`

#### 2. Raw pooled datapoints
- `descriptors/wafer_nonhole_colour/wafer_nonhole_hsl_points.parquet`

#### 3. Optional frame-partitioned datapoints index
- `descriptors/wafer_nonhole_colour/frame_point_counts.csv`

#### 4. Cluster model
- `descriptors/wafer_nonhole_colour/cluster_model.json`
- `descriptors/wafer_nonhole_colour/cluster_summary.csv`

#### 5. Time-series outputs
- `descriptors/wafer_nonhole_colour/cluster_frequency_timeseries.csv`
- `descriptors/wafer_nonhole_colour/cluster_weighted_frequency_timeseries.csv`

#### 6. Diagnostics / plots
- `descriptors/wafer_nonhole_colour/pooled_hsl_cloud_summary.png`
- `descriptors/wafer_nonhole_colour/cluster_frequency_over_time.png`
- `descriptors/wafer_nonhole_colour/cluster_weighted_frequency_over_time.png`

#### 7. Optional spatial diagnostics
- `descriptors/wafer_nonhole_colour/frame_0000_cluster_map.png` (and/or sampled key frames)
- `descriptors/wafer_nonhole_colour/frame_0000_cluster_weight_map.png`

#### 8. Stage status / notes
- `descriptors/wafer_nonhole_colour/stage_status.json`

---

## New code touch points

### New module files to add

#### A. `holecolor/descriptors/wafer_nonhole_colour.py`
Must contain the core logic for:
- region mask assembly,
- raw point extraction,
- HSL embedding,
- pooled cluster fitting,
- hard and weighted time-series,
- summary serialization.

Required helper functions:
- `_wafer_nonhole_mask(...)`
- `_extract_wafer_nonhole_hsl_points(...)`
- `_hsl_embedding(...)`
- `_fit_colour_clusters(...)`
- `_cluster_summary_rows(...)`
- `_cluster_frequency_rows(...)`
- `_cluster_weighted_frequency_rows(...)`
- `_cluster_display_rgb(...)`

#### B. `holecolor/plotting/wafer_nonhole_colour.py`
Must contain plotting helpers for:
- pooled cloud summary,
- hard frequency lines,
- weighted frequency lines,
- optional per-frame cluster maps.

### Existing files to modify

#### 1. `holecolor/pipeline.py`
Add a dedicated stage invocation after current global buffer rows and before coupled interpretation finalization.

#### 2. `holecolor/config/schema.py`
Add a new config block:
- `WaferNonholeColourConfig`

#### 3. `holecolor/config/defaults.py`
Wire defaults into `PipelineConfig`.

#### 4. `holecolor/tests/test_pipeline_artifacts.py`
Extend artifact assertions.

#### 5. Add new focused tests
- `holecolor/tests/test_wafer_nonhole_colour_points.py`
- `holecolor/tests/test_wafer_nonhole_colour_clustering.py`
- `holecolor/tests/test_wafer_nonhole_colour_timeseries.py`

#### 6. Refactor governance files
- `refactor/REFactor_BACKLOG.md`
- `refactor/REFactor_PROGRESS.md`
- `refactor/REFactor_LEDGER.md`
- `refactor/REFactor_STATE.json`

---

## New config block

Add to `holecolor/config/schema.py`:

```python
@dataclass(slots=True)
class WaferNonholeColourConfig:
    enabled: bool = True
    hole_exclusion_policy: Literal["selected", "tiers_1_2", "tiers_1_2_3"] = "selected"
    save_full_point_cloud: bool = True
    max_points_for_fit: int = 250000
    fit_sampling_mode: Literal["uniform", "stratified"] = "stratified"
    cluster_method: Literal["gmm"] = "gmm"
    candidate_k: tuple[int, ...] = (2, 3, 4, 5, 6, 7, 8)
    covariance_type: Literal["full", "diag"] = "full"
    weighted_signal_mode: Literal["gaussian", "posterior_gaussian"] = "posterior_gaussian"
    low_saturation_hue_guard: float = 0.03
    min_region_area_px: int = 500
    save_frame_cluster_maps: bool = False
    frame_cluster_map_stride: int = 25
```

And add to `PipelineConfig`:

```python
wafer_nonhole_colour: WaferNonholeColourConfig = field(default_factory=WaferNonholeColourConfig)
```

### Required config semantics

- `save_full_point_cloud=True` means the canonical dataset is all wafer-nonhole pixels.
- `max_points_for_fit` only limits the fitting subset, **not** the canonical stored dataset.
- `candidate_k` is used for model selection.
- `weighted_signal_mode="posterior_gaussian"` is the required default.

---

## Exact data model for the raw pixel table

The raw canonical table in:

`wafer_nonhole_hsl_points.parquet`

must contain at least these columns:

- `frame_id` (int)
- `time_s` (float)
- `y_px` (int)
- `x_px` (int)
- `R` (float in [0,1])
- `G` (float in [0,1])
- `B` (float in [0,1])
- `H` (float in [0,1])
- `S` (float in [0,1])
- `L` (float in [0,1])
- `hx` (float)
- `hy` (float)
- `region_policy` (str)
- `hole_exclusion_policy` (str)

Optional but recommended:
- `selected_hole_mask_source`
- `buffer_relation_policy_snapshot`

### Important requirement

The canonical persisted point cloud must be **post-mask, pre-cluster**.
This ensures the clustering can be re-run later without re-reading the full video if desired.

---

## Region-mask construction directive

### Required wafer support source

Use the same wafer/support representation already emitted by the archive:
- `geometry/wafer_geometry.json`
- support mask / propagated support logic already present in pipeline runtime

### Required hole union source

Use the currently selected hole geometry from the main pipeline stage.
The hole mask must be built from the actual accepted circles, not from seed points.

Required mask-construction rule:
- rasterize each selected hole as a filled disk using its current center and radius
- union all hole disks into `hole_union_mask`

### Required region overlay

`frame_ref_region_overlay.png` must show:
- wafer support outline
- excluded hole disks
- retained wafer non-hole region

This overlay is mandatory because it visually verifies that the extraction region matches the intended policy.

---

## Cluster fitting directive

### Required model family

Use **Gaussian Mixture Models** on the embedding space.

Reason:
- supports soft membership,
- exposes cluster covariance / spread,
- directly supports the weighted signal you want.

### Required model selection

Fit candidate models over `candidate_k`, choose best by **BIC**.

### Required fitting dataset rule

- canonical dataset = all valid wafer-nonhole pixels across the run,
- fitting subset = optionally sampled subset of canonical dataset if needed for speed.

### Required sampling rule

Default sampling for fit must be **stratified by frame** so early / late video windows are not dominated by whichever frame contains the largest valid region.

### Required cluster ordering

After fitting, clusters must be sorted by **global pooled mass descending**.

This sorted order becomes the public cluster order used in all outputs.

---

## Required cluster summary fields

`cluster_summary.csv` must contain one row per cluster with at least:

- `cluster_id`
- `rank_by_mass`
- `global_pixel_mass`
- `global_pixel_fraction`
- `center_H`
- `center_S`
- `center_L`
- `display_R`
- `display_G`
- `display_B`
- `hue_concentration`
- `std_S`
- `std_L`
- `cov_trace`
- `inner_radius_p50`
- `inner_radius_p90`
- `model_weight`

### Cluster center back-transform

From cluster embedding center `(hx, hy, l)` derive:

- `center_H = atan2(hy, hx) / (2π)` normalized to `[0,1)`
- `center_S = sqrt(hx² + hy²)` clipped to `[0,1]`
- `center_L = l`

---

## Required time-series definitions

For each frame `t`, define `N_t` as the number of valid wafer-nonhole pixels in that frame.

### A. Hard frequency signal

For cluster `k`:

\[
f_k(t)=n_k(t)/N_t
\]

Where `n_k(t)` is the number of valid pixels with hard cluster assignment `k`.

#### Required file
`cluster_frequency_timeseries.csv`

Required fields:
- `frame_id`
- `time_s`
- `cluster_id`
- `cluster_rank`
- `frequency`
- `pixel_count`
- `region_area_px`

### B. Weighted frequency signal

For each pixel `i`, cluster `k`, compute a weight:

\[
w_{ik}=\exp\left(-\frac{1}{2}(d_{ik}/\tau_k)^2\right)
\]

where:
- `d_ik` is the Mahalanobis or covariance-aware distance to cluster `k`,
- `τ_k` is a scale derived from cluster covariance.

Required default signal:

\[
g_k(t)=\frac{1}{N_t}\sum_i r_{ik} w_{ik}
\]

where `r_ik` is the GMM posterior membership.

This is the required default because it reflects both:
- assignment uncertainty,
- closeness to the cluster core.

#### Required file
`cluster_weighted_frequency_timeseries.csv`

Required fields:
- `frame_id`
- `time_s`
- `cluster_id`
- `cluster_rank`
- `weighted_frequency`
- `weighted_mass`
- `region_area_px`

---

## Plotting directive

### 1. Hard frequency line plot

`cluster_frequency_over_time.png`

Must show:
- x-axis: `time_s`
- y-axis: `frequency`
- one line per cluster
- line colour taken from cluster display colour
- legend ordered by cluster mass rank

### 2. Weighted frequency line plot

`cluster_weighted_frequency_over_time.png`

Same rules, but with `weighted_frequency` on y-axis.

### 3. Pooled cloud summary

`pooled_hsl_cloud_summary.png`

Must not attempt a full 3D rendering of all points if unreadable.
Prefer a compact diagnostic such as:
- H vs L scatter of a sampled subset,
- or hx vs hy scatter plus lightness distribution,
- with cluster centers overlaid.

This plot is diagnostic, not the canonical numeric output.

---

## Pipeline integration directive

### Required stage insertion point in `holecolor/pipeline.py`

Insert the new stage **after** these existing writes:
- `global_buffer_timeseries.csv`
- `global_buffer_radial_profiles.csv`
- `global_buffer_band_profiles.csv`
- `global_buffer_region.json`

and **before** or alongside:
- `coupled_hole_buffer_timeseries.csv`

### Required integration behavior

The stage must:
1. assemble the wafer non-hole mask for each frame,
2. extract/store HSL datapoints,
3. fit the pooled cluster model once per run,
4. compute per-frame hard frequency rows,
5. compute per-frame weighted frequency rows,
6. write diagnostics,
7. expose outputs for later coupled interpretation.

### Coupling rule

Do **not** merge these outputs into `global_buffer_timeseries.csv`.

This must remain a **distinct branch** with its own artifact family.

---

## Suggested implementation skeleton

### In `holecolor/descriptors/wafer_nonhole_colour.py`

```python
def build_wafer_nonhole_mask(shape, wafer_geometry, selected_holes) -> np.ndarray: ...

def extract_frame_hsl_points(frame, mask) -> dict[str, np.ndarray] | pd.DataFrame: ...

def hsl_embedding(h, s, l) -> np.ndarray: ...

def fit_gmm_clusters(features, candidate_k, covariance_type="full") -> ClusterFitResult: ...

def summarize_clusters(points_df, fit_result) -> list[dict[str, Any]]: ...

def frame_cluster_frequency_rows(points_df, fit_result) -> list[dict[str, Any]]: ...

def frame_cluster_weighted_rows(points_df, fit_result) -> list[dict[str, Any]]: ...
```

### In `holecolor/pipeline.py`

Add a stage wrapper such as:

```python
def _run_wafer_nonhole_colour_stage(...):
    ...
```

and call it from `run_milestone3(...)`.

---

## Test directive

### A. Unit tests — point extraction

Create `holecolor/tests/test_wafer_nonhole_colour_points.py`

Required tests:
1. region mask excludes hole disks from wafer support
2. extracted point count equals mask sum
3. persisted HSL fields exist and remain bounded in `[0,1]`
4. hue embedding is stable for wrap-around hues

### B. Unit tests — clustering

Create `holecolor/tests/test_wafer_nonhole_colour_clustering.py`

Required tests:
1. synthetic two-colour wafer cloud yields two dominant clusters
2. cluster summary sorts by pooled mass descending
3. center back-transform to HSL is valid
4. posterior probabilities sum to ~1 per point

### C. Unit tests — time-series

Create `holecolor/tests/test_wafer_nonhole_colour_timeseries.py`

Required tests:
1. hard frequencies sum to ~1 across clusters per frame
2. weighted frequencies are bounded and non-negative
3. weighted signal decreases for points far from the cluster center
4. empty/too-small region triggers honest skip behavior

### D. Pipeline artifact test extension

Extend `holecolor/tests/test_pipeline_artifacts.py`

Add required artifact assertions for:
- `descriptors/wafer_nonhole_colour/region_definition.json`
- `descriptors/wafer_nonhole_colour/wafer_nonhole_hsl_points.parquet`
- `descriptors/wafer_nonhole_colour/cluster_model.json`
- `descriptors/wafer_nonhole_colour/cluster_summary.csv`
- `descriptors/wafer_nonhole_colour/cluster_frequency_timeseries.csv`
- `descriptors/wafer_nonhole_colour/cluster_weighted_frequency_timeseries.csv`
- `descriptors/wafer_nonhole_colour/stage_status.json`

### E. Smoke verification

Minimum focused verification command set:

```bash
pytest -q \
  holecolor/tests/test_wafer_nonhole_colour_points.py \
  holecolor/tests/test_wafer_nonhole_colour_clustering.py \
  holecolor/tests/test_wafer_nonhole_colour_timeseries.py \
  holecolor/tests/test_pipeline_artifacts.py
```

---

## Refactor governance directive

This new work must be recorded as an **M5 extension**.

### Required new actions in `refactor/REFactor_BACKLOG.md`

Add:

#### ACT-018 — Add global wafer non-hole HSL point-cloud extraction
Goal:
- persist canonical all-pixel HSL dataset for wafer-minus-holes region

#### ACT-019 — Add pooled dominant colour clustering over wafer non-hole cloud
Goal:
- define a small number of dominant global colour states with center + variability + mass

#### ACT-020 — Add hard and weighted cluster-frequency time-series
Goal:
- emit temporal lines for cluster prevalence in the wafer-minus-holes region

### Required ledger entry rule

For each action, record:
- expected effect
- observed effect
- regression check
- decision

### Required progress update

`REFactor_PROGRESS.md` must mention this branch explicitly, for example:

- current stable path + global wafer non-hole HSL cluster branch

---

## Invariants that must remain true

1. The exact-sequence hole detection logic must remain unchanged.
2. Existing `global_buffer_*` outputs must remain unchanged unless explicitly versioned.
3. Existing `matrix_timeseries.csv` must remain available until M6 consolidation.
4. The new layer must not silently alter hole selection policy.
5. If clustering is skipped, it must do so honestly and explicitly.
6. The canonical dataset for this layer is **wafer minus holes**, not buffer-only.

---

## Do-not-do list

1. **Do not** cluster on raw scalar hue.
2. **Do not** mix this branch into `global_buffer_timeseries.csv`.
3. **Do not** define the region as buffer-only.
4. **Do not** use hole seeds or predicted-only centers as hole masks unless the explicit exclusion policy says so.
5. **Do not** save only aggregated means; the raw canonical point cloud is required.
6. **Do not** silently downsample away the canonical dataset; only the fit subset may be sampled.

---

## Acceptance criteria

This directive pack is satisfied only when all of the following are true:

1. The pipeline writes a canonical `wafer_nonhole_hsl_points.parquet` for the wafer-minus-holes region.
2. The run writes a fitted cluster model and per-cluster summary with center + variability + mass.
3. The run writes both:
   - hard cluster frequency time-series,
   - weighted cluster frequency time-series.
4. The region overlay clearly shows wafer retained area with holes excluded.
5. Focused tests pass.
6. Existing pipeline artifacts still pass regression checks.
7. Refactor memory files are updated with ACT-018/019/020 and verification evidence.

---

## Recommended implementation order

1. Add config block and defaults.
2. Implement region-mask + raw HSL point extraction.
3. Persist canonical parquet + region overlay.
4. Add GMM fitting + cluster summary.
5. Add hard frequency time-series.
6. Add weighted frequency time-series.
7. Add plots.
8. Add tests.
9. Update refactor governance files.
10. Run focused verification.

---

## Final integration intent statement

The new branch must make the archive capable of representing the **global colour composition of the wafer, excluding holes, as a pooled HSL point cloud and a small set of dominant colour states**, then tracking those states over time by both hard occupancy and covariance-aware weighted occupancy.

This is the correct statistical counterpart to the existing hole-centric radial analysis, and it must remain a distinct, auditable, support-aligned branch in the ongoing refactor.
