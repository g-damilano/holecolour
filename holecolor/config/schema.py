from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass(slots=True)
class AuditConfig:
    blur_method: Literal["wavelet", "laplacian_var"] = "wavelet"
    blur_threshold: float = 0.8
    saturation_threshold: int = 250
    frame_jump_hist_bins: int = 64


@dataclass(slots=True)
class PhotometryConfig:
    candidate_methods: tuple[str, ...] = ("none", "gain_norm", "flatfield_poly")
    border_width_px: int = 24
    hole_control_weight: float = 1.0


@dataclass(slots=True)
class RegistrationConfig:
    mode: Literal["rigid", "affine"] = "rigid"
    max_shift_px: int = 50
    reference_frame: Literal["first", "best_qc"] = "best_qc"


@dataclass(slots=True)
class GeometryConfig:
    dark_threshold_quantile: float = 0.20
    min_radius_px: float = 6.0
    max_radius_px: float = 80.0
    min_confidence: float = 0.45
    lattice_ransac_iters: int = 500
    duplicate_suppression_px: float = 0.6
    angle_tolerance_deg: float = 18.0
    propagation_mode: Literal["canonical_static", "dynamic"] = "canonical_static"
    propagation_search_px: float = 6.0
    smoothing_window: int = 3
    detector_watchdog_s: float = 1200.0
    detector_branch_watchdog_s: float = 300.0
    fail_on_sanity_failure: bool = True
    use_photometry_fallback_for_geometry: bool = False


@dataclass(slots=True)
class MaskConfig:
    interior_shrink_px: int = 2
    rim_width_px: int = 2
    n_terraces: int = 8
    terrace_width_mode: Literal["half_gap", "full_gap", "fixed"] = "half_gap"
    terrace_gap_basis: Literal["border_gap", "center_pitch"] = "border_gap"
    terrace_min_width_px: float = 0.0


@dataclass(slots=True)
class DescriptorConfig:
    primary_color_space: Literal["auto", "rgb", "hsv", "hsl"] = "auto"
    primary_descriptor: Literal["auto", "r", "g", "b", "h", "s"] = "auto"
    low_saturation_hue_mask: float = 0.05


@dataclass(slots=True)
class HotspotConfig:
    score_channel: Literal["h", "s", "r", "g", "b", "delta"] = "s"
    threshold_mode: Literal["percentile", "otsu", "zscore"] = "percentile"
    threshold_value: float = 95.0
    min_area_px: int = 12
    link_max_dist_px: float = 18.0
    max_area_ratio: float = 3.0
    min_score_for_tracking: float = 0.0


@dataclass(slots=True)
class RadialConfig:
    perturb_radius_pct: float = 0.05
    perturb_terrace_px: int = 1
    max_mae_threshold: float = 0.25
    max_max_abs_threshold: float = 0.60
    archetype_k: int = 3
    angular_n_sectors: int = 8
    min_asymmetry_valid_fraction: float = 0.50
    min_rdf_archetype_stability: float = 0.60
    sector_lag_onset_threshold: float = 0.01
    rdf_bootstrap_n: int = 128
    min_rdf_bootstrap_class_support: float = 0.55
    min_sector_propagation_valid_fraction: float = 0.50
    min_sector_acceleration_valid_fraction: float = 0.40


@dataclass(slots=True)
class ParallelConfig:
    enabled: bool = True
    backend: Literal["auto", "process", "thread", "none"] = "auto"
    max_workers: int = 0
    min_parallel_tasks: int = 12
    chunksize: int = 1
    show_progress: bool = True
    progress_leave: bool = False
    progress_mininterval_s: float = 0.1
    status_heartbeat_interval_s: float = 0.5
    opencv_threads_per_worker: int = 1


@dataclass(slots=True)
class CheckpointConfig:
    enabled: bool = True
    reuse: bool = True
    write: bool = True


@dataclass(slots=True)
class QCConfig:
    fail_on_gate_error: bool = False


@dataclass(slots=True)
class TemporalConfig:
    cluster_k: int = 3
    min_valid_annuli_onset: int = 3
    min_valid_annuli_peak: int = 3
    min_monotonic_fraction: float = 0.60
    max_negative_lag_fraction: float = 0.40
    stability_reruns: int = 5
    stability_jitter_scale: float = 0.03
    min_phenotype_stability: float = 0.60
    min_neighbor_coherence: float = 0.20
    min_canonical_agreement: float = 0.50
    spatial_neighbor_radius: int = 2
    min_spatial_smoothness: float = 0.20





@dataclass(slots=True)
class WaferNonholeColourConfig:
    enabled: bool = True
    fail_open: bool = True
    max_points_per_frame: int = 5000
    max_total_fit_points: int = 4000
    min_total_points: int = 500
    gmm_k_min: int = 2
    gmm_k_max: int = 8
    random_state: int = 0
    write_recolour_video: bool = True
    write_side_by_side_video: bool = True
    write_labelmap_video: bool = True
    write_baseline_activity_video: bool = True


@dataclass(slots=True)
class RadialClusterAverageHoleConfig:
    enabled: bool = True
    fail_open: bool = True
    n_angle_sectors: int = 8
    front_threshold_fraction: float = 0.15


@dataclass(slots=True)
class ClusterRdfVisualisationConfig:
    enabled: bool = True
    fail_open: bool = True
    baseline_frames: int = 3
    terrace_display_base: int = 1
    time_axis: Literal["frame", "time_s"] = "time_s"
    generate_montage: bool = True
    generate_raw_rdf: bool = True
    generate_active_rdf: bool = True
    generate_dominant_chronogram: bool = True
    generate_radial_snapshots: bool = True
    generate_average_hole_rings: bool = True
    generate_sector_fan: bool = True
    generate_hole_barcode: bool = True
    generate_front_trajectory: bool = True
    generate_cluster_legend: bool = True
    generate_wafer_ring_glyph: bool = False


@dataclass(slots=True)
class VisualisationConfig:
    cluster_rdf: ClusterRdfVisualisationConfig = field(default_factory=ClusterRdfVisualisationConfig)


@dataclass(slots=True)
class ValidationConfig:
    enabled: bool = True
    brightness_factors: tuple[float, ...] = (0.90, 1.00, 1.10)
    radius_scale_factors: tuple[float, ...] = (0.95, 1.00, 1.05)
    min_conclusion_agreement: float = 0.66
    min_mean_profile_correlation: float = 0.70


@dataclass(slots=True)
class PipelineConfig:
    audit: AuditConfig = field(default_factory=AuditConfig)
    photometry: PhotometryConfig = field(default_factory=PhotometryConfig)
    registration: RegistrationConfig = field(default_factory=RegistrationConfig)
    geometry: GeometryConfig = field(default_factory=GeometryConfig)
    masks: MaskConfig = field(default_factory=MaskConfig)
    descriptors: DescriptorConfig = field(default_factory=DescriptorConfig)
    hotspots: HotspotConfig = field(default_factory=HotspotConfig)
    radial: RadialConfig = field(default_factory=RadialConfig)
    temporal: TemporalConfig = field(default_factory=TemporalConfig)
    parallel: ParallelConfig = field(default_factory=ParallelConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    qc: QCConfig = field(default_factory=QCConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    wafer_nonhole_colour: WaferNonholeColourConfig = field(default_factory=WaferNonholeColourConfig)
    radial_cluster_average_hole: RadialClusterAverageHoleConfig = field(default_factory=RadialClusterAverageHoleConfig)
    visualisation: VisualisationConfig = field(default_factory=VisualisationConfig)
