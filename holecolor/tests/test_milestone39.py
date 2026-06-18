from holecolor.radial.columnar import RadialRowTable
from holecolor.temporal.columnar import (
    PhenotypeTable,
    TemporalValidationSummary,
    build_phenotype_archetype_rows_table,
    build_phenotype_neighbor_and_smoothness_table,
)
from holecolor.temporal.phenotypes import (
    build_phenotype_archetype_rows,
    phenotype_neighbor_coherence,
    phenotype_spatial_smoothness,
)


def test_milestone39_temporal_tables_match_row_based_helpers():
    phenotype_rows = [
        {"hole_id": 1, "lattice_u": 0, "lattice_v": 0, "phenotype_label": "P1_a", "semantic_label": "a", "canonical_id": 1, "canonical_label": "P1_a"},
        {"hole_id": 2, "lattice_u": 1, "lattice_v": 0, "phenotype_label": "P1_a", "semantic_label": "a", "canonical_id": 1, "canonical_label": "P1_a"},
        {"hole_id": 3, "lattice_u": 0, "lattice_v": 1, "phenotype_label": "P2_b", "semantic_label": "b", "canonical_id": 2, "canonical_label": "P2_b"},
    ]
    radial_rows = [
        {"hole_id": 1, "frame_id": 0, "annulus_id": 0, "descriptor_value": 1.0},
        {"hole_id": 2, "frame_id": 0, "annulus_id": 0, "descriptor_value": 3.0},
        {"hole_id": 3, "frame_id": 0, "annulus_id": 0, "descriptor_value": 5.0},
        {"hole_id": 1, "frame_id": 1, "annulus_id": 1, "descriptor_value": 2.0},
        {"hole_id": 2, "frame_id": 1, "annulus_id": 1, "descriptor_value": 4.0},
        {"hole_id": 3, "frame_id": 1, "annulus_id": 1, "descriptor_value": 6.0},
    ]
    table = PhenotypeTable.from_rows(phenotype_rows)
    coherence_new, neighbors_new, smooth_new, smooth_rows_new, smooth_summary_new = build_phenotype_neighbor_and_smoothness_table(table, radius=2)
    coherence_old, neighbors_old = phenotype_neighbor_coherence(phenotype_rows)
    smooth_old, smooth_rows_old, smooth_summary_old = phenotype_spatial_smoothness(phenotype_rows, radius=2)
    assert coherence_new == coherence_old
    assert len(neighbors_new) == len(neighbors_old)
    assert abs(smooth_new - smooth_old) < 1e-9
    assert len(smooth_rows_new) == len(smooth_rows_old)
    assert smooth_summary_new["mean_neighbor_count"] == smooth_summary_old["mean_neighbor_count"]

    arche_new = build_phenotype_archetype_rows_table(RadialRowTable.from_rows(radial_rows), table)
    arche_old = build_phenotype_archetype_rows(radial_rows, phenotype_rows)
    assert arche_new == arche_old


def test_milestone39_temporal_validation_summary_matches_direct_fractions():
    per_hole_summary_rows = [
        {"n_valid_annuli_onset": 3, "n_valid_annuli_peak": 2, "monotonic_onset": True, "monotonic_peak": True},
        {"n_valid_annuli_onset": 1, "n_valid_annuli_peak": 0, "monotonic_onset": False, "monotonic_peak": True},
    ]
    propagation_rows = [{"negative_lag_flag": False}, {"negative_lag_flag": True}]
    stability_summary = {"mean_stability_fraction": 0.8, "canonical_agreement_fraction": 0.6}
    summary = TemporalValidationSummary.from_rows(
        per_hole_summary_rows,
        propagation_rows,
        stability_summary,
        coherence_fraction=0.5,
        spatial_smoothness=0.75,
        min_valid_annuli_onset=2,
        min_valid_annuli_peak=1,
    )
    assert summary.valid_onset_fraction == 0.5
    assert summary.valid_peak_fraction == 0.5
    assert summary.lag_monotonic_fraction == 0.5
    assert summary.negative_lag_fraction == 0.5
    assert summary.phenotype_stability_fraction == 0.8
    assert summary.phenotype_canonical_agreement == 0.6
    assert summary.phenotype_neighbor_fraction == 0.5
    assert summary.phenotype_spatial_fraction == 0.75
