from holecolor.pipeline import _write_rdf_bootstrap_plot, _write_rdf_bootstrap_support_plot, _write_rdf_stability_plot
from holecolor.radial.columnar import ValidationHoleTable


def test_milestone46_validation_plot_helpers_accept_column_payloads(tmp_path):
    table = ValidationHoleTable.from_rows(
        rdf_bootstrap_rows=[{
            "hole_id": 1,
            "bootstrap_front_velocity_ci_width": 0.2,
            "bootstrap_mean_front_radius_ci_width": 0.1,
            "bootstrap_delta_front_radius_ci_width": 0.15,
        }],
        rdf_bootstrap_support_rows=[{
            "hole_id": 1,
            "bootstrap_rdf_archetype_support_fraction": 0.8,
        }],
    )
    _write_rdf_bootstrap_plot(tmp_path, table.bootstrap_ci_plot_columns())
    _write_rdf_bootstrap_support_plot(tmp_path, table.bootstrap_support_plot_columns())
    _write_rdf_stability_plot(tmp_path, {
        "brightness_factor": [1.0, 1.1],
        "radius_scale": [1.0, 1.0],
        "rdf_archetype_stability_fraction": [1.0, 0.75],
        "is_base": [True, False],
    })
    assert (tmp_path / "per_hole_rdf_bootstrap_ci.png").exists()
    assert (tmp_path / "rdf_archetype_bootstrap_support.png").exists()
    assert (tmp_path / "per_hole_rdf_stability.png").exists()
