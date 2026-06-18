# Verification for ACT-008 and ACT-010

Timestamp: 2026-03-18T13:06:46Z

Command:
`/opt/pyvenv/bin/python3 -m pytest -q holecolor/tests/test_refactor_models.py holecolor/tests/test_hole_buffer_relations.py holecolor/tests/test_support_geometry.py holecolor/tests/test_support_stage_smoke.py holecolor/tests/test_global_buffer_scaffold.py holecolor/tests/test_global_buffer_radial_profiles.py holecolor/tests/test_pipeline_artifacts.py holecolor/tests/test_milestone53.py`

Observed output:
```text
...........
```

Interpretation:
- 11 focused tests passed
- radial-from-buffer-border profiles are covered by unit tests and pipeline-artifact checks
- border-aware selection is covered for unknown and partial-known border cases
