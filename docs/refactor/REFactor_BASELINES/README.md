# Refactor Baselines

Store frozen baseline outputs here for each smoke/reference dataset.

Recommended layout:

- `JAO25/`
  - baseline_summary.json
  - conservative_overlay.png
  - extended_overlay.png
  - pattern_extended_overlay.png
  - tiered_final_holes.csv
  - smoke_manifest.json

Rules:
- never overwrite a frozen baseline silently
- if a baseline is updated, record the reason in `REFactor_LEDGER.md`
- every baseline update must include before/after effect notes
