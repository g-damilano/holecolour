# Architecture split notes (Milestone 26)

This milestone starts disentangling the package into:

1. **colour analysis**
   - can run without any hole model
   - entrypoint: `holecolor color INPUT --out OUTDIR`

2. **hole-grid calibration**
   - dedicated calibration/export path for the regular hole lattice
   - entrypoint: `holecolor holegrid INPUT --out OUTDIR`
   - writes `hole_grid_model.json`

3. **hole-centered radial analysis**
   - consumes the saved grid bundle when provided
   - entrypoints: `holecolor run ... --hole-grid-model model.json`
     and `holecolor radial ... --hole-grid-model model.json`

The hole-grid model is intended to become the contract between a future GUI-assisted
calibration tool and the main analysis pipeline.
