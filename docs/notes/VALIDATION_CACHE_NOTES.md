# Validation cache notes

This build adds a validation-cache manifest tied to the key downstream inputs of the expensive validation stage.

## What is cached
- RDF archetype perturbation sweep outputs
- RDF bootstrap summary outputs
- RDF bootstrap class-support outputs
- sector propagation and acceleration summaries
- uncertainty comparison tables
- radial perturbation sweep outputs

## Cache behavior
- Cache is keyed by a signature of the chosen descriptor, validation configuration, radial conclusion label, and digests of the main downstream input tables.
- On a rerun into the same output directory, if the signature matches and the expected files exist, the validation stage loads the cached outputs instead of recomputing them.
- The cache manifest is written to `qc/validation_cache_manifest.json`.

## Effect
This improves full-validation reruns and makes the validation stage much more responsive when only the final packaging or report generation is being repeated.
