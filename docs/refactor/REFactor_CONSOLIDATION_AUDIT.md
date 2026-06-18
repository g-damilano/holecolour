# Refactor Consolidation Audit

## Scope
This audit covers the currently explicit legacy milestone compatibility layer.

## Canonical runtime entrypoints
- `run_milestone1`
- `run_milestone16`
- `run_milestone18`

## Legacy alias groups
### Forwarding to milestone16
- `run_milestone2`
- `run_milestone3`
- `run_milestone4`
- `run_milestone7`
- `run_milestone8`
- `run_milestone9`
- `run_milestone10`
- `run_milestone11`
- `run_milestone12`
- `run_milestone13`
- `run_milestone14`
- `run_milestone15`

### Forwarding to milestone18
- `run_milestone19`
- `run_milestone20`
- `run_milestone21`
- `run_milestone22`
- `run_milestone23`
- `run_milestone24`

## Safe reductions made in this round
- centralized forwarding through `_run_legacy_alias_to_milestone16(...)`
- centralized forwarding through `_run_legacy_alias_to_milestone18(...)`
- preserved public wrapper names and behavior

## Remaining consolidation targets
- document and eventually reduce milestone16->17->18 alias chain if safe
- review duplicated artifact-writing boundaries
- retire redundant public milestone aliases only when downstream callers are known

## Important note
This round reduces duplication in the compatibility layer, but does **not** remove public aliases yet.
