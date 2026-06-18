# Refactor Decisions

## DEC-001 — Refactor memory must live inside the archive
Rationale:
- avoids drift between chat memory and repository state
- keeps plan, progress, and verification evidence co-located with code

## DEC-002 — Geometry must be first-class, not implicit
Rationale:
- the target architecture requires explicit wafer/buffer/hole objects
- masks should be derived from vector geometry, not be the only geometry representation

## DEC-003 — Tier 3 predicted holes remain explicit and separate
Rationale:
- geometry completion is useful, but it must not be silently merged with strong detections
- downstream analyses can choose conservative vs extended vs pattern-complete usage explicitly

## DEC-004 — Buffer border uncertainty must be modeled honestly
Rationale:
- border-based hole exclusion is only valid when border geometry is known
- partial border and off-frame centers are valid geometry states, not errors
