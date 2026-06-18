# Refactor Master Plan

## North-star target

The target architecture is a **vector-structured, support-centric pipeline** in which:

- wafer geometry
- buffer geometry
- hole tiers
- hole-buffer relations

are explicit first-class objects, while:

- local hole analysis
- global buffer analysis
- coupled local/global analysis

are separate verified branches.

Every refactor action must be recorded, verified, and compared against frozen baselines.

## Governance rules

Every atomic action must produce:

1. state update  
2. verification evidence  
3. effect statement  
4. progress update  

No silent edits.

## Milestones

### M0 — Freeze baseline
Goal:
- freeze current archive behavior and outputs for smoke datasets
- establish a stable rollback point

### M1 — Formalize geometry objects
Goal:
- separate geometry from analysis

Target objects:
- `WaferGeometry`
- `BufferGeometry`
- `LatticeModel`
- `HoleRecord`
- `HoleTier`
- `HoleBufferRelation`

### M2 — Insert explicit support geometry stage
Goal:
- add a stage dedicated to wafer border and buffer border geometry

Must support:
- full border in frame
- partial arc in frame
- center outside frame
- unknown state

### M3 — Recast hole detection as tiered geometry
Goal:
- the developed detection logic becomes the official tiered hole stage

### M4 — Add hole-buffer classification layer
Goal:
- classify each hole relative to the buffer border

### M5 — Split local and global analysis
Goal:
- separate local hole-centered analysis from global buffer analysis

### M6 — Consolidate and remove compatibility shims
Goal:
- retire transitional glue after verification

## Verification classes

### Structural verification
Examples:
- new dataclass exists
- stage wired into pipeline
- manifest includes new outputs

### Behavioral verification
Examples:
- counts or classifications change as expected
- new files emitted
- new geometry objects available to downstream stages

### Regression verification
Examples:
- smoke run still completes
- required legacy artifacts still exist until retired
- reference geometry still produces usable lattice

## Required effect report format

For every action, record:
- expected effect
- observed effect
- regression check
- decision (keep / revise / rollback)

## Current baseline source

- base archive: `holecolor_milestone53_v0_53_exact_sequence_patch.zip`
- initialization time: `2026-03-17T14:06:15Z`
