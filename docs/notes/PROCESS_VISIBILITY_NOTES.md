# Process visibility notes

This build hardens run visibility in two ways:

1. **Pipeline-stage visibility in the CLI**
   - a master `Pipeline stages` progress bar is shown for the major stages of `holecolor run` / `holecolor radial`
   - existing detailed progress bars remain for long iterable stages such as frame loading, audit, registration, geometry propagation, per-frame analysis, hotspot tracking, and perturbation sweeps

2. **Persistent status files written during the run**
   - `logs/current_status.json` always contains the latest known stage/state
   - `logs/run_status.jsonl` contains append-only stage events
   - `logs/stage_timings.json` contains elapsed times for completed stages

This is designed so users can verify that the process is still alive even if the terminal output scrollback is noisy or they are monitoring the run from another window.


Milestone 23 visibility additions
- Every pipeline stage now has its own live stage bar, not just the master pipeline bar.
- Long-running stages update a persistent heartbeat in `logs/current_status.json` and append events to `logs/run_status.jsonl`.
- The current status payload now includes elapsed time, ETA when available, stage fraction, and overall pipeline fraction.
- Silent aggregate work was split into explicit visible stages: hotspot linking, aggregate outputs, and QC/manifest.
