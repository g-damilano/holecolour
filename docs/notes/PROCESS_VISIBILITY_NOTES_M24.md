# Process visibility notes (Milestone 24)

This build extends CLI visibility beyond the live bars:

- `logs/stage_plan.json` is written at run start with the ordered stage list.
- `logs/current_status.json` mirrors the latest machine-readable state.
- `logs/run_status.jsonl` appends every status event.
- `logs/progress_summary.txt` is updated continuously with a human-readable snapshot.
- `holecolor status RUN_DIR` prints a compact one-line status summary.
- `holecolor status RUN_DIR --watch` tails the current run state until completion.

The goal is that a user can verify liveness in three ways:

1. directly in the original CLI via tqdm stage bars,
2. by inspecting the live files inside `run_dir/logs/`,
3. from another terminal using the `status` subcommand.
