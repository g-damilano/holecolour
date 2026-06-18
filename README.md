# holecolor

`holecolor` is a classical image-analysis pipeline for colour and spatial-colour time-series analysis around regular hole lattices. It was built for microscopy-style videos where colour changes are measured around complete holes and their surrounding concentric terrace regions.

The package provides a command-line interface, reusable Python modules, generated notebooks, and traceable output artifacts for geometry, colour clustering, radial descriptors, temporal events, and quality control.

## Repository Layout

```text
holecolour/
  holecolor/                 Python package and tests
  docs/                      Historical notes, refactor docs, visual option references
  scripts/                   One-off exploratory/helper scripts
  packaging/                 Local app packaging helpers
  data/                      Local data staging area, ignored by Git
  local_outputs/             Local run outputs, ignored by Git
  archives/                  Local zip/package archives, ignored by Git
  pyproject.toml             Package metadata and dependencies
  pytest.ini                 Test configuration
  README.md                  This guide
```

Large videos, generated runs, caches, archives, and virtual environments are intentionally ignored by Git. Keep publishable source code, tests, and documentation in the tracked package/docs folders.

## Install

From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

If the `holecolor` command is not on your PATH, use:

```bash
python -m holecolor.cli
```

in place of `holecolor`.

## Quick Start

Put local videos under `data/videos/`. That folder is ignored by Git so raw data does not get published accidentally.

Run the full pipeline:

```bash
holecolor run "data/videos/JAO25_center_75_percent.mp4" --out "local_outputs/runs/JAO25_center_75_full" --workers 4
```

Run a faster iteration profile:

```bash
holecolor run "data/videos/JAO25_center_75_percent.mp4" --out "local_outputs/runs/JAO25_center_75_fast" --fast --workers 4
```

The `--fast` profile skips the heaviest validation sweeps and reduces repeated/bootstrap work. Use it while tuning inputs, then run without `--fast` for final results.

## Core CLI Commands

Run the whole analysis:

```bash
holecolor run INPUT_VIDEO --out OUTPUT_DIR
```

Run only frame quality audit:

```bash
holecolor audit INPUT_VIDEO --out OUTPUT_DIR
```

Run only photometry correction selection:

```bash
holecolor photometry INPUT_VIDEO --out OUTPUT_DIR
```

Run only geometry calibration:

```bash
holecolor holegrid INPUT_VIDEO --out OUTPUT_DIR --sample-frames 8
```

Reuse a saved hole-grid model:

```bash
holecolor run INPUT_VIDEO --out OUTPUT_DIR --hole-grid-model path/to/hole_grid_model.json
```

Regenerate the generated results notebook for an existing run:

```bash
holecolor notebook OUTPUT_DIR
```

Monitor a running pipeline from a second terminal:

```bash
holecolor status OUTPUT_DIR --watch
```

Show command-specific help:

```bash
holecolor --help
holecolor run --help
holecolor holegrid --help
```

## Important Flags

| Flag | Meaning |
| --- | --- |
| `--out DIR` | Output directory. Prefer `local_outputs/runs/...` for local work. |
| `--every-n N` | Load every Nth frame. Use `1` for final analysis; larger values are useful for smoke tests. |
| `--workers N` | Number of parallel workers. `0` uses automatic selection. |
| `--parallel-backend auto\|process\|thread\|none` | Parallel backend. `auto` is default; `thread` is often friendly on Windows; `none` is easiest to debug. |
| `--fast` | Lighter iteration profile. |
| `--strict-gates` | Treat QC gate failures as hard failures. |
| `--acceptance-gates` | After a run, prompt for Y/N review of key visual artifacts. |
| `--trim-ui` | Open a visual start/end frame selector before processing. |
| `--start-frame N`, `--end-frame N` | Process an inclusive frame range. |
| `--trim-selection PATH` | Reuse a saved frame trim JSON. |
| `--no-black-band-crop` | Disable automatic top/bottom black-band cropping. |
| `--hole-grid-model PATH` | Reuse an existing geometry model. |

## Preprocessing Rules

### Black-Band Cropping

Videos may contain black letterbox bands above or below the usable image. By default, the CLI detects consistent top and bottom black bands before analysis and crops them away. The decision is logged to:

```text
OUTPUT_DIR/logs/black_band_crop.json
```

The detector uses row luminance, dark-pixel fraction, row mean, and row standard deviation across sampled frames. Cropping is conservative and can be disabled with `--no-black-band-crop`.

### Complete-Hole Filtering

Only complete, reliable hole regions are analysed. A detected hole is excluded if:

- the hole disk is cut off by the usable video frame
- the planned outer terrace disk extends beyond the usable video frame
- the hole disk falls outside wafer support
- the planned outer terrace disk falls outside wafer support

Because terraces are concentric annuli, requiring the outermost terrace disk to fit guarantees that all inner terraces are complete. Exclusion decisions are written to:

```text
OUTPUT_DIR/geometry/hole_terrace_exclusion.csv
OUTPUT_DIR/geometry/hole_terrace_exclusion_summary.json
```

This strict filter is intentional. It is better to analyse fewer complete holes than to include partial holes or clipped terrace regions that would bias measurements.

## Output Guide

Start with the curated output layer:

```text
OUTPUT_DIR/outputs/START_HERE.md
OUTPUT_DIR/outputs/index.html
OUTPUT_DIR/outputs/output_manifest.json
```

Then inspect these core outputs:

```text
OUTPUT_DIR/summary.json
OUTPUT_DIR/run_manifest.json
OUTPUT_DIR/logs/current_status.json
OUTPUT_DIR/logs/run_status.jsonl
OUTPUT_DIR/geometry/overlays/frame_ref_geometry_overlay.png
OUTPUT_DIR/geometry/geometry_sanity_checks.json
OUTPUT_DIR/geometry/hole_geometry.csv
OUTPUT_DIR/geometry/hole_lattice_index.csv
OUTPUT_DIR/geometry/hole_terrace_exclusion.csv
OUTPUT_DIR/descriptors/hole_compartment_timeseries.csv
OUTPUT_DIR/descriptors/wafer_nonhole_colour/frame_cluster_summary.csv
OUTPUT_DIR/descriptors/wafer_nonhole_colour/video_cluster_baseline_activity.avi
OUTPUT_DIR/radial/hole_annulus_timeseries.csv
OUTPUT_DIR/notebooks/holecolor_results_explorer.ipynb
```

Recommended review order:

1. Confirm the reference geometry overlay and `geometry_sanity_checks.json`.
2. Check `hole_terrace_exclusion_summary.json` to verify enough complete holes remain.
3. Inspect colour clustering outputs and the baseline activity video.
4. Review radial and terrace outputs only after geometry and colour clustering look valid.
5. Use the generated notebook for exploratory review.

## Jupyter Usage

You can clone the public repository directly inside a Jupyter notebook or JupyterLab terminal:

```python
!git clone https://github.com/g-damilano/holecolour.git
%cd holecolour
!python -m pip install --upgrade pip
!python -m pip install -e .
```

Confirm the CLI is available:

```python
!python -m holecolor.cli --help
```

The repository does not include raw videos. Upload or copy your analysis videos into `data/videos/` in the notebook environment before running the pipeline:

```python
from pathlib import Path

Path("data/videos").mkdir(parents=True, exist_ok=True)
```

Then run the CLI from a notebook cell:

```python
!python -m holecolor.cli run "data/videos/JAO25_center_75_percent.mp4" --out "local_outputs/runs/notebook_run" --fast --workers 4
```

For a final full run, remove `--fast`:

```python
!python -m holecolor.cli run "data/videos/JAO25_center_75_percent.mp4" --out "local_outputs/runs/notebook_full" --workers 4
```

You can also call the CLI entrypoint from Python:

```python
import sys
from holecolor.cli import main

sys.argv = [
    "holecolor",
    "run",
    "data/videos/JAO25_center_75_percent.mp4",
    "--out", "local_outputs/runs/notebook_run",
    "--fast",
    "--workers", "4",
]

main()
```

The shell-style notebook command is usually closest to command-line behavior and is the recommended notebook workflow.

## Development

Install development dependencies:

```bash
python -m pip install -e ".[dev]"
```

Run focused tests:

```bash
python -m pytest holecolor/tests/test_video_io.py holecolor/tests/test_terrace_width_plan.py -q
```

Run the full test suite:

```bash
python -m pytest -q
```

Compile-check the package:

```bash
python -m compileall -q holecolor
```

## Publishing Checklist

Before pushing to GitHub:

1. Keep raw videos in `data/videos/` and generated outputs in `local_outputs/`.
2. Confirm `.gitignore` excludes caches, bytecode, virtualenvs, data, archives, and run outputs.
3. Remove or keep local-only archives under `archives/`; they are ignored by Git.
4. Run the focused tests and at least one CLI smoke test.
5. Review `git status --short` before committing.
6. Do not commit private data, large videos, generated notebooks from local runs, or packaged zip archives unless intentionally publishing a release artifact.

## License

Add a `LICENSE` file before publishing if this repository will be public. Choose a license that matches the intended sharing model for the code and any associated data.
