# Getting Started

This page walks through installing NeuroPose, running your first pose
estimation, and understanding the output. It targets researchers who are
comfortable on a Linux command line but may not have used the package
before.

!!! info "Model loader status"
    The MeTRAbs model loader is pending the commit-11 rewrite, during
    which the upstream model URL and TensorFlow version will be pinned.
    Until it lands, the `neuropose watch` and `neuropose process`
    commands will exit with a clear "pending commit 11" message. The
    Python API still works if you inject a model manually — see the
    *Python API* section below for the current workaround.

## Prerequisites

- Linux (Ubuntu 22.04+ or equivalent)
- Python 3.11
- [`uv`](https://github.com/astral-sh/uv) for dependency management
- CUDA-capable GPU (optional, recommended for long videos)
- Internet access on first run (for the model download, once the loader
  lands)

## Installation

Clone the repository and install in editable mode:

```bash
git clone https://git.levineuwirth.org/neuwirth/neuropose.git
cd neuropose
uv venv --python 3.11
source .venv/bin/activate
uv sync --group dev
```

`uv sync --group dev` installs the runtime dependencies (pydantic, typer,
OpenCV, TensorFlow, matplotlib) plus the dev tooling (pytest, ruff,
pyright, pre-commit, mkdocs-material). The first run will download
TensorFlow, which is roughly 600 MB; subsequent runs hit the uv cache.

Confirm the CLI is installed:

```bash
neuropose --version
# neuropose 0.1.0.dev0
```

## Configuration

NeuroPose reads configuration from one of three sources, in order of
decreasing precedence:

1. A YAML file passed via `--config`.
2. Environment variables prefixed with `NEUROPOSE_` (e.g.
   `NEUROPOSE_DEVICE=/GPU:0`).
3. Built-in defaults.

The default runtime data directory is `$XDG_DATA_HOME/neuropose/jobs`
(typically `~/.local/share/neuropose/jobs`). Runtime data never lives
inside the repository.

A complete example config:

```yaml title="config.yaml"
# TensorFlow device string. "/CPU:0" or "/GPU:N".
device: "/GPU:0"

# Base directory for job inputs, outputs, and failed quarantine.
data_dir: "/srv/neuropose/jobs"

# Where the MeTRAbs model is cached after download.
model_cache_dir: "/srv/neuropose/models"

# How often the interfacer daemon scans the input directory.
poll_interval_seconds: 10

# Horizontal field-of-view passed to MeTRAbs. Override per call if you
# know the camera intrinsics; otherwise MeTRAbs's 55° default is fine.
default_fov_degrees: 55.0
```

See the [`neuropose.config`](api/config.md) API reference for the full
list of fields and their validation rules.

## Processing a single video

The `process` subcommand is the quickest way to run the estimator on one
video:

```bash
neuropose process path/to/video.mp4
```

By default this writes `<video-stem>_predictions.json` in the current
working directory. Override with `--output`:

```bash
neuropose process path/to/video.mp4 --output /srv/results/trial_01.json
```

## Running the daemon

For batch processing, use the `watch` subcommand. Point a config at a
data directory, drop videos into job subdirectories under `data_dir/in/`,
and the daemon processes each one in order.

```bash
# 1. Prepare the data directory
neuropose --config ./config.yaml watch &

# 2. In another shell, add a job
mkdir -p /srv/neuropose/jobs/in/trial_01
cp video_01.mp4 video_02.mp4 /srv/neuropose/jobs/in/trial_01/

# 3. The daemon will pick it up within poll_interval_seconds
#    and write /srv/neuropose/jobs/out/trial_01/results.json
```

The daemon writes a persistent `status.json` tracking every job's
lifecycle. On startup, any jobs left in the `processing` state from a
previous crash are marked failed and their inputs are moved to
`data_dir/failed/` for operator review. See the
[`neuropose.interfacer`](api/interfacer.md) API reference for the full
lifecycle contract.

Stop the daemon with `Ctrl-C` or `kill -TERM <pid>`. The current job
finishes before the loop exits.

## Output schema

Each processed video produces a JSON file with the following shape:

```json
{
  "metadata": {
    "frame_count": 180,
    "fps": 30.0,
    "width": 1920,
    "height": 1080
  },
  "frames": {
    "frame_000000": {
      "boxes": [[10.2, 20.5, 200.0, 400.0, 0.97]],
      "poses3d": [[[x, y, z], ...]],
      "poses2d": [[[x, y], ...]]
    },
    "frame_000001": { ... }
  }
}
```

Key details:

- **Frame identifiers** are `frame_000000`, `frame_000001`, ...
  (six-digit zero-padded). These are identifiers, not filenames — no
  PNG files exist on disk.
- **`boxes`** are `[x, y, width, height, confidence]` in pixels.
- **`poses3d`** are `[x, y, z]` in millimetres, per the MeTRAbs
  convention.
- **`poses2d`** are `[x, y]` in pixels.
- **`metadata`** carries the source video's frame count, fps, and
  resolution. This is essential for reproducibility — downstream
  analysis can convert frame indices to real time without needing the
  original video file.

Use [`neuropose.io.load_video_predictions`](api/io.md) to read the JSON
back into a validated `VideoPredictions` object.

## Python API

For scripting, debugging, or integrating NeuroPose into a larger
pipeline, you can use the `Estimator` class directly. This is also the
current workaround for the pending model loader:

```python
from neuropose.estimator import Estimator
from neuropose.io import save_video_predictions
from pathlib import Path

# Load the MeTRAbs model however you like — e.g. via tensorflow_hub once
# you know the canonical URL. Until commit 11 pins it, you'll need to
# load it yourself here.
import tensorflow_hub as tfhub
model = tfhub.load("...")  # TODO: pin upstream URL

estimator = Estimator(model=model, device="/GPU:0")
result = estimator.process_video(Path("trial_01.mp4"))

print(f"Processed {result.frame_count} frames")
save_video_predictions(Path("trial_01_predictions.json"), result.predictions)
```

You can also wire up a progress callback for long videos:

```python
from rich.progress import Progress

with Progress() as progress:
    task = progress.add_task("Processing", total=None)
    result = estimator.process_video(
        Path("trial_01.mp4"),
        progress=lambda processed, total_hint: progress.update(task, completed=processed),
    )
```

## Visualization

To generate per-frame overlay images (2D skeleton on the source frame
plus a 3D scatter plot), use `neuropose.visualize`:

```python
from neuropose.visualize import visualize_predictions

visualize_predictions(
    video_path=Path("trial_01.mp4"),
    predictions=result.predictions,
    output_dir=Path("trial_01_viz/"),
    frame_indices=[0, 30, 60, 90],  # pick a handful of frames for spot-checking
)
```

Visualization is a separate module to keep the estimator's import graph
free of matplotlib. Matplotlib's `Agg` backend is set inside the
function, so importing `neuropose.visualize` has no global side effects.

## Troubleshooting

| Problem | Resolution |
|---|---|
| `error: pending commit 11` from `neuropose watch` or `process` | The model loader is not yet implemented. Use the Python API with a manually-loaded model. |
| `AlreadyRunningError` from the daemon | Another NeuroPose daemon already holds the lock file. Check `data_dir/.neuropose.lock` for the PID. |
| `VideoDecodeError` on valid-looking video | The file may be corrupted or in a codec OpenCV was built without. Try re-encoding with `ffmpeg -i in.mov -c:v libx264 out.mp4`. |
| Jobs stuck in `processing` state on startup | The daemon now recovers these automatically — they'll be marked failed and quarantined to `data_dir/failed/` on the next run. |
| Daemon not detecting a new job | Check that the job is inside a **subdirectory** of `data_dir/in/`, not directly in `data_dir/in/`. Empty subdirectories are silently skipped (the daemon assumes you are still copying files). |
