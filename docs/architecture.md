# Architecture

This page describes how NeuroPose is structured and why. It is the
document to read if you are about to modify the estimator, the daemon,
or the output schema, and want to understand the constraints the
existing design is trying to honour.

## Component overview

NeuroPose is a three-stage pipeline:

```text
┌───────────────────┐     ┌──────────────────┐     ┌───────────────────┐
│   interfacer      │     │   estimator      │     │    analyzer       │
│   (daemon)        │────▶│   (inference)    │────▶│   (post-process)  │
│                   │     │                  │     │                   │
│ watches filesystem│     │ MeTRAbs wrapper  │     │ DTW, features,    │
│ manages job state │     │ per-video worker │     │  classification   │
└───────────────────┘     └──────────────────┘     └───────────────────┘
       │                           │                       │
       ▼                           ▼                       ▼
 status.json +            VideoPredictions            analysis results
 job directories          (validated schema)          (pending commit 10)
```

Each stage is a separate module with one job, and the contracts between
them are defined by validated pydantic schemas in
[`neuropose.io`](api/io.md).

### estimator

**Role:** pure inference library. Given a video path and a MeTRAbs
model, produces a validated `VideoPredictions` object plus a
`PerformanceMetrics` bundle.

**Does NOT handle:** job directories, status files, polling, locking,
signal handling, visualization, or anywhere-to-save decisions. It is a
library, not a daemon.

The estimator streams frames directly from OpenCV into the model — no
intermediate write-to-disk-then-read-back-as-PNG round trip like the
previous prototype had. `process_video()` returns a typed
`ProcessVideoResult` containing the predictions and an
always-populated `PerformanceMetrics` (per-frame latency, peak RSS,
total wall clock, active TF device, TF version, `tensorflow-metal`
detection, and model load time when the caller went through
`load_model()`). It does not touch the filesystem unless the caller
explicitly asks it to save the result.

See [`neuropose.estimator`](api/estimator.md) for the API reference.

### benchmark

**Role:** multi-pass inference benchmarking layered on top of the
estimator. `run_benchmark()` calls `process_video` N times, discards
the first pass as warmup, and aggregates the remaining
`PerformanceMetrics` into a `BenchmarkAggregate` with distributional
statistics (mean / p50 / p95 / p99 per-frame latency, mean
throughput, max peak RSS).

The benchmark is exposed via the `neuropose benchmark <video>` CLI
subcommand. Its `--compare-cpu` flag spawns a subprocess with GPU
visibility hidden (via `tf.config.set_visible_devices([], "GPU")`
before any TF op) so a Metal-backed Apple Silicon run can be diffed
against a CPU baseline — both the throughput speedup and the maximum
element-wise `poses3d` divergence in millimetres are surfaced in the
output. This is the "is `tensorflow-metal` producing correct
numerics?" check that `RESEARCH.md`'s TensorFlow-version-compatibility
section leaves open for v0.1.

### interfacer

**Role:** job-lifecycle daemon. Watches `input_dir` for new job
subdirectories, dispatches each to an injected `Estimator`, and manages
the persistent `status.json` that tracks every job's lifecycle.

**Owns:** the `input_dir → output_dir → failed_dir` transitions, the
single-instance lock, signal handling, and crash recovery.

**Does NOT handle:** inference — that is the estimator's job, which is
injected via the constructor so tests can supply a fake.

Key guarantees:

- **Single instance.** An exclusive `fcntl.flock` on
  `data_dir/.neuropose.lock` blocks a second daemon from running against
  the same data directory. The lock is released automatically on
  process exit, even SIGKILL.
- **Crash recovery.** On startup, any status entries left in
  `processing` state are marked failed with an "interrupted" error and
  their inputs quarantined. The operator decides whether to retry by
  moving them back to `input_dir`.
- **Graceful shutdown.** SIGINT and SIGTERM request an orderly stop.
  The current job finishes before the loop exits.
- **Structured errors.** Every failed job records a short
  `"<ExceptionType>: <message>"` in its status entry so operators have
  a grep target without digging through logs.

See [`neuropose.interfacer`](api/interfacer.md) for the API reference.

### analyzer

**Role:** post-processing. Takes a `results.json` and produces analysis
output (DTW comparisons, joint-angle features, repetition segmentation,
classification). Each piece is a pure function of the predictions, so
the module is a set of testable utilities rather than a daemon.

Three submodules ship today:

- `analyzer.features` — `predictions_to_numpy`, normalization,
  padding, joint angles, summary statistics, and a thin
  `scipy.signal.find_peaks` wrapper.
- `analyzer.dtw` — three DTW entry points (`dtw_all`, `dtw_per_joint`,
  `dtw_relation`) over `fastdtw`, with a frozen `DTWResult` dataclass.
  See `RESEARCH.md` for the ongoing methodology discussion.
- `analyzer.segment` — **repetition segmentation**. Given a
  `VideoPredictions` of a trial in which the subject performs the
  same movement several times (e.g. lifting a cup repeatedly), the
  module detects the individual repetitions as
  `[start, peak, end)` windows via valley-to-valley peak detection
  on a clinically chosen 1D signal. The signal is one of four
  extractor variants (`joint_axis`, `joint_pair_distance`,
  `joint_speed`, `joint_angle`), and the produced `Segmentation`
  carries its own `SegmentationConfig` so the on-disk
  representation is self-describing. Segmentation is exposed both
  as a Python API and as the `neuropose segment` CLI subcommand,
  which runs post-hoc against an existing `results.json` — the
  daemon stays a pure inference daemon.

Classification wrappers on top of `sktime` are deliberately not
shipped yet; see `RESEARCH.md` for the plan.

## Data flow

```text
                 ┌──────────────────────────┐
                 │ $XDG_DATA_HOME/neuropose/│
                 └──────────────────────────┘
                              │
                ┌─────────────┼─────────────┐
                ▼             ▼             ▼
            jobs/in/      jobs/out/     jobs/failed/
                │             ▲             ▲
                │ discovered  │ on success  │ on failure
                │             │             │
                └─────────▶ process_job ────┘
                              │
                              ▼
                      status.json (atomic)
```

1. The operator drops a video (or several) into
   `data_dir/in/<job_name>/`.
2. The daemon detects the new job directory on its next poll.
3. For each video in the job, the estimator runs inference and returns
   a `VideoPredictions` object.
4. The daemon aggregates per-video predictions into a `JobResults`
   object and writes it to `data_dir/out/<job_name>/results.json`.
5. The status entry is updated to `completed`, with the path to
   `results.json` recorded.
6. On catastrophic failure (no videos, decode error, model crash), the
   job's input directory is moved to `data_dir/failed/<job_name>/` and
   the status entry is updated to `failed` with an error message.

All filesystem writes that affect application state (status file, job
results) go through atomic tmp-file-then-rename helpers in
[`neuropose.io`](api/io.md), so a crash mid-write cannot leave a
truncated file behind.

## Runtime directory layout

The daemon operates within a single base `data_dir`:

```text
$data_dir/
├── .neuropose.lock           # fcntl lock file; contains owner PID
├── in/
│   ├── job_001/              # operator-created
│   │   ├── video_01.mp4
│   │   └── video_02.mp4
│   └── job_002/
│       └── trial.mov
├── out/
│   ├── status.json           # persistent lifecycle state
│   ├── job_001/
│   │   └── results.json      # aggregated JobResults
│   └── job_002/
│       └── results.json
└── failed/
    └── job_003/              # quarantined inputs
        └── broken_video.mov
```

`data_dir` defaults to `$XDG_DATA_HOME/neuropose/jobs` and **is never
inside the repository.** This is deliberate: the previous prototype kept
job directories under `backend/neuropose/in/`, which is exactly how
subject-identifying data ended up on the same tree as `git add`. The
current design makes it mechanically difficult for subject data to
leak into source control.

Model weights are cached separately at `$XDG_DATA_HOME/neuropose/models/`.

## Design principles

A few choices run through every module and are worth knowing if you
plan to extend the package:

**Immutable schemas.** `FramePrediction` and `VideoMetadata` are
frozen pydantic models. The previous prototype had a bug where its
visualizer mutated `poses3d` in place via a numpy view, invisibly
corrupting the data if you visualized before saving. The frozen schema
makes that class of bug impossible.

**Validate at the boundary.** Every load/save helper in `neuropose.io`
validates on entry. Malformed files fail at load time with a pydantic
validation error, not three call sites later as an `AttributeError` on
a missing key.

**Library / daemon separation.** The estimator is pure library — give
it a video and a model, get back validated predictions. The daemon is
the wrapper that adds filesystem semantics. This makes the estimator
trivially testable (inject a fake model, inject any video) and lets
downstream users embed it in other pipelines without inheriting the
daemon's lifecycle.

**Dependency injection.** The `Interfacer` takes its `Estimator` as a
constructor argument. Tests inject fakes; production wires the real
thing. There is no singleton model state.

**No implicit config discovery.** Configuration is loaded explicitly
via `--config` or environment variables. The previous prototype's
`load_config('config.yaml')` was a relative path footgun — it worked
only when the daemon was launched from a specific directory. The new
`Settings` class refuses to guess.

**Atomic writes for all stateful files.** Status file, job results,
predictions — every write goes through a tmp-file-then-rename so a
crash mid-write cannot corrupt state.

**Fail fast, fail specifically.** Each module defines a small hierarchy
of typed exceptions (`EstimatorError`, `InterfacerError`, etc.).
Exception types carry semantic meaning; callers can distinguish
recoverable failures from programmer errors.
