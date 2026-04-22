# Changelog

All notable changes to NeuroPose are recorded in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

This section covers the ground-up rewrite of NeuroPose. The entries
below describe the difference between the previous internal prototype
and the state of the repository at the first tagged release, and will
be split into per-release sections once tagging begins.

### Added

#### Package structure and tooling

- `src/neuropose/` package layout with `py.typed` marker, MIT `LICENSE`,
  policy-enforcing `.gitignore`, pinned Python 3.11 (`.python-version`),
  and `pyproject.toml` with full project metadata, classifiers, and
  URL pointers. The runtime TensorFlow dependency is pinned to
  `tensorflow>=2.16,<2.19` — see *Changed* below for the rationale.
  `psutil>=5.9` is a runtime dependency used by the estimator's
  always-on `PerformanceMetrics` collection to sample peak RSS.
- `[project.optional-dependencies].analysis` extra for fastdtw, scipy,
  scikit-learn, and sktime — install via `pip install neuropose[analysis]`.
- `[project.optional-dependencies].metal` extra pulling
  `tensorflow-metal>=1.2,<2` under `sys_platform == 'darwin' and
  platform_machine == 'arm64'` environment markers. Opt-in only via
  `pip install 'neuropose[metal]'` or `uv sync --extra metal`; silently
  no-op on every non-Apple-Silicon platform. The Metal path is **not**
  exercised in CI and is documented as experimental in
  `docs/getting-started.md` — users enabling it are expected to
  spot-check numerics against the CPU path before trusting results
  downstream.
- `[dependency-groups].dev` (PEP 735) with the full dev + docs + analyzer
  toolchain: pytest, pytest-cov, ruff, pyright, pre-commit,
  mkdocs-material, mkdocstrings, fastdtw, and scipy. `uv sync --group dev`
  gives contributors everything needed to run the whole suite.
- `AUTHORS.md`, `CITATION.cff` (with a MeTRAbs upstream `references:`
  entry), and a MIT-licensed `LICENSE` with an explicit MeTRAbs
  attribution paragraph.
- Pre-commit configuration (`.pre-commit-config.yaml`) running ruff,
  ruff-format, gitleaks (secret scanning), a 500 KB-limit
  large-files hook, end-of-file fixers, trailing-whitespace fixers,
  and YAML/TOML/JSON validators. Pyright is deliberately **not** in
  pre-commit — it runs in CI only, so pre-commit stays fast.
- Ruff configuration in `pyproject.toml` with a deliberately broad
  rule selection (pycodestyle, pyflakes, isort, bugbear, pyupgrade,
  simplify, ruff-specific, pep8-naming, comprehensions, pathlib,
  pytest-style, tidy-imports, numpy-specific, pydocstyle with numpy
  convention). Per-file ignores for tests and private modules.
- Pyright configuration in `standard` mode (not `strict` — TF/OpenCV
  stubs would otherwise drown the signal). Unknown-type reports are
  explicitly silenced until the TensorFlow version pin is settled.
- Pytest configuration with strict markers, an opt-in `slow` marker,
  and a `--runslow` CLI flag implemented in
  `tests/conftest.py::pytest_collection_modifyitems` so integration
  tests stay out of the default run.

#### CI / infrastructure

- GitHub Actions workflow `.github/workflows/ci.yml` running three
  parallel jobs — **lint** (ruff), **typecheck** (pyright), and
  **test** (pytest) — on every push and PR to `main`. Uses `uv` with
  a pinned version (`0.9.16`) and cache-enabled setup for fast reruns.
  Concurrency control cancels superseded runs on the same branch.
- GitHub Actions workflow `.github/workflows/docs.yml` that builds the
  mkdocs-material site on every relevant push and uploads the rendered
  site as a 14-day workflow artifact. GitHub Pages deployment is
  intentionally not wired up yet; the workflow header comment
  describes what to add when the repo flips public.

#### Runtime modules

- **`neuropose.config`** — `Settings` class built on
  `pydantic-settings`. Field-level validation for `device`,
  `poll_interval_seconds`, and `default_fov_degrees`; explicit
  `from_yaml()` classmethod (no implicit config-file discovery); XDG
  defaults for `data_dir` and `model_cache_dir` (`~/.local/share/neuropose/…`)
  so runtime data never lives inside the repository; `ensure_dirs()`
  as an explicit method so construction remains filesystem-side-effect-free.
- **`neuropose.io`** — validated prediction schemas:
  `FramePrediction` (frozen), `VideoMetadata` (frame count, fps,
  width, height), `VideoPredictions` (metadata envelope + frames
  mapping + optional `segmentations` field), `JobResults`,
  `JobStatus` enum, `JobStatusEntry` (with a structured `error`
  field plus optional live-progress fields — `current_video`,
  `frames_processed`, `frames_total`, `videos_completed`,
  `videos_total`, `percent_complete`, `last_update` — populated by
  the interfacer during inference and consumed by
  `neuropose.monitor`), and `StatusFile`. Legacy status files
  written before the progress fields existed still load cleanly
  because every new field is optional with a `None` default. Performance schema: frozen
  `PerformanceMetrics` carrying per-call timings
  (`model_load_seconds`, `total_seconds`, `per_frame_latencies_ms`),
  `peak_rss_mb`, `active_device`, `tensorflow_metal_active`, and
  `tensorflow_version`; `BenchmarkResult` pairing a discarded
  `warmup_pass` with `measured_passes` and a `BenchmarkAggregate`
  (mean / p50 / p95 / p99 per-frame latency, mean throughput, max
  peak RSS); optional `CpuComparisonResult` nested inside
  `BenchmarkResult` for `--compare-cpu` runs, carrying both
  device aggregates, the throughput speedup, and the
  maximum-element-wise `poses3d` divergence in millimetres.
  Segmentation schema: frozen `Segment` windows (`start`, `end`,
  `peak`), `SegmentationConfig` (with a `method` version literal,
  e.g. `valley_to_valley_v1`), a discriminated `ExtractorSpec`
  union over `JointAxisExtractor`, `JointPairDistanceExtractor`,
  `JointSpeedExtractor`, and `JointAngleExtractor`, and
  `Segmentation` pairing a config with its segments so on-disk
  results are self-describing. Load and save helpers with an atomic
  tmp-file-then-rename pattern for every state file.
  `load_benchmark_result` / `save_benchmark_result` follow the same
  atomic pattern. `load_status` is deliberately crash-resilient:
  missing, corrupt, or non-mapping JSON returns an empty
  `StatusFile` rather than raising. Legacy predictions files
  without the `segmentations` field deserialize cleanly to an
  empty mapping.
- **`neuropose.estimator`** — `Estimator` class that streams frames
  directly from OpenCV into the model, with no intermediate write-to-
  disk-then-read-back-as-PNG round trip. Returns a typed
  `ProcessVideoResult` containing a validated `VideoPredictions`
  object and an always-populated `PerformanceMetrics` bundle (per-
  frame latency in ms, total wall clock, peak RSS via `psutil`,
  active TF device string, `tensorflow-metal` detection, TF
  version, and model load time when the caller went through
  `load_model()`). Does not touch the filesystem. Constructor
  accepts an injected model for testability; `load_model()`
  delegates to `neuropose._model.load_metrabs_model()`. Typed
  exception hierarchy: `EstimatorError`, `ModelNotLoadedError`,
  `VideoDecodeError`. Optional per-frame `progress` callback for
  long videos. Frame identifier convention is `frame_000000`
  (six-digit zero-pad, no extension — no file is implied).
- **`neuropose.visualize`** — `visualize_predictions()` for per-frame
  2D + 3D overlay rendering. `matplotlib.use("Agg")` is called inside
  the function rather than at module import, so `import neuropose.visualize`
  has no global side effect. Explicit deep-copy of `poses3d` before
  axis rotation to prevent the aliasing bug from the previous
  prototype. Supports `frame_indices` for rendering a subset of
  frames.
- **`neuropose.interfacer`** — `Interfacer` job-lifecycle daemon with
  dependency-injected `Settings` and `Estimator`. Single-instance
  enforcement via `fcntl.flock` on `data_dir/.neuropose.lock`.
  Crash-recovery `recover_stuck_jobs()` that marks any status entries
  left in `processing` state as failed with an "interrupted"
  message and quarantines their inputs. Graceful shutdown on SIGINT/
  SIGTERM with an interruptible sleep. Structured error fields on
  every failed job. `run_once()` factored out of the main loop so
  tests can drive single iterations without threading. Quarantine
  collision handling (`job_a.1`, `job_a.2`, …) and empty-directory
  silent-skip heuristic (mid-copy directories are not marked failed).
- **`neuropose._model`** — MeTRAbs model loader. Downloads the pinned
  tarball from the upstream RWTH Aachen URL
  (`metrabs_eff2l_y4_384px_800k_28ds.tar.gz`), verifies its SHA-256
  checksum, atomically extracts to a staging directory and renames
  into place, and loads via `tf.saved_model.load`. Streams the
  download and hash computation in 1 MB chunks so memory is flat.
  One automatic retry on SHA-256 mismatch (in case the previous
  download was truncated). Post-load interface check for
  `detect_poses`, `per_skeleton_joint_names`, and
  `per_skeleton_joint_edges`.
- **`neuropose.monitor`** — localhost HTTP status dashboard. A small
  `http.server`-based HTTP server (pure stdlib, zero new runtime
  dependencies) that serves a plain HTML page at `GET /` with an
  auto-refresh meta tag, one row per tracked job, a
  `<progress>` bar, and a stale-entry warning badge for
  `processing` jobs whose `last_update` has not ticked in 60 s.
  `GET /status.json` returns the raw validated `StatusFile` as JSON
  for `curl`/scripted pipelines; `?job=<name>` filters to a single
  entry. `GET /health` is a simple liveness probe. Binds to
  `127.0.0.1:8765` by default — loopback-only, with an explicit
  `--host` override required to expose externally. Every request
  re-reads `status.json`, so the monitor has no in-memory cache, no
  sync protocol with the daemon, and stays useful even if the
  daemon is down (last-known state surfaced with the stale badge).
- **Progress checkpointing in the interfacer.** `Interfacer` now
  updates the currently-running job's `JobStatusEntry` every
  `settings.status_checkpoint_every_frames` frames (default 30, a
  new `Settings` field) during inference via the estimator's
  `progress` callback. Each checkpoint rewrites `status.json`
  atomically through the existing `save_status` helper; writes are
  best-effort and I/O failures are logged without interrupting
  inference. `_run_job_inner` seeds a "videos_total=N" checkpoint
  before calling the estimator so the monitor shows the job's
  scope from the first poll. Checkpoint cadence is knob-exposed for
  operators who want to tune the smoothness-vs-write-rate trade-off.
- **`neuropose.ingest`** — zip-archive intake utility. `ingest_zip()`
  extracts a zip of videos into one job directory per video under
  `$data_dir/in/`, with validation-before-write (path-traversal and
  absolute-path members rejected, oversize archives rejected at the
  20 GB-uncompressed cap), zip-internal and external collision
  detection reported in one shot, non-video members silently
  skipped (`.DS_Store`, `README.md`, etc.), and per-job atomic
  placement via a staging directory + `os.rename`. Nested paths are
  flattened into job names by joining components with underscores
  and sanitising unsafe characters — `patient_001/trial_01.mp4`
  becomes job `patient_001_trial_01`, preserving disambiguation
  against a sibling `patient_002/trial_01.mp4`. Typed exception
  hierarchy: `IngestError`, `ArchiveInvalidError`,
  `ArchiveEmptyError`, `ArchiveTooLargeError`, `JobCollisionError`
  (with a `.collisions` list of offending names). The running
  daemon needs no changes — ingested job dirs are picked up on the
  next poll.
- **`neuropose.migrations`** — schema-migration infrastructure for
  the three top-level serialised payloads (`VideoPredictions`,
  `JobResults`, `BenchmarkResult`). Every payload carries a
  `schema_version` field defaulting to `CURRENT_VERSION`; on load,
  the raw JSON dict is passed through `migrate_video_predictions` /
  `migrate_job_results` / `migrate_benchmark_result` *before*
  pydantic validation so files written by older NeuroPose versions
  upgrade transparently. One shared `CURRENT_VERSION` counter;
  per-schema migration registries populated via
  `register_video_predictions_migration(from_version)` and
  `register_benchmark_result_migration(from_version)` decorators.
  `JobResults` is a `RootModel` with no envelope of its own, so its
  migration runs per-entry across the root mapping. The driver raises
  `FutureSchemaError` for payloads newer than the current build
  (clear upgrade-NeuroPose message), `MigrationNotFoundError` for
  missing chain links (indicates a `CURRENT_VERSION` bump that forgot
  its migration), and logs at INFO on each version advance. Currently
  at `CURRENT_VERSION = 2`, with registered v1 → v2 migrations for
  `VideoPredictions` and `BenchmarkResult` that add the optional
  `provenance` field.
- **`neuropose.analyzer.features.procrustes_align`** — Kabsch
  rigid-alignment helper for pose sequences, plus a
  `ProcrustesMode` literal (`"per_frame"` | `"per_sequence"`) and a
  frozen `AlignmentDiagnostics` dataclass (`rotation_deg`,
  `rotation_deg_max`, `translation`, `translation_max`, `scale`,
  plus the mode that produced them). Per-sequence mode fits one
  rigid transform across the whole trial; per-frame fits an
  independent transform per frame. Optional `scale=True` fits a
  uniform scale factor for cross-subject comparisons. Wired into
  every DTW entry point in `neuropose.analyzer.dtw` via a new
  keyword-only `align: AlignMode = "none"` parameter — `"none"`
  preserves the 0.1 raw-coordinate behaviour, while
  `"procrustes_per_frame"` and `"procrustes_per_sequence"` route
  inputs through `procrustes_align` before DTW runs so the returned
  distance is rotation- and translation-invariant. Paper C's
  pipeline is expected to set `align="procrustes_per_sequence"`;
  see `TECHNICAL.md` Phase 0.
- **`neuropose.analyzer.dtw.Representation`** and
  **`neuropose.analyzer.dtw.NanPolicy`** — two new Literal types
  exposing orthogonal DTW preprocessing knobs on every entry point.
  `representation` (on `dtw_all` and `dtw_per_joint`) switches the
  per-frame feature vector between `"coords"` (the 0.1 default) and
  `"angles"`, which runs `extract_joint_angles` on the supplied
  `angle_triplets` first — yielding distances that are translation-,
  rotation-, and scale-invariant by construction, and directly
  interpretable in clinical terms. `nan_policy` (on all three entry
  points) selects `"propagate"` (surface fastdtw's ValueError on
  NaN — the default), `"interpolate"` (linear fill per feature
  column), or `"drop"` (remove NaN frames before DTW); the
  policy is applied consistently whether NaN originated from the
  angles pipeline or from corrupted upstream coordinates.
  `dtw_relation` stays a standalone convenience entry point for
  two-joint displacement DTW; users who prefer a unified API can
  express the same computation via `dtw_all` with an appropriate
  pair of angle triplets or run `dtw_relation` directly.
- **`neuropose.analyzer.pipeline`** (schemas) — declarative
  analysis-pipeline configuration and output artifact, parseable from
  YAML or JSON via pydantic. `AnalysisConfig` captures a full
  experiment: inputs (primary + optional reference predictions
  files), preprocessing (person index, with room to grow),
  optional segmentation (`gait_cycles` / `gait_cycles_bilateral` /
  `extractor` discriminated union), and a required analysis stage
  (`dtw` / `stats` / `none` discriminated union). `AnalysisReport`
  is the runtime output: carries the originating config, a
  `Provenance` envelope with `analysis_config` populated, per-input
  summaries, produced segmentations, and an analysis-result payload
  that mirrors the stage choice (`DtwResults`, `StatsResults`, or
  `NoResults`). Cross-field invariants — `method="dtw_relation"`
  requires `joint_i`/`joint_j`, `representation="angles"` requires
  non-empty `angle_triplets`, `analysis.kind="dtw"` requires
  `inputs.reference`, `analysis.kind="stats"` refuses a reference —
  are enforced at parse time via `model_validator` so typos fail in
  milliseconds instead of after a multi-minute predictions load.
  `AnalysisReport` carries a `schema_version` field defaulting to
  `CURRENT_VERSION = 2`, with a new
  `register_analysis_report_migration` decorator and
  `migrate_analysis_report` driver in `neuropose.migrations` ready
  for future schema changes. Pipeline execution lands in a
  follow-up commit.
- **`neuropose.analyzer.segment.segment_gait_cycles`** and
  **`segment_gait_cycles_bilateral`** — clinical convenience
  wrappers over `segment_predictions` that pre-fill a `joint_axis`
  extractor with gait-appropriate defaults (`joint="rhee"`,
  `axis="y"`, `min_cycle_seconds=0.4`). The single-side entry point
  accepts any berkeley_mhad_43 joint name and any spatial axis as a
  string literal `"x" | "y" | "z"`, plus an `invert` flag for
  recordings whose vertical axis runs opposite to MeTRAbs's
  Y-down world-coordinate convention. The bilateral wrapper runs
  the detection on both `lhee` and `rhee` and returns the two
  results under `"left_heel_strikes"` / `"right_heel_strikes"`
  keys — shape-compatible with `VideoPredictions.segmentations` so
  the dict can be merged in directly. Degrades gracefully on
  pathological gaits (shuffling, walker-assisted) by returning an
  empty segments list rather than raising. Closes the gait-cycle
  segmentation item in `TECHNICAL.md` Phase 0.
- **`neuropose.io.Provenance`** — reproducibility envelope for every
  inference run. Populated automatically by `Estimator.process_video`
  when the model was loaded via `load_model` (the production path)
  and attached to the output `VideoPredictions`; propagates from
  there into `JobResults` (per-video) and `BenchmarkResult` (via the
  benchmark loop). Captures the MeTRAbs artifact SHA-256 and
  filename, `tensorflow` / `tensorflow-metal` / `numpy` /
  `neuropose` / Python versions, and reserved slots for a `seed`,
  `deterministic` flag (Track 2), and `analysis_config` (Phase 0
  YAML pipeline). `None` on the injected-model test path where
  NeuroPose has no way to fingerprint the supplied artifact. Frozen
  pydantic model with `extra="forbid"` and
  `protected_namespaces=()` so the `model_*` field names do not
  collide with pydantic v2's internal namespace. `_model.load_metrabs_model`
  now returns a `LoadedModel` dataclass bundling the TF handle with
  the pinned SHA and filename so the estimator can build the
  `Provenance` without re-hashing the tarball.
- **`neuropose.reset`** — pipeline-wide reset utility for the
  benchmark / iteration loop. `find_neuropose_processes()` scans the
  OS process table (via `psutil`) for running `neuropose watch` and
  `neuropose serve` instances and classifies each as `daemon` or
  `monitor`. `terminate_processes()` SIGINTs them, polls for graceful
  exit up to a configurable grace period, and optionally escalates
  to SIGKILL with `force_kill=True`. `wipe_state()` removes the
  contents of `$data_dir/in/`, `$data_dir/out/` (including
  `status.json`), `$data_dir/failed/` (unless `keep_failed=True`),
  the `.neuropose.lock` file, and any leftover `.ingest_<uuid>/`
  staging dirs from interrupted ingests; container directories
  themselves are preserved so the daemon does not need to recreate
  them on next startup. `reset_pipeline()` composes the three with
  one safety guard: if any process survives termination, the wipe
  phase is skipped and the returned `ResetReport` flags
  `wipe_skipped_due_to_survivors`, because removing `$data_dir`
  out from under an active daemon would corrupt its in-flight
  writes. Surfaced as `neuropose reset` in the CLI with
  `--yes/-y`, `--keep-failed`, `--force-kill`, `--grace-seconds`,
  and `--dry-run/-n` flags; the command always prints a preview
  before prompting (skipped under `--yes`) and returns
  `EXIT_USAGE=2` when survivors block the wipe.
- **`neuropose.benchmark`** — multi-pass inference benchmarking for
  a single video. `run_benchmark()` runs `process_video` N times
  (default 5), always discards the first pass as warmup (graph
  compilation, file-system cache warmup), and aggregates the
  remaining `PerformanceMetrics` into a `BenchmarkAggregate` with
  mean / p50 / p95 / p99 per-frame latency, mean throughput, and
  max peak RSS. `capture_reference=True` additionally preserves the
  last measured pass's `VideoPredictions` in memory so the
  `--compare-cpu` CLI flow can diff the `poses3d` arrays between a
  GPU and CPU run. `compute_poses3d_divergence()` computes the
  maximum element-wise absolute difference (in millimetres) between
  two prediction sets, skipping frames with mismatched detection
  counts and surfacing the `frame_count_compared` so callers can
  tell if the number is trustworthy. `format_benchmark_report()`
  renders a human-readable summary for CLI stdout.
- **`neuropose.analyzer`** — post-processing subpackage with lazy
  imports for the heavy dependencies:
  - `analyzer.dtw` — three DTW entry points (`dtw_all`,
    `dtw_per_joint`, `dtw_relation`) over fastdtw, with a frozen
    `DTWResult` dataclass and three orthogonal preprocessing knobs
    (`align`, `representation`, `nan_policy`). See `RESEARCH.md`
    for the ongoing
    methodology investigation.
  - `analyzer.features` — `predictions_to_numpy`,
    `normalize_pose_sequence` (uniform and axis-wise),
    `pad_sequences` (edge-padding), `procrustes_align` (Kabsch
    rigid alignment, per-frame or per-sequence, optional uniform
    scaling), `extract_joint_angles` (NaN on degenerate vectors),
    `extract_feature_statistics` (`FeatureStatistics` frozen
    dataclass), and a `find_peaks` thin wrapper around
    `scipy.signal.find_peaks`.
  - `analyzer.segment` — repetition segmentation for trials in
    which a subject performs the same movement several times. A
    three-layer API: `segment_by_peaks` (pure 1D
    valley-to-valley peak detection on a generic signal),
    `segment_predictions` (top-level entry point taking a
    `VideoPredictions` plus an `ExtractorSpec`, converting
    time-based parameters to frame counts via `metadata.fps`), and
    `slice_predictions` (split a `VideoPredictions` into one per
    detected repetition with re-keyed frame names and a rewritten
    `frame_count`). Gait-specific convenience wrappers
    `segment_gait_cycles` (single heel) and
    `segment_gait_cycles_bilateral` (both heels, returning a dict
    keyed by `"left_heel_strikes"` / `"right_heel_strikes"`) sit
    above `segment_predictions` with clinical defaults. Ships four extractor factories —
    `joint_axis`, `joint_pair_distance`, `joint_speed`, and
    `joint_angle` — plus a `JOINT_NAMES` constant for the
    berkeley_mhad_43 skeleton with a `joint_index(name)` lookup,
    so post-processing callers can resolve `"rwri"` → integer
    without loading the MeTRAbs SavedModel. A matching integration
    test (`tests/integration/test_joint_names_drift.py`, marked
    `slow`) loads the real model and asserts the constant still
    matches, so any upstream skeleton drift fails CI.
- **`neuropose.cli`** — Typer-based command-line interface with
  eight subcommands: `watch` (run the daemon), `process <video>`
  (run the estimator on a single video), `ingest <archive>` (unzip
  a video archive into per-video job directories under
  `$data_dir/in/` with validation-before-write and atomic
  placement; `--force` overwrites collisions, otherwise the whole
  operation refuses if any target name already exists),
  `serve` (start the localhost HTTP monitor at `127.0.0.1:8765`
  by default — `--host` and `--port` are the two overrides;
  KeyboardInterrupt exits with the standard shell-interruption
  code and an `OSError` at bind time is translated to a clean
  usage error with the bind target in the message),
  `reset` (stop the daemon and monitor, then wipe pipeline state
  for a clean restart — wraps `neuropose.reset` with a confirmation
  prompt, `--dry-run` preview, `--keep-failed` to preserve the
  forensic quarantine, `--force-kill` to escalate to SIGKILL after
  the SIGINT grace period, and `--grace-seconds` to tune the wait;
  refuses to wipe state while any process survives termination so
  active writes cannot be corrupted),
  `segment <results>` (post-hoc repetition segmentation — loads a
  JobResults or a single VideoPredictions, runs
  `neuropose.analyzer.segment.segment_predictions` with the chosen
  extractor and thresholds, and atomically writes the file back
  with the new segmentation attached under `--name`),
  `benchmark <video>` (multi-pass inference benchmark — runs
  `--repeats N` passes with a discarded first pass and
  `--warmup-frames M` excluded from the head of each measured
  pass, reports aggregates to stdout, and optionally writes a
  structured `BenchmarkResult` to `--output`. Supports
  `--compare-cpu` which spawns a `--force-cpu` subprocess, diffs
  the resulting `poses3d` arrays, and reports throughput speedup
  and max divergence in mm — the missing Apple Silicon numerical
  verification answer from `RESEARCH.md`), and
  `analyze <results>` (stub). The `segment` subcommand accepts
  joint specifiers as either berkeley_mhad_43 names (`lwri`,
  `rwri`, …) or integer indices, and refuses to overwrite an
  existing segmentation of the same name without `--force`.
  Global options `--config/-c`, `--verbose/-v`, `--quiet/-q`,
  `--version`. Structured error handling turns expected exceptions
  (`FileNotFoundError` on config, `ValidationError`,
  `AlreadyRunningError`, `NotImplementedError`,
  `KeyboardInterrupt`) into clear stderr messages and distinct
  exit codes (`EXIT_OK=0`, `EXIT_USAGE=2`, `EXIT_PENDING=3`,
  `EXIT_INTERRUPTED=130`). The CLI entry point is wired in
  `[project.scripts]` as `neuropose = "neuropose.cli:run"`.

#### Documentation

- **mkdocs-material documentation site** under `docs/` with the full
  theme configuration (light/dark toggle, tabs navigation, search),
  `mkdocstrings` Python handler set to numpy docstring style with
  source links, and a `pymdownx` extension set for admonitions,
  tabbed content, collapsible details, and syntax-highlighted code
  blocks. Nav: Home → Getting Started → Architecture → API Reference
  (auto-generated from module docstrings) → Development → Deployment.
- Prose documentation pages: `docs/index.md` (public landing page),
  `docs/getting-started.md` (install, CLI, output schema, Python API,
  visualization, troubleshooting), `docs/architecture.md` (three-stage
  pipeline, data flow, runtime directory layout, design principles),
  `docs/development.md` (contributor setup, tests, lint/type,
  commit hygiene, release process stub), and `docs/deployment.md`
  (systemd user unit, Docker pointer, GPU notes, backup guidance).
- API reference stubs `docs/api/{config,estimator,interfacer,io,visualize}.md`
  — each is a two-line file containing a `:::` mkdocstrings directive,
  so the API documentation is generated from the source docstrings
  at build time and cannot drift out of sync.
- `RESEARCH.md` at the repo root: a living R&D log for DTW
  methodology alternatives and MeTRAbs self-hosting / fine-tuning
  plans. Not user-facing documentation; not linked from the mkdocs
  nav.

#### Tests

- `tests/unit/` covering configuration (defaults, validation, YAML
  loading, env overrides, `ensure_dirs`), IO schema and helpers
  (roundtrip, atomic save, frozen-model guarantees, corruption
  tolerance), the estimator (construction, model-guard, process path
  with fake MeTRAbs model, error paths), the visualize module
  (smoke tests + an anti-regression check for the audit §6 aliasing
  bug), the interfacer (construction, discovery, process-job happy
  and failure paths, stuck-job recovery, lock, run_once,
  interruptible sleep), the CLI (top-level options, config handling,
  each subcommand's error path), the analyzer DTW helpers, and the
  analyzer features helpers.
- `tests/conftest.py` with an autouse `_isolate_environment` fixture
  that redirects `$HOME` and `$XDG_DATA_HOME` at a per-test temp
  directory so no test can accidentally write to the developer's real
  machine, and clears any `NEUROPOSE_*` env vars. Adds a
  `synthetic_video` fixture (cv2-generated 5-frame MJPG AVI sized
  for most unit tests) and a `fake_metrabs_model` fixture.
- `tests/integration/test_estimator_smoke.py` — end-to-end model
  loader + estimator smoke test against the real MeTRAbs tarball,
  marked `@pytest.mark.slow`, skipped by default, opt-in via
  `--runslow`. Uses a session-scoped model cache so the download
  happens at most once per run.

#### Operations

- `Dockerfile` — CPU image based on `python:3.11-slim-bookworm`.
  Installs the package with the `analysis` extra, runs as non-root
  user `neuropose` (UID 1000), exposes `/data` as a volume, sets
  `NEUROPOSE_DATA_DIR` and `NEUROPOSE_MODEL_CACHE_DIR` to point at
  the mounted volume, and uses `ENTRYPOINT ["neuropose"]` with
  `CMD ["watch"]` so the default is the daemon and overrides are
  ergonomic.
- `.dockerignore` that aggressively excludes developer tooling,
  caches, tests, documentation sources, research notes, and
  ancillary scripts from the build context.
- `scripts/download_model.py` — standalone pre-warm script that
  invokes `load_metrabs_model()` with an optional `--cache-dir`
  override. Useful for seeding a deployment's cache before cutting
  off network access.

### Changed

- **Relicensed from AGPL-3.0 (used in the prior internal prototype)
  to MIT.** The prior license was copied from precedent rather than
  chosen deliberately; the MIT relicense better matches both the
  project's "research software others can build on" intent and the
  upstream MeTRAbs license.
- Reorganised from the prior `backend/` + runtime-data layout into
  a `src/neuropose/` Python package. Runtime data now lives outside
  the repository by default (under `$XDG_DATA_HOME/neuropose/`) so
  subject-identifying inputs cannot accidentally end up in a
  `git add`.
- Frame identifier convention changed from `frame_0000.png` (old,
  misleading — no PNG file exists) to `frame_000000` (six-digit
  zero-pad, no extension, pure identifier).
- Estimator API: `process_video()` now returns a typed
  `ProcessVideoResult` containing a validated `VideoPredictions`
  object, instead of a stringly-typed dict with `results_path` and
  `frame_count`. The estimator no longer owns filesystem
  destinations — the caller decides where to save.
- `VideoPredictions` schema now carries a `VideoMetadata` envelope
  (frame count, fps, width, height) alongside the per-frame
  predictions. Downstream analysis can convert frame indices to
  real time without needing access to the original video.
- Interfacer uses `datetime.now(UTC)` instead of the deprecated
  `datetime.utcnow()`, addresses the "no-videos"-vs-exception-path
  inconsistency (both now quarantine), and persists a structured
  `error` string on every failure for grep-friendly diagnostics.
- **TensorFlow pin set to `tensorflow>=2.16,<2.19`.** The 2.16
  floor is the first release with native `darwin/arm64` wheels under
  the `tensorflow` package name on PyPI, so a single dependency line
  works across Linux x86_64, Linux arm64, and Apple Silicon macOS
  without platform markers or a separate `tensorflow-macos` package.
  The `<2.19` ceiling is a `tensorflow-metal` compatibility constraint:
  the latest Metal plugin (1.2.0, January 2025) advertises "TF 2.18+"
  but in practice fails on 2.19 and 2.20 with symbol-not-found errors
  and graph-execution `InvalidArgumentError`s
  ([tensorflow/tensorflow#84167](https://github.com/tensorflow/tensorflow/issues/84167)).
  Cap is global rather than darwin-only so dependency resolution stays
  identical across platforms. The MeTRAbs SavedModel itself
  (`metrabs_eff2l_y4_384px_800k_28ds`, serialized with TF 2.10) was
  separately verified to load and run `detect_poses` end-to-end on
  TF 2.21 + Keras 3 with no errors and zero custom ops, so the cap is
  purely an external-package constraint and can lift once Apple ships
  a Metal plugin that tracks mainline TensorFlow again. Full probe
  data and op inventory in `RESEARCH.md`.
- Operating-system classifiers in `pyproject.toml` extended from
  Linux-only to `POSIX` + `POSIX :: Linux` + `MacOS`, reflecting the
  Apple Silicon support that the TF 2.16 floor makes real.

### Removed

- The previous `backend/analyzer.py` and `backend/validator.py`
  stubs, which were non-functional and had never been run
  successfully. `analyzer.py` is reintroduced as a pure-function
  subpackage (`neuropose.analyzer`) rewritten from the prior
  code's design intent. `validator.py` is reintroduced as a real
  pytest suite (`tests/unit/` and `tests/integration/`).
- The previous `reconstruct_from_frames` helper on the `Estimator`
  — dead code, broken (dereferenced `self.OUTPUT_PATH`, which did
  not exist), hardcoded 10 fps, never called. ffmpeg is a better
  tool for this and can be invoked directly.
- The previous `__main__` placeholder (`print("in main"); sys.exit()`)
  on `estimator.py`. The real CLI now lives in `neuropose.cli`.
- Every file under `docs/` in the previous prototype. All of the
  pydoc-generated HTML, Org-mode sources, and handwritten markdown
  described an older version of the API with methods
  (`bind_and_block`, `construct_paths`, `toggle_visualization`,
  `propagate_fatal_error`, etc.) that no longer exist. The docs are
  now auto-generated from source docstrings via mkdocstrings so
  drift is mechanically impossible.
- The previous Dockerfile, which referenced a non-existent
  `backend/requirements.txt`, attempted to `COPY ./model /app/model`
  (no such directory), and set `CMD ["uvicorn", "main:app"]` for a
  FastAPI app that never existed.
- The previous `install/install.sh`, `install/#install.sh#` (an
  Emacs autosave file), `install/install.sh~` (an Emacs backup file),
  and `install/environment.yml`. The conda + `git+https` install
  story is replaced by `uv` + a single `pyproject.toml`.
- The previous `bit.ly/metrabs_1` URL shortener for the model
  download, replaced by a pinned canonical URL on the upstream
  RWTH Aachen "omnomnom" host, with SHA-256 verification on
  download. See `RESEARCH.md` for the plan to mirror to
  self-hosted storage.

### Security

- Large-files pre-commit hook (`check-added-large-files` with a
  500 KB limit) blocks accidental commits of subject data or model
  weights.
- Gitleaks pre-commit hook scans every staged change for secret
  material.
- Dockerfile runs as a non-root user (UID 1000, `neuropose`) by
  default.
- Tarfile extraction uses the `filter="data"` option to block path
  traversal and other tar-bomb attacks during MeTRAbs model
  extraction.
- SHA-256 pinning of the MeTRAbs model artifact. A change to the
  upstream tarball contents fails the checksum verification and
  requires a human-reviewed diff before the new artifact is
  trusted.

### Known limitations

- Apple Silicon support is established by-construction (TF 2.16+
  publishes native `darwin/arm64` wheels and the MeTRAbs SavedModel
  uses only stock ops verified portable on TF 2.21) but has not yet
  been exercised on real Apple Silicon hardware. A `macos-14` CI
  matrix entry covering the unit tests is the cheapest way to catch
  any regression and is planned as a follow-up.
- Classification wrappers on top of sktime are deliberately **not**
  included in `neuropose.analyzer` for this release. See `RESEARCH.md`
  for the reasoning and the plan.
- GPU support in Docker is not yet shipped (`Dockerfile.gpu` is
  planned). The existing `Dockerfile` runs CPU-only.
- `neuropose analyze` is a CLI stub that exits with a pending
  message. The analyzer subpackage is usable from Python directly;
  the CLI wrapper will follow once the analysis pipeline has a
  concrete shape worth wrapping.
- The data-handling policy referenced from `docs/deployment.md` and
  `docs/index.md` (`docs/data-policy.md`) is being authored
  separately and is not part of this changelog entry.

[Unreleased]: https://git.levineuwirth.org/neuwirth/neuropose/compare/initial...HEAD
