# NeuroPose Technical Ideation Notes

A living engineering roadmap, parallel to `RESEARCH.md`. Where
`RESEARCH.md` captures open methodological questions (DTW, skeleton
choice, hosting the model), this document captures open *engineering*
questions — release readiness, operability, scaling — and the paths
they could take.

This is **not** user-facing documentation. Items here are *candidates*
for future work, and inclusion does not imply commitment.

## How to use this document

- Add a section when you start thinking about a new area of technical
  investment.
- Each section should end with a **Scope**, **Sketch**, or **Open
  questions** block so it's obvious to a future you (or a new
  contributor) what the concrete next move would be.
- When an item in here is decided and implemented, move it to the
  relevant place in `docs/` or in the code itself, and leave a short
  pointer behind (*See `docs/deployment.md` for the resolved design.*).
- The audience is anyone maintaining the codebase — Levi, David,
  Praneeth, Dr. Shu, and whoever comes after us. Assume competence in
  Python and systems work; don't assume familiarity with our specific
  tooling choices.

## Three phases, then a contingent track

There are four distinct technical objectives, ordered by timeline and
by what each enables next. The sequencing is deliberate: each phase
unblocks the next, and doing them in any other order either publishes
Paper C on top of a pipeline its own design notes disavow, or delays
the open-source release past the window where the accompanying paper
is still salient.

1. **Phase 0 — C-enabling pipeline work.** A targeted subset of
   engineering work that has to land *before* Paper C can start. The
   DTW defaults shipped in 0.1 are explicitly a "mechanical port, not
   a methodological choice" (see `RESEARCH.md` §1); running the
   clinical validation study on them would mean publishing results
   from a pipeline the accompanying design notes explicitly criticize.
   Phase 0 fixes the analyzer's methodological foundations (Procrustes
   preprocessing, cycle segmentation, joint-angle DTW representation),
   locks in the reproducibility surface (`Provenance` subobject,
   YAML-configurable analysis pipeline), and sets up schema migration
   so data generated during Phase 1 survives the long write-up.
   **Near-term, well-scoped, weeks of work.**

2. **Phase 1 — Paper C: clinical validation study.** The planned
   clinical-methods paper: cycle-aware joint-angle DTW for clinical
   gait similarity, validated against MoCap ground truth and/or
   clinician ratings. Gated on MoCap data access via Dr. Shu. This is
   research work, not engineering work — this document describes the
   engineering scaffolding *around* it, not the paper itself. Phase 2
   work can happen in the background during this phase as ideal filler
   for research-burnout cycles. **Months; timeline driven by data
   access and experimental design.**

3. **Phase 2 — Coordinated open-source release + Paper A.** The
   engineering-paper companion (A) describing the tech stack, plus
   the tagged 0.1 release: PyPI publication, docs deployment, Docker
   images, CI matrix, supervision artifacts, doctor preflight, all
   the operational items that make the tool credible to external
   users. Timed to arrive *with or slightly before* Paper C's
   submission, producing a paper-plus-tool bundle that reviewers can
   actually run. **Weeks of work, timing driven by Paper C's
   submission window.**

4. **Track 2 — Clinical platform (contingent).** Everything beyond
   the open-source research tool — multi-tenancy, audit logging,
   HTTP/API layer, clinician UI, clinical-system integrations. Not
   sequenced; activates only if specific triggers fire (external
   demand, multi-site ambition, funding mandate, publication
   traction). Most of this is background thinking, not planned work.
   The value of keeping it in this document is so that Phase 0 and
   Phase 2 decisions don't accidentally foreclose Track 2 options.

Phases 0 → 1 → 2 form a near-term sequence that culminates in a
paper-plus-release bundle. Track 2 sits outside that sequence and
does not gate any of it.

## Contents

- [Phase 0: C-enabling pipeline work](#phase-0-c-enabling-pipeline-work)
  - [Procrustes preprocessing](#procrustes-preprocessing)
  - [Gait cycle segmentation](#gait-cycle-segmentation)
  - [Joint-angle DTW representation](#joint-angle-dtw-representation)
  - [Provenance subobject](#provenance-subobject)
  - [YAML-configurable analysis pipeline](#yaml-configurable-analysis-pipeline)
  - [Schema migration for VideoPredictions](#schema-migration-for-videopredictions)
- [Phase 1: Clinical validation study (Paper C)](#phase-1-clinical-validation-study-paper-c)
- [Phase 2: Coordinated open-source release + Paper A](#phase-2-coordinated-open-source-release--paper-a)
  - [Release definition](#release-definition)
  - [Apple Silicon CI matrix](#apple-silicon-ci-matrix)
  - [Mac hardware validation pass](#mac-hardware-validation-pass)
  - [Retention and pruning](#retention-and-pruning)
  - [neuropose doctor preflight](#neuropose-doctor-preflight)
  - [Process supervision artifacts](#process-supervision-artifacts)
  - [Structured logging option](#structured-logging-option)
  - [Monitor authentication](#monitor-authentication)
  - [Docker GPU image](#docker-gpu-image)
  - [Dependency freshness automation](#dependency-freshness-automation)
  - [Release workflow](#release-workflow)
  - [Error-path test coverage expansion](#error-path-test-coverage-expansion)
- [Track 2: Clinical platform (contingent)](#track-2-clinical-platform-contingent)
  - [Triggers to activate Track 2](#triggers-to-activate-track-2)
  - [Multi-tenancy and identity](#multi-tenancy-and-identity)
  - [Audit logging and compliance posture](#audit-logging-and-compliance-posture)
  - [HTTP/API layer](#httpapi-layer)
  - [Clinician-facing UI](#clinician-facing-ui)
  - [Horizontal scaling](#horizontal-scaling)
  - [Backup, replication, and data durability](#backup-replication-and-data-durability)
  - [Clinical-system integrations](#clinical-system-integrations)
  - [Deterministic inference mode](#deterministic-inference-mode)
  - [Observability and SLOs](#observability-and-slos)
  - [Supply-chain attestation and signed releases](#supply-chain-attestation-and-signed-releases)
  - [Deployment orchestration](#deployment-orchestration)
- [Decisions to not prematurely foreclose](#decisions-to-not-prematurely-foreclose)

---

## Phase 0: C-enabling pipeline work

The six items below are prerequisites for Paper C. Until they are
landed, every analysis C would produce would be running on defaults
that `RESEARCH.md` §1 explicitly flags as provisional. Ship these
first, in any order that suits the implementer's cadence, and the
rest of the project can pick up with confidence that Phase 1 results
are trustworthy.

### Procrustes preprocessing

**Status:** Not implemented. `neuropose.analyzer.features` ships
`extract_joint_angles` and feature-statistics helpers; no alignment
step exists between pose sequences.

**Why it matters for Paper C:** without alignment, DTW distance is
translation- and orientation-dependent. Two recordings of the same
subject from different camera positions produce different distances,
which is almost never what a clinician wants. Paper C's methods
section would need to apologize for this in print; cheaper to fix the
method than to defend it.

**Scope:**

- Add `procrustes_align(a: np.ndarray, b: np.ndarray, *, mode:
  Literal["per_frame", "per_sequence"]) -> tuple[np.ndarray,
  np.ndarray, AlignmentDiagnostics]` to `neuropose.analyzer.features`.
  Implements the Kabsch algorithm (closed-form optimal rigid
  transform). Per-frame aligns each frame of A to the corresponding
  frame of B independently; per-sequence computes one transform over
  the whole sequence. Both are useful — per-frame for fine-grained
  matching, per-sequence for preserving within-trial dynamics.
- Return aligned arrays plus an `AlignmentDiagnostics` dataclass with
  the fitted rotation magnitude and translation magnitude so
  downstream code can flag suspiciously large transforms (usually a
  sign of upstream annotation error).
- Expose as an opt-in `align: Literal["none", "procrustes_per_frame",
  "procrustes_per_sequence"] = "none"` parameter on every DTW entry
  point in `neuropose.analyzer.dtw`. Default `none` preserves current
  behavior; Paper C's pipeline sets it to `procrustes_per_sequence`.
- Unit tests: construct a known rotation + translation between two
  synthetic skeletons, verify alignment recovers it to within
  floating-point precision; verify alignment of a sequence with its
  own translated copy produces zero residual.

**Non-scope:**

- Non-rigid alignment (thin-plate splines, learned registration). Not
  needed for skeleton-level comparison and would be a research
  contribution on its own.

**Open question:** should alignment also include optional scaling
(scaled-Procrustes / full Procrustes)? For cross-subject comparison
it almost certainly should. Default to scale-preserving and add a
`scale: bool = False` flag; Paper C can flip it on for cross-subject
figures.

### Gait cycle segmentation

**Status:** `segment_by_peaks` in `neuropose.analyzer.segment`
performs generic valley-to-valley segmentation on a supplied 1D
signal. There is no gait-specific wrapper that knows to look at the
heel's vertical coordinate.

**Why it matters for Paper C:** clinical gait analysis wants to
compare *the 4th heel-strike of trial A* to *the 4th heel-strike of
trial B*, not *frame 120 of A vs frame 120 of B*. Per-cycle DTW is
the standard approach in the biomechanics literature (Sadeghi et al.
2000 and descendants); running full-trial DTW on gait is a choice
reviewers of Paper C would correctly flag as methodologically weak.

**Scope:**

- New `segment_gait_cycles(predictions: VideoPredictions, *, joint:
  str = "rhee", axis: Literal["x", "y", "z"] = "y", min_cycle_seconds:
  float = 0.4) -> Segmentation` in `neuropose.analyzer.segment`.
- Under the hood: extract the specified joint's coordinate along the
  specified axis, apply `segment_by_peaks` with appropriate distance
  and prominence thresholds (derived from `min_cycle_seconds` via
  `predictions.metadata.fps`), return the resulting `Segmentation`
  (the existing `neuropose.io.Segmentation` type) so downstream
  tooling picks it up unchanged.
- Two-sided detection: run the same detection on the opposite heel
  and return *both* per-side segmentations under named keys
  (`left_heel_strikes`, `right_heel_strikes`). Clinical users will
  want both.
- Allow the reference joint and axis to be configurable so trials
  recorded with a different camera orientation (lateral vs frontal
  vs oblique) can still be segmented without a code change.

**Non-scope:**

- HMM-based cycle detection, learned cycle detectors. Peak detection
  on vertical coordinate is standard, well-understood, and the
  method the biomechanics literature expects to see.
- Handling pathological gaits where heel-strikes are absent
  (shuffling, walker-assisted). The function should degrade
  gracefully (return a `Segmentation` with an empty list, not raise),
  and Paper C's data-quality filtering handles the rest.

**Open question:** should the function also emit a "confidence" per
cycle (prominence of the detected peak, regularity of spacing) that
Paper C can use to filter out low-quality detections? Cheap to add,
useful downstream. Recommend yes.

### Joint-angle DTW representation

**Status:** `dtw_all`, `dtw_per_joint`, and `dtw_relation` operate on
raw 3D coordinates or joint-pair displacements. `extract_joint_angles`
produces per-frame angle sequences but is not wired as a DTW input.

**Why it matters for Paper C:** angle-space DTW is translation- and
rotation-invariant by construction, scale-invariant if normalized,
and directly interpretable in clinical terms ("knee flexion angle
during swing phase"). Paper C's headline figures almost certainly
use angle-space distances; raw coordinates would draw the obvious
reviewer question of why we aren't comparing the thing clinicians
actually measure.

**Scope:**

- Add `representation: Literal["coords", "angles", "relation"] =
  "coords"` to every DTW entry point. The `coords` default preserves
  existing behavior; `angles` runs `extract_joint_angles` on each
  input first; `relation` is the existing `dtw_relation` path
  expressed as a representation choice rather than a separate
  function (leaving the `dtw_relation` name as a convenience wrapper
  if preferred).
- Degenerate-vector handling: `extract_joint_angles` returns NaN for
  degenerate (zero-length) vectors. The DTW path needs to decide how
  to handle NaN — skip-and-interpolate, drop, or propagate to the
  distance. Propagation is safest (makes the problem visible);
  interpolation is what clinical users probably want day-to-day.
  Default to propagation and expose `nan_policy: Literal["propagate",
  "interpolate", "drop"]` for experimentation.
- Tests: synthetic pair with known angular difference, assert DTW in
  angle-space recovers it independent of global rotation applied to
  the input.

**Non-scope:**

- Quaternion or SO(3) rotation-space DTW. Interesting but requires a
  rotation parameterization the current skeleton output does not
  produce.
- Mixed-representation (position + angle concatenated, learned
  embeddings). These are experiments Paper C might run; they don't
  belong in Phase 0 infrastructure.

### Provenance subobject

**Status:** `PerformanceMetrics` captures `tensorflow_version`,
`active_device`, and `tensorflow_metal_active`. Model SHA is not
computed or propagated. `numpy_version` and `neuropose_version` are
not recorded. No first-class `Provenance` object.

**Why it matters for Paper C:** reproducibility is the first
question a reviewer asks of a clinical-methods paper. The answer
needs to be "same model artifact, same pipeline config, same
versions, same seeds" — and all four need to be recorded on every
`results.json` that underlies a paper figure. Not having this means
either manually tracking it in a lab notebook (fragile, won't
survive personnel turnover) or running every experiment through a
pinned Docker image (expensive, doesn't capture runtime
non-determinism). The subobject is the cheap right answer.

**Scope:**

- New `Provenance` pydantic model in `neuropose.io` with fields:
  `model_sha256: str`, `model_filename: str`, `tensorflow_version:
  str`, `tensorflow_metal_version: str | None`,
  `numpy_version: str`, `neuropose_version: str`, `python_version:
  str`, `seed: int | None`, `deterministic: bool`, `analysis_config:
  dict | None` (the YAML of the run if the pipeline was invoked via
  `neuropose analyze --config`).
- Optional `provenance: Provenance | None = None` field on
  `VideoPredictions`, `JobResults`, and `BenchmarkResult`. None-valued
  on legacy files (enabled by schema migration — see below), populated
  on every new write.
- `_model.py` hashes the downloaded tarball on first load (after the
  existing SHA verification — the two checks use the same hash so
  compute is amortized) and exposes the hash via a
  `get_model_sha256()` method on the `Estimator`. `Interfacer._run_job_inner`
  constructs the `Provenance` and attaches it to the output.
- Unit test: serialize → JSON → deserialize round-trip identity;
  assert `model_sha256` matches the SHA recorded in
  `neuropose._model`.

**Non-scope:**

- Cryptographic signatures on results.json. That's Phase 2 (sigstore
  on release artifacts) or Track 2 (per-output signing) territory,
  not Phase 0.
- Provenance on arbitrary intermediate products (numpy arrays, DTW
  distance matrices). Top-level JSONs cover Paper C's needs; richer
  intermediates can inherit from a hand-off if needed.

**Open question:** does Paper C need *per-frame* provenance (which
frame was processed with which configuration) or just per-job
provenance? Per-job is enough for reproducibility; per-frame is only
useful if we want to mix configurations within a single job, which
has no current use case.

### YAML-configurable analysis pipeline

**Status:** `neuropose.cli`'s `analyze` subcommand is a stub that
raises `NotImplementedError`. Analysis operations are called
individually from Python, or via CLI flags on `segment` and
`benchmark`. No unified representation of "a complete analysis run."

**Why it matters for Paper C:** the paper will run many experimental
configurations — alignment on/off, per-frame vs per-sequence, raw
coordinates vs joint angles, full-trial vs cycle-segmented DTW,
various distance metrics. Each experiment should be reproducible
from a single file that can be version-controlled, diffed, attached
to the `Provenance` object, and cited in the paper. A Python script
full of kwargs is the alternative, and it's exactly the alternative
the open-source community collectively decided against ten years ago.

This item also resolves the "`neuropose analyze`: ship or remove"
question that was previously open: we are shipping `analyze`, just
specifically in a YAML-driven form. The stub that currently exists
becomes the real command in Phase 0.

**Scope:**

- `AnalysisConfig` pydantic model in `neuropose.analyzer` capturing
  the full pipeline: input source (predictions file path),
  preprocessing (`align`, `normalize`, `segment`), per-segment
  analysis (DTW backend, representation, distance function, extra
  kwargs), output (figures, statistics, distance matrices).
- Parseable from YAML via pydantic; validated on parse so typos in
  field names fail early with a clear error.
- `neuropose analyze --config experiment.yaml [--output
  results_042.json]` runs the pipeline end-to-end. The config YAML
  is serialized into the resulting `Provenance.analysis_config`, so
  the output file is self-describing.
- Ship three or four *example* configs under `examples/analysis/`
  that exercise the full matrix of alignment × representation ×
  segmentation choices Paper C will care about. Double as integration
  tests.

**Non-scope:**

- A DAG / workflow engine (Snakemake, Nextflow). A flat config is
  enough for Paper C's needs; reach for a DAG tool only when
  experiments have genuine inter-stage dependencies, which analysis
  of a single video does not.
- Parallel sweep execution. Run multiple configs via a shell loop
  for now (`for cfg in examples/analysis/*.yaml; do neuropose
  analyze --config "$cfg" --output "out/$(basename "$cfg" .yaml).json"; done`).
  A real sweep orchestrator is Track 2.

**Open question:** should there be a `neuropose analyze compare
<config_a.yaml> <config_b.yaml>` subcommand that runs both and
emits a diff figure? Useful for Paper C but not a gating feature —
post-Phase-0 addition if the need is clear.

### Schema migration for VideoPredictions

**Status:** `VideoPredictions` gained `segmentations: dict[str,
Segmentation] = Field(default_factory=dict)` during recent work. Old
JSON files without the field still load (pydantic default-factories
back-fill), but this is accidental rather than designed-in.

**Why it matters for Paper C:** Paper C will produce analysis results
over the course of 6-12 months. During that window, Phase 0 work
itself will evolve — the `Provenance` object will gain fields, the
`AnalysisConfig` shape will stabilize, maybe the `Segmentation` schema
will extend. Without migration support, every schema change would
invalidate some portion of Paper C's already-generated data, forcing
either a freeze (drops velocity) or a full re-run (wastes compute).
Migration now is the cheap fix.

**Scope:**

- Add a `schema_version: int = 1` field to `VideoPredictions`,
  `JobResults`, and `BenchmarkResult` (the three load-anywhere
  top-level schemas).
- Write `migrate_video_predictions(payload: dict) -> dict` that
  takes a raw JSON-loaded dict, dispatches on `schema_version`, and
  returns a dict conformant with the current version. Default to 1
  when missing (existing files).
- Wire it into `load_video_predictions()` so the migration runs
  before pydantic validation. Log at INFO on migration so users see
  when files are being upgraded.
- When writing, always write the current version.

**Non-scope:**

- A general-purpose migration framework. A function that dispatches
  on an integer is sufficient until we have three versions.
- In-place migration (writing back the upgraded file). Migrations
  should run on read; write-back is a separate operator decision.

---

## Phase 1: Clinical validation study (Paper C)

Phase 1 is *Paper C itself* — the clinical-methods paper this project
exists to produce. The content belongs in the paper, in `RESEARCH.md`,
and in the analysis-config YAMLs under `examples/`, not here. This
section exists only to demarcate the phase and to capture the
engineering commitments that should (and should not) happen during it.

**Engineering posture during Phase 1:**

- **Phase 0 is frozen on entry.** Don't refactor the analyzer during
  Phase 1; refactors invalidate earlier experiments. If a Phase 0
  shortcoming surfaces during paper-writing, log it in `RESEARCH.md`
  and revisit after submission.
- **Phase 2 work is welcome as background.** Writing a launchd plist,
  wiring up Dependabot, tightening error-path tests — all of this is
  ideal filler work during the experimental-design and writing
  phases of Paper C. It consumes different energy than research work
  does, and the tool is in better shape on submission day as a
  result.
- **`RESEARCH.md` gets the bulk of the updates.** Methods decisions,
  reading-list expansions, reviewer-response notes all live there,
  not here.
- **Do add engineering-side notes here** when a Paper C experiment
  reveals a piece of missing tooling that's worth a Phase 2 item
  (for example: "we needed batch-analysis across 200 trials and hit
  this, so Phase 2 should include ..."). Phase 1 is the best
  possible source of prioritization signal for what Phase 2 is
  actually worth.

**Prerequisite outside this document:** a MoCap-data-access
conversation with Dr. Shu. Nothing in Phase 1 can start until that
conversation has resolved. `RESEARCH.md` §3 flags this as the
gating question for fine-tuning; it is equally the gating question
for validation.

---

## Phase 2: Coordinated open-source release + Paper A

Phase 2 is the release. Its content is exactly the items listed here
— the engineering work to take the Phase-0-plus-Phase-1 codebase to a
state where an outside researcher can pick it up, install it, run it,
verify its claims, and cite it. It runs concurrently with the tail
end of Phase 1 (see posture notes above) and culminates in a
coordinated drop: tag → PyPI → Pages → arXiv / JOSS submission for
Paper A → reference in Paper C's Code Availability section.

### Release definition

Before enumerating the remaining work, define what "released" means.
A release candidate should satisfy all of the following:

1. **Installable on a blank machine.** `pip install neuropose` or
   `uv pip install neuropose` works on both Linux x86_64 and Apple
   Silicon Mac, with no manual steps beyond Python 3.11.
2. **Runnable without the author in the room.** The `docs/` site is
   published somewhere persistent (GitHub Pages, Cloudflare Pages),
   the getting-started walkthrough actually works end-to-end, and
   the MeTRAbs model downloads and verifies on first run.
3. **Verifiable by a reviewer.** CI runs on every push, covers both
   Linux and macOS, and a PR from a stranger could be meaningfully
   reviewed without access to the research Mac.
4. **Honest about its limits.** Every surface the release advertises
   is either exercised in CI or clearly marked experimental. No
   false promises in the README or CLI help text. (The `analyze`
   stub that motivated this item pre-Phase-0 is now real per Phase
   0's YAML pipeline, so "ship or remove" is no longer open.)
5. **Versioned.** A git tag exists, `__version__` matches, and
   `CHANGELOG.md` has a real release section, not just `[Unreleased]`.
6. **Bundled.** Paper A (tech-stack writeup) and Paper C (clinical
   validation) cite the release tag, and the release notes cite
   them. The three artifacts arrive together; reviewers of either
   paper can find and run the code.

Items below are the gaps between the end-of-Phase-0 state and that
definition.

### Apple Silicon CI matrix

**Status:** `RESEARCH.md` lists this as an open next step; no
`macos-14` entry in `.github/workflows/ci.yml`.

**Why it matters for release:** every claim of "Apple Silicon
support" is currently "by construction" — the TF 2.16+ floor ships
`darwin/arm64` wheels, the MeTRAbs SavedModel has zero custom ops, and
therefore it should work. It has not been empirically confirmed on
real hardware in an automated way. For a public release, we either
verify in CI or we stop claiming Mac support in the README.

**Scope:**

- Add a `macos-14` matrix entry to the `test` job (lint and typecheck
  stay single-platform, they're platform-independent).
- Exclude `slow` markers on macOS so we don't pay the 2 GB model
  download twice per run.
- Accept that the first green macOS run may require two or three
  hotfixes — path case sensitivity, `multiprocessing` spawn vs fork,
  shared library load order — and budget a day for that.
- Do **not** add a Metal runner. GitHub's `macos-14` runners don't
  expose the GPU to TensorFlow in a useful way, and the `[metal]`
  extra's numerical verification is a separate task that needs real
  M-series silicon we control.

**Sketch:**

```yaml
test:
  strategy:
    fail-fast: false
    matrix:
      os: [ubuntu-latest, macos-14]
  runs-on: ${{ matrix.os }}
```

Everything else in the job stays the same; `uv` works identically on
both platforms.

### Mac hardware validation pass

**Status:** Unexercised. The Shu Lab research Mac (`100.64.15.110`) is
available; we have an rsync script but no cron job, no automated
smoke check, no numerical-divergence report against the Linux
baseline.

**Why it matters for release:** CI on GitHub's `macos-14` runners
validates that the wheels install and the tests pass on Apple
Silicon. It does not validate that the real MeTRAbs model loads, that
inference runs, or that `poses3d` on the Mac matches `poses3d` on
Linux within a sane tolerance. Those are different questions, and
answering them against a throwaway runner each time would be wasteful
and unreliable.

A minimum version of this check — "does `detect_poses` produce
output on the research Mac at all?" — should happen during Phase 0
regardless, because Paper C will likely run on the same hardware and
a silent numerical divergence there would invalidate the paper's
results. The scope below is the full, release-grade version.

**Scope:**

- Run `neuropose benchmark --compare-cpu` against a reference clip on
  the research Mac. Capture the resulting `BenchmarkResult` JSON.
- Commit the JSON as `benchmarks/reference/mac_m3_ultra_cpu_v0_1.json`
  (a tracked file, not gitignored — this is the reference numerics
  we'll compare against going forward).
- Separately, run the `[metal]` path and diff. Record in
  `RESEARCH.md` whether divergence is within the ~1e-2 mm budget the
  research notes propose, or whether the Metal path is in the "use at
  your own risk" column.
- Document the findings as a new section in `RESEARCH.md` ("Apple
  Silicon verification, 2026-0X") and close the corresponding
  open-question entry.

**Open question:** should the reference JSON become a test input
(slow-marked integration test that re-runs benchmark on a developer's
machine and asserts divergence from the committed reference), or just
documentation? The former catches regressions automatically at the
cost of a 2 GB model download in the slow job; the latter is cheaper
but easier to ignore.

### Retention and pruning

**Status:** `out/` and `failed/` grow forever. No retention config.
No `neuropose prune` command.

**Why it matters for release:** a research Mac running the daemon
unattended for months will fill its disk. The first support request
will be "the daemon just stopped working" and the answer will be "you
ran out of disk." We can solve this once now, or a hundred times
later.

**Scope:**

- Add a `retention_days: int | None = None` setting (default None =
  disabled, preserving current behavior).
- When set, the daemon checks on each poll whether any job in
  `out/` or `failed/` is older than the threshold and removes it. The
  corresponding `status.json` entry transitions to a new `PRUNED`
  state (keeping the audit trail) or is removed entirely (keeping the
  status file small) — pick one and document.
- Ship a `neuropose prune [--older-than N] [--dry-run]` one-shot
  command for operators who want manual control.
- Document in `docs/deployment.md` with a recommended default (30
  days feels right for benchmark/iteration workflows; clinical
  deployments would be legal-driven and much longer).

**Open question:** should pruned jobs' `status.json` entries be
preserved as tombstones (so a user asking "where did job X go?" can
see "pruned 2026-05-01") or removed entirely? Tombstones are more
user-friendly; removal keeps the status file bounded. Default to
tombstones since the status file bound is only a problem at a scale
the 0.1 release won't hit.

### neuropose doctor preflight

**Status:** Not implemented.

**Why it matters for release:** pydantic-settings validates the
*schema* of `Settings` (is `device` a valid string, is
`poll_interval_seconds` positive). It does not validate the
*environment* — is `data_dir` writable, is the lock file acquirable,
is `model_cache_dir` on the same filesystem as `data_dir` (so
`os.rename` works atomically), is the configured TF device actually
available. Each of those is a runtime failure mode that shows up with
an ugly traceback ten seconds after `neuropose watch` starts, and
every one is cheaply detectable at startup.

**Scope:**

- New subcommand `neuropose doctor` that runs a battery of
  preflight checks and prints a pass/fail table.
- Checks to include: `data_dir` exists and is writable; lock file
  acquirable (with clean release); all three subdirectories
  (`in/out/failed`) writable; `model_cache_dir` writable and on the
  same filesystem as `data_dir`; TF is importable; configured
  `device` is in `tf.config.list_physical_devices()`;
  `tensorflow-metal` either absent or installed with a version that
  advertises support for the installed TF; XDG envvars are sane;
  Python version matches `pyproject.toml` floor.
- Exit code 0 if all checks pass, 1 if any warning, 2 if any fatal
  failure.
- The daemon's `run()` entry point calls the same underlying
  preflight function before entering the poll loop, so
  `watch`-without-doctor still gets the benefit.

**Non-scope:**

- Do not check for network access to the MeTRAbs download host.
  Network-dependent checks make CI flaky and don't match the offline
  caching behavior of real operators.

### Process supervision artifacts

**Status:** `docs/deployment.md` documents a systemd user unit as
text in prose. No file in `scripts/` that a user can actually copy.
No macOS launchd plist at all.

**Why it matters for release:** copy-paste from a docs page into a
`.service` file works, but it's friction. An open-source project with
"here is the file, here is where it goes, here is the enable command"
ships deployments faster.

**Scope:**

- Ship `scripts/systemd/neuropose.service` as a file with `%h`
  placeholders and a short install README.
- Ship `scripts/launchd/org.levineuwirth.neuropose.plist` as a file
  with an install README. (Consider making the plist label match the
  reverse-DNS of whoever is hosting — either the lab's or
  `org.neuropose.daemon` for a vendor-neutral identity.)
- Optional: a `scripts/install_service.sh` that detects the platform
  and runs the right install command. Probably not worth the
  complexity; a five-line README section per platform is fine.

**Non-scope:**

- Do not write installers for init systems we do not personally run
  (upstart, sysvinit, runit). If someone needs those, the systemd
  unit gives them enough of a template.

### Structured logging option

**Status:** Everything logs to stderr via `logging.basicConfig`
with a human-readable formatter.

**Why it matters for release:** the current format is correct for
interactive use. For any consumer that wants to feed the daemon's
output into Loki, Splunk, Grafana, Datadog, or even `jq`-based
aggregation, JSON-per-line would eliminate a parsing step. This is
a near-free feature if added now and a disruptive formatting change
if added later. It is also a prerequisite for any Track 2
audit-logging work, so building it now keeps Track 2 options open at
near-zero cost.

**Scope:**

- Add a `--log-format={human,json}` global CLI option defaulting to
  `human`.
- Implement the `json` variant as a formatter that emits
  `{"ts": ..., "level": ..., "logger": ..., "message": ..., ...}` per
  line with no log-line wrapping.
- Wire it through `_configure_logging()` so every subcommand benefits
  identically.

**Open question:** do we also want log correlation IDs per job?
That's a bigger change (pushing a context var through the
Interfacer's call stack) and probably Track 2 — skip for 0.1.

### Monitor authentication

**Status:** The monitor binds to `127.0.0.1:8765` by default. No
auth, no tokens. `--host 0.0.0.0` works but has a comment warning the
operator to think.

**Why it matters for release:** loopback-only is a reasonable
default, but the monitor is specifically marketed as the thing
collaborators can watch. "Collaborator" implies a browser somewhere
other than the daemon host. The "correct" answer (TLS, real auth) is
too expensive for 0.1; the "wrong but acceptable" answer (no auth, so
anyone who can reach the port sees everything) is what we have now.
There's a middle ground.

**Scope:**

- Add an optional `monitor_token: str | None = None` setting.
- When set, every request to `/` and `/status.json` must carry
  `?token=<value>` in the query string or `X-Status-Token` in the
  header. No token → 401.
- `neuropose serve` prints a URL including the token on startup, so
  operators can copy-paste it. If `monitor_token` is unset, behavior
  is unchanged.
- `--host 0.0.0.0` emits a stderr warning if `monitor_token` is unset
  — don't block it, just flag it.

**Non-scope:**

- TLS. Use a reverse proxy (Caddy, nginx, `ssh -L`) for any
  internet-facing exposure. The monitor is not the right place to
  terminate TLS.
- Multi-user auth, session cookies, anything with a database. That's
  Track 2.

### Docker GPU image

**Status:** `Dockerfile` exists (CPU-only). `Dockerfile.gpu`
mentioned in CHANGELOG as planned.

**Why it matters for release:** a single-file CUDA deployment story
reduces "can I run this on our lab server?" from a 45-minute dance
with conda and CUDA versions to one `docker run`. For Linux GPU
users this is the friction difference between trying the project and
bouncing.

**Scope:**

- Write `Dockerfile.gpu` on top of `nvidia/cuda:12.x-runtime-ubuntu22.04`
  (pick the version TF 2.18 actually supports — check the
  `tensorflow-gpu` compat matrix, not just "latest").
- Multi-stage: build stage has `uv` and builds the venv; final stage
  just copies the venv and sets entrypoints.
- Add a `docker-build.yml` CI workflow that builds both images on
  every push to main and publishes as `ghcr.io/neuwirth/neuropose:cpu`
  and `:gpu` (or wherever the project ends up hosted).
- Document in `docs/deployment.md` with a `docker run --gpus all`
  example.

**Non-scope:**

- A `tensorflow-metal` Docker image. Mac can't virtualize Metal, so
  there's no point.

### Dependency freshness automation

**Status:** No Dependabot, no Renovate. Everything floats until
somebody notices. The recent TF cap tightening (`<2.19`) was caught
manually because a user happened to ask; a scheduled bot would have
flagged it weeks earlier.

**Why it matters for release:** security CVEs on transitive
dependencies land every few weeks. Without automation, they get
discovered by a downstream user trying to install into an audited
environment. With automation, they become a PR you either merge or
explicitly decline.

**Scope:**

- Add `.github/dependabot.yml` with groups: `python-prod`,
  `python-dev`, `github-actions`. Weekly schedule. Ignore `tensorflow`
  updates until manually cleared (the `tensorflow-metal` constraint
  means auto-bumping TF is destructive).
- Alternative: Renovate via `renovate.json`. Renovate has better
  grouping and scheduling, Dependabot is simpler and needs no setup
  on GitHub. For an open-source Brown-lab project, Dependabot is
  enough.
- Add `uv lock --upgrade-package <name>` to the dev playbook in
  `docs/development.md` so PR authors know how to re-lock.

### Release workflow

**Status:** `[project.scripts]` is wired for `pip install`, but no
tag-triggered publishing pipeline. `.github/workflows/docs.yml`
uploads the built docs as a 14-day artifact, not to Pages.

**Why it matters for release:** "release" without a repeatable
publishing flow is a synonym for "one-off person runs hatch build on
their laptop at 11pm before the paper deadline." That is not a
release.

**Scope:**

- `.github/workflows/release.yml` triggered on version tags
  (`v[0-9]+.[0-9]+.[0-9]+`). Steps: check version matches
  `__version__`; build with `hatch build`; publish to PyPI via
  trusted publisher (no long-lived token); create GitHub release with
  changelog excerpt.
- Flip `docs.yml` to deploy the `site/` output to GitHub Pages on
  every push to `main` once the repo is public. Pin the Pages URL in
  the README and in `site_url` in `mkdocs.yml` (already points at
  `levineuwirth.github.io`, but verify).
- Sign tags with GPG; document the key fingerprint in `SECURITY.md`
  (which does not yet exist; create it).
- Consider wiring sigstore signing at the same time — see Track 2
  supply-chain section. Free after the initial setup and buys
  everything Track 2 would want without committing to the rest of
  that track.

**Open question:** do we publish under `neuropose`, `brown-neuropose`,
or something else on PyPI? Whichever name, squat it before the paper
drops — waiting means risking namesquatter abuse.

### Error-path test coverage expansion

**Status:** Happy paths and a handful of input-validation errors
covered. Not covered: disk full mid-write, corrupt video mid-decode,
OOM during inference, fcntl.flock on NFS (no-op on some kernels),
truncated zip archives, permission denied on data_dir subdirectories.

**Why it matters for release:** shipping a tool where "happy path
works" is different from shipping a tool where "when it fails, it
fails predictably." For a clinical research pipeline where a crash
mid-job quarantines valuable recording data, fault tolerance is a
feature.

**Scope:**

- Systematic pass: for each module, write a `test_<module>_failure_modes.py`
  enumerating the specific exception classes that can escape and the
  corresponding test case that triggers each one. Use `pytest.raises`
  with the exact expected exception class.
- Hardest cases use fixtures that monkeypatch system calls
  (`os.write` raises OSError(ENOSPC), `cv2.VideoCapture.read` returns
  `False, None` partway through, `fcntl.flock` raises OSError(EBADF)).
- Aim: every user-facing error message in the codebase has a test
  that proves it's reachable.

**Non-scope:**

- Chaos-engineering frameworks. `monkeypatch` is enough.
- Covering unrecoverable errors like SIGKILL of the daemon mid-frame.
  That's the recovery-on-startup test, which already exists.

---

## Track 2: Clinical platform (contingent)

Track 2 is everything beyond the open-source research tool —
multi-tenancy, audit logging, HTTP/API layer, clinician UI,
clinical-system integrations, the works. None of it is sequenced
with Phases 0–2; all of it is gated on specific triggers that don't
exist yet.

### Triggers to activate Track 2

Do not start Track 2 work until at least one of the following is
true:

1. **External demand.** Another clinical group has asked for a
   deployment they can run independently. Not a casual "interesting
   project" — a specific ask with a specific cohort and a specific
   timeline.
2. **Multi-site ambition.** The Shu Lab decides to run NeuroPose
   across more than one site within Brown-affiliated clinical
   systems, and the single-host assumption stops working.
3. **Funding mandate.** A grant or contract specifies outputs that
   the Phase 0-1-2 deliverables cannot meet (e.g. "produce a
   HIPAA-compliant deployment," "integrate with the EHR").
4. **Publication traction.** Papers A and C get engagement that
   translates into demand for a hosted version. Clinical-methods
   papers occasionally do. If enough unsolicited inquiries land,
   Track 2 becomes worth the investment.

Before at least one of these triggers: everything below is
background thinking, not planned work. *Do not refactor Phase 0 or
Phase 2 code to make Track 2 easier.* Every such refactor is a bet
on a future that may not arrive.

### Multi-tenancy and identity

**What it would require:**

- A concept of "user" distinct from "OS user." Today `Settings.data_dir`
  is one directory per OS user; multi-tenancy means one `data_dir`
  serving many logical tenants with enforced isolation.
- Per-tenant namespacing in `in/`, `out/`, `failed/`, and
  `status.json`. Cleanest is one subdirectory per tenant with the
  same four-directory layout; the daemon's discovery logic becomes a
  two-level scan.
- Authentication on the control plane. Passing tenant identity as a
  command-line arg is fine for a research prototype; a real
  deployment needs OAuth/OIDC or SAML with the institution's IdP
  (Brown CAS, epic Auth, whatever the target site uses).
- Authorization model: at minimum, "tenant A cannot see tenant B's
  jobs." For clinical deployments, probably also role-based (clinician
  / PI / admin / auditor).

**Cheapest path forward if a trigger fires:** fork the data-directory
layout into `$data_dir/<tenant_id>/{in,out,failed,status.json}`,
teach the daemon to iterate tenants in its poll loop, add a
`--tenant` flag to the CLI. That's enough for an invitation-only
deployment where tenants are identified by opaque string and issued
out-of-band.

**Expensive path:** anything involving an identity provider. Don't
go there without a real operator committing to the deployment.

### Audit logging and compliance posture

**What it would require:**

- Append-only log of every data access, write, and configuration
  change, with actor identity and timestamp. Separate from the
  application log (which rotates).
- Logs streamed to a write-once sink (S3 with object-lock,
  immutable journal) so a compromised host can't rewrite the
  trail.
- Legal review: what exactly does HIPAA require of this tool? What
  about institutional IRB? The answer will differ across sites and
  the project cannot prescribe it — but the *capability* to generate
  the required logs needs to be built in.
- Retention policy wired to the audit log, not just application
  state. Pruning job results is different from pruning audit records.

**Technical prerequisite:** structured logging from Phase 2 (which
is a cheap add and is scheduled anyway). Without JSON-per-line logs,
audit extraction is a grep-and-pray regex problem.

### HTTP/API layer

**What it would require:**

- Today the control plane is "write files to `in/`." For a
  non-filesystem-native consumer (a hosted web UI, a batch scheduler,
  a Jupyter kernel in a different container), an HTTP API is the
  right abstraction.
- FastAPI or Litestar on top of the existing ingest/interfacer/io
  modules. The daemon becomes a long-running process that serves
  requests *and* processes the input directory; or the daemon stays
  headless and the HTTP layer is a separate process talking via the
  same filesystem contract.
- OpenAPI schema published as part of the release so client code can
  be generated.

**Non-obvious pitfall:** the daemon's fcntl-based single-instance
lock assumes one writer. If the HTTP layer is a separate process, it
needs to go through the same ingest API, not directly into `in/`.
That's an easy discipline to establish if designed in from day one,
a painful refactor later.

**Cheap Phase 0/2 precaution:** keep `neuropose.ingest` and
`neuropose.interfacer` API-stable as Python modules. If a future
HTTP layer imports them, we don't want to break the import.

### Clinician-facing UI

**What it would require:**

- More than the `neuropose serve` dashboard — an actual web
  application with clinician-facing views: patient list, session
  list, session-level pose visualization, comparison against
  reference motion, exportable reports.
- Probably React + TypeScript on the frontend, consuming the HTTP
  API above. Backend-rendered templates would be faster to build but
  a worse fit for the per-session interaction model clinicians
  expect.
- WebGL or Three.js for 3D pose playback. The `neuropose.visualize`
  module is a matplotlib-based still-frame tool; rebuilding it for
  interactive 3D is a weeks-to-months project on its own.
- Accessibility: clinician environments include keyboard-only users,
  users on institutional IE holdovers (yes, still), users with
  screen readers. A research-grade UI ignores this; a clinical-grade
  one cannot.

**Scope is enormous.** This is the single largest piece of Track 2
and would likely dwarf all other Track 2 work combined. Would not
start without dedicated frontend engineering effort.

### Horizontal scaling

**What it would require:**

- A message broker (Redis Streams, RabbitMQ, or NATS) in place of the
  filesystem poll. Each job becomes a broker message; multiple
  worker processes consume and process in parallel.
- Shared storage for inputs and outputs (S3, MinIO, NFS). The
  "job_name is a directory" contract generalizes to "job_name is an
  object prefix."
- Per-worker GPU affinity for the multi-GPU case; worker auto-sizing
  based on queue depth.
- Distributed lock for the leader-only work (status file writes,
  retention enforcement).

**Upgrade path that minimizes pain:** the current single-process
daemon is equivalent to the "one worker" case of a horizontal
deployment. If the job object in `neuropose.io` stays the source of
truth (not the filesystem layout), the transition is backend-swap,
not architectural surgery. Keep that option open by treating the
filesystem as an implementation detail of `Interfacer`, not a public
contract.

### Backup, replication, and data durability

**What it would require:**

- Outputs (`out/<job>/results.json`) currently live on one disk on
  one host. For clinical data this is insufficient durability.
- Replication target: another host (hot standby), object storage
  (warm archive), or both. The `out/` directory is the canonical
  store; replicating it periodically is a scriptable cron job today.
- Proper replication: as writes happen, not as a cron. Either a
  daemon-side hook that PUTs to S3 immediately after each
  `save_job_results`, or a sidecar process watching the filesystem
  with `inotify`/`fswatch`.
- Restore story: how do we restore `out/` from backup without
  breaking `status.json` (which refers to job names by convention)?
  Test this annually.

**Minimum viable backup for Phase 2:** add a `scripts/backup.sh`
that rsyncs `$data_dir/out/` to a configurable destination. Not a
feature; a paving-the-path-for-operators artifact.

### Clinical-system integrations

**What it would require:**

- **DICOM** if videos are stored as DICOM instances rather than
  MP4. Clinical motion-analysis devices increasingly output DICOM
  video; reading DICOM means `pydicom` + some decoding logic.
- **FHIR** for patient metadata. If NeuroPose is to accept a
  patient ID and attach it to a session, that ID probably comes
  from a FHIR Patient resource. Means speaking FHIR to the hospital's
  FHIR endpoint (Epic, Cerner).
- **Redcap** integration for clinical-research cohorts (the Brown
  ecosystem uses it heavily). An export script that pulls subject
  metadata from a RedCap project and lays it into the ingest
  directory is cheap and valuable.

**Order of likely need:** RedCap first (easy, valuable, Brown-local),
then DICOM (depends on what the recording device outputs), then
FHIR (only if we're pulling from an EHR, which we probably aren't
for research).

### Deterministic inference mode

**What it would require:**

- Phase 0's `Provenance` object already captures model SHA, TF
  version, NumPy version, and a seed field. The missing piece for
  strict reproducibility is forcing TensorFlow itself to behave
  deterministically —
  `tf.config.experimental.enable_op_determinism()` plus seeding all
  of `random`, `numpy.random`, and `tf.random`.
- A `deterministic: bool = False` setting on `Settings` that flips
  the above. Default off, because deterministic mode costs a
  meaningful fraction of throughput on GPUs and isn't free on CPUs
  either. Clinical deployments would turn it on; benchmark runs
  would turn it off.
- A `Provenance.deterministic` boolean field is already in the Phase
  0 scope; this item closes the loop by actually making that
  boolean mean something.

**Cheap Phase 2 precaution:** wire the setting in Phase 2 even if we
don't flip it on by default. Future Track 2 deployments can flip it
without a code change.

### Observability and SLOs

**What it would require:**

- Prometheus metrics endpoint (separate port from the monitor, no
  auth needed on metrics, loopback or behind a scraper only).
- Counters: jobs_processed, jobs_failed, frames_processed, bytes_read,
  bytes_written. Histograms: per-frame latency, per-job latency,
  per-video latency. Gauges: queue depth, active job count.
- Tracing: OpenTelemetry instrumentation on job_process,
  detect_poses, save_job_results. Again, the interesting spans are
  the long ones, so trace-sampling at 100% is usually fine until
  throughput matters.
- Defined SLOs: "99% of jobs complete within 10× video duration,"
  "95% of monitor requests return in under 100 ms," etc.
  SLO definitions go into a `docs/slos.md`; burn-rate alerting is
  the operational half.

**Order-of-magnitude** dependency: none of this is useful without
Track 2 demand. A single-user research Mac doesn't have SLOs.

### Supply-chain attestation and signed releases

**What it would require:**

- SBOM generation on every release (CycloneDX or SPDX format,
  attached to the GitHub release and published alongside the wheel).
- Signed releases: sigstore / cosign signatures on the wheel, the
  Docker images, and the source tarball. GitHub's OIDC +
  sigstore makes this a ten-line workflow once. For a clinical tool,
  a reviewer being able to verify "this wheel is the one GitHub
  Actions produced from this commit" is non-negotiable.
- Reproducible builds: same source → same wheel hash. Python wheels
  are usually reproducible with `SOURCE_DATE_EPOCH` set and `.pyc`
  exclusion; document the exact command.
- Provenance attestations (SLSA level 2 or 3) for the CI pipeline.
  GitHub's `attestations/build-provenance` action does this.

**Cheapest Phase 2 precaution:** wire sigstore signing into the
release workflow when it's first built (see Phase 2 release workflow
section). Free after the initial setup.

### Deployment orchestration

**What it would require:**

- Kubernetes manifests (Helm chart, probably). Pod specs for the
  daemon, the monitor, the HTTP API. Separate deployments so they
  can scale independently.
- Terraform or Pulumi for the underlying infrastructure: GPU
  node pool, object storage, IAM, TLS termination. Site-dependent;
  Brown runs primarily on-prem with some AWS — the IaC would need
  to target both.
- Secrets management: Vault, AWS Secrets Manager, or K8s
  Secrets + External Secrets Operator. The monitor token, the
  broker credentials, the object-storage keys all need to stop being
  env vars in a `.service` file.

**Strong recommendation:** do not write any of this until there is
a specific deployment with specific operators. Generic K8s manifests
written without a target are a solution in search of a problem, and
they age fast.

---

## Decisions to not prematurely foreclose

A short list of choices we should avoid making in Phase 0 or Phase 2
that would make Track 2 more expensive later:

1. **Keep `neuropose.ingest` and `neuropose.interfacer` API-stable
   as Python modules.** A future HTTP layer should be able to import
   them. Avoid adding `@staticmethod` decorators that hide internal
   state; avoid coupling to global config.
2. **Keep the filesystem layout reversible.** Anything in
   `$data_dir` that is not a user artifact should be treated as
   internal. If Track 2 wants to replace the filesystem with an
   object store, the daemon's only file I/O should be via
   `neuropose.io` helpers — no raw opens scattered through the code.
3. **Keep `VideoPredictions.provenance` extensible.** The Phase 0
   `Provenance` model should be a pydantic model so fields can be
   added backward-compatibly. Don't pack provenance into free-form
   strings or nested dicts that require bespoke parsing.
4. **Keep the CLI subcommands orthogonal.** Do not add subcommands
   that wrap multiple other subcommands for convenience; that
   creates API shape we'd regret if the right composition layer
   later is HTTP, not shell.
5. **Keep model loading behind `neuropose._model`.** A future
   self-hosted model registry, signed-artifact verification, or
   multi-model switching should be a change in one file, not a
   refactor across the estimator.
6. **Keep `Settings` the single source of truth.** No `os.environ`
   reads outside pydantic-settings; no sprinkled `Path.home()`
   calls. Track 2 almost certainly overrides configuration from
   a secret store, and if that override has one place to hook in,
   it's easy.
7. **Keep status-file schema owned by pydantic, not hand-written
   JSON.** Track 2 multi-tenancy means indexing into the status
   file by tenant; a pydantic model refactor is cheap, a
   hand-written dict refactor is not.
8. **Keep the `AnalysisConfig` shape additive.** The Phase 0 YAML
   schema will evolve through Phase 1 as Paper C's experiments
   surface needs. Additions are free (new optional fields);
   renames and removals invalidate prior experiments. Pydantic's
   `extra="forbid"` catches typos at parse time while still
   allowing additive extension.

These are cheap-now / expensive-later items. Every other Track 2
decision can wait for a real trigger.
