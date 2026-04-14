# NeuroPose Research and Ideation Notes

A living R&D log for open design questions, speculative directions, and
planned experiments that are larger in scope than individual commits.
This is **not** user-facing documentation — items in here are
*candidates* for future work, and inclusion does not imply commitment.

## How to use this document

- Add a section when you start thinking about a new area of investigation.
- Each section should end with an **Open questions** or **Next steps**
  block so it's obvious to a future you (or a new contributor) what the
  active threads are.
- When something in here is decided and implemented, move it to the
  relevant place in `docs/` or in the code itself and leave a short
  pointer behind ("*See `docs/architecture.md` for the resolved design.*").
- Consider the audience: yourself, Dr. Shu, David, Praneeth, and future
  contributors. Assume they know pose estimation at a grad-student level
  but may not have followed every prior conversation.

## Contents

- [DTW methodology](#dtw-methodology)
- [TensorFlow version compatibility](#tensorflow-version-compatibility)
- [MeTRAbs hosting and extensibility](#metrabs-hosting-and-extensibility)

---

## DTW methodology

### Current implementation (v0.1, commit 10)

`neuropose.analyzer.dtw` ships three entry points, all built on top of
[`fastdtw`](https://github.com/slaypni/fastdtw) with
`scipy.spatial.distance.euclidean` as the point-distance function:

- **`dtw_all(a, b)`** — single DTW on flattened `(frames, joints × 3)`
  vectors. One scalar distance for the whole sequence.
- **`dtw_per_joint(a, b)`** — one DTW call per joint, returning a list
  of per-joint distances and warping paths. Preserves per-joint
  temporal alignment at J× the cost.
- **`dtw_relation(a, b, joint_i, joint_j)`** — DTW on the per-frame
  displacement vector between two specific joints. The intent here is
  to capture "how does the relationship between these two joints change
  over time", which is translation-invariant and so immune to raw
  camera-frame changes.

These three correspond directly to the three helpers that existed
(broken) in the previous prototype's `analyzer.py`, ported forward with
bug fixes, types, and tests. **The port was mechanical — not a
methodological choice.** We inherited the FastDTW + Euclidean defaults
without validating them against the clinical research use cases, and
that validation is overdue.

### Known limitations of the v0.1 approach

#### FastDTW is an approximation, not exact DTW

[FastDTW](https://cs.fit.edu/~pkc/papers/tdm04.pdf) is a multi-scale
approximation that runs in linear time by recursively refining a coarse
alignment. For the radius-based implementation in
`slaypni/fastdtw`, the distance is not guaranteed to match exact DTW,
and in pathological cases the error can be significant. For a research
codebase where the DTW distance is going to show up in a figure, that
matters.

**Candidate exact alternatives** (all pip-installable):

- [`dtaidistance`](https://github.com/wannesm/dtaidistance) — C-based,
  supports both exact DTW and a `fast=True` approximation; also
  supports shape-DTW and various constraint bands. Actively maintained,
  and the underlying algorithms match the textbook.
- [`tslearn`](https://tslearn.readthedocs.io/) — ML-flavored toolkit
  with exact DTW, soft-DTW (differentiable), Sakoe-Chiba banding, and
  kernel-DTW. Good fit if we ever want to feed DTW distances into an
  sklearn/PyTorch pipeline.
- [`cdtw`](https://github.com/statefb/dtw-python) / `dtw-python` —
  Python port of the R `dtw` package; exhaustive options for windowing,
  step patterns, and open-ended alignment. Less friendly API but the
  most rigorously documented.

#### Euclidean is a choice, not a default

Treating `(x, y, z)` joint positions as a point in R³ and taking
Euclidean distances implicitly assumes the three axes are commensurable
in the same units, which is fine for MeTRAbs (mm) but throws away prior
knowledge about human motion. Alternatives worth considering:

- **Angular distance on joint angles.** Compute joint angles per frame
  (`extract_joint_angles` already exists) and run DTW on the angle
  sequences rather than raw coordinates. Translation- and
  scale-invariant by construction; well-matched to clinical metrics
  like knee flexion angle.
- **Geodesic distance on SO(3)** for local joint rotations. Requires a
  skeleton-rooted rotation parameterization; more work to set up but
  the right metric for "how different are these two poses?" in a
  biomechanics sense.
- **Mahalanobis distance** against a learned pose prior. This is the
  "machine learning" answer — fit a covariance to a reference corpus
  (normal gait from a healthy cohort), then measure distances in the
  whitened space. Requires enough data to fit the prior without
  overfitting, but makes "is this gait abnormal?" a calibrated question.

#### Preprocessing: what invariance do we want?

The v0.1 implementation is not invariant to anything. Two videos of the
same subject with a different camera position will give a different
DTW distance, which is almost certainly not what a clinician wants.
Candidate preprocessing steps:

- **Translation invariance**: subtract the root joint (pelvis or torso
  centroid) from every joint per frame, so all poses are expressed in a
  body-relative coordinate frame. Cheap and almost always desired.
- **Scale invariance**: divide by a reference length (e.g., torso
  length, or total skeleton span) so tall and short subjects produce
  comparable distances. Important for comparing across subjects.
- **Rotation invariance**: align to a canonical frame (e.g., hip-to-hip
  vector = x-axis, hip-to-shoulder = z-axis) per frame. Required if the
  subject's orientation relative to the camera varies between trials.
- **Procrustes alignment per frame**: fit the best rigid transform
  (rotation + translation) between pose A's frame and pose B's frame
  before computing distance. The closed-form
  [Kabsch algorithm](https://en.wikipedia.org/wiki/Kabsch_algorithm) is
  fast and exact. This is likely the *right* thing for most comparison
  use cases but has never been wired up.

The `dtw_relation` helper is translation- and (for unit-vector
displacements) scale-invariant by construction, which is why it ends up
being the most useful of the three existing entry points in practice.

#### Representation: coordinates, angles, velocities, or dual?

The v0.1 DTW operates on **3D joint coordinates** (translation-dependent)
or **joint-pair displacements** (`dtw_relation`). Other representations
worth comparing:

- **Joint angles.** Using `extract_joint_angles` output as the DTW
  input gives a rotation-and-translation-invariant comparison that's
  also directly interpretable in clinical terms.
- **Joint velocities.** Temporal derivatives of position. Emphasizes
  *how the pose changes* rather than *what it is* — good for
  discriminating smooth from jerky motion in gait.
- **Dual (position + angle).** Concatenate normalized position and
  angle features into a single per-frame vector. More expressive but
  requires tuning the relative weights.
- **Learned embeddings.** Feed each frame through a pretrained
  pose-representation network (there are a few) and DTW on the
  embedding space. Expensive and opaque but may capture
  higher-order structure.

#### Multi-scale approaches

FastDTW is already multi-scale internally. Other ideas:

- **Coarse-to-fine DTW.** Downsample aggressively, run exact DTW on
  the coarse version to get a sub-quadratic alignment, then refine
  locally. This is essentially what FastDTW does, but with an explicit
  signal-processing hat on.
- **Wavelet-decomposed DTW.** Decompose each joint's trajectory into
  wavelet coefficients and run DTW on the low-frequency coefficients.
  Unclear whether this actually helps; interesting because it separates
  posture (low-frequency) from tremor / micro-motion (high-frequency).

#### Clinical gait: cycle-aware DTW

Gait is approximately periodic, and "the 4th heel-strike of trial A"
is the clinically meaningful comparison point to "the 4th heel-strike
of trial B", not "frame 120 of A vs frame 120 of B". A natural two-stage
approach:

1. **Cycle detection.** Find heel-strikes (or other gait events) via
   peak detection on a joint's vertical coordinate, and segment each
   trial into individual cycles.
2. **Per-cycle DTW.** Time-warp within each cycle independently to
   normalize cycle duration. The distance between trials is then the
   sum / mean of per-cycle distances.

This is standard in the biomechanics literature
([Sadeghi et al. 2000](https://doi.org/10.1016/S0966-6362(00)00074-3)
and descendants) and is almost certainly a better fit for clinical
comparison than the naive full-trial DTW we ship today.

#### Soft-DTW for learning applications

[Soft-DTW](https://arxiv.org/abs/1703.01541) is a differentiable
relaxation of DTW, which means gradients can flow through it. This
matters if we ever want to train a network to *learn* a distance
metric or an embedding under a DTW objective — for example, a pose
encoder whose output space is calibrated to gait similarity. Worth
keeping on the radar even if we're not training anything today.
`tslearn` implements it.

### Evaluation strategy

Validating a DTW implementation is harder than validating most things.
Some ideas for how to know we got it right:

- **Synthetic perturbations.** Take a reference sequence and apply
  known perturbations (time stretch, added noise, spatial offset) and
  verify that distance scales monotonically with perturbation magnitude
  and that invariance properties are honored.
- **Reference implementation parity.** For a small set of hand-picked
  pairs, compute DTW distance using `dtaidistance` exact DTW and
  our implementation, and verify the approximation error is below a
  documented threshold.
- **Inter-rater clinical benchmark.** When we have labeled clinical
  data, measure how well DTW distance correlates with clinician
  ratings of gait similarity. This is the real test but is gated on
  having data we can use.
- **Pathology discrimination.** Can DTW distance separate healthy
  from impaired gait in a held-out set? This is the usefulness test.

### Open questions

1. Is FastDTW good enough, or should we move to `dtaidistance` exact
   DTW as the default? (First concrete experiment: pick 20 pairs from
   whatever reference data we can source, compute distance both ways,
   see if the approximation error is acceptable.)
2. What's the right representation for clinical gait DTW — raw
   coordinates, joint angles, or per-pair displacements?
3. Should we implement Procrustes alignment as a preprocessing step
   before any DTW call? (If yes, it belongs in `neuropose.analyzer.features`.)
4. Should the clinical pipeline use cycle-segmented DTW instead of
   full-trial DTW? This is a methodological choice with real
   downstream implications.
5. Is soft-DTW useful to us, or is it a solution looking for a
   problem we don't have?
6. What reference corpus do we use to develop and validate any of this?

### Reading list

- Sakoe, H. & Chiba, S. (1978). "Dynamic programming algorithm
  optimization for spoken word recognition." The original DTW paper.
- Salvador, S. & Chan, P. (2007). "Toward accurate dynamic time
  warping in linear time and space."
  [PDF](https://cs.fit.edu/~pkc/papers/tdm04.pdf). The FastDTW paper.
- Cuturi, M. & Blondel, M. (2017). "Soft-DTW: a Differentiable Loss
  Function for Time-Series." [arXiv 1703.01541](https://arxiv.org/abs/1703.01541).
- Sadeghi, H. et al. (2000). "Symmetry and limb dominance in able-bodied
  gait: a review." Biomechanics reference for cycle-aware analysis.
- `dtaidistance` documentation —
  <https://dtaidistance.readthedocs.io/>. Worth reading even if we
  don't switch, for the overview of DTW variants and constraints.

### Next steps

- [ ] Pick 10–20 reference pose-sequence pairs and run both FastDTW and
      exact DTW on them to quantify the approximation error.
- [ ] Prototype a Procrustes-aligned preprocessing wrapper and
      re-run the same pairs.
- [ ] Sketch a cycle-aware DTW pipeline against a gait dataset we can
      actually use (identity- and IRB-safe).
- [ ] Decide whether to keep FastDTW as the default or replace it.
- [ ] If we replace it: migrate `neuropose.analyzer.dtw` to the new
      backend in a single commit with no API change.

---

## TensorFlow version compatibility

### The question

The pinned MeTRAbs model artifact
(`metrabs_eff2l_y4_384px_800k_28ds.tar.gz`) is a TensorFlow SavedModel.
SavedModels embed a producer TF version and depend on a set of TF op
kernels. Picking a TF version pin that is too low risks Apple Silicon
install pain (pre-2.16 has no native `darwin/arm64` wheel under the
`tensorflow` package name); picking one that is too high risks loading
or runtime failures if MeTRAbs uses ops that have been renamed,
deprecated, or removed. The goal of this investigation was to find the
**minimum** pin that works on Linux x86_64, Linux arm64, and macOS arm64
without forcing platform-conditional dependencies or shipping
`tensorflow-metal` as a default.

### Method

Phase 0 of the procedure laid out earlier in this document was to
inspect the SavedModel directly and run `detect_poses` end-to-end on a
synthetic input. The probe script (`test.py` at the repo root, kept
during the investigation and removed in the same commit that landed the
pin) did three things:

1. Parsed `saved_model.pb` with `saved_model_pb2.SavedModel` and read
   the `tensorflow_version` and `tensorflow_git_version` fields out of
   each `meta_info_def` to establish the **producer** version.
2. Walked every `node.op` and `library.function[*].node_def[*].op` in
   the graph to enumerate the **complete set of ops** the model relies
   on. This is the binary-compatibility surface — anything in this set
   that gets removed in a future TF release breaks the model.
3. Called `tf.saved_model.load(MODEL_DIR)`, accessed
   `per_skeleton_joint_names["berkeley_mhad_43"]`, and invoked
   `model.detect_poses(image, intrinsic_matrix=..., skeleton="berkeley_mhad_43")`
   on a 288×384 black frame to confirm the consumer TF version actually
   *runs* the model (not just loads it — these are different failure
   modes).

The probe ran on Linux x86_64 against whatever `uv sync --group dev`
resolved at the time, which was **TensorFlow 2.21.0** with **Keras
3.14.0** — i.e. the most recent TF release as of 2026-04 and a version
that crosses the Keras-3 cutover at TF 2.16.

### Result

- **Producer version:** `tf version: 2.10.0`,
  `producer: v2.10.0-0-g359c3cdfc5f`. The model was serialized in
  September 2022, consistent with the file mtimes in the extracted
  tarball.
- **Custom ops:** **zero**. `tf.raw_ops.__dict__` filtered for
  `"metrabs"` returned `[]`. Every op in the SavedModel is a stock
  TensorFlow kernel that has been stable since at least TF 2.4.
- **Op inventory** (recorded for posterity so a future contributor can
  diff against a newer MeTRAbs release without re-running the probe):

  ```
  Abs, Add, AddV2, All, Any, Assert, AssignVariableOp, AvgPool,
  BatchMatMulV2, BiasAdd, Bitcast, BroadcastArgs, BroadcastTo, Cast,
  Ceil, Cholesky, CombinedNonMaxSuppression, ConcatV2, Const, Conv2D,
  Cos, Cross, Cumsum, DepthwiseConv2dNative, Einsum, EnsureShape, Equal,
  Exp, ExpandDims, Fill, Floor, FloorDiv, FloorMod, FusedBatchNormV3,
  GatherV2, Greater, GreaterEqual, Identity, IdentityN, If,
  ImageProjectiveTransformV3, LeakyRelu, Less, LessEqual, Log,
  LogicalAnd, LogicalNot, LogicalOr, LookupTableExportV2,
  LookupTableFindV2, LookupTableImportV2, MatMul, MatrixDiagV3,
  MatrixInverse, MatrixSolveLs, MatrixTriangularSolve, Max, MaxPool,
  Maximum, Mean, MergeV2Checkpoints, Min, Minimum, Mul,
  MutableDenseHashTableV2, Neg, NoOp, NonMaxSuppressionWithOverlaps,
  NotEqual, Pack, Pad, PadV2, PartitionedCall, Placeholder, Pow, Prod,
  RaggedRange, RaggedTensorFromVariant, RaggedTensorToTensor,
  RaggedTensorToVariant, Range, Rank, ReadVariableOp, RealDiv, Relu,
  Reshape, ResizeArea, ResizeBilinear, RestoreV2, ReverseV2,
  RngReadAndSkip, SaveV2, Select, SelectV2, Shape, ShardedFilename,
  Sigmoid, Sin, Size, Slice, Softplus, Split, SplitV, Sqrt, Square,
  Squeeze, StatefulPartitionedCall, StatelessIf,
  StatelessRandomUniformV2, StatelessWhile, StaticRegexFullMatch,
  StridedSlice, StringJoin, Sub, Sum, Tan, Tanh, TensorListConcatV2,
  TensorListFromTensor, TensorListGetItem, TensorListReserve,
  TensorListSetItem, TensorListStack, Tile, TopKV2, Transpose, Unpack,
  VarHandleOp, Where, While, ZerosLike
  ```

- **Load:** `tf.saved_model.load` returned a `_UserObject` with
  `detect_poses` exposed. No warnings about deprecated kernels, no
  errors. The 11-minor-version forward jump from producer 2.10 to
  consumer 2.21 was a non-event, including the Keras 3 cutover at 2.16.
- **Skeleton check:** `per_skeleton_joint_names["berkeley_mhad_43"]` had
  shape `(43,)` and `per_skeleton_joint_edges["berkeley_mhad_43"]` had
  shape `(42, 2)`, exactly matching what
  `tests/integration/test_estimator_smoke.py` asserts.
- **End-to-end inference:** `model.detect_poses` on a black 288×384
  frame returned `{'poses3d': (0, 43, 3), 'boxes': (0, 5),
  'poses2d': (0, 43, 2)}`, all `float32`. Zero detections is the
  correct output for a black frame — the important signal is that the
  shapes, dtypes, and key names exactly match what `FramePrediction` in
  `neuropose.io` is built to ingest, so the entire estimator pipeline
  is wire-compatible with this TF version.

### Decision

Pin `tensorflow>=2.16,<3.0`. Reasoning:

1. **2.16 is the Apple Silicon floor that matters.** TF 2.16 is the
   first release with native `darwin/arm64` wheels published on PyPI
   under the `tensorflow` package name. Below 2.16, Mac users would
   need `tensorflow-macos` (a separate Apple-maintained package), which
   forces ugly platform markers in `pyproject.toml` and means Linux and
   Mac users run subtly different codebases. Above 2.16, the same
   single dependency line installs cleanly on every supported platform.
2. **MeTRAbs imposes no upper bound below 3.0.** Producer 2.10 → consumer
   2.21 (an 11-minor-version jump across the Keras 3 boundary) loaded
   and ran without a single complaint. The op inventory is 100% stock,
   so future TF 2.x releases would only break this if they removed
   stable kernels — which would itself be a TF 2.x SemVer violation.
3. **`tensorflow-metal` is an opt-in extra, not a default.**
   `tensorflow-metal` is a PluggableDevice that Apple ships separately
   to add a Metal-backed `/GPU:0`. It has its own version-compatibility
   table (Apple maintains it at
   `developer.apple.com/metal/tensorflow-plugin/`), has a documented
   history of producing silently-wrong numerics on specific TF ops,
   and breaks intermittently on Keras 3. For a clinical-research
   pipeline where reproducibility matters more than inference latency,
   CPU inference on Mac is the right default. We do ship a
   `[project.optional-dependencies].metal` extra that pulls
   `tensorflow-metal>=1.2,<2` under darwin/arm64 platform markers, so
   users who want the speedup can opt in via
   `pip install 'neuropose[metal]'` — but the Metal path is not
   exercised in CI, is documented as experimental in
   `docs/getting-started.md`, and users are expected to spot-check
   `poses3d` output against the CPU path before trusting it for any
   clinical measurement.

### What is **not** yet verified

- The probe ran on Linux x86_64 only. macOS arm64 has not been exercised
  on real hardware. The argument that it should work is by construction
  — `tensorflow==2.16+` ships native arm64 macOS wheels, the SavedModel
  uses zero custom ops, and there is no MeTRAbs-side platform code — but
  empirical confirmation is still pending.
- Linux arm64 has likewise not been exercised. Same by-construction
  argument applies.
- A `macos-14` GitHub Actions matrix entry (which would run the unit
  tests on Apple Silicon hardware) is the cheapest way to catch any
  regression and is the intended follow-up.
- Inference-output numerics have not been compared across platforms.
  This is the next layer of rigor below "does it run" — we expect
  fp32 results to match within ~1e-3 mm on `poses3d`, but a real
  cross-platform diff against a reference set has not been done.
- The `[metal]` optional-dependencies extra exists in `pyproject.toml`
  but the Metal code path has never been exercised against the
  pinned MeTRAbs SavedModel. Enabling it is a pure opt-in and comes
  with a documented "verify your own numerics" caveat in
  `docs/getting-started.md`. Whether it actually produces a speedup
  on EfficientNetV2-L-based inference on real clinical videos —
  and whether that speedup is worth the numerical-divergence risk
  — is unknown.

### Open questions

1. Does the same `detect_poses` call produce numerically equivalent
   `poses3d` on macOS arm64 as on Linux x86_64 against a real (non-black)
   reference image? Within what tolerance?
2. If a future MeTRAbs release introduces a custom op (e.g. for a new
   detector head), how do we want the loader to fail? Currently the
   `_REQUIRED_MODEL_ATTRS` interface check would still pass; the failure
   would surface at first `detect_poses` call, which is late.
3. Does it make sense to upper-bound the pin more tightly than `<3.0`
   (e.g. `<2.22` to bound to tested versions), or is the SemVer guard
   sufficient given the all-stock-ops result?

### Next steps

- [ ] Run the same probe on real macOS arm64 hardware and log the
      result (load success, detect_poses success, output numerics
      diff against the Linux baseline).
- [ ] Add a `macos-14` matrix entry to `.github/workflows/ci.yml` for
      the unit tests. Slow tests stay Linux-only to avoid doubling the
      MeTRAbs download cost in CI.
- [ ] Re-run the probe whenever MeTRAbs upstream publishes a new model
      tarball, and diff the op inventory above. Any new op that is not
      in the list above is a flag worth investigating before raising
      the pin.
- [ ] Benchmark `[metal]` vs CPU on a real Apple Silicon Mac against
      a short reference clip: measure (a) per-frame latency, (b) peak
      memory, and (c) `poses3d` divergence from the CPU baseline. If
      the speedup is meaningful and the numerics are within
      ~1e-2 mm, move the `metal` extra from "experimental" to
      "supported" in the docs. If not, document the failure mode
      here and keep the extra where it is.

---

## MeTRAbs hosting and extensibility

### Current state (v0.1, commit 11)

The model loader in `neuropose._model.load_metrabs_model` will pin the
canonical upstream URL:

```
https://omnomnom.vision.rwth-aachen.de/data/metrabs/metrabs_eff2l_y4_384px_800k_28ds.tar.gz
```

This is the RWTH Aachen "omnomnom" host — a raw HTTP file server run
by the MeTRAbs authors' lab. There is no current HuggingFace mirror
of the relevant MeTRAbs variant at the time of commit 11.

The URL encodes the model configuration:
`metrabs_eff2l_y4_384px_800k_28ds` means the EfficientNetV2-L backbone,
YOLOv4 detector head, 384-pixel input, 800k training steps, trained on
28 datasets. This name pattern is worth preserving when we host the
model ourselves so future variants stay self-describing.

### Supply-chain concerns

Pinning a single upstream URL to a third-party academic host is a
real supply-chain risk, and the audit of the previous prototype called
it out explicitly (the old code used `bit.ly/metrabs_1`, which was
even worse). Concrete failure modes:

- The RWTH Aachen host goes down or is decommissioned.
- The URL changes when Sárándi releases a new MeTRAbs version.
- The tarball contents change under the same URL without a version bump.

**Minimum mitigation** (should land in or immediately after commit 11):

- **Pin a SHA-256 checksum** alongside the URL, and verify on download
  before unpacking. If the checksum doesn't match, fail hard with a
  clear error.
- **Cache aggressively.** Once downloaded and verified, never hit the
  network again for the same configuration. `model_cache_dir` is
  already in `Settings`.
- **Document the exact filename and checksum** in `RESEARCH.md` (or
  migrate to a `MODEL_ARTIFACTS.md` file) so operators have a way to
  manually download the model out-of-band if the primary URL is dead.

### Self-hosting options

We want to host the model ourselves, both for reliability and because
it opens the door to future fine-tuning and redistribution of our own
variants. Candidate hosting approaches:

#### Forgejo LFS

Pros:
- Lives next to the code.
- Version-controlled artifacts.
- Access control mirrors repo access.

Cons:
- LFS is designed for git-tracked binary assets, not for large
  infrequently-updated model weights — you pay LFS overhead on every
  clone unless you configure `lfs.fetchexclude`.
- Model is ~2.2 GB; Forgejo LFS performance at that size is untested
  for our instance.
- Pinning is by LFS pointer, which means the model is coupled to a
  particular repo revision. Messy if we want multiple code revisions
  to share the same model.

**Verdict:** Workable but not the best fit.

#### Forgejo generic package registry

Forgejo supports a [generic package
registry](https://forgejo.org/docs/latest/user/packages/generic/) that
can host arbitrary binary artifacts with versioned URLs. This is
closer to what we want:

```
https://git.levineuwirth.org/api/packages/neuwirth/generic/metrabs/eff2l_y4_384px_800k_28ds/metrabs.tar.gz
```

Pros:
- Versioned URLs decoupled from repo revisions.
- Upload once, download many times, no clone coupling.
- Integrated auth if we want to gate access.
- Can be made public even if the repo is private.

Cons:
- Requires uploading the file manually or via an API call.
- Forgejo registry size / bandwidth limits depend on the instance.

**Verdict:** Probably the best fit for "we want it hosted alongside
the project."

#### Plain HTTP server on a VPS subdomain

A dedicated subdomain like `models.levineuwirth.org` backed by a
simple HTTP file server (nginx `autoindex`, or Caddy with a tidy
directory layout). Example URL:

```
https://models.levineuwirth.org/metrabs/metrabs_eff2l_y4_384px_800k_28ds.tar.gz
```

Pros:
- Simplest possible story. No API, no auth machinery.
- Easy to mirror from — anyone can curl the URL.
- Decoupled from the git forge, so we can share models publicly even
  when the repo itself is private.
- Easy to put a CDN in front (Cloudflare) if bandwidth ever matters.

Cons:
- Manual upload via scp/rsync.
- No access control unless we add it.
- No versioning beyond filename convention.

**Verdict:** Strong candidate. This is probably the right choice for
v0.1 of self-hosted models.

#### S3-compatible object storage (MinIO self-hosted)

Run MinIO on the VPS, get S3-compatible API for free, and serve models
via pre-signed URLs or a public bucket.

Pros:
- Proper object storage with ETags, range requests, multipart uploads.
- Integration story is straightforward if we ever move to cloud-hosted
  storage.
- Industry-standard API.

Cons:
- More operational complexity than a plain HTTP server for what might
  be a handful of files.

**Verdict:** Overkill for v0.1 but worth revisiting if model storage
becomes a real operational concern.

### Integrity: SHA-256 pinning

Regardless of which hosting approach we pick, **the model loader should
always verify a SHA-256 checksum** before trusting the downloaded
artifact. This is the one piece of supply-chain hygiene that has to be
in place before we ship commit 11 to any user outside the Shu lab.

Implementation sketch for `neuropose/_model.py`:

```python
def load_metrabs_model(cache_dir: Path | None = None) -> Any:
    cache_dir = cache_dir or _default_model_cache_dir()
    cache_dir.mkdir(parents=True, exist_ok=True)
    tarball = cache_dir / _MODEL_FILENAME
    if not tarball.exists():
        _download(_MODEL_URL, tarball)
    _verify_sha256(tarball, _MODEL_SHA256)
    extracted = _extract_if_needed(tarball, cache_dir)
    return tfhub.load(str(extracted))  # or tf.saved_model.load
```

The `_MODEL_SHA256` constant is the source of truth; if it ever has
to change, the constant change is visible in the git diff and a human
reviews it.

### Fine-tuning

The next research direction after we have inference working is
fine-tuning MeTRAbs on clinical-specific data. Open questions:

- **What data?** Any clinical data is IRB-gated. Even de-identified
  pose data may carry subject information if the recording conditions
  (lighting, room layout) are distinctive enough. Any training plan
  has to run through the data-handling policy that lives (will live)
  in `docs/data-policy.md`.
- **Transfer learning strategy.**
  - *Head-only fine-tuning*: freeze the EfficientNetV2-L backbone and
    re-train the pose regression head on clinical data. Fast, low
    compute, unlikely to overfit, but also unlikely to capture
    clinical-pose idiosyncrasies.
  - *Low-LR full fine-tune*: unfreeze everything, use a learning rate
    1/100th of the original, train for a few epochs. Better
    adaptation, higher risk of catastrophic forgetting.
  - *Adapter layers*: insert small trainable adapters into the frozen
    backbone. Parameter-efficient, well-studied in NLP, less common
    for pose but should work.
- **Compute requirements.** EfficientNetV2-L is roughly 120M parameters;
  fine-tuning on a single modern GPU (24 GB VRAM) is feasible at
  reduced batch size. A multi-GPU node is friendlier but not strictly
  required.
- **Evaluation.** We need held-out clinical data with trusted ground
  truth. MoCap-derived poses are the gold standard; marker-based MoCap
  systems provide sub-millimeter accuracy at the cost of subject
  instrumentation. The Shu lab's access to MoCap is the gating factor.
- **Sharing fine-tuned weights.** If we fine-tune on clinical data, the
  resulting weights may encode subject information in ways that are
  non-obvious and potentially IRB-relevant. Sharing fine-tuned weights
  externally has to be cleared through the same channels as sharing the
  training data.

### Training our own pose estimator

The long-range version of the research direction: train a pose
estimator from scratch that extends MeTRAbs's methodology. MeTRAbs is
a good starting point because the method is well-documented:

- Sárándi, I., et al. (2020). "MeTRAbs: Metric-Scale Truncation-Robust
  Heatmaps for Absolute 3D Human Pose Estimation."
  [arXiv 2007.07227](https://arxiv.org/abs/2007.07227),
  IEEE Transactions on Biometrics, Behavior, and Identity Science.

Core contributions (worth knowing if you modify any of this):

- **Truncation-robust heatmaps.** Instead of predicting a 2D heatmap
  bounded by the image, MeTRAbs predicts a heatmap that extends
  *outside* the image and can place a joint at coordinates the image
  alone could not disambiguate. Critical for crops where the subject
  is partially out of frame.
- **Metric scale regression.** MeTRAbs predicts the absolute 3D
  positions of joints in millimetres by combining a 2D heatmap with a
  per-joint depth regressor. Most 3D pose methods produce only
  relative coordinates, which are useless for clinical measurement.
- **Multi-dataset training with a common skeleton.** The 28-dataset
  training set unifies disparate skeleton topologies into a common
  43-joint Berkeley MHAD skeleton, which we carry forward in
  NeuroPose.

**Natural extensions worth considering:**

- **Temporal smoothing head.** MeTRAbs is a per-frame model. Clinical
  gait analysis wants temporally smooth trajectories. Adding a
  lightweight temporal head (1D CNN or small transformer over frame
  sequences) could produce smoother outputs without touching the
  backbone.
- **Clinical-specific heatmap supervision.** If we have MoCap data for
  clinical poses, we can use it as ground-truth heatmap supervision to
  improve accuracy in the pose ranges the model sees least often in
  the 28-dataset training corpus (e.g., pathological gaits, walker-
  assisted ambulation).
- **Multi-person identity tracking.** MeTRAbs produces detections per
  frame without continuity across frames. Adding a Hungarian-matched
  tracker (or a learned tracker) would solve the multi-person
  identity problem that `predictions_to_numpy` currently dodges with
  a `person_index` parameter.
- **Alternative backbones.** EfficientNetV2-L is a 2020-era choice.
  Newer backbones (ConvNeXt, DINOv2-initialized ViTs) may give
  meaningful gains, especially for clinical poses that are
  under-represented in the original training set.
- **Uncertainty estimation.** Clinical users want to know when the
  model is unsure. A Gaussian output head (mean + variance per joint)
  or an ensemble-based approach would let us propagate uncertainty
  into downstream analysis.

**Compute requirements:** training MeTRAbs from scratch was reported
as "a few weeks" on 8x V100 in the original paper. A from-scratch
re-training is a substantial undertaking. Fine-tuning is much more
accessible.

### Collaboration opportunities

- **István Sárándi** (now at University of Tübingen, formerly RWTH
  Aachen) is the author of MeTRAbs. The code is MIT-licensed and he
  has historically been responsive to collaboration requests. If we
  end up publishing work that significantly extends MeTRAbs, at the
  very least we should reach out about co-authorship or
  acknowledgment; at best we might find an active collaborator.
- **The Shu Lab's existing collaborators** on clinical gait research
  at Brown and partner institutions may have MoCap-validated datasets
  we can use for fine-tuning and evaluation. Worth asking Dr. Shu.

### Open questions

1. Does Forgejo's generic package registry actually handle a 2.2 GB
   upload cleanly, or do we need the plain HTTP server route?
2. What's the right SHA-256 pin to commit alongside the URL? (Need to
   download the tarball first and run `sha256sum`.)
3. Do we have access to MoCap-validated clinical gait data for
   fine-tuning evaluation? This gates every training-related
   experiment.
4. Is fine-tuning even worth pursuing before we have inference results
   that are clearly *not* good enough on clinical data? (I.e.,
   motivate the work with concrete failure cases rather than assuming
   a delta we haven't measured.)
5. Does it make sense to reach out to Sárándi now, or wait until we
   have something concrete to collaborate on?

### Reading list

- Sárándi, I. et al. (2020). "MeTRAbs: Metric-Scale Truncation-Robust
  Heatmaps for Absolute 3D Human Pose Estimation."
  [arXiv 2007.07227](https://arxiv.org/abs/2007.07227). **Essential
  reading** for anyone planning to extend the method.
- Sárándi's personal site and the MeTRAbs GitHub repo
  (<https://github.com/isarandi/metrabs>) — the code, model zoo, and
  training scripts live here.
- Zheng, C. et al. (2023). "Deep Learning-Based Human Pose Estimation: A
  Survey." Good survey paper for orienting on the state of the art.
- The original 28-dataset training composition referenced in the
  MeTRAbs paper — worth tracing through to understand what poses are
  in- and out-of-distribution for the pretrained model.

### Next steps

- [ ] Download the pinned tarball and compute its SHA-256 for the
      commit-11 model loader.
- [ ] Decide between Forgejo generic registry and plain HTTP subdomain
      for self-hosting. Prototype whichever one wins.
- [ ] Mirror the pinned tarball to the chosen self-hosted location so
      we can fall over to it the moment the RWTH URL changes or goes
      down.
- [ ] Write a one-page "MODEL_ARTIFACTS.md" that documents every model
      version we use, its checksum, and its canonical source URL.
- [ ] Have the data-access conversation with Dr. Shu about clinical
      training data. Everything else is blocked on this.
- [ ] (Much later) Reach out to Sárándi about potential collaboration
      once we have something concrete to talk about.
