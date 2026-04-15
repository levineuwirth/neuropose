"""I/O helpers and schema definitions for NeuroPose prediction data.

Defines pydantic models for per-frame predictions, per-video predictions
(with metadata envelope), job-level aggregated results, and the persistent
status file. All models are validated on load, so malformed files are caught
at the boundary rather than at some downstream call site.

Atomicity: :func:`save_status`, :func:`save_job_results`, and
:func:`save_video_predictions` write to a sibling temp file and then
atomically rename, so a crash mid-write will not leave a partially-written
file behind. This matches the crash-resilience guarantee the interfacer
daemon makes to callers.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, RootModel, model_validator


class JobStatus(StrEnum):
    """Lifecycle state of a single processing job."""

    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class FramePrediction(BaseModel):
    """Pose estimation output for a single video frame.

    Each inner list corresponds to one detected person. Coordinate units
    follow MeTRAbs conventions: ``poses3d`` in millimetres, ``poses2d`` in
    pixels, ``boxes`` as ``[x, y, width, height, confidence]`` in pixels.

    Frozen (immutable) to prevent the in-place coordinate-swap aliasing bug
    that affected the previous prototype's visualizer.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    boxes: list[list[float]] = Field(
        description="Per-detection bounding boxes as [x, y, width, height, confidence]."
    )
    poses3d: list[list[list[float]]] = Field(
        description="Per-detection 3D joint positions in millimetres."
    )
    poses2d: list[list[list[float]]] = Field(
        description="Per-detection 2D joint positions in pixels."
    )


class VideoMetadata(BaseModel):
    """Metadata about the source video for a set of predictions.

    Essential for reproducibility: the frame count lets downstream analysis
    verify completeness, and the fps lets it convert frame indices to real
    time without needing access to the original video file.

    Intentionally does NOT include the source file path or filename, which
    may encode subject-identifying information. Callers that need provenance
    should store it out-of-band in accordance with the data-handling policy.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    frame_count: int = Field(ge=0, description="Number of frames actually processed.")
    fps: float = Field(ge=0.0, description="Source video frame rate (frames per second).")
    width: int = Field(ge=0, description="Source video frame width in pixels.")
    height: int = Field(ge=0, description="Source video frame height in pixels.")


class PerformanceMetrics(BaseModel):
    """Timing and resource-usage metrics collected during inference.

    Populated by :meth:`neuropose.estimator.Estimator.process_video` on
    every call so real-world runs always carry their own timing without
    the caller having to opt in. The values describe the estimator's
    behaviour on one specific pass over a single video and are *not*
    aggregates — callers interested in distributional statistics
    (p50/p95/p99) should pass the same video through
    :func:`neuropose.benchmark.run_benchmark`, which runs multiple
    passes and aggregates the resulting :class:`PerformanceMetrics`
    instances.

    Fields intentionally capture "what machine ran this":
    ``active_device`` reports the TensorFlow device string the model
    ran on (``/GPU:0`` on Apple Silicon with ``tensorflow-metal``,
    ``/CPU:0`` otherwise), and ``tensorflow_version`` captures the
    exact TF release so downstream readers can tell whether a
    numerical discrepancy between two runs is a stack-version
    difference rather than a real measurement.

    Frozen so that the metrics captured at return time cannot be
    edited in-place downstream.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    model_load_seconds: float | None = Field(
        default=None,
        ge=0.0,
        description=(
            "Wall-clock time to load the MeTRAbs model, in seconds. "
            "``None`` when the model was injected via ``Estimator(model=...)`` "
            "rather than loaded through ``load_model()``, so the value is "
            "never confused with a zero-cost cache hit."
        ),
    )
    total_seconds: float = Field(
        ge=0.0,
        description="Wall-clock time of ``process_video`` end-to-end.",
    )
    per_frame_latencies_ms: list[float] = Field(
        default_factory=list,
        description=(
            "Per-frame inference latencies in milliseconds, one entry per "
            "decoded frame in insertion order. Excludes video-decode time "
            "and captures only the ``detect_poses`` call, so callers "
            "aggregating over this list are measuring model throughput."
        ),
    )
    peak_rss_mb: float = Field(
        ge=0.0,
        description=(
            "Maximum resident-set size observed during the call, in "
            "megabytes. Sampled once after each frame via ``psutil``."
        ),
    )
    active_device: str = Field(
        description=(
            "TensorFlow device string the inference ran on, e.g. "
            "``/CPU:0`` or ``/GPU:0``. Derived from "
            "``tf.config.list_physical_devices('GPU')`` — when a GPU "
            "device is visible the string is ``/GPU:0``, otherwise "
            "``/CPU:0``."
        ),
    )
    tensorflow_metal_active: bool = Field(
        default=False,
        description=(
            "``True`` if the ``tensorflow-metal`` PluggableDevice is "
            "installed and contributed the visible GPU device, i.e. "
            "MeTRAbs inference is running through Apple's Metal "
            "Performance Shaders backend on Apple Silicon. ``False`` "
            "otherwise, including on Linux CUDA builds."
        ),
    )
    tensorflow_version: str = Field(
        description="Value of ``tensorflow.__version__`` at the time of the call.",
    )


class BenchmarkAggregate(BaseModel):
    """Distributional statistics aggregated across benchmark passes.

    Computed by :mod:`neuropose.benchmark` over the measured
    :class:`PerformanceMetrics` instances of a multi-pass benchmark run
    — the very first pass is always discarded as warmup (graph
    compilation, file-system caches, etc.), and the first
    ``warmup_frames_per_pass`` frames of each remaining pass are also
    excluded so graph-init cost inside a pass does not contaminate the
    steady-state percentiles.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    repeats_measured: int = Field(
        ge=0,
        description="Number of passes that contributed to the aggregate.",
    )
    warmup_frames_per_pass: int = Field(
        ge=0,
        description="Frames discarded from the head of each measured pass.",
    )
    mean_frame_latency_ms: float = Field(ge=0.0)
    p50_frame_latency_ms: float = Field(ge=0.0)
    p95_frame_latency_ms: float = Field(ge=0.0)
    p99_frame_latency_ms: float = Field(ge=0.0)
    stddev_frame_latency_ms: float = Field(ge=0.0)
    mean_throughput_fps: float = Field(ge=0.0)
    peak_rss_mb_max: float = Field(
        ge=0.0,
        description="Maximum peak RSS observed across all measured passes.",
    )
    active_device: str
    tensorflow_metal_active: bool = False
    tensorflow_version: str


class CpuComparisonResult(BaseModel):
    """Result of a ``--compare-cpu`` benchmark run.

    The parent benchmark process runs on the platform's default device
    (GPU if visible), and a subprocess is spawned with ``--force-cpu``
    to run the exact same video on CPU. Both runs' aggregates are
    preserved here alongside the maximum element-wise divergence in the
    resulting ``poses3d`` arrays — that number is the "is Metal
    numerically OK?" answer that ``RESEARCH.md`` §TensorFlow version
    compatibility is asking.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    primary_aggregate: BenchmarkAggregate = Field(
        description="Aggregate from the parent (default-device) run.",
    )
    cpu_aggregate: BenchmarkAggregate = Field(
        description="Aggregate from the ``--force-cpu`` subprocess run.",
    )
    speedup: float = Field(
        description=(
            "``primary_aggregate.mean_throughput_fps / "
            "cpu_aggregate.mean_throughput_fps``. Values greater than 1 "
            "mean the primary device is faster than CPU; values less "
            "than 1 mean CPU won (possible on tiny videos where device "
            "initialisation dominates)."
        ),
    )
    max_poses3d_divergence_mm: float = Field(
        ge=0.0,
        description=(
            "Maximum element-wise absolute difference (in millimetres) "
            "between the primary-device and CPU-device ``poses3d`` "
            "arrays, taken over every frame, detection, joint, and "
            "axis. A small number (~1e-3 mm) is the expected floor; "
            "anything above ~1e-2 mm warrants investigation before "
            "trusting the primary device for clinical measurement."
        ),
    )
    frame_count_compared: int = Field(
        ge=0,
        description="Number of frames that entered the divergence computation.",
    )


class BenchmarkResult(BaseModel):
    """Top-level result of :func:`neuropose.benchmark.run_benchmark`.

    Carries the raw per-pass metrics, the aggregated summary, and (if
    ``--compare-cpu`` was requested) a :class:`CpuComparisonResult`.
    Serialised to JSON for downstream regression tracking; pretty-printed
    to stdout by the :mod:`neuropose.cli.benchmark` subcommand.

    The ``video_name`` field intentionally stores only the file's
    basename, not the full path, to match ``VideoMetadata``'s stance on
    not persisting potentially subject-identifying filesystem paths.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    video_name: str = Field(
        description="Basename of the benchmarked video (no directory components).",
    )
    repeats: int = Field(
        ge=1,
        description="Total passes executed — includes the discarded warmup pass.",
    )
    warmup_frames: int = Field(
        ge=0,
        description=(
            "Frames excluded from the head of each measured pass when "
            "computing the aggregate statistics."
        ),
    )
    warmup_pass: PerformanceMetrics = Field(
        description=(
            "First pass, discarded from the aggregate. Preserved here so "
            "readers can see the graph-compilation cost explicitly."
        ),
    )
    measured_passes: list[PerformanceMetrics] = Field(
        description="Passes 2..N, i.e. those that contributed to the aggregate.",
    )
    aggregate: BenchmarkAggregate
    cpu_comparison: CpuComparisonResult | None = None


class JointAxisExtractor(BaseModel):
    """Segmentation signal extracted from one axis of one joint.

    The simplest extractor: picks a single spatial coordinate of a single
    joint as the segmentation signal. Use ``invert=True`` when the motion
    of interest is a decrease (e.g. a joint that dips down during the
    repetition) so that peak detection can work on a maximum rather than
    a minimum.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["joint_axis"] = "joint_axis"
    joint: int = Field(ge=0, description="Joint index into the (J,) skeleton.")
    axis: int = Field(ge=0, le=2, description="Spatial axis: 0=x, 1=y, 2=z.")
    invert: bool = Field(default=False, description="Negate the signal so valleys become peaks.")


class JointPairDistanceExtractor(BaseModel):
    """Signal = Euclidean distance between two joints per frame.

    Translation- and (partially) rotation-invariant. Good fit for tasks
    where the target motion is a distance change between two body parts
    (e.g. wrist-to-shoulder for a reach).
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["joint_pair_distance"] = "joint_pair_distance"
    joints: tuple[int, int] = Field(description="Ordered pair of joint indices.")

    @model_validator(mode="after")
    def _joints_distinct(self) -> JointPairDistanceExtractor:
        if self.joints[0] == self.joints[1]:
            raise ValueError("joint_pair_distance requires two distinct joints")
        if self.joints[0] < 0 or self.joints[1] < 0:
            raise ValueError("joint indices must be non-negative")
        return self


class JointSpeedExtractor(BaseModel):
    """Signal = frame-to-frame displacement magnitude of one joint.

    The first frame is padded with a 0 so the signal has the same length
    as the input pose sequence. Useful when a repetition is characterised
    by "the joint moves fast, then rests, then moves fast again" rather
    than by any particular coordinate value.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["joint_speed"] = "joint_speed"
    joint: int = Field(ge=0)


class JointAngleExtractor(BaseModel):
    """Signal = angle at joint ``b`` formed by the triplet ``(a, b, c)``.

    Computed in radians in ``[0, pi]`` using
    :func:`neuropose.analyzer.features.extract_joint_angles`. This is the
    most translation- and rotation-invariant of the built-in extractors
    and the natural choice for clinically meaningful metrics like knee
    or elbow flexion.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["joint_angle"] = "joint_angle"
    triplet: tuple[int, int, int] = Field(
        description="``(a, b, c)`` joint indices; angle is computed at ``b``."
    )

    @model_validator(mode="after")
    def _triplet_non_negative(self) -> JointAngleExtractor:
        if any(idx < 0 for idx in self.triplet):
            raise ValueError("joint indices must be non-negative")
        return self


ExtractorSpec = Annotated[
    JointAxisExtractor | JointPairDistanceExtractor | JointSpeedExtractor | JointAngleExtractor,
    Field(discriminator="kind"),
]


class SegmentationConfig(BaseModel):
    """Parameters that define a single segmentation pass.

    The full config is serialized alongside the segments it produced so
    that a reader of ``results.json`` can tell — a year later, without
    access to the code that wrote it — exactly which joint was tracked
    and which thresholds were applied. The ``method`` field is a version
    stamp: if the segmentation algorithm changes, add a new literal here
    and dispatch on it in the engine rather than silently reinterpreting
    old files.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    extractor: ExtractorSpec
    person_index: int = Field(
        default=0,
        ge=0,
        description="Which detected person to extract from each frame.",
    )
    min_distance_seconds: float | None = Field(
        default=None,
        ge=0.0,
        description="Minimum time between successive repetition peaks.",
    )
    min_prominence: float | None = Field(
        default=None,
        description="Minimum scipy-style peak prominence on the raw signal.",
    )
    min_height: float | None = Field(
        default=None,
        description="Minimum signal value for a point to be considered a peak.",
    )
    pad_seconds: float = Field(
        default=0.0,
        ge=0.0,
        description="Amount of time to extend each segment on both sides.",
    )
    method: Literal["valley_to_valley_v1"] = Field(
        default="valley_to_valley_v1",
        description="Name and version of the segmentation algorithm used.",
    )


class Segment(BaseModel):
    """A single repetition window inside a pose sequence.

    Frame indices are integers into the ``VideoPredictions.frames``
    insertion order. ``start`` is inclusive, ``end`` is exclusive (Python
    slice convention), and ``peak`` is the apex of the repetition inside
    ``[start, end)``.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    start: int = Field(ge=0, description="Inclusive start frame index.")
    end: int = Field(gt=0, description="Exclusive end frame index.")
    peak: int = Field(ge=0, description="Frame index of the repetition's apex.")

    @model_validator(mode="after")
    def _check_ordering(self) -> Segment:
        if self.end <= self.start:
            raise ValueError(f"end ({self.end}) must be > start ({self.start})")
        if not (self.start <= self.peak < self.end):
            raise ValueError(
                f"peak ({self.peak}) must be in [start, end) = [{self.start}, {self.end})"
            )
        return self


class Segmentation(BaseModel):
    """A labelled segmentation of a single video.

    Pairs the :class:`SegmentationConfig` that produced the segments with
    the segments themselves, so the serialized form is self-describing.
    Multiple named :class:`Segmentation` objects can coexist on a single
    :class:`VideoPredictions` via its ``segmentations`` field — typically
    one per clinical task or per operator-chosen strategy.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    config: SegmentationConfig
    segments: list[Segment]


class VideoPredictions(BaseModel):
    """Per-frame predictions for a single video, paired with video metadata.

    The ``frames`` mapping is keyed by frame identifier (``frame_<index>`` by
    convention, zero-padded to 6 digits). The identifier is a stable string,
    not a filesystem path — no PNG file is implied.

    The optional ``segmentations`` mapping carries one or more post-hoc
    :class:`Segmentation` objects keyed by operator-chosen name (e.g.
    ``"cup_lift"``). Downstream analysis code that expects rep-level
    windows looks here. The field defaults to empty, so inference output
    written before segmentation round-trips through this schema unchanged.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    metadata: VideoMetadata
    frames: dict[str, FramePrediction]
    segmentations: dict[str, Segmentation] = Field(default_factory=dict)

    def frame_names(self) -> list[str]:
        """Return frame identifiers in insertion order."""
        return list(self.frames.keys())

    def __len__(self) -> int:
        """Return the number of frames."""
        return len(self.frames)

    def __iter__(self) -> Iterator[str]:  # type: ignore[override]
        """Iterate over frame identifiers in insertion order."""
        return iter(self.frames)

    def __getitem__(self, key: str) -> FramePrediction:
        """Return the :class:`FramePrediction` for ``key``."""
        return self.frames[key]


class JobResults(RootModel[dict[str, VideoPredictions]]):
    """Aggregated predictions for an entire job, keyed by video filename.

    This is the shape of the top-level ``results.json`` written by the
    interfacer daemon: one entry per video in the job directory.
    """

    def videos(self) -> list[str]:
        """Return video names in insertion order."""
        return list(self.root.keys())

    def __len__(self) -> int:
        """Return the number of videos in the job."""
        return len(self.root)

    def __iter__(self) -> Iterator[str]:  # type: ignore[override]
        """Iterate over video names in insertion order."""
        return iter(self.root)

    def __getitem__(self, key: str) -> VideoPredictions:
        """Return the :class:`VideoPredictions` for ``key``."""
        return self.root[key]


class JobStatusEntry(BaseModel):
    """Status entry for a single job in the persistent status file."""

    model_config = ConfigDict(extra="forbid")

    status: JobStatus
    started_at: datetime
    completed_at: datetime | None = None
    results_path: Path | None = None
    error: str | None = Field(
        default=None,
        description=(
            "Short human-readable reason if status == failed. "
            "Populated by the interfacer on failure paths."
        ),
    )


class StatusFile(RootModel[dict[str, JobStatusEntry]]):
    """Mapping of job name to its status entry."""

    def is_empty(self) -> bool:
        """Return ``True`` if the status file contains no entries."""
        return len(self.root) == 0

    def __len__(self) -> int:
        """Return the number of job entries."""
        return len(self.root)

    def __iter__(self) -> Iterator[str]:  # type: ignore[override]
        """Iterate over job names in insertion order."""
        return iter(self.root)


# ---------------------------------------------------------------------------
# Load / save helpers
# ---------------------------------------------------------------------------


def load_video_predictions(path: Path) -> VideoPredictions:
    """Load and validate a per-video predictions JSON file."""
    with path.open("r", encoding="utf-8") as f:
        data: Any = json.load(f)
    return VideoPredictions.model_validate(data)


def save_video_predictions(path: Path, predictions: VideoPredictions) -> None:
    """Serialize per-video predictions to a JSON file atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    _write_json_atomic(path, predictions.model_dump(mode="json"))


def load_job_results(path: Path) -> JobResults:
    """Load and validate an aggregated per-job results JSON file."""
    with path.open("r", encoding="utf-8") as f:
        data: Any = json.load(f)
    return JobResults.model_validate(data)


def save_job_results(path: Path, results: JobResults) -> None:
    """Serialize aggregated job results to a JSON file atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    _write_json_atomic(path, results.model_dump(mode="json"))


def load_benchmark_result(path: Path) -> BenchmarkResult:
    """Load and validate a benchmark-result JSON file."""
    with path.open("r", encoding="utf-8") as f:
        data: Any = json.load(f)
    return BenchmarkResult.model_validate(data)


def save_benchmark_result(path: Path, result: BenchmarkResult) -> None:
    """Serialize a :class:`BenchmarkResult` to a JSON file atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    _write_json_atomic(path, result.model_dump(mode="json"))


def load_status(path: Path) -> StatusFile:
    """Load the persistent job status file.

    Returns an empty :class:`StatusFile` if the file is missing, is not valid
    JSON, or does not contain a JSON mapping. This preserves the
    crash-resilient behaviour the daemon relies on: a missing or corrupted
    status file is treated as a clean slate rather than a fatal error.
    """
    if not path.exists():
        return StatusFile(root={})
    try:
        with path.open("r", encoding="utf-8") as f:
            data: Any = json.load(f)
    except json.JSONDecodeError:
        return StatusFile(root={})
    if not isinstance(data, dict):
        return StatusFile(root={})
    return StatusFile.model_validate(data)


def save_status(path: Path, status: StatusFile) -> None:
    """Persist the job status file atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    _write_json_atomic(path, status.model_dump(mode="json"))


def _write_json_atomic(path: Path, payload: Any) -> None:
    """Write ``payload`` to ``path`` as JSON, atomically.

    Writes to a sibling ``<path>.tmp`` first, then atomically renames over
    ``path`` so a crash mid-write cannot leave behind a truncated file.
    """
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    tmp.replace(path)
