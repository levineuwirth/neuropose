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
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, RootModel


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


class VideoPredictions(BaseModel):
    """Per-frame predictions for a single video, paired with video metadata.

    The ``frames`` mapping is keyed by frame identifier (``frame_<index>`` by
    convention, zero-padded to 6 digits). The identifier is a stable string,
    not a filesystem path — no PNG file is implied.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    metadata: VideoMetadata
    frames: dict[str, FramePrediction]

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
