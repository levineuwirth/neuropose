"""3D human pose estimator — MeTRAbs wrapper.

The :class:`Estimator` class is the core of NeuroPose's inference path. It
takes a video file, runs the MeTRAbs 3D pose-estimation model on each frame,
and returns a validated :class:`~neuropose.io.VideoPredictions` object with
the per-frame predictions and video metadata.

Design
------
The estimator is a **library**, not a daemon: it knows nothing about job
directories, status files, or polling. Those concerns live in
:mod:`neuropose.interfacer`. An ``Estimator`` can be constructed directly
from a Python script and called on a single video, which is the path taken
by the documentation's quick-start.

Frames are streamed from the source video in memory — the previous
prototype wrote every frame to disk as a PNG and then re-read each one with
``tf.io.decode_png``. That round-trip is gone; frames are read once and
passed to the model directly.

Model injection
---------------
The MeTRAbs model is supplied either:

1. Through :meth:`Estimator.load_model`, which delegates to
   :func:`neuropose._model.load_metrabs_model` (stubbed pending commit 11).
2. Directly via ``Estimator(model=...)``, which is the path used by the
   test suite with a fake model to exercise the code without TensorFlow.

Either way, attempting to call :meth:`Estimator.process_video` before a
model is present raises :class:`ModelNotLoadedError`.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import psutil

from neuropose._model import load_metrabs_model
from neuropose.io import (
    FramePrediction,
    PerformanceMetrics,
    VideoMetadata,
    VideoPredictions,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class EstimatorError(Exception):
    """Base class for errors raised by :class:`Estimator`."""


class ModelNotLoadedError(EstimatorError):
    """Raised when an inference method is called before the model is loaded."""


class VideoDecodeError(EstimatorError):
    """Raised when a video file cannot be opened or decoded."""


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ProcessVideoResult:
    """Result of :meth:`Estimator.process_video`.

    Attributes
    ----------
    predictions
        The validated :class:`VideoPredictions` object, containing both the
        per-frame predictions and the ``VideoMetadata`` envelope.
    metrics
        Timing and resource-usage metrics for the call. Always populated
        so real-world runs carry their own measurements without the
        caller having to opt in. ``metrics.model_load_seconds`` is
        ``None`` when the model was injected rather than loaded via
        :meth:`Estimator.load_model`.
    """

    predictions: VideoPredictions
    metrics: PerformanceMetrics = field(
        default_factory=lambda: PerformanceMetrics(
            total_seconds=0.0,
            peak_rss_mb=0.0,
            active_device="/CPU:0",
            tensorflow_version="unknown",
        )
    )

    @property
    def frame_count(self) -> int:
        """Convenience accessor for ``predictions.metadata.frame_count``."""
        return self.predictions.metadata.frame_count


# ---------------------------------------------------------------------------
# Estimator
# ---------------------------------------------------------------------------


ProgressCallback = Callable[[int, int], None]
"""Type alias for progress callbacks: ``(frames_processed, frame_count_hint)``."""


class Estimator:
    """3D pose estimator built on MeTRAbs.

    Parameters
    ----------
    device
        TensorFlow device string (e.g. ``"/CPU:0"`` or ``"/GPU:0"``). Passed
        through to the model at inference time.
    skeleton
        Skeleton identifier understood by MeTRAbs. Defaults to
        ``"berkeley_mhad_43"``, the 43-joint skeleton used by the previous
        prototype.
    default_fov_degrees
        Horizontal field of view assumed when a video does not supply
        intrinsics. Overridable per call via
        :meth:`process_video`'s ``fov_degrees`` argument.
    model
        Optional pre-loaded MeTRAbs model. If supplied, the estimator uses
        it directly and :meth:`load_model` need not be called. This path is
        used by tests to inject a fake model, and by callers that want to
        share a single model across many :class:`Estimator` instances.
    """

    def __init__(
        self,
        *,
        device: str = "/CPU:0",
        skeleton: str = "berkeley_mhad_43",
        default_fov_degrees: float = 55.0,
        model: Any | None = None,
    ) -> None:
        self.device = device
        self.skeleton = skeleton
        self.default_fov_degrees = default_fov_degrees
        self._model: Any | None = model
        # ``None`` when the model was injected via the constructor (tests,
        # shared-model callers) — we never fake a zero load time. Set on
        # successful ``load_model`` below so the next ``process_video`` can
        # pass the real number through into ``PerformanceMetrics``.
        self._model_load_seconds: float | None = None

    # -- model lifecycle ----------------------------------------------------

    @property
    def model(self) -> Any:
        """Return the loaded model, or raise :class:`ModelNotLoadedError`."""
        if self._model is None:
            raise ModelNotLoadedError(
                "Estimator model has not been loaded. "
                "Call Estimator.load_model() or pass model=... to the constructor."
            )
        return self._model

    @property
    def is_model_loaded(self) -> bool:
        """Return ``True`` if a model has been supplied or loaded."""
        return self._model is not None

    def load_model(self, cache_dir: Path | None = None) -> None:
        """Load the MeTRAbs model via :func:`neuropose._model.load_metrabs_model`.

        Parameters
        ----------
        cache_dir
            Directory where the downloaded model should be cached. Typically
            ``Settings.model_cache_dir``.

        Notes
        -----
        This is idempotent: calling it again after a successful load is a
        no-op. Callers that want to reload the model should construct a new
        :class:`Estimator` instance.
        """
        if self._model is not None:
            logger.debug("Model already loaded; skipping reload.")
            return
        logger.info("Loading MeTRAbs model (cache_dir=%s)", cache_dir)
        start = time.perf_counter()
        self._model = load_metrabs_model(cache_dir=cache_dir)
        self._model_load_seconds = time.perf_counter() - start
        logger.info("MeTRAbs model loaded in %.2f s", self._model_load_seconds)

    # -- inference ----------------------------------------------------------

    def process_video(
        self,
        video_path: Path,
        *,
        fov_degrees: float | None = None,
        progress: ProgressCallback | None = None,
    ) -> ProcessVideoResult:
        """Run pose estimation on every frame of a video.

        Parameters
        ----------
        video_path
            Path to the input video. Must exist and be openable by OpenCV.
        fov_degrees
            Per-call override for the horizontal field of view. If ``None``,
            the estimator's ``default_fov_degrees`` is used.
        progress
            Optional callback invoked after each processed frame as
            ``progress(processed, total_hint)``. ``total_hint`` is the
            approximate frame count reported by OpenCV's
            ``CAP_PROP_FRAME_COUNT``, which is unreliable for variable-rate
            videos; the authoritative count is available after the call on
            ``result.frame_count``.

        Returns
        -------
        ProcessVideoResult
            A typed result containing the validated :class:`VideoPredictions`.

        Raises
        ------
        FileNotFoundError
            If ``video_path`` does not exist.
        VideoDecodeError
            If OpenCV cannot open the video.
        ModelNotLoadedError
            If the model has not been loaded or injected.
        """
        if not video_path.exists():
            raise FileNotFoundError(f"video file not found: {video_path}")

        # Access the model eagerly so we fail fast before opening the video.
        model = self.model
        fov = fov_degrees if fov_degrees is not None else self.default_fov_degrees

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise VideoDecodeError(f"OpenCV could not open video: {video_path}")

        # Start metrics collection *after* the file-not-found and
        # decode-error paths so total_seconds reflects "work the estimator
        # actually did" rather than setup failures the caller handles.
        process = psutil.Process()
        peak_rss_bytes = process.memory_info().rss
        per_frame_latencies_ms: list[float] = []
        overall_start = time.perf_counter()

        try:
            fps = float(cap.get(cv2.CAP_PROP_FPS))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_hint = max(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), 0)

            logger.info(
                "Processing video %s (%dx%d @ %.2f fps, ~%d frames)",
                video_path,
                width,
                height,
                fps,
                total_hint,
            )

            frames: dict[str, FramePrediction] = {}
            frame_index = 0
            while True:
                ok, bgr_frame = cap.read()
                if not ok:
                    break
                # MeTRAbs was trained on RGB images; OpenCV gives us BGR.
                rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
                # Time only the model call — cv2 decode and colour
                # conversion are not what the benchmark is asking about.
                frame_start = time.perf_counter()
                prediction = self._infer_frame(model, rgb_frame, fov)
                per_frame_latencies_ms.append((time.perf_counter() - frame_start) * 1000.0)
                frames[f"frame_{frame_index:06d}"] = prediction
                frame_index += 1
                # Sample RSS once per frame. Cheap (a single proc/mach
                # syscall) and produces a monotonic peak without needing
                # a background thread.
                rss = process.memory_info().rss
                if rss > peak_rss_bytes:
                    peak_rss_bytes = rss
                if progress is not None:
                    progress(frame_index, total_hint)

            metadata = VideoMetadata(
                frame_count=frame_index,
                fps=fps if fps > 0 else 0.0,
                width=width,
                height=height,
            )
        finally:
            cap.release()

        if frame_index == 0:
            logger.warning("Video %s contained no decodable frames.", video_path)

        total_seconds = time.perf_counter() - overall_start
        device_info = _detect_active_device()
        metrics = PerformanceMetrics(
            model_load_seconds=self._model_load_seconds,
            total_seconds=total_seconds,
            per_frame_latencies_ms=per_frame_latencies_ms,
            peak_rss_mb=peak_rss_bytes / (1024.0 * 1024.0),
            active_device=device_info.device,
            tensorflow_metal_active=device_info.metal_active,
            tensorflow_version=device_info.tf_version,
        )
        logger.info(
            "Processed %d frames in %.2f s (%.1f fps, peak RSS %.0f MB, device %s)",
            frame_index,
            total_seconds,
            frame_index / total_seconds if total_seconds > 0 else 0.0,
            metrics.peak_rss_mb,
            metrics.active_device,
        )

        predictions = VideoPredictions(metadata=metadata, frames=frames)
        return ProcessVideoResult(predictions=predictions, metrics=metrics)

    # -- internals ----------------------------------------------------------

    def _infer_frame(
        self,
        model: Any,
        rgb_frame: Any,
        fov_degrees: float,
    ) -> FramePrediction:
        """Run a single frame through the model and validate the output."""
        pred = model.detect_poses(
            rgb_frame,
            default_fov_degrees=fov_degrees,
            skeleton=self.skeleton,
        )
        return FramePrediction(
            boxes=_to_nested_list(pred["boxes"]),
            poses3d=_to_nested_list(pred["poses3d"]),
            poses2d=_to_nested_list(pred["poses2d"]),
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _to_nested_list(value: Any) -> Any:
    """Normalise a TF tensor, numpy array, or nested list to Python lists.

    The real MeTRAbs model returns ``tf.Tensor`` objects which expose
    ``.numpy()`` returning a ``numpy.ndarray``. Tests inject fake models
    that return plain numpy arrays. Both paths flow through this helper so
    the rest of the code is agnostic to which is in use.
    """
    if hasattr(value, "numpy"):
        value = value.numpy()
    if hasattr(value, "tolist"):
        return value.tolist()
    return list(value)


@dataclass(frozen=True)
class _ActiveDeviceInfo:
    """Small bundle of device info gathered for ``PerformanceMetrics``."""

    device: str
    metal_active: bool
    tf_version: str


def _detect_active_device() -> _ActiveDeviceInfo:
    """Report which TF device the current process would use for inference.

    Returns a synthetic "unknown" bundle if TensorFlow is not importable —
    unit tests that inject a fake model do not have a real TF install at
    the metrics layer, and we do not want to fail the call for that.

    On Apple Silicon with the ``[metal]`` extra installed, TensorFlow
    exposes a ``GPU`` device contributed by the ``tensorflow-metal``
    PluggableDevice. The distinction between that and a real CUDA GPU
    matters for :mod:`neuropose.benchmark`'s ``--compare-cpu`` flow, so
    we surface it via ``metal_active``.
    """
    try:
        import tensorflow as tf
    except ImportError:
        return _ActiveDeviceInfo(
            device="unknown",
            metal_active=False,
            tf_version="unknown",
        )

    tf_version = getattr(tf, "__version__", "unknown")
    try:
        gpu_devices = tf.config.list_physical_devices("GPU")
    except Exception:  # runtime-specific: never hard-fail metrics
        gpu_devices = []

    device = "/GPU:0" if gpu_devices else "/CPU:0"

    metal_active = False
    if gpu_devices:
        try:
            from importlib.metadata import PackageNotFoundError, version

            try:
                version("tensorflow-metal")
                metal_active = True
            except PackageNotFoundError:
                metal_active = False
        except Exception:
            metal_active = False

    return _ActiveDeviceInfo(
        device=device,
        metal_active=metal_active,
        tf_version=tf_version,
    )
