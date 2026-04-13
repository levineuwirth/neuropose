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
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2

from neuropose._model import load_metrabs_model
from neuropose.io import FramePrediction, VideoMetadata, VideoPredictions

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
    """

    predictions: VideoPredictions

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
        self._model = load_metrabs_model(cache_dir=cache_dir)
        logger.info("MeTRAbs model loaded.")

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
                prediction = self._infer_frame(model, rgb_frame, fov)
                frames[f"frame_{frame_index:06d}"] = prediction
                frame_index += 1
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

        predictions = VideoPredictions(metadata=metadata, frames=frames)
        return ProcessVideoResult(predictions=predictions)

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
