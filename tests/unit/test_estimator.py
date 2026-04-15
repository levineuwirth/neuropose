"""Tests for :class:`neuropose.estimator.Estimator`.

These tests exercise the non-model code paths (video decoding, frame loop,
metadata extraction, result construction, progress reporting, and error
handling) using an injected fake MeTRAbs model. The TensorFlow / real-model
integration smoke test lives in ``tests/integration/`` and lands with
commit 11.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from neuropose.estimator import (
    Estimator,
    ModelNotLoadedError,
    ProcessVideoResult,
    VideoDecodeError,
)
from neuropose.io import FramePrediction, PerformanceMetrics, VideoPredictions


class TestConstruction:
    def test_no_model_by_default(self) -> None:
        estimator = Estimator()
        assert not estimator.is_model_loaded

    def test_injected_model_is_loaded(self, fake_metrabs_model) -> None:
        estimator = Estimator(model=fake_metrabs_model)
        assert estimator.is_model_loaded

    def test_defaults(self) -> None:
        estimator = Estimator()
        assert estimator.device == "/CPU:0"
        assert estimator.skeleton == "berkeley_mhad_43"
        assert estimator.default_fov_degrees == pytest.approx(55.0)

    def test_overrides(self, fake_metrabs_model) -> None:
        estimator = Estimator(
            device="/GPU:0",
            skeleton="smpl_24",
            default_fov_degrees=40.0,
            model=fake_metrabs_model,
        )
        assert estimator.device == "/GPU:0"
        assert estimator.skeleton == "smpl_24"
        assert estimator.default_fov_degrees == pytest.approx(40.0)


class TestModelGuard:
    def test_model_property_raises_when_missing(self) -> None:
        estimator = Estimator()
        with pytest.raises(ModelNotLoadedError):
            _ = estimator.model

    def test_process_video_raises_when_missing(self, synthetic_video: Path) -> None:
        estimator = Estimator()
        with pytest.raises(ModelNotLoadedError):
            estimator.process_video(synthetic_video)

    def test_load_model_delegates_to_loader(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """``Estimator.load_model`` should delegate to ``load_metrabs_model``.

        We verify the delegation without actually invoking TensorFlow or the
        network: the loader is monkeypatched to return a sentinel, and we
        assert it ends up as the estimator's model.
        """
        sentinel = object()
        called_with: list[Path | None] = []

        def fake_loader(cache_dir: Path | None = None) -> object:
            called_with.append(cache_dir)
            return sentinel

        monkeypatch.setattr("neuropose.estimator.load_metrabs_model", fake_loader)
        estimator = Estimator()
        estimator.load_model(cache_dir=Path("/tmp/fake-cache"))
        assert estimator.model is sentinel
        assert called_with == [Path("/tmp/fake-cache")]

    def test_load_model_is_idempotent_when_already_loaded(
        self,
        fake_metrabs_model,
    ) -> None:
        estimator = Estimator(model=fake_metrabs_model)
        # Should not raise, and should not clobber the injected model.
        estimator.load_model()
        assert estimator.model is fake_metrabs_model


class TestProcessVideo:
    def test_returns_typed_result(
        self,
        synthetic_video: Path,
        fake_metrabs_model,
    ) -> None:
        estimator = Estimator(model=fake_metrabs_model)
        result = estimator.process_video(synthetic_video)
        assert isinstance(result, ProcessVideoResult)
        assert isinstance(result.predictions, VideoPredictions)

    def test_frame_count_matches_source(
        self,
        synthetic_video: Path,
        fake_metrabs_model,
    ) -> None:
        estimator = Estimator(model=fake_metrabs_model)
        result = estimator.process_video(synthetic_video)
        assert result.frame_count == 5
        assert fake_metrabs_model.call_count == 5

    def test_frame_naming_is_zero_padded(
        self,
        synthetic_video: Path,
        fake_metrabs_model,
    ) -> None:
        estimator = Estimator(model=fake_metrabs_model)
        result = estimator.process_video(synthetic_video)
        names = result.predictions.frame_names()
        assert names == [
            "frame_000000",
            "frame_000001",
            "frame_000002",
            "frame_000003",
            "frame_000004",
        ]

    def test_metadata_populated(
        self,
        synthetic_video: Path,
        fake_metrabs_model,
    ) -> None:
        estimator = Estimator(model=fake_metrabs_model)
        result = estimator.process_video(synthetic_video)
        metadata = result.predictions.metadata
        assert metadata.frame_count == 5
        assert metadata.width == 32
        assert metadata.height == 32
        assert metadata.fps > 0.0

    def test_each_frame_validates_as_frame_prediction(
        self,
        synthetic_video: Path,
        fake_metrabs_model,
    ) -> None:
        estimator = Estimator(model=fake_metrabs_model)
        result = estimator.process_video(synthetic_video)
        for name in result.predictions.frame_names():
            frame = result.predictions[name]
            assert isinstance(frame, FramePrediction)
            assert len(frame.boxes) == 1
            assert len(frame.poses3d) == 1
            assert len(frame.poses3d[0]) == 2  # Two joints per the fake model.

    def test_progress_callback_invoked_per_frame(
        self,
        synthetic_video: Path,
        fake_metrabs_model,
    ) -> None:
        estimator = Estimator(model=fake_metrabs_model)
        calls: list[tuple[int, int]] = []
        estimator.process_video(
            synthetic_video,
            progress=lambda processed, total: calls.append((processed, total)),
        )
        assert len(calls) == 5
        # Processed counts should be strictly increasing.
        assert [c[0] for c in calls] == [1, 2, 3, 4, 5]

    def test_fov_override_is_passed_through(self, synthetic_video: Path) -> None:
        fov_seen: list[float] = []

        class RecordingModel:
            def detect_poses(
                self,
                image,
                *,
                default_fov_degrees: float,
                skeleton: str,
            ):
                del image, skeleton
                fov_seen.append(default_fov_degrees)
                import numpy as np

                return {
                    "boxes": np.array([[0.0, 0.0, 32.0, 32.0, 0.9]]),
                    "poses3d": np.array([[[0.0, 0.0, 0.0]]]),
                    "poses2d": np.array([[[0.0, 0.0]]]),
                }

        estimator = Estimator(model=RecordingModel(), default_fov_degrees=55.0)
        estimator.process_video(synthetic_video, fov_degrees=40.0)
        assert all(fov == pytest.approx(40.0) for fov in fov_seen)
        assert len(fov_seen) == 5


class TestPerformanceMetrics:
    def test_metrics_attached_to_result(
        self,
        synthetic_video: Path,
        fake_metrabs_model,
    ) -> None:
        estimator = Estimator(model=fake_metrabs_model)
        result = estimator.process_video(synthetic_video)
        assert isinstance(result.metrics, PerformanceMetrics)

    def test_per_frame_latencies_length_matches_frames(
        self,
        synthetic_video: Path,
        fake_metrabs_model,
    ) -> None:
        estimator = Estimator(model=fake_metrabs_model)
        result = estimator.process_video(synthetic_video)
        assert len(result.metrics.per_frame_latencies_ms) == result.frame_count

    def test_all_latencies_are_non_negative(
        self,
        synthetic_video: Path,
        fake_metrabs_model,
    ) -> None:
        estimator = Estimator(model=fake_metrabs_model)
        result = estimator.process_video(synthetic_video)
        assert all(v >= 0.0 for v in result.metrics.per_frame_latencies_ms)

    def test_total_seconds_is_positive(
        self,
        synthetic_video: Path,
        fake_metrabs_model,
    ) -> None:
        estimator = Estimator(model=fake_metrabs_model)
        result = estimator.process_video(synthetic_video)
        assert result.metrics.total_seconds > 0.0

    def test_peak_rss_is_positive(
        self,
        synthetic_video: Path,
        fake_metrabs_model,
    ) -> None:
        estimator = Estimator(model=fake_metrabs_model)
        result = estimator.process_video(synthetic_video)
        # psutil always reports at least a few MB of RSS for a running
        # Python process; the exact number varies by platform.
        assert result.metrics.peak_rss_mb > 0.0

    def test_model_load_seconds_none_when_injected(
        self,
        synthetic_video: Path,
        fake_metrabs_model,
    ) -> None:
        estimator = Estimator(model=fake_metrabs_model)
        result = estimator.process_video(synthetic_video)
        assert result.metrics.model_load_seconds is None

    def test_model_load_seconds_set_after_load(
        self,
        synthetic_video: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """``load_model`` should set ``model_load_seconds`` on the next call.

        We stub the loader to return a recording fake model, time how long
        the estimator's ``load_model`` takes, and verify the number ends
        up on the metrics object.
        """
        import numpy as np

        class Recorder:
            def detect_poses(self, image, **kwargs):
                del image, kwargs
                return {
                    "boxes": np.array([[0.0, 0.0, 32.0, 32.0, 0.9]]),
                    "poses3d": np.array([[[0.0, 0.0, 0.0]]]),
                    "poses2d": np.array([[[0.0, 0.0]]]),
                }

        def fake_loader(cache_dir: Path | None = None) -> object:
            del cache_dir
            return Recorder()

        monkeypatch.setattr("neuropose.estimator.load_metrabs_model", fake_loader)
        estimator = Estimator()
        estimator.load_model()
        result = estimator.process_video(synthetic_video)
        assert result.metrics.model_load_seconds is not None
        assert result.metrics.model_load_seconds >= 0.0

    def test_active_device_string_populated(
        self,
        synthetic_video: Path,
        fake_metrabs_model,
    ) -> None:
        estimator = Estimator(model=fake_metrabs_model)
        result = estimator.process_video(synthetic_video)
        # The exact string depends on the runner's TF install, but it
        # must be one of the two canonical forms.
        assert result.metrics.active_device in {"/CPU:0", "/GPU:0", "unknown"}

    def test_tensorflow_version_populated(
        self,
        synthetic_video: Path,
        fake_metrabs_model,
    ) -> None:
        estimator = Estimator(model=fake_metrabs_model)
        result = estimator.process_video(synthetic_video)
        # TF is in the dev deps so the version should always be a real
        # string, not the "unknown" fallback.
        assert result.metrics.tensorflow_version not in {"", "unknown"}


class TestErrors:
    def test_missing_video(
        self,
        tmp_path: Path,
        fake_metrabs_model,
    ) -> None:
        estimator = Estimator(model=fake_metrabs_model)
        with pytest.raises(FileNotFoundError):
            estimator.process_video(tmp_path / "does_not_exist.mp4")

    def test_unreadable_video_raises_decode_error(
        self,
        tmp_path: Path,
        fake_metrabs_model,
    ) -> None:
        # A file that exists but is not a valid video. cv2.VideoCapture
        # returns isOpened() == False for non-video content.
        path = tmp_path / "not_a_video.avi"
        path.write_bytes(b"this is definitely not a video file")
        estimator = Estimator(model=fake_metrabs_model)
        with pytest.raises(VideoDecodeError):
            estimator.process_video(path)
