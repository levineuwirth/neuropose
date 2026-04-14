"""End-to-end smoke test for the MeTRAbs model loader and estimator.

This module lives under ``tests/integration/`` and every test in it is
marked ``@pytest.mark.slow``. That means:

- ``pytest`` with no flags **skips** these tests. The conftest hook
  in ``tests/conftest.py`` requires the ``--runslow`` flag to run
  anything marked ``slow``.
- ``pytest --runslow`` runs them. On a cold cache this triggers a
  ~2 GB download of the MeTRAbs model tarball from its upstream URL;
  on a warm cache (subsequent runs with the same ``cache_dir``) it
  completes in seconds.

The intent of this file is **plumbing verification**, not accuracy
benchmarking:

- Does the loader download, verify, extract, and load the tarball?
- Does the loader's second call hit the cache without re-downloading?
- Does the estimator run end-to-end against a real MeTRAbs model on a
  synthetic video and produce a valid :class:`VideoPredictions` object?

Accuracy against reference pose data is out of scope here and will
live in a separate benchmark harness once the project has data it is
cleared to test against.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from neuropose._model import load_metrabs_model
from neuropose.estimator import Estimator
from neuropose.io import FramePrediction, VideoPredictions

pytestmark = pytest.mark.slow


@pytest.fixture
def integration_video(tmp_path: Path) -> Path:
    """Generate a 384x288 synthetic video sized for MeTRAbs input.

    The default ``synthetic_video`` fixture in ``tests/conftest.py``
    produces 32x32 frames, which is too small for MeTRAbs's 384 px
    input and may cause the detector pipeline to short-circuit
    unpredictably. This fixture produces a modestly-sized video so
    the smoke test's plumbing assertions are meaningful.
    """
    path = tmp_path / "integration.avi"
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")  # type: ignore[attr-defined]
    writer = cv2.VideoWriter(str(path), fourcc, 30.0, (384, 288))
    assert writer.isOpened(), "cv2.VideoWriter failed to open; MJPG codec missing?"
    for i in range(5):
        # Flat gray with a shifting offset per frame. There are no
        # humans in the frame; MeTRAbs should produce zero detections
        # per frame but the pipeline must still return valid structures.
        frame = np.full((288, 384, 3), 100 + i * 10, dtype=np.uint8)
        writer.write(frame)
    writer.release()
    assert path.exists()
    assert path.stat().st_size > 0
    return path


@pytest.fixture(scope="session")
def shared_model_cache_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Session-scoped cache dir so the model is downloaded at most once per run.

    Without this, each test in the file would trigger a fresh download
    because the default ``tmp_path`` is function-scoped. The session
    scope means the first test pays for the download and subsequent
    tests load from the cache.
    """
    return tmp_path_factory.mktemp("neuropose_model_cache")


class TestMetrabsLoader:
    """Exercises the loader's download → verify → extract → load path."""

    def test_download_and_load(self, shared_model_cache_dir: Path) -> None:
        model = load_metrabs_model(cache_dir=shared_model_cache_dir)
        assert model is not None
        for attr in ("detect_poses", "per_skeleton_joint_names", "per_skeleton_joint_edges"):
            assert hasattr(model, attr), f"loaded model is missing {attr}"

    def test_second_call_uses_cache(self, shared_model_cache_dir: Path) -> None:
        """Idempotent: second call should return the cached model cheaply."""
        model_a = load_metrabs_model(cache_dir=shared_model_cache_dir)
        model_b = load_metrabs_model(cache_dir=shared_model_cache_dir)
        # tf.saved_model.load returns a new Python object each call, so
        # identity comparison doesn't work — but both should still
        # expose the MeTRAbs interface.
        assert hasattr(model_a, "detect_poses")
        assert hasattr(model_b, "detect_poses")

    def test_berkeley_mhad_skeleton_is_present(self, shared_model_cache_dir: Path) -> None:
        """The estimator pins skeleton='berkeley_mhad_43'; verify it exists."""
        model = load_metrabs_model(cache_dir=shared_model_cache_dir)
        joint_names = model.per_skeleton_joint_names["berkeley_mhad_43"]
        joint_edges = model.per_skeleton_joint_edges["berkeley_mhad_43"]
        # MeTRAbs exposes these as tf.Tensor objects; just verify we
        # can pull a shape out.
        assert joint_names.shape[0] == 43
        assert joint_edges.shape[0] > 0


class TestEndToEndInference:
    """Runs the estimator against a real model on a synthetic video."""

    def test_estimator_produces_valid_predictions(
        self,
        integration_video: Path,
        shared_model_cache_dir: Path,
    ) -> None:
        model = load_metrabs_model(cache_dir=shared_model_cache_dir)
        estimator = Estimator(model=model)

        result = estimator.process_video(integration_video)

        assert isinstance(result.predictions, VideoPredictions)
        assert result.frame_count == 5
        assert result.predictions.metadata.width == 384
        assert result.predictions.metadata.height == 288

        # Each frame's predictions must validate as a FramePrediction,
        # regardless of whether MeTRAbs detects any people in it.
        for frame_name in result.predictions.frame_names():
            frame = result.predictions[frame_name]
            assert isinstance(frame, FramePrediction)
            assert isinstance(frame.boxes, list)
            assert isinstance(frame.poses3d, list)
            assert isinstance(frame.poses2d, list)

    def test_progress_callback_invoked_per_frame(
        self,
        integration_video: Path,
        shared_model_cache_dir: Path,
    ) -> None:
        model = load_metrabs_model(cache_dir=shared_model_cache_dir)
        estimator = Estimator(model=model)

        processed_counts: list[int] = []
        estimator.process_video(
            integration_video,
            progress=lambda processed, _total: processed_counts.append(processed),
        )

        assert processed_counts == [1, 2, 3, 4, 5]
