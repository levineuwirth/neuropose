"""Shared pytest configuration and fixtures for the NeuroPose test suite."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Slow test opt-in
# ---------------------------------------------------------------------------


def pytest_addoption(parser: pytest.Parser) -> None:
    """Register the ``--runslow`` command-line flag.

    Tests marked ``@pytest.mark.slow`` (typically the integration tests
    under ``tests/integration/`` that download the MeTRAbs model) are
    skipped by default and run only when ``--runslow`` is passed. This
    keeps the default ``pytest`` invocation fast and offline-safe, and
    keeps CI's default test job from burning minutes on a 2 GB download
    on every push.
    """
    parser.addoption(
        "--runslow",
        action="store_true",
        default=False,
        help="run tests marked @pytest.mark.slow (model download required)",
    )


def pytest_collection_modifyitems(
    config: pytest.Config,
    items: list[pytest.Item],
) -> None:
    """Skip ``@slow`` tests unless ``--runslow`` was given on the command line."""
    if config.getoption("--runslow"):
        return
    skip_slow = pytest.mark.skip(reason="need --runslow to run slow tests")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)


# ---------------------------------------------------------------------------
# Environment isolation
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _isolate_environment(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    """Isolate every test from the developer's real home directory.

    Points ``$HOME`` and ``$XDG_DATA_HOME`` at per-test temp directories so
    that any code path that uses the default ``Settings()`` (which reaches
    into ``~/.local/share/neuropose``) cannot accidentally write to the real
    machine. Also clears any ``NEUROPOSE_*`` environment variables that may
    be set in the developer's shell, so test behaviour does not depend on
    who is running the test suite.
    """
    isolated = tmp_path_factory.mktemp("neuropose_env_isolation")
    monkeypatch.setenv("HOME", str(isolated))
    monkeypatch.setenv("XDG_DATA_HOME", str(isolated / "xdg"))
    for key in list(os.environ):
        if key.startswith("NEUROPOSE_"):
            monkeypatch.delenv(key, raising=False)


@pytest.fixture
def xdg_home() -> Path:
    """Return the isolated ``$XDG_DATA_HOME`` set up by ``_isolate_environment``."""
    return Path(os.environ["XDG_DATA_HOME"])


# ---------------------------------------------------------------------------
# Synthetic video + fake MeTRAbs model
# ---------------------------------------------------------------------------


@pytest.fixture
def synthetic_video(tmp_path: Path) -> Path:
    """Generate a tiny synthetic video at test time.

    The fixture writes a 5-frame, 32x32 MJPG-encoded ``.avi`` file. MJPG is
    chosen over ``mp4v`` because it ships with ``opencv-python-headless`` on
    every platform we target, whereas ``mp4v`` occasionally requires an
    ffmpeg binary that may not be present on minimal CI runners.
    """
    path = tmp_path / "synthetic.avi"
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")  # type: ignore[attr-defined]
    writer = cv2.VideoWriter(str(path), fourcc, 30.0, (32, 32))
    assert writer.isOpened(), "cv2.VideoWriter failed to open; MJPG codec missing?"
    for i in range(5):
        # Distinct brightness per frame so a downstream check could verify
        # the test is actually reading frame-by-frame.
        frame = np.full((32, 32, 3), i * 40, dtype=np.uint8)
        writer.write(frame)
    writer.release()
    assert path.exists(), "Synthetic video was not written."
    assert path.stat().st_size > 0, "Synthetic video is empty."
    return path


class _FakeMetrabsModel:
    """Minimal stand-in for the MeTRAbs model used in unit tests.

    Returns deterministic pose data (one person, two joints) per call so
    tests can assert on shapes without importing TensorFlow or downloading
    the real model. The returned arrays are plain numpy so the estimator's
    ``_to_nested_list`` helper exercises its non-``numpy()`` branch.
    """

    def __init__(self) -> None:
        self.call_count = 0

    def detect_poses(
        self,
        image: Any,
        *,
        default_fov_degrees: float,
        skeleton: str,
    ) -> dict[str, np.ndarray]:
        del image, default_fov_degrees, skeleton  # signature-compatible with MeTRAbs
        self.call_count += 1
        return {
            "boxes": np.array([[0.0, 0.0, 32.0, 32.0, 0.95]]),
            "poses3d": np.array([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]]),
            "poses2d": np.array([[[10.0, 20.0], [30.0, 40.0]]]),
        }


@pytest.fixture
def fake_metrabs_model() -> _FakeMetrabsModel:
    """Return a fresh fake MeTRAbs model instance for a single test."""
    return _FakeMetrabsModel()
