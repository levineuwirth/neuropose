"""Tests for :mod:`neuropose.io` schema and helpers."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from neuropose.io import (
    FramePrediction,
    JobResults,
    JobStatus,
    StatusFile,
    VideoPredictions,
    load_job_results,
    load_status,
    load_video_predictions,
    save_job_results,
    save_status,
    save_video_predictions,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def one_frame() -> dict:
    """A minimal valid FramePrediction payload (one person, two joints)."""
    return {
        "boxes": [[10.0, 20.0, 100.0, 200.0, 0.95]],
        "poses3d": [[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]],
        "poses2d": [[[10.0, 20.0], [30.0, 40.0]]],
    }


@pytest.fixture
def video_payload(one_frame: dict) -> dict:
    return {
        "frame_0000.png": one_frame,
        "frame_0001.png": one_frame,
    }


# ---------------------------------------------------------------------------
# FramePrediction
# ---------------------------------------------------------------------------


class TestFramePrediction:
    def test_roundtrip(self, one_frame: dict) -> None:
        frame = FramePrediction.model_validate(one_frame)
        assert frame.boxes == one_frame["boxes"]
        assert frame.poses3d == one_frame["poses3d"]
        assert frame.poses2d == one_frame["poses2d"]

    def test_rejects_extra_fields(self, one_frame: dict) -> None:
        one_frame["extra"] = "bogus"
        with pytest.raises(ValidationError):
            FramePrediction.model_validate(one_frame)

    def test_is_frozen(self, one_frame: dict) -> None:
        frame = FramePrediction.model_validate(one_frame)
        with pytest.raises(ValidationError):
            frame.boxes = []


# ---------------------------------------------------------------------------
# VideoPredictions
# ---------------------------------------------------------------------------


class TestVideoPredictions:
    def test_from_dict(self, video_payload: dict) -> None:
        vp = VideoPredictions.model_validate(video_payload)
        assert len(vp) == 2
        assert vp.frames() == ["frame_0000.png", "frame_0001.png"]
        assert vp["frame_0000.png"].boxes[0][4] == pytest.approx(0.95)

    def test_iteration(self, video_payload: dict) -> None:
        vp = VideoPredictions.model_validate(video_payload)
        assert list(vp) == ["frame_0000.png", "frame_0001.png"]

    def test_save_and_load_roundtrip(self, tmp_path: Path, video_payload: dict) -> None:
        vp = VideoPredictions.model_validate(video_payload)
        path = tmp_path / "preds" / "video.json"
        save_video_predictions(path, vp)
        assert path.exists()
        loaded = load_video_predictions(path)
        assert loaded.frames() == vp.frames()
        assert loaded["frame_0000.png"].poses3d == vp["frame_0000.png"].poses3d


# ---------------------------------------------------------------------------
# JobResults
# ---------------------------------------------------------------------------


class TestJobResults:
    def test_save_and_load_roundtrip(self, tmp_path: Path, video_payload: dict) -> None:
        jr = JobResults.model_validate(
            {"video_a.mp4": video_payload, "video_b.mp4": video_payload}
        )
        path = tmp_path / "results.json"
        save_job_results(path, jr)
        loaded = load_job_results(path)
        assert loaded.videos() == ["video_a.mp4", "video_b.mp4"]
        assert len(loaded["video_a.mp4"]) == 2


# ---------------------------------------------------------------------------
# Status file
# ---------------------------------------------------------------------------


class TestStatusFile:
    def test_load_missing_returns_empty(self, tmp_path: Path) -> None:
        status = load_status(tmp_path / "nope.json")
        assert status.is_empty()

    def test_load_corrupt_json_returns_empty(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.json"
        path.write_text("{ not valid json")
        status = load_status(path)
        assert status.is_empty()

    def test_load_non_mapping_returns_empty(self, tmp_path: Path) -> None:
        path = tmp_path / "list.json"
        path.write_text(json.dumps([1, 2, 3]))
        status = load_status(path)
        assert status.is_empty()

    def test_save_and_load_completed_entry(self, tmp_path: Path) -> None:
        started = datetime(2026, 4, 13, 12, 0, 0, tzinfo=UTC)
        completed = datetime(2026, 4, 13, 12, 5, 0, tzinfo=UTC)
        status = StatusFile.model_validate(
            {
                "job_001": {
                    "status": "completed",
                    "started_at": started.isoformat(),
                    "completed_at": completed.isoformat(),
                    "results_path": "/tmp/results.json",
                    "error": None,
                }
            }
        )
        path = tmp_path / "status.json"
        save_status(path, status)
        loaded = load_status(path)
        entry = loaded.root["job_001"]
        assert entry.status == JobStatus.COMPLETED
        assert entry.started_at == started
        assert entry.completed_at == completed
        assert entry.error is None

    def test_save_is_atomic(self, tmp_path: Path) -> None:
        """``save_status`` leaves no orphan ``.tmp`` file on success."""
        started = datetime(2026, 4, 13, tzinfo=UTC)
        status = StatusFile.model_validate(
            {
                "job_001": {
                    "status": "processing",
                    "started_at": started.isoformat(),
                }
            }
        )
        path = tmp_path / "status.json"
        save_status(path, status)
        assert path.exists()
        tmps = list(tmp_path.glob("status.json.tmp"))
        assert tmps == []

    def test_failed_entry_carries_error_message(self, tmp_path: Path) -> None:
        started = datetime(2026, 4, 13, tzinfo=UTC)
        status = StatusFile.model_validate(
            {
                "job_001": {
                    "status": "failed",
                    "started_at": started.isoformat(),
                    "error": "ffmpeg decode failed: codec not supported",
                }
            }
        )
        path = tmp_path / "status.json"
        save_status(path, status)
        loaded = load_status(path)
        entry = loaded.root["job_001"]
        assert entry.status == JobStatus.FAILED
        assert entry.error is not None
        assert "ffmpeg" in entry.error

    def test_rejects_unknown_status(self, tmp_path: Path) -> None:
        with pytest.raises(ValidationError):
            StatusFile.model_validate(
                {
                    "job_001": {
                        "status": "some-unknown-state",
                        "started_at": datetime(2026, 4, 13, tzinfo=UTC).isoformat(),
                    }
                }
            )
