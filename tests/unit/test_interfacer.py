"""Tests for :class:`neuropose.interfacer.Interfacer`.

Exercises the daemon's job-lifecycle and state-transition logic with an
injected fake estimator. The full fcntl lock test depends on the
behaviour that within a single process, two ``fcntl.flock`` calls on the
same file through independent file descriptors block each other.
"""

from __future__ import annotations

import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from neuropose.config import Settings
from neuropose.estimator import Estimator
from neuropose.interfacer import (
    AlreadyRunningError,
    Interfacer,
    JobProcessingError,
)
from neuropose.io import (
    JobStatus,
    JobStatusEntry,
    StatusFile,
    load_status,
    save_status,
)

# ---------------------------------------------------------------------------
# Stubs and helpers
# ---------------------------------------------------------------------------


class _RaisingEstimator:
    """Stub estimator whose ``process_video`` always raises."""

    def __init__(self, exc: Exception | None = None) -> None:
        self._exc = exc or RuntimeError("simulated estimator failure")
        self.is_model_loaded = True

    def load_model(self, cache_dir: Path | None = None) -> None:
        del cache_dir

    def process_video(self, video_path: Path) -> Any:
        del video_path
        raise self._exc


def _make_settings(tmp_path: Path) -> Settings:
    """Construct a Settings object pointing at an isolated tmp_path."""
    return Settings(
        data_dir=tmp_path / "jobs",
        model_cache_dir=tmp_path / "models",
    )


def _prepare_job(
    settings: Settings,
    job_name: str,
    videos: list[Path] | None = None,
    extra_files: list[tuple[str, bytes]] | None = None,
) -> Path:
    """Create ``input_dir/<job_name>`` and populate it.

    Parameters
    ----------
    videos
        Video files to copy into the job directory. Relative filenames are
        taken from the source path's ``name`` attribute.
    extra_files
        Additional ``(name, bytes)`` tuples to drop into the job directory.
        Useful for exercising the "directory with files but no videos"
        failure path.
    """
    settings.ensure_dirs()
    job_dir = settings.input_dir / job_name
    job_dir.mkdir(parents=True, exist_ok=True)
    for video in videos or []:
        shutil.copy(video, job_dir / video.name)
    for name, blob in extra_files or []:
        (job_dir / name).write_bytes(blob)
    return job_dir


# ---------------------------------------------------------------------------
# Construction and stop flag
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_initial_state(self, tmp_path: Path, fake_metrabs_model) -> None:
        settings = _make_settings(tmp_path)
        estimator = Estimator(model=fake_metrabs_model)
        interfacer = Interfacer(settings, estimator)
        assert not interfacer.is_stopping

    def test_stop_sets_flag(self, tmp_path: Path, fake_metrabs_model) -> None:
        settings = _make_settings(tmp_path)
        estimator = Estimator(model=fake_metrabs_model)
        interfacer = Interfacer(settings, estimator)
        interfacer.stop()
        assert interfacer.is_stopping

    def test_stop_is_idempotent(self, tmp_path: Path, fake_metrabs_model) -> None:
        settings = _make_settings(tmp_path)
        estimator = Estimator(model=fake_metrabs_model)
        interfacer = Interfacer(settings, estimator)
        interfacer.stop()
        interfacer.stop()
        assert interfacer.is_stopping


# ---------------------------------------------------------------------------
# Job discovery
# ---------------------------------------------------------------------------


class TestDiscoverNewJobs:
    def test_empty_input_dir(self, tmp_path: Path, fake_metrabs_model) -> None:
        settings = _make_settings(tmp_path)
        settings.ensure_dirs()
        interfacer = Interfacer(settings, Estimator(model=fake_metrabs_model))
        assert interfacer._discover_new_jobs(StatusFile(root={})) == []

    def test_missing_input_dir(self, tmp_path: Path, fake_metrabs_model) -> None:
        # data_dir not yet created; ensure_dirs has NOT been called.
        settings = _make_settings(tmp_path)
        interfacer = Interfacer(settings, Estimator(model=fake_metrabs_model))
        assert interfacer._discover_new_jobs(StatusFile(root={})) == []

    def test_skips_empty_directories_silently(self, tmp_path: Path, fake_metrabs_model) -> None:
        settings = _make_settings(tmp_path)
        settings.ensure_dirs()
        (settings.input_dir / "empty_job").mkdir()
        interfacer = Interfacer(settings, Estimator(model=fake_metrabs_model))
        assert interfacer._discover_new_jobs(StatusFile(root={})) == []

    def test_returns_non_empty_jobs_in_sorted_order(
        self,
        tmp_path: Path,
        synthetic_video: Path,
        fake_metrabs_model,
    ) -> None:
        settings = _make_settings(tmp_path)
        _prepare_job(settings, "job_c", videos=[synthetic_video])
        _prepare_job(settings, "job_a", videos=[synthetic_video])
        _prepare_job(settings, "job_b", videos=[synthetic_video])
        interfacer = Interfacer(settings, Estimator(model=fake_metrabs_model))
        assert interfacer._discover_new_jobs(StatusFile(root={})) == [
            "job_a",
            "job_b",
            "job_c",
        ]

    def test_excludes_jobs_already_in_status(
        self,
        tmp_path: Path,
        synthetic_video: Path,
        fake_metrabs_model,
    ) -> None:
        settings = _make_settings(tmp_path)
        _prepare_job(settings, "job_a", videos=[synthetic_video])
        _prepare_job(settings, "job_b", videos=[synthetic_video])
        status = StatusFile.model_validate(
            {
                "job_a": {
                    "status": "completed",
                    "started_at": datetime(2026, 4, 13, tzinfo=UTC).isoformat(),
                }
            }
        )
        interfacer = Interfacer(settings, Estimator(model=fake_metrabs_model))
        assert interfacer._discover_new_jobs(status) == ["job_b"]

    def test_dir_with_non_video_files_is_returned(self, tmp_path: Path, fake_metrabs_model) -> None:
        # Dirs that contain files but no *videos* are NOT silently skipped
        # — they should be returned so process_job marks them failed.
        settings = _make_settings(tmp_path)
        _prepare_job(
            settings,
            "job_a",
            extra_files=[("README.txt", b"nothing to see")],
        )
        interfacer = Interfacer(settings, Estimator(model=fake_metrabs_model))
        assert interfacer._discover_new_jobs(StatusFile(root={})) == ["job_a"]


# ---------------------------------------------------------------------------
# process_job happy path
# ---------------------------------------------------------------------------


class TestProcessJobSuccess:
    def test_happy_path_marks_completed(
        self,
        tmp_path: Path,
        synthetic_video: Path,
        fake_metrabs_model,
    ) -> None:
        settings = _make_settings(tmp_path)
        _prepare_job(settings, "job_a", videos=[synthetic_video])
        estimator = Estimator(model=fake_metrabs_model)
        interfacer = Interfacer(settings, estimator)

        entry = interfacer.process_job("job_a")

        assert entry.status == JobStatus.COMPLETED
        assert entry.results_path is not None
        assert entry.results_path.exists()
        assert entry.error is None
        assert entry.completed_at is not None

    def test_happy_path_persists_status(
        self,
        tmp_path: Path,
        synthetic_video: Path,
        fake_metrabs_model,
    ) -> None:
        settings = _make_settings(tmp_path)
        _prepare_job(settings, "job_a", videos=[synthetic_video])
        interfacer = Interfacer(settings, Estimator(model=fake_metrabs_model))

        interfacer.process_job("job_a")

        loaded = load_status(settings.status_file)
        assert "job_a" in loaded.root
        assert loaded.root["job_a"].status == JobStatus.COMPLETED

    def test_happy_path_leaves_inputs_in_place(
        self,
        tmp_path: Path,
        synthetic_video: Path,
        fake_metrabs_model,
    ) -> None:
        # A successful job should NOT be quarantined — its inputs stay in
        # input_dir. (An operator might want to rename / archive them
        # separately, but that's not the daemon's job.)
        settings = _make_settings(tmp_path)
        _prepare_job(settings, "job_a", videos=[synthetic_video])
        interfacer = Interfacer(settings, Estimator(model=fake_metrabs_model))

        interfacer.process_job("job_a")

        assert (settings.input_dir / "job_a").exists()
        assert not (settings.failed_dir / "job_a").exists()


# ---------------------------------------------------------------------------
# process_job failure paths
# ---------------------------------------------------------------------------


class TestProcessJobFailure:
    def test_no_videos_marks_failed_and_quarantines(
        self, tmp_path: Path, fake_metrabs_model
    ) -> None:
        settings = _make_settings(tmp_path)
        _prepare_job(
            settings,
            "job_a",
            extra_files=[("README.txt", b"nothing to see")],
        )
        interfacer = Interfacer(settings, Estimator(model=fake_metrabs_model))

        entry = interfacer.process_job("job_a")

        assert entry.status == JobStatus.FAILED
        assert entry.error is not None
        assert "no supported video files" in entry.error
        # Inputs moved to failed_dir.
        assert not (settings.input_dir / "job_a").exists()
        assert (settings.failed_dir / "job_a").exists()
        assert (settings.failed_dir / "job_a" / "README.txt").exists()

    def test_estimator_exception_marks_failed_and_quarantines(
        self, tmp_path: Path, synthetic_video: Path
    ) -> None:
        settings = _make_settings(tmp_path)
        _prepare_job(settings, "job_a", videos=[synthetic_video])
        interfacer = Interfacer(settings, _RaisingEstimator())  # type: ignore[arg-type]

        entry = interfacer.process_job("job_a")

        assert entry.status == JobStatus.FAILED
        assert entry.error is not None
        assert "RuntimeError" in entry.error
        assert "simulated estimator failure" in entry.error
        assert not (settings.input_dir / "job_a").exists()
        assert (settings.failed_dir / "job_a").exists()

    def test_quarantine_collision_suffixes(self, tmp_path: Path, fake_metrabs_model) -> None:
        settings = _make_settings(tmp_path)
        settings.ensure_dirs()
        # Pre-populate failed_dir with an existing entry for "job_a".
        (settings.failed_dir / "job_a").mkdir()
        _prepare_job(
            settings,
            "job_a",
            extra_files=[("not_a_video.txt", b"x")],
        )
        interfacer = Interfacer(settings, Estimator(model=fake_metrabs_model))

        interfacer.process_job("job_a")

        assert (settings.failed_dir / "job_a").exists()
        assert (settings.failed_dir / "job_a.1").exists()
        assert (settings.failed_dir / "job_a.1" / "not_a_video.txt").exists()

    def test_raising_estimator_error_type_is_reported(
        self, tmp_path: Path, synthetic_video: Path
    ) -> None:
        settings = _make_settings(tmp_path)
        _prepare_job(settings, "job_a", videos=[synthetic_video])
        interfacer = Interfacer(
            settings,
            _RaisingEstimator(exc=JobProcessingError("custom boom")),  # type: ignore[arg-type]
        )

        entry = interfacer.process_job("job_a")

        assert entry.status == JobStatus.FAILED
        assert entry.error is not None
        assert "JobProcessingError" in entry.error
        assert "custom boom" in entry.error


# ---------------------------------------------------------------------------
# Stuck-processing recovery
# ---------------------------------------------------------------------------


class TestRecoverStuckJobs:
    def test_recovers_single_stuck_entry(
        self, tmp_path: Path, synthetic_video: Path, fake_metrabs_model
    ) -> None:
        settings = _make_settings(tmp_path)
        _prepare_job(settings, "job_a", videos=[synthetic_video])
        status = StatusFile.model_validate(
            {
                "job_a": {
                    "status": "processing",
                    "started_at": datetime(2026, 4, 13, tzinfo=UTC).isoformat(),
                }
            }
        )
        save_status(settings.status_file, status)
        interfacer = Interfacer(settings, Estimator(model=fake_metrabs_model))

        interfacer.recover_stuck_jobs()

        loaded = load_status(settings.status_file)
        entry = loaded.root["job_a"]
        assert entry.status == JobStatus.FAILED
        assert entry.error is not None
        assert "interrupted" in entry.error
        assert entry.completed_at is not None
        # Inputs were quarantined.
        assert not (settings.input_dir / "job_a").exists()
        assert (settings.failed_dir / "job_a").exists()

    def test_does_not_touch_completed_entries(self, tmp_path: Path, fake_metrabs_model) -> None:
        settings = _make_settings(tmp_path)
        settings.ensure_dirs()
        completed = datetime(2026, 4, 13, 10, 0, 0, tzinfo=UTC)
        status = StatusFile.model_validate(
            {
                "job_a": {
                    "status": "completed",
                    "started_at": completed.isoformat(),
                    "completed_at": completed.isoformat(),
                },
                "job_b": {
                    "status": "failed",
                    "started_at": completed.isoformat(),
                    "completed_at": completed.isoformat(),
                    "error": "old failure",
                },
            }
        )
        save_status(settings.status_file, status)
        interfacer = Interfacer(settings, Estimator(model=fake_metrabs_model))

        interfacer.recover_stuck_jobs()

        loaded = load_status(settings.status_file)
        assert loaded.root["job_a"].status == JobStatus.COMPLETED
        assert loaded.root["job_b"].status == JobStatus.FAILED
        assert loaded.root["job_b"].error == "old failure"

    def test_no_status_file_is_noop(self, tmp_path: Path, fake_metrabs_model) -> None:
        settings = _make_settings(tmp_path)
        interfacer = Interfacer(settings, Estimator(model=fake_metrabs_model))
        # Must not raise even though status_file does not exist.
        interfacer.recover_stuck_jobs()


# ---------------------------------------------------------------------------
# run_once
# ---------------------------------------------------------------------------


class TestRunOnce:
    def test_no_jobs_is_noop(self, tmp_path: Path, fake_metrabs_model) -> None:
        settings = _make_settings(tmp_path)
        interfacer = Interfacer(settings, Estimator(model=fake_metrabs_model))
        interfacer.run_once()
        loaded = load_status(settings.status_file)
        assert loaded.is_empty()

    def test_processes_all_new_jobs(
        self,
        tmp_path: Path,
        synthetic_video: Path,
        fake_metrabs_model,
    ) -> None:
        settings = _make_settings(tmp_path)
        _prepare_job(settings, "job_a", videos=[synthetic_video])
        _prepare_job(settings, "job_b", videos=[synthetic_video])
        interfacer = Interfacer(settings, Estimator(model=fake_metrabs_model))

        interfacer.run_once()

        loaded = load_status(settings.status_file)
        assert loaded.root["job_a"].status == JobStatus.COMPLETED
        assert loaded.root["job_b"].status == JobStatus.COMPLETED

    def test_stop_between_jobs_defers_remaining(
        self,
        tmp_path: Path,
        synthetic_video: Path,
        fake_metrabs_model,
    ) -> None:
        settings = _make_settings(tmp_path)
        _prepare_job(settings, "job_a", videos=[synthetic_video])
        _prepare_job(settings, "job_b", videos=[synthetic_video])
        interfacer = Interfacer(settings, Estimator(model=fake_metrabs_model))

        # Override process_job so the first call flips the stop flag
        # before returning. The loop should then break before job_b.
        original = interfacer.process_job
        call_log: list[str] = []

        def recording_process_job(job_name: str) -> JobStatusEntry:
            call_log.append(job_name)
            result = original(job_name)
            interfacer.stop()
            return result

        interfacer.process_job = recording_process_job  # type: ignore[method-assign]
        interfacer.run_once()

        assert call_log == ["job_a"]
        loaded = load_status(settings.status_file)
        assert "job_a" in loaded.root
        assert "job_b" not in loaded.root


# ---------------------------------------------------------------------------
# Single-instance lock
# ---------------------------------------------------------------------------


class TestLock:
    def test_first_acquire_succeeds(self, tmp_path: Path, fake_metrabs_model) -> None:
        settings = _make_settings(tmp_path)
        interfacer = Interfacer(settings, Estimator(model=fake_metrabs_model))
        try:
            interfacer._acquire_lock()
            assert interfacer._lock_fd is not None
        finally:
            interfacer._release_lock()

    def test_second_acquire_raises_already_running(
        self, tmp_path: Path, fake_metrabs_model
    ) -> None:
        settings = _make_settings(tmp_path)
        first = Interfacer(settings, Estimator(model=fake_metrabs_model))
        second = Interfacer(settings, Estimator(model=fake_metrabs_model))
        first._acquire_lock()
        try:
            with pytest.raises(AlreadyRunningError):
                second._acquire_lock()
        finally:
            first._release_lock()

    def test_release_allows_subsequent_acquire(self, tmp_path: Path, fake_metrabs_model) -> None:
        settings = _make_settings(tmp_path)
        first = Interfacer(settings, Estimator(model=fake_metrabs_model))
        first._acquire_lock()
        first._release_lock()

        second = Interfacer(settings, Estimator(model=fake_metrabs_model))
        try:
            second._acquire_lock()  # Should succeed after release.
        finally:
            second._release_lock()

    def test_lock_file_contains_pid(self, tmp_path: Path, fake_metrabs_model) -> None:
        settings = _make_settings(tmp_path)
        interfacer = Interfacer(settings, Estimator(model=fake_metrabs_model))
        try:
            interfacer._acquire_lock()
            lock_path = settings.data_dir / ".neuropose.lock"
            content = lock_path.read_text().strip()
            import os

            assert content == str(os.getpid())
        finally:
            interfacer._release_lock()


# ---------------------------------------------------------------------------
# Interruptible sleep
# ---------------------------------------------------------------------------


class TestInterruptibleSleep:
    def test_zero_returns_immediately(self, tmp_path: Path, fake_metrabs_model) -> None:
        settings = _make_settings(tmp_path)
        interfacer = Interfacer(settings, Estimator(model=fake_metrabs_model))
        import time

        start = time.monotonic()
        interfacer._interruptible_sleep(0)
        elapsed = time.monotonic() - start
        assert elapsed < 0.1

    def test_stop_flag_wakes_sleep_early(self, tmp_path: Path, fake_metrabs_model) -> None:
        settings = _make_settings(tmp_path)
        interfacer = Interfacer(settings, Estimator(model=fake_metrabs_model))
        interfacer.stop()
        import time

        start = time.monotonic()
        interfacer._interruptible_sleep(5.0)
        elapsed = time.monotonic() - start
        # With stop flag already set, the sleep should return in well under
        # the 5-second nominal window.
        assert elapsed < 1.0
