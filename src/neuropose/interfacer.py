"""Filesystem-polling daemon for NeuroPose jobs.

The :class:`Interfacer` watches an input directory for new job
subdirectories, dispatches each job to a supplied
:class:`neuropose.estimator.Estimator`, and persists job state to disk via
:mod:`neuropose.io`. It owns the input → output → failed directory lifecycle
and the persistent :class:`~neuropose.io.StatusFile`; it does NOT own
inference, which lives on the injected estimator.

Key guarantees
--------------
- **Single-instance**: an exclusive ``fcntl.flock`` on ``data_dir/.neuropose.lock``
  blocks a second daemon from running against the same data directory. The
  lock is released automatically on process exit, even on SIGKILL.
- **Crash recovery**: on startup, any status entries left in
  :attr:`~neuropose.io.JobStatus.PROCESSING` state (i.e. jobs that were in
  flight when the previous daemon was killed) are marked failed with a
  clear error message and their inputs are quarantined to ``failed_dir``.
  The operator can move them back to ``input_dir`` to retry.
- **Graceful shutdown**: ``SIGINT`` and ``SIGTERM`` request an orderly stop.
  The current job finishes before the loop exits; no partial writes.
- **Structured errors**: every failed job records a short
  ``"<ExceptionType>: <message>"`` in the status file's ``error`` field so
  operators have a grep target without needing the log file.
"""

from __future__ import annotations

import contextlib
import fcntl
import logging
import os
import signal
import time
from datetime import UTC, datetime
from pathlib import Path
from types import FrameType
from typing import TYPE_CHECKING

from neuropose.config import Settings
from neuropose.estimator import Estimator
from neuropose.io import (
    JobResults,
    JobStatus,
    JobStatusEntry,
    StatusFile,
    load_status,
    save_job_results,
    save_status,
)

if TYPE_CHECKING:
    from collections.abc import Iterable

logger = logging.getLogger(__name__)


VIDEO_EXTENSIONS = frozenset({".mp4", ".avi", ".mov", ".mkv"})
"""Filename suffixes accepted as job inputs."""

LOCK_FILENAME = ".neuropose.lock"
"""Name of the single-instance lock file, placed under ``data_dir``."""


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class InterfacerError(Exception):
    """Base class for errors raised by :class:`Interfacer`."""


class AlreadyRunningError(InterfacerError):
    """Raised when another :class:`Interfacer` instance holds the lock file."""


class JobProcessingError(InterfacerError):
    """Raised for recoverable per-job failures (missing videos, decode error)."""


# ---------------------------------------------------------------------------
# Interfacer
# ---------------------------------------------------------------------------


class Interfacer:
    """Job-dispatching daemon.

    Parameters
    ----------
    settings
        Runtime configuration. The daemon reads
        ``input_dir``, ``output_dir``, ``failed_dir``, ``status_file``,
        ``poll_interval_seconds``, and ``model_cache_dir``.
    estimator
        A ready-to-use :class:`~neuropose.estimator.Estimator`. If its
        model has not yet been loaded, the daemon will call
        :meth:`~neuropose.estimator.Estimator.load_model` on its
        first :meth:`run` call, passing ``settings.model_cache_dir``.
    """

    def __init__(self, settings: Settings, estimator: Estimator) -> None:
        self._settings = settings
        self._estimator = estimator
        self._stop = False
        self._lock_fd: int | None = None
        self._prev_sigint: object = None
        self._prev_sigterm: object = None

    # -- public lifecycle ---------------------------------------------------

    def run(self) -> None:
        """Run the main daemon loop until :meth:`stop` is called.

        Acquires the single-instance lock, installs signal handlers, loads
        the model if needed, recovers stuck jobs, then loops calling
        :meth:`run_once` until ``self._stop`` is set. On exit (normal or
        exceptional) the lock and signal handlers are cleaned up.
        """
        self._acquire_lock()
        try:
            self._install_signal_handlers()
            try:
                self.recover_stuck_jobs()
                if not self._estimator.is_model_loaded:
                    logger.info("Loading estimator model before entering main loop")
                    self._estimator.load_model(cache_dir=self._settings.model_cache_dir)
                logger.info(
                    "Interfacer running. Polling %s every %d seconds.",
                    self._settings.input_dir,
                    self._settings.poll_interval_seconds,
                )
                while not self._stop:
                    try:
                        self.run_once()
                    except Exception:
                        logger.exception("Unexpected error in main loop; backing off")
                        self._interruptible_sleep(self._settings.poll_interval_seconds * 2)
                        continue
                    self._interruptible_sleep(self._settings.poll_interval_seconds)
            finally:
                self._restore_signal_handlers()
        finally:
            self._release_lock()
        logger.info("Interfacer stopped.")

    def run_once(self) -> None:
        """Execute exactly one poll iteration.

        Discovers new job subdirectories, processes each one in insertion
        order, and returns. Primarily intended for tests and for external
        schedulers that want to drive the daemon themselves. Called
        internally by :meth:`run`.
        """
        self._settings.ensure_dirs()
        status = load_status(self._settings.status_file)
        new_jobs = self._discover_new_jobs(status)
        if new_jobs:
            logger.info("Discovered %d new job(s): %s", len(new_jobs), new_jobs)
        for job_name in new_jobs:
            if self._stop:
                logger.info("Stop requested; deferring remaining jobs to next run.")
                break
            self.process_job(job_name)

    def stop(self) -> None:
        """Request a graceful shutdown of the run loop.

        Idempotent. Safe to call from a signal handler or a different
        thread. The running job (if any) will finish before the loop exits.
        """
        self._stop = True

    @property
    def is_stopping(self) -> bool:
        """Return ``True`` if :meth:`stop` has been called."""
        return self._stop

    # -- job processing -----------------------------------------------------

    def process_job(self, job_name: str) -> JobStatusEntry:
        """Run a single job from discovery through final status.

        Transitions the job's status entry ``processing → completed|failed``,
        writes ``results.json`` on success, or quarantines the job inputs to
        ``failed_dir`` on failure. Always persists the status file before
        returning.

        Parameters
        ----------
        job_name
            Name of the job subdirectory under ``settings.input_dir``.

        Returns
        -------
        JobStatusEntry
            The final status entry as persisted.
        """
        logger.info("[%s] Starting job", job_name)
        started_at = datetime.now(UTC)

        status = load_status(self._settings.status_file)
        status.root[job_name] = JobStatusEntry(
            status=JobStatus.PROCESSING,
            started_at=started_at,
        )
        save_status(self._settings.status_file, status)

        try:
            final_entry = self._run_job_inner(job_name, started_at)
            logger.info("[%s] Completed successfully", job_name)
        except Exception as exc:
            logger.exception("[%s] Failed", job_name)
            final_entry = JobStatusEntry(
                status=JobStatus.FAILED,
                started_at=started_at,
                completed_at=datetime.now(UTC),
                error=_format_error(exc),
            )
            self._quarantine_job(job_name)

        status = load_status(self._settings.status_file)
        status.root[job_name] = final_entry
        save_status(self._settings.status_file, status)
        return final_entry

    def recover_stuck_jobs(self) -> None:
        """Mark any ``processing`` entries as failed and quarantine their inputs.

        Intended to run once on daemon startup. A job in ``processing``
        state at startup was in flight when the previous daemon instance
        died, so its actual state on disk is unknown. Rather than retrying
        automatically (which would re-process arbitrary amounts of work),
        we mark it failed with a clear interruption message and move its
        inputs to ``failed_dir`` for operator review.
        """
        status = load_status(self._settings.status_file)
        dirty = False
        for job_name, entry in list(status.root.items()):
            if entry.status != JobStatus.PROCESSING:
                continue
            logger.warning("[%s] Recovering stuck job from previous run", job_name)
            status.root[job_name] = JobStatusEntry(
                status=JobStatus.FAILED,
                started_at=entry.started_at,
                completed_at=datetime.now(UTC),
                error="interrupted: daemon shutdown or crash before completion",
            )
            self._quarantine_job(job_name)
            dirty = True
        if dirty:
            save_status(self._settings.status_file, status)

    # -- internals: job lifecycle ------------------------------------------

    def _run_job_inner(self, job_name: str, started_at: datetime) -> JobStatusEntry:
        """Do the actual inference work for a single job.

        Raises on any failure — the caller in :meth:`process_job` handles
        the transition to the failed state and the quarantine move.

        During inference the interfacer updates the job's
        :class:`JobStatusEntry` roughly every
        :attr:`~neuropose.config.Settings.status_checkpoint_every_frames`
        frames, so :mod:`neuropose.monitor` can render a live progress
        bar for collaborators. Each checkpoint is a full atomic rewrite
        of ``status.json`` via :func:`save_status`, which is
        acceptable because scheduled pose inference is many orders of
        magnitude more expensive than the write itself.
        """
        job_in_path = self._settings.input_dir / job_name
        job_out_path = self._settings.output_dir / job_name
        job_out_path.mkdir(parents=True, exist_ok=True)

        videos = sorted(_discover_videos(job_in_path))
        if not videos:
            raise JobProcessingError(
                f"no supported video files found in {job_in_path} "
                f"(accepted extensions: {sorted(VIDEO_EXTENSIONS)})"
            )

        videos_total = len(videos)
        # Seed the initial progress checkpoint so the monitor shows
        # "videos_total=N, videos_completed=0" from the first poll after
        # the job starts, rather than waiting until a callback fires.
        self._checkpoint_progress(
            job_name,
            started_at=started_at,
            current_video=videos[0].name,
            frames_processed=0,
            frames_total=None,
            videos_completed=0,
            videos_total=videos_total,
        )

        per_video_predictions = {}
        checkpoint_every = self._settings.status_checkpoint_every_frames
        for video_index, video_path in enumerate(videos):
            if self._stop:
                raise JobProcessingError(
                    f"stop requested mid-job after processing "
                    f"{len(per_video_predictions)}/{videos_total} videos"
                )
            logger.info("[%s] Processing video %s", job_name, video_path.name)

            def _on_frame(
                processed: int,
                total_hint: int,
                *,
                # Bind the loop-local values so the closure captures
                # them correctly for each iteration — without this the
                # late-binding gotcha would make every callback report
                # the last video's name once the loop advances.
                _job_name: str = job_name,
                _started_at: datetime = started_at,
                _current_video: str = video_path.name,
                _video_index: int = video_index,
                _videos_total: int = videos_total,
                _checkpoint_every: int = checkpoint_every,
            ) -> None:
                if processed % _checkpoint_every != 0:
                    return
                self._checkpoint_progress(
                    _job_name,
                    started_at=_started_at,
                    current_video=_current_video,
                    frames_processed=processed,
                    frames_total=total_hint if total_hint > 0 else None,
                    videos_completed=_video_index,
                    videos_total=_videos_total,
                )

            result = self._estimator.process_video(video_path, progress=_on_frame)
            per_video_predictions[video_path.name] = result.predictions
            logger.info(
                "[%s] Processed %s (%d frames)",
                job_name,
                video_path.name,
                result.frame_count,
            )
            # Post-video checkpoint: snap videos_completed to the end of
            # this video even if the last frame didn't fall on the
            # checkpoint cadence, so the monitor's "N / M videos done"
            # line is always exact after a video finishes.
            self._checkpoint_progress(
                job_name,
                started_at=started_at,
                current_video=video_path.name,
                frames_processed=result.frame_count,
                frames_total=result.frame_count,
                videos_completed=video_index + 1,
                videos_total=videos_total,
            )

        job_results = JobResults(root=per_video_predictions)
        results_path = job_out_path / "results.json"
        save_job_results(results_path, job_results)
        logger.info("[%s] Wrote aggregated results to %s", job_name, results_path)

        return JobStatusEntry(
            status=JobStatus.COMPLETED,
            started_at=started_at,
            completed_at=datetime.now(UTC),
            results_path=results_path,
            videos_completed=videos_total,
            videos_total=videos_total,
            percent_complete=100.0,
            last_update=datetime.now(UTC),
        )

    def _checkpoint_progress(
        self,
        job_name: str,
        *,
        started_at: datetime,
        current_video: str,
        frames_processed: int,
        frames_total: int | None,
        videos_completed: int,
        videos_total: int,
    ) -> None:
        """Rewrite ``status.json`` with the current per-job progress.

        Computes ``percent_complete`` across the whole job: videos that
        have fully finished contribute ``1.0`` each, and the current
        video contributes ``frames_processed / frames_total`` if the
        frame-count hint is known, else a partial-credit estimate of
        ``0.5``. The overall fraction is then averaged across
        ``videos_total`` and scaled to 0-100.

        Never raises on an I/O error — progress checkpoints are
        best-effort. If the write fails we log and move on so the
        inference loop keeps making forward progress.
        """
        if frames_total and frames_total > 0:
            current_fraction = frames_processed / frames_total
        elif frames_processed > 0:
            current_fraction = 0.5
        else:
            current_fraction = 0.0
        overall_fraction = (videos_completed + current_fraction) / max(videos_total, 1)
        percent = max(0.0, min(100.0, overall_fraction * 100.0))

        try:
            status = load_status(self._settings.status_file)
            existing = status.root.get(job_name)
            if existing is None:
                # The entry should have been seeded by process_job before
                # _run_job_inner was called. If it isn't there, skip the
                # checkpoint — something else has already deleted the
                # entry and we do not want to recreate a ghost.
                return
            status.root[job_name] = existing.model_copy(
                update={
                    "current_video": current_video,
                    "frames_processed": frames_processed,
                    "frames_total": frames_total,
                    "videos_completed": videos_completed,
                    "videos_total": videos_total,
                    "percent_complete": percent,
                    "last_update": datetime.now(UTC),
                }
            )
            save_status(self._settings.status_file, status)
        except Exception:
            logger.warning(
                "[%s] Failed to checkpoint progress; continuing inference",
                job_name,
                exc_info=True,
            )

    def _discover_new_jobs(self, status: StatusFile) -> list[str]:
        """Return names of job subdirectories not yet tracked in ``status``.

        Empty directories are **silently skipped** (not returned) so a user
        mid-copy does not trigger a spurious "no videos" failure. A
        directory containing files that are not videos IS returned here; it
        will fail inside :meth:`_run_job_inner` and be quarantined, which
        is the correct behaviour for genuinely broken inputs.
        """
        input_dir = self._settings.input_dir
        if not input_dir.exists():
            return []
        candidates = sorted(
            p for p in input_dir.iterdir() if p.is_dir() and p.name not in status.root
        )
        new: list[str] = []
        for path in candidates:
            if _is_empty_dir(path):
                logger.debug("[%s] Skipping empty job directory", path.name)
                continue
            new.append(path.name)
        return new

    def _quarantine_job(self, job_name: str) -> None:
        """Move a job's input directory to the failed directory.

        If a quarantine target with the same name already exists (e.g.
        because the same job name has failed before), a numeric suffix is
        appended to disambiguate.
        """
        source = self._settings.input_dir / job_name
        if not source.exists():
            logger.debug("[%s] Nothing to quarantine (source absent)", job_name)
            return
        self._settings.failed_dir.mkdir(parents=True, exist_ok=True)
        dest = self._settings.failed_dir / job_name
        suffix = 1
        while dest.exists():
            dest = self._settings.failed_dir / f"{job_name}.{suffix}"
            suffix += 1
        try:
            source.rename(dest)
        except OSError:
            logger.exception("[%s] Failed to quarantine inputs to %s", job_name, dest)
            return
        logger.info("[%s] Quarantined inputs to %s", job_name, dest)

    # -- internals: sleep / signals / lock ---------------------------------

    def _interruptible_sleep(self, seconds: float) -> None:
        """Sleep up to ``seconds`` but wake early if :meth:`stop` is called."""
        if seconds <= 0:
            return
        deadline = time.monotonic() + seconds
        while not self._stop:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return
            time.sleep(min(0.5, remaining))

    def _install_signal_handlers(self) -> None:
        """Install SIGINT/SIGTERM handlers that set the stop flag.

        Signal handlers can only be installed from the main thread; calling
        this from a non-main thread (as tests occasionally do) raises
        :class:`ValueError`, which we downgrade to a warning. The stop
        mechanism still works via :meth:`stop`.
        """
        try:
            self._prev_sigint = signal.signal(signal.SIGINT, self._handle_signal)
            self._prev_sigterm = signal.signal(signal.SIGTERM, self._handle_signal)
        except ValueError:
            logger.warning(
                "Could not install signal handlers (Interfacer.run() is not running "
                "on the main thread); rely on explicit Interfacer.stop() instead."
            )
            self._prev_sigint = None
            self._prev_sigterm = None

    def _restore_signal_handlers(self) -> None:
        """Restore the signal handlers that were in place before :meth:`run`."""
        if self._prev_sigint is not None:
            with contextlib.suppress(ValueError, TypeError):
                signal.signal(signal.SIGINT, self._prev_sigint)  # type: ignore[arg-type]
        if self._prev_sigterm is not None:
            with contextlib.suppress(ValueError, TypeError):
                signal.signal(signal.SIGTERM, self._prev_sigterm)  # type: ignore[arg-type]
        self._prev_sigint = None
        self._prev_sigterm = None

    def _handle_signal(self, signum: int, frame: FrameType | None) -> None:
        """Signal handler: request shutdown."""
        del frame
        logger.info("Received signal %d; requesting graceful shutdown", signum)
        self.stop()

    def _acquire_lock(self) -> None:
        """Acquire the single-instance lock file, or raise :class:`AlreadyRunningError`."""
        self._settings.data_dir.mkdir(parents=True, exist_ok=True)
        lock_path = self._settings.data_dir / LOCK_FILENAME
        fd = os.open(str(lock_path), os.O_WRONLY | os.O_CREAT, 0o600)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            os.close(fd)
            raise AlreadyRunningError(
                f"another NeuroPose daemon already holds {lock_path}"
            ) from exc
        # Overwrite the lock file with our PID so humans can see who owns it.
        os.ftruncate(fd, 0)
        os.write(fd, f"{os.getpid()}\n".encode())
        self._lock_fd = fd
        logger.debug("Acquired lock %s (pid %d)", lock_path, os.getpid())

    def _release_lock(self) -> None:
        """Release the single-instance lock if held."""
        if self._lock_fd is None:
            return
        with contextlib.suppress(OSError):
            fcntl.flock(self._lock_fd, fcntl.LOCK_UN)
        with contextlib.suppress(OSError):
            os.close(self._lock_fd)
        self._lock_fd = None


# ---------------------------------------------------------------------------
# Module helpers
# ---------------------------------------------------------------------------


def _discover_videos(job_dir: Path) -> Iterable[Path]:
    """Yield paths to all supported video files in ``job_dir`` (non-recursive)."""
    return (p for p in job_dir.iterdir() if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS)


def _is_empty_dir(path: Path) -> bool:
    """Return ``True`` if ``path`` is a directory containing no entries."""
    try:
        next(path.iterdir())
    except StopIteration:
        return True
    return False


def _format_error(exc: BaseException) -> str:
    """Return a short ``"ExceptionType: message"`` string for the status file.

    Kept deliberately short so ``status.json`` does not balloon with
    multi-kilobyte tracebacks. Full tracebacks still land in the log via
    :meth:`logging.Logger.exception`.
    """
    message = str(exc) or "<no message>"
    return f"{type(exc).__name__}: {message}"
