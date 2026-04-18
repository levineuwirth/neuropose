"""Tests for :mod:`neuropose.reset` and the ``neuropose reset`` CLI command.

Coverage:

- :func:`find_neuropose_processes` filters by cmdline marker, classifies
  daemon vs monitor, and excludes the calling process.
- :func:`terminate_processes` sends SIGINT, escalates to SIGKILL when
  asked, and reports survivors.
- :func:`wipe_state` removes contents of in/, out/, failed/, the lock
  file, and ``.ingest_*`` staging dirs; honors ``keep_failed`` and
  ``dry_run``.
- :func:`reset_pipeline` skips the wipe phase when termination leaves
  survivors.
- The ``neuropose reset`` CLI command renders previews, honors
  ``--dry-run`` and ``--yes``, and exits non-zero on survivors.

The process-killing tests use monkeypatched ``psutil.process_iter``
and ``os.kill`` so the suite never touches the real process table or
sends real signals.
"""

from __future__ import annotations

import signal
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from neuropose.cli import EXIT_OK, EXIT_USAGE, app
from neuropose.config import Settings
from neuropose.interfacer import LOCK_FILENAME
from neuropose.reset import (
    DEFAULT_GRACE_SECONDS,
    RunningProcess,
    TerminationReport,
    WipeReport,
    find_neuropose_processes,
    reset_pipeline,
    terminate_processes,
    wipe_state,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _FakeProc:
    """Minimal stand-in for ``psutil.Process`` for the discovery tests."""

    def __init__(self, pid: int, cmdline: list[str]) -> None:
        self.info = {"pid": pid, "cmdline": cmdline}


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    """A Settings pointing at an isolated temp data dir, with subdirs created."""
    s = Settings(data_dir=tmp_path / "jobs", model_cache_dir=tmp_path / "models")
    s.ensure_dirs()
    return s


# ---------------------------------------------------------------------------
# find_neuropose_processes
# ---------------------------------------------------------------------------


class TestFindNeuroposeProcesses:
    def test_classifies_watch_as_daemon(self, monkeypatch: pytest.MonkeyPatch) -> None:
        procs = [_FakeProc(1234, ["python", "-m", "neuropose", "watch"])]
        monkeypatch.setattr("psutil.process_iter", lambda attrs: iter(procs))
        found = find_neuropose_processes()
        assert len(found) == 1
        assert found[0].pid == 1234
        assert found[0].role == "daemon"

    def test_classifies_serve_as_monitor(self, monkeypatch: pytest.MonkeyPatch) -> None:
        procs = [_FakeProc(5678, ["uv", "run", "neuropose", "serve", "--port", "8765"])]
        monkeypatch.setattr("psutil.process_iter", lambda attrs: iter(procs))
        found = find_neuropose_processes()
        assert len(found) == 1
        assert found[0].role == "monitor"

    def test_ignores_unrelated_processes(self, monkeypatch: pytest.MonkeyPatch) -> None:
        procs = [
            _FakeProc(1, ["bash"]),
            _FakeProc(2, ["python", "-m", "pip", "install", "neuropose"]),
            _FakeProc(3, ["neuropose", "--help"]),
        ]
        monkeypatch.setattr("psutil.process_iter", lambda attrs: iter(procs))
        assert find_neuropose_processes() == []

    def test_excludes_self(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import os

        self_pid = os.getpid()
        procs = [
            _FakeProc(self_pid, ["python", "-m", "neuropose", "watch"]),
            _FakeProc(9999, ["python", "-m", "neuropose", "watch"]),
        ]
        monkeypatch.setattr("psutil.process_iter", lambda attrs: iter(procs))
        found = find_neuropose_processes()
        assert [rp.pid for rp in found] == [9999]

    def test_includes_self_when_disabled(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import os

        self_pid = os.getpid()
        procs = [_FakeProc(self_pid, ["python", "-m", "neuropose", "watch"])]
        monkeypatch.setattr("psutil.process_iter", lambda attrs: iter(procs))
        found = find_neuropose_processes(exclude_self=False)
        assert [rp.pid for rp in found] == [self_pid]

    def test_handles_dead_processes(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A NoSuchProcess raised mid-iteration must not crash the scan."""
        import psutil

        class _RaisingProc:
            @property
            def info(self) -> dict[str, Any]:
                raise psutil.NoSuchProcess(pid=1)

        procs = [
            _RaisingProc(),
            _FakeProc(2, ["python", "-m", "neuropose", "watch"]),
        ]
        monkeypatch.setattr("psutil.process_iter", lambda attrs: iter(procs))
        found = find_neuropose_processes()
        assert [rp.pid for rp in found] == [2]


# ---------------------------------------------------------------------------
# terminate_processes
# ---------------------------------------------------------------------------


class TestTerminateProcesses:
    def test_empty_list_is_noop(self) -> None:
        report = terminate_processes([])
        assert report.stopped == []
        assert report.survivors == []
        assert report.force_killed == []

    def test_sigint_to_each_process(self, monkeypatch: pytest.MonkeyPatch) -> None:
        sent: list[tuple[int, int]] = []
        monkeypatch.setattr("os.kill", lambda pid, sig: sent.append((pid, sig)))
        # Mark all processes immediately dead so the wait loop exits fast.
        monkeypatch.setattr("neuropose.reset._is_alive", lambda pid: False)

        rps = [
            RunningProcess(pid=10, role="daemon", cmdline="x"),
            RunningProcess(pid=20, role="monitor", cmdline="y"),
        ]
        report = terminate_processes(rps, grace_seconds=0.0)
        assert sent == [(10, signal.SIGINT), (20, signal.SIGINT)]
        assert {p.pid for p in report.stopped} == {10, 20}
        assert report.survivors == []

    def test_survivors_reported_when_force_kill_off(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("os.kill", lambda pid, sig: None)
        # Process 10 always alive; process 20 dies after SIGINT.
        alive = {10}
        monkeypatch.setattr("neuropose.reset._is_alive", lambda pid: pid in alive)

        rps = [
            RunningProcess(pid=10, role="daemon", cmdline="x"),
            RunningProcess(pid=20, role="monitor", cmdline="y"),
        ]
        # Drop pid 20 so it appears dead immediately.
        alive.discard(20)
        report = terminate_processes(rps, grace_seconds=0.0, force_kill=False)
        assert {p.pid for p in report.stopped} == {20}
        assert {p.pid for p in report.survivors} == {10}
        assert report.force_killed == []

    def test_force_kill_escalates_to_sigkill(self, monkeypatch: pytest.MonkeyPatch) -> None:
        sent: list[tuple[int, int]] = []
        monkeypatch.setattr("os.kill", lambda pid, sig: sent.append((pid, sig)))

        # Start alive; SIGKILL "kills" by toggling the flag from inside _is_alive.
        alive = {10}

        def fake_is_alive(pid: int) -> bool:
            if (pid, signal.SIGKILL) in sent:
                return False
            return pid in alive

        monkeypatch.setattr("neuropose.reset._is_alive", fake_is_alive)

        rp = RunningProcess(pid=10, role="daemon", cmdline="x")
        report = terminate_processes([rp], grace_seconds=0.0, force_kill=True)
        assert (10, signal.SIGINT) in sent
        assert (10, signal.SIGKILL) in sent
        assert {p.pid for p in report.force_killed} == {10}
        assert report.survivors == []


# ---------------------------------------------------------------------------
# wipe_state
# ---------------------------------------------------------------------------


class TestWipeState:
    def test_no_op_on_empty_dirs(self, settings: Settings) -> None:
        report = wipe_state(settings)
        assert report.removed_paths == []
        assert report.bytes_freed == 0

    def test_removes_in_out_failed_contents(self, settings: Settings) -> None:
        (settings.input_dir / "job_a").mkdir()
        (settings.input_dir / "job_a" / "video.mp4").write_bytes(b"x" * 100)
        (settings.output_dir / "status.json").write_text("{}")
        (settings.failed_dir / "job_b").mkdir()

        report = wipe_state(settings)
        names = {p.name for p in report.removed_paths}
        assert names == {"job_a", "status.json", "job_b"}
        # Containers themselves preserved.
        assert settings.input_dir.exists()
        assert settings.output_dir.exists()
        assert settings.failed_dir.exists()

    def test_keep_failed_preserves_failed_contents(self, settings: Settings) -> None:
        (settings.input_dir / "job_a").mkdir()
        (settings.failed_dir / "job_b").mkdir()
        (settings.failed_dir / "job_b" / "evidence.log").write_text("crash")

        report = wipe_state(settings, keep_failed=True)
        names = {p.name for p in report.removed_paths}
        assert "job_a" in names
        assert "job_b" not in names
        assert (settings.failed_dir / "job_b" / "evidence.log").exists()

    def test_removes_lock_file(self, settings: Settings) -> None:
        (settings.data_dir / LOCK_FILENAME).write_text("12345\n")
        report = wipe_state(settings)
        assert (settings.data_dir / LOCK_FILENAME) in report.removed_paths
        assert not (settings.data_dir / LOCK_FILENAME).exists()

    def test_removes_ingest_staging_dirs(self, settings: Settings) -> None:
        staging_a = settings.data_dir / ".ingest_abc123"
        staging_b = settings.data_dir / ".ingest_def456"
        staging_a.mkdir()
        staging_b.mkdir()
        (staging_a / "leftover.mp4").write_bytes(b"y" * 50)

        report = wipe_state(settings)
        assert staging_a in report.removed_paths
        assert staging_b in report.removed_paths
        assert not staging_a.exists()
        assert not staging_b.exists()

    def test_dry_run_reports_without_removing(self, settings: Settings) -> None:
        (settings.input_dir / "job_a").mkdir()
        (settings.input_dir / "job_a" / "video.mp4").write_bytes(b"z" * 200)

        report = wipe_state(settings, dry_run=True)
        assert len(report.removed_paths) == 1
        assert report.bytes_freed == 200
        # Nothing actually deleted.
        assert (settings.input_dir / "job_a" / "video.mp4").exists()

    def test_bytes_freed_recurses_into_subdirs(self, settings: Settings) -> None:
        job = settings.input_dir / "job_a"
        job.mkdir()
        (job / "a.mp4").write_bytes(b"a" * 100)
        (job / "nested").mkdir()
        (job / "nested" / "b.mp4").write_bytes(b"b" * 250)

        report = wipe_state(settings, dry_run=True)
        assert report.bytes_freed == 350


# ---------------------------------------------------------------------------
# reset_pipeline
# ---------------------------------------------------------------------------


class TestResetPipeline:
    def test_dry_run_skips_termination(
        self,
        settings: Settings,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        rp = RunningProcess(pid=10, role="daemon", cmdline="x")
        monkeypatch.setattr("neuropose.reset.find_neuropose_processes", lambda: [rp])

        # Sentinel to detect termination calls.
        def _should_not_be_called(*args: Any, **kwargs: Any) -> TerminationReport:
            raise AssertionError("dry_run must not invoke terminate_processes")

        monkeypatch.setattr("neuropose.reset.terminate_processes", _should_not_be_called)

        report = reset_pipeline(settings, dry_run=True)
        assert report.dry_run is True
        assert report.discovered == [rp]
        assert report.termination.stopped == []

    def test_skips_wipe_when_survivors_remain(
        self,
        settings: Settings,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        rp = RunningProcess(pid=10, role="daemon", cmdline="x")
        monkeypatch.setattr("neuropose.reset.find_neuropose_processes", lambda: [rp])
        monkeypatch.setattr(
            "neuropose.reset.terminate_processes",
            lambda procs, **_: TerminationReport(survivors=list(procs)),
        )

        # Seed something to wipe so we can confirm it's untouched.
        (settings.input_dir / "job_a").mkdir()

        report = reset_pipeline(settings, dry_run=False)
        assert report.wipe_skipped_due_to_survivors is True
        assert report.wipe.removed_paths == []
        assert (settings.input_dir / "job_a").exists()

    def test_wipes_when_all_processes_stopped(
        self,
        settings: Settings,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        rp = RunningProcess(pid=10, role="daemon", cmdline="x")
        monkeypatch.setattr("neuropose.reset.find_neuropose_processes", lambda: [rp])
        monkeypatch.setattr(
            "neuropose.reset.terminate_processes",
            lambda procs, **_: TerminationReport(stopped=list(procs)),
        )

        (settings.input_dir / "job_a").mkdir()

        report = reset_pipeline(settings, dry_run=False)
        assert report.wipe_skipped_due_to_survivors is False
        assert any(p.name == "job_a" for p in report.wipe.removed_paths)
        assert not (settings.input_dir / "job_a").exists()


# ---------------------------------------------------------------------------
# CLI: neuropose reset
# ---------------------------------------------------------------------------


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


@pytest.fixture
def env_data_dir(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Point NEUROPOSE_DATA_DIR at an isolated temp dir for CLI tests."""
    data_dir = tmp_path / "jobs"
    data_dir.mkdir()
    (data_dir / "in").mkdir()
    (data_dir / "out").mkdir()
    (data_dir / "failed").mkdir()
    monkeypatch.setenv("NEUROPOSE_DATA_DIR", str(data_dir))
    return data_dir


class TestResetCli:
    def test_reset_dry_run_does_not_modify(
        self,
        runner: CliRunner,
        env_data_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        (env_data_dir / "in" / "job_a").mkdir()
        monkeypatch.setattr("neuropose.reset.find_neuropose_processes", list)

        result = runner.invoke(app, ["reset", "--dry-run"])
        assert result.exit_code == EXIT_OK, result.output
        assert "would remove" in result.output
        assert "(dry-run; no changes made)" in result.output
        assert (env_data_dir / "in" / "job_a").exists()

    def test_reset_yes_skips_confirmation_and_wipes(
        self,
        runner: CliRunner,
        env_data_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        (env_data_dir / "in" / "job_a").mkdir()
        (env_data_dir / "in" / "job_a" / "video.mp4").write_bytes(b"x" * 100)
        monkeypatch.setattr("neuropose.reset.find_neuropose_processes", list)

        result = runner.invoke(app, ["reset", "--yes"])
        assert result.exit_code == EXIT_OK, result.output
        assert "removed" in result.output
        assert not (env_data_dir / "in" / "job_a").exists()

    def test_reset_aborts_on_no_confirmation(
        self,
        runner: CliRunner,
        env_data_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        (env_data_dir / "in" / "job_a").mkdir()
        monkeypatch.setattr("neuropose.reset.find_neuropose_processes", list)

        # typer.confirm reads from stdin; "n\n" declines.
        result = runner.invoke(app, ["reset"], input="n\n")
        assert result.exit_code == EXIT_USAGE, result.output
        assert "aborted" in result.output
        assert (env_data_dir / "in" / "job_a").exists()

    def test_reset_clean_state_is_noop(
        self,
        runner: CliRunner,
        env_data_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        del env_data_dir
        monkeypatch.setattr("neuropose.reset.find_neuropose_processes", list)

        result = runner.invoke(app, ["reset"])
        assert result.exit_code == EXIT_OK, result.output
        assert "nothing to do" in result.output

    def test_reset_reports_survivors_with_nonzero_exit(
        self,
        runner: CliRunner,
        env_data_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        del env_data_dir
        rp = RunningProcess(pid=42, role="daemon", cmdline="neuropose watch")
        monkeypatch.setattr("neuropose.reset.find_neuropose_processes", lambda: [rp])
        monkeypatch.setattr(
            "neuropose.reset.terminate_processes",
            lambda procs, **_: TerminationReport(survivors=list(procs)),
        )

        result = runner.invoke(app, ["reset", "--yes"])
        assert result.exit_code == EXIT_USAGE, result.output
        assert "did not exit" in result.output
        assert "pid 42" in result.output
        assert "--force-kill" in result.output


def test_default_grace_seconds_constant_is_reasonable() -> None:
    """Lock the default grace period so a refactor cannot silently lower it."""
    assert 5.0 <= DEFAULT_GRACE_SECONDS <= 60.0


def test_wipe_report_default_construction() -> None:
    r = WipeReport()
    assert r.removed_paths == []
    assert r.bytes_freed == 0
