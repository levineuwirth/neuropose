"""Pipeline-wide reset utility.

Tear down a running NeuroPose deployment back to a clean state: stop
the watch daemon and the monitor server, then wipe the job queue,
results, status file, and ingest staging directories. Intended for
the rapid iteration loop that comes up during benchmark and validation
work, where you want an empty ``$data_dir/in`` without manually
``rm -rf``-ing five separate paths and pkill'ing two processes.

The module is split into three independently-callable layers so each
piece is testable in isolation and reusable from non-CLI contexts:

- :func:`find_neuropose_processes` enumerates running ``neuropose
  watch`` and ``neuropose serve`` processes by scanning the OS process
  table. Pure read; no side effects.
- :func:`terminate_processes` signals the discovered processes (SIGINT
  first, optionally SIGKILL after a grace period) and reports
  survivors.
- :func:`wipe_state` removes the data-directory paths that the daemon
  and monitor produce. Idempotent; safe against a fresh install.

The top-level :func:`reset_pipeline` orchestrates all three and
returns a :class:`ResetReport` summarizing what happened.

Safety
------
:func:`reset_pipeline` refuses to wipe state while *any* discovered
process is still alive after the termination phase. Wiping
``$data_dir`` out from under an active daemon would leave the daemon
writing into deleted directory entries — a guaranteed mess. Callers
that hit a survivor must either raise the grace period, opt into
``force_kill=True`` (SIGKILL), or kill the survivor manually before
re-running.
"""

from __future__ import annotations

import contextlib
import logging
import os
import shutil
import signal
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import psutil

from neuropose.config import Settings
from neuropose.interfacer import LOCK_FILENAME

logger = logging.getLogger(__name__)


# Cmdline substrings that identify the daemon and monitor.
# Matched against the joined argv so ``uv run neuropose watch -v`` and
# ``python -m neuropose watch`` both hit. The substring choice
# deliberately includes the subcommand name so a generic
# ``neuropose --help`` shell doesn't get caught.
_DAEMON_MARKER = "neuropose watch"
_MONITOR_MARKER = "neuropose serve"

DEFAULT_GRACE_SECONDS = 10.0
"""How long :func:`terminate_processes` waits after SIGINT before
declaring a process a survivor (or escalating to SIGKILL when
``force_kill`` is set). Long enough for an idle daemon to finish its
current poll iteration; short enough that an interactive ``reset``
invocation doesn't feel hung. Override per-call when waiting on a
multi-minute inference."""

_POLL_INTERVAL_SECONDS = 0.2

ProcessRole = Literal["daemon", "monitor"]


@dataclass(frozen=True)
class RunningProcess:
    """A neuropose process discovered in the OS process table."""

    pid: int
    role: ProcessRole
    cmdline: str


@dataclass
class TerminationReport:
    """Outcome of trying to stop a set of running processes."""

    stopped: list[RunningProcess] = field(default_factory=list)
    survivors: list[RunningProcess] = field(default_factory=list)
    force_killed: list[RunningProcess] = field(default_factory=list)


@dataclass
class WipeReport:
    """Outcome of wiping data-directory state."""

    removed_paths: list[Path] = field(default_factory=list)
    bytes_freed: int = 0


@dataclass
class ResetReport:
    """Aggregate report from a full pipeline reset."""

    discovered: list[RunningProcess]
    termination: TerminationReport
    wipe: WipeReport
    dry_run: bool
    wipe_skipped_due_to_survivors: bool = False


def find_neuropose_processes(*, exclude_self: bool = True) -> list[RunningProcess]:
    """Scan the process table for ``neuropose watch`` / ``neuropose serve``.

    Parameters
    ----------
    exclude_self
        Skip the current process. The ``neuropose reset`` command
        itself has ``"neuropose"`` in its argv and would otherwise see
        itself in the result. Set to ``False`` only in tests where the
        caller has constructed a process table that should match
        verbatim.

    Returns
    -------
    list[RunningProcess]
        Processes whose joined argv contains either marker substring.
        Empty list when nothing matches.
    """
    self_pid = os.getpid()
    found: list[RunningProcess] = []
    for proc in psutil.process_iter(["pid", "cmdline"]):
        try:
            pid = proc.info["pid"]
            cmdline = proc.info["cmdline"] or []
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
        if exclude_self and pid == self_pid:
            continue
        joined = " ".join(cmdline)
        # Daemon check first because "neuropose serve" and
        # "neuropose watch" cannot both appear in a single process.
        if _DAEMON_MARKER in joined:
            found.append(RunningProcess(pid=pid, role="daemon", cmdline=joined))
        elif _MONITOR_MARKER in joined:
            found.append(RunningProcess(pid=pid, role="monitor", cmdline=joined))
    return found


def terminate_processes(
    processes: list[RunningProcess],
    *,
    grace_seconds: float = DEFAULT_GRACE_SECONDS,
    force_kill: bool = False,
) -> TerminationReport:
    """Send SIGINT to each process; optionally escalate to SIGKILL.

    Parameters
    ----------
    processes
        The processes to stop. Pass an empty list for a no-op.
    grace_seconds
        Maximum time to wait for processes to exit after SIGINT.
    force_kill
        When ``True``, any process still alive after ``grace_seconds``
        is sent SIGKILL. When ``False``, survivors are reported back
        to the caller untouched.

    Notes
    -----
    SIGTERM is *not* used as an intermediate escalation step. The
    interfacer's signal handler treats SIGINT and SIGTERM identically
    (both call :meth:`Interfacer.stop`), so SIGTERM accomplishes
    nothing that SIGINT did not already attempt. The only escalation
    that actually forces a stuck daemon to exit is SIGKILL, which
    bypasses the handler entirely.
    """
    report = TerminationReport()
    if not processes:
        return report

    for rp in processes:
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.kill(rp.pid, signal.SIGINT)
            logger.info("sent SIGINT to pid %d (%s)", rp.pid, rp.role)

    survivors = _wait_for_exit(processes, grace_seconds)
    stopped = [p for p in processes if p not in survivors]
    report.stopped.extend(stopped)

    if not survivors:
        return report

    if not force_kill:
        report.survivors.extend(survivors)
        return report

    for rp in survivors:
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.kill(rp.pid, signal.SIGKILL)
            logger.warning("escalated to SIGKILL for pid %d (%s)", rp.pid, rp.role)

    # SIGKILL is delivered synchronously enough that a short final
    # poll is sufficient — any remaining "survivor" at this point is
    # a permission error or a kernel-side hang, not graceful shutdown.
    final_survivors = _wait_for_exit(survivors, grace_seconds=2.0)
    killed = [p for p in survivors if p not in final_survivors]
    report.force_killed.extend(killed)
    report.survivors.extend(final_survivors)
    return report


def _wait_for_exit(
    processes: list[RunningProcess],
    grace_seconds: float,
) -> list[RunningProcess]:
    """Poll until every process exits or the deadline passes."""
    deadline = time.monotonic() + grace_seconds
    while time.monotonic() < deadline:
        survivors = [p for p in processes if _is_alive(p.pid)]
        if not survivors:
            return []
        time.sleep(_POLL_INTERVAL_SECONDS)
    return [p for p in processes if _is_alive(p.pid)]


def _is_alive(pid: int) -> bool:
    """Return ``True`` if ``pid`` is still running and not a zombie."""
    try:
        proc = psutil.Process(pid)
    except psutil.NoSuchProcess:
        return False
    try:
        return proc.status() != psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        return False


def wipe_state(
    settings: Settings,
    *,
    keep_failed: bool = False,
    dry_run: bool = False,
) -> WipeReport:
    """Remove data-directory paths produced by the daemon and monitor.

    Removes the *contents* of ``in/``, ``out/``, and (unless
    ``keep_failed`` is set) ``failed/``, plus the daemon lock file and
    any leftover ``.ingest_<uuid>/`` staging directories. The
    container directories themselves are preserved so the daemon does
    not need to recreate them on next startup.

    Parameters
    ----------
    settings
        Resolved :class:`~neuropose.config.Settings`. Determines all
        target paths via the ``input_dir`` / ``output_dir`` /
        ``failed_dir`` properties.
    keep_failed
        Preserve ``$data_dir/failed/`` for forensic review of past
        crashes. Default removes it along with the rest of the
        pipeline state.
    dry_run
        Compute the report without actually deleting anything. Useful
        for previewing the blast radius before confirming a reset.
    """
    report = WipeReport()

    targets: list[Path] = []
    if settings.input_dir.exists():
        targets.extend(settings.input_dir.iterdir())
    if settings.output_dir.exists():
        targets.extend(settings.output_dir.iterdir())
    if not keep_failed and settings.failed_dir.exists():
        targets.extend(settings.failed_dir.iterdir())

    lock_path = settings.data_dir / LOCK_FILENAME
    if lock_path.exists():
        targets.append(lock_path)

    if settings.data_dir.exists():
        targets.extend(settings.data_dir.glob(".ingest_*"))

    for target in targets:
        size = _path_size(target)
        if not dry_run:
            _remove(target)
        report.removed_paths.append(target)
        report.bytes_freed += size

    return report


def _path_size(path: Path) -> int:
    """Return the cumulative size of ``path``, recursing into directories."""
    if path.is_symlink() or path.is_file():
        try:
            return path.stat().st_size
        except OSError:
            return 0
    total = 0
    for sub in path.rglob("*"):
        try:
            if sub.is_file() and not sub.is_symlink():
                total += sub.stat().st_size
        except OSError:
            continue
    return total


def _remove(path: Path) -> None:
    """Remove ``path`` whether file, symlink, or directory."""
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    else:
        path.unlink()


def reset_pipeline(
    settings: Settings,
    *,
    grace_seconds: float = DEFAULT_GRACE_SECONDS,
    force_kill: bool = False,
    keep_failed: bool = False,
    dry_run: bool = False,
) -> ResetReport:
    """Stop daemon + monitor, then wipe pipeline state.

    Composes :func:`find_neuropose_processes`,
    :func:`terminate_processes`, and :func:`wipe_state` into a single
    operation. The wipe phase is *skipped* if any process survives
    termination — wiping ``$data_dir`` out from under an active
    daemon would corrupt its in-flight writes. The returned
    :class:`ResetReport` flags this case via
    ``wipe_skipped_due_to_survivors``.

    Parameters
    ----------
    settings
        Resolved :class:`~neuropose.config.Settings`.
    grace_seconds
        Maximum time to wait for SIGINT to take effect.
    force_kill
        Escalate to SIGKILL on any process still alive after
        ``grace_seconds``. Necessary when the daemon is mid-inference
        on a long video and you do not want to wait for the current
        video to finish.
    keep_failed
        Preserve ``$data_dir/failed/`` during the wipe.
    dry_run
        Discover and report without killing anything or removing any
        paths. Termination phase is skipped entirely.
    """
    discovered = find_neuropose_processes()

    if dry_run:
        wipe = wipe_state(settings, keep_failed=keep_failed, dry_run=True)
        return ResetReport(
            discovered=discovered,
            termination=TerminationReport(),
            wipe=wipe,
            dry_run=True,
        )

    termination = terminate_processes(
        discovered,
        grace_seconds=grace_seconds,
        force_kill=force_kill,
    )

    if termination.survivors:
        return ResetReport(
            discovered=discovered,
            termination=termination,
            wipe=WipeReport(),
            dry_run=False,
            wipe_skipped_due_to_survivors=True,
        )

    wipe = wipe_state(settings, keep_failed=keep_failed, dry_run=False)
    return ResetReport(
        discovered=discovered,
        termination=termination,
        wipe=wipe,
        dry_run=False,
    )
