"""Tests for :mod:`neuropose.monitor`.

The tests boot the real :class:`http.server.HTTPServer` subclass on a
free ephemeral port in a background thread, issue real HTTP requests
with :mod:`urllib.request`, and assert on the responses. This
exercises the handler, routing, JSON serialization, HTML rendering,
and the query-parameter filter end-to-end — exactly the surface
collaborators will hit.
"""

from __future__ import annotations

import json
import socket
import threading
import time
import urllib.error
import urllib.request
from datetime import UTC, datetime, timedelta
from http.server import HTTPServer
from pathlib import Path

import pytest

from neuropose.io import (
    JobStatus,
    JobStatusEntry,
    StatusFile,
    save_status,
)
from neuropose.monitor import (
    STALE_THRESHOLD_SECONDS,
    build_server,
    render_status_html,
    serve_forever,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _free_port() -> int:
    """Grab an unused TCP port and release it for the next caller.

    ``HTTPServer`` does not offer a first-class "bind to an ephemeral
    port" API that also exposes the chosen port to the caller before
    serving. Doing a separate SO_REUSEADDR probe to pick one is
    reliable enough for a test-only fixture.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


@pytest.fixture
def status_path(tmp_path: Path) -> Path:
    """Return a path to a populated status.json for the monitor tests."""
    path = tmp_path / "status.json"
    now = datetime.now(UTC)
    status = StatusFile(
        root={
            "job_running": JobStatusEntry(
                status=JobStatus.PROCESSING,
                started_at=now - timedelta(minutes=2),
                current_video="trial_01.mp4",
                frames_processed=450,
                frames_total=1200,
                videos_completed=0,
                videos_total=3,
                percent_complete=12.5,
                last_update=now,
            ),
            "job_done": JobStatusEntry(
                status=JobStatus.COMPLETED,
                started_at=now - timedelta(minutes=10),
                completed_at=now - timedelta(minutes=5),
                results_path=Path("/tmp/out/job_done/results.json"),
                videos_completed=2,
                videos_total=2,
                percent_complete=100.0,
                last_update=now - timedelta(minutes=5),
            ),
            "job_dead": JobStatusEntry(
                status=JobStatus.FAILED,
                started_at=now - timedelta(minutes=6),
                completed_at=now - timedelta(minutes=5),
                error="VideoDecodeError: OpenCV could not open video",
            ),
        }
    )
    save_status(path, status)
    return path


@pytest.fixture
def running_server(status_path: Path):
    """Boot the monitor in a background thread, yield the base URL."""
    port = _free_port()
    server = build_server(status_path, host="127.0.0.1", port=port)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    # Tiny settle delay so the first request doesn't race with
    # serve_forever's setup. The stdlib server is synchronous enough
    # that a few ms is plenty.
    time.sleep(0.05)
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def _get(url: str, *, timeout: float = 2.0) -> tuple[int, bytes, dict[str, str]]:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return resp.status, resp.read(), dict(resp.headers)
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read(), dict(exc.headers) if exc.headers else {}


# ---------------------------------------------------------------------------
# HTML rendering (pure function, no server)
# ---------------------------------------------------------------------------


class TestRenderStatusHtml:
    def test_empty_status_shows_empty_state(self, tmp_path: Path) -> None:
        html_text = render_status_html(
            StatusFile(root={}),
            status_path=tmp_path / "status.json",
            now=datetime.now(UTC),
        )
        assert "No jobs tracked yet" in html_text
        assert "auto-refresh" in html_text

    def test_processing_entry_renders_progress_bar(self, status_path: Path) -> None:
        from neuropose.io import load_status

        status = load_status(status_path)
        html_text = render_status_html(
            status,
            status_path=status_path,
            now=datetime.now(UTC),
        )
        assert "job_running" in html_text
        assert "<progress" in html_text
        assert "12.5" in html_text
        assert "trial_01.mp4" in html_text

    def test_completed_entry_shows_100_percent(self, status_path: Path) -> None:
        from neuropose.io import load_status

        status = load_status(status_path)
        html_text = render_status_html(
            status,
            status_path=status_path,
            now=datetime.now(UTC),
        )
        assert "job_done" in html_text
        assert "100" in html_text

    def test_failed_entry_shows_error_message(self, status_path: Path) -> None:
        from neuropose.io import load_status

        status = load_status(status_path)
        html_text = render_status_html(
            status,
            status_path=status_path,
            now=datetime.now(UTC),
        )
        assert "job_dead" in html_text
        assert "VideoDecodeError" in html_text

    def test_error_message_is_html_escaped(self, tmp_path: Path) -> None:
        now = datetime.now(UTC)
        status = StatusFile(
            root={
                "x": JobStatusEntry(
                    status=JobStatus.FAILED,
                    started_at=now,
                    completed_at=now,
                    error="<script>alert('xss')</script>",
                )
            }
        )
        html_text = render_status_html(
            status,
            status_path=tmp_path / "status.json",
            now=now,
        )
        assert "<script>" not in html_text
        assert "&lt;script&gt;" in html_text

    def test_stale_processing_entry_gets_badge(self, tmp_path: Path) -> None:
        now = datetime.now(UTC)
        stale_update = now - timedelta(seconds=STALE_THRESHOLD_SECONDS + 30)
        status = StatusFile(
            root={
                "stuck": JobStatusEntry(
                    status=JobStatus.PROCESSING,
                    started_at=now - timedelta(hours=1),
                    current_video="video.mp4",
                    frames_processed=10,
                    frames_total=100,
                    videos_completed=0,
                    videos_total=1,
                    percent_complete=10.0,
                    last_update=stale_update,
                ),
            }
        )
        html_text = render_status_html(status, status_path=tmp_path / "status.json", now=now)
        # The badge text is "stale — no update for X s". The class
        # 'stale' also appears in the inline <style> block, so we
        # match the human-readable badge text to distinguish.
        assert "no update for" in html_text

    def test_fresh_processing_entry_has_no_stale_badge(self, tmp_path: Path) -> None:
        now = datetime.now(UTC)
        status = StatusFile(
            root={
                "ok": JobStatusEntry(
                    status=JobStatus.PROCESSING,
                    started_at=now - timedelta(seconds=5),
                    current_video="x.mp4",
                    frames_processed=5,
                    frames_total=100,
                    videos_completed=0,
                    videos_total=1,
                    percent_complete=5.0,
                    last_update=now,
                ),
            }
        )
        html_text = render_status_html(status, status_path=tmp_path / "status.json", now=now)
        # Same logic: the CSS class is always present, so we match on
        # the badge text which is only injected when the entry is
        # actually stale.
        assert "no update for" not in html_text


# ---------------------------------------------------------------------------
# HTTP server
# ---------------------------------------------------------------------------


class TestHtmlRoute:
    def test_root_returns_html(self, running_server: str) -> None:
        status, body, headers = _get(f"{running_server}/")
        assert status == 200
        assert "text/html" in headers.get("Content-Type", "")
        text = body.decode()
        assert "NeuroPose job status" in text
        assert "job_running" in text
        assert "<progress" in text

    def test_root_has_no_store_cache_control(self, running_server: str) -> None:
        _, _, headers = _get(f"{running_server}/")
        assert "no-store" in headers.get("Cache-Control", "")


class TestJsonRoute:
    def test_status_json_returns_all_entries(self, running_server: str) -> None:
        status, body, headers = _get(f"{running_server}/status.json")
        assert status == 200
        assert "application/json" in headers.get("Content-Type", "")
        data = json.loads(body)
        assert set(data.keys()) == {"job_running", "job_done", "job_dead"}

    def test_status_json_filter_returns_single_entry(self, running_server: str) -> None:
        status, body, _ = _get(f"{running_server}/status.json?job=job_running")
        assert status == 200
        data = json.loads(body)
        assert data["status"] == "processing"
        assert data["current_video"] == "trial_01.mp4"
        assert data["percent_complete"] == 12.5

    def test_status_json_filter_unknown_job_is_404(self, running_server: str) -> None:
        status, _, _ = _get(f"{running_server}/status.json?job=nope")
        assert status == 404


class TestHealthRoute:
    def test_health_returns_ok(self, running_server: str) -> None:
        status, body, _ = _get(f"{running_server}/health")
        assert status == 200
        assert json.loads(body) == {"status": "ok"}


class TestUnknownRoutes:
    def test_unknown_path_is_404(self, running_server: str) -> None:
        status, _, _ = _get(f"{running_server}/wat")
        assert status == 404


# ---------------------------------------------------------------------------
# Monitor survives a missing status file
# ---------------------------------------------------------------------------


class TestMissingStatusFile:
    def test_monitor_reads_missing_file_as_empty(self, tmp_path: Path) -> None:
        port = _free_port()
        path = tmp_path / "status.json"  # deliberately not created
        server = build_server(path, host="127.0.0.1", port=port)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        time.sleep(0.05)
        try:
            status, body, _ = _get(f"http://127.0.0.1:{port}/status.json")
            assert status == 200
            assert json.loads(body) == {}
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)


# ---------------------------------------------------------------------------
# serve_forever wrapper
# ---------------------------------------------------------------------------


class TestServeForever:
    def test_serve_forever_shuts_down_on_keyboard_interrupt(
        self,
        status_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """``serve_forever`` should catch KeyboardInterrupt and clean up.

        We monkeypatch the stdlib HTTPServer's ``serve_forever`` to
        immediately raise, which simulates a Ctrl-C before any request
        is accepted. The wrapper should swallow it, log, and close the
        server.
        """

        def fake_serve_forever(self) -> None:
            raise KeyboardInterrupt

        monkeypatch.setattr(HTTPServer, "serve_forever", fake_serve_forever)
        # Should return cleanly, not propagate the KeyboardInterrupt.
        serve_forever(status_path, host="127.0.0.1", port=_free_port())
