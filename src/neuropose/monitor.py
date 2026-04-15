"""Localhost HTTP status monitor for the NeuroPose daemon.

Serves a small dashboard that collaborators on the same machine can
visit in a browser to watch a batch run make progress. The page reads
the persistent :class:`~neuropose.io.StatusFile` on every request — no
new state, no in-memory cache, no sync protocol with the daemon — so a
``neuropose serve`` process is trivially safe to run alongside
``neuropose watch`` and stays useful even if the daemon is down.

Two URLs are exposed:

- ``GET /`` returns a plain HTML page with a progress bar per job,
  an auto-refresh ``<meta>`` tag, and a stale-entry warning when a
  processing job hasn't checkpointed in a while.
- ``GET /status.json`` returns the raw validated
  :class:`~neuropose.io.StatusFile` as JSON, so any collaborator with
  ``curl`` (or a scripted pipeline) can consume the same data the
  browser sees. ``?job=<name>`` filters to a single entry.
- ``GET /health`` is a simple ``200 OK`` so external uptime
  checks can tell that the server process is running without parsing
  the HTML.

By default the server binds to ``127.0.0.1:8765``. It is **not**
exposed on any external interface by default — collaborators on the
same machine reach it directly, and collaborators elsewhere should go
through an SSH tunnel or explicitly configured reverse proxy. Binding
to ``0.0.0.0`` requires an explicit ``--host`` override, because
that is a real network-exposure decision the operator should make
with eyes open.

Dependencies
------------
Pure stdlib: :mod:`http.server`, :mod:`json`, :mod:`datetime`. No
FastAPI, no Flask, no tornado — this is a localhost tool and the
cost of a framework is not justified. Keeping it in stdlib also
means the monitor has zero runtime dependency surface that could
conflict with the rest of the project's pin.
"""

from __future__ import annotations

import html
import json
import logging
from datetime import UTC, datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlsplit

from neuropose.io import JobStatus, JobStatusEntry, StatusFile, load_status

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tunables
# ---------------------------------------------------------------------------

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765
HTML_REFRESH_SECONDS = 5
"""How often the HTML page auto-refreshes via ``<meta http-equiv>``."""

STALE_THRESHOLD_SECONDS = 60
"""A ``processing`` entry with ``last_update`` older than this many
seconds is flagged as stale in the HTML. 60 s is 20x the default
checkpoint cadence at 10 fps inference, so anything beyond it is very
likely a wedged daemon rather than normal jitter."""


# ---------------------------------------------------------------------------
# HTML rendering
# ---------------------------------------------------------------------------


_HTML_TEMPLATE = """\
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta http-equiv="refresh" content="{refresh}">
<title>NeuroPose job status</title>
<style>
  body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI",
                 "Helvetica Neue", Arial, sans-serif;
    max-width: 960px;
    margin: 2em auto;
    padding: 0 1em;
    color: #1a1a1a;
  }}
  h1 {{ margin-bottom: 0.25em; }}
  .subtitle {{ color: #555; margin-top: 0; font-size: 0.9em; }}
  .empty {{
    padding: 2em;
    background: #f6f6f6;
    border-radius: 8px;
    text-align: center;
    color: #555;
  }}
  table {{
    width: 100%;
    border-collapse: collapse;
    margin-top: 1em;
  }}
  th, td {{
    text-align: left;
    padding: 0.6em 0.8em;
    border-bottom: 1px solid #eee;
    vertical-align: top;
  }}
  th {{
    font-weight: 600;
    background: #fafafa;
    border-bottom: 2px solid #ccc;
  }}
  progress {{ width: 100%; height: 1em; }}
  .status-processing {{ color: #1765bd; font-weight: 600; }}
  .status-completed  {{ color: #2a8f3a; font-weight: 600; }}
  .status-failed     {{ color: #b93232; font-weight: 600; }}
  .stale {{
    display: inline-block;
    background: #fff3cd;
    color: #8a6d1b;
    padding: 0 0.4em;
    border-radius: 4px;
    font-size: 0.8em;
    margin-left: 0.4em;
  }}
  .error-msg {{
    font-family: ui-monospace, "SF Mono", Menlo, Consolas, monospace;
    font-size: 0.85em;
    color: #b93232;
    white-space: pre-wrap;
    word-break: break-word;
  }}
  .muted {{ color: #888; font-size: 0.85em; }}
</style>
</head>
<body>
<h1>NeuroPose job status</h1>
<p class="subtitle">
  Reading {status_path} &middot;
  as of {now_iso} &middot;
  auto-refresh every {refresh} s &middot;
  <a href="/status.json">status.json</a>
</p>
{body}
</body>
</html>
"""


def render_status_html(status: StatusFile, *, status_path: Path, now: datetime) -> str:
    """Render the full HTML dashboard for ``status``."""
    if status.is_empty():
        body = (
            '<div class="empty">No jobs tracked yet. '
            "Ingest a zip archive or drop a job directory into "
            "<code>$data_dir/in/</code> to get started.</div>"
        )
    else:
        rows = "\n".join(_render_row(name, entry, now=now) for name, entry in status.root.items())
        body = (
            "<table>\n"
            "<thead><tr>"
            "<th>Job</th>"
            "<th>Status</th>"
            "<th>Progress</th>"
            "<th>Current video</th>"
            "<th>Started</th>"
            "<th>Last update</th>"
            "</tr></thead>\n"
            f"<tbody>\n{rows}\n</tbody>\n"
            "</table>\n"
        )
    return _HTML_TEMPLATE.format(
        refresh=HTML_REFRESH_SECONDS,
        status_path=html.escape(str(status_path)),
        now_iso=html.escape(now.isoformat(timespec="seconds")),
        body=body,
    )


def _render_row(name: str, entry: JobStatusEntry, *, now: datetime) -> str:
    """Render one ``<tr>`` for a single job entry."""
    status_class = f"status-{entry.status.value}"
    status_cell = f'<span class="{status_class}">{html.escape(entry.status.value)}</span>'

    stale_badge = ""
    if entry.status == JobStatus.PROCESSING and entry.last_update is not None:
        age_seconds = (now - entry.last_update).total_seconds()
        if age_seconds > STALE_THRESHOLD_SECONDS:
            stale_badge = f'<span class="stale">stale — no update for {int(age_seconds)} s</span>'
    status_cell += stale_badge

    progress_cell = _render_progress_cell(entry)

    current_video = html.escape(entry.current_video or "—")
    if entry.videos_total is not None and entry.videos_completed is not None:
        current_video += (
            f' <span class="muted">({entry.videos_completed}/'
            f"{entry.videos_total} videos done)</span>"
        )

    started_cell = html.escape(entry.started_at.isoformat(timespec="seconds"))
    last_update_cell = (
        html.escape(entry.last_update.isoformat(timespec="seconds"))
        if entry.last_update is not None
        else '<span class="muted">—</span>'
    )

    error_row = ""
    if entry.status == JobStatus.FAILED and entry.error:
        error_row = (
            f'<tr><td colspan="6" class="error-msg">error: {html.escape(entry.error)}</td></tr>'
        )

    return (
        f"<tr>"
        f"<td>{html.escape(name)}</td>"
        f"<td>{status_cell}</td>"
        f"<td>{progress_cell}</td>"
        f"<td>{current_video}</td>"
        f"<td>{started_cell}</td>"
        f"<td>{last_update_cell}</td>"
        f"</tr>"
        f"{error_row}"
    )


def _render_progress_cell(entry: JobStatusEntry) -> str:
    """Render the progress-bar cell for one job."""
    if entry.status == JobStatus.COMPLETED:
        return '<progress value="100" max="100"></progress> 100%'
    if entry.status == JobStatus.FAILED:
        return '<span class="muted">—</span>'
    if entry.percent_complete is None:
        return '<progress max="100"></progress> <span class="muted">starting…</span>'
    pct = entry.percent_complete
    detail = ""
    if entry.frames_processed is not None and entry.frames_total:
        detail = (
            f' <span class="muted">'
            f"({entry.frames_processed}/{entry.frames_total} frames"
            f"{_render_eta(entry)})</span>"
        )
    return f'<progress value="{pct:.1f}" max="100"></progress> {pct:.1f}%{detail}'


def _render_eta(entry: JobStatusEntry) -> str:
    """Return an ``, ETA ~XYZ s`` suffix if ETA can be computed."""
    if entry.started_at is None or entry.percent_complete is None:
        return ""
    if entry.percent_complete <= 0.0 or entry.percent_complete >= 100.0:
        return ""
    now = datetime.now(UTC)
    elapsed = (now - entry.started_at).total_seconds()
    if elapsed <= 0.0:
        return ""
    fraction = entry.percent_complete / 100.0
    total_estimated = elapsed / fraction
    remaining = max(0.0, total_estimated - elapsed)
    return f", ETA ~{_format_duration(remaining)}"


def _format_duration(seconds: float) -> str:
    """Format a duration in seconds as ``HH:MM:SS`` or ``MM:SS`` or ``SS s``."""
    seconds_int = round(seconds)
    if seconds_int < 60:
        return f"{seconds_int} s"
    if seconds_int < 3600:
        m, s = divmod(seconds_int, 60)
        return f"{m}:{s:02d}"
    h, rem = divmod(seconds_int, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}"


# ---------------------------------------------------------------------------
# Request handler
# ---------------------------------------------------------------------------


class _StatusRequestHandler(BaseHTTPRequestHandler):
    """HTTP handler that serves the status dashboard and JSON.

    The status file path is injected via the server's ``status_path``
    attribute (see :func:`build_server`). Using a subclass + instance
    attribute avoids a module-level global and keeps the handler
    trivially parametrisable for tests.
    """

    # Silence the stdlib server's default stderr access log; route it
    # through the package logger instead so operators can tune it like
    # any other neuropose module.
    def log_message(self, format: str, *args: Any) -> None:
        logger.debug("%s - - %s", self.address_string(), format % args)

    def do_GET(self) -> None:
        parsed = urlsplit(self.path)
        path = parsed.path
        if path == "/":
            self._serve_html()
        elif path == "/status.json":
            self._serve_json(parse_qs(parsed.query))
        elif path == "/health":
            self._serve_health()
        else:
            self.send_error(HTTPStatus.NOT_FOUND, "unknown path")

    def _serve_html(self) -> None:
        status = self._load_status()
        body = render_status_html(
            status,
            status_path=self.server.status_path,  # type: ignore[attr-defined]
            now=datetime.now(UTC),
        ).encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _serve_json(self, query: dict[str, list[str]]) -> None:
        status = self._load_status()
        payload: Any = status.model_dump(mode="json")
        job_filter = query.get("job", [None])[0]
        if job_filter is not None:
            if job_filter not in status.root:
                self.send_error(
                    HTTPStatus.NOT_FOUND,
                    f"no such job: {job_filter}",
                )
                return
            payload = status.root[job_filter].model_dump(mode="json")
        body = json.dumps(payload, indent=2).encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _serve_health(self) -> None:
        body = b'{"status":"ok"}'
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _load_status(self) -> StatusFile:
        """Load the current status file.

        The file is read on every request — no in-memory cache — which
        is cheap in absolute terms (the file is small) and means the
        monitor never serves stale data relative to the daemon's most
        recent atomic write.
        """
        path: Path = self.server.status_path  # type: ignore[attr-defined]
        return load_status(path)


# ---------------------------------------------------------------------------
# Server construction and lifecycle
# ---------------------------------------------------------------------------


class _StatusServer(HTTPServer):
    """HTTPServer subclass that carries the status file path.

    The path is needed inside every request handler, and the stdlib
    server lets subclasses add arbitrary attributes — cleaner than a
    module-level global or a closure-captured handler class.
    """

    status_path: Path

    def __init__(
        self,
        server_address: tuple[str, int],
        status_path: Path,
    ) -> None:
        super().__init__(server_address, _StatusRequestHandler)
        self.status_path = status_path


def build_server(
    status_path: Path,
    *,
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
) -> _StatusServer:
    """Construct (but do not start) a status monitor HTTP server.

    Parameters
    ----------
    status_path
        Path to the daemon's ``status.json``. Typically
        ``Settings.status_file``.
    host
        Interface to bind. Defaults to ``127.0.0.1`` — explicitly
        loopback so an unconfigured monitor cannot be reached over the
        network without an operator opting in. Pass ``0.0.0.0`` or a
        specific external IP only when you have thought through the
        exposure.
    port
        TCP port to listen on. Defaults to 8765.

    Returns
    -------
    _StatusServer
        A ready-to-serve HTTP server. Call ``.serve_forever()`` to
        block, or ``.handle_request()`` once for tests.
    """
    return _StatusServer((host, port), status_path)


def serve_forever(
    status_path: Path,
    *,
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
) -> None:
    """Start the monitor and block until the process is interrupted.

    Logs the bound address once on startup so operators can copy it
    into a browser. Catches :class:`KeyboardInterrupt` and shuts the
    server down cleanly so Ctrl-C produces a quick, no-traceback exit.
    """
    server = build_server(status_path, host=host, port=port)
    logger.info(
        "NeuroPose monitor listening on http://%s:%d/ (reading %s)",
        host,
        port,
        status_path,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("Interrupt received; shutting down monitor")
    finally:
        server.server_close()
