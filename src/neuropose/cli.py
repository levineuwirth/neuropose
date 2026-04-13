"""NeuroPose command-line interface.

Three subcommands:

- ``neuropose watch`` — run the :class:`~neuropose.interfacer.Interfacer`
  daemon against the configured input directory.
- ``neuropose process <video>`` — run the estimator on a single video and
  write the predictions JSON to disk.
- ``neuropose analyze <results>`` — stubbed placeholder pending the
  analyzer rewrite in commit 10.

User-facing error handling
--------------------------
This module takes responsibility for turning internal exceptions into
clear, short messages on stderr and non-zero exit codes. Users of the CLI
should never see a raw Python traceback for expected failure modes:

===============================  ===============  ==========================
Exception                        Exit code        User-facing message
===============================  ===============  ==========================
``FileNotFoundError`` (config)   2                "config file not found: ..."
``ValidationError`` (config)     2                "invalid config: ..."
``AlreadyRunningError``          2                "another daemon is running"
``NotImplementedError``          3                "pending commit N: ..."
``KeyboardInterrupt``            130              (silent, matches shell)
===============================  ===============  ==========================

Internal errors (bugs) still surface as tracebacks — we only catch the
exception classes we expect from the layers below.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Annotated

import typer
from pydantic import ValidationError

from neuropose import __version__
from neuropose.config import Settings
from neuropose.estimator import Estimator
from neuropose.interfacer import AlreadyRunningError, Interfacer
from neuropose.io import save_video_predictions

logger = logging.getLogger(__name__)

# Exit codes. Kept as module constants so tests can import and compare.
EXIT_OK = 0
EXIT_USAGE = 2
EXIT_PENDING = 3
EXIT_INTERRUPTED = 130

app = typer.Typer(
    name="neuropose",
    help="NeuroPose — 3D human pose estimation pipeline.",
    no_args_is_help=True,
    add_completion=False,
)


# ---------------------------------------------------------------------------
# Global callback
# ---------------------------------------------------------------------------


def _version_callback(value: bool) -> None:
    """Print the package version and exit, when ``--version`` is given."""
    if value:
        typer.echo(f"neuropose {__version__}")
        raise typer.Exit()


def _configure_logging(verbose: bool, quiet: bool) -> None:
    """Set up the root logger based on ``--verbose``/``--quiet`` flags."""
    if verbose and quiet:
        raise typer.BadParameter("--verbose and --quiet are mutually exclusive")
    if verbose:
        level = logging.DEBUG
    elif quiet:
        level = logging.WARNING
    else:
        level = logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )


def _load_settings(config: Path | None) -> Settings:
    """Load settings from ``config`` or from env vars / defaults."""
    if config is None:
        try:
            return Settings()
        except ValidationError as exc:
            typer.echo(f"error: invalid environment configuration: {exc}", err=True)
            raise typer.Exit(code=EXIT_USAGE) from exc
    try:
        return Settings.from_yaml(config)
    except FileNotFoundError as exc:
        typer.echo(f"error: config file not found: {config}", err=True)
        raise typer.Exit(code=EXIT_USAGE) from exc
    except ValidationError as exc:
        typer.echo(f"error: invalid config {config}: {exc}", err=True)
        raise typer.Exit(code=EXIT_USAGE) from exc
    except ValueError as exc:
        typer.echo(f"error: invalid config {config}: {exc}", err=True)
        raise typer.Exit(code=EXIT_USAGE) from exc


@app.callback()
def main(
    ctx: typer.Context,
    config: Annotated[
        Path | None,
        typer.Option(
            "--config",
            "-c",
            help="Path to a YAML configuration file. If omitted, configuration "
            "is read from NEUROPOSE_* environment variables and defaults.",
            exists=False,
            dir_okay=False,
            readable=True,
        ),
    ] = None,
    verbose: Annotated[
        bool,
        typer.Option("--verbose", "-v", help="Enable debug logging."),
    ] = False,
    quiet: Annotated[
        bool,
        typer.Option("--quiet", "-q", help="Suppress info logging."),
    ] = False,
    version: Annotated[
        bool,
        typer.Option(
            "--version",
            help="Show the package version and exit.",
            is_eager=True,
            callback=_version_callback,
        ),
    ] = False,
) -> None:
    """NeuroPose — 3D human pose estimation pipeline."""
    del version  # Handled eagerly in _version_callback.
    _configure_logging(verbose, quiet)
    ctx.obj = _load_settings(config)


# ---------------------------------------------------------------------------
# watch
# ---------------------------------------------------------------------------


@app.command()
def watch(ctx: typer.Context) -> None:
    """Run the NeuroPose job-processing daemon.

    Watches the configured input directory for new job subdirectories,
    processes each one in order, and persists status to disk. Blocks until
    SIGINT, SIGTERM, or another instance of the daemon takes over.
    """
    settings: Settings = ctx.obj
    estimator = Estimator(
        device=settings.device,
        default_fov_degrees=settings.default_fov_degrees,
    )
    interfacer = Interfacer(settings=settings, estimator=estimator)
    try:
        interfacer.run()
    except AlreadyRunningError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=EXIT_USAGE) from exc
    except NotImplementedError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=EXIT_PENDING) from exc
    except KeyboardInterrupt as exc:
        raise typer.Exit(code=EXIT_INTERRUPTED) from exc


# ---------------------------------------------------------------------------
# process
# ---------------------------------------------------------------------------


@app.command()
def process(
    ctx: typer.Context,
    video: Annotated[
        Path,
        typer.Argument(
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            help="Path to an input video file (.mp4, .avi, .mov, .mkv).",
        ),
    ],
    output: Annotated[
        Path | None,
        typer.Option(
            "--output",
            "-o",
            help="Where to write the predictions JSON. Defaults to "
            "<video-stem>_predictions.json in the current working directory.",
            dir_okay=False,
            writable=True,
        ),
    ] = None,
) -> None:
    """Run pose estimation on a single video and write the JSON result.

    Convenience entry point for one-off processing outside the daemon
    workflow. The ``watch`` subcommand is the right choice for batch
    processing of job directories.
    """
    settings: Settings = ctx.obj
    estimator = Estimator(
        device=settings.device,
        default_fov_degrees=settings.default_fov_degrees,
    )
    try:
        estimator.load_model(cache_dir=settings.model_cache_dir)
    except NotImplementedError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=EXIT_PENDING) from exc
    result = estimator.process_video(video)
    out_path = output if output is not None else Path.cwd() / f"{video.stem}_predictions.json"
    save_video_predictions(out_path, result.predictions)
    typer.echo(f"wrote {out_path} ({result.frame_count} frames)")


# ---------------------------------------------------------------------------
# analyze (stub)
# ---------------------------------------------------------------------------


@app.command()
def analyze(
    ctx: typer.Context,
    results: Annotated[
        Path,
        typer.Argument(help="Path to a results.json produced by watch or process."),
    ],
) -> None:
    """Run the analyzer subpackage against a results.json (pending commit 10)."""
    del ctx, results
    typer.echo(
        "error: the analyzer subpackage is pending commit 10. "
        "Until it lands, use neuropose.io to load results.json from Python.",
        err=True,
    )
    raise typer.Exit(code=EXIT_PENDING)


def run() -> None:
    """Entry point referenced by ``pyproject.toml``'s ``[project.scripts]``."""
    app()


if __name__ == "__main__":
    run()
