"""NeuroPose command-line interface.

Five subcommands:

- ``neuropose watch`` — run the :class:`~neuropose.interfacer.Interfacer`
  daemon against the configured input directory.
- ``neuropose process <video>`` — run the estimator on a single video and
  write the predictions JSON to disk.
- ``neuropose segment <results>`` — post-hoc repetition segmentation of
  an existing predictions file. Attaches a named
  :class:`~neuropose.io.Segmentation` to every video it contains and
  writes the file back atomically.
- ``neuropose benchmark <video>`` — multi-pass inference benchmark for
  a single video, with optional ``--compare-cpu`` for Apple-Silicon
  vs CPU numerical-divergence checks. Prints a human report to stdout
  and (optionally) writes a structured :class:`~neuropose.io.BenchmarkResult`
  JSON to ``--output``.
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

import json
import logging
from enum import StrEnum
from pathlib import Path
from typing import Annotated

import typer
from pydantic import ValidationError

from neuropose import __version__
from neuropose.config import Settings
from neuropose.estimator import Estimator
from neuropose.interfacer import AlreadyRunningError, Interfacer
from neuropose.io import (
    BenchmarkResult,
    CpuComparisonResult,
    ExtractorSpec,
    JobResults,
    VideoPredictions,
    load_benchmark_result,
    load_job_results,
    load_video_predictions,
    save_benchmark_result,
    save_job_results,
    save_video_predictions,
)

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
# segment
# ---------------------------------------------------------------------------


class _ExtractorKind(StrEnum):
    """CLI-facing extractor-kind enum for ``neuropose segment --extractor``."""

    JOINT_AXIS = "joint_axis"
    JOINT_PAIR_DISTANCE = "joint_pair_distance"
    JOINT_SPEED = "joint_speed"
    JOINT_ANGLE = "joint_angle"


def _resolve_joint(token: str) -> int:
    """Resolve a joint specifier to an integer index.

    Accepts either a string joint name from the berkeley_mhad_43 skeleton
    (e.g. ``"rwri"``) or a plain integer as a string. Raises
    ``typer.BadParameter`` with a helpful message on failure so CLI users
    get a short error rather than a pydantic traceback.
    """
    # Deferred import so the CLI module itself stays free of the analyzer
    # heavy imports when the user is only running watch/process.
    from neuropose.analyzer.segment import JOINT_INDEX

    stripped = token.strip()
    if not stripped:
        raise typer.BadParameter("joint specifier is empty")
    if stripped in JOINT_INDEX:
        return JOINT_INDEX[stripped]
    try:
        return int(stripped)
    except ValueError as exc:
        raise typer.BadParameter(
            f"unknown joint {stripped!r}; expected an integer index or one of "
            f"the berkeley_mhad_43 names (e.g. 'rwri', 'lkne')"
        ) from exc


def _parse_joint_list(raw: str, *, expected: int) -> tuple[int, ...]:
    """Split a comma-separated joint list into ``expected`` integer indices."""
    tokens = [t for t in raw.split(",") if t.strip()]
    if len(tokens) != expected:
        raise typer.BadParameter(
            f"expected {expected} comma-separated joint specifiers; got {len(tokens)}: {raw!r}"
        )
    return tuple(_resolve_joint(t) for t in tokens)


def _build_extractor_spec(
    kind: _ExtractorKind,
    *,
    joint: str | None,
    joints: str | None,
    triplet: str | None,
    axis: int | None,
    invert: bool,
) -> ExtractorSpec:
    """Translate CLI flags into a serializable :class:`ExtractorSpec`.

    Each extractor kind consumes a different subset of the flags; unused
    flags are ignored for ergonomics (so users can leave defaults in shell
    aliases) but missing required flags raise ``typer.BadParameter`` with
    the specific missing name so the error is obvious.
    """
    from neuropose.analyzer.segment import (
        joint_angle,
        joint_axis,
        joint_pair_distance,
        joint_speed,
    )

    if kind is _ExtractorKind.JOINT_AXIS:
        if joint is None:
            raise typer.BadParameter("--joint is required for joint_axis")
        if axis is None:
            raise typer.BadParameter("--axis is required for joint_axis")
        if not (0 <= axis <= 2):
            raise typer.BadParameter(f"--axis must be 0, 1, or 2; got {axis}")
        return joint_axis(_resolve_joint(joint), axis, invert=invert)

    if kind is _ExtractorKind.JOINT_PAIR_DISTANCE:
        if joints is None:
            raise typer.BadParameter(
                "--joints is required for joint_pair_distance (e.g. --joints lwri,rwri)"
            )
        j1, j2 = _parse_joint_list(joints, expected=2)
        return joint_pair_distance(j1, j2)

    if kind is _ExtractorKind.JOINT_SPEED:
        if joint is None:
            raise typer.BadParameter("--joint is required for joint_speed")
        return joint_speed(_resolve_joint(joint))

    if kind is _ExtractorKind.JOINT_ANGLE:
        if triplet is None:
            raise typer.BadParameter(
                "--triplet is required for joint_angle (e.g. --triplet larm,lelb,lwri)"
            )
        a, b, c = _parse_joint_list(triplet, expected=3)
        return joint_angle(a, b, c)

    raise typer.BadParameter(f"unknown extractor kind: {kind}")


def _load_predictions_or_results(
    path: Path,
) -> tuple[JobResults | VideoPredictions, bool]:
    """Load a results file, auto-detecting JobResults vs VideoPredictions.

    Returns the loaded object and a boolean ``is_job_results`` flag so
    the caller knows which save helper to use when writing back. The
    distinction is made by trying :class:`VideoPredictions` first (the
    more restrictive shape) and falling back to :class:`JobResults`.
    """
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    # A JobResults is a mapping of video-name → VideoPredictions; a bare
    # VideoPredictions always has a top-level "metadata" key. This is
    # cheap enough to dispatch on without a full validation pass.
    if isinstance(raw, dict) and "metadata" in raw and "frames" in raw:
        return load_video_predictions(path), False
    return load_job_results(path), True


@app.command()
def segment(
    ctx: typer.Context,
    results: Annotated[
        Path,
        typer.Argument(
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            help="Path to a results.json (JobResults) or predictions.json "
            "(single VideoPredictions).",
        ),
    ],
    name: Annotated[
        str,
        typer.Option(
            "--name",
            help="Key under which the segmentation will be stored in "
            "VideoPredictions.segmentations (e.g. 'cup_lift').",
        ),
    ],
    extractor: Annotated[
        _ExtractorKind,
        typer.Option(
            "--extractor",
            help="Which segmentation signal to extract from each frame.",
        ),
    ],
    joint: Annotated[
        str | None,
        typer.Option(
            "--joint",
            help="Joint specifier for joint_axis and joint_speed extractors. "
            "Accepts a berkeley_mhad_43 name ('rwri', 'lkne') or an integer index.",
        ),
    ] = None,
    joints: Annotated[
        str | None,
        typer.Option(
            "--joints",
            help="Comma-separated pair for joint_pair_distance (e.g. 'lwri,rwri').",
        ),
    ] = None,
    triplet: Annotated[
        str | None,
        typer.Option(
            "--triplet",
            help="Comma-separated (a,b,c) for joint_angle; angle is at b.",
        ),
    ] = None,
    axis: Annotated[
        int | None,
        typer.Option(
            "--axis",
            help="Spatial axis for joint_axis: 0=x, 1=y, 2=z.",
        ),
    ] = None,
    invert: Annotated[
        bool,
        typer.Option(
            "--invert",
            help="Negate the signal (joint_axis only); useful when the "
            "movement of interest is a decrease in the selected coordinate.",
        ),
    ] = False,
    person_index: Annotated[
        int,
        typer.Option("--person-index", min=0, help="Which detected person to use."),
    ] = 0,
    min_distance_seconds: Annotated[
        float | None,
        typer.Option(
            "--min-distance-seconds",
            min=0.0,
            help="Minimum time between successive repetition peaks.",
        ),
    ] = None,
    min_prominence: Annotated[
        float | None,
        typer.Option(
            "--min-prominence",
            help="Minimum scipy peak prominence on the raw signal.",
        ),
    ] = None,
    min_height: Annotated[
        float | None,
        typer.Option(
            "--min-height",
            help="Minimum signal value to qualify as a peak.",
        ),
    ] = None,
    pad_seconds: Annotated[
        float,
        typer.Option(
            "--pad-seconds",
            min=0.0,
            help="Amount of time to extend each segment on both sides.",
        ),
    ] = 0.0,
    output: Annotated[
        Path | None,
        typer.Option(
            "--output",
            "-o",
            help="Where to write the segmented file. Defaults to overwriting the input atomically.",
            dir_okay=False,
            writable=True,
        ),
    ] = None,
    force: Annotated[
        bool,
        typer.Option(
            "--force",
            help="Overwrite an existing segmentation with the same --name. "
            "Without --force, a collision is a usage error.",
        ),
    ] = False,
) -> None:
    """Run post-hoc repetition segmentation over an existing predictions file.

    Loads the file (auto-detecting JobResults vs single VideoPredictions),
    runs :func:`neuropose.analyzer.segment.segment_predictions` on each
    video with the chosen extractor and thresholds, attaches the result
    under ``--name`` in every video's ``segmentations`` mapping, and
    writes the file back to ``--output`` (or the input path by default).

    The command mutates only the ``segmentations`` field; inference
    output — per-frame poses and video metadata — round-trips unchanged.
    """
    del ctx  # settings not needed; this command is pure post-processing
    # Deferred import keeps the CLI module's top-level imports free of
    # scipy so watch/process do not pay the import cost on startup.
    from neuropose.analyzer.segment import segment_predictions

    try:
        spec = _build_extractor_spec(
            extractor,
            joint=joint,
            joints=joints,
            triplet=triplet,
            axis=axis,
            invert=invert,
        )
    except typer.BadParameter as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=EXIT_USAGE) from exc

    try:
        loaded, is_job_results = _load_predictions_or_results(results)
    except (ValidationError, json.JSONDecodeError) as exc:
        typer.echo(f"error: could not load {results}: {exc}", err=True)
        raise typer.Exit(code=EXIT_USAGE) from exc

    # Normalise to an iterable of (video_name, VideoPredictions) so the
    # per-video segmentation loop is the same for both input shapes.
    if is_job_results:
        assert isinstance(loaded, JobResults)
        videos: dict[str, VideoPredictions] = dict(loaded.root)
    else:
        assert isinstance(loaded, VideoPredictions)
        videos = {results.name: loaded}

    updated: dict[str, VideoPredictions] = {}
    total_segments = 0
    for video_name, preds in videos.items():
        if name in preds.segmentations and not force:
            typer.echo(
                f"error: video {video_name!r} already has a segmentation named "
                f"{name!r}; pass --force to overwrite.",
                err=True,
            )
            raise typer.Exit(code=EXIT_USAGE)
        try:
            result = segment_predictions(
                preds,
                spec,
                person_index=person_index,
                min_distance_seconds=min_distance_seconds,
                min_prominence=min_prominence,
                min_height=min_height,
                pad_seconds=pad_seconds,
            )
        except (ValueError, ImportError) as exc:
            typer.echo(f"error: segmentation failed for {video_name}: {exc}", err=True)
            raise typer.Exit(code=EXIT_USAGE) from exc

        new_segmentations = dict(preds.segmentations)
        new_segmentations[name] = result
        updated[video_name] = preds.model_copy(update={"segmentations": new_segmentations})
        total_segments += len(result.segments)
        typer.echo(f"{video_name}: {len(result.segments)} segment(s) under name={name!r}")

    out_path = output if output is not None else results
    if is_job_results:
        save_job_results(out_path, JobResults(root=updated))
    else:
        # Single-VideoPredictions input: the dict has exactly one entry.
        (only,) = updated.values()
        save_video_predictions(out_path, only)

    typer.echo(f"wrote {out_path} ({len(updated)} video(s), {total_segments} total segment(s))")


# ---------------------------------------------------------------------------
# benchmark
# ---------------------------------------------------------------------------


def _force_cpu_only() -> None:
    """Hide all GPU devices from TensorFlow before any TF op initialises.

    Called from the ``--force-cpu`` benchmark path before the estimator
    loads the model. ``tf.config.set_visible_devices`` must be invoked
    before the runtime has touched a GPU device, which in practice
    means before any other TF API call — so this function imports TF
    itself and then immediately pins visibility to an empty GPU list.

    The import is deferred inside the function so that non-benchmark
    CLI paths (``watch``, ``process``) do not pay the ~2 s TensorFlow
    import cost unless they are actually doing inference.
    """
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise typer.BadParameter(
            "--force-cpu requires TensorFlow to be installed, but import failed"
        ) from exc
    # Silence is fine here: a stale visibility call on a runtime that
    # already initialised GPU raises RuntimeError, but the benchmark CLI
    # calls this as its first TF op so that should never happen.
    tf.config.set_visible_devices([], "GPU")


def _run_compare_cpu_subprocess(
    video: Path,
    *,
    repeats: int,
    warmup_frames: int,
) -> tuple[BenchmarkResult, VideoPredictions]:
    """Spawn a ``--force-cpu`` child to get a CPU baseline + predictions.

    The parent benchmark runs on whatever device the platform exposes
    by default (Apple Silicon → Metal GPU when the extra is installed;
    Linux → CPU by default). For the ``--compare-cpu`` comparison we
    need a second run that is *guaranteed* to hit CPU, and we need the
    resulting ``poses3d`` to diff against the parent's output.

    The cleanest way to guarantee device isolation is to run the CPU
    pass in a subprocess with GPU visibility hidden before any TF
    import. In-process device switching via
    ``tf.device("/CPU:0")`` is not reliable against SavedModels whose
    ConcreteFunctions may carry baked-in device placements, and
    calling ``set_visible_devices`` after the GPU has already been
    touched is a hard error from TF's runtime.

    The subprocess is invoked as ``neuropose benchmark <video> --force-cpu
    --repeats N --warmup-frames M --output <tmp> --predictions-output
    <tmp>`` so the child writes both the benchmark result and the last
    measured pass's predictions to disk. The parent reads them, deletes
    the tempfiles, and returns.
    """
    # Deferred imports: subprocess and tempfile are pure-Python and
    # cheap, but keeping them inside the function keeps the module's
    # top-level import surface small.
    import subprocess
    import sys
    import tempfile

    with tempfile.TemporaryDirectory(prefix="neuropose_cpu_bench_") as td:
        td_path = Path(td)
        result_path = td_path / "result.json"
        predictions_path = td_path / "predictions.json"

        cmd = [
            sys.executable,
            "-m",
            "neuropose.cli",
            "benchmark",
            str(video),
            "--repeats",
            str(repeats),
            "--warmup-frames",
            str(warmup_frames),
            "--output",
            str(result_path),
            "--predictions-output",
            str(predictions_path),
            "--force-cpu",
        ]
        logger.info("spawning cpu-baseline subprocess: %s", " ".join(cmd))
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if proc.returncode != 0:
            raise RuntimeError(
                f"cpu-baseline subprocess failed (exit {proc.returncode}): {proc.stderr.strip()}"
            )
        return (
            load_benchmark_result(result_path),
            load_video_predictions(predictions_path),
        )


@app.command()
def benchmark(
    ctx: typer.Context,
    video: Annotated[
        Path,
        typer.Argument(
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            help="Path to an input video file to benchmark.",
        ),
    ],
    repeats: Annotated[
        int,
        typer.Option(
            "--repeats",
            min=2,
            help="Total passes to run; the first is always discarded as warmup.",
        ),
    ] = 5,
    warmup_frames: Annotated[
        int,
        typer.Option(
            "--warmup-frames",
            min=0,
            help="Frames to discard from the head of each measured pass.",
        ),
    ] = 3,
    compare_cpu: Annotated[
        bool,
        typer.Option(
            "--compare-cpu",
            help=(
                "After the primary benchmark, spawn a subprocess with "
                "--force-cpu to produce a CPU baseline on the same video, "
                "then report throughput speedup and max poses3d divergence "
                "in millimetres. Intended for Apple Silicon numerics "
                "verification."
            ),
        ),
    ] = False,
    force_cpu: Annotated[
        bool,
        typer.Option(
            "--force-cpu",
            help=(
                "Hide all GPU devices from TensorFlow before loading the "
                "model, so inference is guaranteed to run on CPU. Used "
                "internally by --compare-cpu's subprocess; rarely needed "
                "as a user-facing flag."
            ),
        ),
    ] = False,
    output: Annotated[
        Path | None,
        typer.Option(
            "--output",
            "-o",
            help=(
                "Where to write the structured BenchmarkResult JSON. "
                "When omitted, only the human-readable report prints to "
                "stdout."
            ),
            dir_okay=False,
            writable=True,
        ),
    ] = None,
    predictions_output: Annotated[
        Path | None,
        typer.Option(
            "--predictions-output",
            help=(
                "Where to write the last measured pass's VideoPredictions "
                "JSON. Used internally by --compare-cpu to supply the "
                "divergence computation with the CPU pass's poses3d; "
                "leave unset for normal benchmark runs."
            ),
            dir_okay=False,
            writable=True,
        ),
    ] = None,
) -> None:
    """Run a multi-pass inference benchmark for a single video.

    The first pass is always discarded (graph compilation / filesystem
    caches), and subsequent passes contribute to a
    :class:`~neuropose.io.BenchmarkAggregate` with mean / p50 / p95 / p99
    per-frame latencies, throughput, peak RSS, active device, and the
    exact TensorFlow version that ran the measurement. With
    ``--compare-cpu``, a second run is spawned as a subprocess with
    GPU visibility hidden; the resulting CPU baseline is diffed against
    the primary run's ``poses3d`` array to produce a maximum absolute
    divergence in millimetres, which is the answer to the "is
    tensorflow-metal producing correct numerics?" question that
    RESEARCH.md's TensorFlow-version-compatibility section leaves open.
    """
    settings: Settings = ctx.obj
    # Deferred imports: the benchmark module and its dependencies
    # (numpy, time) are needed for this subcommand only.
    from neuropose.benchmark import (
        compute_poses3d_divergence,
        format_benchmark_report,
        run_benchmark,
    )

    if force_cpu and compare_cpu:
        typer.echo(
            "error: --force-cpu and --compare-cpu are mutually exclusive; "
            "--force-cpu is an implementation detail of --compare-cpu's "
            "subprocess and should not be combined with it.",
            err=True,
        )
        raise typer.Exit(code=EXIT_USAGE)

    if force_cpu:
        _force_cpu_only()

    estimator = Estimator(
        device=settings.device,
        default_fov_degrees=settings.default_fov_degrees,
    )
    try:
        estimator.load_model(cache_dir=settings.model_cache_dir)
    except NotImplementedError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=EXIT_PENDING) from exc

    outcome = run_benchmark(
        estimator,
        video,
        repeats=repeats,
        warmup_frames=warmup_frames,
        capture_reference=compare_cpu or predictions_output is not None,
    )
    result = outcome.result

    # CPU comparison path: spawn child, diff poses3d, rewrap result
    # with a cpu_comparison field populated.
    if compare_cpu:
        if outcome.reference_predictions is None:
            # Should be impossible because we asked for it above; guard
            # anyway so a future refactor cannot silently drop it.
            typer.echo(
                "error: --compare-cpu requires reference predictions, "
                "but none were captured. This is a bug.",
                err=True,
            )
            raise typer.Exit(code=EXIT_USAGE)

        try:
            cpu_result, cpu_predictions = _run_compare_cpu_subprocess(
                video,
                repeats=repeats,
                warmup_frames=warmup_frames,
            )
        except RuntimeError as exc:
            typer.echo(f"error: {exc}", err=True)
            raise typer.Exit(code=EXIT_USAGE) from exc

        max_diff, compared = compute_poses3d_divergence(
            outcome.reference_predictions, cpu_predictions
        )
        primary_throughput = result.aggregate.mean_throughput_fps
        cpu_throughput = cpu_result.aggregate.mean_throughput_fps
        speedup = primary_throughput / cpu_throughput if cpu_throughput > 0 else 0.0
        comparison = CpuComparisonResult(
            primary_aggregate=result.aggregate,
            cpu_aggregate=cpu_result.aggregate,
            speedup=speedup,
            max_poses3d_divergence_mm=max_diff,
            frame_count_compared=compared,
        )
        result = result.model_copy(update={"cpu_comparison": comparison})

    typer.echo(format_benchmark_report(result))

    if output is not None:
        save_benchmark_result(output, result)
        typer.echo(f"\nwrote {output}")

    if predictions_output is not None:
        if outcome.reference_predictions is None:
            typer.echo(
                "error: --predictions-output was given but no reference "
                "predictions were captured. This is a bug.",
                err=True,
            )
            raise typer.Exit(code=EXIT_USAGE)
        save_video_predictions(predictions_output, outcome.reference_predictions)


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
