"""Multi-pass inference benchmarking for :mod:`neuropose.estimator`.

The :mod:`neuropose.estimator` module already instruments every call
and attaches a :class:`~neuropose.io.PerformanceMetrics` to its
:class:`~neuropose.estimator.ProcessVideoResult`, so real-world runs
always carry their own timing. This module layers on top of that to
answer the harder question — *what can this machine actually do
steady-state?* — by running the same video multiple times, throwing
out the first pass entirely (graph compilation / file-system cache
warmup), and aggregating the rest into distributional statistics.

The design is intentionally thin: a single :func:`run_benchmark`
entry point, a small divergence helper for the ``--compare-cpu``
flow, and a pretty-printer used by the CLI. The heavy lifting —
schemas, aggregation, serialisation — lives in :mod:`neuropose.io`.

See :class:`neuropose.io.BenchmarkResult` for the result shape and
:mod:`neuropose.cli.benchmark` for the command-line surface.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from neuropose.estimator import Estimator
from neuropose.io import (
    BenchmarkAggregate,
    BenchmarkResult,
    PerformanceMetrics,
    VideoPredictions,
)


@dataclass(frozen=True)
class BenchmarkRunOutcome:
    """Return type of :func:`run_benchmark`.

    Separate from :class:`neuropose.io.BenchmarkResult` because the
    reference :class:`~neuropose.io.VideoPredictions` captured from the
    last measured pass is **not** part of the serialisable benchmark
    record — it is kept in memory only to feed the ``--compare-cpu``
    divergence computation, and would otherwise bloat every JSON output
    with redundant pose data.
    """

    result: BenchmarkResult
    reference_predictions: VideoPredictions | None


def run_benchmark(
    estimator: Estimator,
    video_path: Path,
    *,
    repeats: int = 5,
    warmup_frames: int = 3,
    capture_reference: bool = False,
) -> BenchmarkRunOutcome:
    """Run ``repeats`` passes of ``process_video`` and aggregate timings.

    Parameters
    ----------
    estimator
        A fully-initialised :class:`~neuropose.estimator.Estimator`
        with its model loaded (or injected). The benchmark calls
        ``estimator.process_video(video_path)`` in a loop.
    video_path
        Path to an input video. Must be readable by OpenCV; any
        :class:`~neuropose.estimator.VideoDecodeError` from the first
        pass propagates out (no point retrying a broken file).
    repeats
        Total number of passes to run. Must be at least 2 — the first
        pass is always discarded, and the aggregate needs at least one
        measured pass to be meaningful.
    warmup_frames
        Number of frames to exclude from the head of *each* measured
        pass when computing the aggregate. Graph compilation and XLA
        kernel JIT usually bite only the first 1-3 frames of a pass,
        so the default of 3 is a safe floor for Apple Silicon and
        CUDA alike.
    capture_reference
        When ``True``, the :class:`VideoPredictions` from the last
        measured pass is preserved on the outcome. Used by the
        ``--compare-cpu`` flow to diff poses across devices; callers
        that only want timings should leave this ``False``.

    Returns
    -------
    BenchmarkRunOutcome
        The serialisable :class:`~neuropose.io.BenchmarkResult` plus
        (optionally) the reference predictions.

    Raises
    ------
    ValueError
        If ``repeats`` is less than 2 (no measured passes after
        discarding the warmup pass) or ``warmup_frames`` is negative.
    """
    if repeats < 2:
        raise ValueError(f"repeats must be >= 2; got {repeats}")
    if warmup_frames < 0:
        raise ValueError(f"warmup_frames must be >= 0; got {warmup_frames}")

    passes: list[PerformanceMetrics] = []
    reference_predictions: VideoPredictions | None = None
    # Provenance is identical across every pass of a single run (same
    # estimator, same model, same environment), so we keep just the
    # latest one we see. Doing this on every iteration is cheap — it's
    # one attribute read — and means the benchmark result carries
    # provenance even when ``capture_reference`` is off.
    latest_provenance = None
    for i in range(repeats):
        result = estimator.process_video(video_path)
        passes.append(result.metrics)
        if result.predictions.provenance is not None:
            latest_provenance = result.predictions.provenance
        # Only the *last* measured pass needs to be captured for
        # divergence comparison. Earlier passes would just be
        # overwritten, so we avoid holding their frame dicts in memory.
        if capture_reference and i == repeats - 1:
            reference_predictions = result.predictions

    aggregate = _aggregate_passes(passes[1:], warmup_frames=warmup_frames)
    benchmark_result = BenchmarkResult(
        video_name=video_path.name,
        repeats=repeats,
        warmup_frames=warmup_frames,
        warmup_pass=passes[0],
        measured_passes=passes[1:],
        aggregate=aggregate,
        provenance=latest_provenance,
    )
    return BenchmarkRunOutcome(
        result=benchmark_result,
        reference_predictions=reference_predictions,
    )


def compute_poses3d_divergence(
    reference: VideoPredictions,
    other: VideoPredictions,
) -> tuple[float, int]:
    """Return the maximum absolute ``poses3d`` divergence in mm.

    Walks both prediction sets frame-by-frame and compares each
    ``(person, joint, axis)`` entry. Frames are matched by name (the
    six-digit ``frame_000000`` identifier), which assumes both runs
    processed the same source video and decoded the same number of
    frames — the benchmark always does, since it invokes
    ``process_video`` on the same file with the same OpenCV build.

    Parameters
    ----------
    reference
        Predictions from the primary device pass (typically GPU).
    other
        Predictions from the secondary device pass (typically CPU via
        a ``--force-cpu`` subprocess).

    Returns
    -------
    tuple[float, int]
        ``(max_divergence_mm, frame_count_compared)``. The integer is
        the number of frames that actually contributed to the diff —
        frames with mismatched detection counts are skipped rather
        than raising, so the caller can tell "is the number trustworthy?"
        from the count alone.

    Raises
    ------
    ValueError
        If the two prediction sets report different frame counts (the
        upstream benchmark should never produce that, so it is a real
        bug when it happens).
    """
    ref_names = reference.frame_names()
    other_names = other.frame_names()
    if len(ref_names) != len(other_names):
        raise ValueError(
            f"frame count mismatch: reference has {len(ref_names)}, other has {len(other_names)}"
        )

    max_diff = 0.0
    compared = 0
    for ref_name, other_name in zip(ref_names, other_names, strict=True):
        ref_frame = reference[ref_name]
        other_frame = other[other_name]
        if len(ref_frame.poses3d) != len(other_frame.poses3d):
            # Detection-count mismatches mean the two devices disagreed
            # about *how many people* were in the frame, not about joint
            # positions. Skip these frames for the divergence metric
            # but do not silently mask them — the comparison result
            # surfaces frame_count_compared so the caller can tell.
            continue
        if not ref_frame.poses3d:
            # Both zero detections → nothing to compare, nothing to skip.
            compared += 1
            continue
        ref_arr = np.asarray(ref_frame.poses3d, dtype=float)
        other_arr = np.asarray(other_frame.poses3d, dtype=float)
        if ref_arr.shape != other_arr.shape:
            continue
        diff = float(np.abs(ref_arr - other_arr).max())
        if diff > max_diff:
            max_diff = diff
        compared += 1
    return max_diff, compared


def format_benchmark_report(result: BenchmarkResult) -> str:
    """Return a human-readable report for ``result``.

    Renders a multi-line summary suitable for stdout. The CLI uses this
    for the operator-facing output; the JSON on disk is the machine-
    readable form.
    """
    agg = result.aggregate
    lines = [
        f"Benchmark: {result.video_name}",
        f"  device:       {agg.active_device}"
        + (" (tensorflow-metal)" if agg.tensorflow_metal_active else ""),
        f"  tf version:   {agg.tensorflow_version}",
        f"  model load:   {_format_model_load(result.warmup_pass.model_load_seconds)}",
        f"  repeats:      {result.repeats} ({agg.repeats_measured} measured, 1 discarded)",
        f"  warmup:       {result.warmup_frames} frame(s) excluded per pass",
        "",
        "Per-frame latency (ms, across measured passes after warmup):",
        f"  mean:    {agg.mean_frame_latency_ms:8.2f}",
        f"  p50:     {agg.p50_frame_latency_ms:8.2f}",
        f"  p95:     {agg.p95_frame_latency_ms:8.2f}",
        f"  p99:     {agg.p99_frame_latency_ms:8.2f}",
        f"  stddev:  {agg.stddev_frame_latency_ms:8.2f}",
        "",
        f"Throughput:   {agg.mean_throughput_fps:.2f} fps (mean across measured passes)",
        f"Peak RSS:     {agg.peak_rss_mb_max:.0f} MB",
    ]
    if result.cpu_comparison is not None:
        cmp = result.cpu_comparison
        lines.extend(
            [
                "",
                "CPU comparison:",
                f"  primary throughput: {cmp.primary_aggregate.mean_throughput_fps:.2f} fps",
                f"  cpu throughput:     {cmp.cpu_aggregate.mean_throughput_fps:.2f} fps",
                f"  speedup:            {cmp.speedup:.2f}x",
                f"  max poses3d divergence: {cmp.max_poses3d_divergence_mm:.4f} mm "
                f"({cmp.frame_count_compared} frames compared)",
            ]
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _aggregate_passes(
    measured: list[PerformanceMetrics],
    *,
    warmup_frames: int,
) -> BenchmarkAggregate:
    """Compute a :class:`BenchmarkAggregate` from a list of measured passes."""
    if not measured:
        # Caller guarantees repeats >= 2 so measured is non-empty, but
        # validate here anyway: an empty list would make np.percentile
        # fail with a confusing error downstream.
        raise ValueError("cannot aggregate zero measured passes")

    latencies: list[float] = []
    throughputs: list[float] = []
    peak_rss = 0.0
    for p in measured:
        # Slice off the warmup head of each pass. If warmup_frames
        # exceeds the pass length, the slice is empty and contributes
        # nothing, which is the correct behaviour for a pathologically
        # short video.
        kept = p.per_frame_latencies_ms[warmup_frames:]
        latencies.extend(kept)
        frame_count = len(p.per_frame_latencies_ms)
        if frame_count > 0 and p.total_seconds > 0:
            throughputs.append(frame_count / p.total_seconds)
        if p.peak_rss_mb > peak_rss:
            peak_rss = p.peak_rss_mb

    if not latencies:
        # Could happen if warmup_frames >= frames_per_pass for every
        # measured pass. Aggregate with all-zero timing rather than
        # exploding; the CLI surfaces the zero-throughput number
        # clearly enough that the operator will notice.
        return BenchmarkAggregate(
            repeats_measured=len(measured),
            warmup_frames_per_pass=warmup_frames,
            mean_frame_latency_ms=0.0,
            p50_frame_latency_ms=0.0,
            p95_frame_latency_ms=0.0,
            p99_frame_latency_ms=0.0,
            stddev_frame_latency_ms=0.0,
            mean_throughput_fps=float(np.mean(throughputs)) if throughputs else 0.0,
            peak_rss_mb_max=peak_rss,
            active_device=measured[0].active_device,
            tensorflow_metal_active=measured[0].tensorflow_metal_active,
            tensorflow_version=measured[0].tensorflow_version,
        )

    latencies_arr = np.asarray(latencies, dtype=float)
    return BenchmarkAggregate(
        repeats_measured=len(measured),
        warmup_frames_per_pass=warmup_frames,
        mean_frame_latency_ms=float(latencies_arr.mean()),
        p50_frame_latency_ms=float(np.percentile(latencies_arr, 50)),
        p95_frame_latency_ms=float(np.percentile(latencies_arr, 95)),
        p99_frame_latency_ms=float(np.percentile(latencies_arr, 99)),
        stddev_frame_latency_ms=float(latencies_arr.std()),
        mean_throughput_fps=float(np.mean(throughputs)) if throughputs else 0.0,
        peak_rss_mb_max=peak_rss,
        active_device=measured[0].active_device,
        tensorflow_metal_active=measured[0].tensorflow_metal_active,
        tensorflow_version=measured[0].tensorflow_version,
    )


def _format_model_load(seconds: float | None) -> str:
    """Format the ``model_load_seconds`` field for the text report."""
    if seconds is None:
        return "injected (not loaded by benchmark)"
    return f"{seconds:.2f} s"
