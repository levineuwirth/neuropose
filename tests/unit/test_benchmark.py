"""Tests for :mod:`neuropose.benchmark`.

Coverage:

- :func:`run_benchmark` against a fake-model :class:`Estimator` —
  repeats, warmup-pass discarding, capture-reference, edge cases.
- Aggregate statistics (mean / p50 / p95 / p99 / throughput / peak RSS)
  on synthetic metrics.
- :func:`compute_poses3d_divergence` on matched and mismatched
  prediction pairs.
- :func:`format_benchmark_report` rendering.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from neuropose.benchmark import (
    compute_poses3d_divergence,
    format_benchmark_report,
    run_benchmark,
)
from neuropose.estimator import Estimator
from neuropose.io import (
    BenchmarkResult,
    PerformanceMetrics,
    VideoPredictions,
)


class TestRunBenchmark:
    def test_returns_benchmark_result(
        self,
        synthetic_video: Path,
        fake_metrabs_model,
    ) -> None:
        estimator = Estimator(model=fake_metrabs_model)
        outcome = run_benchmark(estimator, synthetic_video, repeats=3, warmup_frames=0)
        assert isinstance(outcome.result, BenchmarkResult)
        assert outcome.result.video_name == synthetic_video.name

    def test_repeats_reflected_in_result(
        self,
        synthetic_video: Path,
        fake_metrabs_model,
    ) -> None:
        estimator = Estimator(model=fake_metrabs_model)
        outcome = run_benchmark(estimator, synthetic_video, repeats=4, warmup_frames=0)
        assert outcome.result.repeats == 4
        # One pass is always discarded as warmup; the rest are measured.
        assert len(outcome.result.measured_passes) == 3
        assert outcome.result.aggregate.repeats_measured == 3

    def test_first_pass_is_the_warmup(
        self,
        synthetic_video: Path,
        fake_metrabs_model,
    ) -> None:
        estimator = Estimator(model=fake_metrabs_model)
        outcome = run_benchmark(estimator, synthetic_video, repeats=3, warmup_frames=0)
        assert outcome.result.warmup_pass is not None
        # warmup_pass is a separate object from the measured passes.
        assert outcome.result.warmup_pass not in outcome.result.measured_passes

    def test_capture_reference_off_by_default(
        self,
        synthetic_video: Path,
        fake_metrabs_model,
    ) -> None:
        estimator = Estimator(model=fake_metrabs_model)
        outcome = run_benchmark(estimator, synthetic_video, repeats=3, warmup_frames=0)
        assert outcome.reference_predictions is None

    def test_capture_reference_returns_last_measured_pass(
        self,
        synthetic_video: Path,
        fake_metrabs_model,
    ) -> None:
        estimator = Estimator(model=fake_metrabs_model)
        outcome = run_benchmark(
            estimator,
            synthetic_video,
            repeats=3,
            warmup_frames=0,
            capture_reference=True,
        )
        assert isinstance(outcome.reference_predictions, VideoPredictions)
        assert len(outcome.reference_predictions) == 5

    def test_rejects_repeats_below_two(
        self,
        synthetic_video: Path,
        fake_metrabs_model,
    ) -> None:
        estimator = Estimator(model=fake_metrabs_model)
        with pytest.raises(ValueError, match="repeats"):
            run_benchmark(estimator, synthetic_video, repeats=1, warmup_frames=0)

    def test_rejects_negative_warmup_frames(
        self,
        synthetic_video: Path,
        fake_metrabs_model,
    ) -> None:
        estimator = Estimator(model=fake_metrabs_model)
        with pytest.raises(ValueError, match="warmup_frames"):
            run_benchmark(estimator, synthetic_video, repeats=3, warmup_frames=-1)

    def test_warmup_frames_filters_aggregate_head(
        self,
        synthetic_video: Path,
        fake_metrabs_model,
    ) -> None:
        estimator = Estimator(model=fake_metrabs_model)
        # synthetic_video has 5 frames; warmup_frames=4 leaves only
        # the last frame of each measured pass to contribute.
        outcome = run_benchmark(estimator, synthetic_video, repeats=3, warmup_frames=4)
        # The stats should still compute, not error.
        assert outcome.result.aggregate.mean_frame_latency_ms >= 0.0

    def test_warmup_frames_exceeding_pass_length_gives_zero_latencies(
        self,
        synthetic_video: Path,
        fake_metrabs_model,
    ) -> None:
        estimator = Estimator(model=fake_metrabs_model)
        # 5-frame video, warmup_frames=10 → every frame discarded.
        outcome = run_benchmark(estimator, synthetic_video, repeats=3, warmup_frames=10)
        # Latency aggregate drops to zero; throughput is still
        # computed from total wall clock.
        assert outcome.result.aggregate.mean_frame_latency_ms == 0.0
        assert outcome.result.aggregate.p95_frame_latency_ms == 0.0


class TestBenchmarkAggregate:
    def test_percentiles_ordered_correctly(
        self,
        synthetic_video: Path,
        fake_metrabs_model,
    ) -> None:
        estimator = Estimator(model=fake_metrabs_model)
        outcome = run_benchmark(estimator, synthetic_video, repeats=4, warmup_frames=0)
        agg = outcome.result.aggregate
        assert agg.p50_frame_latency_ms <= agg.p95_frame_latency_ms
        assert agg.p95_frame_latency_ms <= agg.p99_frame_latency_ms

    def test_throughput_non_negative(
        self,
        synthetic_video: Path,
        fake_metrabs_model,
    ) -> None:
        estimator = Estimator(model=fake_metrabs_model)
        outcome = run_benchmark(estimator, synthetic_video, repeats=3, warmup_frames=0)
        assert outcome.result.aggregate.mean_throughput_fps > 0.0

    def test_peak_rss_matches_max_across_passes(
        self,
        synthetic_video: Path,
        fake_metrabs_model,
    ) -> None:
        estimator = Estimator(model=fake_metrabs_model)
        outcome = run_benchmark(estimator, synthetic_video, repeats=3, warmup_frames=0)
        individual = [p.peak_rss_mb for p in outcome.result.measured_passes]
        assert outcome.result.aggregate.peak_rss_mb_max == max(individual)


class TestComputePoses3dDivergence:
    def _make_predictions(self, poses3d_values: list[float]) -> VideoPredictions:
        """Build a 3-frame VideoPredictions whose single joint has the given y-values."""
        frames: dict[str, dict] = {}
        for i, value in enumerate(poses3d_values):
            frames[f"frame_{i:06d}"] = {
                "boxes": [[0.0, 0.0, 1.0, 1.0, 0.9]],
                "poses3d": [[[0.0, float(value), 0.0]]],
                "poses2d": [[[0.0, 0.0]]],
            }
        return VideoPredictions.model_validate(
            {
                "metadata": {
                    "frame_count": len(poses3d_values),
                    "fps": 30.0,
                    "width": 64,
                    "height": 64,
                },
                "frames": frames,
            }
        )

    def test_identical_predictions_zero_divergence(self) -> None:
        a = self._make_predictions([1.0, 2.0, 3.0])
        b = self._make_predictions([1.0, 2.0, 3.0])
        max_diff, compared = compute_poses3d_divergence(a, b)
        assert max_diff == 0.0
        assert compared == 3

    def test_divergence_reports_max_abs_diff(self) -> None:
        a = self._make_predictions([1.0, 2.0, 3.0])
        b = self._make_predictions([1.0, 2.1, 2.5])  # diffs 0, 0.1, 0.5
        max_diff, compared = compute_poses3d_divergence(a, b)
        assert max_diff == pytest.approx(0.5)
        assert compared == 3

    def test_mismatched_frame_count_raises(self) -> None:
        a = self._make_predictions([1.0, 2.0])
        b = self._make_predictions([1.0, 2.0, 3.0])
        with pytest.raises(ValueError, match="frame count"):
            compute_poses3d_divergence(a, b)

    def test_mismatched_detection_count_skipped_not_raised(self) -> None:
        a = self._make_predictions([1.0, 2.0])
        # Frame 1 has zero detections instead of one.
        b_payload = {
            "metadata": {
                "frame_count": 2,
                "fps": 30.0,
                "width": 64,
                "height": 64,
            },
            "frames": {
                "frame_000000": {
                    "boxes": [[0, 0, 1, 1, 0.9]],
                    "poses3d": [[[0.0, 1.0, 0.0]]],
                    "poses2d": [[[0.0, 0.0]]],
                },
                "frame_000001": {
                    "boxes": [],
                    "poses3d": [],
                    "poses2d": [],
                },
            },
        }
        b = VideoPredictions.model_validate(b_payload)
        max_diff, compared = compute_poses3d_divergence(a, b)
        # First frame contributes 0 divergence; second frame is skipped.
        assert max_diff == 0.0
        assert compared == 1


class TestFormatBenchmarkReport:
    def _fake_metrics(self, latencies_ms: list[float]) -> PerformanceMetrics:
        return PerformanceMetrics(
            model_load_seconds=12.3,
            total_seconds=sum(latencies_ms) / 1000.0 + 0.01,
            per_frame_latencies_ms=latencies_ms,
            peak_rss_mb=512.0,
            active_device="/CPU:0",
            tensorflow_metal_active=False,
            tensorflow_version="2.18.0",
        )

    def test_report_mentions_key_fields(
        self,
        synthetic_video: Path,
        fake_metrabs_model,
    ) -> None:
        estimator = Estimator(model=fake_metrabs_model)
        outcome = run_benchmark(estimator, synthetic_video, repeats=3, warmup_frames=0)
        text = format_benchmark_report(outcome.result)
        assert "Benchmark:" in text
        assert "device:" in text
        assert "mean:" in text
        assert "p50:" in text
        assert "p95:" in text
        assert "p99:" in text
        assert "Throughput:" in text
        assert "Peak RSS:" in text

    def test_report_mentions_metal_when_active(self) -> None:
        from neuropose.io import BenchmarkAggregate, BenchmarkResult

        metrics = PerformanceMetrics(
            model_load_seconds=None,
            total_seconds=1.0,
            per_frame_latencies_ms=[10.0, 11.0],
            peak_rss_mb=100.0,
            active_device="/GPU:0",
            tensorflow_metal_active=True,
            tensorflow_version="2.18.0",
        )
        agg = BenchmarkAggregate(
            repeats_measured=1,
            warmup_frames_per_pass=0,
            mean_frame_latency_ms=10.5,
            p50_frame_latency_ms=10.5,
            p95_frame_latency_ms=11.0,
            p99_frame_latency_ms=11.0,
            stddev_frame_latency_ms=0.5,
            mean_throughput_fps=95.0,
            peak_rss_mb_max=100.0,
            active_device="/GPU:0",
            tensorflow_metal_active=True,
            tensorflow_version="2.18.0",
        )
        result = BenchmarkResult(
            video_name="test.mp4",
            repeats=2,
            warmup_frames=0,
            warmup_pass=metrics,
            measured_passes=[metrics],
            aggregate=agg,
        )
        text = format_benchmark_report(result)
        assert "tensorflow-metal" in text

    def test_report_model_load_injected_label(
        self,
        synthetic_video: Path,
        fake_metrabs_model,
    ) -> None:
        estimator = Estimator(model=fake_metrabs_model)
        outcome = run_benchmark(estimator, synthetic_video, repeats=3, warmup_frames=0)
        text = format_benchmark_report(outcome.result)
        assert "injected" in text  # fake model was not loaded via load_model
