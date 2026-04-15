"""Tests for :mod:`neuropose.io` schema and helpers."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from neuropose.io import (
    BenchmarkAggregate,
    BenchmarkResult,
    CpuComparisonResult,
    FramePrediction,
    JobResults,
    JobStatus,
    JointAngleExtractor,
    JointAxisExtractor,
    JointPairDistanceExtractor,
    JointSpeedExtractor,
    PerformanceMetrics,
    Segment,
    Segmentation,
    SegmentationConfig,
    StatusFile,
    VideoMetadata,
    VideoPredictions,
    load_benchmark_result,
    load_job_results,
    load_status,
    load_video_predictions,
    save_benchmark_result,
    save_job_results,
    save_status,
    save_video_predictions,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def one_frame() -> dict:
    """A minimal valid FramePrediction payload (one person, two joints)."""
    return {
        "boxes": [[10.0, 20.0, 100.0, 200.0, 0.95]],
        "poses3d": [[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]],
        "poses2d": [[[10.0, 20.0], [30.0, 40.0]]],
    }


@pytest.fixture
def video_metadata_payload() -> dict:
    return {"frame_count": 2, "fps": 30.0, "width": 640, "height": 480}


@pytest.fixture
def video_predictions_payload(one_frame: dict, video_metadata_payload: dict) -> dict:
    return {
        "metadata": video_metadata_payload,
        "frames": {
            "frame_000000": one_frame,
            "frame_000001": one_frame,
        },
    }


# ---------------------------------------------------------------------------
# FramePrediction
# ---------------------------------------------------------------------------


class TestFramePrediction:
    def test_roundtrip(self, one_frame: dict) -> None:
        frame = FramePrediction.model_validate(one_frame)
        assert frame.boxes == one_frame["boxes"]
        assert frame.poses3d == one_frame["poses3d"]
        assert frame.poses2d == one_frame["poses2d"]

    def test_rejects_extra_fields(self, one_frame: dict) -> None:
        one_frame["extra"] = "bogus"
        with pytest.raises(ValidationError):
            FramePrediction.model_validate(one_frame)

    def test_is_frozen(self, one_frame: dict) -> None:
        frame = FramePrediction.model_validate(one_frame)
        with pytest.raises(ValidationError):
            frame.boxes = []


# ---------------------------------------------------------------------------
# VideoMetadata
# ---------------------------------------------------------------------------


class TestVideoMetadata:
    def test_valid(self) -> None:
        meta = VideoMetadata(frame_count=10, fps=29.97, width=1920, height=1080)
        assert meta.frame_count == 10
        assert meta.fps == pytest.approx(29.97)

    def test_zero_frame_count_allowed(self) -> None:
        # Broken or empty videos still produce a valid metadata object so
        # the caller can see frame_count == 0 rather than receiving an
        # exception.
        VideoMetadata(frame_count=0, fps=0.0, width=0, height=0)

    def test_rejects_negative(self) -> None:
        with pytest.raises(ValidationError):
            VideoMetadata(frame_count=-1, fps=30.0, width=640, height=480)

    def test_rejects_extra_fields(self) -> None:
        with pytest.raises(ValidationError):
            VideoMetadata(
                frame_count=1,
                fps=30.0,
                width=640,
                height=480,
                source_path="/leak/me",  # type: ignore[call-arg]
            )

    def test_is_frozen(self) -> None:
        meta = VideoMetadata(frame_count=10, fps=30.0, width=640, height=480)
        with pytest.raises(ValidationError):
            meta.fps = 60.0


# ---------------------------------------------------------------------------
# VideoPredictions
# ---------------------------------------------------------------------------


class TestVideoPredictions:
    def test_from_dict(self, video_predictions_payload: dict) -> None:
        vp = VideoPredictions.model_validate(video_predictions_payload)
        assert len(vp) == 2
        assert vp.frame_names() == ["frame_000000", "frame_000001"]
        assert vp["frame_000000"].boxes[0][4] == pytest.approx(0.95)
        assert vp.metadata.fps == pytest.approx(30.0)

    def test_iteration(self, video_predictions_payload: dict) -> None:
        vp = VideoPredictions.model_validate(video_predictions_payload)
        assert list(vp) == ["frame_000000", "frame_000001"]

    def test_rejects_missing_metadata(self, video_predictions_payload: dict) -> None:
        del video_predictions_payload["metadata"]
        with pytest.raises(ValidationError):
            VideoPredictions.model_validate(video_predictions_payload)

    def test_save_and_load_roundtrip(
        self,
        tmp_path: Path,
        video_predictions_payload: dict,
    ) -> None:
        vp = VideoPredictions.model_validate(video_predictions_payload)
        path = tmp_path / "preds" / "video.json"
        save_video_predictions(path, vp)
        assert path.exists()
        loaded = load_video_predictions(path)
        assert loaded.frame_names() == vp.frame_names()
        assert loaded.metadata == vp.metadata
        assert loaded["frame_000000"].poses3d == vp["frame_000000"].poses3d

    def test_save_is_atomic(
        self,
        tmp_path: Path,
        video_predictions_payload: dict,
    ) -> None:
        vp = VideoPredictions.model_validate(video_predictions_payload)
        path = tmp_path / "video.json"
        save_video_predictions(path, vp)
        assert path.exists()
        tmps = list(tmp_path.glob("video.json.tmp"))
        assert tmps == []


# ---------------------------------------------------------------------------
# JobResults
# ---------------------------------------------------------------------------


class TestJobResults:
    def test_save_and_load_roundtrip(
        self,
        tmp_path: Path,
        video_predictions_payload: dict,
    ) -> None:
        jr = JobResults.model_validate(
            {
                "video_a.mp4": video_predictions_payload,
                "video_b.mp4": video_predictions_payload,
            }
        )
        path = tmp_path / "results.json"
        save_job_results(path, jr)
        loaded = load_job_results(path)
        assert loaded.videos() == ["video_a.mp4", "video_b.mp4"]
        assert len(loaded["video_a.mp4"]) == 2


# ---------------------------------------------------------------------------
# Performance / benchmark schemas
# ---------------------------------------------------------------------------


def _make_metrics(
    *,
    total_seconds: float = 1.0,
    latencies: list[float] | None = None,
    peak_rss_mb: float = 512.0,
    active_device: str = "/CPU:0",
    metal_active: bool = False,
    model_load_seconds: float | None = None,
) -> PerformanceMetrics:
    return PerformanceMetrics(
        model_load_seconds=model_load_seconds,
        total_seconds=total_seconds,
        per_frame_latencies_ms=latencies if latencies is not None else [10.0, 11.0, 9.5],
        peak_rss_mb=peak_rss_mb,
        active_device=active_device,
        tensorflow_metal_active=metal_active,
        tensorflow_version="2.21.0",
    )


def _make_aggregate() -> BenchmarkAggregate:
    return BenchmarkAggregate(
        repeats_measured=4,
        warmup_frames_per_pass=3,
        mean_frame_latency_ms=10.0,
        p50_frame_latency_ms=9.8,
        p95_frame_latency_ms=12.5,
        p99_frame_latency_ms=13.0,
        stddev_frame_latency_ms=0.7,
        mean_throughput_fps=100.0,
        peak_rss_mb_max=512.0,
        active_device="/CPU:0",
        tensorflow_metal_active=False,
        tensorflow_version="2.21.0",
    )


class TestPerformanceMetricsModel:
    def test_roundtrip(self) -> None:
        m = _make_metrics()
        rehydrated = PerformanceMetrics.model_validate(m.model_dump(mode="json"))
        assert rehydrated == m

    def test_rejects_negative_total_seconds(self) -> None:
        with pytest.raises(ValidationError):
            PerformanceMetrics(
                total_seconds=-1.0,
                peak_rss_mb=0.0,
                active_device="/CPU:0",
                tensorflow_version="2.21.0",
            )

    def test_rejects_negative_peak_rss(self) -> None:
        with pytest.raises(ValidationError):
            PerformanceMetrics(
                total_seconds=1.0,
                peak_rss_mb=-5.0,
                active_device="/CPU:0",
                tensorflow_version="2.21.0",
            )

    def test_model_load_seconds_optional(self) -> None:
        m = _make_metrics(model_load_seconds=None)
        assert m.model_load_seconds is None

    def test_is_frozen(self) -> None:
        m = _make_metrics()
        with pytest.raises(ValidationError):
            m.total_seconds = 2.0


class TestBenchmarkResultPersistence:
    def test_roundtrip_to_disk(self, tmp_path: Path) -> None:
        result = BenchmarkResult(
            video_name="trial.mp4",
            repeats=5,
            warmup_frames=3,
            warmup_pass=_make_metrics(total_seconds=20.0),
            measured_passes=[_make_metrics(total_seconds=1.5) for _ in range(4)],
            aggregate=_make_aggregate(),
        )
        path = tmp_path / "bench.json"
        save_benchmark_result(path, result)
        assert path.exists()
        loaded = load_benchmark_result(path)
        assert loaded == result

    def test_rejects_repeats_below_one(self) -> None:
        with pytest.raises(ValidationError):
            BenchmarkResult(
                video_name="x.mp4",
                repeats=0,
                warmup_frames=0,
                warmup_pass=_make_metrics(),
                measured_passes=[],
                aggregate=_make_aggregate(),
            )

    def test_cpu_comparison_nested(self, tmp_path: Path) -> None:
        comparison = CpuComparisonResult(
            primary_aggregate=_make_aggregate(),
            cpu_aggregate=_make_aggregate(),
            speedup=2.5,
            max_poses3d_divergence_mm=0.002,
            frame_count_compared=30,
        )
        result = BenchmarkResult(
            video_name="trial.mp4",
            repeats=5,
            warmup_frames=3,
            warmup_pass=_make_metrics(),
            measured_passes=[_make_metrics() for _ in range(4)],
            aggregate=_make_aggregate(),
            cpu_comparison=comparison,
        )
        path = tmp_path / "bench_with_cmp.json"
        save_benchmark_result(path, result)
        loaded = load_benchmark_result(path)
        assert loaded.cpu_comparison is not None
        assert loaded.cpu_comparison.speedup == pytest.approx(2.5)
        assert loaded.cpu_comparison.max_poses3d_divergence_mm == pytest.approx(0.002)


# ---------------------------------------------------------------------------
# Segmentation schema
# ---------------------------------------------------------------------------


class TestSegmentModel:
    def test_valid(self) -> None:
        seg = Segment(start=0, end=30, peak=15)
        assert seg.start == 0
        assert seg.peak == 15
        assert seg.end == 30

    def test_rejects_end_not_greater_than_start(self) -> None:
        with pytest.raises(ValidationError, match="end"):
            Segment(start=10, end=10, peak=10)

    def test_peak_must_be_inside_window(self) -> None:
        with pytest.raises(ValidationError, match="peak"):
            Segment(start=0, end=30, peak=30)  # peak == end is out of range

    def test_is_frozen(self) -> None:
        seg = Segment(start=0, end=10, peak=5)
        with pytest.raises(ValidationError):
            seg.start = 1


class TestExtractorSpecs:
    def test_joint_pair_distance_rejects_identical_joints(self) -> None:
        with pytest.raises(ValidationError, match="distinct"):
            JointPairDistanceExtractor(joints=(7, 7))

    def test_joint_pair_distance_rejects_negative(self) -> None:
        with pytest.raises(ValidationError, match="non-negative"):
            JointPairDistanceExtractor(joints=(-1, 5))

    def test_joint_angle_rejects_negative(self) -> None:
        with pytest.raises(ValidationError, match="non-negative"):
            JointAngleExtractor(triplet=(0, -1, 2))

    def test_joint_axis_rejects_bad_axis(self) -> None:
        with pytest.raises(ValidationError):
            JointAxisExtractor(joint=0, axis=3)

    def test_discriminator_dispatches_to_correct_variant(self) -> None:
        # Round-trip each extractor variant through a SegmentationConfig
        # dict to confirm the discriminator selects the right class.
        for payload, cls in [
            ({"kind": "joint_axis", "joint": 1, "axis": 2}, JointAxisExtractor),
            ({"kind": "joint_pair_distance", "joints": [1, 2]}, JointPairDistanceExtractor),
            ({"kind": "joint_speed", "joint": 3}, JointSpeedExtractor),
            ({"kind": "joint_angle", "triplet": [1, 2, 3]}, JointAngleExtractor),
        ]:
            cfg = SegmentationConfig.model_validate({"extractor": payload})
            assert isinstance(cfg.extractor, cls)


class TestSegmentationPersistence:
    def test_roundtrip_through_video_predictions(
        self,
        tmp_path: Path,
        video_predictions_payload: dict,
    ) -> None:
        cfg = SegmentationConfig(
            extractor=JointAxisExtractor(joint=15, axis=1),
            min_prominence=20.0,
            pad_seconds=0.1,
        )
        segmentation = Segmentation(
            config=cfg,
            segments=[Segment(start=0, end=1, peak=0), Segment(start=1, end=2, peak=1)],
        )
        video_predictions_payload["segmentations"] = {
            "cup_lift": segmentation.model_dump(mode="json")
        }
        vp = VideoPredictions.model_validate(video_predictions_payload)
        path = tmp_path / "video.json"
        save_video_predictions(path, vp)
        loaded = load_video_predictions(path)
        assert "cup_lift" in loaded.segmentations
        cup = loaded.segmentations["cup_lift"]
        assert cup.config.extractor.kind == "joint_axis"
        assert len(cup.segments) == 2
        assert cup.config.method == "valley_to_valley_v1"

    def test_default_empty_segmentations_on_new_instance(
        self, video_predictions_payload: dict
    ) -> None:
        vp = VideoPredictions.model_validate(video_predictions_payload)
        assert vp.segmentations == {}

    def test_legacy_file_without_segmentations_loads_clean(
        self,
        tmp_path: Path,
        video_predictions_payload: dict,
    ) -> None:
        # Older predictions files never wrote the segmentations field;
        # make sure they still validate and deserialize as if they had
        # an empty mapping.
        assert "segmentations" not in video_predictions_payload
        path = tmp_path / "legacy.json"
        path.write_text(json.dumps(video_predictions_payload))
        vp = load_video_predictions(path)
        assert vp.segmentations == {}


# ---------------------------------------------------------------------------
# Status file
# ---------------------------------------------------------------------------


class TestStatusFile:
    def test_load_missing_returns_empty(self, tmp_path: Path) -> None:
        status = load_status(tmp_path / "nope.json")
        assert status.is_empty()

    def test_load_corrupt_json_returns_empty(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.json"
        path.write_text("{ not valid json")
        status = load_status(path)
        assert status.is_empty()

    def test_load_non_mapping_returns_empty(self, tmp_path: Path) -> None:
        path = tmp_path / "list.json"
        path.write_text(json.dumps([1, 2, 3]))
        status = load_status(path)
        assert status.is_empty()

    def test_save_and_load_completed_entry(self, tmp_path: Path) -> None:
        started = datetime(2026, 4, 13, 12, 0, 0, tzinfo=UTC)
        completed = datetime(2026, 4, 13, 12, 5, 0, tzinfo=UTC)
        status = StatusFile.model_validate(
            {
                "job_001": {
                    "status": "completed",
                    "started_at": started.isoformat(),
                    "completed_at": completed.isoformat(),
                    "results_path": "/tmp/results.json",
                    "error": None,
                }
            }
        )
        path = tmp_path / "status.json"
        save_status(path, status)
        loaded = load_status(path)
        entry = loaded.root["job_001"]
        assert entry.status == JobStatus.COMPLETED
        assert entry.started_at == started
        assert entry.completed_at == completed
        assert entry.error is None

    def test_save_is_atomic(self, tmp_path: Path) -> None:
        """``save_status`` leaves no orphan ``.tmp`` file on success."""
        started = datetime(2026, 4, 13, tzinfo=UTC)
        status = StatusFile.model_validate(
            {
                "job_001": {
                    "status": "processing",
                    "started_at": started.isoformat(),
                }
            }
        )
        path = tmp_path / "status.json"
        save_status(path, status)
        assert path.exists()
        tmps = list(tmp_path.glob("status.json.tmp"))
        assert tmps == []

    def test_failed_entry_carries_error_message(self, tmp_path: Path) -> None:
        started = datetime(2026, 4, 13, tzinfo=UTC)
        status = StatusFile.model_validate(
            {
                "job_001": {
                    "status": "failed",
                    "started_at": started.isoformat(),
                    "error": "ffmpeg decode failed: codec not supported",
                }
            }
        )
        path = tmp_path / "status.json"
        save_status(path, status)
        loaded = load_status(path)
        entry = loaded.root["job_001"]
        assert entry.status == JobStatus.FAILED
        assert entry.error is not None
        assert "ffmpeg" in entry.error

    def test_rejects_unknown_status(self, tmp_path: Path) -> None:
        with pytest.raises(ValidationError):
            StatusFile.model_validate(
                {
                    "job_001": {
                        "status": "some-unknown-state",
                        "started_at": datetime(2026, 4, 13, tzinfo=UTC).isoformat(),
                    }
                }
            )
