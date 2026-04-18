"""Tests for :mod:`neuropose.migrations`.

Covers both the low-level migration driver (version walking, future/missing
errors, INFO logging) and its integration through the
:mod:`neuropose.io` load helpers (legacy payloads round-trip; future
payloads fail with a clear message).

The migration driver is tested by monkey-patching ``CURRENT_VERSION`` and
the per-schema migration registries, so the tests exercise the full
chain-walking machinery without needing the codebase to actually be on a
non-initial schema version.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

from neuropose import migrations
from neuropose.io import (
    BenchmarkResult,
    FramePrediction,
    JobResults,
    VideoMetadata,
    VideoPredictions,
    load_benchmark_result,
    load_job_results,
    load_video_predictions,
    save_benchmark_result,
    save_job_results,
    save_video_predictions,
)
from neuropose.migrations import (
    CURRENT_VERSION,
    FutureSchemaError,
    MigrationError,
    MigrationNotFoundError,
    migrate_benchmark_result,
    migrate_job_results,
    migrate_video_predictions,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _minimal_video_predictions_payload() -> dict:
    """A valid VideoPredictions payload at the current schema version."""
    return {
        "schema_version": CURRENT_VERSION,
        "metadata": {
            "frame_count": 1,
            "fps": 30.0,
            "width": 32,
            "height": 32,
        },
        "frames": {
            "frame_000000": {
                "boxes": [[0.0, 0.0, 32.0, 32.0, 0.95]],
                "poses3d": [[[1.0, 2.0, 3.0]]],
                "poses2d": [[[10.0, 20.0]]],
            }
        },
        "segmentations": {},
    }


def _minimal_video_predictions_object() -> VideoPredictions:
    """Same payload, as a validated pydantic object."""
    return VideoPredictions(
        metadata=VideoMetadata(frame_count=1, fps=30.0, width=32, height=32),
        frames={
            "frame_000000": FramePrediction(
                boxes=[[0.0, 0.0, 32.0, 32.0, 0.95]],
                poses3d=[[[1.0, 2.0, 3.0]]],
                poses2d=[[[10.0, 20.0]]],
            )
        },
    )


@pytest.fixture
def fake_two_version_chain(monkeypatch: pytest.MonkeyPatch) -> None:
    """Patch the module to look like CURRENT_VERSION=2 with a v1->v2 migration.

    Lets the tests exercise the full migration loop even though the real
    codebase is still at CURRENT_VERSION=1.
    """
    monkeypatch.setattr(migrations, "CURRENT_VERSION", 2)

    def _v1_to_v2(payload: dict) -> dict:
        payload = dict(payload)
        payload["schema_version"] = 2
        payload["added_in_v2"] = "hello"
        return payload

    monkeypatch.setattr(migrations, "_VIDEO_PREDICTIONS_MIGRATIONS", {1: _v1_to_v2})
    monkeypatch.setattr(migrations, "_BENCHMARK_RESULT_MIGRATIONS", {1: _v1_to_v2})


@pytest.fixture
def fake_three_version_chain(monkeypatch: pytest.MonkeyPatch) -> None:
    """Patch the module to look like CURRENT_VERSION=3 with v1->v2 and v2->v3.

    Exercises multi-step migration chaining.
    """
    monkeypatch.setattr(migrations, "CURRENT_VERSION", 3)

    def _v1_to_v2(payload: dict) -> dict:
        payload = dict(payload)
        payload["schema_version"] = 2
        payload["added_in_v2"] = "alpha"
        return payload

    def _v2_to_v3(payload: dict) -> dict:
        payload = dict(payload)
        payload["schema_version"] = 3
        payload["added_in_v3"] = "beta"
        return payload

    monkeypatch.setattr(
        migrations,
        "_VIDEO_PREDICTIONS_MIGRATIONS",
        {1: _v1_to_v2, 2: _v2_to_v3},
    )


# ---------------------------------------------------------------------------
# migrate_video_predictions — driver behavior
# ---------------------------------------------------------------------------


class TestMigrateVideoPredictions:
    def test_current_version_payload_is_noop(self) -> None:
        payload = {"schema_version": CURRENT_VERSION, "hello": "world"}
        result = migrate_video_predictions(payload)
        assert result == payload

    def test_missing_version_key_treated_as_v1(self) -> None:
        """A payload with no schema_version is treated as version 1.

        With CURRENT_VERSION == 2, the legacy payload is run through
        the registered v1 → v2 migration (which stamps ``provenance =
        None``) on the way to the current version.
        """
        payload = {"hello": "world"}
        result = migrate_video_predictions(payload)
        assert result["hello"] == "world"
        assert result["schema_version"] == CURRENT_VERSION
        assert result["provenance"] is None

    def test_future_version_raises(self) -> None:
        payload = {"schema_version": CURRENT_VERSION + 99}
        with pytest.raises(FutureSchemaError, match="newer than"):
            migrate_video_predictions(payload)

    def test_non_integer_version_raises(self) -> None:
        payload = {"schema_version": "1.0"}
        with pytest.raises(MigrationError, match="invalid schema_version"):
            migrate_video_predictions(payload)

    def test_zero_version_raises(self) -> None:
        payload = {"schema_version": 0}
        with pytest.raises(MigrationError, match="invalid schema_version"):
            migrate_video_predictions(payload)

    def test_single_step_migration(self, fake_two_version_chain: None) -> None:
        del fake_two_version_chain
        payload = {"schema_version": 1, "original_field": "keep_me"}
        result = migrate_video_predictions(payload)
        assert result == {
            "schema_version": 2,
            "original_field": "keep_me",
            "added_in_v2": "hello",
        }

    def test_missing_version_under_patched_chain_migrates_from_v1(
        self, fake_two_version_chain: None
    ) -> None:
        del fake_two_version_chain
        # Legacy file with no version stamp: should be treated as v1 and
        # upgraded to v2.
        payload = {"legacy": True}
        result = migrate_video_predictions(payload)
        assert result["schema_version"] == 2
        assert result["added_in_v2"] == "hello"
        assert result["legacy"] is True

    def test_multi_step_migration_chains(self, fake_three_version_chain: None) -> None:
        del fake_three_version_chain
        payload = {"schema_version": 1, "original": "yes"}
        result = migrate_video_predictions(payload)
        assert result == {
            "schema_version": 3,
            "original": "yes",
            "added_in_v2": "alpha",
            "added_in_v3": "beta",
        }

    def test_missing_intermediate_migration_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """If CURRENT advances past a version with no registered migration, fail loud."""
        monkeypatch.setattr(migrations, "CURRENT_VERSION", 3)
        # Only v1 -> v2 registered; v2 -> v3 is the missing link.
        monkeypatch.setattr(
            migrations,
            "_VIDEO_PREDICTIONS_MIGRATIONS",
            {1: lambda p: {**p, "schema_version": 2}},
        )
        with pytest.raises(MigrationNotFoundError, match="from schema_version 2"):
            migrate_video_predictions({"schema_version": 1})

    def test_logs_at_info_on_migration(
        self,
        fake_two_version_chain: None,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        del fake_two_version_chain
        caplog.set_level(logging.INFO, logger="neuropose.migrations")
        migrate_video_predictions({"schema_version": 1})
        assert any("Migrating VideoPredictions" in record.message for record in caplog.records)

    def test_starting_from_current_logs_nothing(
        self,
        fake_two_version_chain: None,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        del fake_two_version_chain
        caplog.set_level(logging.INFO, logger="neuropose.migrations")
        migrate_video_predictions({"schema_version": 2})
        assert not any("Migrating" in record.message for record in caplog.records)


# ---------------------------------------------------------------------------
# migrate_benchmark_result — same driver, sibling registry
# ---------------------------------------------------------------------------


class TestMigrateBenchmarkResult:
    def test_uses_benchmark_registry_not_video_registry(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Each schema has its own migration registry; they must not cross-pollinate."""
        monkeypatch.setattr(migrations, "CURRENT_VERSION", 2)
        # Register video migration but NOT benchmark migration.
        monkeypatch.setattr(
            migrations,
            "_VIDEO_PREDICTIONS_MIGRATIONS",
            {1: lambda p: {**p, "schema_version": 2, "from_video_registry": True}},
        )
        monkeypatch.setattr(migrations, "_BENCHMARK_RESULT_MIGRATIONS", {})
        # Video migration works:
        assert migrate_video_predictions({"schema_version": 1})["from_video_registry"] is True
        # Benchmark migration should fail — no entry in its registry.
        with pytest.raises(MigrationNotFoundError):
            migrate_benchmark_result({"schema_version": 1})


# ---------------------------------------------------------------------------
# migrate_job_results — per-entry dispatch
# ---------------------------------------------------------------------------


class TestMigrateJobResults:
    def test_empty_dict_is_noop(self) -> None:
        assert migrate_job_results({}) == {}

    def test_each_video_is_migrated(self, fake_two_version_chain: None) -> None:
        del fake_two_version_chain
        payload = {
            "video_a.mp4": {"schema_version": 1, "content_a": True},
            "video_b.mp4": {"schema_version": 1, "content_b": True},
        }
        result = migrate_job_results(payload)
        assert result["video_a.mp4"]["schema_version"] == 2
        assert result["video_a.mp4"]["content_a"] is True
        assert result["video_a.mp4"]["added_in_v2"] == "hello"
        assert result["video_b.mp4"]["schema_version"] == 2
        assert result["video_b.mp4"]["content_b"] is True


# ---------------------------------------------------------------------------
# register_video_predictions_migration / register_benchmark_result_migration
# ---------------------------------------------------------------------------


class TestRegistration:
    def test_duplicate_registration_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(migrations, "_VIDEO_PREDICTIONS_MIGRATIONS", {})

        @migrations.register_video_predictions_migration(from_version=1)
        def _first(p: dict) -> dict:
            return p

        with pytest.raises(RuntimeError, match="already registered"):

            @migrations.register_video_predictions_migration(from_version=1)
            def _second(p: dict) -> dict:
                return p

    def test_decorator_returns_callable_unchanged(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The decorator must not wrap or rename the function."""
        monkeypatch.setattr(migrations, "_VIDEO_PREDICTIONS_MIGRATIONS", {})

        @migrations.register_video_predictions_migration(from_version=1)
        def _fn(p: dict) -> dict:
            return p

        assert _fn.__name__ == "_fn"
        assert _fn({"x": 1}) == {"x": 1}


# ---------------------------------------------------------------------------
# Integration: load_* functions run migrations before validation
# ---------------------------------------------------------------------------


class TestLoadIntegration:
    def test_load_video_predictions_accepts_legacy_payload(self, tmp_path: Path) -> None:
        """A VideoPredictions JSON written before schema_version existed loads cleanly."""
        legacy = _minimal_video_predictions_payload()
        del legacy["schema_version"]  # Pretend this file predates versioning.
        path = tmp_path / "legacy.json"
        path.write_text(json.dumps(legacy))

        loaded = load_video_predictions(path)
        assert loaded.schema_version == CURRENT_VERSION

    def test_load_video_predictions_rejects_future_version(self, tmp_path: Path) -> None:
        payload = _minimal_video_predictions_payload()
        payload["schema_version"] = CURRENT_VERSION + 42
        path = tmp_path / "future.json"
        path.write_text(json.dumps(payload))

        with pytest.raises(FutureSchemaError):
            load_video_predictions(path)

    def test_save_then_load_roundtrips(self, tmp_path: Path) -> None:
        obj = _minimal_video_predictions_object()
        path = tmp_path / "out.json"
        save_video_predictions(path, obj)
        loaded = load_video_predictions(path)
        assert loaded == obj
        assert loaded.schema_version == CURRENT_VERSION

    def test_load_job_results_migrates_each_video(self, tmp_path: Path) -> None:
        video_a = _minimal_video_predictions_payload()
        video_b = _minimal_video_predictions_payload()
        # Strip schema_version from both to simulate legacy file.
        del video_a["schema_version"]
        del video_b["schema_version"]
        payload = {"a.mp4": video_a, "b.mp4": video_b}
        path = tmp_path / "job.json"
        path.write_text(json.dumps(payload))

        loaded = load_job_results(path)
        assert len(loaded) == 2
        for video in ("a.mp4", "b.mp4"):
            assert loaded[video].schema_version == CURRENT_VERSION

    def test_save_then_load_job_results_roundtrips(self, tmp_path: Path) -> None:
        obj = JobResults(root={"video_a.mp4": _minimal_video_predictions_object()})
        path = tmp_path / "job.json"
        save_job_results(path, obj)
        loaded = load_job_results(path)
        assert loaded == obj

    def test_load_benchmark_result_roundtrips(self, tmp_path: Path) -> None:
        """Save → load round-trip for a realistic benchmark result."""
        from neuropose.io import BenchmarkAggregate, PerformanceMetrics

        metrics = PerformanceMetrics(
            total_seconds=1.0,
            per_frame_latencies_ms=[10.0, 11.0],
            peak_rss_mb=100.0,
            active_device="/CPU:0",
            tensorflow_version="2.18.0",
        )
        aggregate = BenchmarkAggregate(
            repeats_measured=1,
            warmup_frames_per_pass=0,
            mean_frame_latency_ms=10.5,
            p50_frame_latency_ms=10.5,
            p95_frame_latency_ms=11.0,
            p99_frame_latency_ms=11.0,
            stddev_frame_latency_ms=0.5,
            mean_throughput_fps=95.0,
            peak_rss_mb_max=100.0,
            active_device="/CPU:0",
            tensorflow_version="2.18.0",
        )
        result = BenchmarkResult(
            video_name="test.mp4",
            repeats=2,
            warmup_frames=0,
            warmup_pass=metrics,
            measured_passes=[metrics],
            aggregate=aggregate,
        )
        path = tmp_path / "bench.json"
        save_benchmark_result(path, result)
        loaded = load_benchmark_result(path)
        assert loaded == result
        assert loaded.schema_version == CURRENT_VERSION

    def test_load_benchmark_result_rejects_future_version(self, tmp_path: Path) -> None:
        """Future-versioned benchmark file should raise with a clear message."""
        from neuropose.io import BenchmarkAggregate, PerformanceMetrics

        metrics = PerformanceMetrics(
            total_seconds=1.0,
            per_frame_latencies_ms=[10.0],
            peak_rss_mb=100.0,
            active_device="/CPU:0",
            tensorflow_version="2.18.0",
        )
        aggregate = BenchmarkAggregate(
            repeats_measured=1,
            warmup_frames_per_pass=0,
            mean_frame_latency_ms=10.0,
            p50_frame_latency_ms=10.0,
            p95_frame_latency_ms=10.0,
            p99_frame_latency_ms=10.0,
            stddev_frame_latency_ms=0.0,
            mean_throughput_fps=100.0,
            peak_rss_mb_max=100.0,
            active_device="/CPU:0",
            tensorflow_version="2.18.0",
        )
        result = BenchmarkResult(
            video_name="x.mp4",
            repeats=1,
            warmup_frames=0,
            warmup_pass=metrics,
            measured_passes=[metrics],
            aggregate=aggregate,
        )
        # Serialize then hand-edit to inject a future version.
        payload = result.model_dump(mode="json")
        payload["schema_version"] = CURRENT_VERSION + 1
        path = tmp_path / "bench_future.json"
        path.write_text(json.dumps(payload))

        with pytest.raises(FutureSchemaError):
            load_benchmark_result(path)
