"""Tests for :mod:`neuropose.cli`.

Uses ``typer.testing.CliRunner`` to invoke the CLI in-process and assert on
exit codes and captured output. The autouse ``_isolate_environment`` fixture
in ``conftest.py`` ensures each invocation uses an isolated ``$HOME`` /
``$XDG_DATA_HOME`` so the daemon's lock file and data dirs cannot bleed
between tests.

The tests deliberately check ``result.output`` (combined stdout + stderr)
rather than separate ``result.stderr``, because click's ``mix_stderr``
plumbing varies across 8.x releases and we want the tests to be version-
robust.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
import yaml
from typer.testing import CliRunner

from neuropose import __version__
from neuropose.cli import (
    EXIT_INTERRUPTED,
    EXIT_OK,
    EXIT_PENDING,
    EXIT_USAGE,
    app,
)
from neuropose.io import (
    JobResults,
    VideoPredictions,
    load_benchmark_result,
    load_job_results,
    load_video_predictions,
    save_job_results,
    save_video_predictions,
)


@pytest.fixture
def runner() -> CliRunner:
    """Return a default CliRunner.

    We do NOT set ``mix_stderr=False`` because that parameter's semantics
    changed between click 8.1 and 8.2. The default behaviour — combined
    output on ``result.output`` — is stable and good enough for the
    assertions this test suite makes.
    """
    return CliRunner()


@pytest.fixture
def stub_metrabs_loader(monkeypatch: pytest.MonkeyPatch) -> None:
    """Patch the MeTRAbs loader to raise a recognisable stub error.

    The real loader downloads a ~675 MB tarball and loads it through
    TensorFlow — neither suitable for unit tests. This fixture replaces
    it with a function that raises ``NotImplementedError`` tagged with a
    stable "pending commit 11" marker so the CLI's ``NotImplementedError``
    handler still has a testable failure mode. The handler itself is
    defensive code for any future stub; this fixture lets us keep it
    honest without reintroducing a real stub in production code.
    """

    def fake_loader(cache_dir: Path | None = None) -> object:
        del cache_dir
        raise NotImplementedError("pending commit 11: MeTRAbs loader stubbed for unit testing")

    monkeypatch.setattr("neuropose.estimator.load_metrabs_model", fake_loader)


# ---------------------------------------------------------------------------
# Top-level options
# ---------------------------------------------------------------------------


class TestTopLevelOptions:
    def test_version_flag(self, runner: CliRunner) -> None:
        result = runner.invoke(app, ["--version"])
        assert result.exit_code == EXIT_OK
        assert __version__ in result.output

    def test_no_args_shows_help(self, runner: CliRunner) -> None:
        result = runner.invoke(app, [])
        # Typer's ``no_args_is_help`` exits with a help message. The exit
        # code convention varies between click versions; accept either
        # success or a usage-style code.
        assert result.exit_code in (EXIT_OK, EXIT_USAGE)
        assert "watch" in result.output
        assert "process" in result.output
        assert "ingest" in result.output
        assert "serve" in result.output
        assert "segment" in result.output
        assert "benchmark" in result.output
        assert "analyze" in result.output

    def test_help_flag(self, runner: CliRunner) -> None:
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == EXIT_OK
        assert "NeuroPose" in result.output

    def test_subcommand_help(self, runner: CliRunner) -> None:
        for subcommand in (
            "watch",
            "process",
            "ingest",
            "serve",
            "segment",
            "benchmark",
            "analyze",
        ):
            result = runner.invoke(app, [subcommand, "--help"])
            assert result.exit_code == EXIT_OK, f"{subcommand} --help failed"

    def test_verbose_and_quiet_are_mutually_exclusive(self, runner: CliRunner) -> None:
        result = runner.invoke(app, ["--verbose", "--quiet", "watch"])
        assert result.exit_code != EXIT_OK


# ---------------------------------------------------------------------------
# --config handling
# ---------------------------------------------------------------------------


class TestConfigOption:
    def test_missing_config_file(self, runner: CliRunner, tmp_path: Path) -> None:
        result = runner.invoke(app, ["--config", str(tmp_path / "nope.yaml"), "watch"])
        assert result.exit_code == EXIT_USAGE
        # typer validates file existence via the Option's exists/readable
        # machinery, so the error message comes from click, not our handler.
        # We just verify it mentions the missing file's name.
        assert "nope.yaml" in result.output

    def test_invalid_config_yaml_structure(self, runner: CliRunner, tmp_path: Path) -> None:
        path = tmp_path / "bad.yaml"
        path.write_text("- not a mapping\n- another item\n")
        result = runner.invoke(app, ["--config", str(path), "watch"])
        assert result.exit_code == EXIT_USAGE
        lowered = result.output.lower()
        assert "invalid config" in lowered or "mapping" in lowered

    def test_invalid_config_field(self, runner: CliRunner, tmp_path: Path) -> None:
        path = tmp_path / "bad.yaml"
        path.write_text(yaml.safe_dump({"device": "cpu"}))
        result = runner.invoke(app, ["--config", str(path), "watch"])
        assert result.exit_code == EXIT_USAGE
        assert "invalid config" in result.output.lower()

    def test_valid_config_reaches_subcommand(
        self,
        runner: CliRunner,
        tmp_path: Path,
        stub_metrabs_loader: None,
    ) -> None:
        # A valid config should flow through the callback and into the
        # subcommand. ``watch`` will then reach the model-loading step, at
        # which point the stubbed loader raises ``NotImplementedError`` and
        # the CLI exits ``EXIT_PENDING``. Observing that exit code is how
        # we confirm the config made it all the way to the subcommand.
        del stub_metrabs_loader
        data_dir = tmp_path / "data"
        path = tmp_path / "good.yaml"
        path.write_text(
            yaml.safe_dump(
                {
                    "device": "/CPU:0",
                    "data_dir": str(data_dir),
                    "model_cache_dir": str(tmp_path / "models"),
                }
            )
        )
        result = runner.invoke(app, ["--config", str(path), "watch"])
        assert result.exit_code == EXIT_PENDING
        assert "commit 11" in result.output


# ---------------------------------------------------------------------------
# watch
# ---------------------------------------------------------------------------


class TestWatch:
    def test_without_model_exits_pending(
        self,
        runner: CliRunner,
        stub_metrabs_loader: None,
    ) -> None:
        """The stubbed loader raises ``NotImplementedError`` on model load.

        The CLI should catch it and exit with ``EXIT_PENDING`` and a message
        pointing at the pending commit. The real loader is patched out by
        ``stub_metrabs_loader`` so this test does not download the model.
        """
        del stub_metrabs_loader
        result = runner.invoke(app, ["watch"])
        assert result.exit_code == EXIT_PENDING
        assert "commit 11" in result.output


# ---------------------------------------------------------------------------
# process
# ---------------------------------------------------------------------------


class TestProcess:
    def test_missing_video_exits_usage(self, runner: CliRunner, tmp_path: Path) -> None:
        result = runner.invoke(app, ["process", str(tmp_path / "nope.mp4")])
        # click's path existence check fires before our callback, so the
        # exit is a usage error.
        assert result.exit_code == EXIT_USAGE

    def test_existing_video_without_model_exits_pending(
        self,
        runner: CliRunner,
        synthetic_video: Path,
        stub_metrabs_loader: None,
    ) -> None:
        del stub_metrabs_loader
        result = runner.invoke(app, ["process", str(synthetic_video)])
        assert result.exit_code == EXIT_PENDING
        assert "commit 11" in result.output


# ---------------------------------------------------------------------------
# ingest
# ---------------------------------------------------------------------------


def _write_zip_for_cli(path: Path, members: dict[str, bytes]) -> Path:
    """Build a zip fixture for the CLI ingest tests."""
    import zipfile

    with zipfile.ZipFile(path, "w") as z:
        for name, data in members.items():
            z.writestr(name, data)
    return path


class TestIngestSubcommand:
    def test_ingest_happy_path_creates_job_dirs(
        self,
        runner: CliRunner,
        tmp_path: Path,
        xdg_home: Path,
    ) -> None:
        archive = _write_zip_for_cli(
            tmp_path / "session.zip",
            {"clip_01.mp4": b"video one", "clip_02.mp4": b"video two"},
        )
        result = runner.invoke(app, ["ingest", str(archive)])
        assert result.exit_code == EXIT_OK, result.output
        assert "ingested 2 job(s)" in result.output
        # Default Settings.data_dir is $XDG_DATA_HOME/neuropose/jobs, and
        # input_dir = data_dir/in.
        input_dir = xdg_home / "neuropose" / "jobs" / "in"
        assert (input_dir / "clip_01" / "clip_01.mp4").read_bytes() == b"video one"
        assert (input_dir / "clip_02" / "clip_02.mp4").read_bytes() == b"video two"

    def test_ingest_reports_skipped_non_videos(
        self,
        runner: CliRunner,
        tmp_path: Path,
    ) -> None:
        archive = _write_zip_for_cli(
            tmp_path / "a.zip",
            {"clip.mp4": b"v", "README.md": b"readme"},
        )
        result = runner.invoke(app, ["ingest", str(archive)])
        assert result.exit_code == EXIT_OK, result.output
        assert "1 non-video member(s) skipped" in result.output

    def test_ingest_missing_archive_is_usage_error(
        self,
        runner: CliRunner,
        tmp_path: Path,
    ) -> None:
        result = runner.invoke(app, ["ingest", str(tmp_path / "nope.zip")])
        assert result.exit_code == EXIT_USAGE

    def test_ingest_empty_archive_is_usage_error(
        self,
        runner: CliRunner,
        tmp_path: Path,
    ) -> None:
        archive = _write_zip_for_cli(tmp_path / "empty.zip", {})
        result = runner.invoke(app, ["ingest", str(archive)])
        assert result.exit_code == EXIT_USAGE
        assert "no video files" in result.output.lower()

    def test_ingest_collision_without_force_is_usage_error(
        self,
        runner: CliRunner,
        tmp_path: Path,
        xdg_home: Path,
    ) -> None:
        archive = _write_zip_for_cli(tmp_path / "a.zip", {"clip.mp4": b"v"})
        # Pre-create the colliding job directory so the first run has
        # something to collide with.
        runner.invoke(app, ["ingest", str(archive)])
        result = runner.invoke(app, ["ingest", str(archive)])
        assert result.exit_code == EXIT_USAGE
        assert "already exist" in result.output.lower()
        assert "--force" in result.output

    def test_ingest_force_overwrites(
        self,
        runner: CliRunner,
        tmp_path: Path,
        xdg_home: Path,
    ) -> None:
        archive = _write_zip_for_cli(tmp_path / "a.zip", {"clip.mp4": b"original"})
        first = runner.invoke(app, ["ingest", str(archive)])
        assert first.exit_code == EXIT_OK, first.output
        # Build a new archive with the same job name but different bytes.
        archive2 = _write_zip_for_cli(tmp_path / "b.zip", {"clip.mp4": b"overwritten"})
        second = runner.invoke(app, ["ingest", str(archive2), "--force"])
        assert second.exit_code == EXIT_OK, second.output
        input_dir = xdg_home / "neuropose" / "jobs" / "in"
        assert (input_dir / "clip" / "clip.mp4").read_bytes() == b"overwritten"

    def test_ingest_traversal_rejected_as_usage_error(
        self,
        runner: CliRunner,
        tmp_path: Path,
    ) -> None:
        archive = _write_zip_for_cli(tmp_path / "a.zip", {"../escape.mp4": b"v"})
        result = runner.invoke(app, ["ingest", str(archive)])
        assert result.exit_code == EXIT_USAGE
        assert "traversal" in result.output.lower()


# ---------------------------------------------------------------------------
# serve
# ---------------------------------------------------------------------------


class TestServeSubcommand:
    def test_serve_exits_cleanly_on_keyboard_interrupt(
        self,
        runner: CliRunner,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """``neuropose serve`` should translate Ctrl-C into EXIT_INTERRUPTED.

        We patch ``serve_forever`` to raise ``KeyboardInterrupt``
        immediately, which simulates a Ctrl-C before any request is
        served. The CLI's handler should map that to the standard
        shell-interruption exit code.
        """

        def fake_serve_forever(status_path, *, host, port) -> None:
            del status_path, host, port
            raise KeyboardInterrupt

        monkeypatch.setattr("neuropose.monitor.serve_forever", fake_serve_forever)
        result = runner.invoke(app, ["serve"])
        assert result.exit_code == EXIT_INTERRUPTED

    def test_serve_bind_error_is_usage_error(
        self,
        runner: CliRunner,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A port-already-in-use OSError should surface as EXIT_USAGE."""

        def fake_serve_forever(status_path, *, host, port) -> None:
            del status_path, host, port
            raise OSError("address already in use")

        monkeypatch.setattr("neuropose.monitor.serve_forever", fake_serve_forever)
        result = runner.invoke(app, ["serve"])
        assert result.exit_code == EXIT_USAGE
        assert "could not bind" in result.output.lower()


# ---------------------------------------------------------------------------
# segment
# ---------------------------------------------------------------------------


_NUM_JOINTS = 43


def _triple_hump_predictions(joint_name: str = "lwri") -> VideoPredictions:
    """Build a 300-frame synthetic VideoPredictions with three clear humps."""
    from neuropose.analyzer.segment import JOINT_INDEX

    joint = JOINT_INDEX[joint_name]
    t = np.linspace(0.0, 6.0 * math.pi, 300)
    signal = np.maximum(0.0, np.sin(t)) ** 2 * 1000.0

    frames = {}
    for i, value in enumerate(signal):
        poses = [[[0.0, 0.0, 0.0] for _ in range(_NUM_JOINTS)]]
        poses[0][joint][1] = float(value)
        frames[f"frame_{i:06d}"] = {
            "boxes": [[0.0, 0.0, 1.0, 1.0, 0.9]],
            "poses3d": poses,
            "poses2d": [[[0.0, 0.0]] * _NUM_JOINTS],
        }
    return VideoPredictions.model_validate(
        {
            "metadata": {
                "frame_count": 300,
                "fps": 30.0,
                "width": 640,
                "height": 480,
            },
            "frames": frames,
        }
    )


class TestSegmentSubcommand:
    def test_segment_job_results_in_place(self, runner: CliRunner, tmp_path: Path) -> None:
        path = tmp_path / "results.json"
        save_job_results(
            path,
            JobResults(root={"trial_01.mp4": _triple_hump_predictions()}),
        )

        result = runner.invoke(
            app,
            [
                "segment",
                str(path),
                "--name",
                "cup_lift",
                "--extractor",
                "joint_axis",
                "--joint",
                "lwri",
                "--axis",
                "1",
                "--min-prominence",
                "50",
            ],
        )

        assert result.exit_code == EXIT_OK, result.output
        loaded = load_job_results(path)
        vp = loaded["trial_01.mp4"]
        assert "cup_lift" in vp.segmentations
        assert len(vp.segmentations["cup_lift"].segments) == 3

    def test_segment_single_predictions_file(self, runner: CliRunner, tmp_path: Path) -> None:
        path = tmp_path / "video_predictions.json"
        save_video_predictions(path, _triple_hump_predictions())

        result = runner.invoke(
            app,
            [
                "segment",
                str(path),
                "--name",
                "cup_lift",
                "--extractor",
                "joint_pair_distance",
                "--joints",
                "lwri,rwri",
                "--min-prominence",
                "50",
            ],
        )
        # With both wrists at origin except for lwri's y-coordinate, the
        # pair distance is exactly the lwri.y signal, so the three humps
        # are detected the same way as the joint_axis case.
        assert result.exit_code == EXIT_OK, result.output
        loaded = load_video_predictions(path)
        assert "cup_lift" in loaded.segmentations
        assert len(loaded.segmentations["cup_lift"].segments) == 3

    def test_segment_writes_to_output_option(self, runner: CliRunner, tmp_path: Path) -> None:
        src = tmp_path / "src.json"
        dst = tmp_path / "dst.json"
        save_job_results(src, JobResults(root={"trial_01.mp4": _triple_hump_predictions()}))

        result = runner.invoke(
            app,
            [
                "segment",
                str(src),
                "--name",
                "cup_lift",
                "--extractor",
                "joint_axis",
                "--joint",
                "15",  # integer form
                "--axis",
                "1",
                "--min-prominence",
                "50",
                "--output",
                str(dst),
            ],
        )

        assert result.exit_code == EXIT_OK, result.output
        assert dst.exists()
        # Source must be left untouched when --output is given.
        src_loaded = load_job_results(src)
        assert src_loaded["trial_01.mp4"].segmentations == {}
        dst_loaded = load_job_results(dst)
        assert "cup_lift" in dst_loaded["trial_01.mp4"].segmentations

    def test_segment_rejects_collision_without_force(
        self, runner: CliRunner, tmp_path: Path
    ) -> None:
        path = tmp_path / "results.json"
        save_job_results(path, JobResults(root={"trial_01.mp4": _triple_hump_predictions()}))
        base_args = [
            "segment",
            str(path),
            "--name",
            "cup_lift",
            "--extractor",
            "joint_axis",
            "--joint",
            "lwri",
            "--axis",
            "1",
            "--min-prominence",
            "50",
        ]

        first = runner.invoke(app, base_args)
        assert first.exit_code == EXIT_OK, first.output

        collision = runner.invoke(app, base_args)
        assert collision.exit_code == EXIT_USAGE
        assert "force" in collision.output.lower()

    def test_segment_force_overwrites(self, runner: CliRunner, tmp_path: Path) -> None:
        path = tmp_path / "results.json"
        save_job_results(path, JobResults(root={"trial_01.mp4": _triple_hump_predictions()}))
        args = [
            "segment",
            str(path),
            "--name",
            "cup_lift",
            "--extractor",
            "joint_axis",
            "--joint",
            "lwri",
            "--axis",
            "1",
            "--min-prominence",
            "50",
        ]
        runner.invoke(app, args)
        result = runner.invoke(app, [*args, "--force"])
        assert result.exit_code == EXIT_OK, result.output

    def test_segment_unknown_joint_name_is_usage_error(
        self, runner: CliRunner, tmp_path: Path
    ) -> None:
        path = tmp_path / "results.json"
        save_job_results(path, JobResults(root={"trial_01.mp4": _triple_hump_predictions()}))

        result = runner.invoke(
            app,
            [
                "segment",
                str(path),
                "--name",
                "cup_lift",
                "--extractor",
                "joint_axis",
                "--joint",
                "elbow",  # deliberately not a valid berkeley_mhad_43 name
                "--axis",
                "1",
            ],
        )

        assert result.exit_code == EXIT_USAGE
        assert "unknown joint" in result.output.lower()

    def test_segment_missing_required_flag_is_usage_error(
        self, runner: CliRunner, tmp_path: Path
    ) -> None:
        path = tmp_path / "results.json"
        save_job_results(path, JobResults(root={"trial_01.mp4": _triple_hump_predictions()}))

        # joint_axis without --joint
        result = runner.invoke(
            app,
            [
                "segment",
                str(path),
                "--name",
                "cup_lift",
                "--extractor",
                "joint_axis",
                "--axis",
                "1",
            ],
        )
        assert result.exit_code == EXIT_USAGE

    def test_segment_unreadable_file_is_usage_error(
        self, runner: CliRunner, tmp_path: Path
    ) -> None:
        path = tmp_path / "garbage.json"
        path.write_text("{ this is not valid json")
        result = runner.invoke(
            app,
            [
                "segment",
                str(path),
                "--name",
                "cup_lift",
                "--extractor",
                "joint_axis",
                "--joint",
                "lwri",
                "--axis",
                "1",
            ],
        )
        assert result.exit_code == EXIT_USAGE

    def test_segment_preserves_pose_values(self, runner: CliRunner, tmp_path: Path) -> None:
        """Segmentation only touches ``segmentations``; pose data is untouched."""
        path = tmp_path / "results.json"
        original = _triple_hump_predictions()
        save_job_results(path, JobResults(root={"trial_01.mp4": original}))

        result = runner.invoke(
            app,
            [
                "segment",
                str(path),
                "--name",
                "cup_lift",
                "--extractor",
                "joint_axis",
                "--joint",
                "lwri",
                "--axis",
                "1",
                "--min-prominence",
                "50",
            ],
        )
        assert result.exit_code == EXIT_OK, result.output

        loaded = load_job_results(path)
        lv = loaded["trial_01.mp4"]
        assert lv.frame_names() == original.frame_names()
        for name in original.frame_names():
            assert lv[name].poses3d == original[name].poses3d


# ---------------------------------------------------------------------------
# benchmark
# ---------------------------------------------------------------------------


@pytest.fixture
def stub_estimator_with_metrics(monkeypatch: pytest.MonkeyPatch):
    """Monkeypatch the benchmark's Estimator path to use a fake model.

    The CLI's ``benchmark`` subcommand instantiates an :class:`Estimator`
    and calls ``load_model``. We replace ``load_metrabs_model`` with a
    fake that returns a deterministic stand-in so the CLI's
    ``load_model`` succeeds without downloading or touching TF. The
    estimator's own metrics-collection path still runs, so the CLI
    exercise is end-to-end except for the real model.
    """
    import numpy as np

    class RecordingFake:
        def detect_poses(self, image, **kwargs):
            del image, kwargs
            return {
                "boxes": np.array([[0.0, 0.0, 1.0, 1.0, 0.9]]),
                "poses3d": np.array([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]]),
                "poses2d": np.array([[[0.0, 0.0], [1.0, 1.0]]]),
            }

    from neuropose._model import LoadedModel

    def fake_loader(cache_dir: Path | None = None) -> LoadedModel:
        del cache_dir
        return LoadedModel(
            model=RecordingFake(),
            sha256="smoke_sha",
            filename="metrabs_smoke.tar.gz",
        )

    monkeypatch.setattr("neuropose.estimator.load_metrabs_model", fake_loader)


class TestBenchmarkSubcommand:
    def test_benchmark_smoke(
        self,
        runner: CliRunner,
        synthetic_video: Path,
        tmp_path: Path,
        stub_estimator_with_metrics,
    ) -> None:
        del stub_estimator_with_metrics
        output = tmp_path / "bench.json"
        result = runner.invoke(
            app,
            [
                "benchmark",
                str(synthetic_video),
                "--repeats",
                "3",
                "--warmup-frames",
                "0",
                "--output",
                str(output),
            ],
        )
        assert result.exit_code == EXIT_OK, result.output
        # Human-readable report must hit stdout.
        assert "Benchmark:" in result.output
        assert "Throughput:" in result.output
        # JSON output file must exist and validate against the schema.
        assert output.exists()
        loaded = load_benchmark_result(output)
        assert loaded.video_name == synthetic_video.name
        assert loaded.repeats == 3
        assert len(loaded.measured_passes) == 2
        assert loaded.cpu_comparison is None

    def test_benchmark_rejects_repeats_below_two(
        self,
        runner: CliRunner,
        synthetic_video: Path,
        stub_estimator_with_metrics,
    ) -> None:
        del stub_estimator_with_metrics
        result = runner.invoke(app, ["benchmark", str(synthetic_video), "--repeats", "1"])
        # Typer's min=2 validation catches this before our code runs.
        assert result.exit_code != EXIT_OK

    def test_benchmark_missing_video_is_usage_error(
        self,
        runner: CliRunner,
        tmp_path: Path,
    ) -> None:
        result = runner.invoke(app, ["benchmark", str(tmp_path / "nope.mp4")])
        assert result.exit_code == EXIT_USAGE

    def test_benchmark_force_cpu_and_compare_cpu_are_mutually_exclusive(
        self,
        runner: CliRunner,
        synthetic_video: Path,
        stub_estimator_with_metrics,
    ) -> None:
        del stub_estimator_with_metrics
        result = runner.invoke(
            app,
            [
                "benchmark",
                str(synthetic_video),
                "--compare-cpu",
                "--force-cpu",
            ],
        )
        assert result.exit_code == EXIT_USAGE
        assert "mutually exclusive" in result.output


# ---------------------------------------------------------------------------
# analyze
# ---------------------------------------------------------------------------


class TestAnalyze:
    """Covers the ``neuropose analyze --config <yaml>`` subcommand.

    Execution happy path is exercised in detail in
    :mod:`tests.unit.test_analyzer_pipeline` — this file focuses on
    the CLI wiring: argument parsing, config-loading error modes, and
    end-to-end smoke.
    """

    def _make_predictions_file(self, tmp_path: Path, name: str, num_frames: int = 30) -> Path:
        """Write a trivial VideoPredictions file to disk for the CLI to load."""
        import math

        from neuropose.io import VideoPredictions, save_video_predictions

        num_joints = 43
        frames = {}
        for i in range(num_frames):
            poses = [[[0.0, 0.0, 0.0] for _ in range(num_joints)]]
            poses[0][41][1] = float(math.sin(i * 0.3)) * 100.0  # rhee Y
            frames[f"frame_{i:06d}"] = {
                "boxes": [[0.0, 0.0, 1.0, 1.0, 0.9]],
                "poses3d": poses,
                "poses2d": [[[0.0, 0.0]] * num_joints],
            }
        preds = VideoPredictions.model_validate(
            {
                "metadata": {
                    "frame_count": num_frames,
                    "fps": 30.0,
                    "width": 640,
                    "height": 480,
                },
                "frames": frames,
            }
        )
        path = tmp_path / name
        save_video_predictions(path, preds)
        return path

    def _write_dtw_config(
        self,
        tmp_path: Path,
        *,
        primary: Path,
        reference: Path,
        report: Path,
    ) -> Path:
        import yaml as _yaml

        config_path = tmp_path / "config.yaml"
        config_path.write_text(
            _yaml.safe_dump(
                {
                    "inputs": {"primary": str(primary), "reference": str(reference)},
                    "analysis": {"kind": "dtw", "method": "dtw_all"},
                    "output": {"report": str(report)},
                }
            )
        )
        return config_path

    def test_missing_config_is_usage_error(self, runner: CliRunner, tmp_path: Path) -> None:
        result = runner.invoke(app, ["analyze", "--config", str(tmp_path / "nope.yaml")])
        assert result.exit_code == EXIT_USAGE
        assert "config file not found" in result.output

    def test_missing_config_flag_is_usage_error(self, runner: CliRunner) -> None:
        result = runner.invoke(app, ["analyze"])
        assert result.exit_code == EXIT_USAGE

    def test_invalid_yaml_is_usage_error(self, runner: CliRunner, tmp_path: Path) -> None:
        bad = tmp_path / "bad.yaml"
        bad.write_text("inputs: {primary: foo\n")  # unclosed flow mapping
        result = runner.invoke(app, ["analyze", "--config", str(bad)])
        assert result.exit_code == EXIT_USAGE
        assert "could not parse YAML" in result.output

    def test_schema_violation_is_usage_error(self, runner: CliRunner, tmp_path: Path) -> None:
        import yaml as _yaml

        bad = tmp_path / "schema.yaml"
        bad.write_text(
            _yaml.safe_dump(
                {
                    "inputs": {"primary": str(tmp_path / "a.json")},
                    # dtw without reference — violates cross-field invariant.
                    "analysis": {"kind": "dtw", "method": "dtw_all"},
                    "output": {"report": str(tmp_path / "r.json")},
                }
            )
        )
        result = runner.invoke(app, ["analyze", "--config", str(bad)])
        assert result.exit_code == EXIT_USAGE
        assert "invalid config" in result.output

    def test_happy_path_writes_report(self, runner: CliRunner, tmp_path: Path) -> None:
        primary = self._make_predictions_file(tmp_path, "a.json")
        reference = self._make_predictions_file(tmp_path, "b.json")
        report_path = tmp_path / "report.json"
        config = self._write_dtw_config(
            tmp_path, primary=primary, reference=reference, report=report_path
        )
        result = runner.invoke(app, ["analyze", "--config", str(config)])
        assert result.exit_code == EXIT_OK, result.output
        assert report_path.exists()
        assert "wrote analysis report" in result.output
        assert "analysis kind: dtw" in result.output

    def test_output_option_overrides_config_path(self, runner: CliRunner, tmp_path: Path) -> None:
        primary = self._make_predictions_file(tmp_path, "a.json")
        reference = self._make_predictions_file(tmp_path, "b.json")
        # Config points at one report path ...
        config = self._write_dtw_config(
            tmp_path,
            primary=primary,
            reference=reference,
            report=tmp_path / "declared.json",
        )
        # ... but --output overrides.
        override = tmp_path / "override.json"
        result = runner.invoke(app, ["analyze", "--config", str(config), "--output", str(override)])
        assert result.exit_code == EXIT_OK, result.output
        assert override.exists()
        assert not (tmp_path / "declared.json").exists()

    def test_missing_predictions_file_is_usage_error(
        self, runner: CliRunner, tmp_path: Path
    ) -> None:
        # Config points at a primary that does not exist.
        config = self._write_dtw_config(
            tmp_path,
            primary=tmp_path / "missing_primary.json",
            reference=tmp_path / "missing_reference.json",
            report=tmp_path / "report.json",
        )
        result = runner.invoke(app, ["analyze", "--config", str(config)])
        assert result.exit_code == EXIT_USAGE


# ---------------------------------------------------------------------------
# Exit-code module constants
# ---------------------------------------------------------------------------


class TestExitCodes:
    def test_exit_codes_are_distinct(self) -> None:
        codes = {EXIT_OK, EXIT_USAGE, EXIT_PENDING, EXIT_INTERRUPTED}
        assert len(codes) == 4
