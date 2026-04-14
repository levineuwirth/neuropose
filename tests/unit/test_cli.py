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

from pathlib import Path

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
        raise NotImplementedError(
            "pending commit 11: MeTRAbs loader stubbed for unit testing"
        )

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
        assert "analyze" in result.output

    def test_help_flag(self, runner: CliRunner) -> None:
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == EXIT_OK
        assert "NeuroPose" in result.output

    def test_subcommand_help(self, runner: CliRunner) -> None:
        for subcommand in ("watch", "process", "analyze"):
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
# analyze
# ---------------------------------------------------------------------------


class TestAnalyze:
    def test_analyze_stub_exits_with_pending_message(
        self, runner: CliRunner, tmp_path: Path
    ) -> None:
        results_path = tmp_path / "results.json"
        results_path.write_text("{}")
        result = runner.invoke(app, ["analyze", str(results_path)])
        assert result.exit_code == EXIT_PENDING
        assert "commit 10" in result.output

    def test_analyze_requires_an_argument(self, runner: CliRunner) -> None:
        result = runner.invoke(app, ["analyze"])
        assert result.exit_code == EXIT_USAGE


# ---------------------------------------------------------------------------
# Exit-code module constants
# ---------------------------------------------------------------------------


class TestExitCodes:
    def test_exit_codes_are_distinct(self) -> None:
        codes = {EXIT_OK, EXIT_USAGE, EXIT_PENDING, EXIT_INTERRUPTED}
        assert len(codes) == 4
