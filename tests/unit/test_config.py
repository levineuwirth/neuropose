"""Tests for :mod:`neuropose.config`."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from neuropose.config import Settings


class TestDefaults:
    """Default values wire through XDG correctly."""

    def test_data_dir_uses_xdg_data_home(self, xdg_home: Path) -> None:
        settings = Settings()
        assert settings.data_dir == xdg_home / "neuropose" / "jobs"

    def test_model_cache_dir_uses_xdg_data_home(self, xdg_home: Path) -> None:
        settings = Settings()
        assert settings.model_cache_dir == xdg_home / "neuropose" / "models"

    def test_derived_directories(self, xdg_home: Path) -> None:
        settings = Settings()
        assert settings.input_dir == settings.data_dir / "in"
        assert settings.output_dir == settings.data_dir / "out"
        assert settings.failed_dir == settings.data_dir / "failed"
        assert settings.status_file == settings.output_dir / "status.json"

    def test_fallback_when_xdg_unset(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        monkeypatch.delenv("XDG_DATA_HOME", raising=False)
        monkeypatch.setenv("HOME", str(tmp_path))
        settings = Settings()
        assert settings.data_dir == tmp_path / ".local" / "share" / "neuropose" / "jobs"

    def test_default_scalars(self, xdg_home: Path) -> None:
        settings = Settings()
        assert settings.poll_interval_seconds == 10
        assert settings.device == "/CPU:0"
        assert settings.default_fov_degrees == pytest.approx(55.0)


class TestValidation:
    """Field validators reject malformed input."""

    @pytest.mark.parametrize("device", ["/CPU:0", "/GPU:0", "/CPU:1", "/GPU:3"])
    def test_device_accepts_valid(self, device: str, xdg_home: Path) -> None:
        settings = Settings(device=device)
        assert settings.device == device

    @pytest.mark.parametrize("device", ["cpu", "/cpu:0", "GPU:0", "/TPU:0", "", "/GPU"])
    def test_device_rejects_invalid(self, device: str, xdg_home: Path) -> None:
        with pytest.raises(ValidationError):
            Settings(device=device)

    def test_poll_interval_rejects_zero(self, xdg_home: Path) -> None:
        with pytest.raises(ValidationError):
            Settings(poll_interval_seconds=0)

    def test_poll_interval_rejects_negative(self, xdg_home: Path) -> None:
        with pytest.raises(ValidationError):
            Settings(poll_interval_seconds=-5)

    def test_fov_rejects_zero(self, xdg_home: Path) -> None:
        with pytest.raises(ValidationError):
            Settings(default_fov_degrees=0)

    def test_fov_rejects_at_limit(self, xdg_home: Path) -> None:
        with pytest.raises(ValidationError):
            Settings(default_fov_degrees=180)

    def test_extra_fields_rejected(self, xdg_home: Path) -> None:
        with pytest.raises(ValidationError):
            Settings(nonexistent_field=True)


class TestYamlLoad:
    """``Settings.from_yaml`` behaves correctly for valid and malformed files."""

    def test_valid(self, tmp_path: Path, xdg_home: Path) -> None:
        config_path = tmp_path / "config.yaml"
        config_path.write_text(
            yaml.safe_dump({"device": "/GPU:0", "poll_interval_seconds": 30})
        )
        settings = Settings.from_yaml(config_path)
        assert settings.device == "/GPU:0"
        assert settings.poll_interval_seconds == 30

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            Settings.from_yaml(tmp_path / "nope.yaml")

    def test_empty_file_uses_defaults(self, tmp_path: Path, xdg_home: Path) -> None:
        config_path = tmp_path / "config.yaml"
        config_path.write_text("")
        settings = Settings.from_yaml(config_path)
        assert settings.poll_interval_seconds == 10

    def test_non_mapping_rejected(self, tmp_path: Path) -> None:
        config_path = tmp_path / "config.yaml"
        config_path.write_text("- item1\n- item2\n")
        with pytest.raises(ValueError, match="YAML mapping"):
            Settings.from_yaml(config_path)

    def test_invalid_field_rejected(self, tmp_path: Path, xdg_home: Path) -> None:
        config_path = tmp_path / "config.yaml"
        config_path.write_text(yaml.safe_dump({"device": "cpu"}))
        with pytest.raises(ValidationError):
            Settings.from_yaml(config_path)


class TestEnvironmentOverrides:
    """``NEUROPOSE_*`` env vars override field defaults."""

    def test_device_override(
        self,
        xdg_home: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("NEUROPOSE_DEVICE", "/GPU:0")
        settings = Settings()
        assert settings.device == "/GPU:0"

    def test_poll_interval_override(
        self,
        xdg_home: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("NEUROPOSE_POLL_INTERVAL_SECONDS", "60")
        settings = Settings()
        assert settings.poll_interval_seconds == 60

    def test_kwargs_beat_env_vars(
        self,
        xdg_home: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("NEUROPOSE_DEVICE", "/GPU:0")
        settings = Settings(device="/CPU:0")
        assert settings.device == "/CPU:0"


class TestEnsureDirs:
    """``ensure_dirs`` creates all directories idempotently."""

    def test_creates_all_directories(self, tmp_path: Path) -> None:
        settings = Settings(
            data_dir=tmp_path / "jobs",
            model_cache_dir=tmp_path / "models",
        )
        settings.ensure_dirs()
        assert settings.data_dir.is_dir()
        assert settings.input_dir.is_dir()
        assert settings.output_dir.is_dir()
        assert settings.failed_dir.is_dir()
        assert settings.model_cache_dir.is_dir()

    def test_idempotent(self, tmp_path: Path) -> None:
        settings = Settings(
            data_dir=tmp_path / "jobs",
            model_cache_dir=tmp_path / "models",
        )
        settings.ensure_dirs()
        settings.ensure_dirs()
        assert settings.data_dir.is_dir()

    def test_construction_has_no_filesystem_side_effects(self, tmp_path: Path) -> None:
        # Creating Settings() must NOT touch the filesystem.
        target = tmp_path / "jobs"
        assert not target.exists()
        _ = Settings(data_dir=target, model_cache_dir=tmp_path / "models")
        assert not target.exists()
