"""Runtime configuration for NeuroPose.

Central settings model built on pydantic-settings. Configuration can be
supplied by, in order of decreasing precedence:

1. Keyword arguments passed directly to ``Settings(...)``.
2. Environment variables prefixed with ``NEUROPOSE_`` (e.g.
   ``NEUROPOSE_DEVICE="/GPU:0"``).
3. A YAML file loaded explicitly via :meth:`Settings.from_yaml`.
4. Field defaults.

There is intentionally no implicit config-file discovery. The daemon must be
pointed at a config file explicitly via the CLI ``--config`` flag. This
avoids the relative-path footgun from the previous prototype, where the
daemon only worked when launched from a specific working directory.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import yaml
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

_DEVICE_PATTERN = re.compile(r"^/(CPU|GPU):\d+$")


def _xdg_data_home() -> Path:
    """Return ``$XDG_DATA_HOME``, falling back to ``~/.local/share``."""
    env = os.environ.get("XDG_DATA_HOME")
    if env:
        return Path(env)
    return Path.home() / ".local" / "share"


def _default_data_dir() -> Path:
    """Return the default runtime data directory (under XDG)."""
    return _xdg_data_home() / "neuropose" / "jobs"


def _default_model_cache_dir() -> Path:
    """Return the default MeTRAbs model cache directory (under XDG)."""
    return _xdg_data_home() / "neuropose" / "models"


class Settings(BaseSettings):
    """NeuroPose runtime configuration.

    Parameters
    ----------
    data_dir
        Base directory that holds ``in/``, ``out/``, and ``failed/``
        subdirectories for job processing. Defaults to a location under
        ``$XDG_DATA_HOME`` so runtime data never lives inside the repository.
    model_cache_dir
        Directory where the MeTRAbs model is downloaded and cached.
    poll_interval_seconds
        Interval between filesystem scans performed by the interfacer daemon.
    device
        TensorFlow device string, e.g. ``"/CPU:0"`` or ``"/GPU:0"``.
    default_fov_degrees
        Default horizontal field of view passed to MeTRAbs when a video does
        not supply camera intrinsics. The MeTRAbs upstream default is 55°.
    """

    model_config = SettingsConfigDict(
        env_prefix="NEUROPOSE_",
        env_nested_delimiter="__",
        extra="forbid",
    )

    data_dir: Path = Field(default_factory=_default_data_dir)
    model_cache_dir: Path = Field(default_factory=_default_model_cache_dir)
    poll_interval_seconds: int = Field(default=10, ge=1)
    device: str = Field(default="/CPU:0")
    default_fov_degrees: float = Field(default=55.0, gt=0.0, lt=180.0)

    @field_validator("device")
    @classmethod
    def _validate_device(cls, value: str) -> str:
        if not _DEVICE_PATTERN.match(value):
            raise ValueError(
                f"device must match '/(CPU|GPU):<index>' (e.g. '/CPU:0', '/GPU:0'); got {value!r}"
            )
        return value

    @property
    def input_dir(self) -> Path:
        """Return the directory containing job subdirectories to be processed."""
        return self.data_dir / "in"

    @property
    def output_dir(self) -> Path:
        """Return the directory where completed job results are written."""
        return self.data_dir / "out"

    @property
    def failed_dir(self) -> Path:
        """Return the directory where inputs are quarantined after catastrophic failure."""
        return self.data_dir / "failed"

    @property
    def status_file(self) -> Path:
        """Return the path of the persistent job-status JSON file."""
        return self.output_dir / "status.json"

    @classmethod
    def from_yaml(cls, path: Path) -> Settings:
        """Load settings from a YAML file.

        Parameters
        ----------
        path
            Path to a YAML configuration file. The file must be a mapping of
            field names to values; unknown fields are rejected.

        Returns
        -------
        Settings
            A validated settings instance.

        Raises
        ------
        FileNotFoundError
            If ``path`` does not exist.
        ValueError
            If the file is not a YAML mapping.
        pydantic.ValidationError
            If field validation fails.
        """
        if not path.exists():
            raise FileNotFoundError(f"config file not found: {path}")
        with path.open("r", encoding="utf-8") as f:
            data: Any = yaml.safe_load(f)
        if data is None:
            data = {}
        if not isinstance(data, dict):
            raise ValueError(f"config file must contain a YAML mapping; got {type(data).__name__}")
        return cls(**data)

    def ensure_dirs(self) -> None:
        """Create all runtime directories if they do not already exist.

        Called by the interfacer daemon on startup. Kept as an explicit method
        rather than a side effect of construction so that ``Settings()`` is
        safe to call in tests without touching the filesystem.
        """
        for path in (
            self.data_dir,
            self.input_dir,
            self.output_dir,
            self.failed_dir,
            self.model_cache_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)
