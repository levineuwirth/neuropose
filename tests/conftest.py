"""Shared pytest configuration and fixtures for the NeuroPose test suite."""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def _isolate_environment(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path_factory: pytest.TempPathFactory,
) -> Iterator[None]:
    """Isolate every test from the developer's real home directory.

    Points ``$HOME`` and ``$XDG_DATA_HOME`` at per-test temp directories so
    that any code path that uses the default ``Settings()`` (which reaches
    into ``~/.local/share/neuropose``) cannot accidentally write to the real
    machine. Also clears any ``NEUROPOSE_*`` environment variables that may
    be set in the developer's shell, so test behaviour does not depend on
    who is running the test suite.
    """
    isolated = tmp_path_factory.mktemp("neuropose_env_isolation")
    monkeypatch.setenv("HOME", str(isolated))
    monkeypatch.setenv("XDG_DATA_HOME", str(isolated / "xdg"))
    for key in list(os.environ):
        if key.startswith("NEUROPOSE_"):
            monkeypatch.delenv(key, raising=False)
    yield


@pytest.fixture
def xdg_home() -> Path:
    """Return the isolated ``$XDG_DATA_HOME`` set up by ``_isolate_environment``."""
    return Path(os.environ["XDG_DATA_HOME"])
