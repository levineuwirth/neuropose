"""Drift test for the hardcoded berkeley_mhad_43 joint-name constant.

:mod:`neuropose.analyzer.segment` ships ``JOINT_NAMES`` as a frozen
tuple of 43 strings so that post-processing callers can resolve
``"rwri"`` → index without loading a multi-gigabyte TensorFlow model.
That is only safe while the hardcoded tuple actually matches what the
pinned MeTRAbs SavedModel reports.

This test loads the real model via :func:`neuropose._model.load_metrabs_model`
and asserts that ``JOINT_NAMES`` is byte-identical to
``model.per_skeleton_joint_names["berkeley_mhad_43"].numpy().astype(str)``.
If MeTRAbs ever ships a new skeleton under the same name — or if we bump
the model pin to one whose ``berkeley_mhad_43`` skeleton is spelled
differently — this test is the drift detector.

Like every test under ``tests/integration/`` the file is marked
``@pytest.mark.slow`` and only runs under ``pytest --runslow``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from neuropose._model import load_metrabs_model
from neuropose.analyzer.segment import JOINT_NAMES

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def metrabs_model_cache_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Module-scoped cache dir so the model downloads at most once per run.

    Function scope would re-download on every test; session scope would
    collide with the estimator smoke-test cache. Module scope is the
    right middle ground for a file that only needs the model loaded
    once.
    """
    return tmp_path_factory.mktemp("neuropose_joint_names_model_cache")


def test_joint_names_match_pinned_model(metrabs_model_cache_dir: Path) -> None:
    """Hardcoded ``JOINT_NAMES`` must match the loaded MeTRAbs skeleton.

    If this fails, the expected fix is:

    1. Update :data:`neuropose.analyzer.segment.JOINT_NAMES` in the same
       commit that bumps the model pin in :mod:`neuropose._model`.
    2. Cross-check any CLI or docs that embed hardcoded joint names.
    """
    loaded = load_metrabs_model(cache_dir=metrabs_model_cache_dir)
    tensor = loaded.model.per_skeleton_joint_names["berkeley_mhad_43"]
    model_names = tuple(tensor.numpy().astype(str).tolist())
    assert model_names == JOINT_NAMES, (
        "JOINT_NAMES drift detected — the hardcoded tuple in "
        "neuropose.analyzer.segment no longer matches the MeTRAbs model. "
        f"Model reports: {model_names}"
    )
