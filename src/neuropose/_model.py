"""MeTRAbs model loading — stub pending commit 11.

This module exists so that :mod:`neuropose.estimator` can import a single,
well-typed loader function even before the upstream MeTRAbs URL is pinned
and the TensorFlow version is settled.

Commit 11 will replace :func:`load_metrabs_model` with an implementation
that:

1. Pins the canonical MeTRAbs tfhub / Kaggle Models handle (replacing the
   ``bit.ly/metrabs_1`` shortener from the previous prototype).
2. Caches the downloaded model under ``Settings.model_cache_dir`` so the
   first run downloads it and subsequent runs are offline.
3. Returns a typed handle that the estimator can invoke without hitting the
   network on each instantiation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def load_metrabs_model(cache_dir: Path | None = None) -> Any:  # noqa: ARG001
    """Load the MeTRAbs model, downloading and caching on first use.

    Parameters
    ----------
    cache_dir
        Directory where the model should be cached. Typically
        ``Settings.model_cache_dir``. If ``None``, the implementation picks
        a default location.

    Returns
    -------
    object
        An opaque model handle that exposes ``detect_poses`` and the
        per-skeleton joint metadata attributes used by
        :class:`neuropose.estimator.Estimator`.

    Raises
    ------
    NotImplementedError
        Always, at this commit. Commit 11 provides the real implementation
        once the upstream MeTRAbs URL is pinned.
    """
    raise NotImplementedError(
        "load_metrabs_model is stubbed pending commit 11. "
        "Inject a model directly via Estimator(model=...) for now, "
        "or wait for the upstream MeTRAbs URL and TensorFlow version pin."
    )
