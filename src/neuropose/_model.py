"""MeTRAbs model loading with download, checksum verification, and caching.

The :func:`load_metrabs_model` function is the single entry point through
which :class:`neuropose.estimator.Estimator` acquires its TensorFlow
model. It handles:

1. First-call download from the pinned upstream URL.
2. SHA-256 verification of the downloaded tarball against a known-good
   checksum. A mismatch triggers exactly one automatic retry (in case
   the download was truncated), after which the error surfaces.
3. Atomic extraction to a staging directory and a single rename into
   the final cache location, so a crash mid-extraction cannot leave
   the cache in a half-extracted state.
4. SavedModel load via ``tf.saved_model.load``.
5. A post-load interface sanity check that verifies the loaded model
   exposes the ``detect_poses``, ``per_skeleton_joint_names``, and
   ``per_skeleton_joint_edges`` attributes the estimator needs.

Model artifact
--------------
The pinned model is MeTRAbs's EfficientNetV2-L variant
(``metrabs_eff2l_y4_384px_800k_28ds``):

- **URL**: hosted on the RWTH Aachen "omnomnom" server, which is the
  canonical distribution point for the MeTRAbs authors' lab.
- **SHA-256**: pinned below. Any change to the URL or the upstream
  tarball will surface as a verification failure, forcing a human
  review of the new checksum before downstream code trusts the new
  artifact.

See ``RESEARCH.md`` at the repo root for the ongoing discussion of
self-hosting the model on our own infrastructure instead of relying on
a single third-party URL.
"""

from __future__ import annotations

import hashlib
import logging
import os
import shutil
import tarfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LoadedModel:
    """Result of :func:`load_metrabs_model`.

    Bundles the loaded TensorFlow model with the provenance metadata
    that identifies which artifact it came from. Callers that only want
    the model reach for :attr:`model`; callers that build a
    :class:`~neuropose.io.Provenance` (primarily
    :class:`~neuropose.estimator.Estimator`) pull :attr:`sha256` and
    :attr:`filename` too.

    Frozen — once :func:`load_metrabs_model` has produced a
    ``LoadedModel``, nothing downstream should edit the identity of
    the artifact it describes.
    """

    model: Any
    sha256: str
    filename: str


# ---------------------------------------------------------------------------
# Model artifact: pinned URL and checksum.
# ---------------------------------------------------------------------------
#
# If the URL or checksum below changes, the diff should be reviewed by a
# human. These are supply-chain constants.

_MODEL_URL = (
    "https://omnomnom.vision.rwth-aachen.de/data/metrabs/metrabs_eff2l_y4_384px_800k_28ds.tar.gz"
)
_MODEL_SHA256 = "fa31b5b043f227588c3d224e56db89307d021bfbbb52e36028919f90e1f96c89"
_MODEL_ARCHIVE_NAME = "metrabs_eff2l_y4_384px_800k_28ds.tar.gz"
_MODEL_DIR_NAME = "metrabs_eff2l_y4_384px_800k_28ds"

_DOWNLOAD_CHUNK_BYTES = 1024 * 1024  # 1 MB
_DOWNLOAD_SOCKET_TIMEOUT = 120.0  # seconds between bytes, not total
_REQUIRED_MODEL_ATTRS = (
    "detect_poses",
    "per_skeleton_joint_names",
    "per_skeleton_joint_edges",
)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_metrabs_model(cache_dir: Path | None = None) -> LoadedModel:
    """Load the MeTRAbs model, downloading and caching on first use.

    Parameters
    ----------
    cache_dir
        Directory where the model tarball and extracted SavedModel are
        cached. If ``None``, defaults to
        ``$XDG_DATA_HOME/neuropose/models`` (matching
        :attr:`neuropose.config.Settings.model_cache_dir`).

    Returns
    -------
    LoadedModel
        Bundle containing the TensorFlow SavedModel handle alongside
        the pinned artifact SHA-256 and filename that identify which
        model the handle came from. The handle exposes ``detect_poses``
        and the ``per_skeleton_joint_names`` / ``per_skeleton_joint_edges``
        attributes used by :class:`neuropose.estimator.Estimator`.

    Raises
    ------
    RuntimeError
        If the download fails, the SHA-256 does not match (after one
        automatic retry), extraction fails, TensorFlow is not
        installed, or the loaded model does not expose the expected
        interface.

    Notes
    -----
    The returned ``sha256`` is the module-pinned :data:`_MODEL_SHA256`,
    not a re-hash of the on-disk tarball. On the cold-cache path this
    is exactly the hash we verified against before loading. On the
    warm-cache path the tarball is not re-verified (that would cost a
    2 GB I/O pass on every daemon startup), so the reported SHA is an
    attestation of "this is the pinned artifact NeuroPose loads" rather
    than a direct fingerprint of the on-disk bytes. For the threat
    model this supports — reproducibility, not tamper-evidence — that
    is the correct semantics.
    """
    resolved_cache = Path(cache_dir) if cache_dir is not None else _default_cache_dir()
    resolved_cache.mkdir(parents=True, exist_ok=True)

    model_dir = resolved_cache / _MODEL_DIR_NAME

    if model_dir.exists():
        try:
            saved_model_dir = _find_saved_model(model_dir)
        except RuntimeError:
            logger.warning(
                "Cached model at %s appears incomplete; removing and re-downloading.",
                model_dir,
            )
            shutil.rmtree(model_dir, ignore_errors=True)
        else:
            return LoadedModel(
                model=_tf_load(saved_model_dir),
                sha256=_MODEL_SHA256,
                filename=_MODEL_ARCHIVE_NAME,
            )

    tarball = resolved_cache / _MODEL_ARCHIVE_NAME

    if not tarball.exists():
        _download_with_progress(_MODEL_URL, tarball)

    try:
        _verify_sha256(tarball, _MODEL_SHA256)
    except RuntimeError as first_exc:
        logger.warning(
            "SHA-256 mismatch on cached tarball; re-downloading once: %s",
            first_exc,
        )
        tarball.unlink(missing_ok=True)
        _download_with_progress(_MODEL_URL, tarball)
        _verify_sha256(tarball, _MODEL_SHA256)

    _extract_tarball(tarball, model_dir)
    saved_model_dir = _find_saved_model(model_dir)
    return LoadedModel(
        model=_tf_load(saved_model_dir),
        sha256=_MODEL_SHA256,
        filename=_MODEL_ARCHIVE_NAME,
    )


# ---------------------------------------------------------------------------
# Cache directory resolution
# ---------------------------------------------------------------------------


def _default_cache_dir() -> Path:
    """Return the default model cache directory under ``$XDG_DATA_HOME``.

    Duplicates :func:`neuropose.config._default_model_cache_dir` rather
    than importing it, to keep this module free of a dependency on the
    config layer. The two must agree; a regression test in
    :mod:`tests.unit.test_config` verifies the Settings default and any
    future change here should be cross-checked there.
    """
    xdg = os.environ.get("XDG_DATA_HOME")
    base = Path(xdg) if xdg else Path.home() / ".local" / "share"
    return base / "neuropose" / "models"


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------


def _download_with_progress(url: str, dest: Path) -> None:
    """Download ``url`` to ``dest`` with progress reporting via the logger.

    Streams the response in 1 MB chunks so memory usage stays flat
    regardless of the file size. Progress is logged at 10 % increments.
    On any exception, the partial file at ``dest`` is removed so the
    caller does not see a truncated file.
    """
    logger.info("Downloading %s → %s", url, dest)
    dest.parent.mkdir(parents=True, exist_ok=True)

    request = urllib.request.Request(
        url,
        headers={"User-Agent": "neuropose/0.1"},
    )

    try:
        with urllib.request.urlopen(
            request,
            timeout=_DOWNLOAD_SOCKET_TIMEOUT,
        ) as response:
            total_bytes_header = response.headers.get("Content-Length")
            total_bytes = int(total_bytes_header) if total_bytes_header else 0

            downloaded = 0
            next_progress_log = 0.10  # log at 10 %, 20 %, ...

            with dest.open("wb") as out_file:
                while True:
                    chunk = response.read(_DOWNLOAD_CHUNK_BYTES)
                    if not chunk:
                        break
                    out_file.write(chunk)
                    downloaded += len(chunk)
                    if total_bytes > 0:
                        fraction = downloaded / total_bytes
                        if fraction >= next_progress_log:
                            logger.info(
                                "  %d / %d MB (%.0f%%)",
                                downloaded // (1024 * 1024),
                                total_bytes // (1024 * 1024),
                                fraction * 100,
                            )
                            next_progress_log += 0.10
    except Exception as exc:
        # Clean up partial file so the next call re-downloads cleanly.
        dest.unlink(missing_ok=True)
        raise RuntimeError(f"Failed to download MeTRAbs model from {url}: {exc}") from exc

    if total_bytes > 0 and downloaded != total_bytes:
        dest.unlink(missing_ok=True)
        raise RuntimeError(
            f"Download from {url} was truncated: got {downloaded} bytes, expected {total_bytes}."
        )
    logger.info("Download complete: %d bytes", downloaded)


# ---------------------------------------------------------------------------
# Checksum verification
# ---------------------------------------------------------------------------


def _verify_sha256(path: Path, expected_hex: str) -> None:
    """Verify that ``path``'s SHA-256 digest matches ``expected_hex``.

    Streams the file through ``hashlib.sha256`` in 1 MB chunks so we do
    not pay for loading a 2 GB tarball into memory.
    """
    logger.info("Verifying SHA-256 of %s", path)
    hasher = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(_DOWNLOAD_CHUNK_BYTES), b""):
            hasher.update(chunk)
    actual_hex = hasher.hexdigest()
    if actual_hex != expected_hex:
        raise RuntimeError(
            f"SHA-256 mismatch for {path}: "
            f"expected {expected_hex}, got {actual_hex}. "
            f"The downloaded file is corrupt, truncated, or has been tampered with."
        )
    logger.info("SHA-256 verified")


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------


def _extract_tarball(tarball: Path, dest_dir: Path) -> None:
    """Extract ``tarball`` atomically to ``dest_dir``.

    Extracts first to a sibling ``<dest_dir>.staging`` directory, then
    replaces ``dest_dir`` with a single ``rename`` once extraction
    completes. A crash mid-extraction therefore cannot leave behind a
    half-populated ``dest_dir``.

    Uses tarfile's ``data`` filter to block path traversal and other
    tar-bomb patterns.
    """
    logger.info("Extracting %s → %s", tarball, dest_dir)
    staging = dest_dir.parent / (dest_dir.name + ".staging")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True, exist_ok=True)

    try:
        with tarfile.open(tarball, "r:gz") as tf_archive:
            # ``filter="data"`` guards against path traversal and other
            # malicious tar contents. Available in Python 3.11.4+ and
            # required in 3.14+.
            tf_archive.extractall(staging, filter="data")
    except Exception as exc:
        shutil.rmtree(staging, ignore_errors=True)
        raise RuntimeError(f"Failed to extract {tarball}: {exc}") from exc

    if dest_dir.exists():
        shutil.rmtree(dest_dir)
    staging.rename(dest_dir)
    logger.info("Extracted to %s", dest_dir)


# ---------------------------------------------------------------------------
# SavedModel discovery and TF load
# ---------------------------------------------------------------------------


def _find_saved_model(root: Path) -> Path:
    """Return the directory containing ``saved_model.pb`` under ``root``.

    The MeTRAbs tarball extracts to a directory containing a SavedModel
    directory, which itself contains ``saved_model.pb``. The exact
    layout of intermediate directories is not contractually stable, so
    we search rather than hardcoding a path.

    Raises
    ------
    RuntimeError
        If no ``saved_model.pb`` is found, or if multiple candidates
        are found (which would make the choice ambiguous).
    """
    candidates = list(root.rglob("saved_model.pb"))
    if not candidates:
        raise RuntimeError(f"no saved_model.pb found under {root}; tarball layout unexpected")
    if len(candidates) > 1:
        raise RuntimeError(
            f"multiple saved_model.pb files found under {root}: "
            f"{[str(p) for p in candidates]}. "
            f"Cannot determine which one to load."
        )
    return candidates[0].parent


def _tf_load(saved_model_dir: Path) -> Any:
    """Load a SavedModel from ``saved_model_dir`` and sanity-check it.

    TensorFlow is imported lazily here so that importing
    :mod:`neuropose._model` does not require TF for test or docs
    code paths that never reach the loader.
    """
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise RuntimeError(
            "TensorFlow is required to load the MeTRAbs model but is not installed. "
            "Install the NeuroPose runtime dependencies with: "
            "pip install neuropose (or uv sync from a dev checkout)."
        ) from exc

    logger.info("Loading SavedModel from %s", saved_model_dir)
    try:
        model = tf.saved_model.load(str(saved_model_dir))
    except Exception as exc:
        raise RuntimeError(f"Failed to load SavedModel from {saved_model_dir}: {exc}") from exc

    missing = [attr for attr in _REQUIRED_MODEL_ATTRS if not hasattr(model, attr)]
    if missing:
        raise RuntimeError(
            f"Loaded SavedModel at {saved_model_dir} is missing expected "
            f"attributes {missing}. The tarball may not be a MeTRAbs model."
        )

    logger.info("MeTRAbs model loaded successfully from %s", saved_model_dir)
    return model
