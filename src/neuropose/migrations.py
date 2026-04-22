"""Schema migration infrastructure for serialised NeuroPose payloads.

Every top-level JSON schema that NeuroPose persists to disk
(:class:`~neuropose.io.VideoPredictions`,
:class:`~neuropose.io.JobResults`, and
:class:`~neuropose.io.BenchmarkResult`) carries a ``schema_version``
integer. When those files are read back, the raw dict is passed
through :func:`migrate_video_predictions` /
:func:`migrate_job_results` / :func:`migrate_benchmark_result`
*before* pydantic validation runs, so each schema version can be
brought up to the current one transparently.

The pattern is deliberately small: one integer version counter shared
across all top-level schemas, plus a per-schema registry of
``{from_version: migration_fn}``. Each migration is a pure function
``dict -> dict`` responsible for stamping the new ``schema_version``
on its output. The framework chains them.

This module is intentionally separate from :mod:`neuropose.io` so
that migration registrations cannot accidentally import the pydantic
models they migrate — migrations must operate on raw dicts to be
robust to schema drift (a field a migration references may not exist
on the pydantic model by the time CURRENT_VERSION has moved past
it).

Adding a new migration
----------------------
When a schema change lands:

1. Bump :data:`CURRENT_VERSION`.
2. Register a migration from the *previous* version to the new one
   via :func:`register_video_predictions_migration` (or the sibling
   for benchmark results). The function receives the raw dict at the
   old version and must return a dict at the new version *including*
   the updated ``schema_version`` stamp.
3. Update the pydantic model in :mod:`neuropose.io` to reflect the
   new field set.
4. Add a unit test verifying that a fixture at the old version
   round-trips through ``load_*`` to the expected new-version shape.
"""

from __future__ import annotations

import logging
from collections.abc import Callable

logger = logging.getLogger(__name__)


CURRENT_VERSION = 2
"""The current schema version for all NeuroPose-persisted JSON payloads.

Shared across :class:`~neuropose.io.VideoPredictions`,
:class:`~neuropose.io.JobResults`, and
:class:`~neuropose.io.BenchmarkResult` so that coordinated schema
changes (for example, adding a ``provenance`` field to all three at
once) bump a single counter rather than three parallel ones.

Version history
---------------
- **v1:** initial schema, pre-Phase-0.
- **v2:** added optional ``provenance`` field to :class:`~neuropose.io.VideoPredictions`
  and :class:`~neuropose.io.BenchmarkResult` (Phase 0, Paper C reproducibility envelope).
  :class:`~neuropose.analyzer.pipeline.AnalysisReport` also enters the registry at v2
  (no legacy v1 payloads ever existed for it, so no migration is registered)."""


class MigrationError(Exception):
    """Base class for schema-migration failures."""


class FutureSchemaError(MigrationError):
    """Raised when a payload's ``schema_version`` exceeds :data:`CURRENT_VERSION`.

    Produced when a newer NeuroPose version writes a file and an older
    version tries to read it. The fix is upgrading NeuroPose; silently
    stripping fields would corrupt the payload.
    """


class MigrationNotFoundError(MigrationError):
    """Raised when no migration is registered for an intermediate version.

    Indicates a bug — :data:`CURRENT_VERSION` was bumped past a version
    for which no migration function was registered. Should only surface
    in tests or on a corrupted install.
    """


# Per-schema registries. Keys are the *source* version of the migration;
# the value is a callable that takes a dict at that version and returns a
# dict at ``source + 1``.
_VIDEO_PREDICTIONS_MIGRATIONS: dict[int, Callable[[dict], dict]] = {}
_BENCHMARK_RESULT_MIGRATIONS: dict[int, Callable[[dict], dict]] = {}
_ANALYSIS_REPORT_MIGRATIONS: dict[int, Callable[[dict], dict]] = {}


def register_video_predictions_migration(
    from_version: int,
) -> Callable[[Callable[[dict], dict]], Callable[[dict], dict]]:
    """Register a :class:`~neuropose.io.VideoPredictions` migration.

    Usage::

        @register_video_predictions_migration(from_version=1)
        def _v1_to_v2(payload: dict) -> dict:
            payload = dict(payload)
            payload["provenance"] = None
            payload["schema_version"] = 2
            return payload

    The decorator registers the function into the per-schema migration
    registry and returns it unchanged, so it can still be called
    directly from tests.
    """

    def wrap(fn: Callable[[dict], dict]) -> Callable[[dict], dict]:
        if from_version in _VIDEO_PREDICTIONS_MIGRATIONS:
            raise RuntimeError(
                f"video-predictions migration already registered from version {from_version}"
            )
        _VIDEO_PREDICTIONS_MIGRATIONS[from_version] = fn
        return fn

    return wrap


def register_benchmark_result_migration(
    from_version: int,
) -> Callable[[Callable[[dict], dict]], Callable[[dict], dict]]:
    """Register a :class:`~neuropose.io.BenchmarkResult` migration.

    See :func:`register_video_predictions_migration` for usage — this
    is the same pattern for the benchmark-result registry.
    """

    def wrap(fn: Callable[[dict], dict]) -> Callable[[dict], dict]:
        if from_version in _BENCHMARK_RESULT_MIGRATIONS:
            raise RuntimeError(
                f"benchmark-result migration already registered from version {from_version}"
            )
        _BENCHMARK_RESULT_MIGRATIONS[from_version] = fn
        return fn

    return wrap


def register_analysis_report_migration(
    from_version: int,
) -> Callable[[Callable[[dict], dict]], Callable[[dict], dict]]:
    """Register a :class:`~neuropose.analyzer.pipeline.AnalysisReport` migration.

    See :func:`register_video_predictions_migration` for usage — this
    is the same pattern for the analysis-report registry. Unlike the
    other two schemas, :class:`AnalysisReport` first appeared at
    :data:`CURRENT_VERSION = 2`, so no ``from_version=1`` migration
    exists (and none is expected).
    """

    def wrap(fn: Callable[[dict], dict]) -> Callable[[dict], dict]:
        if from_version in _ANALYSIS_REPORT_MIGRATIONS:
            raise RuntimeError(
                f"analysis-report migration already registered from version {from_version}"
            )
        _ANALYSIS_REPORT_MIGRATIONS[from_version] = fn
        return fn

    return wrap


def migrate_video_predictions(payload: dict) -> dict:
    """Migrate a raw :class:`~neuropose.io.VideoPredictions` dict to current.

    Parameters
    ----------
    payload
        Raw JSON-loaded dict. Must not yet have been through pydantic
        validation. A missing ``schema_version`` key is interpreted as
        version ``1`` (the earliest tracked version, shipped before
        the migration infrastructure existed).

    Returns
    -------
    dict
        The payload at :data:`CURRENT_VERSION`. Ready to be passed to
        ``VideoPredictions.model_validate``.

    Raises
    ------
    FutureSchemaError
        If the payload declares a ``schema_version`` higher than
        :data:`CURRENT_VERSION`.
    MigrationNotFoundError
        If an intermediate migration is missing from the registry.
    """
    return _migrate(payload, _VIDEO_PREDICTIONS_MIGRATIONS, schema_name="VideoPredictions")


def migrate_benchmark_result(payload: dict) -> dict:
    """Migrate a raw :class:`~neuropose.io.BenchmarkResult` dict to current.

    See :func:`migrate_video_predictions` for semantics. This is the
    sibling function for benchmark-result payloads.
    """
    return _migrate(payload, _BENCHMARK_RESULT_MIGRATIONS, schema_name="BenchmarkResult")


def migrate_analysis_report(payload: dict) -> dict:
    """Migrate a raw :class:`~neuropose.analyzer.pipeline.AnalysisReport` dict.

    See :func:`migrate_video_predictions` for semantics. Because
    :class:`AnalysisReport` first shipped at schema_version 2, a
    payload missing the key still defaults to 1 (and would require a
    not-yet-registered v1 → v2 migration); this is only reachable for
    deliberately malformed inputs.
    """
    return _migrate(payload, _ANALYSIS_REPORT_MIGRATIONS, schema_name="AnalysisReport")


def migrate_job_results(payload: dict) -> dict:
    """Migrate a :class:`~neuropose.io.JobResults` root dict to current.

    ``JobResults`` is a ``RootModel`` whose root is a mapping of video
    name to :class:`~neuropose.io.VideoPredictions` payload. It has no
    envelope of its own, so the migration is "run
    :func:`migrate_video_predictions` on every value in the mapping."

    Parameters
    ----------
    payload
        Raw JSON-loaded dict of ``{video_name: VideoPredictions-shaped dict}``.

    Returns
    -------
    dict
        The same mapping with each video payload migrated to the
        current schema version.
    """
    return {name: migrate_video_predictions(video) for name, video in payload.items()}


def _migrate(
    payload: dict,
    migrations: dict[int, Callable[[dict], dict]],
    *,
    schema_name: str,
) -> dict:
    """Walk the migration chain to :data:`CURRENT_VERSION` and return the migrated payload.

    Shared driver for :func:`migrate_video_predictions` and
    :func:`migrate_benchmark_result`. Looks up the incoming
    ``schema_version`` (defaulting to 1 when absent), walks the migration
    chain until reaching :data:`CURRENT_VERSION`, and returns the
    migrated payload. Logs at INFO each time it actually advances a
    version so operators see the upgrade happen.
    """
    version = payload.get("schema_version", 1)
    if not isinstance(version, int) or version < 1:
        raise MigrationError(
            f"{schema_name} payload has invalid schema_version {version!r}; must be an integer >= 1"
        )
    if version > CURRENT_VERSION:
        raise FutureSchemaError(
            f"{schema_name} payload declares schema_version {version}, which is newer "
            f"than this build's CURRENT_VERSION ({CURRENT_VERSION}). Upgrade NeuroPose."
        )
    while version < CURRENT_VERSION:
        if version not in migrations:
            raise MigrationNotFoundError(
                f"no {schema_name} migration registered from schema_version {version}"
            )
        logger.info(
            "Migrating %s payload from schema_version %d to %d",
            schema_name,
            version,
            version + 1,
        )
        payload = migrations[version](payload)
        version += 1
    return payload


# ---------------------------------------------------------------------------
# Registered migrations
# ---------------------------------------------------------------------------
#
# Keep registrations *below* the driver so the module's public API surfaces
# at the top and the version-specific diffs live together at the bottom where
# they are easiest to audit chronologically.


@register_video_predictions_migration(from_version=1)
def _video_predictions_v1_to_v2(payload: dict) -> dict:
    """v1 → v2: add the optional ``provenance`` field (Phase 0).

    Phase 0 introduces the :class:`~neuropose.io.Provenance` envelope
    for Paper C reproducibility. v1 files predate it, so we stamp
    ``provenance = None`` on load — the field is optional on the
    pydantic model and ``None`` correctly indicates "we don't have
    provenance metadata for this payload."
    """
    payload = dict(payload)
    payload.setdefault("provenance", None)
    payload["schema_version"] = 2
    return payload


@register_benchmark_result_migration(from_version=1)
def _benchmark_result_v1_to_v2(payload: dict) -> dict:
    """v1 → v2: add the optional ``provenance`` field (Phase 0).

    Sibling of :func:`_video_predictions_v1_to_v2` for benchmark
    payloads; same rationale.
    """
    payload = dict(payload)
    payload.setdefault("provenance", None)
    payload["schema_version"] = 2
    return payload
