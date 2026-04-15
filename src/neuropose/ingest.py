"""Zip-archive ingestion for the NeuroPose job queue.

Operators frequently arrive with a zip archive of videos — pulled off a
camera SD card, exported from a study laptop, etc. — and want every
video inside processed as an independent job without manually
unpacking the archive, renaming collisions, and copying files into
``$data_dir/in/<job_name>/`` one at a time. This module automates
exactly that workflow.

Responsibilities
----------------
- **Validate the archive before touching disk.** Path-traversal members
  (absolute paths, ``..`` components) are refused; the total
  uncompressed size is capped; only files whose suffixes are in
  :data:`neuropose.interfacer.VIDEO_EXTENSIONS` are considered as
  candidates. Non-video members are logged and skipped, not treated
  as errors — zips commonly carry ``.DS_Store``, ``README.md``, etc.
- **Derive one job name per video.** The in-archive path is
  slash-replaced, extension-stripped, and sanitized so that nested
  structure like ``patient_001/trial_01.mp4`` becomes
  ``patient_001_trial_01`` and never collides with a sibling
  ``trial_01.mp4`` in another directory.
- **Atomic per-job placement.** Each video is extracted to a staging
  directory under ``$data_dir/.ingest_<uuid>/`` first. Only after the
  whole archive has been successfully extracted does the ingester
  rename each staged job directory into ``$data_dir/in/<job_name>/``
  with :func:`os.rename`, which is atomic on the same filesystem.
  This way the daemon never sees a half-extracted job directory and
  the "empty directory mid-copy" skip heuristic in the interfacer is
  never exercised during ingest.
- **Collision detection, up-front.** Zip-internal collisions (two
  videos that would flatten to the same job name) and external
  collisions (a job directory of the same name already exists under
  ``in/``) are detected *before* any disk write. The default is to
  refuse the operation and report every colliding name at once;
  ``force=True`` deletes the colliding ``in/<job_name>/`` directories
  and proceeds.

This module is pure library: the CLI wrapper lives in
:mod:`neuropose.cli`, and the function here does not touch the
logger's configuration. Errors surface as typed exceptions so the
CLI can translate them into stable exit codes.
"""

from __future__ import annotations

import logging
import re
import shutil
import uuid
import zipfile
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath

from neuropose.interfacer import VIDEO_EXTENSIONS

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tunables
# ---------------------------------------------------------------------------

MAX_UNCOMPRESSED_BYTES: int = 20 * 1024 * 1024 * 1024  # 20 GB
"""Maximum total uncompressed size of a single zip archive.

Not a hard clinical limit — just a zip-bomb guard. A real research
archive with a few dozen multi-GB recordings stays well below this
number; anything above it is almost certainly a mistake or an attack.
The cap is enforced by summing :attr:`zipfile.ZipInfo.file_size` over
every candidate member *before* any extraction starts, so a bomb that
expands to 100 GB is rejected without writing a single byte to disk.
"""

_JOB_NAME_SAFE_PATTERN = re.compile(r"[^A-Za-z0-9._-]+")
"""Characters allowed in a derived job name. Everything else becomes ``_``."""


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class IngestError(Exception):
    """Base class for errors raised by :func:`ingest_zip`."""


class ArchiveInvalidError(IngestError):
    """The zip archive is malformed, unreadable, or contains unsafe members."""


class ArchiveEmptyError(IngestError):
    """The archive contains no files with supported video extensions."""


class ArchiveTooLargeError(IngestError):
    """The archive's total uncompressed size exceeds :data:`MAX_UNCOMPRESSED_BYTES`."""


class JobCollisionError(IngestError):
    """One or more target job directories already exist, or two videos collapse to the same name.

    The ``collisions`` attribute lists the colliding job names so the
    caller can report them in one shot rather than piecemeal.
    """

    def __init__(self, message: str, collisions: list[str]) -> None:
        super().__init__(message)
        self.collisions = collisions


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class IngestedJob:
    """A single job directory created by :func:`ingest_zip`."""

    job_name: str
    job_dir: Path
    video_filename: str
    uncompressed_bytes: int


@dataclass(frozen=True)
class IngestResult:
    """Aggregate result of :func:`ingest_zip`."""

    archive: Path
    ingested: list[IngestedJob] = field(default_factory=list)
    skipped_non_videos: list[str] = field(default_factory=list)

    @property
    def job_count(self) -> int:
        """Return the number of job directories created."""
        return len(self.ingested)

    @property
    def total_uncompressed_bytes(self) -> int:
        """Return the total uncompressed size of all ingested videos in bytes."""
        return sum(job.uncompressed_bytes for job in self.ingested)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def ingest_zip(
    archive_path: Path,
    input_dir: Path,
    *,
    force: bool = False,
) -> IngestResult:
    """Extract videos from a zip archive into per-video job directories.

    Parameters
    ----------
    archive_path
        Path to the zip archive to ingest.
    input_dir
        The interfacer's input directory — typically
        ``Settings.input_dir`` (``$data_dir/in``). Each video in the
        archive produces a new subdirectory here, and the running
        daemon picks them up on its next poll.
    force
        If ``True``, overwrite any pre-existing ``input_dir/<job_name>/``
        directories that collide with the archive's contents by
        removing them first. If ``False`` (default), raise
        :class:`JobCollisionError` listing every colliding name and
        perform no disk writes.

    Returns
    -------
    IngestResult
        Record of which jobs were created and which archive members
        were skipped as non-video.

    Raises
    ------
    FileNotFoundError
        If ``archive_path`` does not exist.
    ArchiveInvalidError
        If the archive is not a valid zip, contains a member with an
        absolute path or traversal components, or cannot be read.
    ArchiveEmptyError
        If the archive contains no files with supported video
        extensions.
    ArchiveTooLargeError
        If the total uncompressed size of candidate video members
        exceeds :data:`MAX_UNCOMPRESSED_BYTES`.
    JobCollisionError
        If two members of the archive flatten to the same job name,
        or if any target ``input_dir/<job_name>/`` exists and
        ``force`` is ``False``.
    """
    if not archive_path.exists():
        raise FileNotFoundError(f"archive not found: {archive_path}")

    input_dir.mkdir(parents=True, exist_ok=True)

    # Phase 1: inspect the archive, decide every target job name, and
    # catch all collisions before touching disk.
    plan = _plan_ingest(archive_path, input_dir=input_dir, force=force)

    # Phase 2: extract every video into a staging directory under the
    # same filesystem as input_dir, so the final os.rename is atomic.
    # Using a uuid in the staging directory name avoids collisions
    # with concurrent ingests and with the daemon's lock file.
    staging_root = input_dir.parent / f".ingest_{uuid.uuid4().hex}"
    staging_root.mkdir(parents=True, exist_ok=False)

    ingested: list[IngestedJob] = []
    try:
        with zipfile.ZipFile(archive_path, "r") as archive:
            for entry in plan.entries:
                staged_job_dir = staging_root / entry.job_name
                staged_job_dir.mkdir(parents=True, exist_ok=False)
                staged_video = staged_job_dir / entry.output_filename
                with (
                    archive.open(entry.member) as src,
                    staged_video.open("wb") as dst,
                ):
                    shutil.copyfileobj(src, dst)

        # Phase 3: move the staged job directories into place. The
        # delete-then-rename dance under --force is only non-atomic
        # per-job — each individual job still flips from "does not
        # exist" to "fully populated" in one rename(2) call.
        for entry in plan.entries:
            staged_job_dir = staging_root / entry.job_name
            target_job_dir = input_dir / entry.job_name
            if force and target_job_dir.exists():
                shutil.rmtree(target_job_dir)
            staged_job_dir.rename(target_job_dir)
            ingested.append(
                IngestedJob(
                    job_name=entry.job_name,
                    job_dir=target_job_dir,
                    video_filename=entry.output_filename,
                    uncompressed_bytes=entry.member.file_size,
                )
            )
    except Exception:
        # Clean up whatever made it into input_dir so the caller is
        # not left with a half-ingested state on, e.g., a zip that
        # turns out to be truncated partway through extraction. We do
        # NOT unwind under success; this branch is the failure path.
        for job in ingested:
            shutil.rmtree(job.job_dir, ignore_errors=True)
        raise
    finally:
        shutil.rmtree(staging_root, ignore_errors=True)

    logger.info(
        "Ingested %d job(s) from %s (%d non-video members skipped)",
        len(ingested),
        archive_path,
        len(plan.skipped_non_videos),
    )
    return IngestResult(
        archive=archive_path,
        ingested=ingested,
        skipped_non_videos=plan.skipped_non_videos,
    )


# ---------------------------------------------------------------------------
# Private planning helpers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _PlannedEntry:
    """One zip member bound for a specific job directory."""

    member: zipfile.ZipInfo
    job_name: str
    output_filename: str


@dataclass(frozen=True)
class _IngestPlan:
    """Result of :func:`_plan_ingest` — a concrete set of extractions."""

    entries: list[_PlannedEntry]
    skipped_non_videos: list[str]


def _plan_ingest(
    archive_path: Path,
    *,
    input_dir: Path,
    force: bool,
) -> _IngestPlan:
    """Scan the archive and produce a validated extraction plan.

    No disk is touched except for opening the zip read-only. Every
    failure mode that we want to surface before extraction (bad archive,
    traversal, empty, too big, collision) is decided in this function
    so :func:`ingest_zip` is either a no-op or a complete success.
    """
    try:
        archive = zipfile.ZipFile(archive_path, "r")
    except zipfile.BadZipFile as exc:
        raise ArchiveInvalidError(f"not a valid zip archive: {archive_path}") from exc

    with archive:
        try:
            members = archive.infolist()
        except Exception as exc:
            raise ArchiveInvalidError(
                f"could not read archive member list: {archive_path}"
            ) from exc

        entries: list[_PlannedEntry] = []
        skipped: list[str] = []
        job_name_to_member: dict[str, str] = {}
        total_bytes = 0

        for member in members:
            if member.is_dir():
                continue

            member_name = member.filename
            _check_member_path_safe(member_name)

            # Filter by suffix. Non-video members are not errors — the
            # zip can carry README.md, .DS_Store, etc. alongside the
            # real payload.
            if PurePosixPath(member_name).suffix.lower() not in VIDEO_EXTENSIONS:
                skipped.append(member_name)
                continue

            total_bytes += member.file_size
            if total_bytes > MAX_UNCOMPRESSED_BYTES:
                raise ArchiveTooLargeError(
                    f"uncompressed size exceeds {MAX_UNCOMPRESSED_BYTES} bytes "
                    f"(cap) after including {member_name}; refusing to extract"
                )

            job_name = _derive_job_name(member_name)
            output_filename = PurePosixPath(member_name).name

            if job_name in job_name_to_member:
                raise JobCollisionError(
                    "zip contains two videos that flatten to the same job name "
                    f"{job_name!r}: {job_name_to_member[job_name]!r} and {member_name!r}",
                    collisions=[job_name],
                )
            job_name_to_member[job_name] = member_name

            entries.append(
                _PlannedEntry(
                    member=member,
                    job_name=job_name,
                    output_filename=output_filename,
                )
            )

    if not entries:
        raise ArchiveEmptyError(
            f"archive {archive_path} contains no video files with "
            f"supported extensions ({sorted(VIDEO_EXTENSIONS)})"
        )

    # External collisions: a pre-existing job directory with the same
    # name. Report every collision at once so the operator sees the
    # full picture in one error, not one at a time.
    if not force:
        existing = [entry.job_name for entry in entries if (input_dir / entry.job_name).exists()]
        if existing:
            raise JobCollisionError(
                f"{len(existing)} job director{'y' if len(existing) == 1 else 'ies'} "
                f"already exist in {input_dir}; pass --force to overwrite",
                collisions=existing,
            )

    return _IngestPlan(entries=entries, skipped_non_videos=skipped)


def _check_member_path_safe(name: str) -> None:
    """Refuse zip members with absolute paths or ``..`` traversal.

    Python's :mod:`zipfile` does not raise on these by default on
    3.11, so we validate each member name before it is extracted.
    """
    if not name:
        raise ArchiveInvalidError("zip archive contains an empty member name")
    if name.startswith("/") or (len(name) >= 2 and name[1] == ":"):
        raise ArchiveInvalidError(f"zip archive contains an absolute-path member: {name!r}")
    # PurePosixPath handles both forward and back slashes via the
    # component-level check on ``..`` below.
    parts = PurePosixPath(name).parts
    if any(part == ".." for part in parts):
        raise ArchiveInvalidError(
            f"zip archive contains a traversal member ({name!r}); refusing to extract"
        )


def _derive_job_name(member_name: str) -> str:
    """Convert an in-archive path into a safe job-directory name.

    The full in-archive path (minus the final extension) is joined
    with underscores — so ``patient_001/trial_01.mp4`` becomes
    ``patient_001_trial_01`` and stays unambiguous even if a sibling
    directory carries a file named ``trial_01.mp4``. Any character
    outside ``[A-Za-z0-9._-]`` is replaced with an underscore, then
    runs of underscores are collapsed, leading/trailing underscores
    are stripped, and the result is non-empty by construction (a
    pathological all-symbol name falls back to ``video``).
    """
    path = PurePosixPath(member_name)
    stem_parts = [*path.parts[:-1], path.stem]
    raw = "_".join(stem_parts)
    sanitised = _JOB_NAME_SAFE_PATTERN.sub("_", raw)
    # Collapse runs of underscores and strip leading/trailing ones so
    # the result doesn't look like `_foo__bar_`.
    collapsed = re.sub(r"_+", "_", sanitised).strip("_")
    return collapsed or "video"
