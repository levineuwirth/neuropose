"""Tests for :mod:`neuropose.ingest`.

Coverage:

- Happy path — nested and top-level videos produce one job each.
- Job-name derivation — flattening, sanitization, collapsing.
- Non-video members are skipped, not errors.
- Zip-internal collisions (two videos → same job name) reported up
  front.
- External collisions (target job dir already exists) are listed in
  one error; ``--force`` deletes and replaces.
- Security: path-traversal and absolute-path members refused; empty
  archive and oversize archive refused.
- Atomicity: when extraction fails midway, no partial state is left
  behind under ``input_dir``.
"""

from __future__ import annotations

import zipfile
from pathlib import Path

import pytest

from neuropose.ingest import (
    ArchiveEmptyError,
    ArchiveInvalidError,
    ArchiveTooLargeError,
    IngestResult,
    JobCollisionError,
    ingest_zip,
)


def _write_zip(path: Path, members: dict[str, bytes]) -> Path:
    """Create a zip at ``path`` with the given ``{name: bytes}`` members."""
    with zipfile.ZipFile(path, "w") as z:
        for name, data in members.items():
            z.writestr(name, data)
    return path


@pytest.fixture
def input_dir(tmp_path: Path) -> Path:
    """Return a fresh ``input_dir`` for the test."""
    d = tmp_path / "jobs" / "in"
    d.mkdir(parents=True)
    return d


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


class TestHappyPath:
    def test_top_level_video_becomes_job(self, tmp_path: Path, input_dir: Path) -> None:
        archive = _write_zip(tmp_path / "a.zip", {"clip_01.mp4": b"data"})
        result = ingest_zip(archive, input_dir)
        assert result.job_count == 1
        assert result.ingested[0].job_name == "clip_01"
        assert (input_dir / "clip_01" / "clip_01.mp4").read_bytes() == b"data"

    def test_nested_path_flattens_into_job_name(self, tmp_path: Path, input_dir: Path) -> None:
        archive = _write_zip(
            tmp_path / "a.zip",
            {"patient_001/trial_01.mp4": b"vid"},
        )
        result = ingest_zip(archive, input_dir)
        job = result.ingested[0]
        assert job.job_name == "patient_001_trial_01"
        # The video file inside the job dir keeps its basename, not the
        # flattened job name, so the daemon sees a clean filename.
        assert job.video_filename == "trial_01.mp4"
        assert (input_dir / "patient_001_trial_01" / "trial_01.mp4").exists()

    def test_sibling_nested_videos_do_not_collide(self, tmp_path: Path, input_dir: Path) -> None:
        archive = _write_zip(
            tmp_path / "a.zip",
            {
                "patient_001/trial_01.mp4": b"a",
                "patient_002/trial_01.mp4": b"b",
            },
        )
        result = ingest_zip(archive, input_dir)
        names = {j.job_name for j in result.ingested}
        assert names == {"patient_001_trial_01", "patient_002_trial_01"}

    def test_multiple_videos_produce_multiple_jobs(self, tmp_path: Path, input_dir: Path) -> None:
        archive = _write_zip(
            tmp_path / "a.zip",
            {f"clip_{i:02d}.mp4": f"d{i}".encode() for i in range(5)},
        )
        result = ingest_zip(archive, input_dir)
        assert result.job_count == 5
        assert len(list(input_dir.iterdir())) == 5

    def test_non_video_members_skipped(self, tmp_path: Path, input_dir: Path) -> None:
        archive = _write_zip(
            tmp_path / "a.zip",
            {
                "clip.mp4": b"video",
                "README.md": b"notes",
                ".DS_Store": b"junk",
                "notes.txt": b"notes",
            },
        )
        result = ingest_zip(archive, input_dir)
        assert result.job_count == 1
        assert sorted(result.skipped_non_videos) == sorted([".DS_Store", "README.md", "notes.txt"])

    def test_all_accepted_extensions(self, tmp_path: Path, input_dir: Path) -> None:
        archive = _write_zip(
            tmp_path / "a.zip",
            {
                "a.mp4": b"a",
                "b.avi": b"b",
                "c.mov": b"c",
                "d.mkv": b"d",
            },
        )
        result = ingest_zip(archive, input_dir)
        assert result.job_count == 4

    def test_returns_typed_result(self, tmp_path: Path, input_dir: Path) -> None:
        archive = _write_zip(tmp_path / "a.zip", {"clip.mp4": b"data"})
        result = ingest_zip(archive, input_dir)
        assert isinstance(result, IngestResult)
        assert result.total_uncompressed_bytes == 4


# ---------------------------------------------------------------------------
# Job-name sanitization
# ---------------------------------------------------------------------------


class TestJobNameDerivation:
    def test_special_chars_become_underscores(self, tmp_path: Path, input_dir: Path) -> None:
        archive = _write_zip(
            tmp_path / "a.zip",
            {"session 2026-04-15 / trial @1.mp4": b"v"},
        )
        result = ingest_zip(archive, input_dir)
        name = result.ingested[0].job_name
        # Every character ends up in the safe set; runs of underscores
        # are collapsed and leading/trailing stripped.
        assert name == "session_2026-04-15_trial_1"

    def test_all_symbol_name_falls_back_to_video(self, tmp_path: Path, input_dir: Path) -> None:
        archive = _write_zip(tmp_path / "a.zip", {"!!!.mp4": b"v"})
        result = ingest_zip(archive, input_dir)
        assert result.ingested[0].job_name == "video"


# ---------------------------------------------------------------------------
# Collision detection
# ---------------------------------------------------------------------------


class TestCollisions:
    def test_zip_internal_collision_rejects(self, tmp_path: Path, input_dir: Path) -> None:
        # Both entries flatten to the same job name because their
        # stems are the same and both are top-level after derivation.
        archive = _write_zip(
            tmp_path / "a.zip",
            {"a/b.mp4": b"x", "a/b.mp4.bak": b"y"},
        )
        # The second one is non-video (.bak suffix), so this is
        # actually a happy case. Build a real collision:
        archive = _write_zip(
            tmp_path / "b.zip",
            {"x__y.mp4": b"1", "x y.mp4": b"2"},
        )
        with pytest.raises(JobCollisionError):
            ingest_zip(archive, input_dir)
        # No files written.
        assert list(input_dir.iterdir()) == []

    def test_external_collision_without_force(self, tmp_path: Path, input_dir: Path) -> None:
        (input_dir / "clip").mkdir()
        (input_dir / "clip" / "existing.mp4").write_bytes(b"old")
        archive = _write_zip(tmp_path / "a.zip", {"clip.mp4": b"new"})
        with pytest.raises(JobCollisionError) as excinfo:
            ingest_zip(archive, input_dir)
        assert excinfo.value.collisions == ["clip"]
        # Existing job dir is untouched.
        assert (input_dir / "clip" / "existing.mp4").read_bytes() == b"old"

    def test_external_collision_listed_together(self, tmp_path: Path, input_dir: Path) -> None:
        for name in ("a", "b", "c"):
            (input_dir / name).mkdir()
        archive = _write_zip(
            tmp_path / "a.zip",
            {"a.mp4": b"1", "b.mp4": b"2", "c.mp4": b"3", "d.mp4": b"4"},
        )
        with pytest.raises(JobCollisionError) as excinfo:
            ingest_zip(archive, input_dir)
        assert sorted(excinfo.value.collisions) == ["a", "b", "c"]

    def test_force_overwrites_existing(self, tmp_path: Path, input_dir: Path) -> None:
        (input_dir / "clip").mkdir()
        (input_dir / "clip" / "existing.mp4").write_bytes(b"old")
        archive = _write_zip(tmp_path / "a.zip", {"clip.mp4": b"new"})
        result = ingest_zip(archive, input_dir, force=True)
        assert result.job_count == 1
        # The old file is gone; only the new one remains.
        files = list((input_dir / "clip").iterdir())
        assert [f.name for f in files] == ["clip.mp4"]
        assert (input_dir / "clip" / "clip.mp4").read_bytes() == b"new"


# ---------------------------------------------------------------------------
# Security
# ---------------------------------------------------------------------------


class TestSecurity:
    def test_absolute_path_member_rejected(self, tmp_path: Path, input_dir: Path) -> None:
        archive = _write_zip(tmp_path / "a.zip", {"/etc/passwd.mp4": b"x"})
        with pytest.raises(ArchiveInvalidError, match="absolute"):
            ingest_zip(archive, input_dir)
        assert list(input_dir.iterdir()) == []

    def test_traversal_member_rejected(self, tmp_path: Path, input_dir: Path) -> None:
        archive = _write_zip(tmp_path / "a.zip", {"../escape.mp4": b"x"})
        with pytest.raises(ArchiveInvalidError, match="traversal"):
            ingest_zip(archive, input_dir)
        assert list(input_dir.iterdir()) == []

    def test_empty_archive_rejected(self, tmp_path: Path, input_dir: Path) -> None:
        archive = _write_zip(tmp_path / "a.zip", {})
        with pytest.raises(ArchiveEmptyError):
            ingest_zip(archive, input_dir)

    def test_archive_with_only_non_videos_rejected(self, tmp_path: Path, input_dir: Path) -> None:
        archive = _write_zip(
            tmp_path / "a.zip",
            {"README.md": b"no videos here", "notes.txt": b"none"},
        )
        with pytest.raises(ArchiveEmptyError):
            ingest_zip(archive, input_dir)

    def test_too_large_archive_rejected(
        self,
        tmp_path: Path,
        input_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # Lower the cap for the test rather than building a real
        # multi-GB zip. The enforcement path is the same.
        monkeypatch.setattr("neuropose.ingest.MAX_UNCOMPRESSED_BYTES", 10)
        archive = _write_zip(
            tmp_path / "a.zip",
            {"clip.mp4": b"0123456789ABCDEF"},  # 16 bytes > 10
        )
        with pytest.raises(ArchiveTooLargeError):
            ingest_zip(archive, input_dir)
        assert list(input_dir.iterdir()) == []

    def test_bad_zip_file_rejected(self, tmp_path: Path, input_dir: Path) -> None:
        bad = tmp_path / "bad.zip"
        bad.write_bytes(b"this is not a valid zip file at all")
        with pytest.raises(ArchiveInvalidError):
            ingest_zip(bad, input_dir)

    def test_missing_archive_raises(self, tmp_path: Path, input_dir: Path) -> None:
        with pytest.raises(FileNotFoundError):
            ingest_zip(tmp_path / "nope.zip", input_dir)


# ---------------------------------------------------------------------------
# Atomicity
# ---------------------------------------------------------------------------


class TestAtomicity:
    def test_staging_directory_cleaned_up_on_success(self, tmp_path: Path, input_dir: Path) -> None:
        archive = _write_zip(tmp_path / "a.zip", {"clip.mp4": b"v"})
        ingest_zip(archive, input_dir)
        # No stray `.ingest_*` directories left under the parent.
        leftover = [p for p in input_dir.parent.iterdir() if p.name.startswith(".ingest_")]
        assert leftover == []

    def test_no_partial_state_when_planning_fails(self, tmp_path: Path, input_dir: Path) -> None:
        # An archive that will pass the zipfile open but fail at
        # planning (traversal member) should never write to input_dir.
        archive = _write_zip(tmp_path / "a.zip", {"../bad.mp4": b"v"})
        with pytest.raises(ArchiveInvalidError):
            ingest_zip(archive, input_dir)
        assert list(input_dir.iterdir()) == []
        leftover = [p for p in input_dir.parent.iterdir() if p.name.startswith(".ingest_")]
        assert leftover == []
