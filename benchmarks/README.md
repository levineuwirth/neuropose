# `benchmarks/`

Workflow convenience directory for performance benchmarking. Drop
videos into `benchmarks/videos/`, then rsync the whole repo to a
research machine — the videos travel alongside the code without
needing a second path on either end.

This directory is **not** a persistent data store. It is a local
scratch area for ephemeral benchmark inputs. Everything under
`benchmarks/videos/` is gitignored (see the `.gitignore` rules and
the per-directory README) so committing videos by accident is
mechanically prevented.

## Policy

**Only put benchmark test videos here. Never subject or clinical
recordings.** Subject data still goes through
`$XDG_DATA_HOME/neuropose/` via the normal interfacer workflow, and
that rule is load-bearing for the project's data-handling posture.

The distinction in practice:

| Type of video                                   | Goes here | Goes to `$XDG_DATA_HOME` |
| ------------------------------------------------ | --------- | ------------------------ |
| Synthetic test videos (gradient frames, etc.)    | Yes       | No                       |
| Public-domain reference footage                  | Yes       | No                       |
| Recordings you personally filmed for benchmarking | Yes       | No                       |
| Anything a clinician recorded                    | **No**    | Yes                      |
| Anything with an identifiable subject            | **No**    | Yes                      |
| Anything IRB-gated                               | **No**    | Yes                      |

When in doubt, route through `$XDG_DATA_HOME`. That path has
cross-machine isolation by design; this directory does not.

## Usage

Assuming you have a short test video to work with:

```console
$ cp ~/Downloads/short_clip.mp4 benchmarks/videos/
$ uv run neuropose benchmark benchmarks/videos/short_clip.mp4 \
    --repeats 5 --warmup-frames 3 \
    --output benchmarks/videos/short_clip_run.json
```

The `*.json` benchmark output is also gitignored — `.json` is not
a tracked extension inside `benchmarks/videos/` because only
`README.md` is whitelisted in that directory.

## Rsyncing to the research Mac

The directory layout is designed so one `rsync` path covers both
code and videos:

```console
$ rsync -av --delete \
    --exclude='.venv/' \
    --exclude='site/' \
    --exclude='.git/' \
    ~/Repos/research/brown/shu/neuropose/ \
    mac.local:~/Repos/research/brown/shu/neuropose/
```

After the sync, the videos in `benchmarks/videos/` on the Mac are
identical to the ones on Linux, so a benchmark run on the Mac can
reference the same filename the Linux report does — makes cross-
machine comparisons trivial.

Tips:

- Add `--exclude='benchmarks/videos/*.json'` if you want to keep
  per-machine benchmark results isolated.
- `--delete` makes the target exactly mirror the source. Without
  it, old files on the target persist — safer but surprising.
- For one-off pushes, `scp benchmarks/videos/clip.mp4
  mac.local:~/Repos/research/brown/shu/neuropose/benchmarks/videos/`
  works without touching the rest of the repo.

## Bulk intake via `neuropose ingest`

When you have a whole batch of recordings instead of a single
benchmark clip, drop the zip archive anywhere you like (the
`benchmarks/videos/` directory is fine for transient test zips) and
let `neuropose ingest` unpack them into per-video job directories
for the running daemon:

```console
$ uv run neuropose ingest benchmarks/videos/session_2026-04-15.zip
ingested 12 job(s) from benchmarks/videos/session_2026-04-15.zip (1842.3 MB, 2 non-video member(s) skipped)
  patient_001_trial_01/trial_01.mp4
  patient_001_trial_02/trial_02.mp4
  ...
```

Each video becomes its own `$data_dir/in/<job_name>/` directory,
and the daemon picks them up on its next poll with no further
operator action. Pass `--force` to overwrite existing job
directories with the same derived names. See the `neuropose
ingest --help` output for the full flag surface and
[`neuropose.ingest`](../docs/api/ingest.md) for the library API.

**Reminder:** the `benchmarks/videos/` directory is for benchmark
test data only, including the zip archives you pass to `ingest`.
Actual clinical recordings should not transit through this
directory — route those through `$XDG_DATA_HOME` instead.

## Directory layout

```
benchmarks/
├── README.md           # this file (tracked)
└── videos/
    ├── README.md       # placeholder to keep the directory tracked
    └── <your-clip>.mp4 # ignored
```
