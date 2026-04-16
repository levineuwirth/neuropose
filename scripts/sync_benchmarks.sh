#!/usr/bin/env bash
# Sync the benchmarks/ directory from this repo checkout to a remote
# machine — typically the research Mac where inference actually runs.
#
# Defaults match the Shu Lab research Mac (a Tailscale address):
#
#   Levi@100.64.15.110:/Users/levi/Repos/neuropose/benchmarks/
#
# Override either side via environment variables:
#
#   REMOTE_HOST=user@host         # SSH target
#   REMOTE_PATH=/absolute/path/   # Remote destination (trailing slash
#                                 # recommended — see rsync(1))
#
# Flags:
#
#   -n, --dry-run   Show what would be transferred, change nothing.
#   --delete        Make the remote exactly mirror the local source,
#                   deleting any files on the remote that are not in
#                   the local tree. OFF by default because the remote
#                   may hold per-machine benchmark result JSONs you
#                   do not want to lose. Pass --delete explicitly
#                   when you *want* a clean mirror.
#   -h, --help      Print this header.
#
# What gets transferred:
#
#   The entire benchmarks/ directory, including videos/*.mp4 and any
#   *.json result files. The directory's .gitignore status is
#   irrelevant to rsync — rsync copies what is in the working tree,
#   not what is tracked by git — so the videos you drop into
#   benchmarks/videos/ travel to the remote exactly as they sit on
#   disk.
#
# Safety:
#
#   - The script resolves its source directory relative to the repo
#     root via `git rev-parse --show-toplevel`, so it works from any
#     subdirectory of the checkout and never accidentally syncs some
#     other `benchmarks/` that happens to live in $PWD.
#   - --partial keeps half-transferred files on the remote so an
#     interrupted run can resume on the next invocation.
#   - Without --delete, the script is purely additive: it will create
#     and update files on the remote but never remove them.
#
# Examples:
#
#   scripts/sync_benchmarks.sh
#   scripts/sync_benchmarks.sh --dry-run
#   scripts/sync_benchmarks.sh --delete
#   REMOTE_HOST=me@other-host scripts/sync_benchmarks.sh

set -euo pipefail

REMOTE_HOST="${REMOTE_HOST:-Levi@100.64.15.110}"
REMOTE_PATH="${REMOTE_PATH:-/Users/levi/Repos/neuropose/benchmarks/}"

dry_run=0
do_delete=0

while [ $# -gt 0 ]; do
    case "$1" in
        -n|--dry-run)
            dry_run=1
            ;;
        --delete)
            do_delete=1
            ;;
        -h|--help)
            # Echo the leading comment block as the help text so the
            # in-script documentation is the single source of truth.
            sed -n '2,60p' "$0" | sed 's/^# \{0,1\}//'
            exit 0
            ;;
        *)
            echo "error: unknown argument: $1" >&2
            echo "run with --help for usage" >&2
            exit 2
            ;;
    esac
    shift
done

# Resolve the repo root so this script works from any cwd inside the
# checkout. $(dirname "$0") is the scripts/ directory; the parent is
# the repo root.
script_dir="$(cd "$(dirname "$0")" && pwd)"
repo_root="$(git -C "$script_dir" rev-parse --show-toplevel)"
source_dir="$repo_root/benchmarks/"

if [ ! -d "$source_dir" ]; then
    echo "error: source directory does not exist: $source_dir" >&2
    exit 1
fi

# rsync flags:
#   -a                 archive mode (recurse + preserve metadata)
#   -h                 human-readable sizes
#   --partial          resume interrupted transfers on the next run
#   --info=progress2   single running progress line (rsync >= 3.1.0
#                      on the local side; the remote's rsync version
#                      is irrelevant because progress is client-side)
rsync_flags=(-ah --partial --info=progress2)

if [ $dry_run -eq 1 ]; then
    rsync_flags+=(--dry-run)
    echo "[dry-run] no files will actually be transferred"
fi

if [ $do_delete -eq 1 ]; then
    rsync_flags+=(--delete)
    echo "[delete] remote will be pruned to exactly mirror local"
fi

echo "source: $source_dir"
echo "target: $REMOTE_HOST:$REMOTE_PATH"
echo

exec rsync "${rsync_flags[@]}" "$source_dir" "$REMOTE_HOST:$REMOTE_PATH"
