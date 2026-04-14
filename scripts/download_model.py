#!/usr/bin/env python
"""Pre-download the pinned MeTRAbs model into the NeuroPose cache.

Run this on a machine with network access before going offline, or to
pre-warm a deployment target's cache so the first ``neuropose watch``
or ``neuropose process`` invocation does not stall on a ~2 GB
download.

Usage::

    uv run python scripts/download_model.py [--cache-dir PATH]

If ``--cache-dir`` is omitted, the script uses
``Settings().model_cache_dir`` (``$XDG_DATA_HOME/neuropose/models`` by
default), which is the same location the daemon and the direct
``Estimator`` entry points read from.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Destination directory for the cached model. Defaults to "
        "the value of Settings().model_cache_dir.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Deferred imports so --help runs cheaply without importing TF.
    from neuropose._model import load_metrabs_model
    from neuropose.config import Settings

    if args.cache_dir is None:
        settings = Settings()
        cache_dir = settings.model_cache_dir
    else:
        cache_dir = args.cache_dir

    print(f"Fetching MeTRAbs model into {cache_dir}", file=sys.stderr)
    try:
        load_metrabs_model(cache_dir=cache_dir)
    except Exception as exc:
        print(f"error: model download failed: {exc}", file=sys.stderr)
        return 1

    print("Download complete", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
