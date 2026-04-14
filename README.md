# NeuroPose (rewrite)

Ground-up rewrite of the prior NeuroPose internal prototype. The repository
is private while the IRB data-handling policy is being authored; this README
is aimed at contributors working on the rewrite, not external users.

## Layout

```text
neuropose/
├── .github/workflows/           # CI: ruff + pyright + pytest (ci.yml), mkdocs (docs.yml)
├── src/neuropose/
│   ├── __init__.py              # version only
│   ├── config.py                # pydantic-settings Settings class
│   ├── io.py                    # prediction schema, load/save helpers
│   ├── estimator.py             # per-video MeTRAbs worker
│   ├── interfacer.py            # filesystem-polling daemon
│   ├── visualize.py             # per-frame 2D + 3D overlay rendering
│   ├── cli.py                   # typer app (watch | process | analyze)
│   ├── _model.py                # MeTRAbs download + SHA-256 verify + load
│   └── analyzer/                # post-processing subpackage
│       ├── dtw.py               # FastDTW helpers
│       └── features.py          # normalization, padding, joint angles, stats
├── tests/
│   ├── conftest.py              # env isolation, synthetic video, fake model
│   ├── unit/                    # fast, no model download
│   └── integration/             # marked @slow, downloads the real MeTRAbs model
├── docs/                        # mkdocs-material site (mkdocs.yml at repo root)
├── scripts/download_model.py    # pre-warm the model cache
├── pyproject.toml               # hatchling build, dev group (PEP 735)
├── Dockerfile                   # CPU image, non-root, /data volume
├── CHANGELOG.md                 # Keep a Changelog format
├── RESEARCH.md                  # DTW methodology + MeTRAbs self-hosting R&D log
├── AUTHORS.md
├── CITATION.cff
└── LICENSE                      # MIT
```

## Architecture

Three stages, one module each:

- **`estimator`** — per-video worker. Streams frames from an input video via
  OpenCV, runs MeTRAbs on each frame, and returns a validated
  `VideoPredictions` (per-frame `boxes`, `poses3d`, `poses2d` plus a
  `VideoMetadata` envelope with frame count, fps, and resolution). Pure
  library — no filesystem semantics.
- **`interfacer`** — filesystem-polling daemon. Watches the configured input
  directory for new job subdirectories, dispatches each to an `Estimator`,
  and persists job state (`status.json`) across crashes and restarts. Single
  instance enforced via `fcntl.flock`. Owns the input → output → failed
  directory lifecycle.
- **`analyzer`** — post-processing subpackage. FastDTW-based motion
  comparison (`dtw_all`, `dtw_per_joint`, `dtw_relation`) and joint-angle /
  feature-statistics helpers. Pure functions operating on `VideoPredictions`.
  Heavy dependencies (fastdtw, scipy) are lazy-imported so
  `import neuropose.analyzer` works without the `analysis` extra.

Configuration is centralized in `src/neuropose/config.py` (a
pydantic-settings `Settings` class). The runtime data directory defaults to
`$XDG_DATA_HOME/neuropose/jobs/` and never lives inside the repository.

## Development setup

Requires Python 3.11 and [`uv`](https://github.com/astral-sh/uv).

```bash
git clone https://git.levineuwirth.org/neuwirth/neuropose.git
cd neuropose
uv sync --group dev
```

`uv sync --group dev` creates `.venv/` automatically and installs the
runtime stack (pydantic, typer, OpenCV, TensorFlow, matplotlib) plus the
full dev toolchain (pytest, ruff, pyright, pre-commit, mkdocs-material,
fastdtw, scipy). First sync downloads ~600 MB of TensorFlow; subsequent
runs hit the uv cache.

Install the pre-commit hooks:

```bash
uv run pre-commit install
```

### Running tests

```bash
uv run pytest                  # unit tests only (default)
uv run pytest --runslow        # unit + integration; downloads ~2 GB MeTRAbs model
uv run pytest -m "not slow"    # explicitly exclude slow tests
```

Integration tests live under `tests/integration/` and are gated behind
`@pytest.mark.slow` plus a custom `--runslow` flag implemented in
`tests/conftest.py`. Without the flag, slow tests are skipped at collection
time. The first `--runslow` run downloads the pinned MeTRAbs tarball
(~2 GB) into a session-scoped temp cache; subsequent tests in the same run
reuse it.

### Lint and type-check

```bash
uv run ruff check .
uv run ruff format .
uv run pyright
```

CI runs lint, typecheck, and test as three parallel jobs on every push and
PR to `main` — see `.github/workflows/ci.yml`.

### Docs

```bash
uv run mkdocs serve            # live-reload preview at http://127.0.0.1:8000
uv run mkdocs build --strict   # same check CI runs
```

The API reference pages under `docs/api/` are auto-generated from source
docstrings via mkdocstrings, so they cannot drift out of sync.
