# NeuroPose

A 3D pose-estimation and kinematic-analysis system for neurological-recovery research, developed in **Liqi Shu's laboratory** at the Brown University Department of Neurology.

> Four externally-funded sub-projects since 2023; clinical-implications manuscript in preparation, with target submission in 2026–2027. The repository is a ground-up rewrite of the prior internal prototype.

## What it does

NeuroPose ingests video of patients during clinical assessments and produces validated, per-frame 3D pose predictions, joint-angle time series, and motion-comparison statistics suitable for downstream neurological analysis. The pipeline is designed for research-clinical workflows: it accepts video over a filesystem-polling interface, persists job state across crashes and restarts, and emits structured outputs that decouple inference from analysis.

The system underpins four externally funded sub-projects in Liqi Shu's lab investigating motor-function recovery in stroke, Parkinson's disease, and neuromuscular disorders. A clinical-implications manuscript synthesizing results across those sub-projects is in preparation.

## Architecture

Three stages, one module each:

- **`estimator`** — per-video worker. Streams frames from an input video via OpenCV, runs MeTRAbs on each frame, and returns a validated `VideoPredictions` (per-frame `boxes`, `poses3d`, `poses2d` plus a `VideoMetadata` envelope with frame count, fps, and resolution). Pure library; no filesystem semantics.
- **`interfacer`** — filesystem-polling daemon. Watches the configured input directory for new job subdirectories, dispatches each to an `Estimator`, and persists job state (`status.json`) across crashes and restarts. Single instance enforced via `fcntl.flock`. Owns the input → output → failed directory lifecycle.
- **`analyzer`** — post-processing subpackage. FastDTW-based motion comparison (`dtw_all`, `dtw_per_joint`, `dtw_relation`) and joint-angle / feature-statistics helpers. Pure functions operating on `VideoPredictions`. Heavy dependencies (fastdtw, scipy) are lazy-imported so `import neuropose.analyzer` works without the `analysis` extra.

Configuration is centralized in `src/neuropose/config.py` (a pydantic-settings `Settings` class). The runtime data directory defaults to `$XDG_DATA_HOME/neuropose/jobs/` and never lives inside the repository.

## Repository layout

```text
neuropose/
├── .github/workflows/           # CI: ruff + pyright + pytest, mkdocs
├── src/neuropose/
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

## Development

Requires Python 3.11 and [`uv`](https://github.com/astral-sh/uv).

```bash
git clone https://github.com/levineuwirth/neuropose.git
cd neuropose
uv sync --group dev
```

`uv sync --group dev` creates `.venv/` automatically and installs the runtime stack (pydantic, typer, OpenCV, TensorFlow, matplotlib) plus the full dev toolchain (pytest, ruff, pyright, pre-commit, mkdocs-material, fastdtw, scipy). First sync downloads ~600 MB of TensorFlow; subsequent runs hit the uv cache.

Install the pre-commit hooks:

```bash
uv run pre-commit install
```

### Tests

```bash
uv run pytest                  # unit tests only (default)
uv run pytest --runslow        # unit + integration; downloads ~2 GB MeTRAbs model
uv run pytest -m "not slow"    # explicitly exclude slow tests
```

Integration tests live under `tests/integration/` and are gated behind `@pytest.mark.slow` plus a custom `--runslow` flag implemented in `tests/conftest.py`. The first `--runslow` run downloads the pinned MeTRAbs tarball (~2 GB) into a session-scoped temp cache; subsequent tests in the same run reuse it.

### Lint and type-check

```bash
uv run ruff check .
uv run ruff format .
uv run pyright
```

CI runs lint, typecheck, and test as three parallel jobs on every push and PR to `main`.

### Docs

```bash
uv run mkdocs serve            # live-reload preview at http://127.0.0.1:8000
uv run mkdocs build --strict   # same check CI runs
```

API reference pages under `docs/api/` are auto-generated from source docstrings via mkdocstrings, so they cannot drift out of sync with the implementation.

## Data and ethics

The system processes patient videos collected under IRB-approved protocols in Liqi Shu's laboratory. The repository contains code only; no patient data, model weights are downloaded at runtime. Researchers who want to use NeuroPose with their own video should contact the lab.

## Authors

- **Levi Neuwirth** — Technical lead. Department of Computer Science, Brown University.
- **Liqi Shu** — Principal investigator. Department of Neurology, Warren Alpert Medical School, Brown University.

See `AUTHORS.md` for the full contributor list.

## Citation

A `CITATION.cff` is included in the repository for the current rewrite. The clinical-implications manuscript will be added once published. In the meantime:

```bibtex
@software{neuwirth2026neuropose,
  author = {Neuwirth, Levi and Shu, Liqi},
  title  = {NeuroPose: 3D Pose Estimation and Kinematic Analysis for Neurological Recovery Research},
  year   = {2026},
  url    = {https://github.com/levineuwirth/neuropose},
  note   = {Manuscript in preparation.}
}
```

For project context and ongoing work, see [levineuwirth.org/essays/neuropose](https://levineuwirth.org/essays/neuropose/).
