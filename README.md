# NeuroPose (rewrite)

Ground-up rewrite of the prior NeuroPose internal prototype. 

## Target layout

```text
neuropose/
├── .github/workflows/           # CI on the GitHub mirror (ruff + pyright + pytest)
├── src/neuropose/
│   ├── __init__.py              # version only
│   ├── config.py                # pydantic-settings Settings class
│   ├── io.py                    # prediction schema, load/save helpers
│   ├── estimator.py             # per-video MeTRAbs worker (ported, cleaned)
│   ├── interfacer.py            # filesystem-polling daemon (ported, cleaned)
│   ├── cli.py                   # typer app (`neuropose run|watch|analyze`)
│   ├── _model.py                # MeTRAbs model load + local cache
│   └── analyzer/                # rewrite of the prior analyzer.py
│       ├── dtw.py               # FastDTW helpers
│       ├── features.py          # normalization, padding, joint angles
│       └── classification.py    # sktime classifier wrappers
├── tests/
│   ├── unit/                    # fast, no model download
│   ├── integration/             # marked @slow, downloads the model
│   └── fixtures/                # synthetic video + reference predictions
├── docs/                        # mkdocs-material site
├── notebooks/                   # getting_started.ipynb, tested in CI via nbval
├── config/default.yaml          # example runtime config
├── scripts/download_model.py
├── pyproject.toml               # hatchling build, typer + pydantic + TF stack
├── Dockerfile                   # CPU, pinned deps
├── AUTHORS.md
├── CITATION.cff
└── LICENSE                      # MIT
```
## Architecture

Three stages, one module each:

- **`estimator`** — per-video worker. Extracts frames from an input video,
  runs MeTRAbs on each frame, and writes per-frame predictions (`boxes`,
  `poses3d`, `poses2d`) to JSON. No daemon logic; usable directly from Python.
- **`interfacer`** — filesystem-polling daemon. Watches the configured input
  directory for new job subdirectories, dispatches each to an `Estimator`,
  and persists job state (`status.json`) across crashes and restarts. Owns
  the input → output → failed directory lifecycle.
- **`analyzer`** — post-processing subpackage. FastDTW-based motion
  comparison, joint-angle feature extraction, and sktime classifier wrappers.
  Operates on the JSON output of the estimator.

Configuration is centralized in `src/neuropose/config.py` (a
pydantic-settings `Settings` class). The runtime data directory defaults to
`$XDG_DATA_HOME/neuropose/jobs/` and never lives inside the repository.

## Commit plan

| # | Scope | State |
|---|---|---|
| 1 | Scaffolding: package layout, MIT license, authors, citation, policy-aware `.gitignore` | in review |
| 2 | Dev tooling: pre-commit, ruff, pyright, gitleaks | planned |
| 3 | CI workflow on the GitHub mirror | planned |
| 4 | `config.py`, `io.py`, unit tests | planned |
| 5 | Port `estimator.py` with typing and audit §6 fixes | planned |
| 6 | Port `interfacer.py` with audit §7 fixes | planned |
| 7 | Typer CLI (`neuropose run|watch|analyze`) | planned |
| 8 | mkdocs-material docs site | planned |
| 9 | Data-handling policy (gates going public) | blocked on IRB prose |
| 10 | `analyzer/` subpackage rewrite | planned |
| 11 | MeTRAbs model loader + integration smoke test | blocked on upstream URL + TF pin |
| 12 | Dockerfile | blocked on 11 |
| 13 | Comprehensive `CHANGELOG.md` retroactive entry | blocked on 12 |
