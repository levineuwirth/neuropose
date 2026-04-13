# Development

This page is for contributors working on NeuroPose itself.

## Environment setup

NeuroPose uses [`uv`](https://github.com/astral-sh/uv) for dependency
management and Python 3.11. After cloning the repository:

```bash
uv venv --python 3.11
source .venv/bin/activate
uv sync --group dev
uv run pre-commit install
```

`uv sync --group dev` installs the project in editable mode alongside
the full dev dependency set (pytest, ruff, pyright, pre-commit,
mkdocs-material, and mkdocstrings).

`pre-commit install` wires the git hooks declared in
`.pre-commit-config.yaml` into your local repo so every commit is
linted, formatted, and scanned for secrets before it lands.

## Running tests

Unit tests are fast and do not require the MeTRAbs model or TensorFlow
inference:

```bash
uv run pytest
```

Integration tests that require a downloaded model are marked with
`@pytest.mark.slow` and are skipped by default. Run them with:

```bash
uv run pytest -m slow
```

Run a specific test file or test class:

```bash
uv run pytest tests/unit/test_estimator.py
uv run pytest tests/unit/test_estimator.py::TestProcessVideo
uv run pytest -k "frame_count"
```

The autouse `_isolate_environment` fixture in `tests/conftest.py` points
`$HOME` and `$XDG_DATA_HOME` at a per-test temp directory, so no test
can accidentally write to your real home directory. It also clears any
`NEUROPOSE_*` variables from your shell so test outcomes do not depend
on who is running them.

## Linting and formatting

NeuroPose uses [`ruff`](https://docs.astral.sh/ruff/) for both lint and
format. Configuration lives in `pyproject.toml` under `[tool.ruff]`.

```bash
uv run ruff check .          # Lint
uv run ruff check --fix .    # Lint + auto-fix
uv run ruff format .         # Format (equivalent to black)
uv run ruff format --check . # Verify formatted
```

The selected lint rules are deliberately broad — pycodestyle, pyflakes,
isort, bugbear, pyupgrade, simplify, ruff-specific, pep8-naming,
comprehensions, pathlib, pytest-style, tidy-imports, numpy-specific,
and pydocstyle (numpy convention). The rationale is "lint noise early
rather than cruft late": we would rather annoy a contributor with a
style fix than let a real bug slip through because the linter was lax.

## Type checking

NeuroPose uses [`pyright`](https://github.com/microsoft/pyright) in
`standard` mode (not `strict` — the TensorFlow / OpenCV / scikit-learn
stubs would generate thousands of false positives under strict). The
plan is to tighten toward strict after the MeTRAbs stack is pinned in
commit 11.

```bash
uv run pyright
```

## Documentation

Documentation is built with [MkDocs](https://www.mkdocs.org/) and the
[Material theme](https://squidfunk.github.io/mkdocs-material/). API
reference pages are auto-generated from the source docstrings by
[mkdocstrings](https://mkdocstrings.github.io/).

Live preview at `http://localhost:8000`:

```bash
uv run mkdocs serve
```

Strict build (the same one CI runs):

```bash
uv run mkdocs build --strict
```

`--strict` promotes every warning (broken internal link, missing nav
entry, unparseable docstring) to an error, so broken docs fail the
build instead of silently producing a broken site.

Adding a new module means:

1. Write it with numpy-style docstrings (the plugin's
   `docstring_style: numpy` setting).
2. Add a stub page under `docs/api/` containing a single `:::` directive:
   ```markdown
   ::: neuropose.your_module
   ```
3. Add a nav entry in `mkdocs.yml` under `API Reference`.

## Project structure

```text
neuropose/
├── src/neuropose/              # The package itself
│   ├── config.py               # pydantic-settings Settings class
│   ├── estimator.py            # per-video MeTRAbs worker
│   ├── interfacer.py           # filesystem-polling daemon
│   ├── visualize.py            # matplotlib overlay rendering
│   ├── io.py                   # prediction schema + atomic save/load
│   ├── cli.py                  # typer CLI entrypoint
│   ├── _model.py               # MeTRAbs loader (stub pending commit 11)
│   └── analyzer/               # post-processing (pending commit 10)
├── tests/
│   ├── conftest.py             # isolated env + synthetic video fixtures
│   ├── unit/                   # fast, no model download
│   └── integration/            # marked slow, downloads the model
├── docs/                       # this documentation
├── .github/workflows/          # CI + docs workflows
├── pyproject.toml              # package metadata, deps, tool configs
└── mkdocs.yml                  # docs site configuration
```

## Commit hygiene

- **Small commits.** Each commit should do one thing and leave the repo
  in a green-CI state.
- **Descriptive commit messages.** The body should explain *why*, not
  restate the diff. References to audit sections or issue numbers are
  welcome.
- **No force-push on `main`.** Use a feature branch and open a merge
  request on the primary forge. `main` is protected; the CI checks
  must pass before merging.
- **No `git commit --no-verify`.** If a pre-commit hook fails, fix the
  underlying issue rather than skipping the hook. The hooks exist
  because the previous prototype was the poster child for what happens
  when hygiene slips.

## Release process

*To be documented when the first tagged release is cut.* The short
version of the plan:

1. Bump `version` in `pyproject.toml` and `__version__` in
   `src/neuropose/__init__.py`.
2. Update `CHANGELOG.md`.
3. Tag the commit (`git tag v0.1.0`).
4. Push the tag. A release workflow builds the wheel + sdist and
   uploads to PyPI once we claim the name.
