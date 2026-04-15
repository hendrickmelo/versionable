# Contributing to versionable

Thanks for your interest in contributing! This document covers how to set up the project, run checks, and submit
changes.

## Getting Started

External contributions should come from a **fork**. Fork the repo on GitHub, then clone your fork:

```bash
git clone https://github.com/<your-username>/versionable.git
cd versionable
```

versionable uses [Pixi](https://pixi.sh) for environment management:

```bash
pixi install
```

This installs all dependencies, including optional backends (TOML, YAML, HDF5) and dev tools (pytest, mypy, ruff). When
you're ready, open a PR from your fork against `main`.

## Running Checks

Before submitting a PR, run the full cleanup and test suite:

```bash
pixi run cleanup          # formatters, linters, type checkers
pixi run test             # pytest
```

`cleanup` runs ruff (format + lint), mypy, pyright, prettier (markdown), and markdownlint. Fix all errors before pushing
— CI runs the same checks.

To run checks individually:

```bash
pixi run format           # ruff format
pixi run ruff             # ruff check --fix
pixi run mypy             # mypy src/
pixi run pyright          # pyright src/
pixi run test             # pytest
```

## CI Environments

CI runs in two environments:

- **default** — all backends installed (JSON, TOML, YAML, HDF5)
- **minimal** — JSON backend only (no h5py, pyyaml, or toml)

You can run both locally:

```bash
pixi run ci-all
```

## Building Documentation

Documentation uses Sphinx with MyST markdown and lives in the `docs/` directory. It builds in its own pixi environment:

```bash
pixi run -e docs docs             # one-time build → docs/_build/html/
pixi run -e docs docs-live        # live-reload server for local editing
```

CI also builds the docs on every PR, so you'll see a failure if your changes break the build.

## Code Style

### Naming Conventions

This project uses **camelCase** for functions, methods, and variables — see the
[FAQ](https://versionable.readthedocs.io/faq.html#why-camelcase-instead-of-snake-case) for why. Test files use
`snake_case` (pytest convention).

| What                            | Style                            | Example                         |
| ------------------------------- | -------------------------------- | ------------------------------- |
| Functions / methods / variables | `camelCase`                      | `computeHash()`, `fieldType`    |
| Classes / types                 | `PascalCase`                     | `Versionable`, `JsonBackend`    |
| Constants                       | `SCREAMING_SNAKE_CASE`           | `_CANONICAL_NAMES`              |
| Private members                 | Leading underscore               | `_registry`, `_resolveFields()` |
| Test functions                  | `snake_case` with `test_` prefix | `test_roundtrip()`              |

### Type Annotations

All functions and instance variables must have type hints. Use modern syntax (`X | Y`, `list[T]`, `dict[K, V]`). Avoid
`Any` unless strictly necessary — add a short comment explaining why if you do.

Never use `# type: ignore`, `# noqa:`, or pyright `exclude` to suppress type errors. Fix the underlying issue instead.

### Imports

```python
from __future__ import annotations

# 1. Standard library
import hashlib
from typing import Any

# 2. Third-party
import numpy as np

# 3. Local
from versionable._hash import computeHash
```

### Tests

- Tests live in `tests/` and use pytest
- Test file names match `test_*.py`
- Use `snake_case` for all identifiers in test files
- Hardcode schema hashes as string literals (e.g., `hash="74a182"`) — don't call `computeHash()` in tests

## Pull Requests

- Open PRs from your fork
- Target the `main` branch
- Preserve individual commit history — we don't squash merge
- Keep PR descriptions concise: what changed, why, and what you tested

### PR Description Format

```text
**Description:** Brief explanation of what and why

**Changes:**
- Bullet point of change 1
- Bullet point of change 2

**Tests performed:**
- Test scenario 1
- Test scenario 2
```

## Project Structure

```text
src/versionable/       Source code (src layout)
  __init__.py          Public API exports
  _api.py              save() / load() entry points
  _base.py             Versionable mixin, registry, metadata
  _hash.py             Schema hash computation
  _types.py            Type converter registry + built-in converters
  _migration.py        Declarative and imperative migrations
  _backend.py          Backend abstract base class
  _json_backend.py     JSON backend
  _toml_backend.py     TOML backend (optional)
  _yaml_backend.py     YAML backend (optional)
  _hdf5_backend.py     HDF5 backend (optional)
  _hdf5_session.py     Incremental HDF5 sessions
  errors.py            Exception hierarchy
tests/                 pytest test suite
docs/                  Sphinx documentation (MyST markdown)
```

Private modules (prefixed with `_`) are internal — public API is exposed only through `__init__.py`.

## Reporting Issues

File issues on [GitHub](https://github.com/hendrickmelo/versionable/issues). Include:

- What you expected vs. what happened
- A minimal reproducible example
- Python version and versionable version (`python -c "import versionable; print(versionable.__version__)"`)
