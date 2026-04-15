# versionable — Python 3.12+ Library

## General Preferences

- Do not add Claude attribution to commit messages or PRs
- Keep PR summaries concise - avoid verbose explanations
- Prefer better type hints over casting or `# type: ignore` or `# noqa:` comments
- Documentation should be concise and to the point
- If an implementation isn't working after a few iterations, stop and ask the user to intervene
- Always specify a language tag on fenced code blocks in markdown (e.g., `python`, `bash`, `text`)
- Do not fix markdown lint warnings during editing — they are distracting and cosmetic. If a clean pass is needed, run
  it once at the end.
- In examples and tests, always hardcode schema hashes as string literals (e.g., `hash="74a182"`). Only use
  `computeHash()` when strictly necessary (e.g., computing hashes programmatically in library internals).

## Shortcuts

- "qreview" → Check for type errors, missing tests, security issues, unused imports
- "ship it" → Run `pixi run cleanup`, fix any errors, then create a commit
- "quick test" → Run `pixi run pytest -x` (stop on first failure)
- "full check" → Run `pixi run cleanup && pixi run pytest`

## Overview

Versioned persistence framework for Python dataclasses with schema hash validation, declarative migrations, type
converters, and pluggable storage backends (JSON, TOML, YAML, HDF5).

**Project Structure:**

- `src/versionable/` — Package source (src layout)
- `tests/` — pytest test suite
- `README.md` — Usage guide and API documentation

## Build System: Pixi

**Use Pixi for all environment management. The package is also pip-installable via hatchling for consumers.**

```bash
pixi install              # Install all dependencies
pixi add <package>        # Add new dependencies
pixi shell                # Activate environment
pixi run <task>           # Run defined tasks
pixi run -- <cmd>         # Run arbitrary commands in the pixi environment
```

All new dependencies → [pixi.toml](pixi.toml)

## Versioning

The version lives in `pyproject.toml` (`version = "X.Y.Z"` or `"X.Y.Z.dev0"`).

**Release cycle:**

1. Between releases, `pyproject.toml` has a dev version (e.g., `0.1.1.dev0`) — PRs do not need to increment it
2. To release: update `pyproject.toml` to the release version (e.g., `0.1.1`), merge to `main`, create a GitHub Release
   tagged `v0.1.1`
3. The publish workflow (`publish.yml`) triggers on the published GitHub Release and publishes to PyPI
4. After publishing, the workflow automatically opens a PR to bump `pyproject.toml` to the next dev version (e.g.,
   `0.1.2.dev0`)

**Rules:**

- Never leave a release version (without `.dev0`) on `main` — always bump to the next dev version immediately after tagging
- Release branches (e.g., `release/v0.1.1`) are created from the tag only if hotfixes are needed — not before
- `__version__` is read at runtime via `importlib.metadata` — it reflects whatever is installed

## Pre-Commit Checklist

**CRITICAL: Always run cleanup before committing:**

```bash
pixi run cleanup          # Runs formatters, linters, and type checks
```

Fix all errors reported by cleanup before creating commits. Do not commit if cleanup fails.

### Type Checking Exclusions

Never exclude Python source files from pyright or mypy in `pyproject.toml`. Fix the type errors instead. If a
suppression is truly unavoidable (e.g., broken third-party type stubs), use the narrowest possible scope:

1. Inline `# pyright: ignore[ruleCode]` on the specific line (with a comment explaining why)
2. File-level `# pyright: ruleCode=false` only if every line in the file triggers the same stub issue
3. Never add files to the pyright `exclude` list — all source code must be type-checked

## PR Workflow

- Default to creating draft PRs, unless asked otherwise.
- When creating a PR, show the title and summary for review before actually creating it.
- Never modify or update merged PRs.

### Pull Request Format

**Description:** Brief explanation of what and why

**Changes:**

- Bullet point of change 1
- Bullet point of change 2

**Tests performed:** (if applicable)

- Test scenario 1
- Test scenario 2

## Python Coding Standards

### Naming Conventions

- **Functions/methods/variables**: `camelCase`
    - Example: `def computeHash()`, `fieldType`, `nativeTypes`
    - Variables that have units include the unit in the name (e.g., `frequency_Hz`)
- **Classes/types**: `PascalCase`
    - Example: `Versionable`, `Migration`, `JsonBackend`
- **Constants**: `SCREAMING_SNAKE_CASE`
    - Example: `_CANONICAL_NAMES`, `_UNHANDLED`
- **Private members**: Leading underscore
    - Example: `_registry`, `_ops`, `def _resolveFields()`
- **Private modules**: Leading underscore
    - Example: `_api.py`, `_types.py` (public API exposed only through `__init__.py`)
- **Test functions and all code inside test files**: `snake_case` with `test_` prefix for test functions (pytest convention); all other identifiers in test files (helpers, variables, parameters) also use `snake_case`
    - Example: `def test_roundtrip()`, `def test_dtype_preserved()`, `def _has_toml()`, `src_dtype`

### Type Annotations

Use comprehensive type hints on all functions and instance variables:

```python
def computeHash(fields: dict[str, Any]) -> str:
    ...
```

- Avoid using `Any` unless strictly necessary. If found to be necessary, add a short explanation.
- Use modern typing features: `X | Y` unions, `list[T]`, `dict[K, V]`

### Import Organization

```python
from __future__ import annotations

# 1. Standard library
import hashlib
import logging
from typing import Any

# 2. Third-party
import numpy as np

# 3. Local
from versionable.errors import VersionableError
from versionable._hash import computeHash
```

Initialize logger: `logger = logging.getLogger(__name__)`

### Docstrings

- **Simple functions**: One-line docstrings
- **Complex functions**: Google-style with Args/Returns/Raises sections
- **Modules**: Module-level docstring explaining purpose

### Error Handling

- Custom exception hierarchy rooted at `VersionableError` (see `_errors.py`)
- Reraise with context: `raise BackendError(...) from e`

## Architecture

Key design decisions:
- `__init_subclass__` (not metaclass) for `Versionable`
- Hash validated at class definition time (import time)
- Versionable types use their serialization name (not module path) in hashes — stable across file moves
- `save()`/`load()` accessed via qualified import (`import versionable`), not direct import
- `Backend.save()` receives raw values + `cls`; each backend owns its serialization
- HDF5 backend uses native type mapping (no JSON) with `__versionable__` metadata groups
