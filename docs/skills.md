# versionable — Agent Skills Reference

Serialization framework for Python 3.12+ dataclasses with schema versioning, hash validation, declarative migrations,
type converters, and pluggable storage backends (JSON, TOML, YAML, HDF5).

## Project Layout

```text
src/versionable/
├── __init__.py              # Public API re-exports (__all__)
├── _base.py                 # Versionable base class, registry, metadata
├── _api.py                  # save(), load(), loadDynamic()
├── _types.py                # Type converter registry, serialize/deserialize dispatch
├── _hash.py                 # Deterministic schema hash (SHA-256, first 6 hex chars)
├── _backend.py              # Backend ABC, extension-based auto-detection
├── _json_backend.py         # JSON backend (.json)
├── _toml_backend.py         # TOML backend (.toml)
├── _yaml_backend.py         # YAML backend (.yaml, .yml)
├── _hdf5_backend.py         # HDF5 backend (.h5, .hdf5) — optional dependency
├── _hdf5_compression.py     # Compression presets (zstd, gzip, blosc, lzf)
├── hdf5.py                  # HDF5 submodule re-exports (Hdf5Compression + presets)
├── _lazy.py                 # Lazy HDF5 array loading via dynamic subclass
├── _migration.py            # Declarative + imperative migration system
├── errors.py                # Exception hierarchy (public module)
└── py.typed                 # PEP 561 marker
tests/
├── conftest.py              # Shared fixtures and sample dataclasses
├── test_base.py             # Versionable class, metadata, registry
├── test_hash.py             # Hash computation
├── test_types.py            # Type converters
├── test_json_backend.py     # JSON round-trips
├── test_toml_backend.py     # TOML round-trips
├── test_yaml_backend.py     # YAML round-trips
├── test_hdf5_backend.py     # HDF5 round-trips, compression, lazy loading
└── test_migration.py        # Migration system
```

Private modules (prefixed `_`) are internal — the public API is exposed only through `__init__.py` and
`versionable.hdf5`.

## Build System

**Pixi** for environment management. **Hatchling** for pip-installable packaging.

```bash
pixi install                    # Install all dependencies
pixi run cleanup                # Format + lint + type-check (MUST pass before committing)
pixi run pytest                 # Run full test suite
pixi run pytest -x              # Stop on first failure
pixi run ci                     # Full CI pipeline (check-only mode)
```

**Environments:**

- `default` — full dev + HDF5 (`h5py`, `hdf5plugin`)
- `minimal` — dev only (no HDF5); CI tests both
- `docs` — Sphinx documentation build

**Quality tools:** ruff (lint + format), mypy, pyright, prettier (markdown), markdownlint-cli2. All run via
`pixi run cleanup`.

## Coding Conventions

| Element                | Convention          | Example                                         |
| ---------------------- | ------------------- | ----------------------------------------------- |
| Functions/methods/vars | `camelCase`         | `computeHash()`, `fieldType`, `nativeTypes`     |
| Classes/types          | `PascalCase`        | `Versionable`, `Migration`, `JsonBackend`      |
| Constants              | `SCREAMING_SNAKE`   | `_CANONICAL_NAMES`, `ZSTD_DEFAULT`              |
| Private members        | Leading `_`         | `_registry`, `_ops`, `_resolveFields()`         |
| Private modules        | Leading `_`         | `_api.py`, `_types.py`                          |
| Test functions         | `snake_case`        | `test_roundtrip()`, `test_changesOnFieldAdd()`  |
| Units in var names     | Include unit suffix | `frequency_Hz`, `timeout_s`                     |

- Use modern type hints: `X | Y`, `list[T]`, `dict[K, V]`. Avoid `Any` without justification.
- Import order: `from __future__ import annotations`, stdlib, third-party, local.
- Logger: `logger = logging.getLogger(__name__)` per module.

## Public API

### Core Functions

```python
import versionable

# Save/load with auto-detected backend
versionable.save(obj, "config.yaml")
versionable.save(obj, "config.yaml", commentDefaults=True)      # Comment out defaults
versionable.save(obj, "data.h5", compression=ZSTD_DEFAULT)      # HDF5 with compression

loaded = versionable.load(MyClass, "config.yaml")
loaded = versionable.load(MyClass, "data.h5", preload=["big"])   # Eager-load specific arrays
loaded = versionable.load(MyClass, "data.h5", preload="*")       # Eager-load all arrays
loaded = versionable.load(MyClass, "data.h5", metadataOnly=True) # Skip arrays entirely

obj = versionable.loadDynamic("file.yaml")                      # Type from __OBJECT__ metadata

# Introspection
meta = versionable.metadata(MyClass)  # VersionableMetadata(version, hash, name, fields, ...)
fields = versionable.getVersionableFields(MyClass)  # dict[str, type]
classes = versionable.registeredClasses()             # dict[name, type]
computed = MyClass.hash()                                # 6-char hex hash

# Dev mode
versionable.ignoreHashErrors(True)  # Warnings instead of errors on hash mismatch
```

### Defining a Versionable Class

```python
from __future__ import annotations

from dataclasses import dataclass

from versionable import Versionable


@dataclass
class SensorConfig(
    Versionable,
    version=1,
    hash="a1b2c3",           # 6-char schema fingerprint (run .hash() to compute)
    name="SensorConfig",     # Serialization name (default: class name)
    old_names=["OldName"],   # Previous names for backward compat
    register=True,           # Add to global registry (for loadDynamic)
    skip_defaults=False,     # Omit default-valued fields on save
    unknown="ignore",        # "ignore" | "error" | "preserve" for unrecognized fields
):
    sampleRate_Hz: float
    label: str = "default"
```

**Serialized fields:** annotated, non-private (no `_` prefix), non-`ClassVar`. Private fields, unannotated
attributes, and `ClassVar` are excluded.

**Reserved metadata keys (cannot be field names):** `__OBJECT__`, `__VERSION__`, `__HASH__`, `__meta__`, `__ndarray__`,
`__json__`.

### Type System

**Natively supported (no registration):** `int`, `float`, `str`, `bool`, `None`, `list[T]`, `dict[K, V]`, `set[T]`,
`frozenset[T]`, `tuple[T, ...]`, `Optional[T]`, `Union[A, B]`, `Literal[...]`.

**Auto-converted stdlib types:** `datetime`, `date`, `time` (ISO 8601), `timedelta` (seconds float), `Path` (string),
`UUID` (string), `Decimal` (string), `bytes` (base64), `complex` ([real, imag]), `re.Pattern` (string).

**Enums:** Serialized by `.value`. Set `VERSIONABLE_FALLBACK` class attribute for graceful unknown-value handling.

**numpy arrays:** HDF5 stores natively as compressed datasets with lazy loading. JSON/TOML/YAML use base64-compressed
npz blobs.

**Custom types — two approaches:**

```python
# Option 1: registerConverter (for third-party or complex types)
from versionable import registerConverter

registerConverter(
    Coord,
    serialize=lambda v: {"lat": v.lat, "lon": v.lon},
    deserialize=lambda v, _cls: Coord(v["lat"], v["lon"]),
)

# Option 2: VersionableValue protocol (for own types mapping to a single primitive)
class UserId:
    def toValue(self) -> str: ...

    @classmethod
    def fromValue(cls, value: str) -> UserId: ...
```

### Migration System

Migrations upgrade old file schemas to the current version. Declared as a `Migrate` inner class.

```python
@dataclass
class Config(Versionable, version=3, hash="x1y2z3"):
    name: str
    timeout_s: float = 30.0

    class Migrate:
        # v1 → v2: rename field
        v1 = Migration().rename("title", "name")

        # v2 → v3: add field with explicit default for old files
        v2 = Migration().add("timeout_s", default=10.0)
```

**Declarative operations (chainable):**

| Operation                              | Description                          |
| -------------------------------------- | ------------------------------------ |
| `.rename(old, new)`                    | Rename a field                       |
| `.drop(field)`                         | Remove a field from old data         |
| `.add(field, default=value)`           | Add field with default for old files |
| `.convert(field, via=fn)`              | Transform a field's value            |
| `.derive(target, from_=source, via=fn)`| Create new field from existing       |
| `.split(field, into={...})`            | Split one field into multiple        |
| `.merge(fields=[...], into=name, via=fn)` | Merge multiple fields into one   |
| `.requiresUpgrade()`                   | Mark as needing in-place rewrite     |
| `.then(other_migration)`               | Chain another migration              |

**Imperative migrations** for complex logic:

```python
from versionable import MigrationContext, migration

class Migrate:
    @migration(fromVersion=2)
    def from_v2(ctx: MigrationContext) -> None:
        ctx["computed"] = ctx.pop("source") * 2
```

Migrations apply sequentially: file at v1 on a v5 class runs v1 → v2 → v3 → v4 → v5.

### Backends

| Backend | Extensions       | None Support | Large Arrays | Lazy Load | Best For                    |
| ------- | ---------------- | ------------ | ------------ | --------- | --------------------------- |
| YAML    | `.yaml`, `.yml`  | Yes          | Slow         | No        | Config files, data science  |
| JSON    | `.json`          | Yes          | Slow         | No        | Interoperability            |
| TOML    | `.toml`          | No           | Slow         | No        | Hand-editable configs       |
| HDF5    | `.h5`, `.hdf5`   | Yes          | Fast/Native  | Yes       | Large numpy arrays          |

Backend is auto-selected by file extension. Custom backends: subclass `Backend`, implement `save()`/`load()`, call
`registerBackend([".ext"], MyBackend)`.

**TOML caveat:** No native `null` — `None` fields are omitted on save, restored from defaults on load. Every TOML field
should have a default.

**HDF5 compression presets** (from `versionable.hdf5`): `ZSTD_DEFAULT`, `ZSTD_FAST`, `ZSTD_BEST`, `BLOSC_DEFAULT`,
`GZIP_DEFAULT`, `LZF`, `UNCOMPRESSED`. zstd/blosc require `hdf5plugin`; gzip/lzf are universal.

### Error Hierarchy

```text
VersionableError (base)
├── HashMismatchError      — declared hash != computed (raised at class definition)
├── VersionError           — file version > class version or missing migrations
├── MigrationError         — migration application failure
├── ArrayNotLoadedError    — accessing array loaded with metadataOnly=True (also AttributeError)
├── UpgradeRequiredError   — migration needs in-place file modification
├── UnknownFieldError      — unknown field in source data (only with unknown="error")
├── ConverterError         — type conversion failure
└── BackendError           — storage backend operation failure
```

### Schema Hash

Computed from field names and canonical type names (sorted, SHA-256, first 6 hex chars). Validated at class definition
time (import time). If declared hash doesn't match, `HashMismatchError` is raised immediately.

Key behavior: `Versionable` subclasses use their serialization `name` (not module path) in hashes — stable across file
moves.

To compute: call `MyClass.hash()` and paste the result into `hash="..."`.

## Testing Patterns

Tests use shared fixtures from `tests/conftest.py` (`SimpleConfig`, `WithOptional`, `WithArray`, `WithEnum`,
`WithNested`, `WithDatetime`, `WithList`, etc.). All test fixtures use `register=False` to avoid polluting the global
registry.

Standard pattern: construct object → save to `tmp_path` → load back → assert field equality. For numpy arrays use
`np.testing.assert_array_equal()`.

```bash
pixi run pytest                          # Full suite
pixi run pytest -x                       # Stop on first failure
pixi run pytest tests/test_json_backend.py  # Single file
pixi run -- pytest -k "test_name"        # Single test by name
```

## Key Design Decisions

- `__init_subclass__` (not metaclass) for `Versionable` registration
- Hash validated at import time — catch schema drift before runtime
- Versionable types use their serialization `name` in hashes — stable across file moves
- Lazy HDF5 loading via dynamically created subclass with `__getattribute__` override
- `save()`/`load()` accessed via qualified import (`import versionable`), not direct import
- numpy is a base dependency; HDF5 support (`h5py`, `hdf5plugin`) is optional via `[hdf5]` extra
