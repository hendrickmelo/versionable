# versionable — User Skills Reference

Serialization framework for Python 3.12+ dataclasses with schema versioning, hash validation, declarative migrations,
type converters, and pluggable storage backends.

## Installation

```bash
pip install versionable            # Core (JSON backend, no heavy deps)
pip install pyyaml                 # Add YAML backend
pip install toml                   # Add TOML backend
pip install h5py hdf5plugin        # Add HDF5 backend
```

## Quick Start

```python
from __future__ import annotations

from dataclasses import dataclass

import versionable
from versionable import Versionable


@dataclass
class SensorConfig(Versionable, version=1, hash="<TBD>"):
    sampleRate_Hz: float
    label: str = "default"


# First run: compute the hash
print(SensorConfig.hash())  # e.g. "a3f1c9"
# Paste it into hash="a3f1c9", then:

versionable.save(SensorConfig(sampleRate_Hz=1000.0), "config.json")
loaded = versionable.load(SensorConfig, "config.json")
```

During development, call `ignoreHashErrors(True)` to get warnings instead of errors while you iterate on fields. Compute
and set the final hash before shipping.

## Defining Versionable Classes

```python
@dataclass
class MyClass(
    Versionable,
    version=1,                    # Required — increment when schema changes
    hash="a1b2c3",               # 6-char fingerprint (run .hash() to compute)
    name="MyClass",              # Serialization name (default: class name)
    old_names=["PreviousName"],  # Previous names for backward compat
    skip_defaults=False,         # Omit default-valued fields on save
    unknown="ignore",            # "ignore" | "error" | "preserve"
):
    requiredField: float
    optionalField: str = "hello"
```

**What gets serialized:** Fields with a type annotation and no leading underscore. `ClassVar` fields, private fields
(`_name`), and unannotated attributes are excluded.

**Nested composition:** Versionable objects can contain other Versionable objects. Each nested class versions
independently.

```python
@dataclass
class Inner(Versionable, version=1, hash="..."):
    x: float
    y: float


@dataclass
class Outer(Versionable, version=1, hash="..."):
    name: str
    point: Inner
```

## Saving and Loading

```python
import versionable

# Backend auto-selected by extension
versionable.save(obj, "config.json")
versionable.save(obj, "config.yaml")   # requires pyyaml
versionable.save(obj, "config.toml")   # requires toml
versionable.save(obj, "data.h5")       # requires h5py + hdf5plugin

loaded = versionable.load(MyClass, "config.json")
```

**Load without knowing the type** (class must be registered and imported):

```python
obj = versionable.loadDynamic("config.yaml")
```

### Save Options

| Option            | Backends   | Description                                |
| ----------------- | ---------- | ------------------------------------------ |
| `commentDefaults` | YAML, TOML | Comment out fields matching class defaults |
| `compression`     | HDF5       | Compression config (see HDF5 section)      |

### Load Options

| Option           | Backends | Description                                              |
| ---------------- | -------- | -------------------------------------------------------- |
| `preload`        | HDF5     | `["field"]` or `"*"` — eager-load arrays instead of lazy |
| `metadataOnly`   | HDF5     | Skip arrays entirely (fastest for metadata scanning)     |
| `upgradeInPlace` | All      | Allow migrations that rewrite the file                   |
| `assumeVersion`  | All      | Override the version read from file metadata             |

## Backends

| Backend | Extensions      | None | Large Arrays | Lazy Load | Best For                   |
| ------- | --------------- | ---- | ------------ | --------- | -------------------------- |
| YAML    | `.yaml`, `.yml` | Yes  | Slow         | No        | Config files, data science |
| JSON    | `.json`         | Yes  | Slow         | No        | Interoperability           |
| TOML    | `.toml`         | No   | Slow         | No        | Hand-editable configs      |
| HDF5    | `.h5`, `.hdf5`  | Yes  | Fast/Native  | Yes       | Large numpy arrays         |

**TOML caveat:** TOML has no `null` type. Fields holding `None` are omitted on save and restored from the class default
on load. Every TOML field should have a default value.

### HDF5 Details

Every field maps to a native HDF5 construct — no JSON in the file. Scalars become attributes, arrays become datasets,
nested Versionables become subgroups with a `__versionable__` metadata group, and `list[np.ndarray]` /
`dict[str, np.ndarray]` become groups of datasets.

Arrays and array collections are lazy-loaded by default — `load()` returns instantly even for multi-gigabyte files.
Accessing an array field or indexing into a `list[np.ndarray]` triggers the disk read.

```python
import versionable
from versionable.hdf5 import GZIP_DEFAULT, ZSTD_DEFAULT

# Save with compression (gzip is the default)
versionable.save(obj, "data.h5", compression=GZIP_DEFAULT)

# Load with selective preloading
loaded = versionable.load(MyClass, "data.h5", preload=["largeArray"])

# Metadata-only (arrays raise ArrayNotLoadedError on access)
loaded = versionable.load(MyClass, "data.h5", metadataOnly=True)
```

**Compression presets** (from `versionable.hdf5`):

| Preset          | Notes                                    |
| --------------- | ---------------------------------------- |
| `ZSTD_DEFAULT`  | zstd level 3 — fast, good ratio          |
| `ZSTD_FAST`     | zstd level 1 — fastest                   |
| `ZSTD_BEST`     | zstd level 9 — best ratio, slow          |
| `BLOSC_DEFAULT` | Blosc + zstd — fast for large arrays     |
| `GZIP_DEFAULT`  | gzip level 4 — default, universal compat |
| `LZF`           | LZF — fastest, no extra deps             |
| `UNCOMPRESSED`  | No compression                           |

gzip (default) and lzf work everywhere. zstd and blosc require `hdf5plugin` — use them if compatibility with other tools
is not a major concern.

### HDF5 Sessions — Incremental Writes and Random Access

For large or long-running data, `versionable.hdf5.open()` provides incremental writes to chunked, resizable datasets and
random access reads without loading the whole file into memory.

```python
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

import versionable
import versionable.hdf5
from versionable import Versionable


@dataclass
class Experiment(Versionable, version=1, hash="536849"):
    name: str
    traces: NDArray[np.float64] = field(default_factory=lambda: np.empty((0, 1024)))

# Write incrementally — each append extends the dataset on disk
session = versionable.hdf5.open(Experiment, "run.h5")
with session as obj:
    obj.name = "acquisition-001"
    for batch in data_source:
        obj.traces.append(batch)
        session.flush()             # flush HDF5 buffers to OS

# Resume an existing file
session = versionable.hdf5.open(Experiment, "run.h5", mode="resume")
with session as obj:
    obj.traces.append(more_data)

# Random access — read slices directly from disk
with versionable.hdf5.open(Experiment, "run.h5", mode="read") as obj:
    print(obj.traces[1000])         # reads only row 1000
    print(obj.traces[50:100])       # reads only this slice
```

**Session modes:**

| Mode       | Description                                      |
| ---------- | ------------------------------------------------ |
| `"create"` | New file (default). Fails if file exists         |
| `"resume"` | Append to existing file. Version/hash must match |
| `"read"`   | Read-only access. No writes allowed              |

**Field types in sessions:**

| Type                  | Behavior                                                 |
| --------------------- | -------------------------------------------------------- |
| Scalars               | Assignment writes through to disk                        |
| `NDArray` / `ndarray` | `DatasetArray` with `append()`, `resize()`, slice access |
| `list[np.ndarray]`    | `TrackedList` — `append()`/`extend()` write through      |
| `dict[str, ndarray]`  | `TrackedDict` — `__setitem__`/`update()` write through   |

Sessions do not support migrations. The file's version and hash must exactly match the class. `DatasetArray` fields
raise `BackendError` after the session is closed — copy data before closing if needed.

**Compression on resume:** Appending to an existing dataset uses the original dataset's compression filter, not the
session's `compression` parameter. The session compression only applies to newly created datasets.

## Supported Types

### Built-in (no registration needed)

**Primitives:** `int`, `float`, `str`, `bool`, `None`

**Collections:** `list[T]`, `dict[K, V]`, `set[T]`, `frozenset[T]`, `tuple[T, ...]`, `Optional[T]`, `Union[A, B]`,
`Literal[...]`

**Stdlib types (auto-converted):**

| Type                 | Serialized As         |
| -------------------- | --------------------- |
| `datetime.datetime`  | ISO 8601 string       |
| `datetime.date`      | ISO 8601 string       |
| `datetime.time`      | ISO 8601 string       |
| `datetime.timedelta` | Float (total seconds) |
| `pathlib.Path`       | String                |
| `uuid.UUID`          | String                |
| `decimal.Decimal`    | String                |
| `bytes`              | Base64 string         |
| `complex`            | `[real, imag]`        |
| `re.Pattern`         | Pattern string        |

**numpy arrays:** Native HDF5 datasets (compressed, lazy-loaded). Base64-compressed npz blobs in JSON/TOML/YAML.

### Enums

Serialized by `.value`. Set a fallback for graceful handling of removed enum members:

```python
from enum import Enum


class Status(Enum):
    ACTIVE = "active"
    ARCHIVED = "archived"
    UNKNOWN = "unknown"


Status.VERSIONABLE_FALLBACK = Status.UNKNOWN  # Old values deserialize to UNKNOWN
```

### Literal Fields

Use `literalFallback` for graceful handling of invalid literal values from old files:

```python
from versionable import literalFallback


@dataclass
class Config(Versionable, version=1, hash="..."):
    mode: Literal["fast", "balanced", "slow"] = literalFallback("balanced")
```

### Custom Types

**Option 1 — `registerConverter`** (for third-party types or complex serialization):

```python
from versionable import registerConverter


registerConverter(
    Coord,
    serialize=lambda v: {"lat": v.lat, "lon": v.lon},
    deserialize=lambda v, _cls: Coord(v["lat"], v["lon"]),
)
```

**Option 2 — `VersionableValue` protocol** (for your own types mapping to a single primitive):

```python
from versionable import VersionableValue


class UserId(VersionableValue):
    def __init__(self, value: str) -> None:
        self.value = value

    def toValue(self) -> str:
        return self.value

    @classmethod
    def fromValue(cls, value: str) -> UserId:
        return cls(value)
```

## Migrations

When you change a class's fields, increment `version`, update `hash`, and add a migration so old files load correctly.

### Declarative Migrations

```python
@dataclass
class Config(Versionable, version=3, hash="x1y2z3"):
    name: str
    timeout_s: float = 30.0
    retries: int = 3

    class Migrate:
        # v1 → v2: renamed "title" to "name"
        v1 = Migration().rename("title", "name")

        # v2 → v3: added "retries" with default for old files
        v2 = Migration().add("retries", default=1)
```

**Available operations (chainable):**

| Operation                                 | Description                          |
| ----------------------------------------- | ------------------------------------ |
| `.rename(old, new)`                       | Rename a field                       |
| `.drop(field)`                            | Remove a field from old data         |
| `.add(field, default=value)`              | Add field with default for old files |
| `.convert(field, via=fn)`                 | Transform a field's value            |
| `.derive(target, from_=source, via=fn)`   | Create new field from existing       |
| `.split(field, into={...})`               | Split one field into multiple        |
| `.merge(fields=[...], into=name, via=fn)` | Merge multiple fields into one       |
| `.requiresUpgrade()`                      | Mark as needing in-place rewrite     |
| `.then(other_migration)`                  | Chain another migration              |

Chain multiple operations: `Migration().rename("a", "b").drop("c").add("d", default=0)`

### Imperative Migrations

For branching logic or complex transformations:

```python
from versionable import MigrationContext, migration


class Migrate:
    @migration(fromVersion=2)
    def from_v2(ctx: MigrationContext) -> None:
        raw = ctx.pop("rawData")
        ctx["timestamps"] = [row[0] for row in raw]
        ctx["values"] = [row[1] for row in raw]
```

Migrations apply sequentially: a v1 file on a v5 class runs v1 → v2 → v3 → v4 → v5.

### Renaming a Class

Use `old_names` to load files saved under a previous class name:

```python
@dataclass
class SensorConfig(Versionable, version=2, hash="...", old_names=["SensorSettings"]):
    ...
```

## Introspection

```python
import versionable
from versionable import metadata, getVersionableFields, registeredClasses

# Schema metadata
meta = metadata(SensorConfig)
meta.version       # int
meta.hash          # str (6 chars)
meta.name          # str
meta.fields        # list[str]

# Field types
fields = getVersionableFields(SensorConfig)  # dict[str, type]

# Compute hash (paste into hash= parameter)
SensorConfig.hash()  # str

# All registered classes
registeredClasses()  # dict[name, type]
```

## Error Handling

```text
VersionableError (base — catch-all)
├── HashMismatchError      — hash= doesn't match fields (raised at import time)
├── VersionError           — file is newer than class, or missing migrations
├── MigrationError         — migration failed to apply
├── ArrayNotLoadedError    — accessing array loaded with metadataOnly=True
├── UpgradeRequiredError   — migration needs upgradeInPlace=True
├── UnknownFieldError      — file has field not in class (only with unknown="error")
├── ConverterError         — type conversion failed
└── BackendError           — file I/O or backend operation failed
```

All exceptions are importable from `versionable`:

```python
from versionable import VersionableError, HashMismatchError, BackendError
```

## Common Patterns

### Configuration file with commented defaults

```python
versionable.save(config, "defaults.yaml", commentDefaults=True)
```

Produces YAML/TOML where fields at their default value are commented out, making it easy to see what was customized.

### Scanning HDF5 metadata without loading arrays

```python
for path in Path("data/").glob("*.h5"):
    obj = versionable.load(Experiment, path, metadataOnly=True)
    print(f"{path}: {obj.name}, {obj.timestamp}")
```

### Dynamic loading with type dispatch

```python
obj = versionable.loadDynamic("unknown_file.yaml")
match type(obj).__name__:
    case "SensorConfig":
        processSensor(obj)
    case "ExperimentResult":
        processResult(obj)
```

### Registering existing backends for custom extensions

Use `registerBackend` to map new file extensions to a built-in backend class:

```python
from versionable import JsonBackend, registerBackend

registerBackend([".jsonc", ".json5"], JsonBackend)
```

All four backend classes are importable from `versionable`: `JsonBackend`, `TomlBackend`, `YamlBackend`, `Hdf5Backend`.

### Writing a custom backend

```python
from versionable import Backend, registerBackend


class MsgPackBackend(Backend):
    nativeTypes: set[type] = set()

    def save(self, fields: dict, meta: dict, path, *, cls: type, **kwargs) -> None: ...
    def load(self, path) -> tuple[dict, dict]: ...


registerBackend([".msgpack"], MsgPackBackend)
```

The `save()` method receives raw (unserialized) field values and the Versionable class. Call `serialize()` internally
for dict-based formats, or handle type dispatch directly for binary formats.
