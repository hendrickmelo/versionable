# versionable

<p align="center">
  <img src="docs/images/logo-horizontal-tagline.svg" alt="versionable logo"/>
</p>

Future-proof your data files. Save structured Python objects with versioning, declarative migrations, and open file
formats.

**[Full documentation →](https://versionable.readthedocs.io)**

## Why **`versionable`** ?

Your data lives in files. Your code keeps changing. Without versioning, old files silently load with missing fields,
wrong types, or stale values.

**`versionable`** fixes that. Every file is stamped with a version number and a fingerprint of its structure. Files
written by v1 of your code load cleanly into v5, automatically migrated, never silently broken. Save to standard formats
out of the box: JSON, HDF5, YAML, and TOML.

**What you get:**

- **Zero boilerplate** — no schema files, no code generation, no build step. Just inherit from `Versionable`
- **Simple versioning with declarative migrations** — rename, add, remove, or transform fields across versions
- **Rich type support** — datetime, Path, UUID, Enum, numpy arrays, and more — easy to extend with your own
- **Nested objects with independent versioning** — compose complex dataclasses from smaller `Versionable` pieces
- **Incremental HDF5 writes** — append rows as data arrives, no need to hold everything in memory
- **Random access for huge files** — read slices directly from disk without loading the whole file
- **JSON, HDF5, YAML, TOML** — or bring your own backend
- **Import-time safety** — schema hash mismatches are caught when your module loads, not in production
- **Modern, type-safe Python** — fully typed and compatible with mypy, pyright, and other static analyzers

### How does it compare?

| Versionable Features                       | pickle | dc libs¹ | protobuf | raw JSON | sidecars |
| ------------------------------------------ | ------ | -------- | -------- | -------- | -------- |
| ✅ Zero boilerplate                        | ✅     | ✅       | 🛠️       | -        | -        |
| ✅ Versioning with declarative migrations  | 🛠️     | -        | -        | 🛠️       | -        |
| ✅ Rich type support                       | ✅     | ✅       | 🛠️       | 🛠️       | 🟠       |
| ✅ Nested objects, versioned independently | 🟠     | 🛠️       | 🟠       | 🛠️       | -        |
| ✅ Incremental HDF5 writes                 | -      | -        | -        | -        | 🛠️       |
| ✅ Random access for huge files            | -      | -        | -        | -        | 🛠️       |
| ✅ Custom Backends                         | -      | 🟠       | 🟠       | 🟠       | -        |
| ✅ Import-time validation                  | -      | -        | 🛠️       | -        | -        |
| ✅ Modern, type-safe Python                | -      | ✅       | ✅       | -        | -        |

¹ pydantic, dataclasses-json, etc.

- 🛠️ = requires manual effort / build step
- 🟠 = partial

## Installation

The base install includes the JSON backend (no extra dependencies beyond numpy):

```bash
pip install versionable
```

Add backend support as needed (JSON is included by default):

```bash
pip install pyyaml                # YAML backend (.yaml, .yml)
pip install toml                  # TOML backend (.toml)
pip install h5py hdf5plugin       # HDF5 backend (.h5, .hdf5)
```

Or install the latest main from source:

```bash
pip install git+https://github.com/hendrickmelo/versionable.git
```

## Quick Start

### Simple files

You save a config file today:

```python
from dataclasses import dataclass
import versionable
from versionable import Versionable

@dataclass
class SensorConfig(Versionable, version=1, hash="4b7866"):
    name: str
    value: float

config = SensorConfig(name="experiment-A", value=9.81)
versionable.save(config, "config.json")
```

A few weeks later you rename `value` to `magnitude`. Without versionable, old files silently load with missing data.
With it, you bump the version and declare a migration — old files upgrade automatically:

```python
from versionable import Migration

@dataclass
class SensorConfig(Versionable, version=2, hash="a70249"):
    name: str
    magnitude: float  # renamed from "value"

    class Migrate:
        v1 = Migration().rename("value", "magnitude")

# Old v1 file loads and the old field is automatically migrated
loaded = versionable.load(SensorConfig, "config.yaml")
assert loaded.magnitude == 9.81
```

### Catching Version Drift

The `hash` parameter is a fingerprint of your fields and their types. It's optional, but when present it catches version
drift: if you change a field and forget to bump the version, Python raises an error as soon as the module is imported.
Not when a user opens a corrupt file in production. During development, in CI, at deploy time.

For example, if you rename `value` to `magnitude` but keep the old hash:

```python
# This raises at import time — not at runtime, not in production
@dataclass
class SensorConfig(Versionable, version=2, hash="4b7866"):  # ⬅ Old V1 Hash
    name: str
    magnitude: float  # changed, but hash wasn't updated
    ...

# HashMismatchError: SensorConfig: hash mismatch — declared '4b7866',
#   computed 'a70249'. Update the hash parameter to 'a70249'.
```

Update the hash, add a migration, and you're done ✔️. A few weird looking characters now, and your data is future-proof.

### Working with Large Data

For scientific and engineering workflows, fields map to native HDF5 chunked datasets. You can append rows incrementally
and read slices from disk without loading the whole file into memory:

```python
import numpy as np
from numpy.typing import NDArray

@dataclass
class Experiment(Versionable, version=1, hash="536849"):
    name: str
    traces: NDArray[np.float64] = field(default_factory=lambda: np.empty((0, 1024)))

# Append to a chunked, resizable dataset as data arrives
session = versionable.hdf5.open(Experiment, "run.h5")
with session as obj:
    obj.name = "long-running-acquisition"
    for batch in data_source:
        obj.traces.append(batch)    # extends the dataset on disk
        session.flush()             # flush HDF5 buffers to OS

# Read slices directly from disk without loading the whole file
with versionable.hdf5.open(Experiment, "run.h5", mode="read") as obj:
    print(obj.traces[1000])         # reads only row 1000
    print(obj.traces[50:100])       # reads only this slice
```

## Learn More

Want to see how old files get upgraded automatically when your schema changes?

- **[See migrations](docs/migrations.md)** in action
- Explore the available **[backends](docs/backends.md)**

### For AI Agents

If you're an AI agent working with versionable, see **[AGENT.md](docs/AGENT.md)** for a condensed API reference.

### Complete Documentation

For custom type converters, HDF5 support, and more, see the
**[full documentation](https://versionable.readthedocs.io)**.

## Acknowledgements

The idea behind versionable started over 15 years ago in C++, where I first learned the approach from
[Steve Araiza](https://github.com/saraiza). Over the years the idea of a Serializable / Versionable class evolved from
using `CArchive` to make use of C++11, variadic macros, and other fun modern C++ features. Some version of this pattern
has been a part of every project I've worked on since those days.

This is the Python version of the idea. It is built using modern, type-safe Python with great fresh ideas from
[Emma Powers](https://github.com/emmapowers/).

A big thank you to both of them! 🥓🥞🍳

## License

MIT - Copyright ©️ 2026 Hendrick Melo
