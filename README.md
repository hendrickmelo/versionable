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

**`versionable`** fixes that. Like database migrations, but for files. Every file is stamped with a version number and a
fingerprint of its structure. Files written by v1 of your code load cleanly into v5, automatically migrated, never
silently broken. Save to standard formats out of the box: JSON, HDF5, YAML, and TOML.

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

The base install includes the JSON backend with zero heavy dependencies:

```bash
pip install versionable
```

Add backend support as needed (JSON is included by default):

```bash
pip install pyyaml                # YAML backend (.yaml, .yml)
pip install tomlkit               # TOML backend (.toml)
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

### The Schema Hash — Friction as a Feature

The `hash` parameter is optional — everything works without it. But when present, it acts as a tripwire.

Without it, here's what happens: you rename a field, forget to add a migration, and old files load with a missing field
that silently defaults to zero. Your experiment runs with wrong calibration data for a week before anyone notices.

The hash prevents that. It's a fingerprint of your fields and their types, validated at _import time_ — not at runtime,
not in production. Change a field and forget to update the version? Python won't even import:

```python
@dataclass
class SensorConfig(Versionable, version=2, hash="4b7866"):  # ⬅ old hash
    name: str
    magnitude: float  # changed, but hash wasn't updated

# HashMismatchError: SensorConfig: hash mismatch — declared '4b7866',
#   computed 'a70249'. Update the hash parameter to 'a70249'.
```

That error is the point. It means you can't accidentally ship a schema change without a migration. The hash makes
breaking changes visible during development, in CI, at deploy time — never in production. Think of it like a type
checker for your data format: optional, zero runtime cost, catches mistakes before they matter.

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

## Background

The pattern behind **`versionable`** has been used in production C++ systems for over 15 years — from `CArchive`-based
serialization to modern C++11 variadic macros. Some version of this pattern has been a part of every project the authors
have worked on. This is our second Python implementation of a proven approach, built with modern type-safe Python. The
test suite has a ~1:1 ratio of test code to source code, with cross-backend round-trip coverage and edge-case validation
across all four backends.

Have questions? See the **[FAQ](docs/faq.md)**. Want to contribute? See the **[contributing guide](CONTRIBUTING.md)**.

## Acknowledgements

The idea started with [Steve Araiza](https://github.com/saraiza), who first taught me this approach. Over the years it
evolved through many C++ iterations, and every project I've worked on since has used some version of this pattern.
[Emma Powers](https://github.com/emmapowers/) brought great fresh ideas to this Python implementation.

A big thank you to both of them! 🥓🥞🍳

## License

MIT - Copyright ©️ 2026 Hendrick Melo
