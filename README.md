# versionable

<p align="center">
  <img src="docs/images/logo-horizontal-tagline.svg" alt="versionable logo"/>
</p>

Save and load Python dataclasses to files — with schema versioning, type converters, and pluggable storage backends.

**[Full documentation →](https://versionable.readthedocs.io)**

## Why versionable?

Your data lives in files. Your code keeps changing. Without protection, old files silently load with missing fields,
wrong types, or stale values — your data schemes against you.

versionable stops the scheming. Define your data as a Python dataclass, and get `save()` and `load()` functions that
produce human-readable files (YAML, JSON, TOML) or binary-efficient ones (HDF5). Every file is stamped with a schema
fingerprint and version number, so a file written by v1 of your code loads cleanly into v5 — automatically migrated,
never silently broken.

_What to use?_ `.pickle` is [unsafe](https://docs.python.org/3/library/pickle.html#restricting-globals). Pure `.json`
and `.yaml` carry no schema and manual wrappers break the moment your schema changes. `.csv` and `.parquet` are great
for tables but poor at handling structured metadata. `.npz` files aren't even guaranteed to be compatible across numpy
versions. `.proto` (Protocol Buffers) are schema-aware but require a build step and offer no migrations. And if you've
resorted to folders with sidecar files (`data.npy` + `params.json` + `metadata.txt`), you already know how easily those
drift out of sync.

**What you get with versionable:**

- **Zero boilerplate** — no schema files, no code generation, no build step. Just inherit from `Versionable`
- **Simple versioning with declarative migrations** — rename, add, remove, or transform fields across versions
- **Rich type support** — datetime, Path, UUID, Enum, numpy arrays, and more — easy to extend with your own
- **Nested objects with independent versioning** — compose complex dataclasses from smaller `Versionable` pieces
- **Native numpy array support** — with lazy HDF5 loading for large datasets
- **JSON, YAML, TOML, HDF5** — or bring your own backend
- **Import-time safety** — schema hash mismatches are caught when your module loads, not in production
- **Modern, type-safe Python** — fully typed and compatible with mypy, pyright, and other static analyzers

### How does it compare?

| Versionable Features                       | pickle | dc libs¹ | protobuf | raw JSON | sidecars |
| ------------------------------------------ | ------ | -------- | -------- | -------- | -------- |
| ✅ Zero boilerplate                        | ✅     | ✅       | ❌       | ❌       | ❌       |
| ✅ Versioning with declarative migrations  | ❌     | ❌       | ❌       | ❌       | ❌       |
| ✅ Rich type support                       | ✅     | ✅       | 🔧       | ❌       | ❌       |
| ✅ Nested objects, versioned independently | 🟠     | 🟠       | 🟠       | 🟠       | ❌       |
| ✅ Native numpy / lazy HDF5                | 🟠     | ❌       | ❌       | ❌       | 🟠       |
| ✅ Custom Backends                         | ❌     | 🟠       | 🟠       | 🟠       | ❌       |
| ✅ Import-time validation                  | ❌     | ❌       | 🔧       | ❌       | ❌       |
| ✅ Modern, type-safe Python                | ❌     | 🟠       | ✅       | ❌       | ❌       |

¹ pydantic, dataclasses-json, etc. | 🔧 = requires manual effort / build step | 🟠 = partial

## Installation

```bash
pip install git+https://github.com/hendrickmelo/versionable.git

# With HDF5 backend support (h5py + hdf5plugin)
pip install "versionable[hdf5] @ git+https://github.com/hendrickmelo/versionable.git"
```

## Quick Start

You save a config file today:

```python
from dataclasses import dataclass
import versionable
from versionable import Versionable

@dataclass
class SensorConfig(Versionable, version=1, hash="82bc30"):
    name: str
    value: float

config = SensorConfig(name="experiment-A", value=9.81)
versionable.save(config, "config.yaml")
```

A few weeks later you rename `value` to `magnitude`. Without versionable, old files silently load with missing data.
With it, you bump the version and declare a migration — old files upgrade automatically:

```python
from versionable import Versionable, Migration

@dataclass
class SensorConfig(Versionable, version=2, hash="064498"):
    name: str
    magnitude: float  # renamed from "value"

    class Migrate:
        v1 = Migration().rename("value", "magnitude")

# Old file loads cleanly into the new schema
loaded = versionable.load(SensorConfig, "config.yaml")
assert loaded.magnitude == 9.81
```

The schema hash is validated at import time — if your fields change but the hash doesn't, you get an error immediately,
not a silent bug in production.

## Learn More

Want to see how old files get upgraded automatically when your schema changes?

- **[See migrations](docs/migrations.md)** in action
- Explore the available **[backends](docs/backends.md)**

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
