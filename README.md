# versionable

<p align="center">
  <img src="docs/images/logo-horizontal-tagline.svg" alt="versionable logo"/>
</p>

Save and load Python dataclasses to files — with schema versioning, type converters, and pluggable storage backends.

**[Full documentation →](https://versionable.readthedocs.io)**

## Why versionable?

Serializing is a fancy way to say _turning an object into a file and back again_. With versionable, you define your
data as a Python dataclass. versionable gives you a `save()` and a `load()` function that produce human-readable
files (YAML, JSON, TOML) or binary-efficient ones (HDF5) — each carrying schema metadata so that a file written by v1 of
your code 5 months ago loads cleanly into v5, automatically.

**The problem:** you save parameters to a file, refactor your code, and the old file no longer loads — or worse, loads
silently with wrong data!

_What to use?_ `.pickle` is [unsafe](https://docs.python.org/3/library/pickle.html#restricting-globals). Pure `.json`
and `.yaml` carry no schema and manual wrappers break the moment your schema changes. `.csv` and `.parquet` are great
for tables but poor at handling structured metadata. `.npz` files aren't even guaranteed to be compatible across numpy
versions. `.proto` (Protocol Buffers) are schema-aware but require a build step and offer no migrations. And if you've
resorted to folders with sidecar files (`data.npy` + `params.json` + `metadata.txt`), you already know how easily those
drift out of sync.

**What you get with versionable:**

- **Zero boilerplate** — no schema files, no code generation, no build step. Just inherit from `Versionable`
- **Import-time safety** — schema hash mismatches are caught when your module loads, not in production
- **Declarative migrations** — rename, add, remove, or transform fields across versions
- **Nested composition** — compose complex dataclasses from smaller `Versionable` pieces that version independently
- **Pluggable backends** — YAML, JSON, TOML, HDF5 (with lazy loading for large arrays)
- **Built-in type converters** — datetime, Path, UUID, Enum, numpy arrays, and more
- **Human-readable output** — files you can inspect, diff, and hand-edit with any editor
- **Modern, type-safe Python** — fully typed and compatible with mypy, pyright, and other static analyzers

### How does it compare?

| Feature                      | versionable | pickle | JSON/YAML | protobuf | JSON libs\* | sidecars |
| ---------------------------- | -------------- | ------ | --------- | -------- | ----------- | -------- |
| Zero boilerplate             | ✅             | ✅     | ✅        | ❌       | ✅          | ❌       |
| Import-time validation       | ✅             | ❌     | ❌        | 🔧       | ❌          | ❌       |
| Schema versioning            | ✅             | ❌     | ❌        | 🔧       | ❌          | ❌       |
| Automatic migrations         | ✅             | ❌     | ❌        | ❌       | ❌          | ❌       |
| Nested composition           | ✅             | ✅     | ⚠️        | ✅       | ⚠️          | ❌       |
| Large array support (HDF5)   | ✅             | ❌     | ❌        | ❌       | ❌          | ⚠️       |
| Human-readable files         | ✅             | ❌     | ✅        | ❌       | ✅          | ⚠️       |
| Type-safe                    | ✅             | ❌     | ❌        | ✅       | ⚠️          | ❌       |
| Safe to load untrusted files | ✅             | ❌     | ✅        | ✅       | ✅          | ⚠️       |
| Data + metadata in one file  | ✅             | ✅     | ✅        | ✅       | ✅          | ❌       |

\* dataclasses-json, dacite, pydantic, cattrs, etc. · 🔧 = requires manual effort / build step · ⚠️ = partial

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
[Steve Araiza](https://github.com/saraiza). Over the years the idea evolved from using `CArchive` to make use of C++11,
variadic macros, and other fun modern C++ features. Some version of serializable has been a part of every project I've
worked on since those days.

This is the Python version of the idea. It is built using modern, type-safe Python with great fresh ideas from
[Emma Powers](https://github.com/emmapowers/).

A big thank you to both of them! 🥓🥞🍳

## License

MIT - Copyright ©️ 2026 Hendrick Melo
