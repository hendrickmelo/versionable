# Backends

versionable provides the option of several backends, each targeting a different trade-off between human-readability,
interoperability with other tools, and performance with large binary data.

You never have to instantiate a backend directly — versionable picks the right one automatically based on the file
extension you pass to `save()` / `load()`. If you use `.json`, you get JSON. If you use `.toml`, you get TOML. The same
object can be saved and loaded with different backends just by changing the filename extension, which makes it easy to
migrate between formats or write tests against a lighter-weight backend than you use in production.

| Extension       | Backend       | Best for                             |
| --------------- | ------------- | ------------------------------------ |
| `.yaml`, `.yml` | `YamlBackend` | Config files, data-science workflows |
| `.json`         | `JsonBackend` | Simple data, interoperability        |
| `.toml`         | `TomlBackend` | Human-editable config files          |
| `.h5`, `.hdf5`  | `Hdf5Backend` | Large numpy arrays, lazy loading     |

All backends store the same schema metadata (`__OBJECT__`, `__VERSION__`, `__HASH__`) alongside your data, so `load()`
can validate the schema and apply migrations regardless of which backend wrote the file.

## Feature comparison

| Feature               | YAML | JSON | TOML | HDF5  |
| --------------------- | ---- | ---- | ---- | ----- |
| Human-readable        | Yes  | Yes  | Yes  | No    |
| `None` / `null`       | Yes  | Yes  | No   | Yes   |
| Comment-out defaults  | Yes  | No   | Yes  | No    |
| Nested objects        | Yes  | Yes  | Yes  | Yes   |
| Large numpy arrays    | Slow | Slow | Slow | Fast  |
| Lazy loading          | No   | No   | No   | Yes   |
| Hand-editable         | Good | Fair | Best | No    |
| External tool support | Wide | Wide | Good | Niche |

## YAML

YAML is a good choice when you want human-readable files with support for comments (added by hand), `null` values, and a
syntax that is already familiar in data-science and DevOps workflows. Unlike TOML, YAML handles `None` natively — fields
with `None` survive the round-trip without any special treatment.

```python
versionable.save(config, "config.yaml")
loaded = versionable.load(SensorConfig, "config.yaml")
```

Produces:

```yaml
name: probe-A
sampleRate_Hz: 120000
channels:
  - 0
  - 1
  - 2
__meta__:
  __OBJECT__: SensorConfig
  __VERSION__: 1
  __HASH__: 9d6951
```

Both `.yaml` and `.yml` extensions are supported.

Metadata is stored in a `__meta__` mapping at the end of the file — your data comes first, schema metadata stays out of
the way.

### Missing fields

Any field absent from the file is filled in from the dataclass default on load. This means older files with fewer fields
load cleanly as new fields are added to the schema (as long as those fields have defaults).

### Comment-Out Defaults

Pass `commentDefaults=True` when saving to comment out fields whose values match the class default. This is useful for
config files where you want users to see all available options without all of them being "active":

```python
versionable.save(config, "config.yaml", commentDefaults=True)
```

```yaml
name: probe-A
sampleRate_Hz: 120000
# channels:
# - 0
# - 1
# - 2
__meta__:
  __OBJECT__: SensorConfig
  __VERSION__: 1
  __HASH__: 9d6951
```

## JSON

JSON is the most common choice when the file will be read by tools outside of Python — a web service, a JavaScript
front-end, or a data pipeline that expects a standard format. It handles all primitive types, lists, and nested objects,
and the output is human-readable even if not particularly easy to hand-edit.

```python
versionable.save(config, "config.json")
loaded = versionable.load(SensorConfig, "config.json")
```

The output includes schema metadata alongside the data:

```json
{
  "__OBJECT__": "SensorConfig",
  "__VERSION__": 1,
  "__HASH__": "9d6951",
  "name": "probe-A",
  "sampleRate_Hz": 120000,
  "channels": [0, 1, 2]
}
```

## TOML

TOML is the best choice for configuration files that users will open and edit by hand. The format is designed to be
obvious at a glance, supports comments (via `commentDefaults`), and maps cleanly to nested sections. If your dataclass
represents application settings that ship with the software and users are expected to tweak, prefer TOML over JSON.

```python
versionable.save(config, "config.toml")
loaded = versionable.load(SensorConfig, "config.toml")
```

Produces human-readable TOML:

```toml
name = "probe-A"
sampleRate_Hz = 120000
channels = [0, 1, 2]

[__meta__]
__OBJECT__ = "SensorConfig"
__VERSION__ = 1
__HASH__ = "9d6951"
```

Fields come first deliberately — if a user opens the file to hand-edit a value, the data is right at the top and the
schema metadata stays out of the way at the bottom.

### Missing fields and None values

TOML is flexible about missing keys — any field absent from the file is silently filled in from the dataclass default on
load. This means you can hand-edit a config file and freely delete any line whose value you want to reset to default,
and it will just work. It also means older files with fewer fields load cleanly as new fields are added to the schema
(as long as those fields have defaults).

**We recommend that every field in a class saved to TOML defines a default value.** Required fields (no default) work
fine for new files, but they become a liability the moment a file is hand-edited, partially written, or migrated from an
older schema version — any of which can leave the field absent, causing load to fail.

The one case to be careful about is `None`. TOML has no native `null` type, so a field holding `None` is omitted on save
— the same as a missing key. On load it is restored from the dataclass default, which is fine if a default exists. But
for a required field (no default), `None` at save time means the field disappears from the file and cannot be recovered
on load. JSON and YAML handle this safely because `null` is a first-class value that survives the round-trip. If your
schema has required optional fields that may genuinely be `None`, prefer YAML or JSON.

Nested `Versionable` objects become native TOML tables. For example, given:

```python
@dataclass
class RetryPolicy(Versionable, version=1, hash="f907a9"):
    retries: int = 3
    backoff_s: float = 1.0

@dataclass
class WorkerConfig(Versionable, version=1, hash="8bdfa7"):
    name: str = "worker"
    retry: RetryPolicy = field(default_factory=RetryPolicy)
```

The saved TOML looks like:

```toml
name = "worker"

[__meta__]
__OBJECT__ = "WorkerConfig"
__VERSION__ = 1
__HASH__ = "8bdfa7"

[retry]
__OBJECT__ = "RetryPolicy"
__VERSION__ = 1
__HASH__ = "f907a9"
retries = 3
backoff_s = 1.0
```

### Comment-Out Defaults

Pass `commentDefaults=True` to comment out fields whose values match the class default. This is useful for config files
where you want users to see all available options without all of them being "active":

```python
versionable.save(config, "config.toml", commentDefaults=True)
```

```toml
name = "probe-A"
sampleRate_Hz = 120000
# channels = [0, 1, 2]

[__meta__]
__OBJECT__ = "SensorConfig"
__VERSION__ = 1
__HASH__ = "9d6951"
```

## HDF5

HDF5 is the right choice when your dataclasses contain large numpy arrays — recordings, images, simulation outputs, or
any dataset where reading the whole file into memory upfront would be slow or wasteful. Unlike JSON and TOML, HDF5
stores arrays as binary compressed datasets, so a 100 MB array saves and loads in a fraction of the time it would take
as text.

The HDF5 backend depends on `h5py`, which in turn requires the HDF5 C library — a non-trivial native dependency that
adds significant installation overhead. It is therefore kept as an optional extra so that projects using only JSON or
TOML don't pay that cost.

**Installation:**

On most platforms (macOS, Windows, Linux x86_64), pip ships a pre-built wheel:

```bash
pip install "versionable[hdf5] @ git+https://github.com/hendrickmelo/versionable.git"
```

On Linux ARM or systems with an older glibc (e.g. RHEL 7), no pre-built wheel is available and pip will fall back to
building from source. Install the HDF5 system library first:

```bash
sudo apt install libhdf5-dev   # Debian/Ubuntu
sudo yum install hdf5-devel    # RHEL/CentOS
```

Then run the pip install above. Users on conda-based environments can skip this — conda manages the HDF5 C library as a
first-class package.

```python
import numpy as np
import numpy.typing as npt
from dataclasses import dataclass
import versionable
from versionable import Versionable

@dataclass
class Recording(Versionable, version=1, hash="..."):
    name: str
    sampleRate_Hz: int
    data: npt.NDArray[np.float64]

rec = Recording(name="capture-1", sampleRate_Hz=240000, data=np.random.rand(1_000_000))
versionable.save(rec, "recording.h5")
```

Arrays are stored as compressed HDF5 datasets. Scalar fields (`str`, `int`, `float`, etc.) are stored as dataset
attributes and are always loaded immediately.

### Compression

By default, array datasets are compressed with **zstd (level 3)** — a good balance of speed and compression ratio. You
can change the algorithm and level per-save by passing a `compression` kwarg:

```python
from versionable.hdf5 import Hdf5Compression, BLOSC_DEFAULT, GZIP_DEFAULT, UNCOMPRESSED

# Use a preset
versionable.save(rec, "recording.h5", compression=GZIP_DEFAULT)

# Or build a custom configuration
comp = Hdf5Compression(algorithm="zstd", level=9)
versionable.save(rec, "recording.h5", compression=comp)
```

Compression is a storage concern — it does not affect the schema hash and has no impact on `load()`. Any compressed file
can be read back regardless of what compression was used to write it, as long as the HDF5 extra is installed.

#### Available presets

| Preset          | Speed | Size | When to use                                                                    |
| --------------- | ----- | ---- | ------------------------------------------------------------------------------ |
| `ZSTD_DEFAULT`  | 🚀    | 🗜️   | Default — good ratio and speed                                                 |
| `GZIP_DEFAULT`  | 🐢    | 🗜️   | Files shared with MATLAB, HDFView, or other HDF5 tools                         |
| `ZSTD_FAST`     | ⚡⚡  | 📦   | Write speed matters more than file size                                        |
| `ZSTD_BEST`     | 🐢    | 🗜️🗜️ | Archival — smallest files, slower writes                                       |
| `BLOSC_DEFAULT` | ⚡⚡  | 🗜️   | Large arrays — parallel blosc2 with zstd inside                                |
| `LZF`           | ⚡    | 📦   | Fastest round-trip when ratio matters less than compatibility with other tools |
| `UNCOMPRESSED`  | 🐰    | 📦📦 | Debugging, or data that doesn't compress well                                  |

#### Hdf5Compression fields

- **`algorithm`** — `"zstd"` | `"gzip"` | `"lzf"` | `"blosc"` | `None`. Default: `"zstd"`. Set to `None` for
  uncompressed.
- **`level`** — `int | None`. Default: `3`. Algorithm-specific level (zstd: 1–22, gzip: 0–9, blosc: 0–9).
- **`shuffle`** — `bool`. Default: `True`. Byte-shuffle filter (improves compression ratio for numeric data).
- **`bloscCompressor`** — `"zstd"` | `"blosclz"` | `"lz4"` | `"lz4hc"` | `"zlib"`. Default: `"zstd"`. Sub-compressor
  used when `algorithm="blosc"`.

The zstd and blosc algorithms are provided by the [hdf5plugin](https://hdf5plugin.readthedocs.io/en/latest/usage.html)
package, which is included in the `[hdf5]` extra. See the hdf5plugin docs for full details on filter parameters and
tuning options. The gzip and lzf algorithms are built into h5py and work without hdf5plugin.

The `BLOSC_DEFAULT` preset uses [blosc2](https://www.blosc.org/pages/blosc-in-depth/) — a meta-compressor that adds
parallel blocking, byte-shuffle, and cache-aligned chunking on top of the chosen sub-compressor. Buffer alignment and
block sizes are handled automatically.

#### Compatibility note

The default `BLOSC_DEFAULT` (and `ZSTD_*` presets) produce files that require `hdf5plugin` to be installed on the
reading side as well — any environment with `versionable[hdf5]` has it, but external tools (MATLAB, HDFView, bare
h5py without the plugin) may not. If your HDF5 files need to be read by software outside this library, use
`GZIP_DEFAULT` instead — gzip is supported by every HDF5 implementation:

```python
versionable.save(rec, "recording.h5", compression=GZIP_DEFAULT)
```

### Lazy Loading

By default, array fields are not read from disk until first access. This means `load()` returns almost instantly even
for large files — the file handle stays open in the background and the array is fetched only when your code actually
uses it:

```python
loaded = versionable.load(Recording, "recording.h5")
loaded.name    # Loaded immediately (scalar)
loaded.data    # Read from disk on first access, then cached
```

Lazy loading is particularly useful when you have many recordings on disk and only need to inspect metadata (name,
sample rate, channel count) before deciding which ones to process.

### Preload

If you know you'll need an array right away, you can opt into eager loading to avoid the latency hit at first access
time — useful when you're about to iterate over the data in a tight loop:

```python
# Preload specific fields
loaded = versionable.load(Recording, "recording.h5", preload=["data"])

# Preload all arrays
loaded = versionable.load(Recording, "recording.h5", preload="*")
```

### Metadata Only

Load only scalar fields and skip arrays entirely. Accessing an array field raises `ArrayNotLoadedError`. This is the
fastest possible load — ideal for scanning a directory of files to build an index or filter by metadata before loading
the full data:

```python
loaded = versionable.load(Recording, "recording.h5", metadataOnly=True)
loaded.name    # Works
loaded.data    # Raises ArrayNotLoadedError
```
