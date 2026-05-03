# Native HDF5 Storage (Eliminate JSON)

> **Historical plan — implemented.** Envelope key names in this document
> (`__OBJECT__`, `__VERSION__`, `__HASH__`, `__ndarray__`) were renamed in
> 0.2.0; see `docs/plans/envelope-keys.md`. Current names are `object`,
> `version`, `hash`, `__ver_ndarray__`.

## Problem

The HDF5 backend previously bundled all non-array fields into a single `__scalars__` JSON attribute:

```python
# Current _writeGroup behavior
scalars: dict[str, Any] = {}
for key, value in fields.items():
    if isinstance(value, np.ndarray):
        group.create_dataset(key, data=value, **datasetKwargs)
    elif isinstance(value, dict) and "__OBJECT__" in value:
        _writeNestedGroup(group, key, value, comp)
    else:
        scalars[key] = value                           # ← everything else
if scalars:
    group.attrs["__scalars__"] = json.dumps(scalars)   # ← one big JSON blob
```

This defeated the purpose of using HDF5:

- **No individual field access** — can't read or update one scalar without parsing the whole blob.
- **No type fidelity** — `float` becomes JSON `number`, `int` becomes JSON `number`, bools and
  None are coerced through JSON round-trips.
- **No interop** — other tools (MATLAB, HDFView, h5dump) see a single opaque JSON string.
- **`list[np.ndarray]` is base64-encoded** — each array becomes a `{"__ndarray__": true, "data": "..."}` dict
  inside the JSON blob, losing compression, lazy loading, and direct dataset access.
- **Blocks save-as-you-go** — can't incrementally update a JSON blob efficiently.

## Goal

Every field maps to a native HDF5 construct. No JSON anywhere in `.h5` files. The file should be
human-readable in HDFView / h5dump and accessible from any HDF5-compatible tool.

## Native Type Mapping

The write/read paths use **fully recursive type dispatch**. Any value is classified into one of
three categories, and collections recurse for their elements:

| Category | Types | HDF5 representation |
|---|---|---|
| Scalar | `int`, `float`, `bool`, `str`, `Enum`, converted types, `None` | Attribute |
| Array | `np.ndarray` | Dataset (compressed) |
| Scalar sequence | `list[scalar]`, `set[scalar]`, `tuple[scalar, ...]`, `frozenset[scalar]` | 1-D dataset |
| Collection of non-scalars | `list[T]`, `set[T]`, `tuple[T, ...]`, `frozenset[T]` | Group with integer keys, recurse for each element |
| Dict | `dict[K, V]` | Group with string-converted keys, recurse for each value |
| Versionable | Nested `Versionable` | Subgroup with `__versionable__` metadata group, recurse for fields |

`None` is stored as `h5py.Empty("f")` — a native HDF5 null.

### Dict Key Conversion

HDF5 group keys are always strings. Non-string dict keys are converted to strings on write and
back to the original type on read using the type annotation:

- `int` → `str(42)` → `int("42")`
- `float` → `str(3.14)` → `float("3.14")`
- `Enum` → `str(value)` → reconstruct from value
- Converted types (UUID, Path, etc.) → use the converter registry
- `str` → identity

### Recursive Nesting

Because the dispatch is recursive, arbitrarily nested structures work naturally:

- `dict[str, list[float]]` — group of 1-D datasets
- `list[dict[str, int]]` — group of subgroups, each with int attributes
- `dict[int, dict[str, Versionable]]` — groups all the way down

### Distinguishing Groups

Groups are used for three purposes, distinguished by the presence of a `__versionable__` child group:

| Has `__versionable__` child group? | Meaning |
|---|---|
| Yes | Versionable object; `__versionable__` holds `__OBJECT__`, `__VERSION__`, `__HASH__` (+ `__FORMAT__` reserved for future versionable versioning) |
| No | Collection group (list, set, tuple, frozenset, or dict); type determined from the parent class's field annotations |

No type marker attributes are stored on collection groups. The deserializer always has the class's type
annotations available and uses them to determine how to reconstruct the group.

## Target File Layout

```text
experiment.h5
├── group: __versionable__/
│   ├── attr: __OBJECT__ = "Experiment"
│   ├── attr: __VERSION__ = 1
│   └── attr: __HASH__ = "a1b2c3"
├── attr:  name = "baseline"
├── attr:  sampleRate_Hz = 48000.0
├── dataset: timestamps  [shape=(3,), dtype=float64]
└── group: traces/
    ├── dataset: 0  [shape=(1024,), dtype=float64, zstd]
    ├── dataset: 1  [shape=(1024,), dtype=float64, zstd]
    └── dataset: 2  [shape=(1024,), dtype=float64, zstd]
```

## Serialization Architecture

### Previous Flow (Problem)

```text
save(obj, path)
  → serialize(value, fieldType, nativeTypes={ndarray})  # generic dict-oriented serializer
  → backend.save(serializedDict, meta, path)            # backend gets pre-serialized data
```

The backend received a flat dict of already-serialized values. It didn't know the original field types,
so it couldn't distinguish `list[np.ndarray]` (→ group of datasets) from `list[dict]` (→ unsupported). It
just saw a Python list and dumped it to JSON.

### New Flow

```text
save(obj, path)
  → backend.save(rawValues, meta, path, cls=type(obj))
```

Every backend receives **raw field values** and the **Versionable class**. Each backend decides how
to serialize for its format:

- `_api.py` no longer calls `serialize()` — it passes raw values and the class to the backend.
- Dict-based backends (JSON, YAML, TOML) call `serialize()` internally at the start of their
  `save()` method, resolving field types from the class. This is a one-line change per backend.
- The HDF5 backend resolves field types from the class and does its own type dispatch.
- Converted types (datetime, Enum, etc.) are resolved by each backend using the same converter
  registry.

This keeps `_api.py` simple (no conditional paths) and gives each backend full control over how
data is represented.

### Backend ABC Change

```python
class Backend(ABC):
    nativeTypes: ClassVar[set[type]] = set()

    @abstractmethod
    def save(
        self,
        fields: dict[str, Any],
        meta: dict[str, Any],
        path: Path,
        *,
        cls: type,                                     # new, required
        **kwargs: Any,
    ) -> None: ...
```

All backends receive `cls`. Dict-based backends use it to resolve field types for `serialize()`;
the HDF5 backend uses it for native type dispatch.

## Implementation Plan

### 1. Refactor Backend Interface (commit: pure refactor, no behavior change)

- Add required `cls` parameter to `Backend.save()`.
- Update `save()` in `_api.py`: pass raw values + the class instead of calling `serialize()`.
- Update dict-based backends (JSON, YAML, TOML) to call `serialize()` internally in their
  `save()` methods, resolving field types from `cls`.
- Run full existing test suite to confirm no regressions.

### 2. Write Path (`_hdf5_backend.py`)

- `_writeValue()` does fully recursive type dispatch on each field:
  - `None` → `h5py.Empty("f")` attribute
  - `np.ndarray` → dataset with compression
  - `Versionable` → subgroup with `__versionable__` metadata, recurse for fields
  - `Enum` → attribute (`.value`)
  - Scalar primitives (`int`, `float`, `str`, `bool`) → attribute
  - Converted types (datetime, Path, etc.) → apply converter, write attribute
  - `list`/`set`/`frozenset`/`tuple` of scalars → 1-D dataset
  - `list`/`set`/`frozenset`/`tuple` of non-scalars → group with integer keys, recurse
  - `dict[K, V]` → group with percent-encoded keys, recurse for values
- Dict keys are percent-encoded via `_keyToStr()`: `/` → `%2F`, `%` → `%25`,
  bare `.` → `%2E`. Null bytes and empty keys are rejected. Decoded on read via
  `urllib.parse.unquote()`.
- Nested Versionable writes operate on live instances, not pre-serialized dicts.
  Recursion handles arbitrarily deep nesting (e.g., `dict[str, Versionable]`
  where each Versionable contains `list[np.ndarray]`).
- Metadata into `__versionable__` child group. No `json.dumps` anywhere.

### 3. Read Path (`_hdf5_backend.py`)

- `_readFields()` reconstructs fields with recursive type dispatch:
  - Scalar attributes → Python primitives (`np.int64` → `int`, etc. via `_readAttr()`)
  - Datasets → `np.ndarray` or `list[scalar]` (via `_readDataset()`, type from class annotation)
  - Groups with `__versionable__` → `_readVersionableGroup()`, which resolves the class from the
    declared type annotation first, falling back to `_resolveClass()` by `__OBJECT__` name
  - Groups without `__versionable__` → `_readGroup()` with recursive dispatch through
    `_readSequenceGroup()`, `_readDictGroup()`, `_readChild()`
- Dict keys are decoded from percent-encoding via `_strToKey()` using `urllib.parse.unquote()`
  and converted back to the original type (int, float, Enum, etc.) from the type annotation.
- `load()` and `loadLazy()` both raise `BackendError` when `_resolveClass()` returns `None`.

### 4. Lazy Loading Updates

Lazy loading is **recursive** — it applies at every level of the object tree, not just the root.

#### Current architecture

`loadLazy` returns a flat `(fields, meta, lazyFields)` tuple. `_api.py` skips `deserialize()` for
fields in `lazyFields` and wraps the instance with `makeLazyInstance`, which installs a
`__getattribute__` override that resolves `LazyArray` sentinels on first access.

The problem: nested Versionable groups are read eagerly via `_readGroup` → `_readVersionableGroup`,
which loads everything including arrays. Lazy loading only applies to top-level fields.

#### Design

The read path already reconstructs nested Versionables as dicts with `__OBJECT__` metadata (which
`_deserializeVersionable` in `_types.py` turns into instances). The fix is to make the **HDF5 read
helpers** lazy-aware when building those dicts:

1. **`_readFields` accepts a lazy context** — a `_LazyContext` dataclass holding `path`,
   `preloadAll`, `preloadSet`, and `metadataOnly`. When `None`, everything is eager (used by
   `Backend.load()`). When present, the same lazy logic that `loadLazy` applies to the root is
   applied recursively at every level.

2. **`_readVersionableGroup` propagates the lazy context** — it passes the context down to
   `_readFields`, so nested Versionable objects get the same lazy treatment as the root.

3. **`_readFields` with lazy context** classifies each child the same way `loadLazy` does today:
   - `np.ndarray` dataset → `LazyArray(path, fullDatasetPath)` (uses the full HDF5 path)
   - Scalar dataset (`list[float]`, etc.) → eager
   - `list[np.ndarray]` / `dict[str, np.ndarray]` group → `LazyArrayList` / `LazyArrayDict`
   - Nested Versionable group → recurse with the same lazy context
   - Scalar attributes → eager

4. **`_api.py` deserialization handles lazy sentinels at any depth** — `_deserializeVersionable`
   in `_types.py` already iterates fields and calls `deserialize()`. It needs to:
   - Skip deserialization for `LazyArray` / `LazyArrayList` / `LazyArrayDict` / `ArrayNotLoaded`
     sentinels (pass them through as-is)
   - Call `makeLazyInstance` on the nested instance if it has lazy fields

5. **`loadLazy` simplifies** — instead of duplicating the field classification logic, it calls
   `_readFields(group, fieldTypes, lazyContext)` and collects `lazyFields` from the result.
   The root-level lazy field tracking is just for the `makeLazyInstance` call in `_api.py`.

#### Sentinel recognition

`_types.py` must not import `_lazy.py` at module level (it would pull in `h5py` on systems
without it). Instead, each sentinel class has `_isLazySentinel = True`. Detection is via
`getattr(value, "_isLazySentinel", False)`. The `makeLazyInstance` import is deferred inside
the `if lazyFields:` block that only runs in HDF5 paths.

#### What stays the same

- `LazyArray`, `LazyArrayList`, `LazyArrayDict`, `ArrayNotLoaded` — unchanged
- `makeLazyInstance` / `_getLazyClass` — unchanged, just applied at every Versionable level
- `preload` / `metadataOnly` semantics — unchanged, just applied recursively
- The write path — unchanged (lazy loading is read-only)

### 5. Deserialization

- The HDF5 backend returns values that are closer to their final Python types (not JSON-derived).
- `deserialize()` still runs on the loaded data (to handle Enum reconstruction, converter types, etc.),
  but the data is no longer coming from JSON — it's native Python/numpy types.
- May need small adjustments to `deserialize()` to handle numpy scalars (`np.int64` → `int`, etc.).

### 6. `loadDynamic` Support

- `loadDynamic()` reads `__OBJECT__` from `__versionable__`, looks up the class in the registry,
  then has `cls` available for type-aware reconstruction. No special handling needed.

### 7. Migration Considerations

- Migrations transform raw field dicts and must see the same types regardless of backend.
- The HDF5 backend's `load()` normalizes values to standard Python types: `np.int64` → `int`,
  `np.float64` → `float`, 1-D datasets → `list`, etc. The only native type that passes through
  is `np.ndarray` (same as today).
- This keeps migrations backend-agnostic — the HDF5-native layout is purely a file format concern.

### 8. Tests

- **Cross-backend kitchen sink** — every common type (str, int, float, bool, Enum, datetime,
  date, timedelta, Path, UUID, Literal, set, frozenset, tuple, dict with int/str keys, nested
  dict[str, list[float]], nested Versionable, list[Versionable]) roundtripped through all 4
  backends with exact type and value assertions.
- **Numpy dtypes** — 13 dtypes × 4 backends, plus 2D, 3D, and empty array shape preservation.
- **HDF5-specific** — native attribute storage (no `__scalars__`), `__versionable__` metadata
  layout, compression algorithms, file extensions.
- **Lazy loading** — per-element lazy for `list[np.ndarray]` (`LazyArrayList`) and
  `dict[str, np.ndarray]` (`LazyArrayDict`), preload, metadataOnly, slicing, iteration,
  lazy dict with percent-encoded keys and cache state verification.
- **Recursive lazy loading** — arrays inside nested Versionables get `LazyArray` sentinels;
  deeply nested `dict[str, Versionable]` with `list[np.ndarray]` fields verified lazy at
  two levels deep; `preload="*"` eagerly loads everything.
- **Dict key edge cases** — `/`, `%2F` literal, `.` all roundtrip correctly via percent-encoding.
- **Error cases** — missing file, unregistered class in `load()` and `loadLazy()`.
- **None handling** — `None` vs default, optional fields, TOML omit behavior.
- **Empty collections** — empty lists, empty `list[np.ndarray]`, empty `dict[str, np.ndarray]`.
- **No JSON anywhere** — recursive assertion that no attribute in the file contains JSON strings.
- **CI** — all tests gated on HDF5 availability for minimal (no h5py) environments.

## Design Decisions

1. **Integer keys for list groups.** Bare integers (`0`, `1`, …) without `track_order`. On read,
   keys are sorted numerically (`sorted(group.keys(), key=int)`).

2. **No type hints for converted types.** The class's type annotations are sufficient at load time.
   External tools will see the stored value (e.g., a string for datetime) without type context.

3. **`None` via `h5py.Empty`.** HDF5 attributes can't store `None`. Fields with value `None` are
   stored as `group.attrs[name] = h5py.Empty("f")` — a native HDF5 null. On read, detected via
   `isinstance(value, h5py.Empty)` and converted back to `None`. This is needed because omitting
   the attribute would fall back to the dataclass default, which may not be `None`.
   (`skipDefaults` still works: if the default is `None` and the value is `None`, the field is
   omitted entirely — no empty attribute written.)

4. **Dict key percent-encoding.** HDF5 interprets `/` as a path separator and `.` as the current
   group. Keys are encoded on write (`/` → `%2F`, `%` → `%25`, bare `.` → `%2E`) and decoded
   on read via `urllib.parse.unquote()`. Null bytes and empty keys are rejected.

5. **Strict class resolution.** Both `load()` and `loadLazy()` raise `BackendError` when the
   file's `__OBJECT__` class can't be resolved. Silently falling back to `fieldTypes = {}` would
   cause arrays to be misinterpreted as lists and vice versa.
