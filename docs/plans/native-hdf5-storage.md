# Native HDF5 Storage (Eliminate JSON)

## Problem

The HDF5 backend currently bundles all non-array fields into a single `__scalars__` JSON attribute:

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

This defeats the purpose of using HDF5:

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

### Current Flow (Problem)

```text
save(obj, path)
  → serialize(value, fieldType, nativeTypes={ndarray})  # generic dict-oriented serializer
  → backend.save(serializedDict, meta, path)            # backend gets pre-serialized data
```

The backend receives a flat dict of already-serialized values. It doesn't know the original field types,
so it can't distinguish `list[np.ndarray]` (→ group of datasets) from `list[dict]` (→ unsupported). It
just sees a Python list and dumps it to JSON.

### Proposed Flow

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

- Resolve field types from `cls` and dispatch each field to the appropriate HDF5 construct.
- Rewrite `_writeGroup()` to dispatch on field type:
  - Primitives → `group.attrs[name] = value`
  - `np.ndarray` → `group.create_dataset(name, data=value, **comp)`
  - `None` → `group.attrs[name] = h5py.Empty("f")`
  - `list[numeric]` / `list[str]` / `list[bool]` → `group.create_dataset(name, data=value)`
  - `list[np.ndarray]` → create subgroup with integer-keyed datasets (`0`, `1`, …)
  - `dict[str, np.ndarray]` → create subgroup with named datasets
  - `Versionable` → create subgroup with a `__versionable__` child group holding metadata attrs
  - `list[Versionable]` → create subgroup with integer-keyed subgroups, each with `__versionable__`
  - Enum → `group.attrs[name] = value.value`
  - Converted types → apply converter, write attribute
- **Recursive write:** The write path operates on live Versionable instances, not pre-serialized
  dicts. When a field is a nested Versionable, the writer resolves its metadata and field types
  from the class, then recursively writes each of its fields as native HDF5 constructs. The same
  applies to `list[Versionable]` — each element is a live instance whose fields must be
  recursively written. This handles arbitrarily deep nesting (e.g., a Versionable containing a
  `list[Versionable]` where each element contains arrays, nested Versionables, etc.).
- Write metadata (`__OBJECT__`, `__VERSION__`, `__HASH__`) into a `__versionable__` child group at the root.
- Remove all `json.dumps` from write path.

### 3. Read Path (`_hdf5_backend.py`)

- Reconstruct fields from attributes + datasets + groups:
  - Scalar attributes → field values directly.
  - Datasets → `np.ndarray` or `list[numeric]` (distinguished by the class's field type).
  - Groups with `__versionable__` child → nested Versionable; read `__versionable__` attrs for metadata, recurse for fields.
  - Groups without `__versionable__` → collection; the class's field type determines `list` vs `dict` reconstruction.

### 4. Lazy Loading Updates

- `loadLazy` needs to handle the new group types:
  - `list[np.ndarray]` group → could return a `LazyList` where each element is a `LazyArray`.
  - Individual scalar attributes → always eagerly loaded (they're tiny).
  - `list[float]` dataset → eagerly loaded (small) or lazy like any other dataset.

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

- Roundtrip for every type in the mapping table.
- `list[np.ndarray]` with varying shapes and dtypes.
- `dict[str, np.ndarray]` roundtrip.
- `list[Versionable]` roundtrip.
- Lazy loading: per-element lazy for `list[np.ndarray]`.
- Error case: unsupported nested type raises clear error.
- Compression still works with all new storage patterns.
- Nested Versionable containing array collections.

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
