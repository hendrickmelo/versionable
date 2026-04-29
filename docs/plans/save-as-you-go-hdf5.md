# Save-As-You-Go for HDF5

> **Historical plan — implemented.** Envelope key names in this document
> (`__OBJECT__`, `__VERSION__`, `__HASH__`) were renamed in 0.2.0; see
> `docs/plans/envelope-keys.md`. Current names are `object`, `version`, `hash`.

## Problem

When collecting data incrementally (e.g., appending traces from a DAQ, accumulating simulation results), users
want to persist data as it arrives so that:

- Progress survives crashes.
- Memory isn't wasted holding a complete copy just for serialization.
- The file is always (or nearly always) in a loadable state.

Today, `versionable.save()` writes the entire object at once. There's no way to persist incremental mutations.

## Foundation

[Native HDF5 Storage](native-hdf5-storage.md) eliminated the JSON `__scalars__` blob and stores every field
as a native HDF5 construct (attributes, datasets, groups). This is the foundation save-as-you-go builds on:

- **Individual field access** — each field is a separate HDF5 attribute/dataset/group, so one field can be
  written or updated without touching others.
- **Type-aware write helpers** — `_writeValue()` dispatches by type and writes any supported value to an
  HDF5 group. The session reuses these helpers directly.
- **`__versionable__` metadata** — stored in a child group, independent of field data.
- **Per-element lazy loading** — `LazyArrayList`/`LazyArrayDict` support per-element access, useful for
  resume mode.

## Goal

A user defines a normal Versionable dataclass and gets transparent, incremental persistence to HDF5:

```python
@dataclass
class Experiment(Versionable, version=1, hash="..."):
    name: str
    sampleRate_Hz: float
    traces: list[np.ndarray]
    timestamps: list[float]
    waveform: Annotated[np.ndarray, Appendable(chunkRows=64)]

# Open a live, file-backed instance
with versionable.hdf5.open(Experiment, "run001.h5") as exp:
    exp.name = "baseline"
    exp.sampleRate_Hz = 48000.0
    exp.waveform = np.empty((0, 1024))

    for chunk in daq.stream():
        exp.traces.append(chunk.data)        # new dataset written to disk
        exp.timestamps.append(chunk.time)    # 1-D dataset resized on disk
        exp.waveform.append(chunk.raw)       # resizable dataset grows on disk
    # file is valid and loadable at any point

# Later: load normally, no special API
exp = versionable.load(Experiment, "run001.h5")
```

## Design

### Core Concept: File-Backed Versionable Instance

`versionable.hdf5.open()` returns a **live instance** of the Versionable type backed by an open HDF5 file.
Mutations to the instance are intercepted and persisted:

| Mutation | What happens on disk |
|----------|---------------------|
| `exp.name = "foo"` | HDF5 attribute `name` updated |
| `exp.sampleRate_Hz = 48000` | HDF5 attribute `sampleRate_Hz` updated |
| `exp.traces.append(arr)` | New dataset `traces/N` created |
| `exp.traces[2] = newArr` | Dataset `traces/2` deleted and recreated |
| `exp.timestamps.append(t)` | `timestamps` dataset resized and value appended |
| `exp.waveform.append(chunk)` | Resizable dataset grows along declared/inferred axis |
| `exp.waveform[0] = newRow` | Direct write to dataset row |
| `exp.waveform = newArray` | Dataset deleted and recreated (resizable) |
| `exp.data = bigArray` | Dataset `data` replaced (fixed-size) |

### `Appendable` — Annotation for Growable ndarray Fields

`Appendable` is field metadata declared via `Annotated` that marks an `np.ndarray` field as
growable. It controls how the session creates the HDF5 dataset and how the field is wrapped
at runtime.

```python
from versionable import Appendable

@dataclass
class Experiment(Versionable, version=1, hash="..."):
    waveform: Annotated[np.ndarray, Appendable()]                     # auto chunk size, auto axis
    highRes: Annotated[np.ndarray, Appendable(chunkRows=128)]         # explicit chunks
    channels: Annotated[np.ndarray, Appendable(axis=1)]               # grows along axis 1
    calibration: np.ndarray                                            # fixed-size
```

```python
@dataclass(frozen=True)
class Appendable:
    """Marks an np.ndarray field as appendable.

    When used inside an HDF5 session, the field is backed by a resizable
    dataset and wrapped with a TrackedArray that supports .append().
    Outside sessions, this annotation is ignored — the field is a normal
    np.ndarray.

    Args:
        chunkRows: Number of elements per chunk along the append axis.
            If None, a heuristic targets ~256 KB per chunk based on
            dtype and the shape of the non-append dimensions.
        axis: The axis along which the dataset grows. If None, inferred
            from the first assigned array (the axis with size 0). If no
            axis has size 0, defaults to 0.
    """
    chunkRows: int | None = None
    axis: int | None = None
```

#### How `Appendable` Integrates

- **Hash system**: `_isNdarrayType()` sees through `Annotated[np.ndarray, Appendable()]` — the
  field hashes identically to `np.ndarray`. (This already works: `_isNdarrayType` strips
  `Annotated` wrappers.)
- **`save()`**: `_writeValue()` writes the array as a normal dataset. The `Appendable` marker
  is ignored — the dataset is contiguous.
- **`load()`**: Returns a normal `np.ndarray`. The `Appendable` marker is ignored.
- **Non-HDF5 backends**: Ignore the annotation entirely. `Annotated` metadata is invisible
  to JSON/YAML/TOML serialization.
- **HDF5 session**: Detects the `Appendable` marker in the field's type annotation, creates a
  resizable dataset with the appropriate `maxshape`, and wraps the field with `TrackedArray`.

#### Append Axis Resolution

The append axis is determined when the dataset is first created (on first assignment). Resolution
order:

1. **`Appendable(axis=1)`** — explicit, always used regardless of data shape.
2. **`Appendable()`** + exactly one axis has size 0 — inferred from shape:
   - `np.empty((0, 1024))` → axis 0
   - `np.empty((16, 0))` → axis 1
3. **`Appendable()`** + no axis has size 0 — defaults to axis 0.
4. **`Appendable()`** + multiple axes have size 0 — raises `BackendError` ("ambiguous append
   axis, specify `Appendable(axis=...)`").

Examples:

```python
# Axis inferred from shape
exp.waveform = np.empty((0, 1024))        # axis=0, maxshape=(None, 1024)
exp.channels = np.empty((16, 0))          # axis=1, maxshape=(16, None)

# Axis from Appendable declaration
# channels: Annotated[np.ndarray, Appendable(axis=1)]
exp.channels = np.zeros((16, 100))        # axis=1, maxshape=(16, None)

# Existing data, no zero axis, defaults to 0
exp.waveform = existing_data              # shape (100, 1024), axis=0, maxshape=(None, 1024)
```

The resolved axis is stored on the `TrackedArray` instance and used for all subsequent
`.append()` calls. Shape validation on append checks that all non-append dimensions match:

```python
# exp.waveform created with shape (0, 1024), axis=0
exp.waveform.append(np.zeros((10, 1024)))   # OK — 10 rows, columns match
exp.waveform.append(np.zeros((3, 1024)))    # OK — different row count, columns match
exp.waveform.append(np.zeros((5, 2048)))    # ValueError — column mismatch
```

#### Chunk Size Heuristic

When `chunkRows` is `None`, the session computes a chunk size targeting ~256 KB per chunk,
based on the non-append dimensions and dtype:

```python
TARGET_CHUNK_BYTES = 256 * 1024  # 256 KB

def _computeChunkSize(shape: tuple[int, ...], dtype: np.dtype, axis: int) -> int:
    """Compute elements per chunk along the append axis targeting TARGET_CHUNK_BYTES."""
    nonAppendShape = shape[:axis] + shape[axis + 1:]
    sliceBytes = int(np.prod(nonAppendShape)) * dtype.itemsize if nonAppendShape else dtype.itemsize
    if sliceBytes == 0:
        return 1
    return max(1, TARGET_CHUNK_BYTES // sliceBytes)
```

Example: axis=0, 1024 float64 columns → 8 KB/slice → `256 KB / 8 KB = 32` elements per chunk.

### `TrackedArray` — Wrapper for Appendable Fields

Inside an HDF5 session, `Appendable` fields are wrapped with `TrackedArray`. This is a wrapper
class (not an ndarray subclass) that holds a reference to the live HDF5 dataset and supports
append + element assignment.

```python
class TrackedArray:
    """Wrapper around a resizable HDF5 dataset.

    Supports .append() for growing the dataset along the declared axis,
    __setitem__ for direct element writes, and __array__() for
    numpy interop.
    """

    _dataset: h5py.Dataset
    _session: Hdf5Session[Any]
    _fieldName: str
    _axis: int  # resolved append axis

    def append(self, data: np.ndarray) -> None:
        """Append data along the append axis, resizing the dataset."""
        self._validateShape(data)
        axis = self._axis
        currentSize = self._dataset.shape[axis]
        newSlices = data.shape[axis] if data.ndim > axis else 1
        self._dataset.resize(currentSize + newSlices, axis=axis)
        # Build a slice that selects the newly appended region
        idx = [slice(None)] * self._dataset.ndim
        idx[axis] = slice(currentSize, None)
        self._dataset[tuple(idx)] = data

    def __setitem__(self, index: Any, value: Any) -> None:
        """Write directly to the HDF5 dataset."""
        self._dataset[index] = value

    def __getitem__(self, index: Any) -> np.ndarray:
        """Read from the HDF5 dataset."""
        return self._dataset[index]

    def __array__(self) -> np.ndarray:
        """Return full dataset as ndarray (for numpy interop)."""
        return self._dataset[()]

    def __len__(self) -> int:
        return self._dataset.shape[0]

    @property
    def shape(self) -> tuple[int, ...]:
        return self._dataset.shape

    @property
    def dtype(self) -> np.dtype:
        return self._dataset.dtype
```

#### Shape Validation

`TrackedArray.append()` validates that non-append dimensions match the existing dataset:

```python
def _validateShape(self, data: np.ndarray) -> None:
    """Raise ValueError if data shape is incompatible with the dataset."""
    expected = list(self._dataset.shape)
    del expected[self._axis]
    got = list(data.shape)
    if data.ndim > self._axis:
        del got[self._axis]
    if expected != got:
        raise ValueError(
            f"Cannot append to '{self._fieldName}': non-append dimensions must match. "
            f"Dataset shape is {self._dataset.shape} (axis={self._axis}), "
            f"but data shape is {data.shape}"
        )
```

#### Operations Summary

| Operation | Works? | Mechanism |
|-----------|--------|-----------|
| `.append(data)` | Yes | Resize dataset along append axis + write |
| `[i] = value` | Yes | `__setitem__` → direct dataset write |
| `[slice] = data` | Yes | `__setitem__` → direct dataset write |
| `= new_array` | Yes | `__setattr__` → delete + recreate resizable dataset |
| `np.mean(exp.field)` | Yes | `__array__()` protocol |
| `len(exp.field)` | Yes | `__len__` (size along append axis) |
| `.shape`, `.dtype` | Yes | Properties delegating to dataset |
| `.axis` | Yes | Property returning the append axis |
| `+= 1` | No | Use `exp.field = exp.field + 1` instead |

#### Type Checking

The field annotation is `Annotated[np.ndarray, Appendable()]`. The type checker sees `np.ndarray`.
Inside a session, the runtime value is `TrackedArray`. This means:

- `exp.waveform[0] = row` — type-safe (`np.ndarray` supports `__setitem__`)
- `np.mean(exp.waveform)` — type-safe (`np.ndarray` is accepted by numpy)
- `exp.waveform.append(data)` — **not type-safe** (ndarray has no `.append()`)

The `.append()` gap is the one place where the type checker doesn't match runtime behavior.
This is a narrow, predictable gap that only occurs inside `with hdf5.open(...)` blocks. Users
can suppress with a targeted `# type: ignore[attr-defined]` if desired.

### API Surface

```python
# --- versionable/hdf5.py ---

type SessionMode = Literal["create", "resume", "overwrite"]

def open(
    cls: type[T],
    path: str | Path,
    *,
    mode: SessionMode = "create",
    compression: Hdf5Compression | None = None,
) -> Hdf5Session[T]:
    """Open a file-backed Versionable instance for incremental writes.

    Args:
        cls: The Versionable dataclass type.
        path: HDF5 file path.
        mode: How to handle the target file:
            "create" (default) — new file, error if exists.
            "resume" — open existing file, restore state, continue.
            "overwrite" — delete existing file if present, create new.
        compression: Compression preset (default: ZSTD_DEFAULT).

    Returns:
        Context manager yielding a file-backed instance of cls.
    """
```

### `Hdf5Session` — The File-Backed Wrapper

```python
class Hdf5Session(Generic[T]):
    """Context manager wrapping a file-backed Versionable instance."""

    _file: h5py.File
    _root: h5py.Group
    _cls: type[T]
    _fieldTypes: dict[str, Any]
    _comp: Hdf5Compression
    _proxy: T  # the live proxy instance

    def __enter__(self) -> T:
        # Write __versionable__ metadata group
        # Return the proxy instance
        ...

    def __exit__(self, *exc) -> None:
        # Close the HDF5 file
        ...
```

The proxy returned by `__enter__` is an instance of a dynamic subclass of `T` that overrides
`__setattr__` to persist field assignments, and wraps mutable collection fields with tracked
proxies.

### Session Internals

#### File and Metadata Setup

On `__enter__` (new file):

1. Open HDF5 file in `"w"` mode.
2. Write `__versionable__` child group with `__OBJECT__`, `__VERSION__`, `__HASH__` from
   `metadata(cls)`.
3. Resolve `fieldTypes` via `_resolveFields(cls)`.
4. Create the proxy instance (dynamic subclass of `cls`).
5. Return the proxy.

The file is valid immediately after `__enter__` — it contains the metadata group and no fields.
Fields appear as the user sets them. This is safe because `load()` applies dataclass defaults for
missing fields.

#### Proxy Subclass

Created dynamically, analogous to how `_lazy.py` creates `_Lazy{ClassName}`:

```python
def _makeProxyClass(cls: type[T]) -> type[T]:
    """Create a dynamic subclass that persists field assignments."""

    def __setattr__(self: Any, name: str, value: Any) -> None:
        session: Hdf5Session[Any] = object.__getattribute__(self, "_session")
        fieldTypes = session._fieldTypes

        if name in fieldTypes:
            # Persist to HDF5
            session._persistField(name, value)

            # Wrap with tracked proxy if applicable
            value = session._wrapValue(name, value, fieldTypes[name])

        object.__setattr__(self, name, value)

    return type(f"_Live{cls.__name__}", (cls,), {"__setattr__": __setattr__})
```

The proxy holds a reference to the session via `_session` (set with `object.__setattr__` to avoid
triggering the override).

#### `_persistField()` — Delegating to Existing Write Helpers

This is where the session reuses the native HDF5 backend's write path:

```python
def _persistField(self, name: str, value: Any) -> None:
    """Write a single field to the HDF5 file."""
    fieldType = self._fieldTypes[name]
    datasetKwargs = self._comp.datasetKwargs()

    # Unwrap TrackedArray to get the raw ndarray for writing
    if isinstance(value, TrackedArray):
        value = np.asarray(value)

    # Remove existing data for this field (attribute, dataset, or group)
    if name in self._root.attrs:
        del self._root.attrs[name]
    if name in self._root:
        del self._root[name]

    # Check if this is an Appendable field
    appendable = _getAppendable(fieldType)
    if appendable is not None:
        # Create resizable dataset
        self._createResizableDataset(name, value, appendable)
    else:
        # Reuse the backend's _writeValue for type dispatch
        _writeValue(self._root, name, value, fieldType, datasetKwargs, self._comp)
```

For non-`Appendable` fields, `_writeValue()` handles all types the backend supports: scalars,
arrays, enums, converted types, nested Versionables, and all collection variants. No type
dispatch is reimplemented.

#### `_createResizableDataset()`

```python
def _createResizableDataset(
    self, name: str, data: np.ndarray, appendable: Appendable,
) -> tuple[h5py.Dataset, int]:
    """Create a chunked, resizable dataset for an Appendable field.

    Returns:
        Tuple of (dataset, resolvedAxis).
    """
    axis = _resolveAppendAxis(data.shape, appendable)

    # Chunk shape: user-specified or heuristic along the append axis
    chunkShape = list(data.shape)
    if appendable.chunkRows is not None:
        chunkShape[axis] = appendable.chunkRows
    else:
        chunkShape[axis] = _computeChunkSize(data.shape, data.dtype, axis)
    # Handle zero-size axis: chunk size must be ≥ 1
    chunkShape[axis] = max(1, chunkShape[axis])

    # maxshape: None (unlimited) along the append axis, fixed elsewhere
    maxshape = list(data.shape)
    maxshape[axis] = None

    ds = self._root.create_dataset(
        name,
        data=data,
        chunks=tuple(chunkShape),
        maxshape=tuple(maxshape),
        **self._comp.datasetKwargs(),
    )
    return ds, axis
```

#### `_resolveAppendAxis()`

```python
def _resolveAppendAxis(shape: tuple[int, ...], appendable: Appendable) -> int:
    """Determine the append axis from the Appendable config and data shape."""
    # 1. Explicit axis always wins
    if appendable.axis is not None:
        return appendable.axis

    # 2. Infer from zero-size axes
    zeroAxes = [i for i, s in enumerate(shape) if s == 0]
    if len(zeroAxes) == 1:
        return zeroAxes[0]
    if len(zeroAxes) > 1:
        raise BackendError(
            f"Ambiguous append axis: shape {shape} has multiple zero-size axes "
            f"{zeroAxes}. Specify Appendable(axis=...) explicitly."
        )

    # 3. Default to axis 0
    return 0
```

#### `_wrapValue()` — Wrapping Fields with Tracked Proxies

After persisting, the session wraps values that need mutation tracking:

```python
def _wrapValue(self, name: str, value: Any, fieldType: Any) -> Any:
    """Wrap a value with a tracked proxy if applicable."""
    # Appendable ndarray → TrackedArray backed by the just-created dataset
    appendable = _getAppendable(fieldType)
    if appendable is not None and name in self._root:
        return TrackedArray(self._root[name], self, name)

    # list → TrackedList
    if isinstance(value, list) and not isinstance(value, TrackedList):
        return TrackedList(value, self, name, fieldType)

    # dict → TrackedDict
    if isinstance(value, dict) and not isinstance(value, TrackedDict):
        return TrackedDict(value, self, name, fieldType)

    return value
```

#### `_getAppendable()` — Extracting the Annotation

```python
def _getAppendable(fieldType: Any) -> Appendable | None:
    """Extract Appendable metadata from an Annotated type, if present."""
    if typing.get_origin(fieldType) is Annotated:
        for arg in typing.get_args(fieldType)[1:]:
            if isinstance(arg, Appendable):
                return arg
    return None
```

### Mutation Interception

#### Scalar & Array Field Assignment (`__setattr__`)

Handled by the proxy subclass override (above). Assigning to any field:

1. Persists via `_persistField()` (delete old + write new).
2. Wraps the value if it's a tracked type (list, dict, or `Appendable` ndarray).
3. Sets it on the instance.

#### List Mutations (`.append()`, `[i] = ...`, `.extend()`)

The proxy wraps `list` fields with a `TrackedList` that intercepts mutations:

```python
class TrackedList(list, Generic[T]):
    """List subclass that notifies the session on mutation."""

    _session: Hdf5Session[Any]
    _fieldName: str
    _fieldType: Any  # the full field type (e.g., list[np.ndarray])

    def append(self, value: T) -> None:
        super().append(value)
        self._session._onListAppend(self._fieldName, len(self) - 1, value)

    def __setitem__(self, index: int, value: T) -> None:
        super().__setitem__(index, value)
        self._session._onListSetItem(self._fieldName, index, value)

    def extend(self, values: Iterable[T]) -> None:
        startIdx = len(self)
        super().extend(values)
        self._session._onListExtend(self._fieldName, startIdx, list(values))
```

Unsupported operations that would reorder HDF5 datasets (`insert`, `pop`, `remove`, `del`,
`sort`, `reverse`) raise `NotImplementedError` with a clear message. These can be added later
if needed.

#### Session Callbacks for List Mutations

The session handles list mutations differently depending on the storage format:

**`list[np.ndarray]` / `list[Versionable]` / `list[non-scalar]`** — stored as a group with
integer-keyed children:

```python
def _onListAppend(self, fieldName: str, index: int, value: Any) -> None:
    elemType = typing.get_args(self._fieldTypes[fieldName])[0]
    group = self._root.require_group(fieldName)
    datasetKwargs = self._comp.datasetKwargs()
    _writeValue(group, str(index), value, elemType, datasetKwargs, self._comp)

def _onListSetItem(self, fieldName: str, index: int, value: Any) -> None:
    elemType = typing.get_args(self._fieldTypes[fieldName])[0]
    group = self._root[fieldName]
    key = str(index)
    if key in group:
        del group[key]
    datasetKwargs = self._comp.datasetKwargs()
    _writeValue(group, key, value, elemType, datasetKwargs, self._comp)
```

**`list[float]` / `list[int]` / `list[str]` / `list[bool]`** — stored as a resizable 1-D dataset:

```python
def _onListAppend(self, fieldName: str, index: int, value: Any) -> None:
    if fieldName not in self._root:
        # First append: create resizable dataset
        elemType = typing.get_args(self._fieldTypes[fieldName])[0]
        dtype = _dtypeForElementType(elemType)
        self._root.create_dataset(
            fieldName, shape=(1,), maxshape=(None,), dtype=dtype,
            data=[value], **self._comp.datasetKwargs(),
        )
    else:
        ds = self._root[fieldName]
        ds.resize(ds.shape[0] + 1, axis=0)
        ds[-1] = value
```

The session inspects `_isScalarType(elemType)` to choose between group-based and dataset-based
dispatch. This check is done once when the field is first accessed or wrapped.

#### Dict Mutations

`TrackedDict` wraps `dict` fields similarly:

```python
class TrackedDict(dict, Generic[K, V]):
    """Dict subclass that notifies the session on mutation."""

    _session: Hdf5Session[Any]
    _fieldName: str

    def __setitem__(self, key: K, value: V) -> None:
        super().__setitem__(key, value)
        self._session._onDictSetItem(self._fieldName, key, value)

    def __delitem__(self, key: K) -> None:
        super().__delitem__(key)
        self._session._onDictDelItem(self._fieldName, key)

    def update(self, other: dict[K, V], **kwargs: V) -> None:
        super().update(other, **kwargs)
        for k, v in {**other, **kwargs}.items():
            self._session._onDictSetItem(self._fieldName, k, v)
```

The session callbacks use `_keyToStr()` for key encoding and `_writeValue()` for value dispatch
— the same functions used by `_writeDict()` in the backend.

### Resume Mode

When `mode="resume"`:

1. Verify the file exists; open in `"a"` mode.
2. Read `__versionable__` group and validate `__OBJECT__`, `__VERSION__`, `__HASH__` match `cls`.
3. Load existing fields using the backend's `_readFields()` path (with the same type dispatch).
4. Populate the proxy instance with loaded values.
5. Wrap collection fields with tracked proxies (preserving existing data).
6. For `Appendable` fields: the resizable dataset already exists on disk. Wrap with
   `TrackedArray` pointing to the existing dataset. Subsequent `.append()` calls resize it.
7. Subsequent mutations append/overwrite as usual.

```python
with versionable.hdf5.open(Experiment, "run001.h5", mode="resume") as exp:
    # exp.traces already has the 500 arrays from the previous run
    print(len(exp.traces))  # 500
    # exp.waveform is a TrackedArray backed by the existing dataset
    print(exp.waveform.shape)  # (50000, 1024)
    for chunk in daq.stream():
        exp.traces.append(chunk.data)     # appends starting at index 500
        exp.waveform.append(chunk.raw)    # resizes existing dataset
```

On resume, `list[float]` datasets are loaded from the existing 1-D dataset. The `TrackedList`
wrapper is initialized with the loaded values. The dataset already exists on disk, so
subsequent `.append()` calls resize it.

### Interaction with `load()`

No changes to `load()` needed. The file produced by `Hdf5Session` is structurally identical to what
`save()` produces — the same `__versionable__` metadata, the same attribute/dataset/group layout.
`load()` reads it with the standard HDF5 backend path.

The only difference: `Appendable` and `list[scalar]` datasets created by the session are chunked
(because they were created with `maxshape=(None,)` for resizing). This doesn't affect `load()` —
h5py reads chunked and contiguous datasets identically.

### Error Handling

| Scenario | Behavior |
|----------|----------|
| `"create"` and file exists | Raise `BackendError` |
| `"resume"` and file doesn't exist | Raise `BackendError` |
| `"resume"` with mismatched class/version/hash | Raise `BackendError` with details |
| `"overwrite"` and file exists | Delete existing file, create new |
| `"overwrite"` and file doesn't exist | Create new (same as `"create"`) |
| Setting a field not in the class | Normal `__setattr__` (no persistence) |
| `insert`/`pop`/`remove` on `TrackedList` | Raise `NotImplementedError` |
| `.append()` on non-`Appendable` ndarray field | Field is plain ndarray, `AttributeError` |
| `TrackedArray.append()` with incompatible shape | `ValueError` with clear message showing expected vs actual dimensions |
| `Appendable()` with multiple zero-size axes | `BackendError` ("ambiguous append axis, specify `Appendable(axis=...)`") |
| Exception inside `with` block | `__exit__` closes the file; partial data is on disk |
| Setting a field to unsupported type | `_writeValue()` raises `BackendError` (same as `save()`) |

## Implementation Plan

### Phase 1: `Appendable` Annotation + `TrackedArray`

**New file: `_appendable.py`**

1. `Appendable` dataclass — field metadata with `chunkRows: int | None` and `axis: int | None`.
2. `_getAppendable()` — extract `Appendable` from `Annotated` type.
3. `_resolveAppendAxis()` — determine axis from config + data shape (explicit → zero-axis
   inference → default 0; error on ambiguous multiple zero axes).
4. `_computeChunkSize()` — heuristic for auto chunk size along the append axis.
5. Verify `_isNdarrayType()` already handles `Annotated[np.ndarray, Appendable()]`.

**New file: `_tracked_array.py`**

6. `TrackedArray` class — wraps `h5py.Dataset`, stores resolved `_axis`. Implements `append`
   (with shape validation), `__setitem__`, `__getitem__`, `__array__`, `__len__`, `shape`,
   `dtype`, `axis`.

**Update: `__init__.py`**

7. Export `Appendable` in `__all__`.

**Tests:**

9. `Appendable` field hashes the same as plain `np.ndarray`.
10. `save()` / `load()` roundtrip ignores `Appendable` marker (field is normal ndarray).
11. Non-HDF5 backends ignore `Appendable`.
12. Axis inference: `np.empty((0, 1024))` → axis 0, `np.empty((16, 0))` → axis 1.
13. Axis inference: multiple zero axes → `BackendError`.
14. Explicit `Appendable(axis=1)` overrides shape inference.
15. No zero axis + no explicit axis → defaults to 0.

### Phase 2: Core Session + Scalar/Array Persistence

**New file: `_hdf5_session.py`**

16. `Hdf5Session` class — context manager that opens/closes the HDF5 file, writes `__versionable__`
    metadata on enter, stores `_file`, `_root`, `_cls`, `_fieldTypes`, `_comp`, `_mode`.
17. `_makeProxyClass()` — dynamic subclass factory with `__setattr__` override.
18. `_persistField()` — delete existing + call `_writeValue()` or `_createResizableDataset()`.
19. `_wrapValue()` — wrap `Appendable` fields with `TrackedArray`.
20. Session mode handling in `__enter__`:
    - `"create"` — open `"w"`, error if file exists.
    - `"overwrite"` — delete file if exists, open `"w"`.
    - `"resume"` — deferred to Phase 4.
21. Scalar field assignment works end-to-end: `exp.name = "foo"` writes an HDF5 attribute.
22. Plain ndarray field assignment works: `exp.data = arr` writes/replaces a dataset.
23. `Appendable` ndarray assignment: creates resizable dataset (with axis resolution),
    returns `TrackedArray`.
24. `TrackedArray.append()`: resizes dataset along resolved axis and writes new data.
25. `TrackedArray.__setitem__()`: writes directly to dataset.

**Update: `hdf5.py`**

26. Add `open()` factory function that creates and returns `Hdf5Session`.
27. Export `SessionMode` type alias.

**Tests:**

28. Create session, assign scalar fields, close, load with `versionable.load()`.
29. Assign plain ndarray field, verify dataset is contiguous.
30. Assign `Appendable` ndarray, verify dataset is resizable/chunked.
31. `TrackedArray.append()` loop, load and verify full array.
32. `TrackedArray.append()` with axis=1 (e.g., `(16, 0)` → grow columns).
33. `TrackedArray.append()` with varying chunk sizes (different row counts per append).
34. `TrackedArray.append()` shape mismatch → `ValueError`.
35. `TrackedArray.__setitem__()` partial writes.
36. `TrackedArray` works with `np.mean()`, `len()`, `.shape`, `.dtype`, `.axis`.
37. Chunk size heuristic produces reasonable values for various shapes/dtypes.
38. Explicit `chunkRows` overrides heuristic.
39. Overwrite a field, verify old value replaced.
40. Context manager closes file on exit.
41. `"create"` mode errors on existing file.
42. `"overwrite"` mode deletes existing file and creates new.

### Phase 3: Tracked Collections

43. `TrackedList` — list subclass with `append`, `__setitem__`, `extend`.
44. `TrackedDict` — dict subclass with `__setitem__`, `__delitem__`, `update`.
45. `_wrapValue()` extended for list and dict fields.
46. Session callbacks: `_onListAppend`, `_onListSetItem`, `_onListExtend`, `_onDictSetItem`,
    `_onDictDelItem`.
47. `list[np.ndarray]` append: creates datasets in group (`traces/0`, `traces/1`, ...).
48. `list[float]` append: creates resizable 1-D dataset, resizes on subsequent appends.
49. `dict[str, np.ndarray]` setitem: creates/replaces dataset in group.
50. Unsupported list operations (`insert`, `pop`, `remove`, `sort`, `reverse`) raise
    `NotImplementedError`.

**Tests:**

51. Append loop for `list[np.ndarray]`, load and verify all arrays.
52. Append loop for `list[float]`, load and verify values.
53. Dict setitem, load and verify.
54. Overwrite list element via `__setitem__`.
55. `NotImplementedError` on unsupported list ops.
56. Mixed: set scalars + append to lists + append to TrackedArray in same session.

### Phase 4: Resume Mode

57. `"resume"` handling in `__enter__`: open in `"a"`, validate metadata, load existing fields.
58. Populate proxy with loaded values, wrap collections and `Appendable` fields.
59. `list[np.ndarray]` resume: new appends start at correct index.
60. `list[float]` resume: resizable dataset is reused.
61. `Appendable` ndarray resume: `TrackedArray` wraps existing resizable dataset, axis
    re-resolved from existing dataset's `maxshape` (the unlimited dimension).
62. Scalar overwrite on resume.

**Tests:**

63. Write N items, close, resume, write more, load all.
64. `"resume"` validates class identity (error on mismatch).
65. `"resume"` with empty file (metadata only, no fields set yet).
66. `"resume"` preserves existing scalar values.
67. `"resume"` `Appendable` field: append continues from existing size.
68. `"resume"` on nonexistent file → `BackendError`.

### Phase 5: Polish

69. `flush()` method for in-place-mutated fields (non-`Appendable` ndarrays).
70. `list[Versionable]` append support (creates subgroups with `__versionable__` metadata).
71. `dict[str, Versionable]` setitem support.
72. Nested Versionable field assignment (`exp.config = SomeConfig(...)`).
73. `TrackedList.extend()` for batch appends.
74. Completeness warning: on `__exit__`, optionally warn about fields that were never set
    (fields without defaults that are still uninitialized).

**Tests:**

75. Nested Versionable roundtrip through session.
76. `list[Versionable]` append + load.
77. `flush()` after in-place mutation of non-`Appendable` ndarray.
78. Warning on unset required fields.

### Phase 6: Documentation

79. **`Appendable` and `TrackedArray` guide** in README — covers:
    - Declaring `Appendable` fields with `Annotated[np.ndarray, Appendable()]`.
    - `axis` parameter: explicit (`Appendable(axis=1)`) vs inferred from zero-size
      dimensions vs default (axis 0). Include examples for each case.
    - `chunkRows` parameter: when to use explicit chunk sizes vs the auto heuristic,
      with guidance on how chunk size affects performance (read/write tradeoffs,
      storage overhead).
    - `TrackedArray` operations: `.append()`, `__setitem__`, `__getitem__`,
      `__array__()` interop with numpy, what works and what doesn't (`+=`).
    - Shape validation: what happens when appended data has incompatible dimensions.
    - Common patterns: DAQ-style row append `(0, channels)`, column-growing
      `(rows, 0)`, starting with existing data.
    - Type-checking note: `.append()` is the one gap, how to suppress if desired.
80. **`hdf5.open()` session guide** in README — covers:
    - Session mode options: `"create"`, `"resume"`, `"overwrite"` with examples.
    - Scalar, array, list, and dict field persistence behavior.
    - `TrackedList` / `TrackedDict` supported operations and limitations.
    - Error scenarios and how each mode handles existing/missing files.
    - Interaction with `load()` — files produced by sessions are standard HDF5.
81. **Docstrings** on all public API: `Appendable`, `TrackedArray`, `Hdf5Session`,
    `hdf5.open()`, `TrackedList`, `TrackedDict`.

## Decisions

1. **Reuse `_writeValue()` for all non-`Appendable` persistence.** The session is a thin
   mutation-tracking layer on top of the existing write path, not a parallel implementation.

2. **`Appendable` annotation for growable ndarray fields.** Declared via
   `Annotated[np.ndarray, Appendable()]`. Carries optional `chunkRows` for chunk size control
   and optional `axis` for the append direction. Both are resolved at runtime when the dataset
   is first created — `chunkRows` defaults to a ~256 KB heuristic based on actual data shape
   and dtype, `axis` is inferred from zero-size dimensions or defaults to 0. Lives in the
   top-level `versionable` namespace (not `versionable.hdf5`) because it's field metadata, not
   a backend feature. Ignored by non-HDF5 backends and by `save()`/`load()`.

3. **`TrackedArray` is a wrapper, not an ndarray subclass.** ndarray is fixed-size and can't
   support `.append()` via subclassing. `TrackedArray` wraps an `h5py.Dataset` and implements
   `__array__()` for numpy interop, `__setitem__` for direct writes, and `.append()` for
   resizing. The type-checking gap (`.append()` not on `np.ndarray`) is narrow and only occurs
   inside session blocks.

4. **Write metadata on `__enter__`, fields lazily.** The file exists immediately (crash safety)
   but no empty structure is pre-created for fields. A field appears in the file only when first
   set.

5. **Resizable datasets for `Appendable` ndarrays and `list[scalar]`.** `list[np.ndarray]` uses
   group-of-datasets (no resizing needed — each append creates a new dataset). `list[float]`
   uses a chunked 1-D dataset with `maxshape=(None,)`. `Appendable` ndarray fields use chunked
   datasets with `maxshape` set to `None` along the append axis and fixed along all other axes.
   Non-`Appendable` ndarray fields are always replaced on assignment (contiguous, fixed-size).

6. **No reorder operations on `TrackedList`.** `insert`, `pop`, `remove`, `sort`, `reverse`
   would require renaming/shifting integer-keyed datasets in HDF5 groups, which is expensive
   and error-prone. These raise `NotImplementedError`. Users who need these patterns should
   build their data in memory and assign the whole list.

7. **No nested `TrackedList`/`TrackedDict`.** Only top-level collection fields are tracked.
   `list[list[float]]` would track the outer list but not inner lists. This avoids unbounded
   proxy depth and keeps the implementation simple. If an inner value is mutated, reassign
   the outer element: `exp.matrix[0] = updated_row`.

8. **Proxy class cached per Versionable type.** Same pattern as `_lazyClassCache` in `_lazy.py`.

### Phase 7: Instance-based `open()`

Currently `open()` only accepts a **type** and returns an empty proxy. This forces
users to re-set every field inside the `with` block even if they already have a
populated instance. Phase 7 adds an overload that accepts an **instance**:

```python
rec = Recording(name="baseline", sampleRate_Hz=48000.0, ...)

with versionable.hdf5.open(rec, "recording.h5") as rec:
    # All fields persisted on enter — just append
    for chunk in daq.stream():
        rec.time_s.append(chunk.time)
```

**Changes:**

82. Update `open()` signature: accept `type[T] | T` as the first argument.
    When an instance is passed, infer the class from `type(obj)`.
83. Update `Hdf5Session.__init__()` to store the optional source instance.
84. In `__enter__`, when a source instance is provided:
    - Create the proxy as usual.
    - Copy all field values from the source instance to the proxy via
      `__setattr__` (which triggers `_persistField` + `_wrapValue`).
    - This means all fields are written to disk on enter.
85. For `mode="resume"` with an instance: raise `BackendError` — resume
    restores state from file, so passing initial values is contradictory.

**Tests:**

86. Open with an instance, load back, verify all fields match.
87. Open with an instance that has Appendable fields, append more, verify.
88. Open with an instance in overwrite mode.
89. Open with an instance in resume mode raises `BackendError`.
90. Open with an instance — verify the proxy is a different object (not the
    original), so mutations don't affect the source.

## Open Questions

1. **Thread safety.** The session holds an open `h5py.File`. HDF5 is not thread-safe by default
   (unless built with `--enable-threadsafe`). Should the session document this limitation, or
   add a lock around file operations?

2. **Multiple sessions on same file.** Should `open()` check for an existing lock file or
   `fcntl` lock to prevent concurrent writers? HDF5 does not support concurrent write access.
   SWMR (single-writer-multiple-reader) mode is a possibility for future work.
