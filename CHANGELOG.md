# Changelog

## 0.2.0 (unreleased)

- Cycles in object graphs now raise `CircularReferenceError` at save time, with the field path of the revisit, instead
  of `RecursionError`. Detection covers all four backends (JSON, YAML, TOML, HDF5).
- Shared references are still duplicated on save and load as separate instances. Lossless shared-reference support (and
  therefore cycles, on opt-in) is planned for 0.3.0.
- File format: dropped the redundant dunders inside the `__versionable__` envelope (`__OBJECT__` → `object`,
  `__VERSION__` → `version`, `__HASH__` → `hash`, `__FORMAT__` → `format`), and re-namespaced the user-data sentinels
  with a `__ver_*__` prefix (`__ndarray__` → `__ver_ndarray__`, `__json__` → `__ver_json__`). The `__versionable__`
  wrapper key itself is unchanged.
- File format: nested `Versionable` values now carry their own `__versionable__` envelope, just like the root.
  Previously the envelope keys were flat alongside data fields in JSON/YAML/TOML; HDF5 already wrapped at every level.
  For TOML this is emitted as a `[parent.__versionable__]` sub-table. The deserialize path is structurally unchanged —
  envelope keys are skipped during field iteration whether flat or wrapped, so 0.1.x files (with flat nested envelopes)
  continue to load.
- Backwards compatibility: `load()` continues to accept the old key names from 0.1.x files for the entire 0.2.x line
  (preferring new keys when both are present); the legacy read path will be removed in 1.0. Saved files always use the
  new keys.
- The warning emitted by `load()` for files missing version metadata now reads `No version found …` (was
  `No __VERSION__ found …`).
- HDF5 backend now auto-registers `hdf5plugin` filters on import, so files written with zstd/blosc compression can be
  read without the consumer having to import `hdf5plugin` themselves. When `hdf5plugin` is missing and a load fails on a
  filter-related error, the raised `BackendError` now suggests `pip install hdf5plugin` (#20).

## 0.1.0

First stable release of **versionable**.

- Versioned persistence for Python dataclasses with schema hash validation and declarative migrations
- JSON, TOML, YAML, and HDF5 backends — base install requires only numpy
- Rich type support: numpy arrays, datetime, enums, nested Versionable types, and more
- Save-as-you-go HDF5 sessions for incremental writes and random access on large files
