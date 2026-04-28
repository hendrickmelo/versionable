# Changelog

## 0.2.0 (unreleased)

- Cycles in object graphs now raise `CircularReferenceError` at save time, with the field path of the revisit, instead
  of `RecursionError`. Detection covers all four backends (JSON, YAML, TOML, HDF5).
- Shared references are still duplicated on save and load as separate instances. Lossless shared-reference support (and
  therefore cycles, on opt-in) is planned for 0.3.0.

## 0.1.0

First stable release of **versionable**.

- Versioned persistence for Python dataclasses with schema hash validation and declarative migrations
- JSON, TOML, YAML, and HDF5 backends — base install requires only numpy
- Rich type support: numpy arrays, datetime, enums, nested Versionable types, and more
- Save-as-you-go HDF5 sessions for incremental writes and random access on large files
