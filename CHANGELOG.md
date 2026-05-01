# Changelog

## 0.2.0 (unreleased)

- Migrations now apply recursively to nested `Versionable` values — direct fields, `list[B]` / `dict[K, B]` /
  `tuple[B, ...]` / `set[B]` elements, and any depth of nesting. Previously migrations only ran at the root of a load,
  so nested data with a schema change between save and load failed (or silently corrupted) at deserialize time. Each
  nested file version is migrated against its own class's `Migrate` chain; a newer nested version raises `VersionError`
  identifying the type.
- Polymorphism is preserved across save/load: `list[Animal]` saved with `Dog` and `Cat` subclass instances reconstructs
  as a list of the original subclass types. The per-element envelope's `object` name drives class lookup in the global
  registry. Unknown names or wrong-subclass mismatches raise `BackendError` identifying the nested type. Combines with
  migrations and `old_names`: each subclass migrates against its own chain, and old files referencing renamed subclasses
  load via `old_names`.
- `unknown="error"` / `"ignore"` / `"preserve"` now applies at every nesting level — each nested class's setting governs
  its own field data, mirroring root behavior.
- `versionable.load(..., validateLiterals=...)` now propagates to nested `Versionable` values; previously the override
  was silently dropped at every nesting boundary. When no override is set, each class's own `validate_literals` setting
  still applies at its own boundary.
- Save-side guard: `dict[Versionable, X]` now raises `ConverterError` at save time. Dict keys can't carry envelope
  information and previously round-tripped as Python repr strings.
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

## 0.1.0

First stable release of **versionable**.

- Versioned persistence for Python dataclasses with schema hash validation and declarative migrations
- JSON, TOML, YAML, and HDF5 backends — base install requires only numpy
- Rich type support: numpy arrays, datetime, enums, nested Versionable types, and more
- Save-as-you-go HDF5 sessions for incremental writes and random access on large files
