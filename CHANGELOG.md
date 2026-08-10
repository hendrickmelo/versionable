# Changelog

## Unreleased

**Breaking change: every schema hash changes.** The canonical type grammar — the string a schema hashes to — was cleaned
up so that two implementations of versionable can compute the same hash for the same schema. The file format is
untouched and old files still load; what changes is the `hash="..."` literal every `Versionable` subclass declares.

**Upgrading.** Import your modules and let the hash check tell you the new values: each class raises `HashMismatchError`
at definition time with the message `declared 'abc123', computed 'def456'. Update the hash parameter to 'def456'.` Fix
one, re-import, repeat. To collect them all in one pass instead of one per import, call
`versionable.ignoreHashErrors(True)` first — every mismatch is then logged as a warning carrying the same computed hash,
and nothing raises.

Grammar changes, all of which move hashes:

- Enums, converter-backed types, and any type reached by fallthrough render as a **bare class name** instead of a
  module-qualified path, so moving a class between files no longer changes a hash. Names must now be unique across a
  schema's reachable types; duplicates are rejected at class definition. Override with `VERSIONABLE_NAME`,
  `registerConverter(name=...)`, or `setSerializationName()`.
- numpy arrays render as `ndarray[<dtype>]` against a closed dtype-token table rather than whatever `np.dtype.name`
  returns, which drifted between numpy 1.x and 2.x. The dtype is now enforced at runtime as well as hashed: safe casts
  are applied per element and unsafe ones raise `DtypeMismatchError`, on lazy HDF5 loads included
  ([ADR-0002](docs/adr/0002-array-dtype-hash-significant.md)).
- `Literal` values render unprefixed, with strings quoted and integers bare, so `Literal['1']` and `Literal[1]` are now
  distinct schemas. Enum members take precedence over their values. `Literal` members that are not `str`, `int`, `bool`,
  `None`, or an enum member are rejected at class definition.
- Variadic tuples render `tuple[T, ...]`, and non-container types never carry type parameters.
- Self-referential annotations resolve at class definition, so a self-referencing class hashes to what the grammar says
  it should. Annotations that still cannot resolve — mutual recursion between two classes — fall back to their raw
  source text and now warn that the resulting hash may not be reproducible by another implementation.

Also in this release:

- **A C# implementation.** The `Versionable` NuGet package is the same library for .NET: same canonical grammar, same
  file format, same migration semantics, and files written by either implementation load in the other. A Roslyn analyzer
  validates each type's declared hash at compile time and a source generator emits its metadata, so the runtime never
  reflects over user types and supports Native AOT and trimming
  ([ADR-0003](docs/adr/0003-csharp-compile-time-validation-codegen.md)). The two packages share their `major.minor`:
  matching that digit pair means interchangeable files ([ADR-0004](docs/adr/0004-shared-generation-versioning.md)). See
  [`dotnet/README.md`](dotnet/README.md), including its HDF5 compression notes — gzip interchanges everywhere, some
  other filters do not.
- `conformance/` holds the cross-language contract: the grammar specification, hash vectors, and a golden corpus of
  files with per-fixture manifests. CI reads it from both languages and, on every pull request, has each
  implementation's writers checked against the other's readers.
- `python -m versionable.tools.to_csharp` scaffolds a C# `Versionable` type from an existing Python one in one shot
  ([docs/to-csharp.md](docs/to-csharp.md)).

## 0.2.1 (2026-05-08)

- Nested `Versionable` dataclasses with a field named `object`, `version`, `hash`, `format`, `format_be`, or
  `shared_refs` now load correctly across all backends. Previously these field names collided with the internal
  envelope-stripping logic and got silently dropped during deserialization, raising
  `TypeError: missing required argument` when reconstructing the instance.

## 0.2.0 (2026-05-05)

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
- The class-level `validate_literals` setting is now honored at every nesting level — each nested class's declaration
  governs its own Literal fields independently of any enclosing class. Previously `validate_literals` was silently
  ignored on nested values.
- **Breaking change:** removed the `validateLiterals` kwarg from `versionable.load()`. The class-level
  `validate_literals=False` (or `literalFallback("...")` for individual fields) covers the cases the load-time override
  addressed; the override turned out to be redundant once the class-level setting actually reached nested boundaries.
- Save-side guard: `dict[Versionable, X]` now raises `ConverterError` at save time. Dict keys can't carry envelope
  information and previously round-tripped as Python repr strings.
- `versionable.load(..., upgradeInPlace=True)` is now actually honored. Previously the flag was accepted but silently
  dropped at the root migration call, so `requiresUpgrade()` migrations always raised `UpgradeRequiredError`. The flag
  now reaches both root and nested migrations.
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
- TOML backend: switched the underlying library from `toml` (unmaintained since 2020) to
  [`tomlkit`](https://github.com/python-poetry/tomlkit). File format is unchanged; output formatting may differ
  byte-wise (whitespace, quote style). The `commentDefaults=True` code path is reimplemented on top of tomlkit's
  structural document API, with commented-out default lines now placed alongside their parent table for cleaner
  uncomment-to-override workflows. Round-trip preservation of user-added comments is not yet supported (planned for a
  follow-up).

## 0.1.0

First stable release of **versionable**.

- Versioned persistence for Python dataclasses with schema hash validation and declarative migrations
- JSON, TOML, YAML, and HDF5 backends — base install requires only numpy
- Rich type support: numpy arrays, datetime, enums, nested Versionable types, and more
- Save-as-you-go HDF5 sessions for incremental writes and random access on large files
