[//]: # (vim: set ft=markdown:)

# Nested Migrations and Polymorphic Deserialize

- **Created:** 2026-05-01
- **Last updated:** 2026-05-01
- **Status:** In progress
- **Branch:** `feature/nested-migrations`
- **Targets release:** 0.2.0

## Problem

Two related correctness bugs in nested deserialize, both hidden today by the lack of nested-migration and
nested-polymorphism test coverage:

**1. Migrations don't recurse.** Migrations are applied only at the **root** of a load. A nested `Versionable`
value (e.g. `B` at field `A.point`, or each element of `list[B]`, or values of `dict[K, B]`) is deserialized
via `_types.py::_deserializeVersionable` using the **current** schema, with no migration step. If `B`'s schema
has changed between save time and load time, the nested data fails to deserialize — or worse, silently
corrupts because field names happen to line up.

**2. Polymorphism doesn't work.** `_deserializeVersionable(data, cls)` always constructs `cls`, ignoring the
envelope's `object` name. So `list[Animal]` saved with `Dog` and `Cat` elements round-trips as a list of
`Animal` instances — the subclass identity is lost. Save records `object="Dog"` correctly; load just doesn't
honor it.

These two bugs interact. Fixing migrations alone would make polymorphism *worse* — `Animal.Migrate` would run
on data that was saved with `Dog`'s schema. So they're fixed together: the deserializer resolves the actual
class from the envelope's `object` first, then runs that class's migrations.

### Empirical demonstration

```python
from dataclasses import dataclass
from pathlib import Path
import tempfile
import versionable
from versionable import Versionable, Migration
from versionable._base import _REGISTRY

with tempfile.TemporaryDirectory() as td:
    p = Path(td) / "data.yaml"

    @dataclass
    class B_v1(Versionable, version=1, name="NestedTest", register=True):
        title: str

    @dataclass
    class A_v1(Versionable, version=1, name="Container", register=True):
        items: list[B_v1]

    versionable.save(A_v1(items=[B_v1(title="hello"), B_v1(title="world")]), p)

    _REGISTRY.pop("NestedTest", None)
    _REGISTRY.pop("Container", None)

    @dataclass
    class B_v2(Versionable, version=2, name="NestedTest", register=True):
        name: str

        class Migrate:
            v1 = Migration().rename("title", "name")

    @dataclass
    class A_v1_with_v2_B(Versionable, version=1, name="Container", register=True):
        items: list[B_v2]

    print(versionable.load(A_v1_with_v2_B, p))
```

Today this raises `TypeError: B_v2.__init__() missing 1 required positional argument: 'name'`. After this PR,
it prints `Container(items=[B_v2(name='hello'), B_v2(name='world')])`.

## Goal

For every nested `Versionable` encountered during deserialize:

1. Read the envelope from the data dict.
2. **Resolve the actual class** from the envelope's `object` name (polymorphism). Falls back to the declared
   field type when the envelope has no object name or the envelope name matches the declared class.
3. Extract the source version.
4. If `fileVersion < actualCls.version`: apply migrations on `actualCls` from `fileVersion` to
   `actualCls.version`, then deserialize fields.
5. If `fileVersion > actualCls.version`: raise `VersionError` identifying the nested type.
6. If `fileVersion == actualCls.version`: deserialize as today (no migration step).
7. If envelope is missing: log a warning and assume the current version of the declared type.

## Design

### Where the recursion lives

Inside `_deserializeVersionable(data, cls)` in `src/versionable/_types.py`. Every nested `Versionable` —
direct field, list/tuple/set element, dict value — funnels through this function via
`_deserializeConcrete` (see `_types.py:464-465`). Centralizing migration + polymorphism logic here covers
the entire nested space without touching `_deserializeSequence` / `_deserializeDict`.

### Polymorphic class resolution

`_deserializeVersionable(data, cls)` accepts a *declared* type `cls`. The actual class to construct is
derived from the envelope's `object` name plus `cls`:

```python
def _resolveNestedClass(envelope: dict[str, Any], cls: type[Versionable]) -> type[Versionable]:
    """Resolve the concrete class for a nested deserialize using the envelope's object name."""
    objectName = envelope.get("object")
    declaredName = cls._serializer_meta_.name if hasattr(cls, "_serializer_meta_") else cls.__name__

    # Back-compat: missing envelope or no object key — fall back to declared type
    if not objectName:
        return cls

    # Same name — declared type is the right one (handles register=False classes)
    if objectName == declaredName:
        return cls

    # Look up in registry
    fromRegistry = _REGISTRY.get(objectName)
    if fromRegistry is None:
        raise BackendError(
            f"Unknown nested object type {objectName!r}. "
            f"Class is not registered or has been removed."
        )

    if not issubclass(fromRegistry, cls):
        raise BackendError(
            f"Nested object type {objectName!r} resolves to {fromRegistry.__qualname__}, "
            f"which is not a subclass of declared type {cls.__qualname__}."
        )

    return fromRegistry
```

Resolution rules:

| Condition                                                  | Result                                                |
| ---------------------------------------------------------- | ----------------------------------------------------- |
| Envelope has no `object` (or envelope missing entirely)    | Use declared `cls` (back-compat with envelope-less)   |
| `object` matches `cls`'s registered name                   | Use `cls` (handles `register=False`)                  |
| `object` is in `_REGISTRY` and resolves to a subclass of `cls` | Use the resolved class                            |
| `object` is in `_REGISTRY` but not a subclass of `cls`     | `BackendError` (file has wrong type)                  |
| `object` is not in `_REGISTRY` and doesn't match `cls`     | `BackendError` (unknown type)                         |

Subclasses still load correctly when no polymorphism is involved (`object == cls.name`). The `old_names`
mechanism is honored automatically: a renamed class registers itself under both its current and old names,
so a lookup for the old name finds the current class.

### Reading the nested envelope

The data dict for a nested Versionable can have its envelope in three shapes:

1. **Wrapped (current 0.2.0+):** `data["__versionable__"] = {"object": ..., "version": N, "hash": ...}`
2. **Flat new keys (transient, not emitted on disk):** `data` has top-level `"object"`, `"version"`, `"hash"`.
3. **Flat dunder keys (0.1.x files):** `data` has top-level `"__OBJECT__"`, `"__VERSION__"`, `"__HASH__"`.

A small helper extracts the version (and object/hash for diagnostics) regardless of shape:

```python
def _readNestedEnvelope(data: dict[str, Any]) -> dict[str, Any]:
    """Extract version/object/hash from a nested Versionable's data dict."""
    if isinstance(data.get("__versionable__"), dict):
        envelope = data["__versionable__"]
    else:
        envelope = data
    return {
        "object": envelope.get("object", envelope.get("__OBJECT__")),
        "version": envelope.get("version", envelope.get("__VERSION__")),
        "hash": envelope.get("hash", envelope.get("__HASH__")),
    }
```

### Strip-then-migrate-then-deserialize

`applyMigrations(data, migrations)` operates on field-name keys. The data dict passed to a nested
deserializer also contains envelope keys. We strip envelope keys before migration:

```python
_ENVELOPE_KEYS = frozenset({"__versionable__", "__OBJECT__", "__VERSION__", "__HASH__",
                            "object", "version", "hash", "__FORMAT__", "format"})

# In _deserializeVersionable (sketch):
fileVersion = _readNestedEnvelope(data)["version"]
meta = cls._serializer_meta_
if fileVersion is None:
    logger.warning(...)  # missing nested envelope
    fields = _stripEnvelope(data)
elif fileVersion < meta.version:
    fields = _stripEnvelope(data)
    fields = _applyMigrations(cls, fields, fileVersion, meta.version)
elif fileVersion > meta.version:
    raise VersionError(f"Nested {meta.name}: file version {fileVersion} > class version {meta.version}.")
else:
    fields = _stripEnvelope(data)
# continue with the existing deserialize loop, using `fields` instead of `data`
```

The strip predicate uses an explicit allow-list (the keys above) — not a `startswith("__")` check —
because user data fields can technically begin with underscores even though `_resolveFields` already
filters them out at the dataclass level.

### Lazy field interaction (HDF5)

`_deserializeVersionable` pops `__ver_lazy__` early. Migration runs *after* the lazy pop, so lazy
sentinels are not exposed to migration ops. Lazy array migrations are out of scope (the array data
isn't loaded yet); the dataclass field holding the lazy sentinel still gets migrated like any other
field name.

### `assumeVersion` interaction

`assumeVersion` at `versionable.load(...)` only applies to the root. For nested elements with
missing envelopes, log a warning and assume the current class version. Different classes have
different version spaces; threading `assumeVersion` through nesting would conflate them. Document
this in the `versionable.load` docstring and `migrations.md`.

### Imperative migrations

`_applyMigrations` already handles both `Migration` and `_ImperativeMigration`. No special handling
needed at the nested layer — confirm via test.

### `validateLiterals` propagation (fix bug-adjacent)

Today `versionable.load(..., validateLiterals=False)` only affects the root deserialize. Nested
Versionables fall back to each class's own `meta.validateLiterals`, so the load-time override is
silently dropped at nesting boundaries. Fix: thread `validateLiterals` through `deserialize` →
`_deserializeConcrete` → `_deserializeVersionable` (and through `_deserializeSequence` /
`_deserializeDict`) so the override applies uniformly.

`_deserializeVersionable` chooses: if a value is passed in, use it; else fall back to the class's
own `meta.validateLiterals`. The root loader's behavior (`validateLiterals if validateLiterals is
not None else meta.validateLiterals`) is preserved.

### Unknown-field handling at nested level

Today `unknown="error"`, `"ignore"`, `"preserve"` are only honored at the root.
`_deserializeVersionable` ignores them — extra fields in nested data are silently dropped (because
field iteration only pulls declared names). Fix: after migration, compute the unknown set against
the resolved class's fields and apply the same root-level policy:

```python
unknownFields = set(fields_dict.keys()) - knownFieldNames
if unknownFields:
    if meta.unknown == "error":
        raise UnknownFieldError(f"Unknown fields in nested {meta.name}: {sorted(unknownFields)}")
    if meta.unknown == "ignore":
        for name in unknownFields:
            del fields_dict[name]
    # "preserve" mode: leave them; field iteration won't pick them up
```

Each nested class's own `unknown` setting governs its own data.

### Save-side guard: no Versionable dict keys

Polymorphism at dict *values* is supported (`dict[str, Animal]` works). Polymorphism at dict
*keys* (`dict[Animal, str]`) is not — keys serialize via `str(k)` (see `_types.py:311`), which
silently produces a Python repr for Versionable instances and round-trips incorrectly. Today this
fails confusingly on load. After this PR, `serialize()` raises `ConverterError` at save time when
a dict key type is a Versionable subclass:

```python
if isinstance(value, dict):
    keyType = args[0] if args else Any
    if isinstance(keyType, type) and issubclass(keyType, Versionable):
        raise ConverterError(
            f"Dict keys cannot be Versionable types ({keyType.__name__}). "
            f"Use Versionable as dict values, not keys."
        )
    ...
```

## Touchpoints

### Source

| File                      | What                                                                       |
| ------------------------- | -------------------------------------------------------------------------- |
| `_types.py`               | `_readNestedEnvelope`, `_stripEnvelope`, `_resolveNestedClass`, polymorphism + migration + unknown-field handling in `_deserializeVersionable`. Thread `validateLiterals` through `_deserializeConcrete`/`_deserializeSequence`/`_deserializeDict`. Save-side dict-key Versionable guard. |
| `_api.py`                 | Move `_applyMigrations` (or expose it in a way `_types.py` can import without circularity). |
| `_migration.py`           | Receives the moved `_applyMigrations` helper.                              |

`_applyMigrations` currently lives in `_api.py` and imports from `_migration`. `_types.py` cannot
import from `_api.py` (circular — `_api.py` already imports `deserialize` from `_types.py`).
Cleanest fix: move the helper to `_migration.py` (which has no `_types.py` dependency), then import
it from both `_api.py` and `_types.py`.

### Tests

New file: `tests/test_nested_migrations.py`. Existing `tests/test_migration.py` is root-only — keep
unchanged.

### Docs

| File              | Change                                                                       |
| ----------------- | ---------------------------------------------------------------------------- |
| `migrations.md`   | New "Nested migrations" section explaining recursion at every level.         |
| `reference.md`    | Note in the deserialize/load reference that migrations apply recursively.    |
| `CHANGELOG.md`    | 0.2.0 entry: nested migrations + missing-envelope warning behavior.          |

## Test matrix

Place all tests in `tests/test_nested_migrations.py`. Use snake_case identifiers per project
convention. Hardcode hashes as string literals (compute once via `MyClass.hash()` in a REPL, then
paste).

### Migration recursion

| # | Scenario | Backends |
|---|----------|----------|
| 1 | Single-field nested: `A.point: B`, save B-v1, load B-v2 + rename migration | JSON, YAML, TOML, HDF5 |
| 2 | `list[B]` element migrations | JSON |
| 3 | `dict[str, B]` value migrations | JSON |
| 4 | `tuple[B, ...]` element migrations | JSON |
| 5 | `set[B]` element migrations (use hash-stable `B`) | JSON |
| 6 | Multi-step chain at nested level: B v1 → v2 → v3 | JSON |
| 7 | Both parent and child migrated: A-v1 → A-v2 + B-v1 → B-v2 | JSON |
| 8 | Newer nested version raises `VersionError` mentioning the nested type | JSON |
| 9 | Missing nested envelope: logs a warning, assumes current version | JSON |
| 10 | Imperative `@migration` decorator on nested type | JSON |
| 11 | Three-level deep migration: A.b: B, B.c: C, all at v1 → all at v2 | JSON |
| 12 | Cross-backend: save YAML at v1, load TOML at v2 | YAML → TOML |
| 13 | Nested migration on HDF5 (covers the `loadLazy` path) | HDF5 |

### Polymorphism

| # | Scenario | Backends |
|---|----------|----------|
| 14 | `list[Animal]` saved with `[Dog, Cat]` round-trips with subclass identity preserved | JSON, HDF5 |
| 15 | Polymorphism + migration: `Dog` v1 → v2 applied, `Cat` v1 → v2 applied | JSON |
| 16 | Unknown nested object name raises `BackendError` | JSON |
| 17 | Resolved class is not a subclass of declared type → `BackendError` | JSON |
| 18 | `old_names` covers polymorphism rename: file says `object="OldDog"`, current class registers `name="Dog", old_names=["OldDog"]` | JSON |

### Save-side guards

| # | Scenario | Backends |
|---|----------|----------|
| 19 | Saving `dict[Versionable, X]` raises `ConverterError` | (save side; backend-agnostic) |

### `validateLiterals` propagation

| # | Scenario | Backends |
|---|----------|----------|
| 20 | `load(..., validateLiterals=False)` skips Literal validation in nested Versionables | JSON |

### Unknown-field handling at nested

| # | Scenario | Backends |
|---|----------|----------|
| 21 | Nested class with `unknown="error"` raises `UnknownFieldError` on extra field | JSON |
| 22 | Nested class with `unknown="ignore"` silently drops extras | JSON |

JSON is the representative backend for tests where dispatch is backend-agnostic; the per-backend
coverage in tests 1, 12, 13, 14 confirms each backend delivers nested envelopes in a shape the
recursion can read.

## Backwards compatibility

- 0.1.x files with **flat dunder envelopes** at the nested level continue to load — the envelope
  reader handles all three shapes.
- 0.1.x files with **no nested envelope** (pre-envelope-keys format, or hand-crafted) load with a
  warning and assume the current version. This matches root-level behavior. No file in the wild
  should be affected — 0.1.x files always wrote envelopes.
- Existing root-only migration tests (`tests/test_migration.py`) continue to pass unchanged.

## Acceptance

- `pixi run --environment default cleanup && pixi run --environment default pytest` green.
- The empirical demonstration above prints
  `Container(items=[B_v2(name='hello'), B_v2(name='world')])` instead of raising `TypeError`.
- Test matrix above is covered.
- Plan doc committed at `docs/plans/nested-migrations.md`.
- `CHANGELOG.md` 0.2.0 entry mentions nested migrations, polymorphism, `validateLiterals`
  propagation, nested unknown-field handling, and the dict-keys save-side guard.
- `docs/migrations.md` and `docs/reference.md` updated to state migrations and polymorphism apply
  recursively.

## Risk

Medium. The change touches the central deserialize loop, which is exercised by every
`load()` call across all backends. Mitigations:

- Test matrix covers each backend separately at the nested level.
- The fast path (file version equals class version, envelope object matches declared type) is
  unchanged: same field iteration, same envelope-key skipping. Only the migration and polymorphism
  branches are new.
- Strip-and-migrate is keyed off explicit envelope key names, not heuristics.
- Polymorphism resolution falls back to the declared type when the envelope is missing or matches
  the declared name — pre-existing files keep loading.

## Out of scope

- **Load-time hash validation.** Today the file's recorded hash is read into `fileMeta` but never
  compared to the class's hash. Adding it (uniformly at root and nested) is a separate hardening.
- **Polymorphic dict keys** (`dict[Animal, str]`). Save raises `ConverterError` (in scope for this
  PR — see save-side guard above), but the underlying support is not added: dict keys serialize
  via `str(k)` and don't carry envelope information.
- **`upgradeInPlace=True` for nested `_RequiresUpgradeOp`.** The flag rewrites the root file once,
  which already covers all nested data. A nested `requiresUpgrade()` op fires the same root-level
  rewrite when the flag is set; no nested-specific propagation needed. (Verify with a test.)
- **`assumeVersion` propagation to nested.** Different classes have different version spaces;
  threading one int through doesn't compose. Nested missing-envelope falls back to the resolved
  class's current version with a warning.
- **Polymorphism for `register=False` classes.** Without registry presence, the resolver can't
  find the class by name. The declared field type is used as a fallback only when names match.
  Documented as a limitation in `migrations.md`.
- **Polymorphism rename without `old_names`.** Old files with the previous name produce
  `BackendError` (clear failure, not silent corruption). Documented as a limitation.
- **The envelope-deduplication optimization** tracked in #28.
- **HDF5 native lazy array migrations.** Lazy fields aren't loaded at load time, so migrating
  their values doesn't apply. The dataclass field holding the lazy sentinel still gets migrated
  like any other field name.

## Suggested commit order

1. Plan doc (this file).
2. Move `_applyMigrations` from `_api.py` to `_migration.py`; update import in `_api.py`. Verify
   tests still pass.
3. Thread `validateLiterals` through `_deserializeConcrete` / `_deserializeSequence` /
   `_deserializeDict` / `_deserializeVersionable`. Verify tests still pass.
4. Implement `_readNestedEnvelope`, `_stripEnvelope`, `_resolveNestedClass`, the migration call,
   and nested unknown-field handling in `_deserializeVersionable`.
5. Add the save-side `dict[Versionable, X]` guard.
6. Add `tests/test_nested_migrations.py`.
7. Update `docs/migrations.md`, `docs/reference.md`.
8. Add `CHANGELOG.md` entry.
9. `pixi run --environment default cleanup` — fix anything reported.
10. Show staged diff + commit message for approval, then commit + push.
