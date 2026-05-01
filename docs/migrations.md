# Migrations

Serialized files are long-lived. A config written today may be loaded by code six months from now, after fields have
been renamed, added, or removed. Without a migration story, you'd be forced to either keep old field names forever, or
break all existing files on every schema change.

**`versionable`** solves this with versioned migrations. When you change a schema, you:

1. Bump `version`
2. Recompute `hash` (via `MyClass.hash()`)
3. Add a `Migrate` inner class that tells **`versionable`** how to transform old data into the new shape

When `load()` reads a file, it checks `version` in the `__versionable__` metadata envelope. If it's behind the current
version, migrations are applied in order — v1 → v2 → v3 — before the object is constructed.

## Declarative Operations

The examples below all evolve the same `WorkerConfig` class through a series of changes. Each section adds one migration
step on top of the previous version.

**v1 — starting schema:**

```python
from dataclasses import dataclass
from versionable import Versionable, Migration

@dataclass
class WorkerConfig(Versionable, version=1, hash="5556c8"):
    title: str
    debug: bool
    retries: int = 3
```

A v1 file on disk:

```yaml
title: batch-processor
debug: false
retries: 5
__versionable__:
  object: WorkerConfig
  version: 1
  hash: 5556c8
```

### Rename a Field

In v2, `title` is renamed to `name`:

```python
@dataclass
class WorkerConfig(Versionable, version=2, hash="ed3a90"):
    name: str           # renamed from "title"
    debug: bool
    retries: int = 3

    class Migrate:
        v1 = Migration().rename("title", "name")
```

`versionable.load(WorkerConfig, "config.yaml")` reads the v1 file, applies the rename, and returns
`WorkerConfig(name="batch-processor", debug=False, retries=5)`.

### Drop a Field

In v3, `debug` is removed. Old files still carry it, so we drop it explicitly:

```python
@dataclass
class WorkerConfig(Versionable, version=3, hash="beb912"):
    name: str
    retries: int = 3

    class Migrate:
        v1 = Migration().rename("title", "name")
        v2 = Migration().drop("debug")
```

A v1 file now goes through both migrations: `title` → `name`, then `debug` is discarded.

### Add a Field

When a new field has a dataclass default, no migration and no version bump are needed — **`versionable`** fills in the
default automatically for any file that doesn't have the field. You only need to update the hash:

```python
# still version=3 — only the hash changes because the fields changed

@dataclass
class WorkerConfig(Versionable, version=3, hash="ea7fc2"):
    name: str
    retries: int = 3
    timeout_s: float = 30.0  # not in older files — default fills in automatically

    class Migrate:
        v1 = Migration().rename("title", "name")
        v2 = Migration().drop("debug")
```

Bump the version (and add a migration) only when the value you want old files to receive differs from the dataclass
default. For example — `timeout_s = 30.0` is a sensible default for new configs, but old worker files predate the
concept of a timeout entirely, so you want them to load as `0.0` (meaning "no timeout limit") rather than silently
imposing a 30-second limit on them:

```python
@dataclass
class WorkerConfig(Versionable, version=4, hash="ea7fc2"):
    name: str
    retries: int = 3
    timeout_s: float = 30.0  # default for new configs

    class Migrate:
        v1 = Migration().rename("title", "name")
        v2 = Migration().drop("debug")
        v3 = Migration().add("timeout_s", default=0.0)  # old files get 0.0, not 30.0
```

### Convert a Field's Value

Use `convert` when a field is kept but its unit or representation changes. Back to `WorkerConfig` — in v5 we rename
`timeout_s` to `timeout_ms` and store it as an integer milliseconds value:

```python
@dataclass
class WorkerConfig(Versionable, version=5, hash="aac8a2"):
    name: str
    retries: int = 3
    timeout_ms: int = 30000  # was timeout_s: float in v4

    class Migrate:
        v1 = Migration().rename("title", "name")
        v2 = Migration().drop("debug")
        v3 = Migration().add("timeout_s", default=0.0)
        v4 = Migration().rename("timeout_s", "timeout_ms").convert("timeout_ms", via=lambda s: int(s * 1000))
```

- A **v3** file with `timeout_s = 5.0` loads as `timeout_ms = 5000`.
- A **v4** file with `timeout_s = 1.5` loads as `timeout_ms = 1500`.
- A **v2** file (which has no `timeout_s`) gets `timeout_s = 0.0` injected by the **v3** migration, then converted to
  `timeout_ms = 0`.

## Chaining Operations

Multiple operations can be chained on a single `Migration` object:

```python
class Migrate:
    v1 = Migration().rename("title", "name").drop("debug")
```

For more complex histories, use `.then()` to link two separate migration objects:

```python
class Migrate:
    v2 = Migration().drop("debug")
    v1 = Migration().rename("title", "name").then(v2)
```

This is equivalent to declaring `v1` and `v2` separately — choose whichever reads more clearly for your use case.

## Multi-Version Chains

The full `WorkerConfig` history in one place — **`versionable`** applies every migration from the file's version up to
the class's current version, in ascending order:

```python
@dataclass
class WorkerConfig(Versionable, version=5, hash="aac8a2"):
    name: str
    retries: int = 3
    timeout_ms: int = 30000

    class Migrate:
        v1 = Migration().rename("title", "name")
        v2 = Migration().drop("debug")
        v3 = Migration().add("timeout_s", default=0.0)
        v4 = Migration().rename("timeout_s", "timeout_ms").convert("timeout_ms", via=lambda s: int(s * 1000))
        # no v5 needed — timeout_ms default (30000) is sufficient for files without the field
```

| File version | Migrations applied        |
| ------------ | ------------------------- |
| v1           | `v1` → `v2` → `v3` → `v4` |
| v2           | `v2` → `v3` → `v4`        |
| v3           | `v3` → `v4`               |
| v4           | `v4` only                 |
| v5           | none (already current)    |

## Derive from Another Field

Use `derive` when a schema refactor splits or restructures an existing field into a new one. Rather than requiring users
to re-export their data, you compute the new field's value directly from what is already in the file.

A common case: v1 stored all sensor data in a single `raw_data` matrix with timestamps packed into the first column. In
v2, timestamps are promoted to their own field for easier access. The `derive` migration extracts them from the old
matrix so that existing v1 files load correctly into the new schema without any manual intervention:

```python
# v1 — timestamps were the first column of raw_data
@dataclass
class Recording(Versionable, version=1, hash="c3a812"):
    name: str
    raw_data: npt.NDArray[np.float64]  # shape (N, M) — first column is timestamps


# v2 — timestamps promoted to their own field
@dataclass
class Recording(Versionable, version=2, hash="d0155b"):
    name: str
    timestamps: npt.NDArray[np.float64]  # new in v2; extracted from the first column of raw_data
    raw_data: npt.NDArray[np.float64]    # still present — derive keeps the source field by default

    class Migrate:
        v1 = Migration().derive("timestamps", from_="raw_data", via=lambda d: d[:, 0])
```

If the source field was removed in the new schema, chain a `drop` to clean it up:

```python
v1 = Migration().derive("timestamps", from_="raw_data", via=lambda d: d[:, 0]).drop("raw_data")
```

## Renaming a Class

When you rename a `Versionable` class, existing files on disk still contain the old name as the `object` attribute in
their `__versionable__` metadata. Use `old_names` to register the old name(s) so those files can still be loaded:

```python
# Was previously called "SensorReading"
@dataclass
class Measurement(
    Versionable, version=1, hash="...", name="Measurement", old_names=["SensorReading"]
):
    timestamp: datetime
    value: float
```

With this declaration:

- New files are saved with `object: "Measurement"`
- Files saved with `object: "SensorReading"` can still be loaded via `loadDynamic()`
- Multiple old names are supported: `old_names=["SensorReading", "DataPoint"]`

If another class already owns one of the old names, class definition raises `VersionableError` instead of silently
reusing or overwriting that registry entry.

## Imperative Migrations

When the transformation involves branching logic or multiple fields at once, use the `@migration` decorator. Continuing
the `WorkerConfig` story — suppose v2 had a `mode` field that controlled how `retries` was interpreted, and v3 folds
that logic in and drops `mode`:

```python
from versionable import migration, MigrationContext

@dataclass
class WorkerConfig(Versionable, version=3, hash="beb912"):
    name: str
    retries: int = 3

    class Migrate:
        v1 = Migration().rename("title", "name")

        @migration(fromVersion=2)
        def from_v2(ctx: MigrationContext) -> None:
            # "aggressive" mode stored retries as a multiplier — convert back to absolute count
            if ctx["mode"] == "aggressive":
                ctx["retries"] = ctx["retries"] * 10
            ctx.drop("mode")  # mode no longer exists in v3
```

`MigrationContext` behaves like a mutable dict over the raw deserialized data. Read and write fields by key; call
`ctx.drop(key)` to remove a field that was deleted in the new version.

## Nested Migrations

Migrations apply at **every** level of the object graph, not just the root. A `Versionable` value appearing as a direct
field, a list/dict/tuple/set element, or a nested field of a nested field gets migrated using its own class's `Migrate`
chain when its file version differs from the class's current version.

```python
@dataclass
class Address(Versionable, version=2, hash="..."):
    street: str  # renamed from "addr"
    city: str

    class Migrate:
        v1 = Migration().rename("addr", "street")

@dataclass
class Person(Versionable, version=1, hash="..."):
    name: str
    addresses: list[Address]
```

Loading a file saved with `Address` v1 inside a `Person` reads each nested address's envelope, applies
`Address.Migrate.v1`, then deserializes the migrated fields. The same applies through any level — a nested field of a
nested field migrates just as smoothly.

Migration recursion works for direct fields, `list[B]`, `dict[K, B]`, `tuple[B, ...]`, and `set[B]` (where `B` is
hashable). Each element gets its own envelope read and migration step.

A few corner cases:

- **Newer nested version.** If the file's nested version exceeds the class's current version, `load()` raises
  `VersionError` identifying the nested type. (The framework can't downgrade.)
- **Missing nested envelope.** If a nested data dict has no envelope at all (e.g., a hand-crafted file or an older
  format), `load()` logs a warning naming the nested type and assumes the class's current version.
- **Imperative migrations** (`@migration`-decorated functions) work at every nested level, the same way declarative
  `Migration` objects do.

## Polymorphic Collections

`list[Animal]` saved with subclass instances — `Dog`, `Cat` — round-trips with subclass identity preserved. The
per-element envelope's `object` name drives class lookup at load time:

```python
@dataclass
class Animal(Versionable, version=1, hash="..."):
    name: str

@dataclass
class Dog(Animal, version=1, hash="..."):
    breed: str

@dataclass
class Cat(Animal, version=1, hash="..."):
    indoor: bool

@dataclass
class Zoo(Versionable, version=1, hash="..."):
    animals: list[Animal]

zoo = Zoo(animals=[Dog(name="Rex", breed="lab"), Cat(name="Whiskers", indoor=True)])
versionable.save(zoo, "zoo.json")

loaded = versionable.load(Zoo, "zoo.json")
assert isinstance(loaded.animals[0], Dog)  # subclass preserved
assert loaded.animals[0].breed == "lab"
```

The resolver looks the per-element `object` name up in the global registry (the same one used by `loadDynamic`). Two
error cases:

- **Unknown name.** The file's `object` is not in the registry → `BackendError`. Common cause: the class was deleted or
  renamed without `old_names`.
- **Wrong subclass.** The resolved class is registered but is not a subclass of the declared field type →
  `BackendError`. The file is malformed or you're loading the wrong file.

Both errors identify the nested type and the declared field type so the failure points at the right place.

`old_names` works across polymorphism, too: rename `Dog` to `Puppy` with `old_names=["Dog"]`, and old files with
`object="Dog"` resolve to the new class. Each subclass migrates against its own `Migrate` chain — `Dog`'s migration runs
for `Dog` elements, `Cat`'s for `Cat` elements.

### Limitations

- **Polymorphic dict keys.** `dict[Animal, X]` is rejected at save time with `ConverterError`. Dict keys serialize via
  `str(k)` and can't carry envelope information; use `Animal` as a dict value, not a key.
- **`register=False` polymorphism.** If `Dog` opts out of the registry, the resolver can't find it by name and falls
  back to the declared field type. Polymorphism through a base class field requires every concrete subclass to be
  registered.
