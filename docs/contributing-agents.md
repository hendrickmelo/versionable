# versionable — Contributor Reference for Agents

What an agent changing the library needs beyond the public API. Conventions, build tasks and design decisions are in the
repo's `AGENTS.md`; the public API, supported types, migrations and backends are in [AGENT.md](AGENT.md). This file
holds only what neither of those covers.

## Serialization envelope

**Reserved metadata keys (cannot be field names):** `__versionable__` (envelope wrapper), `__ver_ndarray__`,
`__ver_json__`. Inside the `__versionable__` envelope: `object`, `version`, `hash`, `format`. The 0.1.x dunder forms
(`__OBJECT__`, `__ndarray__`, `__json__`, etc.) are still accepted on read for back-compat.

**numpy arrays:** HDF5 stores them natively as compressed datasets with lazy loading. JSON, TOML and YAML store a
base64-compressed npz blob.

## Migration internals

Migrations apply sequentially: a file at v1 loaded into a v5 class runs v1 → v2 → v3 → v4 → v5. Each step is the
`Migrate.vN` entry for the version it upgrades from; a missing step raises `MigrationError`. `.requiresUpgrade()` marks
a migration whose result must be written back to the file; applying it raises `UpgradeRequiredError` unless the caller
passed `upgradeInPlace=True` to `load()`. `.then(other)` appends another migration's operations to the same step.

## Error hierarchy

```text
VersionableError (base)
├── HashMismatchError      — declared hash != computed (raised at class definition)
├── VersionError           — file version newer than the class version (no downgrade)
├── MigrationError         — missing migration step, or a step that failed to apply
├── ArrayNotLoadedError    — accessing array loaded with metadataOnly=True (also AttributeError)
├── UpgradeRequiredError   — migration needs in-place file modification
├── UnknownFieldError      — unknown field in source data (only with unknown="error")
├── ConverterError         — type conversion failure
└── BackendError           — storage backend operation failure
```

## Schema hash

Computed from field names and canonical type names (sorted, SHA-256, first 6 hex chars) and validated at class
definition time; a mismatch raises `HashMismatchError` at import. Versionable types contribute their serialization
`name`, not their module path, so moving a class between modules does not change the hash. `MyClass.hash()` returns the
value to paste into `hash="..."`.

## Testing patterns

Shared fixtures live in `tests/conftest.py` (`SimpleConfig`, `WithOptional`, `WithArray`, `WithEnum`, `WithNested`,
`WithDatetime`, `WithList`, …). Every fixture class uses `register=False` so the global registry stays clean between
tests.

The standard test is construct → save to `tmp_path` → load → assert field equality; numpy arrays compare with
`np.testing.assert_array_equal()`. Identifiers inside test files are `snake_case`.

```bash
pixi run test                              # full suite
pixi run -- pytest -x                      # stop on first failure
pixi run -- pytest tests/test_json_backend.py
pixi run -- pytest -k "test_name"
```

CI runs the suite in the `default` environment (with HDF5) and the `minimal` one (without); `pixi run ci-all` does the
same locally.
