[//]: # (vim: set ft=markdown:)

# Envelope Keys: Drop Redundant Dunders

- **Created:** 2026-04-27
- **Last updated:** 2026-05-01
- **Status:** Implemented
- **Branch:** `feature/envelope-keys`
- **Targets release:** 0.2.0

## Problem

Today's metadata envelope uses SCREAMING_DUNDER keys *inside* a wrapper that is itself already namespaced as
`__versionable__`. The result reads like an unfriendly internal:

```yaml
__versionable__:
  __OBJECT__: MyConfig
  __VERSION__: 1
  __HASH__: 4b7866
name: probe-A
```

The wrapper is doing namespace work; the inner dunders are noise. They make hand-edited config files harder to scan
and don't match the lowercase / snake_case convention used elsewhere (YAML/TOML idioms, Python kwargs like
`skip_defaults`, `old_names`).

A separate but related cleanup: the user-data sentinels (`__ndarray__`, `__json__`, `__lazy__`) are *not* inside the
wrapper — they live in user values and need to remain visually distinct. But their bare names risk colliding with
anything a user might put in a dict. Re-namespacing them to `__ver_*__` keeps the dunder visibility and adds clear
package ownership.

## Goal

A single PR, lands in 0.2.0:

1. **Inside `__versionable__`, drop the dunders.** Lowercase for single words, snake_case for multi-word.
2. **Outside `__versionable__`, keep dunders but add a `__ver_*__` prefix** so user-data sentinels are
   unambiguously package-owned.
3. **Read both old and new keys** for one release cycle so 0.1.x files still load. Always write only new keys.

## Renames

### Inside `__versionable__` (drop dunders)

| Old              | New           | Notes                                                  |
| ---------------- | ------------- | ------------------------------------------------------ |
| `__OBJECT__`     | `object`      | Class serialization name                               |
| `__VERSION__`    | `version`     | Schema version                                         |
| `__HASH__`       | `hash`        | 6-char schema hash                                     |
| `__FORMAT__`     | `format`      | Reserved for future versionable file format versioning |
| `__FORMAT_BE__`  | `format_be`   | Reserved for future per-backend format versioning      |
| `__SHARED_REFS__`| `shared_refs` | PR 2 (0.3.0) — opt-in shared-reference encoding flag   |

`__FORMAT__`, `__FORMAT_BE__`, `__SHARED_REFS__` are **never written today**. The first two are reserved
constants in error messages; the third lives in the PR 2 plan doc. Renaming them is documentation-only for now —
no files in the wild contain them.

### User-data sentinels (keep dunders, add `__ver_*__` prefix)

| Old           | New                | Notes                                                |
| ------------- | ------------------ | ---------------------------------------------------- |
| `__ndarray__` | `__ver_ndarray__`  | Marks a JSON/TOML/YAML dict as an inline numpy array |
| `__json__`    | `__ver_json__`     | YAML/TOML wrapper for values without native encoding |
| `__lazy__`    | `__ver_lazy__`     | In-memory only (never on disk) — internal HDF5 reader marker |

Plus PR 2 sentinels (planned for 0.3.0, not implemented in this PR but spec'd in
`docs/plans/reference-handling.md`):

| Planned (old) | Planned (new)  | Notes                                                |
| ------------- | -------------- | ---------------------------------------------------- |
| `__ID__`      | `__ver_id__`   | Identity marker for shared references                |
| `__REF__`     | `__ver_ref__`  | Reference to an earlier `__ver_id__`                 |

### Stays as-is

- `__versionable__` (the wrapper key itself) — it *is* the namespace marker; the dunder is doing real work.
- Internal meta-dict keys passed between `_api.load` and backends (`__OBJECT__`, `__VERSION__`, `__HASH__` in the
  `tuple[fields, meta]` returned by `Backend.load`). Optional: rename these too for symmetry — see
  [Open questions](#open-questions). Default: leave alone, change is local to file format.

## Backwards compatibility

### Read

`Backend.load()` accepts both old and new keys, preferring new when both are present (no real file should have
both). Concretely, in each backend's `load`:

```python
metaTable = data.pop("__versionable__", {})
meta = {
    "__OBJECT__": metaTable.get("object", metaTable.get("__OBJECT__", "")),
    "__VERSION__": metaTable.get("version", metaTable.get("__VERSION__")),
    "__HASH__": metaTable.get("hash", metaTable.get("__HASH__", "")),
}
fileFormat = metaTable.get("format", metaTable.get("__FORMAT__"))
```

For user-data sentinels (`__ndarray__`, `__json__`), the converters / unwrappers accept both forms:

```python
if isinstance(value, dict):
    if "__ver_ndarray__" in value or "__ndarray__" in value:
        ...
```

### Write

Write only new keys. No flag, no opt-out — old-key writes don't exist after this PR.

### Deprecation timing

Read-both stays through the 0.2.x line. Drop old-key reads at 1.0, with a migration tool (or instructions to
re-save) for any straggler 0.1.x files. Tracked in `docs/plans/envelope-keys.md` (this file) under
[Drop schedule](#drop-schedule).

## Nested Versionable layout

Nested `Versionable` values get their own `__versionable__` envelope, just like the root. Before this change,
nested objects had envelope keys flat alongside data fields:

```json
"point": {"object": "Inner", "version": 1, "hash": "...", "x": 1.0, "y": 2.0}
```

After:

```json
"point": {"__versionable__": {"object": "Inner", "version": 1, "hash": "..."}, "x": 1.0, "y": 2.0}
```

Same shape across JSON, YAML, TOML, and HDF5. HDF5 already used a `__versionable__` child group at every level
— this change brings the text backends in line. For TOML the `toml` library emits this as a
`[parent.__versionable__]` sub-table, which is the canonical TOML form for the equivalent payload.

The deserialize path is structurally indifferent to envelope key location — `_deserializeVersionable` iterates
dataclass fields and pulls values; envelope keys (whether flat or wrapped) are skipped because they aren't
fields. So no read-side back-compat code is needed for the wrap.

## File-format examples (after the rename)

### JSON

```json
{
  "__versionable__": {
    "object": "MyConfig",
    "version": 1,
    "hash": "4b7866"
  },
  "name": "probe-A",
  "data": {
    "__ver_ndarray__": true,
    "dtype": "float64",
    "shape": [3],
    "data": "<base64>"
  }
}
```

### YAML

```yaml
__versionable__:
  object: MyConfig
  version: 1
  hash: 4b7866
name: probe-A
```

### TOML

```toml
[__versionable__]
object = "MyConfig"
version = 1
hash = "4b7866"

name = "probe-A"
```

### HDF5

```text
/__versionable__/
  attrs:
    object = "MyConfig"
    version = 1
    hash = "4b7866"
attrs (root):
  name = "probe-A"
```

## Touchpoints

### Source (8 files, ~70 sites)

| File                          | What                                                        |
| ----------------------------- | ----------------------------------------------------------- |
| `_json_backend.py`            | save: write new keys; load: read both                       |
| `_yaml_backend.py`            | save: write new keys; load: read both                       |
| `_toml_backend.py`            | save: write new keys; load: read both                       |
| `_hdf5_backend.py`            | save: write new attrs on `__versionable__` group; load: read both |
| `_hdf5_session.py`            | resume: read both; create: write new                        |
| `_types.py`                   | `__ndarray__` ↔ `__ver_ndarray__`; `__lazy__` ↔ `__ver_lazy__` |
| `_yaml_backend.py`, `_toml_backend.py` | `__json__` ↔ `__ver_json__` (wrappers)             |
| `_backend.py`, `_api.py`      | Comments / strings if any reference the old names           |

### Tests

Per-backend test fixtures (`test_json_backend.py`, `test_yaml_backend.py`, `test_toml_backend.py`,
`test_hdf5_backend.py`) and golden-file tests that hardcode the dunder strings. Plus new back-compat tests:

- For each backend: an old-format file (hand-rolled with `__OBJECT__` etc.) loads correctly.
- For each backend: an old-format file with `__ndarray__` / `__json__` loads correctly.
- Mixed old + new on the same file: prefer new (or error — see [Open questions](#open-questions)).

### Docs (in `docs/`)

- `reference.md` — "Reserved Keys" table (rewrite).
- `AGENT.md`, `getting-started.md`, `migrations.md`, `backends.md`, `types.md`, `skills.md` — any prose mentioning
  the dunders.
- `plans/reference-handling.md` — update the PR 2 spec to use `shared_refs` and `__ver_id__`/`__ver_ref__`. (PR
  #25, which ships the cycle-detection part of that plan, also lands the new key names in its plan-doc copy.)
- `plans/native-hdf5-storage.md`, `plans/save-as-you-go-hdf5.md` — historical plans, mention the dunders. Update
  for accuracy or annotate with a "superseded by …" note.

### Examples

- `examples/comment_defaults.py` — verify output still looks right.

## Acceptance

- All existing tests pass after touchups.
- New back-compat tests cover at least one old-format file per backend, plus old-format ndarray/json sentinels.
- `pixi run cleanup && pixi run pytest` green.
- `docs/reference.md` reserved-keys table reflects the new shape.
- `CHANGELOG.md` 0.2.0 entry mentions the rename and the back-compat read window.
- `docs/plans/reference-handling.md` updated to use `shared_refs`, `__ver_id__`, `__ver_ref__` so PR 2 starts on
  the right names.

## Drop schedule

| Release | Behavior                                                                     |
| ------- | ---------------------------------------------------------------------------- |
| 0.2.0   | Read both old and new. Write only new.                                       |
| 0.2.x   | Same as 0.2.0.                                                               |
| 1.0     | Drop old-key read support. Provide migration: `versionable migrate <file>` or "load and re-save". |

## Risk

Low–medium. Mechanical rename, but file format change. The mitigations:

- Read-both back-compat means existing 0.1.x files keep working.
- The user-data sentinel renames (`__ver_ndarray__` etc.) are the only place where a careful test suite per
  backend is needed — converters use these as type discriminators inside arbitrary dicts.

## Open questions

1. **Internal meta-dict keys.** `Backend.load()` returns `(fields, meta)` where `meta` uses `__OBJECT__` /
   `__VERSION__` / `__HASH__` as dict keys (matching the file format keys it's translating from). After the
   rename, should this internal contract switch to `object` / `version` / `hash`? Default: yes, for symmetry —
   the contract is internal, no external callers, and the save-side already uses lowercase (`metaDict = {"name":
   ..., "version": ..., "hash": ...}`).
2. **Mixed old + new in the same file.** A handcrafted file could in theory have both. Read prefers new; should
   it warn? Default: no — silent prefer-new keeps the load path simple. Collisions in real files won't happen.
3. **`__ver_lazy__` is internal-only.** It's never written to disk; it's a marker on in-memory dicts during the
   HDF5 lazy-read path. The rename is purely cosmetic. Confirm no external code depends on the name.
4. **Migration tool at 1.0.** Out of scope for this PR but worth filing a tracking issue: a `versionable
   migrate <path>` CLI that rewrites old-key files in place. Or just document "load and re-save" as the
   migration path.
