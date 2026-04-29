# Reference Handling: Cycles and Shared Instances

- **Created:** 2026-04-26
- **Last updated:** 2026-04-26
- **Status:** Proposed
- **Branch:** `feature/reference-handling`

## Problem

Today's serializer is a pure tree walk: `serialize()` recurses through dicts/lists/sets/tuples, and `_serializeVersionable()` /
`_writeVersionable()` inline a fresh dict (or HDF5 subgroup) for every nested `Versionable`. There is no `id()`
memoization, no `visited` set, and no reference table at any layer.

This breaks four user-visible cases:

| Case | Example | Today |
| --- | --- | --- |
| **Self-cycle** | `n.children.append(n)` | `RecursionError` (Python stack limit) |
| **Mutual cycle** | `a.partner = b; b.partner = a` | `RecursionError` |
| **Diamond / shared** | `parent.left = parent.right = shared` | Saves `shared` twice; loads as two distinct instances |
| **DAG with deep sharing** | One calibration referenced by 50 sensors | File grows ~50×; load is N× slower |

The first two are loud (a stack trace) but cryptic. The last two are silent and arguably worse — files look fine,
identity is lost, files balloon.

## Goal

Two-step rollout, two PRs:

1. **PR 1 — Cycle detection (lands before 0.2.0).** Replace `RecursionError` with a clear
   `CircularReferenceError` carrying the field path. Strict additive change; no file format change; no public
   API change beyond a new exception class.
2. **PR 2 — Shared references (ships in 0.3.0).** Opt-in, per-class, lossless preservation of shared instances
   (and therefore cycles, for non-frozen classes). Each backend uses its idiomatic primitive: `__ver_id__` /
   `__ver_ref__` envelopes for JSON/TOML, native YAML anchors, HDF5 hard links.

PR 1 is a strict correctness improvement that any release should ship. It also establishes the threading and
error machinery that PR 2 builds on.

---

## PR 1 — Cycle detection (target: before 0.2.0)

### Scope

- Detect cycles in `serialize()` (covers JSON/YAML/TOML) and `_writeVersionable()` (HDF5).
- Raise a new `CircularReferenceError(VersionableError)` with the field path of the revisit.
- Add tests for the four cycle shapes (self, mutual, three-way, cycle through a list, cycle through a dict).
- Document the limitation in `docs/AGENT.md` and `docs/reference.md`.

### Non-goals for PR 1

- No support for shared (non-cyclic) references — diamonds still duplicate.
- No file format changes.
- No two-pass load.
- No HDF5 link support.

### Design

Thread a `_visited: set[int]` (set of `id(obj)`) plus a `_path: tuple[str, ...]` through the recursive write
paths. On entry to `_serializeVersionable` / `_writeVersionable`:

```python
if id(obj) in _visited:
    raise CircularReferenceError(
        f"Circular reference detected at field path "
        f"{'.'.join(_path) or '<root>'} → {type(obj).__name__}@{id(obj):x}. "
        f"versionable cannot serialize cycles in 0.2.x. "
        f"This will be supported in 0.3.0 via opt-in shared_refs=True."
    )
_visited.add(id(obj))
try:
    ... existing recursion ...
finally:
    _visited.discard(id(obj))
```

`_visited` only tracks `Versionable` instances. Lists/dicts/tuples don't get tracked — Python's own protections
catch a list-containing-itself (`RecursionError`), but cycles must pass through a `Versionable` for the schema
to be legal, so tracking only `Versionable` is sufficient and simpler.

The `finally` discard is important: it allows a `Versionable` to legitimately appear at two unrelated tree
positions (a diamond). PR 1 still duplicates that data on disk — PR 2 fixes it — but PR 1 must not falsely
flag diamonds as cycles. Removing from `_visited` on the way back up gives us "currently-on-stack" semantics,
which is exactly cycle detection.

### Touchpoints

| File | Change |
| --- | --- |
| `_types.py::serialize` | Accept `_visited` and `_path` (defaulted), pass through recursive calls. |
| `_types.py::_serializeVersionable` | Cycle check at entry; add to `_visited`; remove on exit. |
| `_types.py::_serializeCollection` | Pass `_visited`/`_path` through to inner `serialize` calls; extend `_path` with `[i]` or `[k]`. |
| `_hdf5_backend.py::_writeFields` | Accept `_visited`/`_path`. |
| `_hdf5_backend.py::_writeValue` | Pass through. |
| `_hdf5_backend.py::_writeVersionable` | Cycle check at entry. |
| `_hdf5_backend.py::_writeSequence`, `_writeDict` | Pass through with extended path. |
| `errors.py` | Add `CircularReferenceError(VersionableError)`. |
| `__init__.py` | Re-export `CircularReferenceError`. |

The `_visited`/`_path` parameters are private (leading underscore) and not part of the public `serialize()`
signature contract — backends call into `serialize()` from `JsonBackend.save()`, `YamlBackend.save()`, etc.,
which can pass fresh empty values.

### Path representation

Use dotted field names with `[i]` for list indices and `[key]` for dict keys, matching pytest-style:

- Self-cycle on a Node's children: `children[0] → Node@7f8a...`
- Cycle through a dict: `partners["alice"] → Person@... → partners["bob"] → Person@7f8a...`
- The starting point is `<root>`.

Including the `id()` (hex) makes it possible to distinguish "two separate instances of the same class"
(legal diamond, today still duplicated) from "actual revisit of the same instance" (cycle).

### Tests (`tests/test_cycles.py`)

```python
def test_self_cycle_raises():
    n = Node(name="root")
    n.children.append(n)
    with pytest.raises(CircularReferenceError, match="children\\[0\\]"):
        versionable.save(n, tmp_path / "out.json")

def test_mutual_cycle_raises(): ...
def test_three_way_cycle_raises(): ...
def test_cycle_through_dict_raises(): ...
def test_cycle_through_list_of_versionables_raises(): ...

def test_diamond_does_not_falsely_flag():
    # Same instance referenced twice, no cycle. PR 1 must not raise.
    shared = Inner(value=42)
    parent = Outer(left=shared, right=shared)
    versionable.save(parent, tmp_path / "out.json")  # succeeds (with duplication)

def test_cycle_detection_works_across_backends():
    # parametrize over .json, .yaml, .toml, .h5
    ...
```

The diamond test is the most important — it pins down the `finally`-discard semantics so we don't accidentally
ship a stricter check than intended.

### Documentation

- `docs/AGENT.md` — under "Common Patterns" or a new "Limitations" section, one paragraph: "versionable serializes
  trees, not graphs. Cycles raise `CircularReferenceError`. Shared references are duplicated on save and load
  as separate instances. See [reference handling plan]."
- `docs/reference.md` — add `CircularReferenceError` to the error tree.
- `CHANGELOG.md` — under 0.2.0: "Cycles in object graphs now raise `CircularReferenceError` instead of
  `RecursionError`. Shared references are still duplicated; lossless shared-ref support is planned for 0.3.0."

### Hash impact

None. Hashes are computed from field types via `computeHash()`. The `_visited` parameter doesn't influence
hashing.

### Migration impact

None. Migrations operate on raw `dict[str, Any]` data after backend load. Cycles can't appear in raw data
written by the current code (the `RecursionError` happens on save, not load).

### Risk

Low. The change is additive — code paths that didn't cycle before still don't cycle. The only behavior
change is `RecursionError` → `CircularReferenceError`, which any code catching `RecursionError` from
`save()` would have been doing as a workaround anyway.

### Acceptance

- All existing tests pass.
- `pixi run cleanup && pixi run pytest` green.
- New tests cover the five cycle shapes and the diamond non-cycle.
- `CircularReferenceError` is exported from the top-level package and documented.

---

## PR 2 — Shared references (target: 0.3.0)

### Scope

Opt-in, lossless preservation of shared instances. After load, two fields that pointed to the same instance
before save still satisfy `loaded.left is loaded.right`. Cycles are representable for non-frozen classes.
Backends use idiomatic primitives.

### Opt-in mechanism

Per-class flag on `Versionable.__init_subclass__`:

```python
@dataclass
class Node(Versionable, version=1, hash="...", shared_refs=True):
    name: str
    children: list[Node] = field(default_factory=list)
```

Default is `False`. The flag lives on `_InternalMeta` and is exposed via `VersionableMetadata.sharedRefs`.

The flag applies to the **root** class — once a save starts in shared-refs mode, the whole tree is encoded with
ID/REF semantics regardless of inner classes. Mixing modes within one file would produce inconsistent semantics
(some shared, some duplicated). Document this clearly.

Optional override at call site: `versionable.save(obj, path, sharedRefs=True)` — useful for loading existing
trees you don't own. If the kwarg is set, it overrides the class flag.

### Hash impact

The `shared_refs` flag does **not** participate in `computeHash()`. It's a serialization-mode choice, not a
schema change. Confirm by inspection in `_hash.py::computeHash` — only `fields: dict[str, Any]` (name → type)
flows in.

### File format — JSON / TOML

A first-occurrence object gets an `__ver_id__` field; subsequent occurrences are replaced by a one-key envelope:

```json
{
  "__versionable__": {"object": "Node", "version": 1, "hash": "...", "shared_refs": true},
  "__ver_id__": "#0",
  "name": "root",
  "children": [
    {"object": "Node", "version": 1, "hash": "...", "__ver_id__": "#1", "name": "left", "children": []},
    {"__ver_ref__": "#1"}
  ]
}
```

Notes:

- `shared_refs: true` on the root metadata declares the file uses ref semantics — readers can fast-path
  files without it (no two-pass needed).
- IDs are local to the file (`"#0"`, `"#1"`, …) and assigned in pre-order traversal. They're opaque strings, not
  numbers, so users don't read meaning into them.
- Refs to the root use `{"__ver_ref__": "#0"}` like any other.
- Non-`Versionable` shared values (numpy arrays, dataclasses-but-not-Versionable, dicts) are still duplicated.
  This is a deliberate scope limit — Versionable owns identity tracking only for its own instances.

### File format — YAML

Use PyYAML's native anchor/alias support. PyYAML emits `&id001` / `*id001` automatically when the same
Python object is encountered twice during dumping. To get this for free we need to **preserve identity in the
intermediate representation**: when serializing a `Versionable` we've already seen, emit the *same* dict
object (not a fresh copy with `__REF__`).

This makes YAML files cleaner than JSON's `__ver_ref__` envelopes:

```yaml
__versionable__:
  object: Node
  version: 1
  hash: "..."
  shared_refs: true
__ver_id__: "#0"
name: root
children:
  - &id001
    object: Node
    __ver_id__: "#1"
    name: left
    children: []
  - *id001
```

On load, PyYAML restores identity automatically — `parent["children"][0] is parent["children"][1]` is `True`.
We then walk the loaded dict and `__ver_ref__`-resolution becomes mostly a no-op; we just need to map dict
identity to instance identity.

### File format — TOML

TOML has no anchor concept and no inline references in the spec. Use the JSON envelope approach
(`__ver_id__` / `__ver_ref__`). TOML's table syntax handles this fine — a `__ver_ref__` table is just a
one-key inline table.

### File format — HDF5

HDF5 has hard links and soft links natively. Use **hard links**:

```text
/__versionable__/
/__versionable__/shared_refs (attr) = True
/name (attr) = "root"
/children/
/children/0/                    ← first occurrence: full subgroup
/children/0/__versionable__/
/children/0/name (attr) = "left"
/children/1 → /children/0       ← hard link to same group
```

`h5py` supports this via `group["alias"] = group["original"]`. On read, `h5py` doesn't tell us "this is a
link" by default — `group["children/1"]` returns the group at the linked location. We need to detect shared
groups by tracking `h5py.Group.id` (HDF5 internal object IDs) during read and emitting the same Python
instance for groups we've seen.

For cycles in HDF5: a child group can hard-link to an *ancestor*. Reading this requires the same identity
tracking we use elsewhere — when the recursive reader reaches a group it has already visited (by HDF5
object ID), it returns the (possibly partially-built) Python instance for that group.

This is the **only** backend where the storage primitive matches the in-memory semantics natively. The other
three need synthetic IDs.

### Two-pass load

For JSON/TOML (and YAML pre-identity-preservation), refs need resolution after the dict is parsed. The flow:

1. **Parse pass** — backend produces a raw dict tree with `__ver_ref__` markers at the leaves where shared.
2. **Index pass** — walk the tree; collect every `__ver_id__` → its dict.
3. **Construct pass** — for each unique ID, build a *shell* `Versionable` instance:
   - Mutable classes (default): construct with placeholder values for ref-typed fields, fill via `setattr` later.
   - Frozen classes: collect into a topological build order. If the graph has a cycle and the class is frozen,
     raise `BackendError("Cannot load cycle into frozen class X")`.
4. **Resolve pass** — walk the tree replacing `__ver_ref__: "#1"` with the constructed instance for `"#1"`.
5. **Post-init pass** — call `__post_init__` (skipped during shell construction) on each instance in topological
   order. For cycles, run `__post_init__` last and document the caveat.

Step (5) is the most fragile. A class whose `__post_init__` validates field values will see the fully-resolved
graph for non-cyclic refs, and a fully-resolved-but-self-referential graph for cycles. Document explicitly:
"`__post_init__` is called after all references are resolved. For shared_refs=True classes participating in
cycles, `__post_init__` may observe `self` in fields before construction has fully returned."

### Touchpoints — implementation

| File | Change |
| --- | --- |
| `_base.py::__init_subclass__` | Accept `shared_refs: bool = False`; store on `_InternalMeta`. |
| `_base.py::VersionableMetadata` | Add `sharedRefs: bool` field. |
| `_types.py::serialize` | When root `shared_refs=True`, switch to ID-tracking serializer. |
| `_types.py` | New `_serializeWithRefs(obj)` that maintains `id(obj) → assignedId` and emits `__ver_id__`/`__ver_ref__`. |
| `_types.py` | New `_deserializeWithRefs(data, cls)` that runs the four-pass load. |
| `_api.py::save` | Accept `sharedRefs: bool | None = None`; resolve override vs. class flag. |
| `_api.py::load` | Detect `shared_refs: True` in metadata; route to ref-aware path. |
| `_json_backend.py` | No format-specific changes — uses generic `serialize`/`deserialize`. |
| `_yaml_backend.py` | Preserve dict identity in the intermediate IR so PyYAML emits anchors. Document the trick. |
| `_toml_backend.py` | No format-specific changes. |
| `_hdf5_backend.py::_writeFields` | When in shared-refs mode, track `id(obj) → group_path`; emit hard links. |
| `_hdf5_backend.py::_readFields` | Track `h5py.Group.id` → Python instance; recurse into hard-linked groups only on first sight. |
| `errors.py` | New `FrozenCycleError(BackendError)` for cycles into frozen classes. |
| `_migration.py` | Resolve refs *before* migrations run. Migrations see Python objects, never `__ver_ref__` dicts. Document. |
| `__init__.py` | Re-export new symbols. |

### Tests (`tests/test_shared_refs.py`)

Cover the matrix:

| Shape × Backend | JSON | YAML | TOML | HDF5 |
| --- | --- | --- | --- | --- |
| Diamond | ✓ | ✓ | ✓ | ✓ |
| Self-cycle | ✓ | ✓ | ✓ | ✓ |
| Mutual cycle | ✓ | ✓ | ✓ | ✓ |
| 3-node cycle through list | ✓ | ✓ | ✓ | ✓ |
| Cycle through dict | ✓ | ✓ | ✓ | ✓ |
| Shared deep in DAG (50× refs) | ✓ | ✓ | ✓ | ✓ |
| Frozen class + diamond (works) | ✓ | ✓ | ✓ | ✓ |
| Frozen class + cycle (raises) | ✓ | ✓ | ✓ | ✓ |
| `__post_init__` runs once per unique instance | ✓ | ✓ | ✓ | ✓ |

Plus identity assertions on every loaded graph: `loaded.left is loaded.right`.

Plus migration tests: a v1 file with shared refs migrates to v2 with the shared structure preserved.

Plus a backwards-compat test: a 0.2.x file (no `shared_refs` marker) loads under a 0.3.0 reader.

### Migration impact

Refs are resolved before migrations run, so migration code stays simple. Document this for users writing
imperative migrations: their `MigrationContext` sees a real Python object graph (with cycles intact), not
`__ver_ref__` dicts. Existing migrations are unaffected because the resolution pass happens transparently.

### Backwards compatibility

- A 0.2.x file (no `shared_refs: true` in metadata) reads under 0.3.0 with the existing code path —
  no two-pass load, no perf hit. Detection is a single attribute check.
- A 0.3.0 file *with* `shared_refs: true` cannot be read by 0.2.x. This is fine — the feature is opt-in
  per class, so users who don't enable it keep producing 0.2.x-readable files. Document the forward-compat
  break clearly.
- `shared_refs: false` is never written (omission means false).

### Risk

Medium-high. The two-pass load with `__post_init__` ordering is the trickiest part. Plan to ship behind the
opt-in flag specifically so users who don't want the complexity don't pay for it.

### Acceptance

- Full test matrix passes on all four backends.
- 0.2.x files round-trip unchanged on a 0.3.0 reader.
- Identity assertions hold on every loaded graph.
- Documentation covers the opt-in flag, `__post_init__` ordering, frozen-class limitation, and forward-compat
  break.
- Reserved-keys docs in `docs/reference.md` list `__ver_id__`, `__ver_ref__`, and the `shared_refs` envelope
  flag as reserved.
- `pixi run cleanup && pixi run pytest` green.

---

## Out of scope (both PRs)

- **Sharing of non-Versionable values.** A numpy array referenced twice still duplicates. HDF5 users who
  want to dedupe arrays can use `Hdf5FieldInfo` / hard links manually. Versionable doesn't track identity for
  primitives, dicts, or arrays.
- **Cross-file references.** Refs are local to one file. No `__ver_ref__: "other_file.json#1"` plans.
- **External graph databases.** This isn't a graph DB — it's a serializer that doesn't break when graphs
  show up.
- **Reference-aware diffing.** Comparing two saved graphs by structure rather than text — interesting, not
  this plan.

## Open questions

1. **HDF5 hard-link detection on read** — `h5py.Group.id` is an `h5py.h5g.GroupID` object; equality should
   work but I haven't verified. If `id` comparison is unreliable, fall back to comparing `group.file.id` +
   `group.id.get_objinfo().fileno + objno` tuples.
2. **YAML identity-preservation trick** — needs validation that PyYAML reliably emits anchors when the same
   dict object appears twice. Edge case: PyYAML's default representer copies dicts in some paths. May need a
   custom representer.
3. **`__post_init__` ordering for cycles** — is "run after the cycle is closed" acceptable, or do we forbid
   `__post_init__` on shared_refs classes? Lean toward "run after, document the caveat" but want a concrete
   user case before committing.
4. **Should the sidecar-table layout (`__objects__: {id: {...}}, __root__: ref`) be considered as an
   alternative to inline `__ver_id__`?** It's cleaner to migrate but worse for hand-editing. Inline matches
   YAML's anchor model. Tentative: inline.
5. **Exposing `sharedRefs` on `VersionableMetadata`** — should it be public introspection? Probably yes,
   to mirror `skipDefaults` / `unknown`.

---

## Rollout

| Milestone | Includes | Targets release |
| --- | --- | --- |
| PR 1 | Cycle detection + `CircularReferenceError` + tests + docs | 0.2.0 |
| 0.2.0 release | Existing 0.2.0 scope + PR 1 | (unchanged target) |
| PR 2 | Shared refs (all four backends) + opt-in flag + tests + docs | 0.3.0 |
| 0.3.0 release | PR 2 | First release after 0.2.0 |

PR 1 must merge before the 0.2.0 release branch is cut. PR 2 lands on `main` after 0.2.0 ships.
