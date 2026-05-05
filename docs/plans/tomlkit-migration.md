[//]: # (vim: set ft=markdown:)

# TOML Backend: Migrate from `toml` to `tomlkit`

- **Created:** 2026-05-03
- **Last updated:** 2026-05-03
- **Status:** Planned
- **Branch:** `feature/tomlkit` (worktree: `versionable.worktrees/feature-tomlkit/`)
- **Targets release:** 0.2.0
- **Tracking issue:** <https://github.com/hendrickmelo/versionable/issues/27>

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or
> superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the unmaintained `toml` package with `tomlkit` in the TOML backend, and rebuild the
`commentDefaults=True` code path on top of tomlkit's structural document API.

**Architecture:** Boundary conversion (option A in the handoff). Read path calls
`tomlkit.parse(text).unwrap()` to produce a plain `dict[str, Any]` — every downstream caller stays
unchanged. Write path builds output two ways: the default branch hands the data dict to
`tomlkit.dumps(data)`; the `commentDefaults=True` branch builds a `tomlkit.document()` and recursively
emits each field either as a key-value pair or as a `tomlkit.comment(...)` line, using the dataclass's
own defaults to decide. Section headers and `__versionable__` envelope tables are never commented.

**Tech stack:** Python 3.12+, [`tomlkit` ≥ 0.13](https://github.com/python-poetry/tomlkit), pixi.

---

## Problem

### 1. The `toml` package is unmaintained

`toml` (PyPI) had its last release as 0.10.2 in November 2020. The ecosystem has consolidated around
`tomli` / `tomllib` (read) and `tomli_w` / `tomlkit` (write). Shipping with a stale dependency is
both a maintenance smell and a signal to downstream users that the project isn't keeping current.

### 2. `commentDefaults` is implemented as fragile string-walking

`src/versionable/_toml_backend.py::_commentDefaultLines` dumps the class defaults to TOML, splits the
result into a set of lines, then walks the actual `toml.dumps(data)` output line-by-line and prepends
`# ` to lines that match the default-line set. This works today but:

- Every new TOML construct (multi-line arrays, dotted keys, inline tables) is a potential edge case
  the line-walker has to know about.
- It cannot preserve user-added comments — they are wiped on every `save()` because we re-emit from
  the parsed Python dict.

### 3. Layout for nested envelopes is dense

After PR #26's wrap commit, nested Versionables emit their envelope under `[parent.__versionable__]`.
With `toml.dumps`, sub-tables get pushed to the end of the file — so a config file with two nested
objects ends up with the data sections at the top and the metadata sub-tables clustered awkwardly at
the bottom. tomlkit preserves insertion order, so explicit document construction can place each
`[parent.__versionable__]` directly after its `[parent]` section. (Verify in **Task 3**.)

## Design

### Read path (option A — boundary conversion)

```python
import tomlkit
from tomlkit.exceptions import TOMLKitError

text = path.read_text(encoding="utf-8")
data: dict[str, Any] = tomlkit.parse(text).unwrap()
```

`unwrap()` returns plain Python primitives (`dict`, `list`, `str`, `int`, …) so the existing
back-compat reader code (`metaTable.get("object", metaTable.get("__OBJECT__", ""))` etc.) works
unchanged.

Comments on the parsed document are *discarded* at the boundary. Round-trip preservation of
user-added comments is **out of scope** for this PR — see [Out of scope](#out-of-scope).

### Write path — default branch

```python
import tomlkit
content = tomlkit.dumps(data)
```

`tomlkit.dumps` accepts any `Mapping`, so a plain dict works directly. Output formatting will differ
byte-wise from `toml.dumps` but is structurally equivalent and round-trips through `tomlkit.parse`.

### Write path — `commentDefaults=True` branch

Replace the line-walking pass with structural document construction. For each field whose serialized
value equals the dataclass default, emit a `tomlkit.comment(...)` line in place of the key-value pair.
Recurse into nested Versionable tables to apply the same logic at every level.

```python
def _emitWithCommentedDefaults(data: dict[str, Any], cls: type) -> str:
    doc = tomlkit.document()
    _addContainerWithDefaults(doc, data, cls)
    return tomlkit.dumps(doc)
```

`_addContainerWithDefaults` is recursive and handles:

- Scalar fields at default → `tomlkit.comment("name = \"my-server\"")`
- Scalar fields not at default → `container[key] = value`
- `__versionable__` sub-tables → always emitted uncommented
- Nested Versionable dicts → recurse with the field's resolved Versionable class

### Touchpoints

| File                                        | Change                                                                       |
| ------------------------------------------- | ---------------------------------------------------------------------------- |
| `pixi.toml`                                 | `[feature.toml.dependencies]`: `toml` → `tomlkit`                            |
| `pyproject.toml`                            | `[project.optional-dependencies]`: `toml` extra and `all` extra; mypy override |
| `pixi.lock`                                 | Regenerated by `pixi install`                                                |
| `src/versionable/_toml_backend.py`          | Library swap + structural rewrite of `_commentDefaultLines`                  |
| `tests/test_toml_backend.py`                | `pytest.importorskip("toml")` → `tomlkit`; 4 sites of `import toml` replaced |
| `tests/test_cross_backend_roundtrip.py`     | `import toml as _toml` → `import tomlkit as _tomlkit`                        |
| `tests/test_cycles.py`                      | Same                                                                         |
| `tests/test_migration.py`                   | Same                                                                         |
| `examples/comment_defaults.py`              | Update commented-out expected-output if formatting differs                   |
| `docs/backends.md`                          | Regenerate the TOML example block if formatting differs                      |
| `CHANGELOG.md`                              | 0.2.0 entry: dependency swap + any user-visible formatting changes           |

## Test strategy

The existing `tests/test_toml_backend.py` is the spec for user-visible behavior. The plan keeps every
existing test passing (with assertion adjustments where a test pins exact whitespace that tomlkit
emits differently). Specific suites:

- **`TestTomlMetadata`** — semantic assertions on parsed structure. Should pass unchanged once the
  library is swapped.
- **`TestTomlCommentDefaults`** — pins enough output structure that some assertions may need to relax
  whitespace. The *intent* of every test is preserved by the structural rewrite.
- **`TestTomlLiteral`, `TestTomlMissingVersion`, `TestTomlErrors`** — semantic. Should pass unchanged.
- **`TestTomlBackCompat`** — legacy 0.1.x reads. The backward-compat code operates on Python dicts
  after parsing, so it's library-agnostic. Verify it still passes.

New tests added in **Task 3**:

- **`test_commentDefaultsEmitsValidToml`** — assert that `commentDefaults=True` output round-trips
  through `tomlkit.parse` without raising and produces a parseable document.
- **`test_dependencyImport`** — bare `import tomlkit` succeeds. Trivial guardrail against future
  accidental dependency removal.

Out-of-scope (documented as a limitation, not tested as a requirement):

- **`test_userCommentsSurviveRoundTrip`** — would require option B (passing `TOMLDocument` through
  the pipeline). Skipped this PR; tracked as a future enhancement.

## Backwards compatibility

- **File format unchanged.** The on-disk shape (`__versionable__` envelope, nested
  `[parent.__versionable__]` sub-tables, `__ver_json__` wrappers for ndarray blobs) is identical.
  Files written by 0.1.x or by the in-flight 0.2.x envelope-keys PR continue to load.
- **Output formatting differs byte-wise.** `tomlkit.dumps` makes different whitespace and
  quote-style choices than `toml.dumps`. Files saved with the new code load identically; old files
  load identically. Diffs on regenerated example files are expected and called out in CHANGELOG.
- **No API change.** The public `versionable.save() / versionable.load()` signatures are unchanged.
  `commentDefaults=True` keyword is unchanged.

## Risks

| Risk                                                                                  | Mitigation                                                                              |
| ------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------- |
| `tomlkit.dumps` emits `__ver_json__` ndarray wrappers as section headers, not key-val   | Verified by `TestTomlBackCompat::test_loadOldFormatNdarray` round-trip.                 |
| Nested-envelope layout (`[parent.__versionable__]`) ordering differs from `toml.dumps`  | This is the *desired* improvement; verify by running examples and visual inspection.    |
| `tomlkit.dumps` does not reorder scalar fields before `[table]` headers like `toml` did | Existing tests use substring assertions (insensitive to order). If layout is awkward, reorder the `data` dict construction in `save()` to put scalars first, then `__versionable__`, then nested tables. |
| `TOMLKitError` doesn't catch every parse failure mode `toml.TomlDecodeError` did        | `tests/test_toml_backend.py::TestTomlErrors` exercises malformed-file paths.            |
| Structural `commentDefaults` rewrite breaks an edge case the line-walker handled        | All `TestTomlCommentDefaults` assertions kept; new round-trip parseability test added.  |
| `tomlkit.comment("foo")` emits `#foo` (no space) instead of `# foo`                     | Existing tests assert `'# debug = false' in text` (with space). If tomlkit omits the space, pass `tomlkit.comment(" " + line)` to force it. |

## Acceptance

- `pixi run --environment default cleanup && pixi run --environment default pytest` green.
- `toml` no longer in `pixi.toml`, `pyproject.toml`, or `pixi.lock`; `tomlkit` is.
- All existing TOML tests pass (with assertion updates for whitespace where appropriate).
- New `test_commentDefaultsEmitsValidToml` and `test_dependencyImport` added and passing.
- `examples/comment_defaults.py` runs and its in-file expected output is regenerated if differs.
- `docs/backends.md` TOML example regenerated if differs.
- `CHANGELOG.md` 0.2.0 entry mentions the dependency swap.

## Out of scope

- **Round-trip user comment preservation** (read a file with comments, save preserving them).
  Requires passing `TOMLDocument` through the internal pipeline (option B in the handoff). Type
  contract on `Backend.load()` would change. Documented as a limitation; future PR.
- **TOML format-version bump** (introducing `format` in `__versionable__`). Separate concern.
- **Migration of project's own `pixi.toml` / `pyproject.toml` parsing**. Out of scope — only the
  library's runtime use of `toml` is in scope.

---

## Tasks

Each task is one commit. Steps inside a task are checkbox-trackable for the executor.

### Task 1: Land this plan doc

**Files:**

- Create: `docs/plans/tomlkit-migration.md` (this file)

- [ ] **Step 1: Verify the file is present and well-formed**

```bash
ls docs/plans/tomlkit-migration.md
```

Expected: file listed.

- [ ] **Step 2: Stage and review**

```bash
git add docs/plans/tomlkit-migration.md
git diff --cached --stat
```

Expected: 1 file changed, ~280 insertions.

- [ ] **Step 3: Commit (after user approval per CLAUDE.md)**

```bash
git commit -m "Add tomlkit migration plan"
```

---

### Task 2: Swap dependency + library imports + tests in lockstep

This task is one atomic change because pulling the rug on `toml` (removing the dep) breaks every
test file that imports it directly. Code, deps, tests all move together.

**Files:**

- Modify: `pixi.toml` (lines 30–31)
- Modify: `pyproject.toml` (line 28, line 31, line 86)
- Modify: `pixi.lock` (regenerated)
- Modify: `src/versionable/_toml_backend.py` (lines 41–44, 93, 103–104, 222–224)
- Modify: `tests/test_toml_backend.py` (lines 8, 44–50, 57–63, 151–161, 167–177)
- Modify: `tests/test_cross_backend_roundtrip.py` (line 117)
- Modify: `tests/test_cycles.py` (line 29)
- Modify: `tests/test_migration.py` (line 23)

- [ ] **Step 1: Update `pixi.toml`**

Replace lines 30–31:

```toml
[feature.toml.dependencies]
toml = ">=0.10"
```

with:

```toml
[feature.toml.dependencies]
tomlkit = ">=0.13"
```

- [ ] **Step 2: Update `pyproject.toml`**

Line 28 — replace:

```toml
toml = ["toml>=0.10"]
```

with:

```toml
toml = ["tomlkit>=0.13"]
```

Line 31 — replace:

```toml
all = ["numpy>=1.26", "h5py>=3.10", "hdf5plugin>=4.0", "toml>=0.10", "pyyaml>=6.0"]
```

with:

```toml
all = ["numpy>=1.26", "h5py>=3.10", "hdf5plugin>=4.0", "tomlkit>=0.13", "pyyaml>=6.0"]
```

Line 86 — replace:

```toml
module = ["numpy.*", "h5py.*", "hdf5plugin.*", "toml.*", "yaml.*"]
```

with:

```toml
module = ["numpy.*", "h5py.*", "hdf5plugin.*", "tomlkit.*", "yaml.*"]
```

- [ ] **Step 3: Regenerate `pixi.lock`**

```bash
pixi install
```

Expected: lockfile updated to drop `toml`, add `tomlkit`. Repo memory note from
`feedback_pixi_lock.md` — `pixi install --locked` would fail until the lock is regenerated; use the
plain `pixi install` here.

- [ ] **Step 4: Verify both deps resolved correctly**

```bash
pixi list -e default | grep -E "^(toml|tomlkit)\b"
```

Expected: `tomlkit` listed; `toml` *not* listed.

- [ ] **Step 5: Rewrite the `_toml_backend.py` import + load path**

Replace lines 41–44:

```python
try:
    import toml
except ImportError as e:
    raise ImportError("TOML backend requires toml — install it with: `pip install toml`") from e
```

with:

```python
try:
    import tomlkit
    from tomlkit.exceptions import TOMLKitError
except ImportError as e:
    raise ImportError("TOML backend requires tomlkit — install it with: `pip install tomlkit`") from e
```

Replace line 93:

```python
content = toml.dumps(data)
```

with:

```python
content = tomlkit.dumps(data)
```

Replace lines 103–104:

```python
data = toml.loads(text)
except (OSError, toml.TomlDecodeError) as e:
```

with:

```python
data = tomlkit.parse(text).unwrap()
except (OSError, TOMLKitError) as e:
```

Add the `dict[str, Any]` annotation if pyright requires it:

```python
data: dict[str, Any] = tomlkit.parse(text).unwrap()
```

- [ ] **Step 6: Patch `_commentDefaultLines` to use tomlkit (minimal change — full structural rewrite is Task 3)**

Replace lines 222–224:

```python
buf = io.StringIO()
toml.dump(defaultSerialized, buf)
defaultLineSet = set(buf.getvalue().splitlines())
```

with:

```python
defaultLineSet = set(tomlkit.dumps(defaultSerialized).splitlines())
```

Remove the now-unused `import io` at the top of the file (line 36).

- [ ] **Step 7: Update test imports in `tests/test_toml_backend.py`**

Line 8 — replace `pytest.importorskip("toml")` with `pytest.importorskip("tomlkit")`.

Lines 44, 57 — `import toml` → `import tomlkit`.
Lines 50, 63 — `toml.loads(p.read_text())` → `tomlkit.parse(p.read_text()).unwrap()`.

Lines 151–161 — replace:

```python
import toml

p = tmp_path / "out.toml"
p.write_text(
    toml.dumps(
        {
            "__versionable__": {"object": "WithLiteral", "version": 1, "hash": ""},
            "name": "test",
            "mode": "banana",
        }
    )
)
```

with:

```python
import tomlkit

p = tmp_path / "out.toml"
p.write_text(
    tomlkit.dumps(
        {
            "__versionable__": {"object": "WithLiteral", "version": 1, "hash": ""},
            "name": "test",
            "mode": "banana",
        }
    )
)
```

Apply the identical replacement at lines 167–177.

- [ ] **Step 8: Update import-skip patterns in cross-backend test files**

In `tests/test_cross_backend_roundtrip.py` line 117, `tests/test_cycles.py` line 29, and
`tests/test_migration.py` line 23, replace:

```python
import toml as _toml  # noqa: F401
```

(or the equivalent) with:

```python
import tomlkit as _tomlkit  # noqa: F401
```

Also rename the variable elsewhere in the file if used (`grep -n "_toml" <file>`).

- [ ] **Step 9: Run the test suite and verify everything passes**

```bash
pixi run --environment default pytest tests/test_toml_backend.py -v
```

Expected: all tests pass. If `TestTomlCommentDefaults` fails because of whitespace differences in
the line-walking match, loosen the assertion to a `.splitlines()` comparison or update the expected
substring. Note any failures with their fix in this step's commit.

```bash
pixi run --environment default pytest -v
```

Expected: all tests across the suite pass.

- [ ] **Step 10: Run cleanup**

```bash
pixi run --environment default cleanup
```

Expected: clean. Fix any reported errors before committing.

- [ ] **Step 11: Show staged diff for review, then commit (after user approval)**

```bash
git add pixi.toml pyproject.toml pixi.lock src/versionable/_toml_backend.py \
        tests/test_toml_backend.py tests/test_cross_backend_roundtrip.py \
        tests/test_cycles.py tests/test_migration.py
git diff --cached --stat
```

Commit:

```bash
git commit -m "Swap toml dependency for tomlkit in TOML backend"
```

---

### Task 3: Rewrite `commentDefaults` as structural document construction

The line-walking trick from `_commentDefaultLines` is library-agnostic but doesn't gain any of
tomlkit's advantages. This task replaces it with explicit `tomlkit.comment(...)` insertion in a
constructed document.

**Files:**

- Modify: `src/versionable/_toml_backend.py` (rewrite `_commentDefaultLines` and its caller)
- Modify: `tests/test_toml_backend.py` (add new tests)

- [ ] **Step 1: Write the failing test for parseability**

Append to `tests/test_toml_backend.py`, inside `TestTomlCommentDefaults`:

```python
def test_commentDefaultsEmitsValidToml(self, tmp_path: Path) -> None:
    """commentDefaults output is parseable as valid TOML and round-trips."""
    import tomlkit

    obj = SimpleConfig(name="test")
    p = tmp_path / "out.toml"
    versionable.save(obj, p, commentDefaults=True)

    # Parsing should not raise — the file with commented defaults is valid TOML
    parsed = tomlkit.parse(p.read_text()).unwrap()
    assert parsed["__versionable__"]["object"] == "SimpleConfig"
    assert parsed["name"] == "test"
    # Commented defaults are not present as keys
    assert "debug" not in parsed
    assert "retries" not in parsed
```

- [ ] **Step 2: Write the failing test for the trivial dependency import guardrail**

Append to `tests/test_toml_backend.py` as a top-level test (outside any class):

```python
def test_dependencyImport() -> None:
    """tomlkit imports cleanly — guardrail against accidental dep removal."""
    import tomlkit

    assert hasattr(tomlkit, "parse")
    assert hasattr(tomlkit, "dumps")
```

- [ ] **Step 3: Run both new tests**

```bash
pixi run --environment default pytest \
    tests/test_toml_backend.py::TestTomlCommentDefaults::test_commentDefaultsEmitsValidToml \
    tests/test_toml_backend.py::test_dependencyImport -v
```

Expected: `test_commentDefaultsEmitsValidToml` PASSES already (the line-walker output is also valid
TOML). `test_dependencyImport` PASSES. The new tests pin behavior; the structural rewrite below
must keep them passing.

- [ ] **Step 4: Replace `_commentDefaultLines` with structural construction**

In `src/versionable/_toml_backend.py`, delete the entire `_commentDefaultLines` function (lines
178–256 in the original) and replace with:

```python
def _emitWithCommentedDefaults(data: dict[str, Any], cls: type) -> str:
    """Build TOML output where fields at their default value appear as comment lines.

    Walks `data` in parallel with the dataclass's defaults. For each scalar/list field
    whose serialized value matches the default, emit a ``# key = value`` comment instead
    of a key-value pair. Section headers and ``__versionable__`` envelope tables are
    never commented. Nested Versionable tables recurse with their own defaults.
    """
    doc = tomlkit.document()
    _addContainerWithDefaults(doc, data, cls)
    return tomlkit.dumps(doc)


def _addContainerWithDefaults(
    container: Any,  # tomlkit.TOMLDocument | tomlkit.items.Table
    data: dict[str, Any],
    cls: type | None,
) -> None:
    """Populate `container` with `data`; commented defaults driven by `cls`."""
    defaults = _classDefaultsToml(cls) if cls is not None else {}
    fieldTypes = _resolveFields(cls) if cls is not None else {}

    for key, value in data.items():
        if key == "__versionable__":
            sub = tomlkit.table()
            for metaKey, metaVal in value.items():
                sub[metaKey] = metaVal
            container[key] = sub
            continue

        if isinstance(value, dict):
            sub = tomlkit.table()
            nestedCls = _findVersionableType(fieldTypes.get(key))
            _addContainerWithDefaults(sub, value, nestedCls)
            container[key] = sub
            continue

        if key in defaults and value == defaults[key]:
            scratch = tomlkit.document()
            scratch[key] = value
            line = tomlkit.dumps(scratch).rstrip("\n")
            container.add(tomlkit.comment(line))
        else:
            container[key] = value


def _classDefaultsToml(cls: type) -> dict[str, Any]:
    """Compute serialized + TOML-safe defaults for each field of `cls`."""
    import dataclasses

    from versionable._types import serialize

    dcFields = {f.name: f for f in dataclasses.fields(cls)}
    resolvedFields = _resolveFields(cls)
    out: dict[str, Any] = {}
    for name, tp in resolvedFields.items():
        dcField = dcFields.get(name)
        if dcField is None:
            continue
        if dcField.default is not dataclasses.MISSING:
            val = serialize(dcField.default, tp)
            if val is not None:
                out[name] = _toTomlSafe(val)
        elif dcField.default_factory is not dataclasses.MISSING:
            val = serialize(dcField.default_factory(), tp)
            if val is not None:
                out[name] = _toTomlSafe(val)
    return out


def _findVersionableType(fieldType: Any) -> type | None:
    """If `fieldType` is a Versionable subclass, return it; else None.

    Parameterized types like ``list[Inner]`` are not unwrapped — commentDefaults
    does not recurse into list elements (matches existing behavior).
    """
    from versionable._base import Versionable

    if isinstance(fieldType, type) and issubclass(fieldType, Versionable):
        return fieldType
    return None
```

Update the imports at the top of `_toml_backend.py`:

```python
from versionable._base import _resolveFields
```

(already imported — confirm). Remove the now-unused inline imports inside the deleted function.

- [ ] **Step 5: Update the call site in `save()`**

Replace lines 92–96:

```python
try:
    content = toml.dumps(data)
    if commentDefaults:
        content = _commentDefaultLines(content, fields, meta["name"])
    path.write_text(content, encoding="utf-8")
```

with:

```python
try:
    if commentDefaults:
        content = _emitWithCommentedDefaults(data, cls)
    else:
        content = tomlkit.dumps(data)
    path.write_text(content, encoding="utf-8")
```

Note: the existing `objectName` lookup (`meta["name"]`) is no longer needed because `cls` is already
in scope (passed by `_api.save`). The structural rewrite uses `cls` directly.

- [ ] **Step 6: Run the full TOML test suite**

```bash
pixi run --environment default pytest tests/test_toml_backend.py -v
```

Expected: all tests pass, including:

- `TestTomlCommentDefaults::test_defaultsCommentedOut`
- `TestTomlCommentDefaults::test_nonDefaultsNotCommented`
- `TestTomlCommentDefaults::test_commentedFileLoadsWithDefaults`
- `TestTomlCommentDefaults::test_nestedSectionHeadersNotCommented`
- `TestTomlCommentDefaults::test_commentDefaultsEmitsValidToml` (new)
- `test_dependencyImport` (new)

If any test fails: investigate the structural output. Likely failure modes:

- **Whitespace around section headers.** Adjust the assertion to be whitespace-tolerant (e.g.,
  `'object = "Inner"' in text` is robust; `'\n[__versionable__]\n' in text` is brittle — prefer the
  former).
- **Missing space after `#` in commented lines.** `tomlkit.comment("foo")` may emit `#foo` instead
  of `# foo`. The fix is to construct the comment with a leading space:
  `tomlkit.comment(" " + line)` inside `_addContainerWithDefaults`. Verify with a quick repl check
  before adjusting.
- **Section ordering.** If `[__versionable__]` ends up before scalar fields and that breaks any
  substring assertion that *also* checks line position, prefer reordering the `data` dict in
  `save()` to put scalar fields first, then `[__versionable__]`, then nested tables.

- [ ] **Step 7: Run the full test suite to catch cross-test regressions**

```bash
pixi run --environment default pytest -v
```

Expected: green.

- [ ] **Step 8: Run cleanup**

```bash
pixi run --environment default cleanup
```

Expected: clean.

- [ ] **Step 9: Show diff and commit (after user approval)**

```bash
git add src/versionable/_toml_backend.py tests/test_toml_backend.py
git diff --cached --stat
git commit -m "Rewrite commentDefaults with structural tomlkit document construction"
```

---

### Task 4: Refresh docs, examples, CHANGELOG

The library output formatting may differ from `toml.dumps`. Verify and regenerate the affected
documentation and example output blocks.

**Files:**

- Modify: `examples/comment_defaults.py` (in-file expected output, lines 57–113)
- Modify: `docs/backends.md` (TOML example block, lines 130–146 and 178–217)
- Modify: `CHANGELOG.md` (0.2.0 entry)

- [ ] **Step 1: Run the example and capture current output**

```bash
pixi run --environment default python examples/comment_defaults.py
```

Expected: the example runs cleanly and prints `=== TOML ===` and `=== YAML ===` blocks.

Visually compare the printed TOML block against the commented expected output in
`examples/comment_defaults.py` lines 57–87 (the `# Output TOML file:` block). The YAML block is not
affected by this PR, so its expected output should match unchanged.

- [ ] **Step 2: Update the in-file expected output if it differs**

Open `examples/comment_defaults.py`. The expected output is in commented lines 57–113. If the actual
output from Step 1 differs, replace the commented block to match the new output verbatim, preserving
the leading `# ` on each line.

If the output matches: skip this step.

- [ ] **Step 3: Verify `docs/backends.md` TOML examples match real output**

Construct a minimal `SensorConfig` matching the docs (lines 132–141 — `name = "probe-A"`,
`sampleRate_Hz = 120000`, `channels = [0, 1, 2]`) and save it:

```bash
pixi run --environment default python -c '
from dataclasses import dataclass, field
import versionable
from versionable import Versionable

@dataclass
class SensorConfig(Versionable, version=1, hash="9d6951"):
    name: str
    sampleRate_Hz: int
    channels: list[int] = field(default_factory=list)

cfg = SensorConfig(name="probe-A", sampleRate_Hz=120000, channels=[0, 1, 2])
versionable.save(cfg, "/tmp/sensor.toml")
print(open("/tmp/sensor.toml").read())
'
```

Compare with the docs block at `docs/backends.md` lines 132–141. If the order of sections, the
quoting style, or the whitespace differs, update the docs block to match.

Apply the same check for the `commentDefaults=True` example block (lines 209–217) and the
nested-Versionable example block (lines 178–195).

- [ ] **Step 4: Append a CHANGELOG entry**

In `CHANGELOG.md`, under `## 0.2.0 (unreleased)`, append (a new bullet at the end of the existing
list):

```markdown
- TOML backend: switched the underlying library from `toml` (unmaintained since 2020) to
  [`tomlkit`](https://github.com/python-poetry/tomlkit). File format is unchanged; output formatting
  may differ byte-wise (whitespace, quote style). The `commentDefaults=True` code path is reimplemented
  on top of tomlkit's structural document API. Round-trip preservation of user-added comments is not
  yet supported (planned for a follow-up).
```

- [ ] **Step 5: Run tests and cleanup once more to catch any stray references**

```bash
pixi run --environment default cleanup && pixi run --environment default pytest
```

Expected: green.

- [ ] **Step 6: Show diff and commit (after user approval)**

```bash
git add examples/comment_defaults.py docs/backends.md CHANGELOG.md
git diff --cached --stat
git commit -m "Refresh docs, example output, and CHANGELOG for tomlkit migration"
```

---

## Suggested PR description (after all tasks land)

**Description:** Migrate the TOML backend from the unmaintained `toml` package to `tomlkit`, and
rebuild the `commentDefaults=True` code path on top of tomlkit's structural document API.

**Changes:**

- Swap `toml` for `tomlkit` in `pixi.toml`, `pyproject.toml`, `pixi.lock`, and the `[tool.mypy]`
  module override.
- Rewrite `_toml_backend.py` to call `tomlkit.parse(text).unwrap()` on read and `tomlkit.dumps(data)`
  on write.
- Replace the `_commentDefaultLines` line-walker with `_emitWithCommentedDefaults`, which constructs
  a `tomlkit.document()` and emits default-valued fields as `tomlkit.comment(...)` lines.
- Update tests that imported `toml` directly; switch `pytest.importorskip` targets.
- Refresh `examples/comment_defaults.py` and `docs/backends.md` example output blocks.

**Tests performed:**

- `pixi run --environment default pytest` — full suite green.
- `examples/comment_defaults.py` runs and produces the documented output.

## Open questions resolved during planning

1. **Round-trip user comment preservation.** Out of scope for this PR. Documented as a future
   enhancement in `Out of scope` and the CHANGELOG entry.
2. **Existing 0.1.x-saved files with `commentDefaults=True`.** Output formatting will differ
   byte-wise from new saves; loading still works. Mentioned in CHANGELOG.
3. **Nested-envelope layout improvement.** Likely fixed for free by tomlkit's insertion-order
   serialization, but explicitly verified in **Task 4 Step 3** by visual comparison with the docs.
