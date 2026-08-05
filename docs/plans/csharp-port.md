# C# (.NET) Port with Interchangeable Files

Created: 2026-08-05 16:15 EDT
Last updated: 2026-08-05 16:15 EDT

## Goal

A C# implementation of versionable (NuGet `Versionable`) with full file interchange: files written by either
language load in the other, including HDF5. Schema hashes are byte-identical across languages for the same logical
schema ([ADR-0001](../adr/0001-cross-language-schema-hash.md)).

## Decisions (from design interview 2026-08-05)

### Canonical type grammar (cross-language, [ADR-0001](../adr/0001-cross-language-schema-hash.md))

- The existing Python-flavored canonical strings are the language-neutral spec; C# maps into them.
- Serialization Names default to the bare class name (enums and custom converter types included); explicit override
  available. Removes module-path leaks (`mypkg.Status` → `Status`, `datetime.datetime` → `datetime`,
  `pathlib.Path` → `Path`, etc.).
- Scalar width erasure: all C# integral types → `int`, `float`/`double` → `float`; out-of-range at load → error.
- Array dtype is hash-significant ([ADR-0002](../adr/0002-array-dtype-hash-significant.md)):
  `NDArray[np.float64]`/`Tensor<double>` → `ndarray[float64]`; bare `np.ndarray` → `ndarray`; shape erased. Python
  gains runtime dtype validation (safe casts applied, unsafe → error).
- `Literal` cleanup: `Literal['fast', 'slow']` (no `typing.` prefix, strings quoted, ints bare, order-significant).
- C# mappings: `T?` (incl. NRT refs) → `Union[None, T]` (members sorted); `List<T>`/`T[]` → `list[T]`;
  `Dictionary<K,V>` → `dict[K, V]`; `HashSet<T>` → `set[T]`; `FrozenSet<T>` → `frozenset[T]`; value tuples →
  `tuple[...]`; `byte[]` → `bytes` (carve-out from the array rule). NRT (`<Nullable>enable</Nullable>`) required.

### Wire format / converters

- `Guid`↔UUID (lowercase-hyphenated), `decimal`↔Decimal (string; out-of-range → error), `byte[]`↔bytes (base64),
  `Complex`↔complex (`[re, im]`), `DateOnly`/`TimeOnly`↔date/time (ISO), `TimeSpan`↔timedelta (total seconds double),
  `Regex`↔Pattern (pattern only; flags dropped both sides — existing quirk).
- Datetime: naive ↔ `DateTime`, aware ↔ `DateTimeOffset`; C# may write 7-digit fractions (Python truncates);
  sub-microsecond ticks lost on C#→Python→C# round trip (documented).
- `Path`: C# ships a small `FilePath` wrapper (canonical `Path`, string on wire). Pure path variants deferred.
- Enums: numeric bare; string-valued via `[EnumValue("...")]` per member; fallback via `[EnumFallback]`
  (= `VERSIONABLE_FALLBACK`).
- Literals: `[LiteralValues("fast", "slow", Fallback = "fast")]` on plain properties; hashes as `Literal[...]`.
- Arrays: only `Tensor<T>` maps to `ndarray`; JSON/YAML/TOML wire form is base64 NPZ via a small in-library NPY/NPZ
  codec (no mainstream .NET NPY package exists).

### C# architecture ([ADR-0003](../adr/0003-csharp-compile-time-validation-codegen.md))

- Roslyn analyzer: compile-time hash validation (build error; severity configurable via `.editorconfig` =
  `ignoreHashErrors`). Diagnostic prints the canonical payload for debuggability.
- Source generator: per-type metadata + typed accessors + `[ModuleInitializer]` registration; one shared runtime
  engine (envelope, migrations, converters, backends). Native AOT/trimming safe from v1.
- Field naming: verbatim property names on the wire; `[VersionableField("...")]` is the only override — no naming
  policies, no Python-side knob.
- Migrations: nested `public static class Migrate` with `Migration` builder fields (`V1`, `V2`, …) and
  `[Migration(FromVersion = n)]` static methods; op parity with Python. Analyzer checks chain *contiguity*
  (gaps below the oldest present migration are fine).
- TFM: `net8.0` single-target (runs on 8/9/10; bump when 8 EOLs).

### Packages

| Concern | Choice |
|---|---|
| JSON | System.Text.Json (in-box) |
| YAML | YamlDotNet |
| TOML | Tomlyn (document model for `commentDefaults`) |
| HDF5 | PureHDF 2.x (+ `PureHDF.Filters.Blosc2`); single-maintainer risk accepted, isolated behind backend abstraction |
| Arrays | System.Numerics.Tensors 10.x (`Tensor<T>`, targets net8.0) |

### Scope

- v1 = Tier 1 (full file-format interchange, all four backends) + Tier 2 (behavioral parity: `loadDynamic`,
  `assumeVersion`, literal validation, `commentDefaults`, dev-mode hash severity).
- Tier 3 (HDF5 append sessions, lazy loading/slicing API) deferred: v1 must read/write Python-produced chunked +
  compressed HDF5 files, but the session/lazy API lands post-v1. **File a follow-up issue at v1 completion.**

### Versioning & releases ([ADR-0004](../adr/0004-shared-generation-versioning.md))

- Shared generation: `major.minor` coupled across PyPI/NuGet; patch per-package. "Match the minor and files
  interchange."
- Python grammar cleanup ships as the same generation the C# port debuts in (one breaking release; CHANGELOG
  documents hash migration — error messages supply new hashes).
- C# releases tag `dotnet-vX.Y.Z`, published by a new `publish-nuget.yml`.

### Conformance (`conformance/`)

- `hash-vectors.json`: canonical payload → expected hash.
- Parallel fixture classes in both languages sharing hardcoded hash literals (compile/import-time mirror proof).
- Checked-in golden corpus: 4 backends × type matrix × migration scenarios, with expected-value manifests. The
  checked-in bytes are the contract.
- CI on every PR: Python-writes→C#-reads, C#-writes→Python-reads, plus both read the golden corpus.

## Implementation phases

1. **Grammar spec + Python cleanup** — write the canonical grammar spec doc (versioned, in `conformance/`);
   implement bare Serialization Names, `ndarray[<dtype>]` + runtime dtype validation, Literal quoting; update all
   fixture hashes; hash vectors.
2. **C# core** — `Versionable` runtime skeleton, canonical grammar over Roslyn symbols, analyzer (hash check),
   source generator (metadata/accessors/registration), converter set, JSON backend. Golden-file tests against
   Python-written JSON from phase 1.
3. **Remaining backends** — YAML (YamlDotNet), TOML (Tomlyn, incl. `commentDefaults`), HDF5 (PureHDF: envelope
   groups, native type mapping, chunked+compressed dataset read/write incl. gzip+shuffle default and Blosc2).
4. **Migrations + polymorphism + Tier 2 parity** — builder/imperative migrations, chain-contiguity analyzer check,
   polymorphic collections, `loadDynamic`, literal/enum fallbacks, unknown-field policies, `skip_defaults`.
5. **Conformance CI + joint release** — bidirectional CI jobs, golden corpus, `publish-nuget.yml`, joint generation
   release (Python breaking release + NuGet debut). File the Tier-3 follow-up issue.
6. **Stretch: converter** — `python -m versionable.tools.to_csharp`: one-shot scaffolder via runtime introspection;
   converts fields/types/metadata/declarative migration ops/enums/literals; emits `// TODO` + commented Python
   source for lambdas, imperative migrations, and `default_factory`.
