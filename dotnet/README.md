# Versionable (.NET)

C# implementation of versionable. Files written by either implementation load in the other, and the same logical schema
hashes byte-identically in both ([ADR-0001](../docs/adr/0001-cross-language-schema-hash.md)).

## Layout

| Path                         | What it is                                                                     |
| ---------------------------- | ------------------------------------------------------------------------------ |
| `src/Versionable/`           | Runtime library (`net8.0`): contracts, envelope, converters, backends.         |
| `src/Versionable.Analyzers/` | Roslyn analyzer + source generator (`netstandard2.0`), consumed by both above. |
| `tests/Versionable.Tests/`   | xUnit suite.                                                                   |

The analyzer validates each type's declared schema hash at compile time and the generator emits per-type metadata, typed
accessors, and registry registration — so the runtime never reflects over user types and stays Native AOT and trimming
safe ([ADR-0003](../docs/adr/0003-csharp-compile-time-validation-codegen.md)). Both the runtime library and the test
project reference the analyzer project with `OutputItemType="Analyzer"`, so the pipeline runs on every build here, not
only at consumer sites. It reaches consumers the same way: `Versionable.csproj` packs the analyzer assembly into the
`Versionable` package under `analyzers/dotnet/cs`, and fails the pack if it is missing — there is no separate analyzer
package to forget to install. `dotnet pack src/Versionable/Versionable.csproj` builds it; releases go through
`.github/workflows/publish-nuget.yml`, triggered by a GitHub Release tagged `dotnet-vX.Y.Z`
([ADR-0004](../docs/adr/0004-shared-generation-versioning.md)).

Shared build settings live in `Directory.Build.props`, package versions in `Directory.Packages.props` (central package
management — `PackageReference` items carry no `Version`), and the SDK floor in `global.json`.

## Build and test

```bash
dotnet restore
dotnet build -warnaserror
dotnet test
dotnet format --verify-no-changes
```

Cleanup — format, then build with analyzers — runs as one command from the repo root:

```bash
pixi run dotnet-cleanup
```

`pixi run cleanup` is the **Python** cleanup task and does not include `dotnet-cleanup`; the .NET SDK is not a pixi
dependency, so a Python-only checkout would fail on it. Run both before pushing a change that touches each side.

### Conformance

The suite reads the golden corpus from `conformance/golden/`. Setting `VERSIONABLE_GOLDEN_ROOT` points it at another
corpus directory instead, and `CorpusWriterTests` writes one to `VERSIONABLE_CORPUS_OUT`. The `conformance` job in
`.github/workflows/ci.yml` uses both to run each language's writers against the other's readers on every PR; its steps
are the reproduction recipe if a cross-language failure needs chasing locally.

## Conventions

Enforced from `.editorconfig`, so they are settled rather than argued:

- **Every `[Versionable]` type is declared `partial`.** The generator implements `IVersionableMetadataProvider` by
  emitting a second part of the type, and a static abstract member cannot be implemented from outside its declaring type
  — a non-partial type silently drops out of the compile-time path. Reading the generated member also needs a generic
  constraint: APIs that want it are written `Method<T>(...) where T : IVersionableMetadataProvider`. The diagnostic that
  catches a missing `partial` lands in task 2b; until then it surfaces as a missing member.
- **Namespace matches folder.** `src/Versionable/Errors/*.cs` is `Versionable.Errors`; anything directly under
  `src/Versionable/` is the root `Versionable` namespace. `IDE0130` fails the build otherwise.
- **File-scoped namespaces**, Allman braces, explicit types over `var`, braces never optional.
- **Naming** is standard .NET: PascalCase types and members, camelCase parameters and locals, `_camelCase` private
  fields (instance and static alike). Test methods use `scenario_expectation`, so `tests/` relaxes the method rule alone
  — private fields, parameters, locals, properties, and types stay enforced there.
- Naming violations surface through `dotnet format`, **not** `dotnet build` — the naming analyzer is IDE-side only, so
  `EnforceCodeStyleInBuild` does not cover it. `dotnet format --verify-no-changes` in CI is the gate that catches them.

## `dotnet` on PATH

The pixi task calls `dotnet` unqualified, so the SDK must be on `PATH`. It is not a pixi dependency. If the SDK lives in
`~/.dotnet` rather than a system location, export both variables — `DOTNET_ROOT` is what lets the CLI find its shared
runtime:

```bash
export PATH="$HOME/.dotnet:$PATH"
export DOTNET_ROOT="$HOME/.dotnet"
```

Target framework is `net8.0` (runs on .NET 8, 9, and 10); the SDK must be 8.0 or newer.

## Packages

| Concern | Package                                          |
| ------- | ------------------------------------------------ |
| JSON    | `System.Text.Json` (in-box)                      |
| YAML    | `YamlDotNet`                                     |
| TOML    | `Tomlyn` (document model, for `commentDefaults`) |
| HDF5    | `PureHDF` + `PureHDF.Filters.Blosc2`             |
| Arrays  | `System.Numerics.Tensors` (`Tensor<T>`)          |

## HDF5 compression

HDF5 is the one backend where the two implementations do not have the same filter set, because `PureHDF` and
`h5py`/`hdf5plugin` register different filters. Values always round-trip; what varies is which compression settings each
side can _write_ and _read_. Nothing below affects any other backend.

### gzip — the default, and the only one guaranteed both ways

`PureHDF` accepts only levels `0, 1, 6, 9`, so `Hdf5Compression` maps a requested level onto the nearest one it can
write:

| Requested | Written |
| --------- | ------- |
| `0`       | `0`     |
| `1`–`3`   | `1`     |
| `4`–`7`   | `6`     |
| `8`–`9`   | `9`     |

**Python's default is level 4, so a C#-written file records level 6** — zlib's own default, one step from Python's. This
is a difference in the recorded `compression_opts`, not in the format: the filter is ordinary gzip, no reader consults
the level to decompress, and h5py opens the file without any plugin. The corpus is written by Python at level 4 and read
by C# unchanged.

### Blosc — writable here, but not the variant Python writes

Despite the package name, `PureHDF.Filters.Blosc2` implements **filter 32001 (blosc v1)**. Python's
`Hdf5Compression(algorithm="blosc")` routes through `hdf5plugin.Blosc2`, which writes **filter 32026**. The two
directions are therefore asymmetric:

| Direction                        | Result                                                |
| -------------------------------- | ----------------------------------------------------- |
| C# writes Blosc → Python reads   | Works, with `hdf5plugin` imported. Plain h5py cannot. |
| Python writes Blosc (32026) → C# | **Fails on every platform.** Nothing registers 32026. |

A second and independent constraint applies to the write side: `Blosc2.PInvoke` ships native binaries for `win-x86`,
`win-x64`, and `linux-x64` only, so saving with Blosc on macOS (including Apple silicon) fails. The failure is rewritten
into a `BackendException` naming the missing library, listing the runtime identifiers that have one, and pointing at
gzip.

### zstd and lzf — not readable here

`PureHDF` ships neither filter, so `Hdf5CompressionAlgorithm` has no member for them rather than offering something no
reader could open. A Python file written with either preset reports an unregistered filter when C# opens it.

### If you need interchange today

**Use gzip.** It is the default on both sides, needs no plugin anywhere, and is the only preset guaranteed readable in
both directions. There is no workaround for the other three: `Hdf5Compression(algorithm="blosc")` routes through
`hdf5plugin.Blosc2` unconditionally, so a Python writer cannot opt into the v1 filter C# can read, and zstd and lzf have
no C# reader at all. Read support for filters 32026, 32015, and 32000 is a post-v1 follow-up tracked on the issue
tracker.
