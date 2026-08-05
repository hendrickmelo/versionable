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
only at consumer sites. Shipping the analyzer inside the `Versionable` NuGet package is phase 5's job; nothing packs
today.

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
