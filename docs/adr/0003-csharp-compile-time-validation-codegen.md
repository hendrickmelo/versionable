# C# uses Roslyn for compile-time hash validation and metadata codegen

The C# implementation ships a Roslyn analyzer + source generator alongside the library. The analyzer computes the schema
hash from the compiler's symbol model and reports a mismatch against the declared `Hash = "..."` as a build error
(severity configurable via `.editorconfig` — the analogue of Python's `ignoreHashErrors` dev mode). The generator emits
per-type metadata and typed accessors (field lists with canonical type info, getters/setters, constructor invokers,
`[ModuleInitializer]` registry registration); a single shared runtime engine (envelope, migrations, converters,
backends) consumes that metadata. This is stronger than Python's import-time tripwire — mismatches fail the build, not
the process start.

## Considered Options

- Runtime reflection with first-use + explicit assembly validation: simpler, but no Native AOT/trimming, and weaker than
  compile-time checking for no ongoing benefit once the Roslyn cost is paid.
- Full per-type serializer generation (à la System.Text.Json source-gen): fastest, but 4 backends × migrations ×
  polymorphism generated as text is a large fragile surface; migrations are user lambdas that run at runtime anyway.

## Consequences

- The canonical type grammar gets a Roslyn-side implementation (symbol → canonical name), shared by analyzer and
  generator — and reusable as the reference for the planned Python→C# converter.
- Native AOT and trimming are supported from v1 (no runtime reflection over user types).
