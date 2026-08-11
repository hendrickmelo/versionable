; Unshipped analyzer release
; https://github.com/dotnet/roslyn-analyzers/blob/main/src/Microsoft.CodeAnalysis.Analyzers/ReleaseTrackingAnalyzers.Help.md
; Not Markdown despite the extension: the release tracker parses a pipe-separated table with no
; leading or trailing pipes, so this file is excluded from prettier and markdownlint. Full rule
; descriptions and the severity knobs live in dotnet/.editorconfig.

### New Rules

Rule ID | Category | Severity | Notes
--------|----------|----------|-------
VSN0001 | Versionable | Error | Declared Hash disagrees with the hash computed from the fields; the message carries the canonical payload.
VSN0002 | Versionable | Error | A field type has no form in the canonical type grammar.
VSN0003 | Versionable | Error | The migration chain skips a version above its oldest entry, or declares a member that cannot be run as a migration.
VSN0004 | Versionable | Error | Two reachable types claim one Serialization Name.
VSN0005 | Versionable | Error | A [Versionable] type, or an enclosing type, is not declared partial.
VSN0006 | Versionable | Error | A [LiteralValues] option is outside the grammar's closed member list.
VSN0007 | Versionable | Error | A wire name contains a payload separator or leaves the Basic Multilingual Plane.
VSN0008 | Versionable | Error | Two members claim one wire name.
VSN0009 | Versionable | Warning | The type is compiled without nullable reference types, so optionality cannot be seen.
VSN0010 | Versionable | Error | No constructor can rebuild the type from its serializable members.
VSN0011 | Versionable | Warning | The type cannot carry a module initializer, so it never enters VersionableRegistry.
