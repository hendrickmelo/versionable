# Scaffolding C# from Python

`python -m versionable.tools.to_csharp` turns Python `Versionable` classes into C# `[Versionable]` partial classes. It
is a one-shot scaffolder for porting an existing schema to the
[.NET implementation](https://www.nuget.org/packages/Versionable), not a code generator you run on every build: convert
once, review the output, and maintain the C# from then on.

```bash
python -m versionable.tools.to_csharp mypkg.schemas
python -m versionable.tools.to_csharp mypkg.schemas:Config --namespace MyPkg.Schemas --out ./cs
```

A target is a module (every `Versionable` it defines) or `module:Name` (one class or enum). Targets are **imported**,
not parsed, so what gets converted is exactly what Python resolved — the same field mapping `computeHash` hashes.

Output is one `.cs` file per source module, named after the module, printed to stdout unless `--out DIR` is given. C#
convention is one type per file, but a Python module is the unit that holds a schema together with the enums it
references; split the output afterwards if your project prefers it.

## The hash is copied verbatim, and that is the point

A canonical payload is language-neutral by construction ([GRAMMAR.md §1](../conformance/GRAMMAR.md)), so the same
logical schema hashes to the same six characters in both languages. The scaffolder copies each `hash=` literal straight
into `[Versionable(Hash = "...")]`, and the Roslyn analyzer recomputes it from the C# symbols on the next build. If the
emitted C# does not mean what the Python meant, you get a compile error naming both hashes and the payload — not a load
failure in production.

Nothing else is trusted to a scaffolder. Review the file, then let the compiler check it.

## What converts

| Python                                     | C#                                                      |
| ------------------------------------------ | ------------------------------------------------------- |
| `str`, `bool`, `bytes`, `complex`          | `string`, `bool`, `byte[]`, `Complex`                   |
| `int`, `float`                             | `long`, `double` (see below)                            |
| `list[T]`, `set[T]`                        | `List<T>`, `HashSet<T>`                                 |
| `dict[K, V]`, `frozenset[T]`               | `Dictionary<K, V>`, `FrozenSet<T>`                      |
| `tuple[A, B]`                              | `(A, B)`                                                |
| `T \| None`                                | `T?`                                                    |
| `datetime` / `date` / `time` / `timedelta` | `DateTime` / `DateOnly` / `TimeOnly` / `TimeSpan`       |
| `Decimal`, `UUID`, `re.Pattern`, `Path`    | `decimal`, `Guid`, `Regex`, `FilePath`                  |
| `npt.NDArray[np.float64]`                  | `Tensor<double>` (per the §7 dtype table)               |
| `Enum`                                     | `enum`, with `[EnumValue]` / `[EnumFallback]`           |
| `Literal[...]`                             | `[LiteralValues(...)]`, `Fallback` included             |
| nested `Versionable`, subclassing          | the type by name, `: Base` for a subclass               |
| `name="..."`, `VERSIONABLE_NAME`           | `[SerializationName("...")]`                            |
| field name                                 | PascalCase property + `[VersionableField("wire_name")]` |
| `Migrate.vN = Migration().rename(...)`     | `Migrate.VN = new Migration().Rename(...)`, op for op   |

Every class parameter — `version=`, `hash=`, `old_names=`, `register=`, `skip_defaults=`, `unknown=`,
`validate_literals=` — carries over to the matching `[Versionable]` member, and only the non-default ones are written
out.

**`int` becomes `long`, `float` becomes `double`.** Python `int` is unbounded, so `long` is the widest C# integral that
cannot lose a value the Python side could hold. Scalar width is erased from the grammar
([§4](../conformance/GRAMMAR.md)), so narrowing a field to `int` afterwards is a local choice that leaves the hash
alone.

**`datetime` becomes `DateTime`, or `DateTimeOffset` when the field's default is timezone-aware.** A Python annotation
does not record awareness, so the default is the only evidence available. Both spellings render the canonical
`datetime`, so a wrong guess costs a review comment, not a hash.

## What gets a `// TODO`

Anything with no C# spelling is emitted as a `// TODO` citing the grammar section that closes it off, with the Python
source commented underneath. **A file with no TODOs left compiles; a file with TODOs is a starting point.**

| Construct                                    | Why                                                              |
| -------------------------------------------- | ---------------------------------------------------------------- |
| `convert` / `derive` / `split` / `merge` ops | They carry a Python lambda. The emitted C# lambda throws.        |
| `@migration`-decorated methods               | Same. Emitted as a `[Migration]` stub that throws — see below.   |
| `default_factory=<callable>`                 | Only the empty-container builtins convert; the rest are bodies.  |
| A default with no C# literal form            | The property becomes `required` until you assign one.            |
| Sibling names PascalCasing alike             | `timeout_ms` + `timeoutMs` are one C# property (CS0102).         |
| `int \| str` and other n-ary unions          | C# spells optionality as `T?` and has no union type (§13).       |
| `tuple[T, ...]`                              | No variadic tuple in C# (§5). `List<T>` **hashes differently**.  |
| `PurePosixPath`, `PureWindowsPath`           | No counterpart (§9). `FilePath` **hashes differently**.          |
| Bare `np.ndarray`                            | `Tensor<T>` is always typed (§7). Declare a dtype in Python.     |
| `Literal[Colour.RED]`, or a nested `Literal` | No C# spelling in v1 (§8).                                       |
| A standalone `None` field                    | C# has no type whose only value is null (§4).                    |
| An unregistered type                         | Renders as its bare Serialization Name (§9); supply the C# type. |

The three rows that say **hashes differently** are the ones where the emitted C# loads the same bytes but does not agree
with the copied hash literal. The TODO says so at the field, so the analyzer's complaint on the next build is not a
surprise.

An imperative migration is emitted as a live `[Migration]` method whose body throws, rather than commented out: a gap in
the chain would fail the analyzer's contiguity check, and an empty body would let an unported migration silently drop
data at load time.

**A converted default is exact or it is not emitted.** Python's temporal types carry microseconds, and the obvious C#
spellings quietly lose them — `TimeSpan.FromSeconds` rounds to the nearest millisecond, the short `DateTime`
constructors stop at seconds. The scaffolder uses the microsecond-precision overloads .NET 7 added, so the full value
survives. Where it cannot — a UTC offset that is not a whole number of minutes, which `DateTimeOffset` rejects outright,
or a duration outside `TimeSpan`'s range — the field becomes `required` with a TODO rather than silently shifting.

## Other limitations

- **Referenced types are not pulled in.** Anything a converted class refers to but that this run did not emit — from
  another module, or from the same module when the target named a single class — is referenced by name and listed in the
  file header. Convert those too.
- **A subclass re-declares its base's `Migrate` with `new`.** Python finds an inherited `Migrate` through the MRO; C#
  does not inherit nested types, so the chain is emitted on both, and the subclass's copy carries `new` to keep CS0108
  quiet. Delete it if the subclass genuinely needs no chain.
- **`Migration().then(other)` is flattened.** Python composes the op lists at declaration time, so the emitted chain is
  one sequence rather than two.
- **Enum members are PascalCased** (`RED` becomes `Red`). The wire value is unaffected — it travels in `[EnumValue]`.
- **A lambda's commented source is the whole statement it was written in**, because that is all `inspect.getsource` can
  see. Source longer than 20 lines is truncated with a `file:line` pointer.
- **No XML doc comments on properties.** A project building with `GenerateDocumentationFile` will want to add them, or
  the emitted types will raise CS1591.
- The emitted code assumes `<Nullable>enable</Nullable>` (required by the grammar, §6) and C# 12 collection expressions,
  both of which are the `net8.0` defaults.
