# Canonical Type Grammar

Grammar version: **1**

Created: 2026-08-05 Last updated: 2026-08-05

The language-neutral contract for versionable Schema Hashes. Every implementation (Python, C#) renders its native field
types into the strings defined here, joins them into a payload, and hashes that payload. Two schemas that are logically
the same **must** produce byte-identical hashes in every language
([ADR-0001](../docs/adr/0001-cross-language-schema-hash.md)).

The grammar is Python-flavored in appearance. That is a historical accident, not a statement about ownership: it is a
spec, and C# maps into it.

Machine-readable vectors: [`hash-vectors.json`](hash-vectors.json). Both implementations must pass every vector.

## 1. Schema Hash algorithm

Given a schema's fields as `{wireName: nativeType}`:

1. Render each `nativeType` to its **canonical type string** (sections 3–10).
2. Form one `name:canonicalType` pair per field — no spaces around the `:`.
3. Sort the pairs **by wire name**, ascending, by Unicode code point (see §2).
4. Join the sorted pairs with `,` — a comma, **no space**.
5. Encode the payload as UTF-8, take `sha256`, and keep the **first 6 characters** of the lowercase hex digest.

```text
hash = sha256(payload_utf8).hexdigest()[:6]
```

Worked example — fields declared as `label: str`, `count: int`, `tags: set[str]`:

```text
pairs (declaration order):  label:str  count:int  tags:set[str]
pairs (sorted by name):     count:int  label:str  tags:set[str]
payload:                    count:int,label:str,tags:set[str]
sha256 hex:                 fe82a7...
hash:                       fe82a7
```

A schema with no fields hashes the empty payload: `sha256("")[:6]` = `e3b0c4`.

**Separator trap.** Field pairs are joined with a bare comma, no space. Type arguments _inside_ brackets are separated
by a comma followed by exactly one space (`", "`). Getting this backwards is the single most likely cross-language hash
divergence.

## 2. Sorting is ordinal

Both sorts in this spec — field names (§1 step 3) and `Union` members (§6) — compare strings by **Unicode code point**,
ascending. This is Python's default `sorted()` on `str`.

C# must use `StringComparer.Ordinal` (or `string.CompareOrdinal`). The default `string.Compare` / `List<string>.Sort()`
are culture-aware and **will** produce a different order — and therefore a different hash — for mixed-case names.

```text
fields: zeta: int, alpha: str, Mid: bool
payload: Mid:bool,alpha:str,zeta:int        # all uppercase sorts before all lowercase
hash: 0ed89c
```

Consequences worth internalising:

- `"Mid" < "alpha"` (`M` = U+004D, `a` = U+0061).
- `"Node" < "None"` (`d` = U+0064 < `n` = U+006E) — a natural trap in unions.
- `"None" < "int"` but `"Decimal" < "None"` — `None` has **no** privileged position (§6).

**Sort the names, not the pairs.** Step 3 of §1 sorts by **wire name only**. Sorting the assembled `name:type` pairs
instead gives a different order whenever one name is a prefix of another and the longer name's next character sorts
below `:` (U+003A) — which includes every digit, `-`, `.`, and `/`:

```text
fields: a: str, a1: int
sort by name  (correct):  "a" < "a1"          ->  a:str,a1:int      hash 479c1a
sort by pair  (wrong):    "a1:int" < "a:str"  ->  a1:int,a:str      hash 193875
```

Both orders are self-consistent, so this bug is invisible until a cross-language hash comparison fails. Vector
`sort-by-name-not-pair` locks the correct result.

### Wire name restrictions

- A wire name MUST NOT contain `:` or `,` — they are the payload's field and pair separators, and a name containing
  either makes the payload ambiguous. `[` and `]` are likewise reserved, so that a payload can always be re-split by
  bracket depth.
- In grammar version 1, wire names and `Literal` string values are restricted to the Basic Multilingual Plane
  (U+0000–U+FFFF). Outside the BMP, C#'s `StringComparer.Ordinal` compares UTF-16 code units — surrogate pairs
  (U+D800–U+DFFF) then sort _below_ U+E000–U+FFFF, the opposite of code-point order. Restricting to the BMP makes
  ordinal and code-point order agree, so both implementations can use their native ordinal comparison unchanged.
- The payload is hashed as **UTF-8 bytes**, never UTF-16. C# must use `Encoding.UTF8.GetBytes`. Vector `non-ascii-utf8`
  locks this.

## 3. Grammar

```text
canonicalType := scalar | container | union | array | literal | serializationName

scalar        := "int" | "float" | "str" | "bool" | "bytes" | "complex" | "None"

container     := "list["      canonicalType "]"
               | "set["       canonicalType "]"
               | "frozenset[" canonicalType "]"
               | "dict["      canonicalType ", " canonicalType "]"
               | fixedTuple
               | variadicTuple

fixedTuple    := "tuple[" canonicalType { ", " canonicalType } "]"   ; order-significant
variadicTuple := "tuple[" canonicalType ", ...]"                     ; homogeneous, any length

union         := "Union[" canonicalType { ", " canonicalType } "]"   ; >= 2 members, ordinal-sorted

array         := "ndarray" | "ndarray[" dtype "]"

literal       := "Literal[" literalValue { ", " literalValue } "]"   ; order-significant
literalValue  := "'" escapedString "'"                               ; str
               | [ "-" ] digit { digit }                             ; int
               | "True" | "False"                                    ; bool
               | "None"                                              ; null
               | serializationName "." memberName                    ; enum member
                                                                     ; any other kind is REJECTED

serializationName := identifier                                      ; bare; never parameterized
```

## 4. Scalars

| Token     | Meaning                            |
| --------- | ---------------------------------- |
| `int`     | Integer of any width               |
| `float`   | Binary floating point of any width |
| `str`     | Unicode text                       |
| `bool`    | Boolean                            |
| `bytes`   | Opaque byte string                 |
| `complex` | Complex number                     |
| `None`    | The null/absent value              |

**Scalar width is erased** ([ADR-0002](../docs/adr/0002-array-dtype-hash-significant.md)). C# `byte`, `sbyte`, `short`,
`ushort`, `int`, `uint`, `long`, `ulong` all render `int`; `float` and `double` both render `float`. A Python
`int`/`float` field is width-less, so widening a C# field is a lossless, language-local detail. Out-of-range values fail
loudly at load, not at hash time.

C# `char` erases the same way, into `str`: Python has no character type, and a one-character string is what a `char` is
on the wire. The same erasure applies to a `Literal` option — `[LiteralValues('a')]` renders `Literal['a']`,
indistinguishable from the string spelling.

`bool` is **never** collapsed into `int`, in either direction.

`None` is most often seen as a `Union` member (§6), but it is also legal as a **standalone** field type — a field whose
only possible value is null (Python `x: None`). That is a Python-only construct: C# has no type whose sole value is
null, so a C# mirror of such a schema cannot be written. Vector `all-scalars` includes a standalone `None` field and is
labelled Python-only.

Array element dtype is the deliberate exception to width erasure (§7).

## 5. Containers

| Canonical       | Meaning                         | Python          | C#                            |
| --------------- | ------------------------------- | --------------- | ----------------------------- |
| `list[T]`       | ordered, homogeneous            | `list[T]`       | `List<T>`, `T[]`              |
| `dict[K, V]`    | mapping                         | `dict[K, V]`    | `Dictionary<K, V>`            |
| `set[T]`        | unordered, unique               | `set[T]`        | `HashSet<T>`                  |
| `frozenset[T]`  | unordered, unique, immutable    | `frozenset[T]`  | `FrozenSet<T>`                |
| `tuple[A, B]`   | fixed arity, per-position types | `tuple[A, B]`   | `(A, B)` / `ValueTuple<A, B>` |
| `tuple[T, ...]` | any arity, one element type     | `tuple[T, ...]` | (no equivalent)               |

- Arguments are separated by `", "` — comma plus exactly one space.
- Containers nest to any depth with no special casing: `list[dict[str, tuple[int, float]]]`.
- A **fixed** tuple's arguments are **order-significant**: `tuple[int, str]` and `tuple[str, int]` are different
  schemas.
- A **variadic** tuple renders the trailing `...` as the literal three-character token `...` — the canonical string is
  spelled exactly as the Python annotation, `tuple[int, ...]`. The token `...` is legal only in this one position: as
  the second and last argument of a `tuple`. C# has no variadic-tuple type, so C# implementations need only _read_ this
  form (for cross-language hash verification), never emit it.
- `byte[]` in C# renders `bytes`, **not** `list[int]` — see §7.
- These six forms, plus `Union`, `ndarray`, and `Literal`, are the **complete** set of parameterized canonical types.
  Every other type renders as a bare Serialization Name with any type parameters dropped (§9).

## 6. Unions

```text
Union[A, B, ...]
```

- Members are rendered first, then sorted **ordinally by their rendered string** (§2). Declaration order is irrelevant,
  so `Union[str, int, bool]` and `Union[bool, str, int]` both render `Union[bool, int, str]`.
- Optionality is a `None` member: Python `Optional[T]` / `T | None` and C# `T?` both render `Union[None, T]` **subject
  to sorting**.
- Duplicate members collapse; a "union" of one member renders as that member alone.

`None` is not pinned to any position — it sorts like every other token:

| Declared                         | Canonical                    | Hash     |
| -------------------------------- | ---------------------------- | -------- |
| `Optional[int]` / `int?`         | `Union[None, int]`           | `3621d6` |
| `Optional[Decimal]` / `decimal?` | `Union[Decimal, None]`       | `2aed1f` |
| `Decimal \| None \| Path`        | `Union[Decimal, None, Path]` | `70861c` |
| `Leaf \| Node \| None`           | `Union[Leaf, Node, None]`    | `ee8d0e` |

That last row is the classic trap: `Node` sorts _before_ `None`.

C# requires `<Nullable>enable</Nullable>`; nullable reference types and nullable value types are treated identically.

## 7. Arrays

```text
ndarray            # dtype dynamic (un-parameterized)
ndarray[<dtype>]   # dtype fixed and hash-significant
```

Array element dtype is **hash-significant** and is the deliberate exception to scalar width erasure
([ADR-0002](../docs/adr/0002-array-dtype-hash-significant.md)): arrays are bulk, physically-typed data (an HDF5 dataset
has a real on-disk dtype), and a silent `float64` → `float32` change is exactly the drift the hash exists to catch.
**Shape is erased** — only the dtype token appears, so a 3-D `float64` array and a 1-D `float64` array hash identically.

Supported dtype tokens (numpy dtype `.name` values) and their C# `Tensor<T>` element types:

| dtype token  | numpy           | C# `Tensor<T>`            |
| ------------ | --------------- | ------------------------- |
| `bool`       | `np.bool_`      | `bool`                    |
| `int8`       | `np.int8`       | `sbyte`                   |
| `int16`      | `np.int16`      | `short`                   |
| `int32`      | `np.int32`      | `int`                     |
| `int64`      | `np.int64`      | `long`                    |
| `uint8`      | `np.uint8`      | `byte`                    |
| `uint16`     | `np.uint16`     | `ushort`                  |
| `uint32`     | `np.uint32`     | `uint`                    |
| `uint64`     | `np.uint64`     | `ulong`                   |
| `float16`    | `np.float16`    | `Half`                    |
| `float32`    | `np.float32`    | `float`                   |
| `float64`    | `np.float64`    | `double`                  |
| `complex64`  | `np.complex64`  | (none — Python-only)      |
| `complex128` | `np.complex128` | `System.Numerics.Complex` |

Notes:

- `Tensor<int>` → `ndarray[int32]`, **not** `ndarray[int]`. Array dtypes are never width-erased; scalars always are.
- `byte[]` is a **carve-out**: it renders `bytes` (§4), not an array. Use `Tensor<byte>` for a `uint8` array.
- Bare `ndarray` is the dynamic-dtype escape hatch. C# has no equivalent (`Tensor<T>` is always typed), so a bare
  `ndarray` field is Python-only and cannot be mirrored in C#.
- **0-d (scalar) arrays cannot materialize in C#.** Shape is erased from the hash, so a 0-d `float64` array and a 1-D
  one render the same `ndarray[float64]` and a Python schema holding one hashes identically in C#; the file, however,
  will not load. `System.Numerics.Tensors` normalizes an empty shape to rank 1 length 0 —
  `Tensor.Create(new[] { 7.0 }, [])` reports `FlattenedLength == 0` — so a numpy scalar array has no faithful
  `Tensor<T>` representation. The C# NPY/NPZ codec reads and writes shape `()` correctly; the converter refuses to hand
  one to `Tensor<T>` rather than silently reshaping it. Store the value as a scalar field, or as a shape-`(1,)` array,
  if the file must be readable from both languages.
- Python performs runtime dtype validation on dtype-annotated fields: safe casts are applied silently, unsafe mismatches
  raise.
- The pre-version-1 leaked form `ndarray[tuple[typing.Any, Ellipsis], numpy.dtype[numpy.float64]]` is gone (§12).

## 8. Literals

```text
Literal['fast', 'slow']
Literal[1, 2, 3]
Literal['auto', 0, 'off', 1]
```

- No `typing.` prefix.
- Members are **order-significant** — they are _not_ sorted. `Literal['fast', 'slow']` (`08e2ae`) and
  `Literal['slow', 'fast']` (`23d37e`) are different schemas.
- Members are separated by `", "`.
- Because strings are quoted, `Literal['1']` (`7696a1`) and `Literal[1]` (`24682e`) are distinguishable. Prior to
  grammar version 1 both rendered `1` — that was a bug (§12).

Member rendering by value kind, in this precedence order:

| Kind        | Rendering                             | Example                   |
| ----------- | ------------------------------------- | ------------------------- |
| enum member | `<SerializationName>.<MEMBER_NAME>`   | `Literal[Status.ACTIVE]`  |
| null        | `None`                                | `Literal['a', None]`      |
| boolean     | `True` / `False`                      | `Literal[True, False]`    |
| integer     | bare decimal, leading `-` if negative | `Literal[-1, 0, 1]`       |
| string      | single-quoted, escaped (below)        | `Literal['fast', 'slow']` |

Precedence matters twice, and the order of that table is normative:

- **Enum member is tested first**, before string and integer. A mixin enum (`class Colour(str, Enum)` in Python, an
  `enum` with an underlying integral type in C#) _is_ a string / an integer, so a value-first test would render
  `Literal[Colour.RED]` as `'red'` — collapsing it into the distinct schema `Literal['red']`. The member reference wins.
  Vector `literal-mixin-enum-member` locks this.
- **Boolean is tested before integer**, because in Python `bool` is a subclass of `int`. `Literal[True]` must render
  `True`, never `1`.

String escaping inside the single quotes, applied in this order: `\` → `\\`, then `'` → `\'`. Nothing else is escaped —
the payload is UTF-8, so non-ASCII characters are emitted verbatim. So the value `it's` renders `'it\'s'` and the value
`a\b` renders `'a\\b'`.

**The list above is closed.** A `Literal` member of any other kind — `bytes`, `float`, a tuple, an arbitrary object — is
**rejected**: the implementation must raise at schema-definition time rather than fall back to a language-specific
representation such as Python's `repr()`. A `repr()` fallback would leak Python object formatting into a
language-neutral grammar and silently produce a hash no other implementation can reproduce.

C# declares literal sets with `[LiteralValues("fast", "slow")]` on a plain property; the attribute argument order is the
canonical order. Enum-valued members have no C# counterpart in v1.

## 9. Serialization Names

Enums, Versionable classes, and custom converter types render as their **Serialization Name**: a bare identifier, with
no module path, namespace, assembly, or enclosing-class qualification
([ADR-0001](../docs/adr/0001-cross-language-schema-hash.md)).

- The default is the bare class name: `Status`, not `mypkg.config.Status`; `datetime`, not `datetime.datetime`; `Path`,
  not `pathlib.Path`. A class nested inside another renders only its own name (`Inner`, not `Outer.Inner`).
- An explicit override is available in both languages, and the declared name is what hashes. Use it when two classes in
  different modules/namespaces would otherwise collide, or when a class is renamed but its files must keep loading.
- Serialization Names are decoupled from language namespaces on purpose: moving a file, renaming a module, or choosing a
  different C# namespace must not change a hash.
- A type that is not a Versionable, an enum, or a registered converter type renders its bare class name too.

**Uniqueness is required.** Serialization Names MUST be unique across every type reachable from a schema. Dropping
module paths and namespaces flattens all types into one namespace, so two same-named classes in different modules now
collide and would hash identically while meaning different things. Implementations MUST detect a duplicate and reject it
with an error naming both types and pointing at the override mechanism — silently accepting the collision defeats the
point of the hash.

**Serialization Names are never parameterized.** Type parameters on any type outside the closed parameterized set (§5:
the six containers, plus `Union`, `ndarray`, `Literal`) are **dropped**. So `re.Pattern[str]` renders `Pattern`,
`MyBox[int]` renders `MyBox`, and a generic Versionable renders its bare name. The parameter is erased on the wire
anyway, and preserving it would require every implementation to agree on how to render language-specific generic
arguments. Vector `parameterized-non-container` locks this.

Declaring an override:

| Kind                | Python                                            | C#                                 |
| ------------------- | ------------------------------------------------- | ---------------------------------- |
| `Versionable` class | `class Foo(Versionable, ..., name="Bar")`         | `[SerializationName("Bar")]`       |
| Enum                | `VERSIONABLE_NAME` attribute, after the enum body | `[SerializationName("Bar")]`       |
| Converter type      | `registerConverter(..., name="Bar")`              | `IWireConverter.SerializationName` |
| Anything else       | `setSerializationName(SomeType, "Bar")`           | `[SerializationName("Bar")]`       |

C# unifies Python's three declaration sites into one attribute, which applies to classes, structs, enums, and interfaces
alike; `[Versionable]` therefore has no `Name` member. The one case C# cannot express is Python's
`setSerializationName()` applied to a type the caller does not own — an attribute has to go on the declaration. A
foreign type reachable from a schema either keeps its bare name or is wrapped. Note also that an explicit
`[SerializationName]` wins over the built-in converter names in the table below: a type declaring one renders that name
even where a converter would otherwise have supplied one.

Built-in converter types and their canonical names:

| Canonical         | Python                    | C#                           | Wire form                      |
| ----------------- | ------------------------- | ---------------------------- | ------------------------------ |
| `datetime`        | `datetime.datetime`       | `DateTime`, `DateTimeOffset` | ISO 8601                       |
| `date`            | `datetime.date`           | `DateOnly`                   | ISO 8601                       |
| `time`            | `datetime.time`           | `TimeOnly`                   | ISO 8601                       |
| `timedelta`       | `datetime.timedelta`      | `TimeSpan`                   | total seconds (float)          |
| `Path`            | `pathlib.Path`            | `FilePath` (library type)    | string                         |
| `PurePosixPath`   | `pathlib.PurePosixPath`   | (none)                       | string                         |
| `PureWindowsPath` | `pathlib.PureWindowsPath` | (none)                       | string                         |
| `Decimal`         | `decimal.Decimal`         | `decimal`                    | string                         |
| `UUID`            | `uuid.UUID`               | `Guid`                       | lowercase-hyphenated string    |
| `Pattern`         | `re.Pattern`              | `Regex`                      | pattern string (flags dropped) |

Note `decimal` maps to `Decimal`, **not** to the `float` scalar — it is a converter type, not a numeric width.

Enums hash by name only; their members and values are not part of the hash. Changing a member's _value_ therefore does
not change the schema hash.

## 10. Annotated / attribute metadata

`Annotated[T, ...]` unwraps to `T`; the metadata is ignored at any nesting depth. So `Annotated[float, 'dB']` renders
`float`, and `list[Annotated[float, 'volts']]` renders `list[float]`.

The same rule holds for C# attributes that carry documentation or validation metadata: they do not appear in the hash.
Two are exceptions, because both change the payload itself (§1): `[VersionableField("...")]` changes a field's **wire
name**, and `[VersionableIgnore]` removes the field from the schema outright. Python needs no counterpart to the latter
— a dataclass field is an annotated attribute, so anything that should not persist simply goes unannotated.

## 11. C# → grammar mapping (summary)

| C#                                                                 | Canonical                      |
| ------------------------------------------------------------------ | ------------------------------ |
| `byte`, `sbyte`, `short`, `ushort`, `int`, `uint`, `long`, `ulong` | `int`                          |
| `nint`, `nuint`                                                    | `int`                          |
| `char`                                                             | `str` (a one-character string) |
| `float`, `double`                                                  | `float`                        |
| `string`                                                           | `str`                          |
| `bool`                                                             | `bool`                         |
| `byte[]`                                                           | `bytes` (carve-out)            |
| `System.Numerics.Complex`                                          | `complex`                      |
| `decimal`                                                          | `Decimal`                      |
| `Guid`                                                             | `UUID`                         |
| `DateTime`, `DateTimeOffset`                                       | `datetime`                     |
| `DateOnly` / `TimeOnly` / `TimeSpan`                               | `date` / `time` / `timedelta`  |
| `Regex`                                                            | `Pattern`                      |
| `FilePath`                                                         | `Path`                         |
| `T?` (nullable value type or nullable reference type)              | `Union[None, T]` (sorted)      |
| `List<T>`, `T[]` (except `byte[]`)                                 | `list[T]`                      |
| `Dictionary<K, V>`                                                 | `dict[K, V]`                   |
| `HashSet<T>`                                                       | `set[T]`                       |
| `FrozenSet<T>`                                                     | `frozenset[T]`                 |
| `(A, B)` / `ValueTuple<A, B>`                                      | `tuple[A, B]`                  |
| `Tensor<double>`                                                   | `ndarray[float64]` (§7)        |
| `enum` type                                                        | Serialization Name             |
| Versionable class                                                  | Serialization Name             |
| `[LiteralValues(...)]` property                                    | `Literal[...]`                 |

## 12. Changes in grammar version 1

Version 1 is the first _versioned_ grammar. It is a **breaking change** relative to the pre-1 Python output and ships in
the same generation as the C# debut ([ADR-0004](../docs/adr/0004-shared-generation-versioning.md)). Schemas containing
any of the following get new hashes:

| Construct          | Pre-1 rendering                                                    | Version 1                  |
| ------------------ | ------------------------------------------------------------------ | -------------------------- |
| Enum               | `mypkg.config.Status`                                              | `Status`                   |
| Converter type     | `datetime.datetime`, `pathlib.Path`, `uuid.UUID`                   | `datetime`, `Path`, `UUID` |
| Unregistered type  | `mypkg.helpers.Thing`                                              | `Thing`                    |
| Parameterized type | `re.Pattern[str]`                                                  | `Pattern`                  |
| Typed array        | `ndarray[tuple[typing.Any, Ellipsis], numpy.dtype[numpy.float64]]` | `ndarray[float64]`         |
| Variadic tuple     | `tuple[int, Ellipsis]`                                             | `tuple[int, ...]`          |
| Literal            | `typing.Literal[fast, slow]`                                       | `Literal['fast', 'slow']`  |
| Literal `'1'`/`1`  | both `1` (collision bug)                                           | `'1'` vs `1`               |
| Literal (other)    | Python `repr()` of the value                                       | rejected (§8)              |

Unchanged: the hash algorithm itself, scalar tokens, non-tuple container forms, union sorting, and `Annotated`
unwrapping.

Newly enforced in version 1 (no hash change, but previously-accepted schemas may now be rejected): Serialization Name
uniqueness (§9), the closed `Literal` member-kind list (§8), and the wire-name character and BMP restrictions (§2).

### Version policy

- `grammarVersion` is an integer, bumped on any change that can alter a hash for an existing schema.
- A bump is a lockstep, two-implementation release. Python and C# never ship different grammar versions in the same
  generation.
- `hash-vectors.json` records the version it describes; vectors are appended, and existing vectors' hashes change only
  alongside a version bump.

## 13. `hash-vectors.json`

```json
{
  "_comment": "...",
  "grammarVersion": 1,
  "vectors": [
    {
      "name": "ndarray-float64",
      "note": "Python NDArray[np.float64]; C# Tensor<double>.",
      "fields": { "data": "ndarray[float64]" },
      "payload": "data:ndarray[float64]",
      "hash": "9ffa65",
      "mustMatch": ["ndarray-shape-erased"],
      "mustDiffer": ["ndarray-float32"]
    }
  ]
}
```

| Key                | Required | Meaning                                                                              |
| ------------------ | -------- | ------------------------------------------------------------------------------------ |
| `name`             | yes      | Unique, stable vector id. Test failures cite it.                                     |
| `note`             | yes      | Description, incl. the source annotation when it differs from the canonical string.  |
| `fields`           | yes      | `wireName` → canonical type string.                                                  |
| `payload`          | yes      | The `,`-joined, name-sorted pairs (§1). Redundant, so a bad sort is caught directly. |
| `hash`             | yes      | 6 lowercase hex characters.                                                          |
| `pythonOnly`       | no       | `true` when the construct has no C# equivalent; the C# suite skips these.            |
| `csharpDeclarable` | no       | `false` when C# can _read_ but not _declare_ the form; the C# suite excuses these.   |
| `mustMatch`        | no       | Vector names whose `hash` must equal this one's. Reciprocal.                         |
| `mustDiffer`       | no       | Vector names whose `hash` must differ from this one's. Reciprocal.                   |

`pythonOnly` and `csharpDeclarable` are not opposites and never apply to the same vector. `pythonOnly` marks a construct
with no C# form at all — a standalone `None` field, a bare `ndarray`, a variadic tuple — so neither language can be
asked to agree about it. `csharpDeclarable: false` marks a form C# must still _reproduce the hash of_, because a
Python-written file can contain it, but which no C# type declaration can produce: the only current case is a `Union` of
three or more members, since C# spells optionality as `T?` and has no n-ary union type. A C# implementation passes such
a vector by reading its `payload`, not by declaring a fixture.

There is a third case, which the vectors do not carry but the golden corpus does: a schema whose C# mirror is _loadable
but not hash-identical_, because one of its fields uses a construct C# renders differently. The `containers` fixture
(`tuple[float, ...]` → `list[float]`, `846dc8` → `51917e`) and the `stdlib` fixture (`PurePosixPath` / `PureWindowsPath`
→ `Path`, `c3ff88` → `b38e82`) are the two current instances, recorded under `knownGaps` in
[`golden/index.json`](golden/index.json). **The consequence is a standing constraint: the envelope hash cannot become a
load-time cross-language gate.** A reader that rejected a file whose stored hash disagreed with the loading type's would
refuse those two fixtures in C# while accepting them in Python. The hash is a definition-time schema-drift check — the
analyzer in C#, class definition in Python — and any future load-time validation must exclude or special-case them.

**`fields` key order is informational only.** Some vectors deliberately declare fields out of sorted order to exercise
§1 step 3, but JSON object key order is not guaranteed by every parser. `payload` is the **authoritative** ordered form:
an implementation that reads `fields` must sort the names itself, and a consumer that cannot rely on key order should
read `payload`.

An implementation passes a vector by (a) declaring a fixture type whose fields render to the given canonical strings,
(b) rebuilding the payload from its own rendering, and (c) reproducing `hash`. Checking only `payload` → `hash` tests
`sha256`, not the grammar — the rendering step is the part that matters.

`mustMatch` / `mustDiffer` make the interesting relationships mechanically checkable rather than prose: e.g.
`ndarray-float64` and `ndarray-shape-erased` share a payload and hash on purpose (shape erasure), while
`ndarray-float64` and `ndarray-float32` must never collide (dtype significance, ADR-0002).
