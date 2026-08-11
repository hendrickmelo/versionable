# versionable

Versioned persistence for structured objects: files carry a version number and a schema fingerprint, and load cleanly
across schema changes via declarative migrations. Originally Python; a C# (.NET) implementation with interchangeable
files is planned.

## Language

**Schema Hash**: The first 6 hex characters of the SHA-256 of a class's fields rendered in the Canonical Type Grammar.
Identical across language implementations for the same logical schema — it fingerprints the schema, not the language.
_Avoid_: checksum, version hash

**Canonical Type Grammar**: The language-neutral textual form of field types used in Schema Hash payloads (`int`,
`float`, `str`, `list[T]`, `Union[None, T]`, `ndarray`). Python-flavored in appearance, but a cross-language spec: each
implementation maps its native types into it. Numeric width is erased for scalars: every integral type maps to `int`,
every floating type to `float`; out-of-range values fail loudly at load. Array element dtype is NOT erased: parametrized
arrays canonicalize to `ndarray[<dtype>]` (numpy dtype names as tokens, e.g. `ndarray[float64]`), enforced at runtime;
bare `ndarray` is the dynamic-dtype escape hatch. _Avoid_: type name, Python type string

**Serialization Name**: The stable name a Versionable class — and any enum or custom type appearing in a schema — is
known by in files and hashes. Defaults to the bare class name; explicitly overridable. Decoupled from language
namespaces (Python module paths, C# namespaces) so renames/moves and cross-language use don't change identity. _Avoid_:
class name, qualified name

**Wire Name**: The key a field is stored under in files and hashed by. Always the user's declared field/property name,
verbatim, in every language; the only exception is an explicit per-property override in C# (`[VersionableField]`). No
implicit casing transformation, ever. _Avoid_: serialized name, JSON name

**Envelope**: The `__versionable__` metadata block written alongside an object's fields: its Serialization Name
(`object`), `version`, and `hash`. At load time only `version` is authoritative (drives migrations); `hash` is
provenance. _Avoid_: header, metadata dict

**Migration**: A declared transformation of serialized field data from version N to N+1 of a schema, applied at load
time.
