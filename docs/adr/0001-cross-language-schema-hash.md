# Schema hashes are cross-language identical

The C# (.NET) implementation must produce byte-identical schema hashes for the same logical schema as Python. The
existing canonical type grammar (`int`, `float`, `list[T]`, `Union[None, T]`, `ndarray`, …) is declared the
language-neutral spec — it happens to look like Python, but C# maps into it (`double`→`float`, `long`→`int`,
`List<T>`→`list[T]`, `T?`→`Union[None, T]`, `byte[]`→`bytes`, `string`→`str`).

The alternative — language-local hashes, with the stored hash treated as opaque provenance — was considered twice and
rejected (note: nothing _mechanically_ forces matching; load never compares the file hash, only `version`). The deciding
argument: copying a hash literal from a Python class into its C# mirror makes the C# analyzer's independent
recomputation an exhaustive compile-time proof that the two schemas are structurally identical (optionality, dtypes,
nested names) — data round-trip tests only check the values they happen to exercise. The incremental cost is near zero
because the cross-language type mapping must exist anyway for wire compatibility, and the Python grammar cleanup (bare
names, ndarray form, Literal quoting) is justified independently — module-path leaks mean file moves change hashes
today. The accepted coupling: future grammar changes are lockstep two-implementation releases.

## Consequences

- Enums (and custom converter types) currently hash by Python module path (`mypkg.config.Status`), which cannot be
  reproduced from C#. They will get a declared serialization name instead. This changes existing hashes for schemas
  containing enums — a documented breaking change shipped under a new version number.
