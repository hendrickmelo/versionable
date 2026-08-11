# Array dtype is hash-significant; scalar width is not

Scalar numeric width is erased in the canonical type grammar (C# `int`/`long` → `int`, `float`/`double` → `float`)
because a Python `int`/`float` field is width-less and widening a C# field is a lossless, language-local detail. Array
element dtype is the deliberate exception: `NDArray[np.float64]` / `Tensor<double>` canonicalize to `ndarray[float64]`
(numpy dtype names as canonical tokens; shape is erased; bare `np.ndarray` stays `ndarray`). Arrays are bulk,
physically-typed data — HDF5 datasets have a real on-disk dtype — and a precision change (float64 → float32) is exactly
the kind of silent schema drift the hash exists to catch: without it, old float64 files would load silently in Python
while statically-typed C# (`Tensor<float>`) errors on the lossy conversion, diverging cross-language.

## Consequences

- Python gains runtime dtype validation for parametrized array annotations (safe casts applied, unsafe casts error) —
  otherwise the hash would fingerprint an annotation that nothing enforces.
- Changing an array field's dtype now changes the hash, forcing a version bump and (typically) a `convert` migration.
- The previously leaked canonical form (`ndarray[tuple[typing.Any, Ellipsis], numpy.dtype[numpy.float64]]`) is replaced
  by `ndarray[float64]` in the same breaking release as ADR-0001.
