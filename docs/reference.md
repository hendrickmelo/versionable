# Reference

## Versionable Class Parameters

Full signature:

```python
@dataclass
class MyClass(
    Versionable,
    version=1,
    hash="abc123",
    name="MyClass",
    old_names=["OldName"],
    register=True,
    skip_defaults=False,
    unknown="ignore",
):
    ...
```

| Parameter       | Type        | Default     | Description                                               |
| --------------- | ----------- | ----------- | --------------------------------------------------------- |
| `version`       | `int`       | required    | Schema version. Increment on breaking field changes.      |
| `hash`          | `str`       | recommended | 6-char hash of field names/types. Checked at import.      |
| `name`          | `str`       | class name  | Serialization name for output metadata and registry.      |
| `old_names`     | `list[str]` | `[]`        | Previous names; allows loading old files.                 |
| `register`      | `bool`      | `True`      | Add class to global registry (for `loadDynamic`).         |
| `skip_defaults` | `bool`      | `False`     | Omit fields equal to their class default from output.     |
| `unknown`       | `str`       | `"ignore"`  | Unrecognised fields: `"ignore"`, `"error"`, `"preserve"`. |

## Opting Out of Registration

By default, every `Versionable` subclass is added to a global registry keyed by its serialization name. This registry
powers `loadDynamic()`, which looks up the class from the `__OBJECT__` metadata in a file.

Set `register=False` when a class shouldn't participate in this registry:

```python
# Abstract base — never serialized directly
@dataclass
class SensorBase(Versionable, version=1, register=False):
    timestamp: datetime

# Test fixture — avoid name collisions with production classes
@dataclass
class Sample(Versionable, version=1, register=False):
    value: int
```

Common use cases:

- **Test classes** — test suites often define throwaway classes with generic names like `Sample`. Without
  `register=False`, these collide with each other across tests.
- **Abstract base classes** — intermediate classes that define shared fields but are never saved to disk.
- **Multiple versions in one process** — if you need two definitions of the same schema (e.g. for migration testing),
  only one can be registered.

Duplicate name detection: if two registered classes share the same serialization name, versionable raises a
`VersionableError` at class definition time with a suggested fix:

```text
VersionableError: Versionable name 'MyClass' is already registered to
mypackage.models.MyClass. Give one of the classes an explicit name to
disambiguate, e.g.: class MyClass(Versionable, ..., name="other.MyClass")
```

## Field Serialization Rules

A field is included in serialization if it:

- Has a type annotation
- Does **not** have a leading underscore in its name
- Is not a `ClassVar`

```python
@dataclass
class Example(Versionable, version=1, hash="..."):
    included: int                    # serialized
    _private: int = 0               # excluded — underscore prefix
    class_var = "constant"          # excluded — no annotation
    CONSTANT: ClassVar[int] = 42    # excluded — ClassVar
```

## metadata()

Inspect the schema metadata registered for a class:

```python
from versionable import metadata

meta = metadata(SensorConfig)
meta.version   # int — current version
meta.hash      # str — 6-char hash
meta.name      # str — serialization name
meta.fields    # list[str] — serialized field names
```

## Versionable.hash()

Compute the schema hash for a `Versionable` subclass. Use this to get the value to put in the `hash=` parameter:

```python
from dataclasses import dataclass
from versionable import Versionable

@dataclass
class MyConfig(Versionable, version=1):
    name: str
    value: float

print(MyConfig.hash())  # e.g. "4b7866"
```

Then add the result to the class definition:

```python
@dataclass
class MyConfig(Versionable, version=1, hash="4b7866"):
    name: str
    value: float
```

## Reserved Keys

The following dunder keys are used internally by versionable and must not be used as field names or as keys in
user-provided dict values:

| Key           | Purpose                                                   |
| ------------- | --------------------------------------------------------- |
| `__OBJECT__`  | Serialization class name (stored in metadata)             |
| `__VERSION__` | Schema version (stored in metadata)                       |
| `__HASH__`    | Schema hash (stored in metadata)                          |
| `__meta__`    | Metadata mapping in YAML and TOML files                   |
| `__ndarray__` | Marks a dict as a serialized numpy array                  |
| `__json__`    | YAML-only wrapper for values with no native YAML encoding |

> ⚠️ **Warning:** Using any of these as a field name or dict key will cause incorrect serialization or deserialization.
