# Supported Types

versionable handles a wide range of Python types out of the box. Primitives, collections, and common standard-library
types all serialize automatically — no registration needed. For anything else, you can register a custom converter or
implement the `VersionableValue` protocol.

## Primitives

`int`, `float`, `str`, `bool`, `None`

## Collections

`list[T]`, `dict[K, V]`, `set[T]`, `frozenset[T]`, `tuple[T, ...]`, `Optional[T]`

## Built-in Converters

These types are automatically serialized and deserialized without any registration:

| Type                 | Serialized as         |
| -------------------- | --------------------- |
| `datetime.datetime`  | ISO 8601 string       |
| `datetime.date`      | ISO 8601 string       |
| `datetime.time`      | ISO 8601 string       |
| `datetime.timedelta` | float (total seconds) |
| `pathlib.Path`       | string                |
| `uuid.UUID`          | string                |
| `decimal.Decimal`    | string                |
| `bytes`              | base64 string         |
| `complex`            | `[real, imag]`        |
| `re.Pattern`         | pattern string        |

## Enums

Enums are serialized by their `.value`:

```python
from enum import Enum
from dataclasses import dataclass
from versionable import Versionable

class Mode(Enum):
    FAST = "fast"
    SLOW = "slow"

@dataclass
class Config(Versionable, version=1, hash="..."):
    mode: Mode = Mode.FAST
```

### Enum Fallback

When you deprecate and remove an enum value, old files that still contain it will fail to load. Set
`VERSIONABLE_FALLBACK` on the enum class to gracefully handle this — unknown values deserialize to the fallback instead
of raising:

```python
class Status(Enum):
    ACTIVE = "active"
    UNKNOWN = "unknown"

Status.VERSIONABLE_FALLBACK = Status.UNKNOWN
```

## Nested Versionable

Fields typed as another `Versionable` subclass are serialized recursively:

```python
@dataclass
class Point(Versionable, version=1, hash="..."):
    x: float
    y: float

@dataclass
class Shape(Versionable, version=1, hash="..."):
    name: str
    origin: Point
```

## Numpy Arrays

- **HDF5:** Stored as native compressed datasets with lazy loading by default. See [Backends](backends.md).
- **JSON / TOML:** Stored as base64-compressed npz blobs.
- **YAML:** Stored as a `__json__` wrapper containing the base64-compressed npz blob as a JSON string.

## Custom Converters

If your dataclass uses a type that versionable doesn't handle natively, you have two options: register a converter
(this section) or implement the `VersionableValue` protocol (next section).

Use `registerConverter` when you need full control over serialization — for example, when the type comes from a
third-party library you can't modify, or when the serialized representation doesn't map cleanly to a single primitive
value.

```python
from versionable import registerConverter

class Coord:
    def __init__(self, lat: float, lon: float):
        self.lat = lat
        self.lon = lon

registerConverter(
    Coord,
    serialize=lambda v: {"lat": v.lat, "lon": v.lon},
    deserialize=lambda v, _cls: Coord(v["lat"], v["lon"]),
)
```

The `serialize` callable receives the value and returns a JSON-serializable object (primitives, lists, or dicts). The
`deserialize` callable receives that serialized value and the target type, and returns an instance.

Converters are registered globally and apply to all backends. Register them once at module level before any `save()` or
`load()` calls — typically alongside your class definition.

## VersionableValue Protocol

For types you own that map naturally to a single primitive value, implement the `toValue` / `fromValue` protocol
instead. This is lighter than `registerConverter` — no separate registration call needed.

```python
class UserId:
    def __init__(self, value: str):
        self._value = value

    def toValue(self) -> str:
        return self._value

    @classmethod
    def fromValue(cls, value: str) -> "UserId":
        return cls(value)
```

Any class with both `toValue` and `fromValue` is automatically detected — no registration required. This is the simplest
way to add serialization to your own types if you are not worried about version.
