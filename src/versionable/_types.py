"""Type converter registry and built-in converters.

Converts between Python types and serialization-friendly primitives.
Each converter handles serialization (Python → primitive) and
deserialization (primitive → Python) for a specific type.
"""

from __future__ import annotations

import base64
import datetime
import io
import logging
import re
import typing
import uuid
from decimal import Decimal
from enum import Enum
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Protocol, runtime_checkable

import numpy as np

from versionable._base import Versionable, _resolveFields
from versionable.errors import ConverterError

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# VersionableValue protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class VersionableValue(Protocol):
    """Protocol for types that know how to serialize themselves."""

    def toValue(self) -> Any: ...

    @classmethod
    def fromValue(cls, value: Any) -> Any: ...


# ---------------------------------------------------------------------------
# Converter registry
# ---------------------------------------------------------------------------


class _TypeConverter:
    """Stores a serializer/deserializer pair for a type."""

    def __init__(
        self,
        tp: type,
        serialize: Any,
        deserialize: Any,
        *,
        matchSubclasses: bool = False,
    ) -> None:
        self.type = tp
        self.serialize = serialize
        self.deserialize = deserialize
        self.matchSubclasses = matchSubclasses


class ConverterRegistry:
    """Registry of type converters with resolution by type."""

    def __init__(self) -> None:
        self._exact: dict[type, _TypeConverter] = {}
        self._subclass: list[_TypeConverter] = []

    def register(
        self,
        tp: type,
        serialize: Any,
        deserialize: Any,
        *,
        matchSubclasses: bool = False,
    ) -> None:
        conv = _TypeConverter(tp, serialize, deserialize, matchSubclasses=matchSubclasses)
        if matchSubclasses:
            self._subclass.append(conv)
        else:
            self._exact[tp] = conv

    def get(self, tp: type) -> _TypeConverter | None:
        """Find a converter for *tp*: exact match first, then subclass match."""
        if tp in self._exact:
            return self._exact[tp]
        for conv in self._subclass:
            if isinstance(tp, type) and issubclass(tp, conv.type):
                return conv
        return None


# Module-level singleton registry
_registry = ConverterRegistry()


def registerConverter(
    tp: type,
    serialize: Any,
    deserialize: Any,
    *,
    matchSubclasses: bool = False,
) -> None:
    """Register a custom type converter."""
    _registry.register(tp, serialize, deserialize, matchSubclasses=matchSubclasses)


# ---------------------------------------------------------------------------
# Built-in Tier 1 converters
# ---------------------------------------------------------------------------

_registry.register(
    datetime.datetime,
    serialize=lambda v: v.isoformat(),
    deserialize=lambda v, _tp: datetime.datetime.fromisoformat(v),
)
_registry.register(
    datetime.date,
    serialize=lambda v: v.isoformat(),
    deserialize=lambda v, _tp: datetime.date.fromisoformat(v),
)
_registry.register(
    datetime.time,
    serialize=lambda v: v.isoformat(),
    deserialize=lambda v, _tp: datetime.time.fromisoformat(v),
)
_registry.register(
    datetime.timedelta,
    serialize=lambda v: v.total_seconds(),
    deserialize=lambda v, _tp: datetime.timedelta(seconds=v),
)
_registry.register(
    Path,
    serialize=lambda v: str(v),
    deserialize=lambda v, _tp: Path(v),
    matchSubclasses=True,
)
_registry.register(
    PurePosixPath,
    serialize=lambda v: str(v),
    deserialize=lambda v, _tp: PurePosixPath(v),
)
_registry.register(
    PureWindowsPath,
    serialize=lambda v: str(v),
    deserialize=lambda v, _tp: PureWindowsPath(v),
)
_registry.register(
    uuid.UUID,
    serialize=lambda v: str(v),
    deserialize=lambda v, _tp: uuid.UUID(v),
)
_registry.register(
    Decimal,
    serialize=lambda v: str(v),
    deserialize=lambda v, _tp: Decimal(v),
)
_registry.register(
    bytes,
    serialize=lambda v: base64.b64encode(v).decode("ascii"),
    deserialize=lambda v, _tp: base64.b64decode(v),
)
_registry.register(
    complex,
    serialize=lambda v: [v.real, v.imag],
    deserialize=lambda v, _tp: complex(v[0], v[1]),
)
_registry.register(
    re.Pattern,
    serialize=lambda v: v.pattern,
    deserialize=lambda v, _tp: re.compile(v),
    matchSubclasses=True,
)


# ---------------------------------------------------------------------------
# Literal validation
# ---------------------------------------------------------------------------

_LITERAL_FALLBACK_KEY = "_literalFallback"


def literalFallback(value: Any) -> Any:
    """Declare a ``Literal`` field default with a fallback for invalid values.

    Use this as the default for a ``Literal``-typed field. When a file
    contains a value outside the declared options, versionable logs a
    warning and returns *value* instead of raising ``ConverterError``.

    Example::

        @dataclass
        class Config(Versionable, version=1, hash="..."):
            mode: Literal["fast", "slow"] = literalFallback("fast")
    """
    import dataclasses

    return dataclasses.field(default=value, metadata={_LITERAL_FALLBACK_KEY: value})


# ---------------------------------------------------------------------------
# Serialize / Deserialize entry points
# ---------------------------------------------------------------------------


def serialize(value: Any, fieldType: Any, *, nativeTypes: set[type] | None = None) -> Any:
    """Serialize *value* to a JSON-compatible primitive.

    Args:
        value: The Python value to serialize.
        fieldType: The declared type annotation for the field.
        nativeTypes: Types that the backend handles natively (skip conversion).
    """
    if value is None:
        return None

    nativeTypes = nativeTypes or set()

    # Try typed value dispatch first
    result = _serializeTyped(value, nativeTypes)
    if result is not _UNHANDLED:
        return result

    # Collections (need fieldType for element type info)
    return _serializeCollection(value, fieldType, nativeTypes)


_UNHANDLED = object()  # sentinel


def _serializeTyped(value: Any, nativeTypes: set[type]) -> Any:  # noqa: PLR0911 — type-dispatch; many returns are inherent
    """Try to serialize based on value type.  Returns _UNHANDLED if not matched."""
    valueType = type(value)

    if valueType in nativeTypes:
        return value
    if isinstance(value, (int, float, str, bool)):
        return value
    if isinstance(value, VersionableValue):
        return value.toValue()

    conv = _registry.get(valueType)
    if conv is not None:
        return conv.serialize(value)

    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Versionable):
        return _serializeVersionable(value)
    if isinstance(value, np.ndarray):
        return _serializeNdarray(value)

    return _UNHANDLED


def _serializeCollection(value: Any, fieldType: Any, nativeTypes: set[type]) -> Any:
    """Serialize collection types (dict, list, tuple, set, frozenset)."""
    args = typing.get_args(fieldType)

    if isinstance(value, dict):
        valType = args[1] if len(args) > 1 else Any
        return {str(k): serialize(v, valType, nativeTypes=nativeTypes) for k, v in value.items()}

    if isinstance(value, (list, tuple)):
        elemType = args[0] if args else Any
        return [serialize(v, elemType, nativeTypes=nativeTypes) for v in value]

    if isinstance(value, (set, frozenset)):
        elemType = args[0] if args else Any
        return [serialize(v, elemType, nativeTypes=nativeTypes) for v in sorted(value, key=repr)]

    # Fallback
    return value


def deserialize(
    data: Any,
    fieldType: Any,
    *,
    nativeTypes: set[type] | None = None,
    fieldMetadata: Any | None = None,
    validateLiterals: bool = True,
) -> Any:
    """Deserialize *data* back to the declared *fieldType*.

    Args:
        data: The primitive value from storage.
        fieldType: The declared type annotation for the field.
        nativeTypes: Types the backend handles natively (skip conversion).
        fieldMetadata: Optional dataclass field metadata (for literal fallbacks).
        validateLiterals: Whether to validate Literal type values.
    """
    if data is None:
        return None

    nativeTypes = nativeTypes or set()

    # Unwrap Annotated — strip metadata, keep the inner type
    origin = typing.get_origin(fieldType)
    if origin is typing.Annotated:
        fieldType = typing.get_args(fieldType)[0]
        origin = typing.get_origin(fieldType)

    args = typing.get_args(fieldType)

    # Literal — validate value against allowed options
    if origin is typing.Literal:
        if validateLiterals and data not in args:
            fallback = (fieldMetadata or {}).get(_LITERAL_FALLBACK_KEY)
            if fallback is not None:
                logger.warning(
                    "Value %r is not a valid Literal option %s. Using fallback %r.",
                    data,
                    list(args),
                    fallback,
                )
                return fallback
            raise ConverterError(f"Value {data!r} is not a valid Literal option. Allowed values: {list(args)}")
        return data

    if origin is typing.Union or _isUnionType(origin):
        return _deserializeUnion(
            data, args, nativeTypes, fieldMetadata=fieldMetadata, validateLiterals=validateLiterals
        )

    # Resolve the concrete type and dispatch
    concreteType = origin or fieldType
    return _deserializeConcrete(data, concreteType, args, nativeTypes)


def _deserializeUnion(
    data: Any,
    args: tuple[Any, ...],
    nativeTypes: set[type],
    fieldMetadata: Any | None = None,
    validateLiterals: bool = True,
) -> Any:
    """Deserialize a Union type by trying each member."""
    nonNoneArgs = [a for a in args if a is not type(None)]
    if len(nonNoneArgs) == 1:
        return deserialize(
            data,
            nonNoneArgs[0],
            nativeTypes=nativeTypes,
            fieldMetadata=fieldMetadata,
            validateLiterals=validateLiterals,
        )
    for arg in nonNoneArgs:
        try:
            return deserialize(
                data, arg, nativeTypes=nativeTypes, fieldMetadata=fieldMetadata, validateLiterals=validateLiterals
            )
        except (TypeError, ValueError, ConverterError):
            continue
    return data


def _deserializeConcrete(  # noqa: PLR0911 — type-dispatch; many returns are inherent
    data: Any,
    concreteType: Any,
    args: tuple[Any, ...],
    nativeTypes: set[type],
) -> Any:
    """Deserialize to a concrete (non-union) type."""
    # Backend native types — pass through
    if isinstance(concreteType, type) and concreteType in nativeTypes:
        return data

    # Primitives
    if concreteType in (int, float, str, bool):
        return data if isinstance(data, concreteType) else concreteType(data)

    # VersionableValue protocol
    if isinstance(concreteType, type) and _implementsVersionableValue(concreteType):
        # _implementsVersionableValue guard above ensures fromValue exists; mypy can't see it.
        return concreteType.fromValue(data)  # type: ignore[attr-defined]

    # Registered converter
    if isinstance(concreteType, type):
        conv = _registry.get(concreteType)
        if conv is not None:
            return conv.deserialize(data, concreteType)

    # Enum
    if isinstance(concreteType, type) and issubclass(concreteType, Enum):
        return _deserializeEnum(data, concreteType)

    # Nested Versionable
    if isinstance(concreteType, type) and issubclass(concreteType, Versionable):
        return _deserializeVersionable(data, concreteType)

    # numpy ndarray (both explicit ndarray and npt.NDArray alias)
    if concreteType is np.ndarray or (isinstance(concreteType, type) and issubclass(concreteType, np.ndarray)):
        return _deserializeNdarray(data)

    # Collections
    if concreteType in (list, tuple, set, frozenset):
        return _deserializeSequence(data, concreteType, args, nativeTypes)

    if concreteType is dict:
        return _deserializeDict(data, args, nativeTypes)

    # Fallback
    return data


def _deserializeSequence(
    data: Any,
    concreteType: type,
    args: tuple[Any, ...],
    nativeTypes: set[type],
) -> Any:
    """Deserialize list, tuple, set, or frozenset."""
    elemType = args[0] if args else Any
    items = [deserialize(v, elemType, nativeTypes=nativeTypes) for v in data]
    if concreteType is tuple:
        return tuple(items)
    if concreteType is set:
        return set(items)
    if concreteType is frozenset:
        return frozenset(items)
    return items


def _deserializeDict(
    data: Any,
    args: tuple[Any, ...],
    nativeTypes: set[type],
) -> dict[str, Any]:
    """Deserialize a dict."""
    keyType = args[0] if args else str
    valType = args[1] if len(args) > 1 else Any
    return {
        deserialize(k, keyType, nativeTypes=nativeTypes): deserialize(v, valType, nativeTypes=nativeTypes)
        for k, v in data.items()
    }


# ---------------------------------------------------------------------------
# Versionable helpers
# ---------------------------------------------------------------------------


def _serializeVersionable(obj: Versionable) -> dict[str, Any]:
    """Serialize a Versionable instance to a dict with metadata envelope."""
    from versionable._base import metadata as getMeta

    meta = getMeta(type(obj))
    fields = _resolveFields(type(obj))
    result: dict[str, Any] = {
        "__OBJECT__": meta.name,
        "__VERSION__": meta.version,
        "__HASH__": meta.hash,
    }
    for fieldName, fieldType in fields.items():
        value = getattr(obj, fieldName)
        result[fieldName] = serialize(value, fieldType)
    return result


def _deserializeVersionable(data: dict[str, Any], cls: type[Versionable]) -> Versionable:
    """Deserialize a dict to a Versionable instance."""
    import dataclasses

    meta = cls._serializer_meta_
    fields = _resolveFields(cls)
    dcFields = {f.name: f for f in dataclasses.fields(cls)}  # type: ignore[arg-type]
    kwargs: dict[str, Any] = {}
    for fieldName, fieldType in fields.items():
        if fieldName in data:
            dcField = dcFields.get(fieldName)
            dcMeta = dcField.metadata if dcField is not None else None
            kwargs[fieldName] = deserialize(
                data[fieldName], fieldType, fieldMetadata=dcMeta, validateLiterals=meta.validateLiterals
            )
        elif fieldName in dcFields:
            dcField = dcFields[fieldName]
            if dcField.default is not dataclasses.MISSING:
                kwargs[fieldName] = dcField.default
            elif dcField.default_factory is not dataclasses.MISSING:
                kwargs[fieldName] = dcField.default_factory()
    return cls(**kwargs)


# ---------------------------------------------------------------------------
# ndarray helpers (inline JSON fallback)
# ---------------------------------------------------------------------------


def _serializeNdarray(arr: np.ndarray) -> dict[str, Any]:
    """Serialize a numpy array to a base64-encoded npz representation."""
    buf = io.BytesIO()
    np.savez_compressed(buf, data=arr)
    encoded = base64.b64encode(buf.getvalue()).decode("ascii")
    return {"__ndarray__": True, "dtype": str(arr.dtype), "shape": list(arr.shape), "data": encoded}


def _deserializeNdarray(data: Any) -> np.ndarray:
    """Deserialize a numpy array from its serialized representation."""
    if isinstance(data, np.ndarray):
        return data
    if isinstance(data, dict) and data.get("__ndarray__"):
        raw = base64.b64decode(data["data"])
        buf = io.BytesIO(raw)
        arr: np.ndarray = np.load(buf)["data"]
        return arr
    if isinstance(data, list):
        return np.asarray(data)
    raise ConverterError(f"Cannot deserialize ndarray from {type(data).__name__}")


# ---------------------------------------------------------------------------
# Enum helpers
# ---------------------------------------------------------------------------


def _deserializeEnum(value: Any, enumCls: type[Enum]) -> Enum:
    """Deserialize an enum value with fallback support.

    Fallback is declared via ``VERSIONABLE_FALLBACK`` class variable
    (must be set after the enum body, not inside it, to avoid sunder
    name restrictions)::

        class Status(Enum):
            ACTIVE = "active"
            UNKNOWN = "unknown"

        Status.VERSIONABLE_FALLBACK = Status.UNKNOWN
    """
    try:
        return enumCls(value)
    except ValueError:
        fallback: Enum | None = getattr(enumCls, "VERSIONABLE_FALLBACK", None)
        if fallback is not None:
            logger.warning(
                "Unknown %s value %r, using fallback %r",
                enumCls.__name__,
                value,
                fallback,
            )
            return fallback
        raise ConverterError(f"Unknown {enumCls.__name__} value: {value!r}") from None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _implementsVersionableValue(cls: type) -> bool:
    """Check if *cls* implements the VersionableValue protocol."""
    return hasattr(cls, "toValue") and hasattr(cls, "fromValue") and callable(cls.fromValue)


def _isUnionType(tp: Any) -> bool:
    """Check if *tp* is a union type (Python 3.10+ ``X | Y`` syntax)."""
    import types

    return tp is types.UnionType
