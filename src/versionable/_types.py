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

from versionable._base import _REGISTRY, Versionable, _resolveFields
from versionable._numpy_compat import _np as np
from versionable._numpy_compat import requireNumpy
from versionable.errors import (
    BackendError,
    CircularReferenceError,
    ConverterError,
    UnknownFieldError,
    VersionError,
)

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


def serialize(
    value: Any,
    fieldType: Any,
    *,
    nativeTypes: set[type] | None = None,
    _visited: set[int] | None = None,
    _path: str = "",
) -> Any:
    """Serialize *value* to a JSON-compatible primitive.

    Args:
        value: The Python value to serialize.
        fieldType: The declared type annotation for the field.
        nativeTypes: Types that the backend handles natively (skip conversion).
        _visited: Set of ``id()`` values for ``Versionable`` instances
            currently on the serialization stack.  Used internally for
            cycle detection.  Backends should not pass this argument.
        _path: Field path of *value* relative to the root being
            serialized, used in cycle-error messages.  Internal.
    """
    if value is None:
        return None

    nativeTypes = nativeTypes or set()
    if _visited is None:
        _visited = set()

    # Try typed value dispatch first
    result = _serializeTyped(value, nativeTypes, _visited, _path)
    if result is not _UNHANDLED:
        return result

    # Collections (need fieldType for element type info)
    return _serializeCollection(value, fieldType, nativeTypes, _visited, _path)


_UNHANDLED = object()  # sentinel


# ---------------------------------------------------------------------------
# Cycle-detection path helpers
# ---------------------------------------------------------------------------
#
# Path format mirrors pytest-style: dotted field names (``parent.children``)
# with bracketed list indices (``children[0]``) and dict keys
# (``partners["alice"]``).  An empty path prints as ``<root>``.


def _appendField(path: str, fieldName: str) -> str:
    """Return *path* with *fieldName* appended as a dotted field segment."""
    return f"{path}.{fieldName}" if path else fieldName


def _appendIndex(path: str, index: int) -> str:
    """Return *path* with a bracketed list/sequence *index* appended."""
    return f"{path}[{index}]"


def _appendKey(path: str, key: Any) -> str:
    """Return *path* with a bracketed dict *key* appended (repr'd)."""
    return f"{path}[{key!r}]"


def _serializeTyped(value: Any, nativeTypes: set[type], visited: set[int], path: str) -> Any:
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
        return _serializeVersionable(value, visited, path)
    if np is not None and isinstance(value, np.ndarray):
        return _serializeNdarray(value)

    return _UNHANDLED


def _serializeCollection(
    value: Any,
    fieldType: Any,
    nativeTypes: set[type],
    visited: set[int],
    path: str,
) -> Any:
    """Serialize collection types (dict, list, tuple, set, frozenset)."""
    args = typing.get_args(fieldType)

    if isinstance(value, dict):
        # Reject Versionable dict keys: dict keys serialize via str(k), which would
        # silently produce a Python repr for Versionable instances and corrupt on
        # round-trip. Use Versionable as dict values, not keys.
        if args and isinstance(args[0], type) and issubclass(args[0], Versionable):
            raise ConverterError(
                f"Dict keys cannot be Versionable types ({args[0].__name__}) at field "
                f"{path or '<root>'}. Use Versionable as dict values, not keys."
            )
        valType = args[1] if len(args) > 1 else Any
        return {
            str(k): serialize(
                v,
                valType,
                nativeTypes=nativeTypes,
                _visited=visited,
                _path=_appendKey(path, k),
            )
            for k, v in value.items()
        }

    if isinstance(value, (list, tuple)):
        elemType = args[0] if args else Any
        return [
            serialize(
                v,
                elemType,
                nativeTypes=nativeTypes,
                _visited=visited,
                _path=_appendIndex(path, i),
            )
            for i, v in enumerate(value)
        ]

    if isinstance(value, (set, frozenset)):
        elemType = args[0] if args else Any
        return [
            serialize(
                v,
                elemType,
                nativeTypes=nativeTypes,
                _visited=visited,
                _path=_appendIndex(path, i),
            )
            for i, v in enumerate(sorted(value, key=repr))
        ]

    # Fallback
    return value


def deserialize(
    data: Any,
    fieldType: Any,
    *,
    nativeTypes: set[type] | None = None,
    fieldMetadata: Any | None = None,
    validateLiterals: bool | None = None,
    _classFallback: bool = True,
) -> Any:
    """Deserialize *data* back to the declared *fieldType*.

    Args:
        data: The primitive value from storage.
        fieldType: The declared type annotation for the field.
        nativeTypes: Types the backend handles natively (skip conversion).
        fieldMetadata: Optional dataclass field metadata (for literal fallbacks).
        validateLiterals: Whether to validate Literal type values. ``None`` means
            "no explicit override" — Literal validation falls back to the enclosing
            ``Versionable``'s class default (passed via ``_classFallback``).
        _classFallback: Internal. The enclosing ``Versionable``'s
            ``meta.validateLiterals``, used when ``validateLiterals`` is ``None``.
            Defaults to ``True`` for direct callers without a class context.
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
        effective = validateLiterals if validateLiterals is not None else _classFallback
        if effective and data not in args:
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
            data,
            args,
            nativeTypes,
            fieldMetadata=fieldMetadata,
            validateLiterals=validateLiterals,
            _classFallback=_classFallback,
        )

    # Resolve the concrete type and dispatch
    concreteType = origin or fieldType
    return _deserializeConcrete(
        data,
        concreteType,
        args,
        nativeTypes,
        validateLiterals=validateLiterals,
        _classFallback=_classFallback,
    )


def _deserializeUnion(
    data: Any,
    args: tuple[Any, ...],
    nativeTypes: set[type],
    fieldMetadata: Any | None = None,
    validateLiterals: bool | None = None,
    _classFallback: bool = True,
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
            _classFallback=_classFallback,
        )
    for arg in nonNoneArgs:
        try:
            return deserialize(
                data,
                arg,
                nativeTypes=nativeTypes,
                fieldMetadata=fieldMetadata,
                validateLiterals=validateLiterals,
                _classFallback=_classFallback,
            )
        except (TypeError, ValueError, ConverterError):
            continue
    return data


def _deserializeConcrete(
    data: Any,
    concreteType: Any,
    args: tuple[Any, ...],
    nativeTypes: set[type],
    *,
    validateLiterals: bool | None = None,
    _classFallback: bool = True,
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

    # Nested Versionable — pass the override through; the nested class will reset
    # _classFallback to its own meta.validateLiterals for its own field deserialization.
    if isinstance(concreteType, type) and issubclass(concreteType, Versionable):
        return _deserializeVersionable(data, concreteType, validateLiterals=validateLiterals)

    # numpy ndarray (both explicit ndarray and npt.NDArray alias)
    if np is not None and (
        concreteType is np.ndarray or (isinstance(concreteType, type) and issubclass(concreteType, np.ndarray))
    ):
        return _deserializeNdarray(data)

    # Collections
    if concreteType in (list, tuple, set, frozenset):
        return _deserializeSequence(
            data, concreteType, args, nativeTypes, validateLiterals=validateLiterals, _classFallback=_classFallback
        )

    if concreteType is dict:
        return _deserializeDict(
            data, args, nativeTypes, validateLiterals=validateLiterals, _classFallback=_classFallback
        )

    # Fallback
    return data


def _deserializeSequence(
    data: Any,
    concreteType: type,
    args: tuple[Any, ...],
    nativeTypes: set[type],
    *,
    validateLiterals: bool | None = None,
    _classFallback: bool = True,
) -> Any:
    """Deserialize list, tuple, set, or frozenset."""
    elemType = args[0] if args else Any
    items = [
        deserialize(
            v, elemType, nativeTypes=nativeTypes, validateLiterals=validateLiterals, _classFallback=_classFallback
        )
        for v in data
    ]
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
    *,
    validateLiterals: bool | None = None,
    _classFallback: bool = True,
) -> dict[str, Any]:
    """Deserialize a dict."""
    keyType = args[0] if args else str
    valType = args[1] if len(args) > 1 else Any
    return {
        deserialize(
            k, keyType, nativeTypes=nativeTypes, validateLiterals=validateLiterals, _classFallback=_classFallback
        ): deserialize(
            v, valType, nativeTypes=nativeTypes, validateLiterals=validateLiterals, _classFallback=_classFallback
        )
        for k, v in data.items()
    }


# ---------------------------------------------------------------------------
# Versionable helpers
# ---------------------------------------------------------------------------


def _serializeVersionable(obj: Versionable, visited: set[int], path: str) -> dict[str, Any]:
    """Serialize a Versionable instance to a dict with metadata envelope.

    Tracks ``id(obj)`` on a "currently-on-stack" set so cycles raise
    :class:`CircularReferenceError` instead of recursing until Python's
    stack limit.  The id is removed on the way back up so that a single
    instance reachable from two unrelated branches (a diamond) does not
    trip the check — diamonds are still duplicated on disk in 0.2.x.
    """
    from versionable._base import metadata as getMeta

    objId = id(obj)
    if objId in visited:
        raise CircularReferenceError(path, obj)

    meta = getMeta(type(obj))
    fields = _resolveFields(type(obj))
    result: dict[str, Any] = {
        "__versionable__": {
            "object": meta.name,
            "version": meta.version,
            "hash": meta.hash,
        },
    }
    visited.add(objId)
    try:
        for fieldName, fieldType in fields.items():
            value = getattr(obj, fieldName)
            result[fieldName] = serialize(
                value,
                fieldType,
                _visited=visited,
                _path=_appendField(path, fieldName),
            )
    finally:
        visited.discard(objId)
    return result


# ---------------------------------------------------------------------------
# Nested envelope reading and polymorphism resolution
# ---------------------------------------------------------------------------
#
# Nested ``Versionable`` data dicts can carry envelope keys in three layouts:
#
# 1. Wrapped (current 0.2.0+ JSON/YAML/TOML files):
#    ``data["__versionable__"] = {"object": ..., "version": N, "hash": ...}``
# 2. Flat new keys (current HDF5 read output, transient on the wire):
#    ``data`` has top-level ``"object"``, ``"version"``, ``"hash"``.
# 3. Flat dunder keys (0.1.x files):
#    ``data`` has top-level ``"__OBJECT__"``, ``"__VERSION__"``, ``"__HASH__"``.
#
# ``_readNestedEnvelope`` reads any of these layouts; ``_stripEnvelope`` removes
# all envelope keys before migrations operate on field-name keys.

_ENVELOPE_KEYS = frozenset(
    {
        "__versionable__",
        # Flat new (0.2.0+, current HDF5 read output)
        "object",
        "version",
        "hash",
        "format",
        "format_be",
        "shared_refs",
        # Flat dunder (0.1.x file format)
        "__OBJECT__",
        "__VERSION__",
        "__HASH__",
        "__FORMAT__",
        "__FORMAT_BE__",
        "__SHARED_REFS__",
    }
)


def _readNestedEnvelope(data: dict[str, Any]) -> dict[str, Any]:
    """Extract envelope fields (object, version, hash) from a nested Versionable's data dict.

    Handles wrapped (``data["__versionable__"]``), flat-new (``object``/``version``/``hash``
    at top level), and flat-dunder (``__OBJECT__``/``__VERSION__``/``__HASH__``) layouts.

    Returns a dict with keys ``object``, ``version``, ``hash``. Values are ``None`` when
    the corresponding envelope field is not present.
    """
    envelope = data["__versionable__"] if isinstance(data.get("__versionable__"), dict) else data
    return {
        "object": envelope.get("object", envelope.get("__OBJECT__")),
        "version": envelope.get("version", envelope.get("__VERSION__")),
        "hash": envelope.get("hash", envelope.get("__HASH__")),
    }


def _stripEnvelope(data: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of *data* with envelope keys removed.

    Migration ops operate on field-name keys; envelope keys (``__versionable__``,
    ``object``/``version``/``hash``, and the legacy dunder forms) must be removed before
    migrations run so they don't get treated as user fields.
    """
    return {k: v for k, v in data.items() if k not in _ENVELOPE_KEYS}


def _resolveNestedClass(envelope: dict[str, Any], cls: type[Versionable]) -> type[Versionable]:
    """Resolve the concrete class for a nested deserialize using the envelope's object name.

    Supports polymorphic collections: ``list[Animal]`` saved with ``Dog`` and ``Cat``
    elements is reconstructed with the original subclass identities. Falls back to
    *cls* (the declared field type) when the envelope is missing or matches the
    declared name, supporting back-compat and ``register=False`` classes.

    Args:
        envelope: Output of :func:`_readNestedEnvelope`.
        cls: The declared field type (e.g., ``Animal`` for ``field: list[Animal]``).

    Raises:
        BackendError: If the envelope's ``object`` is not in the registry, or resolves
            to a class that is not a subclass of *cls*.
    """
    objectName = envelope.get("object")
    if not objectName:
        # Missing envelope or no object key — back-compat with envelope-less data.
        return cls

    declaredName = cls._serializer_meta_.name if hasattr(cls, "_serializer_meta_") else cls.__name__
    if objectName == declaredName:
        # Same identity — use the declared type (also handles register=False classes
        # whose declared type isn't in the registry).
        return cls

    fromRegistry = _REGISTRY.get(objectName)
    if fromRegistry is None:
        raise BackendError(
            f"Unknown nested object type {objectName!r} (declared field type: {cls.__qualname__}). "
            f"Class is not registered or has been removed."
        )
    if not issubclass(fromRegistry, cls):
        raise BackendError(
            f"Nested object type {objectName!r} resolves to {fromRegistry.__qualname__}, "
            f"which is not a subclass of declared field type {cls.__qualname__}."
        )
    return fromRegistry


def _deserializeVersionable(
    data: dict[str, Any],
    cls: type[Versionable],
    *,
    validateLiterals: bool | None = None,
) -> Versionable:
    """Deserialize a dict to a Versionable instance.

    Reads the per-element envelope to:
    - Resolve the concrete class for polymorphic collections (e.g., ``list[Animal]``
      with ``Dog`` and ``Cat`` elements preserves subclass identity).
    - Apply migrations from the file's recorded version to the resolved class's version.
    - Honor the resolved class's ``unknown="error"``/``"ignore"``/``"preserve"`` policy
      against extra fields after migration.

    If the dict contains a ``__ver_lazy__`` key (set by the HDF5 backend's recursive
    lazy reader), lazy sentinels are passed through without deserialization and the
    instance is wrapped with ``makeLazyInstance``.

    Args:
        data: The serialized field dict for the Versionable instance, including
            envelope keys (any of the three supported layouts).
        cls: The declared field type. Used as a fallback when no envelope is present
            and as the polymorphism upper bound (``object`` must resolve to a subclass).
        validateLiterals: Explicit Literal-validation override. ``None`` falls back
            to each Versionable's class default; an explicit ``True``/``False``
            propagates everywhere.

    Raises:
        VersionError: File version is newer than the resolved class's version.
        BackendError: Polymorphism resolution failed (unknown name or wrong subclass).
        UnknownFieldError: Resolved class has ``unknown="error"`` and the data
            contains fields not declared on the class.
    """
    import dataclasses

    lazyFields: set[str] = data.pop("__ver_lazy__", set())

    # Polymorphism: resolve the concrete class from the envelope's object name.
    envelope = _readNestedEnvelope(data)
    actualCls = _resolveNestedClass(envelope, cls)
    meta = actualCls._serializer_meta_

    # Migration: strip envelope, then apply if file version differs.
    fileVersion = envelope.get("version")
    if fileVersion is None:
        # Missing envelope (0.1.x without envelope, hand-crafted file) — assume current.
        logger.warning(
            "Nested %s: no version found in data envelope; assuming current version (%d). "
            "If this data was written by an older version of the code, the load may fail.",
            meta.name,
            meta.version,
        )
        fieldsData = _stripEnvelope(data)
    elif fileVersion < meta.version:
        from versionable._migration import applyMigrationRange

        fieldsData = _stripEnvelope(data)
        fieldsData = applyMigrationRange(actualCls, fieldsData, fileVersion, meta.version)
    elif fileVersion > meta.version:
        raise VersionError(
            f"Nested {meta.name}: file version ({fileVersion}) is newer than class version "
            f"({meta.version}). Cannot downgrade."
        )
    else:
        fieldsData = _stripEnvelope(data)

    # Unknown-field handling per the resolved class's policy.
    fields = _resolveFields(actualCls)
    knownFieldNames = set(fields.keys())
    unknownFields = set(fieldsData.keys()) - knownFieldNames
    if unknownFields:
        if meta.unknown == "error":
            raise UnknownFieldError(f"Unknown fields in nested {meta.name}: {sorted(unknownFields)}")
        if meta.unknown == "ignore":
            for name in unknownFields:
                del fieldsData[name]
        # "preserve" mode: leave them; field iteration won't include them in kwargs.

    dcFields = {f.name: f for f in dataclasses.fields(actualCls)}  # type: ignore[arg-type]
    kwargs: dict[str, Any] = {}
    for fieldName, fieldType in fields.items():
        if fieldName in fieldsData:
            rawValue = fieldsData[fieldName]
            if fieldName in lazyFields or getattr(rawValue, "_isLazySentinel", False):
                # Lazy sentinel — pass through without deserialization
                kwargs[fieldName] = rawValue
            else:
                dcField = dcFields.get(fieldName)
                dcMeta = dcField.metadata if dcField is not None else None
                # Pass `validateLiterals` (the override) through unchanged so nested
                # Versionables can re-resolve at their own boundaries. Set
                # `_classFallback` to this class's setting so this class's own Literal
                # fields validate per its own declaration when no override is set.
                kwargs[fieldName] = deserialize(
                    rawValue,
                    fieldType,
                    fieldMetadata=dcMeta,
                    validateLiterals=validateLiterals,
                    _classFallback=meta.validateLiterals,
                )
        elif fieldName in dcFields:
            dcField = dcFields[fieldName]
            if dcField.default is not dataclasses.MISSING:
                kwargs[fieldName] = dcField.default
            elif dcField.default_factory is not dataclasses.MISSING:
                kwargs[fieldName] = dcField.default_factory()

    instance = actualCls(**kwargs)
    if lazyFields:
        from versionable._lazy import makeLazyInstance

        instance = makeLazyInstance(instance, lazyFields)
    return instance


# ---------------------------------------------------------------------------
# ndarray helpers (inline JSON fallback)
# ---------------------------------------------------------------------------


def _serializeNdarray(arr: Any) -> dict[str, Any]:
    """Serialize a numpy array to a base64-encoded npz representation."""
    numpy = requireNumpy("ndarray serialization")
    buf = io.BytesIO()
    numpy.savez_compressed(buf, data=arr)
    encoded = base64.b64encode(buf.getvalue()).decode("ascii")
    return {"__ver_ndarray__": True, "dtype": str(arr.dtype), "shape": list(arr.shape), "data": encoded}


def _deserializeNdarray(data: Any) -> Any:
    """Deserialize a numpy array from its serialized representation."""
    numpy = requireNumpy("ndarray deserialization")
    if isinstance(data, numpy.ndarray):
        return data
    if isinstance(data, dict) and (data.get("__ver_ndarray__") or data.get("__ndarray__")):
        raw = base64.b64decode(data["data"])
        buf = io.BytesIO(raw)
        return numpy.load(buf)["data"]
    if isinstance(data, list):
        return numpy.asarray(data)
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
