"""Schema hash computation for Versionable classes.

Produces a deterministic short hash from field names and their type
annotations.  The hash is used to detect accidental schema drift — if a
field is added, removed, renamed, or its type changes, the hash changes.

The canonical type names produced here are a language-neutral grammar
(ADR-0001): they must be reproducible from another implementation of
versionable, so they never contain module paths.  Every type renders as its
*serialization name* — the bare class name by default, overridable per type.
"""

from __future__ import annotations

import hashlib
import types
import typing
from enum import Enum
from typing import Any, Union

from versionable._arrays import canonicalArrayName, isNdarrayOrigin
from versionable.errors import UnsupportedTypeError, VersionableError

# Enum class attribute declaring an explicit serialization name.  Must be
# assigned after the enum body (like VERSIONABLE_FALLBACK) — an assignment
# inside the body would become an enum member.
_ENUM_NAME_ATTR = "VERSIONABLE_NAME"

# The only types that carry type parameters in the grammar.  Union, ndarray and
# Literal are handled before this set is consulted; every other type renders as
# a bare serialization name with its parameters dropped.
_PARAMETERIZED_CONTAINERS: frozenset[type] = frozenset({list, dict, set, frozenset, tuple})


def computeHash(fields: dict[str, Any]) -> str:
    """Compute a 6-character hex hash from a mapping of field names to types.

    Args:
        fields: Mapping of field name to resolved type annotation.

    Returns:
        First 6 hex characters of the SHA-256 digest.
    """
    parts = [f"{name}:{canonicalTypeName(fields[name])}" for name in sorted(fields)]
    payload = ",".join(parts)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:6]


def collectSerializationNames(fields: dict[str, Any]) -> dict[str, type]:
    """Return ``{serialization name: type}`` for every named type reachable from *fields*.

    Dropping module paths flattens every type into one namespace, so two
    same-named classes reachable from one schema would hash identically while
    meaning different things.  This walk follows containers, unions, literals'
    enum members and nested ``Versionable`` fields (cycle-safe) and rejects such
    a collision.

    Builtins, ``ndarray`` and unresolved forward references carry no
    serialization name and are skipped.

    Raises:
        VersionableError: Two different types share a serialization name.
    """
    names: dict[str, type] = {}
    visited: set[int] = set()
    for fieldType in fields.values():
        _collectNames(fieldType, names, visited)
    return names


def _collectNames(tp: Any, names: dict[str, type], visited: set[int]) -> None:
    """Walk *tp*, recording the serialization name of every named type it reaches."""
    origin = typing.get_origin(tp)
    args = typing.get_args(tp)

    if origin is typing.Annotated:
        _collectNames(args[0], names, visited)
        return

    if origin is typing.Literal:
        for value in args:
            if isinstance(value, Enum):
                _recordName(type(value), names, visited)
        return

    if origin is not None:
        if origin is Union or origin is types.UnionType or origin in _PARAMETERIZED_CONTAINERS:
            for arg in args:
                _collectNames(arg, names, visited)
            return
        # Parameters on a non-container are dropped; the origin carries the name.
        _recordName(origin, names, visited)
        return

    if isinstance(tp, type):
        _recordName(tp, names, visited)


def _recordName(tp: type, names: dict[str, type], visited: set[int]) -> None:
    """Record *tp*'s serialization name, then recurse into it if it is Versionable."""
    if tp in _CANONICAL_NAMES or isNdarrayOrigin(tp):
        return

    name = _baseTypeName(tp)
    existing = names.get(name)
    if existing is not None and existing is not tp:
        raise VersionableError(
            f"Serialization name {name!r} is claimed by two different types reachable from this schema: "
            f"{_describeType(existing)} and {_describeType(tp)}. Serialization names must be unique — the "
            f"canonical type grammar drops module paths, so both would hash identically. Give one of them an "
            f"explicit distinct name: name='...' for a Versionable class, a VERSIONABLE_NAME attribute assigned "
            f"after an enum body, registerConverter(..., name='...') for a converter type, or "
            f"setSerializationName(...) for anything else."
        )
    names[name] = tp

    if getattr(tp, "_serializer_meta_", None) is None or id(tp) in visited:
        return
    visited.add(id(tp))
    # Local import: _base imports this module at import time.
    from versionable._base import _resolveFields

    for fieldType in _resolveFields(tp).values():
        _collectNames(fieldType, names, visited)


def _describeType(tp: type) -> str:
    """Return a module-qualified description of *tp* for error messages only."""
    module = getattr(tp, "__module__", "")
    qualname = getattr(tp, "__qualname__", getattr(tp, "__name__", repr(tp)))
    return f"{module}.{qualname}" if module and module != "builtins" else str(qualname)


def canonicalTypeName(tp: Any) -> str:
    """Return a stable, canonical string representation of a type.

    This must be deterministic across Python versions.  It normalises
    generic aliases, unions, and common types to a consistent form.
    """
    # None / NoneType
    if tp is type(None):
        return "None"

    # typing special forms: Optional, Union, etc.
    origin = typing.get_origin(tp)
    args = typing.get_args(tp)

    # Annotated — unwrap and use only the underlying type (metadata is ignored)
    if origin is typing.Annotated:
        return canonicalTypeName(args[0])

    # Literal — values are rendered, not types; order is significant
    if origin is typing.Literal:
        return f"Literal[{', '.join(_literalValueName(a) for a in args)}]"

    # Union (including Optional[T] which is Union[T, None])
    if origin is Union or origin is types.UnionType:
        inner = sorted(canonicalTypeName(a) for a in args)
        return f"Union[{', '.join(inner)}]"

    # numpy arrays — shape is erased, dtype is hash-significant (ADR-0002)
    if isNdarrayOrigin(origin):
        return canonicalArrayName(args)

    # Generic aliases.  Only the closed container set is parameterized; type
    # parameters on anything else are dropped, so re.Pattern[str] renders
    # "Pattern" and MyBox[int] renders "MyBox".
    if origin is not None:
        originName = _baseTypeName(origin)
        if args and origin in _PARAMETERIZED_CONTAINERS:
            innerParts = ", ".join(canonicalTypeName(a) for a in args)
            return f"{originName}[{innerParts}]"
        return originName

    # Variadic tuple marker: tuple[int, ...] renders the literal token "..."
    if tp is Ellipsis:
        return "..."

    # Plain types
    if isinstance(tp, type):
        return _baseTypeName(tp)

    # Forward references (strings)
    if isinstance(tp, (str, typing.ForwardRef)):
        return tp.__forward_arg__ if isinstance(tp, typing.ForwardRef) else tp

    # Fallback: use repr
    return repr(tp)


# Canonical names for well-known types to avoid module-path differences
_CANONICAL_NAMES: dict[type, str] = {
    int: "int",
    float: "float",
    str: "str",
    bool: "bool",
    bytes: "bytes",
    complex: "complex",
    type(None): "None",
    list: "list",
    dict: "dict",
    set: "set",
    frozenset: "frozenset",
    tuple: "tuple",
}


# Explicit serialization-name overrides, keyed by exact type.  Populated by
# `setSerializationName` and by `registerConverter(..., name=...)`.
_SERIALIZATION_NAMES: dict[type, str] = {}


def setSerializationName(tp: type, name: str) -> None:
    """Declare the canonical serialization name used for *tp* in schema hashes.

    Types render as their bare class name by default.  Use this to pin a
    different name — for a type whose class name collides with another, or one
    whose Python name differs from the name the schema is defined by::

        setSerializationName(np.datetime64, "timestamp")

    ``Versionable`` subclasses declare their name with the ``name=`` class
    argument, enums with a ``VERSIONABLE_NAME`` class attribute, and converter
    types with ``registerConverter(..., name=...)``; this is the general form
    behind all three.

    Raises:
        VersionableError: *tp* is a ``Versionable`` subclass (use the ``name=``
            class argument, which the registry also keys on) or a builtin whose
            canonical name is fixed by the grammar.
    """
    if getattr(tp, "_serializer_meta_", None) is not None:
        raise VersionableError(
            f"{tp.__qualname__} is a Versionable subclass — declare its serialization name with "
            f'class {tp.__name__}(Versionable, ..., name="{name}") so the class registry uses it too, '
            f"rather than setSerializationName()."
        )
    if tp in _CANONICAL_NAMES:
        raise VersionableError(
            f"The canonical name of {tp.__name__!r} is fixed by the type grammar and cannot be overridden."
        )
    _SERIALIZATION_NAMES[tp] = name


def _baseTypeName(tp: type) -> str:
    """Return the serialization name for a type.

    Names are always bare — no module path — so that moving a class between
    modules does not change any hash, and so another language implementation
    can reproduce the name from the schema alone (ADR-0001).
    """
    explicit = _SERIALIZATION_NAMES.get(tp)
    if explicit is not None:
        return explicit

    if tp in _CANONICAL_NAMES:
        return _CANONICAL_NAMES[tp]

    # Versionable subclasses: use the declared serialization name
    serMeta = getattr(tp, "_serializer_meta_", None)
    if serMeta is not None:
        name: str = serMeta.name
        return name

    # Enum subclasses: VERSIONABLE_NAME if declared, else the bare class name.
    # Read from the class's own __dict__ so an enum never inherits a sibling's
    # serialization name from a shared mixin base.
    if isinstance(tp, type) and issubclass(tp, Enum):
        declared = tp.__dict__.get(_ENUM_NAME_ATTR)
        if isinstance(declared, str):
            return declared
        return tp.__name__

    # numpy ndarray (bare, unparametrised)
    module = getattr(tp, "__module__", "")
    qualname = getattr(tp, "__qualname__", getattr(tp, "__name__", repr(tp)))

    if module == "numpy" and qualname == "ndarray":
        return "ndarray"

    # Everything else — converter types (datetime, Path, Decimal, …) and
    # unregistered types alike — renders as the bare class name.
    return getattr(tp, "__name__", qualname)


def _literalValueName(value: Any) -> str:
    """Render a single ``Literal`` option.

    The accepted kinds are closed.  Precedence matters twice: enum members are
    tested first, so ``Literal[Colour.RED]`` of a ``str``-mixin enum renders
    ``Colour.RED`` rather than its value (two distinct schemas would otherwise
    collide); and ``bool`` is tested before ``int`` because it subclasses it.

    Raises:
        UnsupportedTypeError: *value* is of any other kind (float, bytes,
            tuple, arbitrary object).  Falling back to ``repr()`` would leak
            Python formatting into a language-neutral grammar.
    """
    if isinstance(value, Enum):
        return f"{_baseTypeName(type(value))}.{value.name}"
    if value is None:
        return "None"
    if isinstance(value, bool):
        return "True" if value else "False"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, str):
        escaped = value.replace("\\", "\\\\").replace("'", "\\'")
        return f"'{escaped}'"
    raise UnsupportedTypeError(
        f"Literal option {value!r} of type {type(value).__name__} cannot be expressed in the canonical type "
        f"grammar. Literal options must be a string, int, bool, None, or an enum member."
    )
