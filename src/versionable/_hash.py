"""Schema hash computation for Versionable classes.

Produces a deterministic short hash from field names and their type
annotations.  The hash is used to detect accidental schema drift — if a
field is added, removed, renamed, or its type changes, the hash changes.
"""

from __future__ import annotations

import hashlib
import types
import typing
from enum import Enum
from typing import Any, Union


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

    # Union (including Optional[T] which is Union[T, None])
    if origin is Union or origin is types.UnionType:
        inner = sorted(canonicalTypeName(a) for a in args)
        return f"Union[{', '.join(inner)}]"

    # Generic aliases: list[int], dict[str, float], etc.
    if origin is not None:
        originName = _baseTypeName(origin)
        if args:
            innerParts = ", ".join(canonicalTypeName(a) for a in args)
            return f"{originName}[{innerParts}]"
        return originName

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


def _baseTypeName(tp: type) -> str:
    """Return the canonical name for a type, handling numpy and other common types."""
    if tp in _CANONICAL_NAMES:
        return _CANONICAL_NAMES[tp]

    # Versionable subclasses: use serialization name (not module path)
    # so that moving files doesn't change the hash
    serMeta = getattr(tp, "_serializer_meta_", None)
    if serMeta is not None:
        name: str = serMeta.name
        return name

    # Enum subclasses: use fully qualified name
    if isinstance(tp, type) and issubclass(tp, Enum):
        return f"{tp.__module__}.{tp.__qualname__}"

    # numpy ndarray
    module = getattr(tp, "__module__", "")
    qualname = getattr(tp, "__qualname__", getattr(tp, "__name__", repr(tp)))

    if module == "numpy" and qualname == "ndarray":
        return "ndarray"

    # For types in builtins, omit module
    if module == "builtins":
        return qualname

    return f"{module}.{qualname}"
