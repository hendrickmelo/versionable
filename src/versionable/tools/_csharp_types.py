"""Python type annotations rendered as C# type spellings.

The inverse of the C#-to-canonical table in ``conformance/GRAMMAR.md`` section 11: given a
resolved Python annotation, produce the C# type whose canonical rendering is the *same*
string.  That equality is what lets the scaffolder copy a Python ``hash=`` literal into a C#
``[Versionable(Hash = ...)]`` verbatim and have the Roslyn analyzer recompute it and agree.

Two widths are chosen rather than derived.  Python ``int`` is unbounded and Python ``float``
is a C ``double``, so ``long`` and ``double`` are the widest spellings that cannot lose a
value the Python side could hold.  Scalar width is erased from the grammar (section 4), so
narrowing either afterwards is a language-local decision that leaves the hash alone.

Anything the grammar defines but C# cannot declare — an n-ary union, a variadic tuple, a bare
``ndarray``, a pure path — comes back as a :class:`Todo` citing the section that says so,
never as a guess that would silently produce a different hash.
"""

from __future__ import annotations

import datetime
import re
import types as pytypes
import typing
import uuid
from dataclasses import dataclass
from decimal import Decimal
from enum import Enum
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

from versionable._arrays import dtypeFromArgs, dtypeToken, isNdarrayOrigin
from versionable._base import Versionable
from versionable._hash import canonicalTypeName
from versionable.errors import UnsupportedTypeError

# Namespaces the scaffolder emits `using` directives for.  `System`,
# `System.Collections.Generic` and `System.Linq` are deliberately absent: the repo's
# Directory.Build.props turns on ImplicitUsings, which already supplies them.
NAMESPACE_VERSIONABLE = "Versionable"
NAMESPACE_NUMERICS = "System.Numerics"
NAMESPACE_TENSORS = "System.Numerics.Tensors"
NAMESPACE_REGEX = "System.Text.RegularExpressions"
NAMESPACE_FROZEN = "System.Collections.Frozen"


@dataclass(frozen=True)
class Todo:
    """One construct the scaffolder refused to guess at.

    Attributes:
        message: What could not be converted and why, citing the grammar section that
            closes the construct off.
        source: The Python source the human has to read to finish the job, if any.  Emitted
            as commented-out lines under the ``// TODO``.
    """

    message: str
    source: str = ""


@dataclass(frozen=True)
class MappedType:
    """A Python annotation's C# spelling, plus whatever it could not carry across.

    Attributes:
        text: The C# type as it is written in source.  When ``todos`` is non-empty this is a
            best-effort placeholder that will not necessarily hash to the same string —
            the TODO says so.
        usings: Namespaces the emitted file needs for ``text`` to resolve.
        todos: Unconvertible constructs found anywhere inside the annotation.
    """

    text: str
    usings: frozenset[str] = frozenset()
    todos: tuple[Todo, ...] = ()

    def withTodo(self, message: str, *, source: str = "") -> MappedType:
        """Return a copy of this mapping with one more :class:`Todo` appended."""
        return MappedType(self.text, self.usings, (*self.todos, Todo(message, source)))


# ---------------------------------------------------------------------------
# Static tables (GRAMMAR.md sections 4, 7, 9, 11)
# ---------------------------------------------------------------------------

# Scalars.  Width erasure (section 4) makes every integral spelling render `int` and both
# floating spellings render `float`, so the widest one is free.
_SCALARS: dict[type, tuple[str, frozenset[str]]] = {
    str: ("string", frozenset()),
    int: ("long", frozenset()),
    float: ("double", frozenset()),
    bool: ("bool", frozenset()),
    bytes: ("byte[]", frozenset()),
    complex: ("Complex", frozenset({NAMESPACE_NUMERICS})),
}

# Built-in converter types, matched by exact type (section 9).  `datetime` maps to
# `DateTime`, the naive spelling; `DateTimeOffset` is the timezone-aware one and renders the
# same canonical `datetime`, so the emitter picks between them from the field's default
# rather than from the annotation, which does not record awareness.
#
# Exact matches are a dict and are tried first, so resolution does not depend on declaration
# order.  It otherwise would: `datetime.datetime` is a subclass of `datetime.date`, so a
# subclass walk that happened to reach `date` first would render every `datetime` field
# `DateOnly` — a silently wrong type, and a silently wrong hash.
_CONVERTERS_EXACT: dict[type, tuple[str, frozenset[str]]] = {
    datetime.datetime: ("DateTime", frozenset()),
    datetime.date: ("DateOnly", frozenset()),
    datetime.time: ("TimeOnly", frozenset()),
    datetime.timedelta: ("TimeSpan", frozenset()),
    Decimal: ("decimal", frozenset()),
    uuid.UUID: ("Guid", frozenset()),
    re.Pattern: ("Regex", frozenset({NAMESPACE_REGEX})),
    Path: ("FilePath", frozenset({NAMESPACE_VERSIONABLE})),
}

# Converter types whose *subclasses* map too, mirroring the `matchSubclasses=True` converters
# in `_types.py`: `Path` is registered that way (so `PosixPath` and `WindowsPath` resolve),
# and `re.Pattern` is generic, so `re.compile(...)`'s type is a subclass rather than the
# class itself.  Ordered most-derived first, and consulted only after every exact match has
# missed.
_CONVERTERS_BY_SUBCLASS: tuple[tuple[type, tuple[str, frozenset[str]]], ...] = (
    (re.Pattern, ("Regex", frozenset({NAMESPACE_REGEX}))),
    (Path, ("FilePath", frozenset({NAMESPACE_VERSIONABLE}))),
)

# Array dtype tokens to `Tensor<T>` element types (section 7).  `complex64` is absent on
# purpose: the table in the grammar marks it Python-only.
_DTYPE_ELEMENTS: dict[str, tuple[str, frozenset[str]]] = {
    "bool": ("bool", frozenset()),
    "int8": ("sbyte", frozenset()),
    "int16": ("short", frozenset()),
    "int32": ("int", frozenset()),
    "int64": ("long", frozenset()),
    "uint8": ("byte", frozenset()),
    "uint16": ("ushort", frozenset()),
    "uint32": ("uint", frozenset()),
    "uint64": ("ulong", frozenset()),
    "float16": ("Half", frozenset()),
    "float32": ("float", frozenset()),
    "float64": ("double", frozenset()),
    "complex128": ("Complex", frozenset({NAMESPACE_NUMERICS})),
}

# Containers whose C# spelling is a single generic taking the same arguments in the same
# order (section 5).  `tuple` is not here: fixed and variadic tuples differ.
_CONTAINERS: dict[type, tuple[str, frozenset[str]]] = {
    list: ("List", frozenset()),
    set: ("HashSet", frozenset()),
    frozenset: ("FrozenSet", frozenset({NAMESPACE_FROZEN})),
    dict: ("Dictionary", frozenset()),
}

# The pure path flavours, which section 9 lists with a C# column of `(none)`.  Checked before
# the converter table so neither can be mistaken for `pathlib.Path`.
_UNCONVERTIBLE_PATHS: frozenset[type] = frozenset({PurePosixPath, PureWindowsPath})


def mapType(tp: Any) -> MappedType:
    """Return the C# spelling of the resolved Python annotation *tp*.

    Mirrors the shape of :func:`versionable._hash.canonicalTypeName`, production for
    production, so a change to one is obvious in the other.

    Args:
        tp: A resolved annotation, as :func:`versionable._base._resolveFields` returns.

    Returns:
        The C# type text plus the namespaces and :class:`Todo` entries it implies.  Never
        raises: an unconvertible construct becomes a ``Todo``, because the scaffolder's job
        is to hand a human a reviewable file, not to refuse one.
    """
    if tp is type(None):
        # A standalone `None` field is Python-only: C# has no type whose sole value is null.
        return MappedType("object?").withTodo(
            "a standalone `None` field has no C# type — GRAMMAR.md section 4 marks it Python-only. "
            "Drop the field or give it a real type."
        )

    origin = typing.get_origin(tp)
    args = typing.get_args(tp)

    if origin is typing.Annotated:
        # Metadata is ignored at any depth (section 10).
        return mapType(args[0])
    if origin is typing.Literal:
        return _mapNestedLiteral(args)
    if origin is typing.Union or origin is pytypes.UnionType:
        return _mapUnion(args)
    if isNdarrayOrigin(origin) or isNdarrayOrigin(tp):
        return mapArray(args)
    if origin is not None:
        return _mapGeneric(origin, args)
    return _mapPlain(tp)


def mapArray(args: tuple[Any, ...]) -> MappedType:
    """Return the ``Tensor<T>`` spelling for an ``ndarray[...]`` annotation's arguments.

    Shape is erased and the dtype is hash-significant (section 7), so the element type is
    the whole of the mapping.
    """
    try:
        dtype = dtypeFromArgs(args)
    except UnsupportedTypeError as exc:  # pragma: no cover — the class would not have defined
        return MappedType("object").withTodo(f"unmappable array dtype: {exc}")

    if dtype is None:
        return MappedType("object").withTodo(
            "a bare `ndarray` (no declared dtype) has no C# spelling — `Tensor<T>` is always typed, "
            "so GRAMMAR.md section 7 marks it Python-only. Declare a dtype "
            "(`npt.NDArray[np.float64]`) on the Python side, or drop the field."
        )

    token = dtypeToken(dtype)
    element = _DTYPE_ELEMENTS.get(token)
    if element is None:
        return MappedType("object").withTodo(
            f"array dtype `{token}` has no C# element type — GRAMMAR.md section 7 marks it Python-only."
        )
    text, usings = element
    return MappedType(f"Tensor<{text}>", usings | {NAMESPACE_TENSORS})


def literalKind(values: tuple[Any, ...]) -> MappedType:
    """Return the C# property type that spans a ``Literal``'s declared options.

    A literal field renders from its options, never from its declared type (section 8), so
    the type only has to *hold* every option.  ``object`` is the honest answer for a mixed
    set and is accepted here for exactly that reason — the C# golden fixture spells the
    mixed ``Literal['auto', 0, 'off', 1]`` the same way.
    """
    if not values:
        return MappedType("object").withTodo("an empty `Literal[]` has no options to declare.")

    todos: tuple[Todo, ...] = ()
    for value in values:
        if isinstance(value, Enum):
            todos = (
                *todos,
                Todo(
                    f"`Literal[{type(value).__name__}.{value.name}]` has no C# counterpart in v1 — "
                    f"GRAMMAR.md section 8. Replace the enum member with its wire value, or type the "
                    f"field as the enum itself."
                ),
            )

    hasNull = any(value is None for value in values)
    concrete = [value for value in values if value is not None]
    if all(isinstance(value, bool) for value in concrete):
        text = "bool"
    elif all(isinstance(value, str) for value in concrete):
        text = "string"
    elif all(isinstance(value, int) and not isinstance(value, bool) for value in concrete):
        text = "int"
    else:
        text = "object"

    if hasNull and text != "object":
        text = f"{text}?"
    elif hasNull:
        text = "object?"
    return MappedType(text, frozenset(), todos)


def _mapNestedLiteral(values: tuple[Any, ...]) -> MappedType:
    """Map a ``Literal`` found *inside* another type, which C# cannot declare.

    ``[LiteralValues]`` applies to a whole property, so ``list[Literal['a', 'b']]`` has no
    attribute spelling — the C# suite reaches that schema only through hand-written
    metadata.  The element type is still emitted so the rest of the field is usable.
    """
    inner = literalKind(values)
    return inner.withTodo(
        f"a `Literal` nested inside another type has no C# declaration — `[LiteralValues]` applies to a "
        f"whole property, so `{inner.text}` here loses the option set and hashes as a plain "
        f"`{inner.text}` rather than `Literal[...]` (GRAMMAR.md section 8)."
    )


def _mapUnion(args: tuple[Any, ...]) -> MappedType:
    """Map a union.  Only ``Optional[T]`` has a C# spelling (sections 6 and 13)."""
    members = [arg for arg in args if arg is not type(None)]
    hasNone = len(members) != len(args)

    if len(members) == 1:
        inner = mapType(members[0])
        if not hasNone:
            return inner
        # Nullable value type and nullable reference type are spelled and rendered
        # identically (section 6), so no distinction is needed here.
        return MappedType(f"{inner.text}?", inner.usings, inner.todos)

    rendered = f"Union[{', '.join(sorted(canonicalTypeName(arg) for arg in args))}]"
    return MappedType("object").withTodo(
        f"`{rendered}` is a union of {len(members)} non-null members, which C# has no type for — it spells "
        f"optionality as `T?` and nothing else. GRAMMAR.md section 13 records this as `csharpDeclarable: "
        f"false`: C# can reproduce the hash by reading the payload but cannot declare a mirror. Split the "
        f"field, or wrap the alternatives in a Versionable."
    )


def _mapGeneric(origin: Any, args: tuple[Any, ...]) -> MappedType:
    """Map a parameterized type: the closed container set, or a name with its parameters dropped."""
    if origin is tuple:
        return _mapTuple(args)

    container = _CONTAINERS.get(origin)
    if container is None:
        # Outside the closed set, type parameters are dropped from the hash (section 9), so
        # the C# mirror is the bare type.
        return _mapPlain(origin)

    name, usings = container
    if not args:
        return MappedType("object").withTodo(
            f"`{name.lower()}` without type arguments has no C# spelling — C# generics are always "
            f"parameterized. Annotate the element type on the Python side."
        )

    mapped = [mapType(arg) for arg in args]
    inner = ", ".join(item.text for item in mapped)
    return MappedType(
        f"{name}<{inner}>",
        usings.union(*(item.usings for item in mapped)),
        tuple(todo for item in mapped for todo in item.todos),
    )


def _mapTuple(args: tuple[Any, ...]) -> MappedType:
    """Map a tuple: a fixed tuple becomes a value tuple, a variadic one has no C# type."""
    if len(args) == 2 and args[1] is Ellipsis:  # noqa: PLR2004 — a variadic tuple is exactly (T, ...)
        element = mapType(args[0])
        return MappedType(f"List<{element.text}>", element.usings, element.todos).withTodo(
            f"`tuple[{element.text}, ...]` is a variadic tuple, which C# has no type for (GRAMMAR.md "
            f"section 5). `List<{element.text}>` loads the same bytes but renders `list[...]`, so the "
            f"declared hash above will NOT match — the analyzer will report the one it computes."
        )
    if not args:
        return MappedType("object").withTodo(
            "an empty `tuple[()]` has no C# spelling. Annotate the element types on the Python side."
        )

    mapped = [mapType(arg) for arg in args]
    inner = ", ".join(item.text for item in mapped)
    return MappedType(
        f"({inner})",
        frozenset().union(*(item.usings for item in mapped)),
        tuple(todo for item in mapped for todo in item.todos),
    )


def _mapPlain(tp: Any) -> MappedType:
    """Map an unparameterized type: scalar, converter type, enum, Versionable, or unknown."""
    scalar = _SCALARS.get(tp)
    if scalar is not None:
        return MappedType(*scalar)

    if isinstance(tp, type):
        if tp in _UNCONVERTIBLE_PATHS:
            return MappedType("FilePath", frozenset({NAMESPACE_VERSIONABLE})).withTodo(
                f"`{tp.__name__}` has no C# counterpart (GRAMMAR.md section 9 lists it as `(none)`). "
                f"`FilePath` reads and writes the same string but renders `Path`, so the declared hash "
                f"above will NOT match. Use `pathlib.Path` on the Python side if the schemas must agree."
            )

        exact = _CONVERTERS_EXACT.get(tp)
        if exact is not None:
            return MappedType(*exact)

        for base, mapping in _CONVERTERS_BY_SUBCLASS:
            if _isSubclass(tp, base):
                return MappedType(*mapping)

        if _isSubclass(tp, Versionable) or _isSubclass(tp, Enum):
            # Emitted alongside, or referenced by name and reported in the file header.
            return MappedType(tp.__name__)

    if tp is Any:
        return MappedType("object").withTodo(
            "`Any` declares no type, so there is nothing to render. Annotate the field concretely."
        )

    name = getattr(tp, "__name__", None) or str(tp)
    return MappedType(name).withTodo(
        f"`{name}` is not a scalar, container, enum, Versionable, or built-in converter type. It renders "
        f"as the bare Serialization Name `{name}` (GRAMMAR.md section 9), so declare a C# type of that "
        f"name and register a converter for it, or replace the field."
    )


def _isSubclass(tp: Any, base: type) -> bool:
    """Return True when *tp* is a class deriving from *base*, tolerating non-classes."""
    return isinstance(tp, type) and issubclass(tp, base)
