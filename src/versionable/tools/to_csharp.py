"""Scaffold C# ``[Versionable]`` declarations from Python ``Versionable`` classes.

Usage::

    python -m versionable.tools.to_csharp mypkg.schemas
    python -m versionable.tools.to_csharp mypkg.schemas:Config --namespace MyPkg.Schemas --out ./cs

Each target is imported and introspected at runtime — no source parsing — so what is
converted is what Python itself resolved: the same field mapping ``computeHash`` hashes.
That is the point of the tool. The ``hash=`` literal is copied across **verbatim**, because
a canonical payload is language-neutral by construction (``conformance/GRAMMAR.md`` section
1), and the Roslyn analyzer recomputes it from the emitted C# on the next build. A hash the
two implementations disagree about is therefore a compile error at the destination rather
than a load failure in production.

**Output is one ``.cs`` file per Python module**, named after the module. C# convention is
one type per file, but a Python module is the unit that holds a schema together with the
enums it references, and splitting them would scatter types that have to be reviewed as a
set. Rename and split the output afterwards if the destination project prefers it; nothing
in the emitted code depends on the file name.

**What the scaffolder will not guess.** Anything with no C# spelling — a lambda in a
migration op, an imperative migration body, a ``default_factory`` that is not an empty
container, an n-ary union, a variadic tuple, a bare ``ndarray``, a pure path — is emitted as
a ``// TODO`` citing the grammar section that closes it off, with the Python source
commented underneath. A file with no TODOs left compiles; a file with TODOs is a starting
point.
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime
import importlib
import inspect
import math
import re
import sys
import typing
import uuid
from collections.abc import Iterable
from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum
from pathlib import Path
from typing import Any

from versionable._base import Versionable, _InternalMeta, _resolveFields
from versionable._hash import _ENUM_NAME_ATTR
from versionable._migration import (
    Migration,
    _AddOp,
    _ConvertOp,
    _DeriveOp,
    _DropOp,
    _ImperativeMigration,
    _MergeOp,
    _RenameOp,
    _RequiresUpgradeOp,
    _SplitOp,
)
from versionable._types import _LITERAL_FALLBACK_KEY
from versionable.tools._csharp_types import (
    NAMESPACE_VERSIONABLE,
    MappedType,
    Todo,
    literalKind,
    mapType,
)

_INDENT = "    "

# Namespace holding the `Migration` builder and `[Migration]` attribute.
_NAMESPACE_MIGRATIONS = "Versionable.Migrations"

# The empty-container factories.  `field(default_factory=list)` is how a Python dataclass
# spells "starts empty", and refusing to convert the overwhelmingly common case would put a
# TODO on nearly every real schema.  Every *other* factory is a callable whose body only a
# human can port, and gets one.
_EMPTY_FACTORY_LITERALS: dict[Any, str] = {
    list: "[]",
    dict: "new()",
    set: "new()",
    frozenset: "FrozenSet<{0}>.Empty",
    tuple: "default",
    bytes: "[]",
    str: '""',
    int: "0",
    float: "0.0",
    bool: "false",
}


# ---------------------------------------------------------------------------
# Emitted-file model
# ---------------------------------------------------------------------------


@dataclass
class CsharpFile:
    """One emitted ``.cs`` file.

    Attributes:
        fileName: Suggested file name, derived from the source module.
        namespace: File-scoped namespace the declarations sit in.
        text: The complete file contents, newline-terminated.
        todos: Every unconvertible construct found while emitting it.
        moduleName: The Python module the declarations came from.
    """

    fileName: str
    namespace: str
    text: str
    todos: list[Todo]
    moduleName: str


@dataclass
class _Block:
    """Lines of C# under construction, with the namespaces and TODOs they accumulated."""

    lines: list[str] = field(default_factory=list)
    usings: set[str] = field(default_factory=set)
    todos: list[Todo] = field(default_factory=list)

    def absorb(self, mapped: MappedType) -> str:
        """Record *mapped*'s namespaces and TODOs, and return its C# type text."""
        self.usings |= mapped.usings
        self.todos.extend(mapped.todos)
        return mapped.text

    def extend(self, other: _Block, *, indent: int = 0) -> None:
        """Append *other*'s lines at *indent* extra levels, merging its namespaces and TODOs."""
        prefix = _INDENT * indent
        self.lines.extend(prefix + line if line else "" for line in other.lines)
        self.usings |= other.usings
        self.todos.extend(other.todos)

    def todo(self, message: str, *, source: str = "") -> None:
        """Emit a ``// TODO`` comment block at this block's own indent, and record it."""
        self.todos.append(Todo(message, source))
        self.lines.extend(f"// {line}" for line in _wrap(f"TODO: {message}"))
        for line in source.splitlines():
            self.lines.append(f"//     {line}".rstrip())


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------


def convert(targets: list[str], *, namespace: str | None = None) -> list[CsharpFile]:
    """Convert one or more ``module`` / ``module:Class`` targets to C# files.

    Args:
        targets: Import targets. ``pkg.mod`` takes every ``Versionable`` the module defines;
            ``pkg.mod:Name`` takes one class or enum.
        namespace: C# namespace for every emitted file. Defaults per file to the PascalCase
            of the source module's dotted path.

    Returns:
        One :class:`CsharpFile` per source module, in first-seen order.

    Raises:
        LookupError: A target names a module attribute that does not exist.
        TypeError: A target names something that is not a ``Versionable`` or an enum.
    """
    byModule: dict[str, list[type]] = {}
    for target in targets:
        for tp in _resolveTarget(target):
            moduleTypes = byModule.setdefault(tp.__module__, [])
            if tp not in moduleTypes:
                moduleTypes.append(tp)

    return [convertTypes(types, moduleName=moduleName, namespace=namespace) for moduleName, types in byModule.items()]


def convertTypes(types: list[type], *, moduleName: str, namespace: str | None = None) -> CsharpFile:
    """Emit one C# file holding *types* plus the enums they reach in the same module.

    Args:
        types: ``Versionable`` subclasses and/or enums, in the order to emit them.
        moduleName: Dotted Python module the types came from; names the file.
        namespace: C# namespace. Defaults to the PascalCase of *moduleName*.

    Returns:
        The emitted file.
    """
    ordered = _orderTypes(types, moduleName)
    body = _Block()
    external: set[str] = set()

    for index, tp in enumerate(ordered):
        if index:
            body.lines.append("")
        if isinstance(tp, type) and issubclass(tp, Enum):
            body.extend(_emitEnum(tp))
        else:
            body.extend(_emitClass(tp, known={item.__name__ for item in ordered}, external=external))

    resolved = namespace or _namespaceFor(moduleName)
    header = _emitHeader(moduleName, body.todos, external)
    usings = sorted(body.usings | {NAMESPACE_VERSIONABLE})

    lines = [*header, ""]
    lines.extend(f"using {name};" for name in usings)
    lines.extend(["", f"namespace {resolved};", ""])
    lines.extend(body.lines)

    return CsharpFile(
        fileName=f"{_pascalCase(moduleName.rsplit('.', 1)[-1])}.cs",
        namespace=resolved,
        text="\n".join(lines).rstrip() + "\n",
        todos=body.todos,
        moduleName=moduleName,
    )


# ---------------------------------------------------------------------------
# Target resolution
# ---------------------------------------------------------------------------


def _resolveTarget(target: str) -> list[type]:
    """Import *target* and return the types it names."""
    moduleName, _, attribute = target.partition(":")
    module = importlib.import_module(moduleName)

    if attribute:
        found = getattr(module, attribute, None)
        if found is None:
            raise LookupError(f"{moduleName!r} has no attribute {attribute!r}.")
        if not _isConvertible(found):
            raise TypeError(f"{target!r} is a {type(found).__name__}, not a Versionable subclass or an enum.")
        return [found]

    types = [
        value
        for value in vars(module).values()
        if _isConvertible(value) and value.__module__ == moduleName and issubclass(value, Versionable)
    ]
    if not types:
        raise LookupError(f"{moduleName!r} defines no Versionable subclasses.")
    return types


def _isConvertible(value: Any) -> bool:
    """Return True when *value* is a ``Versionable`` subclass or an enum class."""
    if not isinstance(value, type) or value is Versionable:
        return False
    return (issubclass(value, Versionable) and getattr(value, "_serializer_meta_", None) is not None) or issubclass(
        value, Enum
    )


def _orderTypes(types: list[type], moduleName: str) -> list[type]:
    """Return *types* with the same-module enums they reference prepended.

    Enums come first so the file reads top-down, and are pulled in automatically because a
    class that references one does not compile without it. Types from *other* modules are
    left alone — they belong to that module's file — and are reported in the header instead.
    """
    enums: list[type] = []
    for tp in types:
        if issubclass(tp, Enum):
            continue
        for fieldType in _resolveFields(tp).values():
            for referenced in _referencedTypes(fieldType):
                if (
                    issubclass(referenced, Enum)
                    and referenced.__module__ == moduleName
                    and referenced not in enums
                    and referenced not in types
                ):
                    enums.append(referenced)

    declared = [tp for tp in types if issubclass(tp, Enum)]
    classes = [tp for tp in types if not issubclass(tp, Enum)]
    return [*declared, *enums, *classes]


def _referencedTypes(tp: Any) -> list[type]:
    """Return every named type reachable from the annotation *tp*, one level of nesting deep."""
    found: list[type] = []
    origin = typing.get_origin(tp)
    args = typing.get_args(tp)

    if origin is typing.Literal:
        return [type(value) for value in args if isinstance(value, Enum)]
    if args:
        for arg in args:
            found.extend(_referencedTypes(arg))
        return found
    if isinstance(tp, type):
        found.append(tp)
    return found


# ---------------------------------------------------------------------------
# Class emission
# ---------------------------------------------------------------------------


def _metaOf(cls: type) -> _InternalMeta:
    """Return the internal metadata of a ``Versionable`` subclass.

    ``_serializer_meta_`` is declared on the base as a ``ClassVar`` but reached here through a
    bare ``type``, which no type checker can narrow. Asserting it once is cheaper than a
    suppression at every call site, and it turns a would-be ``AttributeError`` deep in
    emission into a message naming the class.
    """
    meta = getattr(cls, "_serializer_meta_", None)
    if not isinstance(meta, _InternalMeta):
        raise TypeError(f"{cls.__qualname__} is not a Versionable schema: it has no serialization metadata.")
    return meta


def _emitClass(cls: type, *, known: set[str], external: set[str]) -> _Block:
    """Emit the C# declaration of the ``Versionable`` subclass *cls*."""
    block = _Block()
    meta = _metaOf(cls)

    block.lines.extend(_docComment(_docstringOf(cls)))
    block.lines.append(f"[Versionable({', '.join(_versionableArguments(meta))})]")
    if meta.name != cls.__name__:
        block.lines.append(f'[SerializationName("{_escape(meta.name)}")]')

    base = _versionableBase(cls)
    inheritance = f" : {base.__name__}" if base is not None else ""
    sealed = "" if any(_versionableBase(other) is cls for other in _subclassesOf(cls)) else "sealed "
    block.lines.append(f"public {sealed}partial class {cls.__name__}{inheritance}")
    block.lines.append("{")

    own = _ownFields(cls, base)
    dcFields = {item.name: item for item in dataclasses.fields(cls)} if dataclasses.is_dataclass(cls) else {}

    collisions = _propertyNameCollisions(cls, own)
    members = _Block()
    for index, (wireName, fieldType) in enumerate(own.items()):
        if index:
            members.lines.append("")
        if wireName in collisions:
            members.todo(collisions[wireName])
        members.extend(_emitField(cls, wireName, fieldType, dcFields.get(wireName), known=known, external=external))

    migrations = _emitMigrations(cls)
    if migrations.lines:
        if members.lines:
            members.lines.append("")
        members.extend(migrations)

    block.extend(members, indent=1)
    block.lines.append("}")
    return block


def _versionableArguments(meta: _InternalMeta) -> list[str]:
    """Return the ``[Versionable(...)]`` arguments, omitting every C# default."""
    parts = [f"Version = {meta.version}", f'Hash = "{_escape(meta.hash)}"']
    if meta.oldNames:
        joined = ", ".join(f'"{_escape(name)}"' for name in meta.oldNames)
        parts.append(f"OldNames = new[] {{ {joined} }}")
    if not meta.register:
        parts.append("Register = false")
    if meta.skipDefaults:
        parts.append("SkipDefaults = true")
    if meta.unknown != "ignore":
        parts.append(f"Unknown = UnknownFieldPolicy.{meta.unknown.capitalize()}")
    if not meta.validateLiterals:
        parts.append("ValidateLiterals = false")
    return parts


def _propertyNameCollisions(cls: type, fields: dict[str, Any]) -> dict[str, str]:
    """Return ``{wireName: message}`` for fields whose PascalCase property names collide.

    ``timeout_ms`` and ``timeoutMs`` are distinct wire names and distinct schemas, but both
    PascalCase to ``TimeoutMs`` — two properties of one name, which is CS0102. The scaffolder
    will not pick a winner, because either choice silently renames a wire field: it names both
    sources and leaves the human to add a ``[VersionableField]`` and a different property name.
    """
    byProperty: dict[str, list[str]] = {}
    for wireName in fields:
        byProperty.setdefault(_propertyName(wireName, cls.__name__), []).append(wireName)

    messages: dict[str, str] = {}
    for propertyName, wireNames in byProperty.items():
        if len(wireNames) < 2:  # noqa: PLR2004 — a collision needs two claimants
            continue
        sources = ", ".join(f"`{name}`" for name in wireNames)
        for wireName in wireNames:
            messages[wireName] = (
                f"{sources} all PascalCase to `{propertyName}`, so this class declares that property more than "
                f"once (CS0102). Rename all but one of the C# properties and keep its `[VersionableField]` — the "
                f"wire names are distinct and hash-significant, so none of them may change."
            )
    return messages


def _versionableBase(cls: type) -> type | None:
    """Return the nearest ``Versionable`` base of *cls* that is itself a schema, or None."""
    for base in cls.__mro__[1:]:
        if base is Versionable or not issubclass(base, Versionable):
            continue
        if getattr(base, "_serializer_meta_", None) is not None:
            return base
    return None


def _subclassesOf(cls: type) -> list[type]:
    """Return the direct subclasses of *cls* Python currently knows about."""
    return list(cls.__subclasses__())


def _ownFields(cls: type, base: type | None) -> dict[str, Any]:
    """Return the fields *cls* declares itself.

    A derived C# type inherits its base's members and the generator folds them into the
    field set, exactly as Python's MRO walk does — so re-declaring them would double them.
    """
    fields = _resolveFields(cls)
    if base is None:
        return fields
    inherited = set(_resolveFields(base))
    return {name: tp for name, tp in fields.items() if name not in inherited}


def _emitField(
    cls: type,
    wireName: str,
    fieldType: Any,
    dcField: dataclasses.Field[Any] | None,
    *,
    known: set[str],
    external: set[str],
) -> _Block:
    """Emit one property: its attributes, its C# type, and its default or ``required``."""
    block = _Block()
    propertyName = _propertyName(wireName, cls.__name__)

    literalValues = _literalOptions(fieldType)
    if literalValues is not None:
        mapped = literalKind(literalValues)
    else:
        mapped = mapType(fieldType)
        for referenced in _referencedTypes(fieldType):
            if _isConvertible(referenced) and referenced.__name__ not in known:
                external.add(f"{referenced.__name__} ({referenced.__module__})")

    typeText = _awarenessOf(block.absorb(mapped), dcField)
    initializer, defaultTodo = _initializer(dcField, typeText)
    fieldTodos = list(mapped.todos)
    if defaultTodo is not None:
        block.todos.append(defaultTodo)
        fieldTodos.append(defaultTodo)

    # Every TODO first, then the attributes, then the property: an attribute list broken by a
    # comment block reads as though the comment belonged to the attribute under it.
    for item in fieldTodos:
        block.lines.extend(f"// {line}" for line in _wrap(f"TODO ({wireName}): {item.message}"))
        for line in item.source.splitlines():
            block.lines.append(f"//     {line}".rstrip())

    if propertyName != wireName:
        block.lines.append(f'[VersionableField("{_escape(wireName)}")]')
    if literalValues is not None:
        block.lines.append(_literalAttribute(literalValues, dcField))

    modifier = "required " if initializer is None else ""
    suffix = f" = {initializer};" if initializer is not None else ""
    block.lines.append(f"public {modifier}{typeText} {propertyName} {{ get; init; }}{suffix}")
    return block


def _awarenessOf(typeText: str, dcField: dataclasses.Field[Any] | None) -> str:
    """Upgrade a ``DateTime`` to ``DateTimeOffset`` when the field's default is timezone-aware.

    A Python annotation is just ``datetime.datetime`` either way — awareness is a property of
    the value, not the type — so the default is the only evidence the scaffolder has. Both
    spellings render the canonical ``datetime`` (GRAMMAR.md section 9), so guessing wrong
    costs a review comment, not a hash.
    """
    if not typeText.startswith("DateTime") or dcField is None:
        return typeText
    default = dcField.default
    if isinstance(default, datetime.datetime) and default.tzinfo is not None:
        return typeText.replace("DateTime", "DateTimeOffset", 1)
    return typeText


def _propertyName(wireName: str, className: str) -> str:
    """Return the C# property name for *wireName*, kept distinct from its declaring type."""
    name = _pascalCase(wireName)
    # C# forbids a member with the same name as its enclosing type (CS0542).
    return f"{name}Value" if name == className else name


# ---------------------------------------------------------------------------
# Literals
# ---------------------------------------------------------------------------


def _literalOptions(fieldType: Any) -> tuple[Any, ...] | None:
    """Return a field's ``Literal`` options, unwrapping ``Annotated``, or None."""
    origin = typing.get_origin(fieldType)
    if origin is typing.Annotated:
        return _literalOptions(typing.get_args(fieldType)[0])
    if origin is typing.Literal:
        return typing.get_args(fieldType)
    return None


def _literalAttribute(values: tuple[Any, ...], dcField: dataclasses.Field[Any] | None) -> str:
    """Return the ``[LiteralValues(...)]`` attribute, with a fallback when one is declared.

    Argument order is the canonical order and is hash-significant (section 8), so the Python
    declaration order is reproduced exactly.
    """
    rendered = ", ".join(_literalArgument(value) for value in values)
    fallback = dcField.metadata.get(_LITERAL_FALLBACK_KEY) if dcField is not None else None
    if fallback is not None:
        return f"[LiteralValues({rendered}, Fallback = {_literalArgument(fallback)})]"
    return f"[LiteralValues({rendered})]"


def _literalArgument(value: Any) -> str:
    """Render one ``Literal`` option as a C# attribute argument."""
    if isinstance(value, Enum):
        # Rejected by literalKind with a TODO; rendered anyway so the reviewer sees the intent.
        return f"{type(value).__name__}.{_pascalCase(value.name)}"
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    return f'"{_escape(str(value))}"'


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------


def _initializer(dcField: dataclasses.Field[Any] | None, typeText: str) -> tuple[str | None, Todo | None]:
    """Return the C# initializer for a field's default, or ``(None, todo)`` to mark it required.

    A field with no expressible default becomes ``required``, which is both the honest
    spelling and the one that keeps the nullable analysis quiet.
    """
    if dcField is None or (dcField.default is dataclasses.MISSING and dcField.default_factory is dataclasses.MISSING):
        return None, None

    if dcField.default_factory is not dataclasses.MISSING:
        empty = _EMPTY_FACTORY_LITERALS.get(dcField.default_factory)
        if empty is not None:
            return empty.format(_genericArgument(typeText)), None
        return None, Todo(
            "`default_factory` is a callable, which has no C# spelling. Port the initial value by hand — "
            "the property is `required` until you do.",
            source=_pythonSource(dcField.default_factory),
        )

    if dcField.default is None:
        # A nullable property already defaults to null; spelling it out adds nothing.
        return None if not typeText.endswith("?") else "null", None

    literal = _csharpLiteral(dcField.default, typeText)
    if literal is None:
        return None, Todo(
            f"the Python default `{dcField.default!r}` has no C# literal form. Assign an equivalent by "
            f"hand — the property is `required` until you do."
        )
    return literal, None


def _genericArgument(typeText: str) -> str:
    """Return the single type argument of ``Name<T>``, or the text unchanged."""
    start = typeText.find("<")
    return typeText[start + 1 : -1] if start != -1 and typeText.endswith(">") else typeText


def _csharpLiteral(value: Any, typeText: str) -> str | None:
    """Return *value* as a C# initializer expression, or None when it has no literal form."""
    if isinstance(value, Enum):
        return f"{type(value).__name__}.{_pascalCase(value.name)}"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return _floatLiteral(value)
    if isinstance(value, str):
        return f'"{_escape(value)}"'
    if isinstance(value, bytes):
        return "[]" if not value else "[" + ", ".join(f"0x{byte:02x}" for byte in value) + "]"
    if isinstance(value, Decimal):
        return f"{value}m"
    if isinstance(value, tuple):
        if not value:
            # Only a variadic tuple can default to (), and that maps to a List<T>.
            return "[]" if typeText.startswith("List<") else None
        parts = [_csharpLiteral(item, "") for item in value]
        return None if any(part is None for part in parts) else f"({', '.join(part for part in parts if part)})"
    if isinstance(value, (list, set, frozenset, dict)) and not value:
        return _EMPTY_FACTORY_LITERALS[type(value)].format(_genericArgument(typeText))
    return _converterLiteral(value)


def _converterLiteral(value: Any) -> str | None:
    """Return the C# construction expression for a built-in converter type's value.

    These are values a schema routinely defaults to — an epoch datetime, a zero duration, a
    nil UUID — and every one has an unambiguous C# spelling, so leaving them to a TODO would
    put one on nearly every temporal schema for no gain.

    **A converted default is exact or it is not emitted.** Python's temporal types carry
    microseconds, and the obvious C# spellings quietly lose them: ``TimeSpan.FromSeconds``
    rounds to the nearest millisecond, and the short ``DateTime`` constructors stop at
    seconds. .NET 7 added microsecond-precision overloads for all of them, so the full value
    is expressible; where it is not — a UTC offset that is not a whole number of minutes, a
    duration outside ``TimeSpan``'s range — this returns None and the field becomes
    ``required`` with a TODO. Silently shifting a default is not an option the scaffolder has.
    """
    if isinstance(value, complex):
        return f"new Complex({_floatLiteral(value.real)}, {_floatLiteral(value.imag)})"
    if isinstance(value, datetime.datetime):
        return _datetimeLiteral(value)
    if isinstance(value, datetime.date):
        return f"new DateOnly({value.year}, {value.month}, {value.day})"
    if isinstance(value, datetime.time):
        return f"new TimeOnly({value.hour}, {value.minute}, {value.second}{_subSecond(value.microsecond)})"
    if isinstance(value, datetime.timedelta):
        return _timedeltaLiteral(value)
    if isinstance(value, uuid.UUID):
        return "Guid.Empty" if value.int == 0 else f'Guid.Parse("{value}")'
    if isinstance(value, re.Pattern):
        return f'new Regex("{_escape(str(value.pattern))}")'
    if isinstance(value, Path):
        return f'new FilePath("{_escape(str(value))}")'
    return None


_SECONDS_PER_MINUTE = 60
# TimeSpan counts 100-nanosecond ticks in an Int64, so it spans just under 10 675 200 days —
# far short of timedelta's 999 999 999.
_TIMESPAN_MAX_DAYS = 10_675_199


def _subSecond(microsecond: int) -> str:
    """Return the trailing ``, millisecond, microsecond`` arguments, or nothing when whole.

    Both are separate constructor arguments in .NET 7+; the argument is a *milli*second count
    plus a remainder, not a microsecond count, so it has to be split rather than passed on.
    """
    if microsecond == 0:
        return ""
    return f", {microsecond // 1000}, {microsecond % 1000}"


def _datetimeLiteral(value: datetime.datetime) -> str | None:
    """Return a ``DateTime`` or ``DateTimeOffset`` expression preserving the full value."""
    date = f"{value.year}, {value.month}, {value.day}"
    clock = f"{value.hour}, {value.minute}, {value.second}"
    fraction = _subSecond(value.microsecond)

    if value.tzinfo is None:
        return f"new DateTime({date}, {clock}{fraction})"

    offset = value.utcoffset() or datetime.timedelta()
    seconds = int(offset.total_seconds())
    if offset.microseconds or seconds % _SECONDS_PER_MINUTE:
        # DateTimeOffset rejects an offset that is not a whole number of minutes, so there is
        # no expression to emit — and rounding one would move the instant.
        return None
    return f"new DateTimeOffset({date}, {clock}{fraction}, TimeSpan.FromMinutes({seconds // _SECONDS_PER_MINUTE}))"


def _timedeltaLiteral(value: datetime.timedelta) -> str | None:
    """Return a ``TimeSpan`` expression preserving the full microsecond value.

    Built from components rather than ``TimeSpan.FromSeconds``, which rounds to the nearest
    millisecond, and rather than a raw tick count, which is exact but unreadable. The
    component constructor sums its arguments, so Python's normalisation of a negative
    duration into a negative day count plus positive parts carries over unchanged.
    """
    if not value:
        return "TimeSpan.Zero"
    if abs(value.days) > _TIMESPAN_MAX_DAYS:
        return None

    hours, remainder = divmod(value.seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    parts = f"{value.days}, {hours}, {minutes}, {seconds}"
    return f"new TimeSpan({parts}{_subSecond(value.microseconds)})"


def _floatLiteral(value: float) -> str:
    """Return a C# ``double`` literal for *value*, including the non-finite spellings."""
    if math.isnan(value):
        return "double.NaN"
    if math.isinf(value):
        return "double.PositiveInfinity" if value > 0 else "double.NegativeInfinity"
    text = repr(value)
    return text if ("." in text or "e" in text or "E" in text) else f"{text}.0"


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


def _emitEnum(cls: type[Enum]) -> _Block:
    """Emit a Python enum as a C# enum, carrying string values and the fallback member."""
    block = _Block()
    block.lines.extend(_docComment(_docstringOf(cls)))

    declaredName = cls.__dict__.get(_ENUM_NAME_ATTR)
    if isinstance(declaredName, str) and declaredName != cls.__name__:
        block.lines.append(f'[SerializationName("{_escape(declaredName)}")]')

    block.lines.append(f"public enum {cls.__name__}")
    block.lines.append("{")

    fallback = getattr(cls, "VERSIONABLE_FALLBACK", None)
    for index, member in enumerate(cls):
        if index:
            block.lines.append("")
        memberBlock = _emitEnumMember(member, isFallback=member is fallback)
        block.extend(memberBlock, indent=1)

    block.lines.append("}")
    return block


def _emitEnumMember(member: Enum, *, isFallback: bool) -> _Block:
    """Emit one enum member with its wire value and, when it is the fallback, the marker."""
    block = _Block()
    value = member.value
    assignment = ""

    if isinstance(value, str):
        block.lines.append(f'[EnumValue("{_escape(value)}")]')
    elif isinstance(value, int) and not isinstance(value, bool):
        assignment = f" = {value}"
    else:
        block.todo(
            f"enum member `{member.name}` has the value {value!r}, which a C# enum cannot hold — members "
            f"are integral, and a string value is carried by `[EnumValue]`. Give it a string or an "
            f"integer on the Python side."
        )

    if isFallback:
        block.lines.append("[EnumFallback]")
    block.lines.append(f"{_pascalCase(member.name)}{assignment},")
    return block


# ---------------------------------------------------------------------------
# Migrations
# ---------------------------------------------------------------------------


def _emitMigrations(cls: type) -> _Block:
    """Emit the nested ``Migrate`` class, op for op, from the Python ``Migrate`` class."""
    block = _Block()
    migrateClass = getattr(cls, "Migrate", None)
    if migrateClass is None:
        return block

    declared = _collectMigrations(migrateClass)
    if not declared:
        return block

    # A derived class inherits Python's `Migrate` through the MRO, and `resolveMigrations`
    # finds it there — so the C# mirror has to carry one too, because Roslyn's GetTypeMembers
    # does not see a base type's nested types and the chain would otherwise be missing.
    # Re-declaring a name a base type already has needs `new`, or CS0108 fires, which
    # `TreatWarningsAsErrors` turns into a failed build.
    inherited = "Migrate" not in cls.__dict__
    summary = "Migrations, transcribed from the Python `Migrate` class. Op semantics are identical."
    if inherited:
        base = _versionableBase(cls)
        summary += (
            f" Inherited from `{base.__name__}` in Python, and re-declared here because a nested type"
            f" is not inherited in C#."
            if base is not None
            else " Inherited in Python, and re-declared here because a nested type is not inherited in C#."
        )

    block.usings.add(_NAMESPACE_MIGRATIONS)
    block.lines.extend(_docComment(summary))
    block.lines.append(f"public {'new ' if inherited else ''}static class Migrate")
    block.lines.append("{")

    body = _Block()
    for index, (version, migration) in enumerate(sorted(declared.items())):
        if index:
            body.lines.append("")
        # The doc comment is emitted by each branch rather than here, so it stays adjacent to
        # the declaration: a `//` block between `///` and the member detaches the XML doc.
        if isinstance(migration, Migration):
            body.extend(_emitDeclarative(version, migration))
        else:
            body.extend(_emitImperative(version, migration))

    block.extend(body, indent=1)
    block.lines.append("}")
    return block


def _collectMigrations(migrateClass: type) -> dict[int, Migration | _ImperativeMigration]:
    """Return ``{fromVersion: migration}`` for a ``Migrate`` class, mirroring ``resolveMigrations``."""
    found: dict[int, Migration | _ImperativeMigration] = {}
    for name in dir(migrateClass):
        attr = getattr(migrateClass, name)
        if name.startswith("v") and name[1:].isdigit() and isinstance(attr, Migration):
            found[int(name[1:])] = attr
        elif isinstance(attr, _ImperativeMigration):
            found[attr.fromVersion] = attr
    return found


def _emitDeclarative(version: int, migration: Migration) -> _Block:
    """Emit one declarative migration as a chained ``Migration`` builder field.

    ``Migration.then()`` has already flattened its operand into this op list on the Python
    side, so a chained declaration arrives here as one sequence and is emitted as one.
    """
    block = _Block()
    calls = [rendered for op in migration.ops if (rendered := _emitOp(op, block))]

    block.lines.extend(_docComment(f"Takes a version {version} file to version {version + 1}."))
    declaration = f"public static readonly Migration V{version} = new Migration()"
    if not calls:
        block.lines.append(f"{declaration};")
        return block

    # One op per line. A migration is the part of a scaffold most likely to be wrong, and a
    # one-line chain of six calls is the shape a reviewer skims past; a stacked one is a list
    # to check off. It is also what a diff can point at when one op changes.
    block.lines.append(declaration)
    block.lines.extend(f"{_INDENT}{call}" for call in calls[:-1])
    block.lines.append(f"{_INDENT}{calls[-1]};")
    return block


def _emitOp(op: Any, block: _Block) -> str:
    """Return the C# builder call for one migration op, recording any TODO it implies."""
    if isinstance(op, _RenameOp):
        return f'.Rename("{_escape(op.old)}", "{_escape(op.new)}")'
    if isinstance(op, _DropOp):
        return f'.Drop("{_escape(op.field)}")'
    if isinstance(op, _RequiresUpgradeOp):
        return ".RequiresUpgrade()"
    if isinstance(op, _AddOp):
        if callable(op.default):
            block.todo(_callableMessage(f'AddComputed("{op.field}", ...)'), source=_pythonSource(op.default))
            return f'.AddComputed("{_escape(op.field)}", () => {_notImplemented(op.default)})'
        literal = _csharpLiteral(op.default, "")
        if literal is None:
            block.todo(
                f"the default `{op.default!r}` added by this migration has no C# literal form. Assign an "
                f"equivalent by hand."
            )
            literal = "null"
        return f'.Add("{_escape(op.field)}", {literal})'
    if isinstance(op, _ConvertOp):
        block.todo(_callableMessage(f'Convert("{op.field}", ...)'), source=_pythonSource(op.via))
        reverse = "" if op.reverse is None else f", value => {_notImplemented(op.reverse)}"
        return f'.Convert("{_escape(op.field)}", value => {_notImplemented(op.via)}{reverse})'
    if isinstance(op, _DeriveOp):
        block.todo(_callableMessage(f'Derive("{op.field}", ...)'), source=_pythonSource(op.via))
        return f'.Derive("{_escape(op.field)}", "{_escape(op.fromField)}", value => {_notImplemented(op.via)})'
    if isinstance(op, _SplitOp):
        targets = ", ".join(
            f'new SplitTarget("{_escape(name)}", value => {_notImplemented(fn)})' for name, fn in op.into.items()
        )
        block.todo(
            _callableMessage(f'Split("{op.field}", ...)'),
            # Every target's callable is usually written in the one statement, so getsource
            # returns the same text once per target. Quote each distinct source once.
            source="\n".join(_uniqueInOrder(_pythonSource(fn) for fn in op.into.values())),
        )
        return f'.Split("{_escape(op.field)}", {targets})'
    if isinstance(op, _MergeOp):
        names = ", ".join(f'"{_escape(name)}"' for name in op.fields)
        block.todo(_callableMessage(f'Merge(..., "{op.into}", ...)'), source=_pythonSource(op.via))
        return f'.Merge(new[] {{ {names} }}, "{_escape(op.into)}", values => {_notImplemented(op.via)})'

    block.todo(f"migration op {type(op).__name__} has no C# builder call. Port it by hand.")
    return ""


def _emitImperative(version: int, migration: _ImperativeMigration) -> _Block:
    """Emit an imperative migration as a stubbed ``[Migration]`` method.

    The method is emitted live rather than commented out so the chain stays contiguous — the
    analyzer's VSN0003 check would otherwise fail on the gap — and its body throws, so an
    unported migration cannot silently drop data at load time.
    """
    block = _Block()
    block.todo(
        f"imperative migration from v{version}: transcribe the body by hand. `MigrationContext` is the "
        f"same dict-like wrapper it is in Python (`ctx[key]`, `ctx.Pop(key)`, `ctx.Drop(key)`).",
        source=_pythonSource(migration.fn),
    )
    block.lines.extend(_docComment(f"Takes a version {version} file to version {version + 1}."))
    block.lines.append(f"[Migration(FromVersion = {version})]")
    block.lines.append(f"public static void FromV{version}(MigrationContext ctx) =>")
    block.lines.append(f'{_INDENT}throw new NotImplementedException("Port {migration.fn.__name__} from Python.");')
    return block


def _callableMessage(call: str) -> str:
    """Return the standard TODO text for an op whose behaviour lives in a Python callable."""
    return (
        f"`{call}` carries a Python callable, which cannot be converted. The emitted lambda throws until "
        f"you port the body; the Python source is below."
    )


def _notImplemented(fn: Any) -> str:
    """Return a C# throw-expression standing in for the Python callable *fn*."""
    name = getattr(fn, "__name__", "callable")
    label = "lambda" if name == "<lambda>" else name
    return f'throw new NotImplementedException("Port the Python {label} — see the TODO above.")'


def _pythonSource(fn: Any, *, maxLines: int = 20) -> str:
    """Return the Python source of *fn*, for commenting under a TODO.

    A lambda's source is the whole statement it was written in, which is noisier than the
    lambda alone but is what the reader would otherwise have to open the file to see.

    A *class* used as a factory — ``field(default_factory=Path)`` — is named rather than
    quoted: its source is the entire class, which for ``pathlib.Path`` is several hundred
    lines of standard library nobody needs pasted into a C# comment.
    """
    if isinstance(fn, type) or inspect.isbuiltin(fn):
        module = getattr(fn, "__module__", "")
        name = getattr(fn, "__qualname__", repr(fn))
        return f"{module}.{name}" if module and module != "builtins" else str(name)

    try:
        source = inspect.getsource(fn)
    except (OSError, TypeError):
        return f"# source unavailable for {fn!r}"

    lines = inspect.cleandoc(source).strip().splitlines()
    if len(lines) > maxLines:
        lines = [*lines[:maxLines], f"# ... {len(lines) - maxLines} more line(s); see {_definedAt(fn)}"]
    return "\n".join(lines)


def _uniqueInOrder(values: Iterable[str]) -> list[str]:
    """Return *values* with duplicates removed, keeping first-seen order."""
    return list(dict.fromkeys(values))


def _definedAt(fn: Any) -> str:
    """Return a ``file:line`` reference for *fn*, for a truncated-source pointer."""
    try:
        return f"{inspect.getsourcefile(fn)}:{inspect.getsourcelines(fn)[1]}"
    except (OSError, TypeError):  # pragma: no cover — getsource already succeeded to get here
        return repr(fn)


# ---------------------------------------------------------------------------
# File header and formatting helpers
# ---------------------------------------------------------------------------


def _emitHeader(moduleName: str, todos: list[Todo], external: set[str]) -> list[str]:
    """Return the file's leading comment block.

    Deliberately *not* the `<auto-generated/>` marker: the schema-hash analyzer calls
    `ConfigureGeneratedCodeAnalysis(GeneratedCodeAnalysisFlags.None)`, so Roslyn would skip
    every hash check in a file carrying it — and the hash check is the reason this file
    exists.
    """
    lines = [
        f"// Scaffolded by `python -m versionable.tools.to_csharp {moduleName}`.",
        "// A starting point, not a finished port: review every declaration before use.",
        "//",
        "// Schema hashes are copied from the Python declarations verbatim. A canonical payload is",
        "// language-neutral (conformance/GRAMMAR.md section 1), so the same schema hashes the same in both",
        "// languages, and the Roslyn analyzer recomputes each one on the next build. A mismatch is a",
        "// compile error here rather than a load failure in production.",
        "//",
        "// Python `int` is unbounded, so integral fields are emitted as `long`, and `float` as `double`.",
        "// Scalar width is erased from the grammar (section 4), so narrowing either afterwards is a local",
        "// choice that leaves the hash alone.",
    ]
    if external:
        lines.append("//")
        # Not "from other modules": a `module:Class` target converts one class, so a type it
        # references from its *own* module is missing here too.
        lines.append("// Referenced by these declarations but not emitted in this run — convert them too:")
        lines.extend(f"//   {name}" for name in sorted(external))
    if todos:
        lines.append("//")
        lines.append(f"// {len(todos)} TODO(s) below need a human.")
    return lines


def _docstringOf(cls: type) -> str:
    """Return *cls*'s docstring, ignoring the signature ``@dataclass`` synthesises for it.

    A dataclass with no docstring of its own gets ``Config(name: 'str' = '')`` — a Python
    signature, which says nothing a reader of the C# declaration below it cannot see.
    """
    doc = (cls.__doc__ or "").strip()
    if not doc or doc.startswith(f"{cls.__name__}("):
        return f"Scaffolded from Python `{cls.__qualname__}`."
    return doc


def _docComment(text: str) -> list[str]:
    """Return *text*'s first paragraph as an XML ``<summary>`` doc comment."""
    paragraph = " ".join(line.strip() for line in text.strip().split("\n\n")[0].splitlines() if line.strip())
    wrapped = _wrap(_escapeXml(paragraph), width=100)
    if len(wrapped) == 1:
        return [f"/// <summary>{wrapped[0]}</summary>"]
    return ["/// <summary>", *(f"/// {line}" for line in wrapped), "/// </summary>"]


def _wrap(text: str, *, width: int = 106) -> list[str]:
    """Wrap *text* to *width* columns, never splitting a word."""
    words = text.split()
    if not words:
        return [""]
    lines = [words[0]]
    for word in words[1:]:
        if len(lines[-1]) + 1 + len(word) <= width:
            lines[-1] = f"{lines[-1]} {word}"
        else:
            lines.append(word)
    return lines


def _pascalCase(name: str) -> str:
    """Return *name* in PascalCase.

    Splits on underscores and keeps inner capitals, so ``timeout_ms`` becomes ``TimeoutMs``
    and ``byIndex`` becomes ``ByIndex``.  An all-uppercase part is lowercased past its first
    letter, because that is a Python constant or enum member (``RED``) rather than an
    acronym the author capitalised deliberately.
    """
    parts = [part for part in name.split("_") if part]
    if not parts:
        return name
    return "".join(part[0].upper() + (part[1:].lower() if part.isupper() else part[1:]) for part in parts)


def _namespaceFor(moduleName: str) -> str:
    """Return the default C# namespace for a dotted Python module path."""
    return ".".join(_pascalCase(part) for part in moduleName.split("."))


def _escapeXml(text: str) -> str:
    """Escape *text* for an XML doc comment."""
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _escape(text: str) -> str:
    """Escape *text* for a C# double-quoted string literal."""
    return text.replace("\\", "\\\\").replace('"', '\\"')


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    """Run the scaffolder. Returns a process exit code."""
    parser = argparse.ArgumentParser(
        prog="python -m versionable.tools.to_csharp",
        description=(
            "Scaffold C# [Versionable] partial classes from Python Versionable classes, by importing and "
            "introspecting them. Emits one .cs file per source module."
        ),
        epilog="Constructs with no C# spelling become // TODO comments citing conformance/GRAMMAR.md.",
    )
    parser.add_argument(
        "targets",
        nargs="+",
        metavar="MODULE[:CLASS]",
        help="Module to convert, or a single class within it (e.g. mypkg.schemas:Config).",
    )
    parser.add_argument(
        "--namespace",
        help="C# namespace for every emitted file. Defaults per file to the PascalCase of its module path.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        metavar="DIR",
        help="Write the files into DIR instead of stdout. Created if it does not exist.",
    )
    args = parser.parse_args(argv)

    try:
        files = convert(args.targets, namespace=args.namespace)
    except (ImportError, LookupError, TypeError) as exc:
        parser.error(str(exc))

    todoCount = sum(len(emitted.todos) for emitted in files)
    for emitted in files:
        if args.out is None:
            if len(files) > 1:
                print(f"// ==== {emitted.fileName} ====")
            print(emitted.text, end="")
        else:
            args.out.mkdir(parents=True, exist_ok=True)
            destination = args.out / emitted.fileName
            destination.write_text(emitted.text, encoding="utf-8")
            print(f"wrote {destination} ({len(emitted.todos)} TODO)", file=sys.stderr)

    if todoCount:
        print(f"{todoCount} TODO(s) need a human before this compiles.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
