"""Versionable base class, metadata access, and class registry.

Provides the ``Versionable`` mixin that dataclasses inherit from to
opt in to the serialization framework.  Schema version, hash, and
serialization options are declared as class parameters and validated
at definition time.
"""

from __future__ import annotations

import logging
import typing
from dataclasses import dataclass
from typing import Any, ClassVar

from versionable._hash import collectSerializationNames, computeHash
from versionable.errors import HashMismatchError, VersionableError

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level state
# ---------------------------------------------------------------------------

_REGISTRY: dict[str, type[Versionable]] = {}
_ignoreHashErrors: bool = False

# Classes already warned about for unresolvable annotations, keyed by id() so a
# class is never kept alive by this set.
_WARNED_UNRESOLVED: set[int] = set()


def ignoreHashErrors(enable: bool = True) -> None:
    """Treat hash mismatches as warnings instead of errors (dev mode)."""
    global _ignoreHashErrors
    _ignoreHashErrors = enable


def registeredClasses() -> dict[str, type[Versionable]]:
    """Return a copy of the class registry (keyed by serialization name)."""
    return dict(_REGISTRY)


# ---------------------------------------------------------------------------
# Metadata dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class VersionableMetadata:
    """Serialization metadata for a ``Versionable`` subclass."""

    version: int
    hash: str
    name: str
    fields: list[str]
    skipDefaults: bool
    unknown: str
    validateLiterals: bool = True
    minReversibleVersion: int | None = None


def metadata(cls: type[Versionable]) -> VersionableMetadata:
    """Return the ``VersionableMetadata`` for a ``Versionable`` subclass."""
    meta: _InternalMeta = cls._serializer_meta_
    fields = _resolveFields(cls)
    return VersionableMetadata(
        version=meta.version,
        hash=meta.hash,
        name=meta.name,
        fields=list(fields.keys()),
        skipDefaults=meta.skipDefaults,
        unknown=meta.unknown,
        validateLiterals=meta.validateLiterals,
    )


# ---------------------------------------------------------------------------
# Internal metadata (stored on each subclass)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _InternalMeta:
    version: int
    hash: str
    name: str
    oldNames: list[str]
    register: bool
    skipDefaults: bool
    unknown: str  # "ignore" | "error" | "preserve"
    validateLiterals: bool


# ---------------------------------------------------------------------------
# Versionable base class
# ---------------------------------------------------------------------------


class Versionable:
    """Mixin for dataclasses that participate in the serialization framework.

    Declare as a base class with version and hash parameters::

        @dataclass
        class Config(Versionable, version=1, hash='a1b2c3'):
            name: str
            debug: bool = False

    The hash is validated at class definition time.  Use
    ``ignoreHashErrors(True)`` during development to get warnings
    instead of errors.
    """

    _serializer_meta_: ClassVar[_InternalMeta]

    @classmethod
    def hash(cls) -> str:
        """Compute the schema hash for this Versionable subclass.

        Returns:
            First 6 hex characters of the SHA-256 digest.
        """
        fields = _resolveFields(cls)
        return computeHash(fields)

    def __init_subclass__(
        cls,
        *,
        version: int = 0,
        hash: str = "",  # noqa: A002 — intentional; matches the class definition API (Versionable, hash='...')
        name: str | None = None,
        old_names: list[str] | None = None,
        register: bool = True,
        skip_defaults: bool = False,
        unknown: str = "ignore",
        validate_literals: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init_subclass__(**kwargs)

        # Skip intermediate base classes that don't declare version
        if version == 0 and hash == "" and not _hasOwnAnnotations(cls):
            return

        serializationName = name or cls.__name__
        oldNames = old_names or []

        meta = _InternalMeta(
            version=version,
            hash=hash,
            name=serializationName,
            oldNames=oldNames,
            register=register,
            skipDefaults=skip_defaults,
            unknown=unknown,
            validateLiterals=validate_literals,
        )
        cls._serializer_meta_ = meta

        # Validate the schema itself, then the declared hash.  The hash is
        # computed even when none is declared: canonicalisation is what rejects
        # constructs the grammar closes off (unsupported Literal options, array
        # dtypes outside the token table, colliding serialization names), and
        # those are errors in the schema, not in the declared hash.
        fields = _resolveFields(cls)
        collectSerializationNames(fields)
        computed = computeHash(fields)

        if hash and computed != hash:
            if _ignoreHashErrors:
                logger.warning(
                    "%s: hash mismatch — declared %r, computed %r",
                    cls.__qualname__,
                    hash,
                    computed,
                )
            else:
                raise HashMismatchError(cls, hash, computed)

        # Register class
        if register:
            # Validate all names before mutating the registry so a collision
            # on an old_name doesn't leave a partially-registered class.
            existing = _REGISTRY.get(serializationName)
            if existing is not None and existing is not cls:
                raise VersionableError(
                    f"Versionable name {serializationName!r} is already registered to "
                    f"{existing.__qualname__}. Give one of the classes an explicit, distinct "
                    f"name to disambiguate, e.g.: "
                    f"class {cls.__name__}(Versionable, ..., "
                    f'name="{cls.__name__}V2")  # any unused bare name'
                )
            for oldName in oldNames:
                existing = _REGISTRY.get(oldName)
                if existing is not None and existing is not cls:
                    raise VersionableError(
                        f"old_names entry {oldName!r} is already registered to "
                        f"{existing.__qualname__}. Use a different old_name or set "
                        f"register=False on one of the classes."
                    )
            _REGISTRY[serializationName] = cls
            for oldName in oldNames:
                _REGISTRY[oldName] = cls


# ---------------------------------------------------------------------------
# Field introspection helpers
# ---------------------------------------------------------------------------


def _hasOwnAnnotations(cls: type) -> bool:
    """Return True if *cls* defines its own ``__annotations__`` (not inherited)."""
    return "__annotations__" in cls.__dict__


def _warnUnresolvedAnnotations(cls: type, hints: dict[str, Any]) -> None:
    """Warn that *cls* has annotations that could not be resolved to real types.

    An unresolved annotation reaches ``canonicalTypeName`` as a raw source
    string and renders verbatim, so ``partner: Node | None`` yields the
    canonical string ``Node | None`` rather than the grammar's
    ``Union[None, Node]``.  The hash is still deterministic within Python, but
    another implementation of the grammar cannot reproduce it.
    """
    unresolved = sorted(name for name, hint in hints.items() if isinstance(hint, str) and not name.startswith("_"))
    # _resolveFields runs on every save and load, so warn once per class rather
    # than once per operation.
    if not unresolved or id(cls) in _WARNED_UNRESOLVED:
        return
    _WARNED_UNRESOLVED.add(id(cls))
    logger.warning(
        "%s: could not resolve the type annotations for %s; they will be hashed as their raw "
        "source text, which may not match the canonical type grammar and may therefore produce "
        "a schema hash another language implementation cannot reproduce. This usually means two "
        "classes refer to each other; define them in one module and reference the later one by "
        "name, or declare the field with an explicit resolvable type.",
        cls.__qualname__,
        ", ".join(repr(name) for name in unresolved),
    )


def _resolveFields(cls: type) -> dict[str, Any]:
    """Return the serializable fields for *cls* as ``{name: type}``.

    A field is serializable if it:
    - Has a type annotation
    - Does not start with ``_``

    Uses ``typing.get_type_hints`` to resolve forward references and
    ``Annotated`` wrappers.

    During ``__init_subclass__`` the class's own name is not yet bound in its
    module, so a self-referential annotation (``parent: Node | None``) cannot be
    resolved from the module globals alone.  ``cls`` is passed in as a local
    name for the retry: without it the raw annotation *string* would reach
    ``canonicalTypeName`` and render verbatim as ``Node | None`` instead of the
    canonical ``Union[Node, None]`` (GRAMMAR.md section 6), giving a
    definition-time hash no other implementation can reproduce.

    That retry does not cover *mutual* recursion (``A`` referring to a ``B``
    defined later), where neither name exists yet.  Those annotations fall back
    to their raw strings and the resulting hash may not be conformant, so the
    fallback warns rather than failing silently — see the follow-up tracked in
    ``docs/plans/csharp-port.md`` phase 5.
    """
    try:
        hints = typing.get_type_hints(cls, include_extras=True)
    except Exception:
        try:
            hints = typing.get_type_hints(cls, localns={cls.__name__: cls}, include_extras=True)
        except Exception:
            # Fallback to raw annotations if resolution still fails (e.g. mutual
            # recursion, where the other class does not exist yet either).
            hints = {}
            for klass in reversed(cls.__mro__):
                hints.update(getattr(klass, "__annotations__", {}))
            _warnUnresolvedAnnotations(cls, hints)

    fields: dict[str, Any] = {}
    for fieldName, fieldType in hints.items():
        # Skip private fields
        if fieldName.startswith("_"):
            continue
        # Skip ClassVar
        if typing.get_origin(fieldType) is ClassVar:
            continue
        # Skip ClassVar written as string
        if isinstance(fieldType, str) and fieldType.startswith("ClassVar"):
            continue
        fields[fieldName] = fieldType

    return fields


def getVersionableFields(cls: type[Versionable]) -> dict[str, Any]:
    """Public API to get the serializable fields for a Versionable subclass.

    Returns:
        Mapping of field name to resolved type annotation.
    """
    return _resolveFields(cls)
