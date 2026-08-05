"""Exception classes for the versionable framework.

All exceptions are defined here to avoid circular imports between modules.
"""

from __future__ import annotations

from typing import Literal

# Where a dtype check was triggered from — used only in error messages.
DtypeContext = Literal["save", "load"]


class VersionableError(Exception):
    """Base exception for all versionable errors."""


class UnsupportedTypeError(VersionableError):
    """A declared type cannot be expressed in the canonical type grammar.

    Raised at class definition time (while the schema hash is computed) for
    constructs the language-neutral grammar deliberately closes off: a
    ``Literal`` option that is not a string, int, bool, ``None`` or enum
    member, and an array dtype outside the supported token table.  Falling
    back to a Python-specific rendering would produce a hash no other
    implementation could reproduce.
    """


class HashMismatchError(VersionableError):
    """Declared hash does not match computed hash.

    Raised at class definition time when the ``hash`` parameter on a
    ``Versionable`` subclass disagrees with the hash computed from the
    current field names and types.
    """

    def __init__(self, cls: type, declared: str, computed: str) -> None:
        self.cls = cls
        self.declared = declared
        self.computed = computed
        super().__init__(
            f"{cls.__qualname__}: hash mismatch — "
            f"declared {declared!r}, computed {computed!r}. "
            f"Update the hash parameter to {computed!r}."
        )


class VersionError(VersionableError):
    """Version-related error (e.g., missing migration for a version gap)."""


class MigrationError(VersionableError):
    """A migration could not be applied."""


class ArrayNotLoadedError(VersionableError, AttributeError):
    """Raised when accessing an array field loaded with ``metadataOnly=True``.

    Inherits from ``AttributeError`` so ``hasattr()`` returns ``False``
    for unloaded array fields.
    """


class UpgradeRequiredError(VersionableError):
    """Migration requires in-place file modification.

    Raised when a migration cannot be performed in memory and the caller
    did not pass ``upgradeInPlace=True``.
    """


class UnknownFieldError(VersionableError):
    """Source data contains a field not declared on the target class.

    Only raised when the class is configured with ``unknown='error'``.
    """


class ConverterError(VersionableError):
    """A type conversion failed during serialization or deserialization."""


class DtypeMismatchError(ConverterError):
    """An array's dtype cannot be safely cast to the dtype its annotation declares.

    Array dtype is hash-significant: ``NDArray[np.float64]`` canonicalises to
    ``ndarray[float64]``.  Safe casts (``np.can_cast(..., casting='safe')``) are
    applied silently; anything lossy raises this error rather than letting the
    file disagree with the schema its hash describes.

    Attributes:
        declared: Canonical name of the declared dtype (e.g. ``float32``).
        actual: Canonical name of the value's dtype.
        fieldPath: Dotted path of the offending field, empty for the root.
        context: ``'save'`` or ``'load'``.
    """

    def __init__(self, *, declared: str, actual: str, fieldPath: str = "", context: DtypeContext = "save") -> None:
        self.declared = declared
        self.actual = actual
        self.fieldPath = fieldPath
        self.context = context
        super().__init__(
            f"Array dtype mismatch at {fieldPath or '<root>'} during {context}: "
            f"declared ndarray[{declared}], data is {actual}. "
            f"Casting {actual} to {declared} is not safe. "
            f"Cast explicitly with .astype('{declared}') if the loss is intended, "
            f"or declare ndarray[{actual}] and update the schema hash."
        )


class BackendError(VersionableError):
    """A storage backend operation failed."""


class CircularReferenceError(VersionableError):
    """A cycle was detected in the object graph during serialization.

    Raised when ``serialize()`` (JSON/YAML/TOML) or the HDF5 writer
    encounters a ``Versionable`` instance that is already on the
    serialization stack — i.e. an object reaches itself by following its
    own fields.

    Carries structured data so callers can inspect the cycle without
    parsing the message string:

    Attributes:
        path: Field path to the revisit (e.g. ``children[0]``,
            ``partner.partner``, ``partners['alice']``).  Empty string
            denotes the root object (rare — the root is only reported
            here when the same instance is reassigned to itself in a
            session).
        objType: Type of the revisited instance.
        objId: ``id()`` of the revisited instance, as an int.  The
            message renders this in hex so a self-cycle can be told
            apart from a legitimate diamond (same instance referenced
            from two unrelated branches; not a cycle).
    """

    def __init__(self, path: str, obj: object) -> None:
        self.path = path
        self.objType = type(obj)
        self.objId = id(obj)
        super().__init__(
            f"Circular reference detected at field path "
            f"{path or '<root>'} → {self.objType.__name__}@{self.objId:x}. "
            f"versionable cannot serialize cycles in 0.2.x. "
            f"Lossless shared-reference support is planned for 0.3.0 "
            f"via an opt-in shared_refs=True flag."
        )
