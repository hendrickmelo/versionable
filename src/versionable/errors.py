"""Exception classes for the versionable framework.

All exceptions are defined here to avoid circular imports between modules.
"""

from __future__ import annotations


class VersionableError(Exception):
    """Base exception for all versionable errors."""


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


class BackendError(VersionableError):
    """A storage backend operation failed."""


class CircularReferenceError(VersionableError):
    """A cycle was detected in the object graph during serialization.

    Raised when ``serialize()`` (JSON/YAML/TOML) or the HDF5 writer
    encounters a ``Versionable`` instance that is already on the
    serialization stack — i.e. an object reaches itself by following its
    own fields.

    The message includes the field path of the revisit and the type and
    ``id()`` of the offending instance, so a self-cycle can be told
    apart from a legitimate diamond (same instance referenced from two
    unrelated branches; not a cycle).
    """
