"""Public API: save() and load() free functions.

These are the primary entry points for serializing and deserializing
``Versionable`` objects.  Access via qualified import::

    import versionable
    versionable.save(obj, "output.yaml")
    loaded = versionable.load(MyClass, "output.yaml")
"""

from __future__ import annotations

import contextlib
import logging
from pathlib import Path
from typing import Any, TypeVar

from versionable._backend import Backend, getBackend
from versionable._base import Versionable, _resolveFields, metadata
from versionable._types import deserialize
from versionable.errors import BackendError, UnknownFieldError, VersionError

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=Versionable)


def save(
    obj: Versionable,
    path: str | Path,
    *,
    backend: type[Backend] | None = None,
    commentDefaults: bool = False,
    **kwargs: Any,
) -> None:
    """Serialize *obj* and write to *path*.

    Args:
        obj: A ``Versionable`` dataclass instance.
        path: Output file path.  Backend is auto-detected from extension.
        backend: Explicit backend class override.
        commentDefaults: If True and the backend supports it (TOML, YAML),
            fields at their default value are written as commented-out
            lines.  Ignored by backends that don't support comments.
        **kwargs: Additional backend-specific options (e.g. ``compression``
            for the HDF5 backend).
    """
    _ensureBackendsRegistered()

    path = Path(path)
    be = getBackend(path, explicit=backend)

    objType = type(obj)
    meta = metadata(objType)
    fields = _resolveFields(objType)

    rawFields: dict[str, Any] = {}
    for fieldName in fields:
        value = getattr(obj, fieldName)

        # skip_defaults: omit fields at their default value
        if meta.skipDefaults:
            import dataclasses

            # Versionable subclasses are always @dataclass; mypy/pyright can't prove that statically.
            dcFields = {f.name: f for f in dataclasses.fields(objType)}  # type: ignore[arg-type]
            if fieldName in dcFields:
                dcField = dcFields[fieldName]
                if dcField.default is not dataclasses.MISSING and value == dcField.default:
                    continue
                if dcField.default_factory is not dataclasses.MISSING and value == dcField.default_factory():
                    continue

        rawFields[fieldName] = value

    metaDict = {
        "name": meta.name,
        "version": meta.version,
        "hash": meta.hash,
    }
    be.save(rawFields, metaDict, path, cls=objType, commentDefaults=commentDefaults, **kwargs)


def load[T: Versionable](
    cls: type[T],
    path: str | Path,
    *,
    backend: type[Backend] | None = None,
    preload: list[str] | str | None = None,
    metadataOnly: bool = False,
    upgradeInPlace: bool = False,
    assumeVersion: int | None = None,
    validateLiterals: bool | None = None,
) -> T:
    """Load a ``Versionable`` object from *path*.

    Args:
        cls: The target dataclass type.
        path: Input file path.
        backend: Explicit backend class override.
        preload: Array field names to eagerly load, or ``'*'`` for all.
        metadataOnly: If True, array fields raise ``ArrayNotLoadedError``.
        upgradeInPlace: If True, allow file modification during migration.
        assumeVersion: Version to assume when the file has no ``__VERSION__``
            metadata.  Defaults to the class's current version.
        validateLiterals: Whether to validate ``Literal`` type values.
            Overrides the class-level ``validate_literals`` setting.
            Defaults to the class setting (``True`` unless overridden).

    Returns:
        An instance of *cls*.
    """
    # Ensure backend module is imported so backends are registered
    _ensureBackendsRegistered()

    path = Path(path)
    be = getBackend(path, explicit=backend)

    # Use lazy loading for HDF5 backend
    lazyFields: set[str] = set()
    loadLazy = getattr(be, "loadLazy", None)
    if loadLazy is not None and (preload != "*" or metadataOnly):
        rawFields, fileMeta, lazyFields = loadLazy(path, cls=cls, preload=preload, metadataOnly=metadataOnly)
    elif loadLazy is not None:
        # HDF5 with preload="*" — use loadLazy so cls is available for type dispatch
        rawFields, fileMeta, lazyFields = loadLazy(path, cls=cls, preload="*", metadataOnly=False)
    else:
        rawFields, fileMeta = be.load(path)

    meta = metadata(cls)
    fileVersion = fileMeta.get("__VERSION__")
    if fileVersion is None:
        if assumeVersion is not None:
            fileVersion = assumeVersion
        else:
            fileVersion = meta.version
            logger.warning(
                "No __VERSION__ found in '%s'. Treating as current version (%d) for %s. "
                "If this file was written by an older version of the code, "
                "pass assumeVersion= to load() to apply the correct migrations.",
                path,
                meta.version,
                meta.name,
            )

    # Version check — apply migrations if needed
    if fileVersion < meta.version:
        rawFields = _applyMigrations(cls, rawFields, fileVersion, meta.version)
    elif fileVersion > meta.version:
        raise VersionError(
            f"File version ({fileVersion}) is newer than class version "
            f"({meta.version}) for {meta.name}. Cannot downgrade."
        )

    # Handle unknown fields
    fields = _resolveFields(cls)
    knownFieldNames = set(fields.keys())
    unknownFields = set(rawFields.keys()) - knownFieldNames

    if unknownFields:
        if meta.unknown == "error":
            raise UnknownFieldError(f"Unknown fields in data for {meta.name}: {sorted(unknownFields)}")
        if meta.unknown == "ignore":
            for name in unknownFields:
                del rawFields[name]
        # "preserve" mode: keep them (will be passed through)

    # Deserialize fields
    import dataclasses

    nativeTypes = be.nativeTypes
    effectiveValidateLiterals = validateLiterals if validateLiterals is not None else meta.validateLiterals
    # Versionable subclasses are always @dataclass; mypy/pyright can't prove that statically.
    dcFields = {f.name: f for f in dataclasses.fields(cls)}  # type: ignore[arg-type]
    kwargs: dict[str, Any] = {}
    for fieldName, fieldType in fields.items():
        if fieldName in rawFields:
            rawValue = rawFields[fieldName]
            # Skip deserialization for lazy sentinels
            if fieldName in lazyFields:
                kwargs[fieldName] = rawValue
            else:
                dcField = dcFields.get(fieldName)
                dcMeta = dcField.metadata if dcField is not None else None
                kwargs[fieldName] = deserialize(
                    rawValue,
                    fieldType,
                    nativeTypes=nativeTypes,
                    fieldMetadata=dcMeta,
                    validateLiterals=effectiveValidateLiterals,
                )
        # Use dataclass default
        elif fieldName in dcFields:
            dcField = dcFields[fieldName]
            if dcField.default is not dataclasses.MISSING:
                kwargs[fieldName] = dcField.default
            elif dcField.default_factory is not dataclasses.MISSING:
                kwargs[fieldName] = dcField.default_factory()
            else:
                raise BackendError(
                    f"Field '{fieldName}' is missing from '{path}' and has no default value. "
                    f"Class '{meta.name}' (version {meta.version}) requires this field. "
                    f"Either add '{fieldName}' to the file or give it a default in the dataclass."
                )

    instance = cls(**kwargs)

    # Apply lazy loading wrapper if there are lazy fields
    if lazyFields:
        from versionable._lazy import makeLazyInstance

        instance = makeLazyInstance(instance, lazyFields)

    return instance


def loadDynamic(
    path: str | Path,
    *,
    backend: type[Backend] | None = None,
    baseClass: type[Versionable] | None = None,
) -> Versionable:
    """Load a Versionable object whose type is determined from file metadata.

    Args:
        path: Input file path.
        backend: Explicit backend class override.
        baseClass: Optional base class to restrict the type.
    """
    from versionable._base import registeredClasses

    _ensureBackendsRegistered()

    path = Path(path)
    be = getBackend(path, explicit=backend)
    _rawFields, fileMeta = be.load(path)

    objectName = fileMeta.get("__OBJECT__", "")
    registry = registeredClasses()

    if objectName not in registry:
        raise BackendError(f"Unknown object type {objectName!r} in {path}")

    cls = registry[objectName]
    if baseClass is not None and not issubclass(cls, baseClass):
        raise BackendError(f"Object type {objectName!r} is not a subclass of {baseClass.__name__}")

    return load(cls, path, backend=backend)


# ---------------------------------------------------------------------------
# Migration helpers
# ---------------------------------------------------------------------------


def _applyMigrations(
    cls: type[Versionable],
    data: dict[str, Any],
    fromVersion: int,
    toVersion: int,
) -> dict[str, Any]:
    """Apply migrations to upgrade *data* from *fromVersion* to *toVersion*.

    This is a placeholder — the full migration system is implemented in
    Phase D (_migration.py).
    """
    try:
        from versionable._migration import applyMigrations, resolveMigrations
    except ImportError:
        raise VersionError(
            f"File is version {fromVersion} but class is version {toVersion}. Migration system not available."
        ) from None

    migrations = resolveMigrations(cls, fromVersion, toVersion)
    return applyMigrations(data, migrations)


# ---------------------------------------------------------------------------
# Backend registration
# ---------------------------------------------------------------------------

_backendsRegistered = False


def _ensureBackendsRegistered() -> None:
    """Import backend modules to ensure they register themselves."""
    global _backendsRegistered
    if _backendsRegistered:
        return
    _backendsRegistered = True

    # Import backends so they call registerBackend()
    import versionable._json_backend
    import versionable._toml_backend
    import versionable._yaml_backend

    with contextlib.suppress(ImportError):
        import versionable._hdf5_backend  # noqa: F401 — side-effect import registers the HDF5 backend
