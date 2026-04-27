"""Unified serialization framework for Python dataclasses.

Handles both structured metadata and large array data with versioning,
migration support, and pluggable storage backends.

Usage::

    from dataclasses import dataclass
    import versionable
    from versionable import Versionable

    @dataclass
    class Config(Versionable, version=1, hash='1336b2'):
        name: str
        debug: bool = False

    config = Config(name="probe-A")
    versionable.save(config, "config.yaml")
    loaded = versionable.load(Config, "config.yaml")
"""

from __future__ import annotations

from importlib.metadata import version as _version

from versionable._api import load, loadDynamic, save
from versionable._backend import Backend, registerBackend
from versionable._base import (
    Versionable,
    VersionableMetadata,
    getVersionableFields,
    ignoreHashErrors,
    metadata,
    registeredClasses,
)
from versionable._hdf5_field import Hdf5FieldInfo
from versionable._json_backend import JsonBackend
from versionable._migration import Migration, MigrationContext, migration
from versionable._types import VersionableValue, literalFallback, registerConverter

__version__ = _version("versionable")
from versionable.errors import (
    ArrayNotLoadedError,
    BackendError,
    CircularReferenceError,
    ConverterError,
    HashMismatchError,
    MigrationError,
    UnknownFieldError,
    UpgradeRequiredError,
    VersionableError,
    VersionError,
)


def __getattr__(name: str) -> type:
    """Lazily import optional backend classes."""
    if name == "Hdf5Backend":
        from versionable._hdf5_backend import Hdf5Backend

        return Hdf5Backend
    if name == "TomlBackend":
        from versionable._toml_backend import TomlBackend

        return TomlBackend
    if name == "YamlBackend":
        from versionable._yaml_backend import YamlBackend

        return YamlBackend
    msg = f"module 'versionable' has no attribute {name!r}"
    raise AttributeError(msg)


__all__ = [
    "ArrayNotLoadedError",
    "Backend",
    "BackendError",
    "CircularReferenceError",
    "ConverterError",
    "HashMismatchError",
    "Hdf5Backend",
    "Hdf5FieldInfo",
    "JsonBackend",
    "Migration",
    "MigrationContext",
    "MigrationError",
    "TomlBackend",
    "UnknownFieldError",
    "UpgradeRequiredError",
    "VersionError",
    "Versionable",
    "VersionableError",
    "VersionableMetadata",
    "VersionableValue",
    "YamlBackend",
    "__version__",
    "getVersionableFields",
    "ignoreHashErrors",
    "literalFallback",
    "load",
    "loadDynamic",
    "metadata",
    "migration",
    "registerBackend",
    "registerConverter",
    "registeredClasses",
    "save",
]
