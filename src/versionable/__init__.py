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
from versionable._migration import Migration, MigrationContext, migration
from versionable._types import VersionableValue, literalFallback, registerConverter

__version__ = _version("versionable")
from versionable.errors import (
    ArrayNotLoadedError,
    BackendError,
    ConverterError,
    HashMismatchError,
    MigrationError,
    UnknownFieldError,
    UpgradeRequiredError,
    VersionableError,
    VersionError,
)

__all__ = [
    "ArrayNotLoadedError",
    "Backend",
    "BackendError",
    "ConverterError",
    "HashMismatchError",
    "Migration",
    "MigrationContext",
    "MigrationError",
    "UnknownFieldError",
    "UpgradeRequiredError",
    "VersionError",
    "Versionable",
    "VersionableError",
    "VersionableMetadata",
    "VersionableValue",
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
