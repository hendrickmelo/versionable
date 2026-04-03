"""Lazy loading support for HDF5 array fields.

When loading a Versionable from HDF5, array fields are replaced with
``LazyArray`` sentinels.  On first access, the sentinel loads the data
from the HDF5 file and caches it in the instance dict.

The mechanism uses a dynamically created subclass that overrides
``__getattribute__`` to intercept access to lazy fields.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import h5py

if TYPE_CHECKING:
    import numpy as np

from versionable.errors import ArrayNotLoadedError

logger = logging.getLogger(__name__)


class LazyArray:
    """Sentinel that loads an HDF5 dataset on first access.

    Stores the file path and dataset name; loads and returns the
    numpy array when ``load()`` is called.
    """

    def __init__(self, filePath: Path, datasetPath: str) -> None:
        self.filePath = filePath
        self.datasetPath = datasetPath

    def load(self) -> np.ndarray:
        with h5py.File(self.filePath, "r") as f:
            arr: np.ndarray = f[self.datasetPath][()]
            return arr

    def __repr__(self) -> str:
        return f"LazyArray({self.datasetPath!r})"


class ArrayNotLoaded:
    """Sentinel for ``metadataOnly=True`` — raises on access."""

    def __init__(self, fieldName: str) -> None:
        self.fieldName = fieldName

    def __repr__(self) -> str:
        return f"ArrayNotLoaded({self.fieldName!r})"


# Cache of dynamically created lazy subclasses
_lazyClassCache: dict[type, type] = {}


def makeLazyInstance(obj: Any, lazyFields: set[str]) -> Any:
    """Replace *obj* with an instance of a lazy subclass.

    The lazy subclass overrides ``__getattribute__`` to resolve
    ``LazyArray`` sentinels on first access and ``ArrayNotLoaded``
    sentinels with an error.
    """
    if not lazyFields:
        return obj

    cls = type(obj)
    lazyCls = _getLazyClass(cls)

    # Change the instance's class to the lazy subclass
    obj.__class__ = lazyCls
    return obj


def _getLazyClass(cls: type) -> type:
    """Get or create a lazy subclass for *cls*."""
    if cls in _lazyClassCache:
        return _lazyClassCache[cls]

    def __getattribute__(self: Any, name: str) -> Any:
        # Use object.__getattribute__ to avoid recursion
        val = object.__getattribute__(self, name)

        if isinstance(val, LazyArray):
            # Load and cache
            loaded = val.load()
            object.__setattr__(self, name, loaded)
            logger.debug("Lazy-loaded %s.%s from %s", type(self).__name__, name, val.datasetPath)
            return loaded

        if isinstance(val, ArrayNotLoaded):
            raise ArrayNotLoadedError(
                f"Array field {val.fieldName!r} was not loaded "
                f"(loaded with metadataOnly=True). "
                f"Reload without metadataOnly to access array data."
            )

        return val

    lazyCls = type(
        f"_Lazy{cls.__name__}",
        (cls,),
        {"__getattribute__": __getattribute__},
    )
    _lazyClassCache[cls] = lazyCls
    return lazyCls
