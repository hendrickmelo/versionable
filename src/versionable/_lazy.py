"""Lazy loading support for HDF5 array fields.

When loading a Versionable from HDF5, array fields are replaced with
``LazyArray`` sentinels.  On first access, the sentinel loads the data
from the HDF5 file and caches it in the instance dict.

For ``list[np.ndarray]`` and ``dict[str, np.ndarray]`` fields,
``LazyArrayList`` and ``LazyArrayDict`` provide per-element lazy loading:
the collection structure is known immediately but individual arrays are
loaded only when accessed.

The mechanism uses a dynamically created subclass that overrides
``__getattribute__`` to intercept access to lazy fields.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, overload

import h5py

if TYPE_CHECKING:
    from collections.abc import Iterator

    import numpy as np

from versionable._hdf5_plugin import missingFilterHint
from versionable.errors import ArrayNotLoadedError, BackendError

logger = logging.getLogger(__name__)


@contextmanager
def _rewrapFilterErrors(filePath: Path) -> Iterator[None]:
    """Re-raise filter-related OSError as BackendError with a hdf5plugin install hint."""
    try:
        yield
    except OSError as e:
        hint = missingFilterHint(e)
        if hint:
            raise BackendError(f"Failed to read HDF5 from {filePath}: {e}{hint}") from e
        raise


def _loadDataset(f: h5py.File, path: str) -> np.ndarray:
    """Load a dataset from an open HDF5 file, with type narrowing."""
    ds = f[path]
    if not isinstance(ds, h5py.Dataset):
        raise TypeError(f"Expected Dataset at {path!r}, got {type(ds).__name__}")
    result: np.ndarray = ds[()]
    return result


@dataclass
class LazyContext:
    """Controls lazy loading behavior during HDF5 reads.

    Passed through the recursive read path so every level of the object
    tree applies the same lazy/eager/metadataOnly policy.
    """

    path: Path
    preloadAll: bool = False
    preloadSet: set[str] = field(default_factory=set)
    metadataOnly: bool = False


def isLazySentinel(value: Any) -> bool:
    """Check if *value* is a lazy loading sentinel that should skip deserialization."""
    return isinstance(value, (LazyArray, LazyArrayList, LazyArrayDict, ArrayNotLoaded))


class LazyArray:
    """Sentinel that loads an HDF5 dataset on first access.

    Has ``_isLazySentinel = True`` so generic code can detect lazy sentinels
    without importing this module.

    Stores the file path and dataset name; loads and returns the
    numpy array when ``load()`` is called.
    """

    _isLazySentinel = True

    def __init__(self, filePath: Path, datasetPath: str) -> None:
        self.filePath = filePath
        self.datasetPath = datasetPath

    def load(self) -> np.ndarray:
        with _rewrapFilterErrors(self.filePath), h5py.File(self.filePath, "r") as f:
            return _loadDataset(f, self.datasetPath)

    def __repr__(self) -> str:
        return f"LazyArray({self.datasetPath!r})"


class LazyArrayList:
    """A list-like container where each element is lazily loaded from HDF5.

    Supports ``len()``, indexing, iteration, and slicing.
    Each element is loaded and cached on first access.
    """

    _isLazySentinel = True

    def __init__(self, filePath: Path, groupPath: str, keys: list[str]) -> None:
        self.filePath = filePath
        self.groupPath = groupPath
        self._keys = keys
        self._cache: dict[int, np.ndarray] = {}

    def __len__(self) -> int:
        return len(self._keys)

    @overload
    def __getitem__(self, index: int) -> np.ndarray: ...

    @overload
    def __getitem__(self, index: slice) -> list[np.ndarray]: ...

    def __getitem__(self, index: int | slice) -> np.ndarray | list[np.ndarray]:
        if isinstance(index, slice):
            return [self[i] for i in range(*index.indices(len(self)))]
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(f"index {index} out of range for LazyArrayList of length {len(self)}")
        if index not in self._cache:
            key = self._keys[index]
            with _rewrapFilterErrors(self.filePath), h5py.File(self.filePath, "r") as f:
                self._cache[index] = _loadDataset(f, f"{self.groupPath}/{key}")
            logger.debug("Lazy-loaded %s/%s", self.groupPath, key)
        return self._cache[index]

    def __iter__(self) -> Any:
        for i in range(len(self)):
            yield self[i]

    def __repr__(self) -> str:
        loaded = len(self._cache)
        return f"LazyArrayList({self.groupPath!r}, {len(self)} items, {loaded} loaded)"


class LazyArrayDict:
    """A dict-like container where each value is lazily loaded from HDF5.

    Supports ``len()``, key access, ``keys()``, ``values()``, ``items()``,
    and iteration.  Each value is loaded and cached on first access.
    """

    _isLazySentinel = True

    def __init__(
        self,
        filePath: Path,
        groupPath: str,
        keys: list[Any],
        hdf5Keys: list[str] | None = None,
    ) -> None:
        self.filePath = filePath
        self.groupPath = groupPath
        self._keys = keys
        self._keySet: set[Any] = set(keys)
        # hdf5Keys are the raw (possibly percent-encoded) names in the file
        self._hdf5Keys = hdf5Keys or [str(k) for k in keys]
        self._keyToHdf5: dict[Any, str] = dict(zip(self._keys, self._hdf5Keys, strict=True))
        self._cache: dict[Any, np.ndarray] = {}

    def __len__(self) -> int:
        return len(self._keys)

    def __getitem__(self, key: Any) -> np.ndarray:
        if key not in self._cache:
            if key not in self._keyToHdf5:
                raise KeyError(key)
            hdf5Key = self._keyToHdf5[key]
            with _rewrapFilterErrors(self.filePath), h5py.File(self.filePath, "r") as f:
                self._cache[key] = _loadDataset(f, f"{self.groupPath}/{hdf5Key}")
            logger.debug("Lazy-loaded %s/%s", self.groupPath, hdf5Key)
        return self._cache[key]

    def __contains__(self, key: object) -> bool:
        return key in self._keySet

    def __iter__(self) -> Any:
        return iter(self._keys)

    def keys(self) -> list[Any]:
        return list(self._keys)

    def values(self) -> list[np.ndarray]:
        return [self[k] for k in self._keys]

    def items(self) -> list[tuple[Any, np.ndarray]]:
        return [(k, self[k]) for k in self._keys]

    def __repr__(self) -> str:
        loaded = len(self._cache)
        return f"LazyArrayDict({self.groupPath!r}, {len(self)} items, {loaded} loaded)"


class ArrayNotLoaded:
    """Sentinel for ``metadataOnly=True`` — raises on access."""

    _isLazySentinel = True

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
