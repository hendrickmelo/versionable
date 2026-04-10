"""HDF5 save-as-you-go session.

Provides ``Hdf5Session``, a context manager that opens an HDF5 file and
returns a live, file-backed instance of a Versionable type. Field
assignments are intercepted and persisted incrementally.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, cast

import h5py
import numpy as np

from versionable._appendable import (
    Appendable,
    _computeChunkSize,
    _getAppendable,
    _resolveAppendAxis,
)
from versionable._base import Versionable, _resolveFields, metadata
from versionable._hdf5_backend import (
    _VERSIONABLE_GROUP,
    _writeValue,
)
from versionable._hdf5_compression import ZSTD_DEFAULT, Hdf5Compression
from versionable._tracked_array import TrackedArray
from versionable.errors import BackendError

logger = logging.getLogger(__name__)

# Cache of dynamically created proxy subclasses
_proxyClassCache: dict[type, type] = {}


class Hdf5Session[T: Versionable]:
    """Context manager wrapping a file-backed Versionable instance.

    Mutations to the proxy instance returned by ``__enter__`` are
    intercepted and persisted to the HDF5 file incrementally.

    Not thread-safe — HDF5 does not support concurrent writes by default.
    """

    _file: h5py.File
    _root: h5py.Group
    _cls: type[T]
    _fieldTypes: dict[str, Any]
    _comp: Hdf5Compression
    _mode: str
    _proxy: T
    _path: Path

    def __init__(
        self,
        cls: type[T],
        path: str | Path,
        *,
        mode: str = "create",
        compression: Hdf5Compression | None = None,
    ) -> None:
        self._cls = cls
        self._path = Path(path)
        self._mode = mode
        self._comp = compression or ZSTD_DEFAULT
        self._fieldTypes = _resolveFields(cls)

    def __enter__(self) -> T:
        if self._mode == "create":
            if self._path.exists():
                raise BackendError(
                    f"File {self._path} already exists. Use mode='overwrite' to replace or mode='resume' to continue."
                )
            self._file = h5py.File(self._path, "w")
        elif self._mode == "overwrite":
            if self._path.exists():
                os.remove(self._path)
            self._file = h5py.File(self._path, "w")
        elif self._mode == "resume":
            raise BackendError("Resume mode is not yet implemented.")
        else:
            raise BackendError(f"Unknown session mode: {self._mode!r}")

        self._root = self._file

        # Write __versionable__ metadata
        meta = metadata(self._cls)
        metaGroup = self._root.create_group(_VERSIONABLE_GROUP)
        metaGroup.attrs["__OBJECT__"] = meta.name
        metaGroup.attrs["__VERSION__"] = meta.version
        metaGroup.attrs["__HASH__"] = meta.hash

        # Create proxy instance
        proxyCls = _getProxyClass(self._cls)
        # Use object.__new__ to skip __init__ (no args needed)
        self._proxy = proxyCls.__new__(proxyCls)
        # Set session reference without triggering __setattr__ override
        object.__setattr__(self._proxy, "_session", self)

        return self._proxy

    def __exit__(
        self,
        excType: type[BaseException] | None,
        excVal: BaseException | None,
        excTb: Any,
    ) -> None:
        try:
            self._file.close()
        except Exception:
            logger.exception("Error closing HDF5 file %s", self._path)

    def _persistField(self, name: str, value: Any) -> None:
        """Write a single field to the HDF5 file."""
        fieldType = self._fieldTypes[name]
        datasetKwargs = self._comp.datasetKwargs()

        # Unwrap TrackedArray to get the raw ndarray for writing
        if isinstance(value, TrackedArray):
            value = np.asarray(value)

        # Remove existing data for this field
        if name in self._root.attrs:
            del self._root.attrs[name]
        if name in self._root:
            del self._root[name]

        # Check if this is an Appendable field
        appendable = _getAppendable(fieldType)
        if appendable is not None and isinstance(value, np.ndarray):
            self._createResizableDataset(name, value, appendable)
        else:
            _writeValue(self._root, name, value, fieldType, datasetKwargs, self._comp)

    def _createResizableDataset(
        self,
        name: str,
        data: np.ndarray,
        appendable: Appendable,
    ) -> tuple[h5py.Dataset, int]:
        """Create a chunked, resizable dataset for an Appendable field.

        Returns:
            Tuple of (dataset, resolvedAxis).
        """
        axis = _resolveAppendAxis(data.shape, appendable)

        # Chunk shape
        chunkShape = list(data.shape)
        if appendable.chunkRows is not None:
            chunkShape[axis] = appendable.chunkRows
        else:
            chunkShape[axis] = _computeChunkSize(data.shape, data.dtype, axis)
        chunkShape[axis] = max(1, chunkShape[axis])

        # maxshape: None (unlimited) along the append axis
        maxshape = list(data.shape)
        maxshape[axis] = None

        ds = self._root.create_dataset(
            name,
            data=data,
            chunks=tuple(chunkShape),
            maxshape=tuple(maxshape),
            **self._comp.datasetKwargs(),
        )
        return ds, axis

    def _wrapValue(self, name: str, value: Any, fieldType: Any) -> Any:
        """Wrap a value with a tracked proxy if applicable."""
        # Appendable ndarray -> TrackedArray backed by the just-created dataset
        appendable = _getAppendable(fieldType)
        if appendable is not None and name in self._root:
            dataset = self._root[name]
            if isinstance(dataset, h5py.Dataset):
                axis = _resolveAppendAxis(dataset.shape, appendable)
                return TrackedArray(dataset, name, axis)

        return value


def _getProxyClass[T: Versionable](cls: type[T]) -> type[T]:
    """Get or create a dynamic proxy subclass that persists field assignments."""
    if cls in _proxyClassCache:
        return _proxyClassCache[cls]

    def __setattr__(self: Any, name: str, value: Any) -> None:
        session: Hdf5Session[Any] = object.__getattribute__(self, "_session")
        fieldTypes = session._fieldTypes  # noqa: SLF001

        if name in fieldTypes:
            session._persistField(name, value)  # noqa: SLF001
            value = session._wrapValue(name, value, fieldTypes[name])  # noqa: SLF001

        object.__setattr__(self, name, value)

    proxyCls = cast("type[T]", type(f"_Live{cls.__name__}", (cls,), {"__setattr__": __setattr__}))
    _proxyClassCache[cls] = proxyCls
    return proxyCls
