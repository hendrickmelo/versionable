"""HDF5 save-as-you-go session.

Provides ``Hdf5Session``, a context manager that opens an HDF5 file and
returns a live, file-backed instance of a Versionable type. Field
assignments are intercepted and persisted incrementally.
"""

from __future__ import annotations

import logging
import os
import typing
from collections.abc import Iterable
from pathlib import Path
from typing import Any, SupportsIndex, cast

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
    _dtypeForElementType,
    _isScalarType,
    _keyToStr,
    _readFields,
    _readMeta,
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
            if not self._path.exists():
                raise BackendError(f"Cannot resume: file {self._path} does not exist.")
            self._file = h5py.File(self._path, "a")
        else:
            raise BackendError(f"Unknown session mode: {self._mode!r}")

        self._root = self._file

        # Create proxy instance
        proxyCls = _getProxyClass(self._cls)
        self._proxy = proxyCls.__new__(proxyCls)
        object.__setattr__(self._proxy, "_session", self)

        if self._mode == "resume":
            self._resumeFromFile()
        else:
            # Write __versionable__ metadata for new files
            meta = metadata(self._cls)
            metaGroup = self._root.create_group(_VERSIONABLE_GROUP)
            metaGroup.attrs["__OBJECT__"] = meta.name
            metaGroup.attrs["__VERSION__"] = meta.version
            metaGroup.attrs["__HASH__"] = meta.hash

        return self._proxy

    def _resumeFromFile(self) -> None:
        """Restore state from an existing HDF5 file for resume mode."""
        # Validate metadata
        fileMeta = _readMeta(self._root)
        meta = metadata(self._cls)
        if fileMeta["__OBJECT__"] != meta.name:
            raise BackendError(
                f"Cannot resume: file contains type {fileMeta['__OBJECT__']!r}, "
                f"but session was opened with {meta.name!r}."
            )
        if fileMeta["__VERSION__"] != meta.version:
            raise BackendError(
                f"Cannot resume: file has version {fileMeta['__VERSION__']}, but class has version {meta.version}."
            )
        if fileMeta["__HASH__"] != meta.hash:
            raise BackendError(
                f"Cannot resume: file has hash {fileMeta['__HASH__']!r}, but class has hash {meta.hash!r}."
            )

        # Load existing fields
        fields, _ = _readFields(self._root, self._fieldTypes)

        # Populate proxy and wrap with tracked proxies
        for name, value in fields.items():
            fieldType = self._fieldTypes.get(name)
            if fieldType is not None:
                wrapped = self._wrapResumedValue(name, value, fieldType)
                object.__setattr__(self._proxy, name, wrapped)

    def _wrapResumedValue(self, name: str, value: Any, fieldType: Any) -> Any:
        """Wrap a loaded value with a tracked proxy for resume mode.

        For Appendable fields, wraps with TrackedArray pointing to the
        existing resizable dataset. For list/dict fields, wraps with
        TrackedList/TrackedDict.
        """
        # Appendable ndarray -> TrackedArray backed by existing dataset
        appendable = _getAppendable(fieldType)
        if appendable is not None and name in self._root:
            dataset = self._root[name]
            if isinstance(dataset, h5py.Dataset):
                # Re-resolve axis from dataset's maxshape (unlimited dim)
                axis = self._resolveAxisFromDataset(dataset, appendable)
                return TrackedArray(dataset, name, axis)

        # list -> TrackedList
        if isinstance(value, list):
            return TrackedList(value, self, name, fieldType)

        # dict -> TrackedDict
        if isinstance(value, dict):
            return TrackedDict(value, self, name, fieldType)

        return value

    @staticmethod
    def _resolveAxisFromDataset(dataset: h5py.Dataset, appendable: Appendable) -> int:
        """Re-resolve the append axis from an existing dataset's maxshape."""
        if appendable.axis is not None:
            return appendable.axis
        # Find the unlimited dimension
        maxshape = dataset.maxshape
        if maxshape is not None:
            for i, m in enumerate(maxshape):
                if m is None:
                    return i
        return 0

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

        # Unwrap TrackedList/TrackedDict to plain list/dict
        if isinstance(value, TrackedList):
            value = list(value)
        if isinstance(value, TrackedDict):
            value = dict(value)

        # Remove existing data for this field
        if name in self._root.attrs:
            del self._root.attrs[name]
        if name in self._root:
            del self._root[name]

        # Check if this is an Appendable field
        appendable = _getAppendable(fieldType)
        if appendable is not None and isinstance(value, np.ndarray):
            self._createResizableDataset(name, value, appendable)
            return

        # For scalar list fields, create resizable datasets so append works
        if isinstance(value, list):
            origin = typing.get_origin(fieldType)
            args = typing.get_args(fieldType)
            if origin is list and args and _isScalarType(args[0]):
                self._createResizableScalarList(name, value, args[0])
                return

        _writeValue(self._root, name, value, fieldType, datasetKwargs, self._comp)

    def _createResizableScalarList(self, name: str, values: list[Any], elemType: type) -> None:
        """Create a resizable 1-D dataset for a scalar list field."""
        datasetKwargs = self._comp.datasetKwargs()
        dtype = h5py.string_dtype() if elemType is str else _dtypeForElementType(elemType)

        if len(values) == 0:
            # Don't create a dataset for empty lists — _onListAppend will create it
            return

        self._root.create_dataset(
            name,
            data=values,
            dtype=dtype,
            maxshape=(None,),
            chunks=True,
            **datasetKwargs,
        )

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

        # list -> TrackedList
        if isinstance(value, list) and not isinstance(value, TrackedList):
            return TrackedList(value, self, name, fieldType)

        # dict -> TrackedDict
        if isinstance(value, dict) and not isinstance(value, TrackedDict):
            return TrackedDict(value, self, name, fieldType)

        return value

    # ------------------------------------------------------------------
    # List mutation callbacks
    # ------------------------------------------------------------------

    def _elemType(self, fieldName: str) -> Any:
        """Extract the element type from a list or dict field type."""
        args = typing.get_args(self._fieldTypes[fieldName])
        return args[0] if args else Any

    def _valType(self, fieldName: str) -> Any:
        """Extract the value type from a dict field type."""
        args = typing.get_args(self._fieldTypes[fieldName])
        return args[1] if len(args) > 1 else Any

    def _isScalarList(self, fieldName: str) -> bool:
        """Check if a list field stores scalars (1-D dataset) vs non-scalars (group)."""
        return _isScalarType(self._elemType(fieldName))

    def _onListAppend(self, fieldName: str, index: int, value: Any) -> None:
        elemType = self._elemType(fieldName)
        datasetKwargs = self._comp.datasetKwargs()

        if _isScalarType(elemType):
            if fieldName not in self._root:
                dtype = _dtypeForElementType(elemType)
                if elemType is str:
                    self._root.create_dataset(
                        fieldName,
                        shape=(1,),
                        maxshape=(None,),
                        dtype=h5py.string_dtype(),
                        data=[value],
                        **datasetKwargs,
                    )
                else:
                    self._root.create_dataset(
                        fieldName,
                        shape=(1,),
                        maxshape=(None,),
                        dtype=dtype,
                        data=[value],
                        **datasetKwargs,
                    )
            else:
                ds = self._root[fieldName]
                if isinstance(ds, h5py.Dataset):
                    ds.resize(ds.shape[0] + 1, axis=0)
                    ds[-1] = value
        else:
            group = self._root.require_group(fieldName)
            _writeValue(group, str(index), value, elemType, datasetKwargs, self._comp)

    def _onListSetItem(self, fieldName: str, index: int, value: Any) -> None:
        elemType = self._elemType(fieldName)
        datasetKwargs = self._comp.datasetKwargs()

        if _isScalarType(elemType):
            ds = self._root[fieldName]
            if isinstance(ds, h5py.Dataset):
                ds[index] = value
        else:
            item = self._root[fieldName]
            if isinstance(item, h5py.Group):
                key = str(index)
                if key in item:
                    del item[key]
                _writeValue(item, key, value, elemType, datasetKwargs, self._comp)

    def _onListExtend(self, fieldName: str, startIdx: int, values: list[Any]) -> None:
        for i, v in enumerate(values):
            self._onListAppend(fieldName, startIdx + i, v)

    # ------------------------------------------------------------------
    # Dict mutation callbacks
    # ------------------------------------------------------------------

    def _onDictSetItem(self, fieldName: str, key: Any, value: Any) -> None:
        valType = self._valType(fieldName)
        datasetKwargs = self._comp.datasetKwargs()
        group = self._root.require_group(fieldName)
        strKey = _keyToStr(key)
        if strKey in group:
            del group[strKey]
        if strKey in group.attrs:
            del group.attrs[strKey]
        _writeValue(group, strKey, value, valType, datasetKwargs, self._comp)

    def _onDictDelItem(self, fieldName: str, key: Any) -> None:
        item = self._root[fieldName]
        if not isinstance(item, h5py.Group):
            return
        strKey = _keyToStr(key)
        if strKey in item:
            del item[strKey]
        if strKey in item.attrs:
            del item.attrs[strKey]


# ---------------------------------------------------------------------------
# Tracked collections
# ---------------------------------------------------------------------------

_UNSUPPORTED_LIST_OPS = ("insert", "pop", "remove", "sort", "reverse", "__delitem__")


class TrackedList[T](list[T]):
    """List subclass that notifies the session on mutation.

    Supports ``append``, ``__setitem__``, and ``extend``.
    Reorder operations raise ``NotImplementedError``.
    """

    def __init__(
        self,
        values: Iterable[T],
        session: Hdf5Session[Any],
        fieldName: str,
        fieldType: Any,
    ) -> None:
        super().__init__(values)
        object.__setattr__(self, "_session", session)
        object.__setattr__(self, "_fieldName", fieldName)
        object.__setattr__(self, "_fieldType", fieldType)

    def append(self, value: T) -> None:
        super().append(value)
        session: Hdf5Session[Any] = object.__getattribute__(self, "_session")
        fieldName: str = object.__getattribute__(self, "_fieldName")
        session._onListAppend(fieldName, len(self) - 1, value)  # noqa: SLF001

    def __setitem__(self, index: Any, value: Any) -> None:
        super().__setitem__(index, value)
        session: Hdf5Session[Any] = object.__getattribute__(self, "_session")
        fieldName: str = object.__getattribute__(self, "_fieldName")
        session._onListSetItem(fieldName, index, value)  # noqa: SLF001

    def extend(self, values: Iterable[T]) -> None:
        startIdx = len(self)
        valuesList = list(values)
        super().extend(valuesList)
        session: Hdf5Session[Any] = object.__getattribute__(self, "_session")
        fieldName: str = object.__getattribute__(self, "_fieldName")
        session._onListExtend(fieldName, startIdx, valuesList)  # noqa: SLF001

    def _unsupported(self, op: str) -> None:
        raise NotImplementedError(
            f"TrackedList does not support '{op}'. Build the data in memory and assign the whole list instead."
        )

    def insert(self, index: SupportsIndex, value: T) -> None:
        self._unsupported("insert")

    def pop(self, index: SupportsIndex = -1) -> T:
        self._unsupported("pop")
        raise AssertionError("unreachable")

    def remove(self, value: T) -> None:
        self._unsupported("remove")

    def sort(self, *, key: Any = None, reverse: bool = False) -> None:
        self._unsupported("sort")

    def reverse(self) -> None:
        self._unsupported("reverse")

    def __delitem__(self, index: Any) -> None:
        self._unsupported("__delitem__")


class TrackedDict[K, V](dict[K, V]):
    """Dict subclass that notifies the session on mutation."""

    def __init__(
        self,
        values: dict[K, V],
        session: Hdf5Session[Any],
        fieldName: str,
        fieldType: Any,
    ) -> None:
        super().__init__(values)
        object.__setattr__(self, "_session", session)
        object.__setattr__(self, "_fieldName", fieldName)
        object.__setattr__(self, "_fieldType", fieldType)

    def __setitem__(self, key: K, value: V) -> None:
        super().__setitem__(key, value)
        session: Hdf5Session[Any] = object.__getattribute__(self, "_session")
        fieldName: str = object.__getattribute__(self, "_fieldName")
        session._onDictSetItem(fieldName, key, value)  # noqa: SLF001

    def __delitem__(self, key: K) -> None:
        super().__delitem__(key)
        session: Hdf5Session[Any] = object.__getattribute__(self, "_session")
        fieldName: str = object.__getattribute__(self, "_fieldName")
        session._onDictDelItem(fieldName, key)  # noqa: SLF001

    def update(self, other: Any = None, **kwargs: V) -> None:
        items: dict[Any, Any] = {}
        if other is not None:
            items.update(other)
        items.update(kwargs)
        super().update(items)
        session: Hdf5Session[Any] = object.__getattribute__(self, "_session")
        fieldName: str = object.__getattribute__(self, "_fieldName")
        for k, v in items.items():
            session._onDictSetItem(fieldName, k, v)  # noqa: SLF001


# ---------------------------------------------------------------------------
# Proxy class factory
# ---------------------------------------------------------------------------


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
