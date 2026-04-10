"""DatasetArray — wrapper for HDF5 dataset-backed ndarray fields.

Inside an HDF5 session, ndarray fields are wrapped with ``DatasetArray``.
This is a wrapper class (not an ndarray subclass) that holds a reference
to the live HDF5 dataset and supports append, element assignment,
resize, and numpy interop.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import TYPE_CHECKING, Any

import numpy as np

from versionable.errors import BackendError

if TYPE_CHECKING:
    import h5py

_CLOSED_MSG = "DatasetArray is no longer accessible — the session has been closed."
_READONLY_MSG = "DatasetArray is read-only (session opened in read mode)."


class DatasetArray:
    """Wrapper around a resizable HDF5 dataset.

    Supports ``.append()`` for growing the dataset along the declared axis,
    ``__setitem__`` for direct element writes, ``resize()`` for
    pre-allocation/truncation, and ``__array__()`` for numpy interop.

    Not thread-safe — HDF5 does not support concurrent writes by default.
    """

    __slots__ = ("_axis", "_closed", "_dataset", "_fieldName", "_writable")

    def __init__(
        self,
        dataset: h5py.Dataset,
        fieldName: str,
        axis: int,
        *,
        writable: bool = True,
    ) -> None:
        self._dataset = dataset
        self._fieldName = fieldName
        self._axis = axis
        self._writable = writable
        self._closed = False

    def _checkOpen(self) -> None:
        if self._closed:
            raise BackendError(_CLOSED_MSG)

    def _checkWritable(self) -> None:
        self._checkOpen()
        if not self._writable:
            raise BackendError(_READONLY_MSG)

    def append(self, data: np.ndarray) -> None:
        """Append data along the append axis, resizing the dataset."""
        self._checkWritable()
        self._validateShape(data)
        axis = self._axis
        currentSize = self._dataset.shape[axis]
        newSlices = data.shape[axis] if data.ndim > axis else 1
        self._dataset.resize(currentSize + newSlices, axis=axis)
        idx = [slice(None)] * self._dataset.ndim
        idx[axis] = slice(currentSize, None)
        self._dataset[tuple(idx)] = data

    def resize(self, size: int, axis: int | None = None) -> None:
        """Resize the dataset along the given axis (defaults to append axis)."""
        self._checkWritable()
        self._dataset.resize(size, axis=axis if axis is not None else self._axis)

    def __setitem__(self, index: Any, value: Any) -> None:
        """Write directly to the HDF5 dataset."""
        self._checkWritable()
        self._dataset[index] = value

    def __getitem__(self, index: Any) -> np.ndarray:
        """Read from the HDF5 dataset."""
        self._checkOpen()
        result: np.ndarray = self._dataset[index]
        return result

    def __array__(self, dtype: np.typing.DTypeLike | None = None, copy: bool | None = None) -> np.ndarray:
        """Return full dataset as ndarray (for numpy interop)."""
        self._checkOpen()
        arr: np.ndarray = self._dataset[()]
        if dtype is not None:
            arr = np.asarray(arr, dtype=dtype)
        return arr

    def __len__(self) -> int:
        self._checkOpen()
        return int(self._dataset.shape[self._axis])

    def __iter__(self) -> Iterator[np.ndarray]:
        """Iterate over the first axis (rows)."""
        self._checkOpen()
        for i in range(self._dataset.shape[0]):
            yield self._dataset[i]

    @property
    def shape(self) -> tuple[int, ...]:
        self._checkOpen()
        result: tuple[int, ...] = self._dataset.shape
        return result

    @property
    def dtype(self) -> np.dtype[Any]:
        self._checkOpen()
        result: np.dtype[Any] = self._dataset.dtype
        return result

    @property
    def axis(self) -> int:
        """The append axis for this array."""
        return self._axis

    @property
    def ndim(self) -> int:
        self._checkOpen()
        return int(self._dataset.ndim)

    @property
    def chunks(self) -> tuple[int, ...] | None:
        """Chunk shape of the underlying HDF5 dataset."""
        self._checkOpen()
        result: tuple[int, ...] | None = self._dataset.chunks
        return result

    @property
    def maxshape(self) -> tuple[int | None, ...] | None:
        """Maximum shape of the underlying HDF5 dataset."""
        self._checkOpen()
        result: tuple[int | None, ...] | None = self._dataset.maxshape
        return result

    def _validateShape(self, data: np.ndarray) -> None:
        """Raise ValueError if data shape is incompatible with the dataset."""
        expected = list(self._dataset.shape)
        del expected[self._axis]
        got = list(data.shape)
        if data.ndim > self._axis:
            del got[self._axis]
        if expected != got:
            raise ValueError(
                f"Cannot append to '{self._fieldName}': non-append dimensions must match. "
                f"Dataset shape is {self._dataset.shape} (axis={self._axis}), "
                f"but data shape is {data.shape}"
            )
