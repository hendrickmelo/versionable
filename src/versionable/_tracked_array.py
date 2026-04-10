"""TrackedArray — wrapper for appendable HDF5 dataset fields.

Inside an HDF5 session, ``Appendable`` fields are wrapped with
``TrackedArray``. This is a wrapper class (not an ndarray subclass)
that holds a reference to the live HDF5 dataset and supports append
and element assignment.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    import h5py


class TrackedArray:
    """Wrapper around a resizable HDF5 dataset.

    Supports ``.append()`` for growing the dataset along the declared axis,
    ``__setitem__`` for direct element writes, and ``__array__()`` for
    numpy interop.

    Not thread-safe — HDF5 does not support concurrent writes by default.
    """

    __slots__ = ("_axis", "_dataset", "_fieldName")

    def __init__(self, dataset: h5py.Dataset, fieldName: str, axis: int) -> None:
        self._dataset = dataset
        self._fieldName = fieldName
        self._axis = axis

    def append(self, data: np.ndarray) -> None:
        """Append data along the append axis, resizing the dataset."""
        self._validateShape(data)
        axis = self._axis
        currentSize = self._dataset.shape[axis]
        newSlices = data.shape[axis] if data.ndim > axis else 1
        self._dataset.resize(currentSize + newSlices, axis=axis)
        idx = [slice(None)] * self._dataset.ndim
        idx[axis] = slice(currentSize, None)
        self._dataset[tuple(idx)] = data

    def __setitem__(self, index: Any, value: Any) -> None:
        """Write directly to the HDF5 dataset."""
        self._dataset[index] = value

    def __getitem__(self, index: Any) -> np.ndarray:
        """Read from the HDF5 dataset."""
        result: np.ndarray = self._dataset[index]
        return result

    def __array__(self, dtype: np.typing.DTypeLike | None = None, copy: bool | None = None) -> np.ndarray:
        """Return full dataset as ndarray (for numpy interop)."""
        arr: np.ndarray = self._dataset[()]
        if dtype is not None:
            arr = np.asarray(arr, dtype=dtype)
        return arr

    def __len__(self) -> int:
        return int(self._dataset.shape[0])

    @property
    def shape(self) -> tuple[int, ...]:
        result: tuple[int, ...] = self._dataset.shape
        return result

    @property
    def dtype(self) -> np.dtype[Any]:
        result: np.dtype[Any] = self._dataset.dtype
        return result

    @property
    def axis(self) -> int:
        """The append axis for this tracked array."""
        return self._axis

    @property
    def ndim(self) -> int:
        return int(self._dataset.ndim)

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
