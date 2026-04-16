"""Hdf5FieldInfo — optional layout hints for ndarray fields in HDF5 sessions.

All ndarray fields in HDF5 sessions are automatically backed by resizable
datasets. Use ``Hdf5FieldInfo`` only when the defaults aren't right.

Usage::

    from typing import Annotated
    from versionable import Hdf5FieldInfo

    @dataclass
    class Experiment(Versionable, version=1, hash="..."):
        highRes: Annotated[np.ndarray, Hdf5FieldInfo(chunkRows=128)]
        channels: Annotated[np.ndarray, Hdf5FieldInfo(axis=1)]
"""

from __future__ import annotations

import typing
from dataclasses import dataclass
from typing import Annotated, Any

from versionable.errors import BackendError

TARGET_CHUNK_BYTES = 256 * 1024  # 256 KB


@dataclass(frozen=True)
class Hdf5FieldInfo:
    """Optional layout hints for ndarray fields in HDF5 sessions.

    All ndarray fields are resizable by default. Use this annotation
    only when the defaults aren't right.

    Args:
        chunkRows: Chunk size along the append axis (default: ~256 KB heuristic).
        axis: Which axis grows on append (default: inferred from zero-size dim, or 0).
    """

    chunkRows: int | None = None
    axis: int | None = None


def _getHdf5FieldInfo(fieldType: Any) -> Hdf5FieldInfo | None:
    """Extract Hdf5FieldInfo metadata from an Annotated type, if present."""
    if typing.get_origin(fieldType) is Annotated:
        for arg in typing.get_args(fieldType)[1:]:
            if isinstance(arg, Hdf5FieldInfo):
                return arg
    return None


def _resolveAppendAxis(shape: tuple[int, ...], hdf5Field: Hdf5FieldInfo) -> int:
    """Determine the append axis from the Hdf5FieldInfo config and data shape.

    Resolution order:
    1. Explicit ``Hdf5FieldInfo(axis=N)`` — always used.
    2. Exactly one axis has size 0 — inferred from shape.
    3. No axis has size 0 — defaults to axis 0.
    4. Multiple axes have size 0 — raises ``BackendError``.
    """
    if hdf5Field.axis is not None:
        axis = hdf5Field.axis
        ndim = len(shape)
        if ndim > 0 and not (-ndim <= axis < ndim):
            raise BackendError(
                f"Invalid append axis {axis} for shape {shape}. Axis must be in range [{-ndim}, {ndim - 1}]."
            )
        return axis % ndim if ndim > 0 else axis

    zeroAxes = [i for i, s in enumerate(shape) if s == 0]
    if len(zeroAxes) == 1:
        return zeroAxes[0]
    if len(zeroAxes) > 1:
        raise BackendError(
            f"Ambiguous append axis: shape {shape} has multiple zero-size axes "
            f"{zeroAxes}. Specify Hdf5FieldInfo(axis=...) explicitly."
        )

    return 0


def _computeChunkSize(shape: tuple[int, ...], dtype: Any, axis: int) -> int:
    """Compute elements per chunk along the append axis targeting TARGET_CHUNK_BYTES."""
    import numpy as np  # deferred to keep Hdf5FieldInfo importable without numpy

    nonAppendShape = shape[:axis] + shape[axis + 1 :]
    sliceBytes = int(np.prod(nonAppendShape)) * dtype.itemsize if nonAppendShape else dtype.itemsize
    if sliceBytes == 0:
        return 1
    return int(max(1, TARGET_CHUNK_BYTES // sliceBytes))
