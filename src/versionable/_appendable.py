"""Appendable annotation for growable ndarray fields.

Marks an ``np.ndarray`` field as appendable inside HDF5 sessions.
Outside sessions, this annotation is ignored — the field is a normal
``np.ndarray``.

Usage::

    from typing import Annotated
    from versionable import Appendable

    @dataclass
    class Experiment(Versionable, version=1, hash="..."):
        waveform: Annotated[np.ndarray, Appendable()]
        highRes: Annotated[np.ndarray, Appendable(chunkRows=128)]
        channels: Annotated[np.ndarray, Appendable(axis=1)]
"""

from __future__ import annotations

import typing
from dataclasses import dataclass
from typing import Annotated, Any

import numpy as np

from versionable.errors import BackendError

TARGET_CHUNK_BYTES = 256 * 1024  # 256 KB


@dataclass(frozen=True)
class Appendable:
    """Marks an np.ndarray field as appendable.

    When used inside an HDF5 session, the field is backed by a resizable
    dataset and wrapped with a TrackedArray that supports ``.append()``.
    Outside sessions, this annotation is ignored — the field is a normal
    ``np.ndarray``.

    Args:
        chunkRows: Number of elements per chunk along the append axis.
            If None, a heuristic targets ~256 KB per chunk based on
            dtype and the shape of the non-append dimensions.
        axis: The axis along which the dataset grows. If None, inferred
            from the first assigned array (the axis with size 0). If no
            axis has size 0, defaults to 0.
    """

    chunkRows: int | None = None
    axis: int | None = None


def _getAppendable(fieldType: Any) -> Appendable | None:
    """Extract Appendable metadata from an Annotated type, if present."""
    if typing.get_origin(fieldType) is Annotated:
        for arg in typing.get_args(fieldType)[1:]:
            if isinstance(arg, Appendable):
                return arg
    return None


def _resolveAppendAxis(shape: tuple[int, ...], appendable: Appendable) -> int:
    """Determine the append axis from the Appendable config and data shape.

    Resolution order:
    1. Explicit ``Appendable(axis=N)`` — always used.
    2. Exactly one axis has size 0 — inferred from shape.
    3. No axis has size 0 — defaults to axis 0.
    4. Multiple axes have size 0 — raises ``BackendError``.
    """
    if appendable.axis is not None:
        return appendable.axis

    zeroAxes = [i for i, s in enumerate(shape) if s == 0]
    if len(zeroAxes) == 1:
        return zeroAxes[0]
    if len(zeroAxes) > 1:
        raise BackendError(
            f"Ambiguous append axis: shape {shape} has multiple zero-size axes "
            f"{zeroAxes}. Specify Appendable(axis=...) explicitly."
        )

    return 0


def _computeChunkSize(shape: tuple[int, ...], dtype: np.dtype[Any], axis: int) -> int:
    """Compute elements per chunk along the append axis targeting TARGET_CHUNK_BYTES."""
    nonAppendShape = shape[:axis] + shape[axis + 1 :]
    sliceBytes = int(np.prod(nonAppendShape)) * dtype.itemsize if nonAppendShape else dtype.itemsize
    if sliceBytes == 0:
        return 1
    return max(1, TARGET_CHUNK_BYTES // sliceBytes)
