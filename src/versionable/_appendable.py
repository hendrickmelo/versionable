"""Backwards-compatibility alias — Appendable is now Hdf5FieldInfo."""

from __future__ import annotations

from versionable._hdf5_field import Appendable as Appendable
from versionable._hdf5_field import Hdf5FieldInfo as Hdf5FieldInfo
from versionable._hdf5_field import _computeChunkSize as _computeChunkSize
from versionable._hdf5_field import _getHdf5FieldInfo as _getHdf5FieldInfo
from versionable._hdf5_field import _resolveAppendAxis as _resolveAppendAxis

# Keep old name available for any external consumers
_getAppendable = _getHdf5FieldInfo

__all__ = [
    "Appendable",
    "Hdf5FieldInfo",
    "_computeChunkSize",
    "_getAppendable",
    "_getHdf5FieldInfo",
    "_resolveAppendAxis",
]
