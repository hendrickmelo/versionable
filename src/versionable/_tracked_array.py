"""Backwards-compatibility alias — TrackedArray is now DatasetArray."""

from __future__ import annotations

from versionable._dataset_array import DatasetArray

TrackedArray = DatasetArray

__all__ = ["TrackedArray"]
