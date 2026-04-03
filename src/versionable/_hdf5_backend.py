"""HDF5 storage backend.

Stores Versionable objects in HDF5 files with:
- Metadata as root-level attributes
- Scalar/collection fields as JSON-encoded attributes on a ``fields`` group
- Array fields as HDF5 datasets with dtype/shape preserved
- Nested Versionable objects as subgroups (recursive)

Supports lazy loading of array fields via ``LazyArray`` sentinels.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, ClassVar

import h5py
import numpy as np

from versionable._backend import Backend, registerBackend
from versionable._hdf5_compression import ZSTD_DEFAULT, Hdf5Compression
from versionable.errors import BackendError

logger = logging.getLogger(__name__)


class Hdf5Backend(Backend):
    """HDF5 file backend for large data with lazy array loading."""

    nativeTypes: ClassVar[set[type]] = {np.ndarray}

    def save(
        self,
        fields: dict[str, Any],
        meta: dict[str, Any],
        path: Path,
        *,
        compression: Hdf5Compression | None = None,
        **kwargs: Any,
    ) -> None:
        comp = compression or ZSTD_DEFAULT
        try:
            with h5py.File(path, "w") as f:
                # Write metadata as root attributes
                f.attrs["__OBJECT__"] = meta["name"]
                f.attrs["__VERSION__"] = meta["version"]
                f.attrs["__HASH__"] = meta["hash"]

                # Write fields
                _writeGroup(f, fields, comp)
        except OSError as e:
            raise BackendError(f"Failed to write HDF5 to {path}: {e}") from e

    def load(self, path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
        try:
            with h5py.File(path, "r") as f:
                meta = {
                    "__OBJECT__": str(f.attrs.get("__OBJECT__", "")),
                    "__VERSION__": int(f.attrs.get("__VERSION__", 1)),
                    "__HASH__": str(f.attrs.get("__HASH__", "")),
                }
                fields = _readGroup(f)
            return fields, meta
        except OSError as e:
            raise BackendError(f"Failed to read HDF5 from {path}: {e}") from e

    def loadLazy(
        self,
        path: Path,
        *,
        preload: list[str] | str | None = None,
        metadataOnly: bool = False,
    ) -> tuple[dict[str, Any], dict[str, Any], set[str]]:
        """Load with lazy array support.

        Returns:
            Tuple of (fields_dict, metadata_dict, lazy_field_names).
            Array fields not in *preload* are returned as ``LazyArray``
            sentinels.
        """
        from versionable._lazy import ArrayNotLoaded, LazyArray

        preloadAll = preload == "*"
        preloadSet = set(preload) if isinstance(preload, list) else set()

        try:
            with h5py.File(path, "r") as f:
                meta = {
                    "__OBJECT__": str(f.attrs.get("__OBJECT__", "")),
                    "__VERSION__": int(f.attrs.get("__VERSION__", 1)),
                    "__HASH__": str(f.attrs.get("__HASH__", "")),
                }

                fields: dict[str, Any] = {}
                lazyFields: set[str] = set()

                for key in f:
                    item = f[key]
                    if isinstance(item, h5py.Dataset):
                        if metadataOnly:
                            fields[key] = ArrayNotLoaded(key)
                            lazyFields.add(key)
                        elif preloadAll or key in preloadSet:
                            fields[key] = item[()]
                        else:
                            fields[key] = LazyArray(path, key)
                            lazyFields.add(key)
                    elif isinstance(item, h5py.Group):
                        # Check if it's a nested Versionable (has __OBJECT__ attr)
                        if "__OBJECT__" in item.attrs:
                            fields[key] = _readNestedGroup(item)
                        else:
                            # Regular group: read as dict
                            fields[key] = _readGroup(item)

                # Read JSON-encoded scalar attributes
                if "__scalars__" in f.attrs:
                    scalars = json.loads(f.attrs["__scalars__"])
                    fields.update(scalars)

            return fields, meta, lazyFields
        except OSError as e:
            raise BackendError(f"Failed to read HDF5 from {path}: {e}") from e


# ---------------------------------------------------------------------------
# Write helpers
# ---------------------------------------------------------------------------


def _writeGroup(group: h5py.Group, fields: dict[str, Any], comp: Hdf5Compression) -> None:
    """Write fields into an HDF5 group.

    Arrays → datasets, nested dicts with __OBJECT__ → subgroups,
    everything else → JSON-encoded in the __scalars__ attribute.
    """
    scalars: dict[str, Any] = {}
    datasetKwargs = comp.datasetKwargs()

    for key, value in fields.items():
        if isinstance(value, np.ndarray):
            group.create_dataset(key, data=value, **datasetKwargs)
        elif isinstance(value, dict) and "__OBJECT__" in value:
            _writeNestedGroup(group, key, value, comp)
        else:
            scalars[key] = value

    if scalars:
        group.attrs["__scalars__"] = json.dumps(scalars, default=str)


def _writeNestedGroup(parent: h5py.Group, name: str, data: dict[str, Any], comp: Hdf5Compression) -> None:
    """Write a nested Versionable as a subgroup."""
    subgroup = parent.create_group(name)
    subgroup.attrs["__OBJECT__"] = data.get("__OBJECT__", "")
    subgroup.attrs["__VERSION__"] = data.get("__VERSION__", 1)
    subgroup.attrs["__HASH__"] = data.get("__HASH__", "")

    # Write the remaining fields (exclude metadata keys)
    nested = {k: v for k, v in data.items() if not k.startswith("__")}
    _writeGroup(subgroup, nested, comp)


# ---------------------------------------------------------------------------
# Read helpers
# ---------------------------------------------------------------------------


def _readGroup(group: h5py.Group) -> dict[str, Any]:
    """Read all fields from an HDF5 group (eager)."""
    fields: dict[str, Any] = {}

    for key in group:
        item = group[key]
        if isinstance(item, h5py.Dataset):
            fields[key] = item[()]
        elif isinstance(item, h5py.Group):
            if "__OBJECT__" in item.attrs:
                fields[key] = _readNestedGroup(item)
            else:
                fields[key] = _readGroup(item)

    # Read scalar attributes
    if "__scalars__" in group.attrs:
        scalars = json.loads(group.attrs["__scalars__"])
        fields.update(scalars)

    return fields


def _readNestedGroup(group: h5py.Group) -> dict[str, Any]:
    """Read a nested Versionable subgroup as a dict with metadata."""
    result: dict[str, Any] = {
        "__OBJECT__": str(group.attrs.get("__OBJECT__", "")),
        "__VERSION__": int(group.attrs.get("__VERSION__", 1)),
        "__HASH__": str(group.attrs.get("__HASH__", "")),
    }
    result.update(_readGroup(group))
    return result


# Register for .h5 and .hdf5 extensions
registerBackend([".h5", ".hdf5"], Hdf5Backend)
