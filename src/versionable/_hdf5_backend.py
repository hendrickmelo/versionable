"""HDF5 storage backend with native type mapping.

Every field maps to a native HDF5 construct — no JSON anywhere:

- Scalar primitives (int, float, bool, str) → HDF5 attributes
- ``np.ndarray`` → HDF5 datasets (with compression)
- ``list[numeric]`` / ``list[str]`` / ``list[bool]`` → 1-D datasets
- ``list[np.ndarray]`` → group of integer-keyed datasets
- ``dict[str, np.ndarray]`` → group of named datasets
- Nested ``Versionable`` → subgroup with ``__versionable__`` metadata group
- ``list[Versionable]`` → group of integer-keyed subgroups
- ``None`` → ``h5py.Empty("f")`` attribute
- Enum → attribute (stores ``.value``)
- Converted types (datetime, Path, etc.) → attribute (converter output)

Metadata (``__OBJECT__``, ``__VERSION__``, ``__HASH__``) is stored in a
``__versionable__`` child group, distinguishing Versionable groups from
plain collection groups.

Supports lazy loading of array fields via ``LazyArray`` sentinels.
"""

from __future__ import annotations

import logging
import typing
from enum import Enum
from pathlib import Path
from typing import Any, ClassVar

import h5py
import numpy as np

from versionable._backend import Backend, registerBackend
from versionable._base import Versionable, _resolveFields, metadata
from versionable._hdf5_compression import ZSTD_DEFAULT, Hdf5Compression
from versionable._types import _registry
from versionable.errors import BackendError

logger = logging.getLogger(__name__)

_VERSIONABLE_GROUP = "__versionable__"


class Hdf5Backend(Backend):
    """HDF5 file backend for large data with lazy array loading."""

    nativeTypes: ClassVar[set[type]] = {np.ndarray}

    def save(
        self,
        fields: dict[str, Any],
        meta: dict[str, Any],
        path: Path,
        *,
        cls: type,
        compression: Hdf5Compression | None = None,
        **kwargs: Any,
    ) -> None:
        comp = compression or ZSTD_DEFAULT
        fieldTypes = _resolveFields(cls)
        try:
            with h5py.File(path, "w") as f:
                # Write metadata into __versionable__ child group
                metaGroup = f.create_group(_VERSIONABLE_GROUP)
                metaGroup.attrs["__OBJECT__"] = meta["name"]
                metaGroup.attrs["__VERSION__"] = meta["version"]
                metaGroup.attrs["__HASH__"] = meta["hash"]

                # Write fields
                _writeFields(f, fields, fieldTypes, comp)
        except OSError as e:
            raise BackendError(f"Failed to write HDF5 to {path}: {e}") from e

    def load(self, path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
        try:
            with h5py.File(path, "r") as f:
                meta = _readMeta(f)
                objectName = meta["__OBJECT__"]
                cls = _resolveClass(objectName)
                fieldTypes = _resolveFields(cls) if cls is not None else {}
                fields = _readFields(f, fieldTypes)
            return fields, meta
        except OSError as e:
            raise BackendError(f"Failed to read HDF5 from {path}: {e}") from e

    def loadLazy(
        self,
        path: Path,
        *,
        cls: type | None = None,
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
                meta = _readMeta(f)
                if cls is None:
                    cls = _resolveClass(meta["__OBJECT__"])
                fieldTypes = _resolveFields(cls) if cls is not None else {}

                fields: dict[str, Any] = {}
                lazyFields: set[str] = set()

                for key in f:
                    if key == _VERSIONABLE_GROUP:
                        continue

                    item = f[key]
                    fieldType = fieldTypes.get(key)

                    if isinstance(item, h5py.Dataset):
                        if _isArrayField(fieldType):
                            # np.ndarray field — lazy loading applies
                            if metadataOnly:
                                fields[key] = ArrayNotLoaded(key)
                                lazyFields.add(key)
                            elif preloadAll or key in preloadSet:
                                fields[key] = item[()]
                            else:
                                fields[key] = LazyArray(path, key)
                                lazyFields.add(key)
                        else:
                            # list[numeric], list[str], etc. — always eager
                            fields[key] = _readDataset(item, fieldType)
                    elif isinstance(item, h5py.Group):
                        fields[key] = _readGroup(item, fieldType)

                # Read scalar attributes
                for attrName in f.attrs:
                    if attrName == _VERSIONABLE_GROUP:
                        continue
                    fields[attrName] = _readAttr(f.attrs[attrName])

            return fields, meta, lazyFields
        except OSError as e:
            raise BackendError(f"Failed to read HDF5 from {path}: {e}") from e


# ---------------------------------------------------------------------------
# Write helpers
# ---------------------------------------------------------------------------


def _writeFields(
    group: h5py.Group,
    fields: dict[str, Any],
    fieldTypes: dict[str, Any],
    comp: Hdf5Compression,
) -> None:
    """Write fields into an HDF5 group using native type dispatch."""
    datasetKwargs = comp.datasetKwargs()

    for name, value in fields.items():
        fieldType = fieldTypes.get(name)
        _writeField(group, name, value, fieldType, datasetKwargs, comp)


def _writeField(  # noqa: PLR0911 — type dispatch; many returns are inherent
    group: h5py.Group,
    name: str,
    value: Any,
    fieldType: Any,
    datasetKwargs: dict[str, Any],
    comp: Hdf5Compression,
) -> None:
    """Write a single field to the group."""
    # None → h5py.Empty
    if value is None:
        group.attrs[name] = h5py.Empty("f")
        return

    # np.ndarray → dataset with compression
    if isinstance(value, np.ndarray):
        group.create_dataset(name, data=value, **datasetKwargs)
        return

    # Nested Versionable → subgroup with __versionable__
    if isinstance(value, Versionable):
        _writeVersionable(group, name, value, comp)
        return

    # Enum → store .value as attribute
    if isinstance(value, Enum):
        group.attrs[name] = value.value
        return

    # Scalar primitives → attribute
    if isinstance(value, (int, float, str, bool)):
        group.attrs[name] = value
        return

    # Converted types (datetime, Path, etc.) → apply converter, write attribute
    conv = _registry.get(type(value))
    if conv is not None:
        group.attrs[name] = conv.serialize(value)
        return

    # Collections — need to inspect fieldType
    origin = typing.get_origin(fieldType)
    args = typing.get_args(fieldType)

    if isinstance(value, list) and origin is list and args:
        elemType = args[0]
        _writeList(group, name, value, elemType, datasetKwargs, comp)
        return

    if isinstance(value, dict) and origin is dict and args:
        _valType = args[1]
        _writeDict(group, name, value, _valType, datasetKwargs, comp)
        return

    # Fallback: try as attribute (will raise if h5py can't store it)
    try:
        group.attrs[name] = value
    except TypeError as e:
        raise BackendError(
            f"Cannot store field '{name}' of type {type(value).__name__} in HDF5. "
            f"Consider restructuring your data or using a dict-based backend (JSON/YAML)."
        ) from e


def _writeList(
    group: h5py.Group,
    name: str,
    values: list[Any],
    elemType: Any,
    datasetKwargs: dict[str, Any],
    comp: Hdf5Compression,
) -> None:
    """Write a list field to the group."""
    # list[np.ndarray] → group of integer-keyed datasets
    if elemType is np.ndarray or (typing.get_origin(elemType) is not None and _isNdarrayType(elemType)):
        subgroup = group.create_group(name)
        for i, arr in enumerate(values):
            subgroup.create_dataset(str(i), data=arr, **datasetKwargs)
        return

    # list[Versionable] → group of integer-keyed subgroups
    concreteElem = typing.get_origin(elemType) or elemType
    if isinstance(concreteElem, type) and issubclass(concreteElem, Versionable):
        subgroup = group.create_group(name)
        for i, item in enumerate(values):
            _writeVersionable(subgroup, str(i), item, comp)
        return

    # list[numeric] / list[str] / list[bool] → 1-D dataset
    if len(values) == 0:
        dtype = _dtypeForElementType(elemType)
        group.create_dataset(name, shape=(0,), dtype=dtype)
    elif isinstance(values[0], str):
        dt = h5py.string_dtype()
        group.create_dataset(name, data=values, dtype=dt)
    else:
        group.create_dataset(name, data=values)


def _writeDict(
    group: h5py.Group,
    name: str,
    values: dict[str, Any],
    valType: Any,
    datasetKwargs: dict[str, Any],
    comp: Hdf5Compression,
) -> None:
    """Write a dict field to the group."""
    # dict[str, np.ndarray] → group of named datasets
    if valType is np.ndarray or _isNdarrayType(valType):
        subgroup = group.create_group(name)
        for key, arr in values.items():
            subgroup.create_dataset(key, data=arr, **datasetKwargs)
        return

    # dict[str, Versionable] → group of named subgroups
    concreteVal = typing.get_origin(valType) or valType
    if isinstance(concreteVal, type) and issubclass(concreteVal, Versionable):
        subgroup = group.create_group(name)
        for key, item in values.items():
            _writeVersionable(subgroup, key, item, comp)
        return

    raise BackendError(
        f"Cannot store field '{name}' of type dict[str, {valType}] in HDF5. "
        f"Consider restructuring your data or using a dict-based backend (JSON/YAML)."
    )


def _writeVersionable(
    parent: h5py.Group,
    name: str,
    obj: Versionable,
    comp: Hdf5Compression,
) -> None:
    """Write a nested Versionable as a subgroup with __versionable__ metadata."""
    subgroup = parent.create_group(name)
    objType = type(obj)
    meta = metadata(objType)
    fieldTypes = _resolveFields(objType)

    # Write metadata into __versionable__ child group
    metaGroup = subgroup.create_group(_VERSIONABLE_GROUP)
    metaGroup.attrs["__OBJECT__"] = meta.name
    metaGroup.attrs["__VERSION__"] = meta.version
    metaGroup.attrs["__HASH__"] = meta.hash

    # Recursively write fields
    fields = {fieldName: getattr(obj, fieldName) for fieldName in fieldTypes}
    _writeFields(subgroup, fields, fieldTypes, comp)


# ---------------------------------------------------------------------------
# Read helpers
# ---------------------------------------------------------------------------


def _readMeta(group: h5py.Group) -> dict[str, Any]:
    """Read metadata from the __versionable__ child group."""
    if _VERSIONABLE_GROUP in group:
        metaGroup = group[_VERSIONABLE_GROUP]
        return {
            "__OBJECT__": str(metaGroup.attrs.get("__OBJECT__", "")),
            "__VERSION__": int(metaGroup.attrs.get("__VERSION__", 1)),
            "__HASH__": str(metaGroup.attrs.get("__HASH__", "")),
        }
    # Should not happen for well-formed files
    return {"__OBJECT__": "", "__VERSION__": 1, "__HASH__": ""}


def _readFields(group: h5py.Group, fieldTypes: dict[str, Any]) -> dict[str, Any]:
    """Read all fields from an HDF5 group (eager)."""
    fields: dict[str, Any] = {}

    # Read child items (datasets and groups)
    for key in group:
        if key == _VERSIONABLE_GROUP:
            continue
        item = group[key]
        fieldType = fieldTypes.get(key)

        if isinstance(item, h5py.Dataset):
            fields[key] = _readDataset(item, fieldType)
        elif isinstance(item, h5py.Group):
            fields[key] = _readGroup(item, fieldType)

    # Read scalar attributes
    for attrName in group.attrs:
        fields[attrName] = _readAttr(group.attrs[attrName])

    return fields


def _readDataset(dataset: h5py.Dataset, fieldType: Any) -> Any:
    """Read a dataset, returning ndarray or list based on field type."""
    data = dataset[()]

    # If the field type is a list, convert to Python list
    if _isListField(fieldType):
        result = data.tolist() if isinstance(data, np.ndarray) else list(data)
        # h5py returns bytes for string datasets — decode to str
        if result and isinstance(result[0], bytes):
            return [v.decode("utf-8") for v in result]
        return result

    return data


def _readGroup(group: h5py.Group, fieldType: Any) -> Any:
    """Read a group, dispatching based on whether it's a Versionable or collection."""
    # Versionable group: has __versionable__ child
    if _VERSIONABLE_GROUP in group:
        return _readVersionableGroup(group)

    # Collection group: list or dict, determined by field type
    origin = typing.get_origin(fieldType)
    args = typing.get_args(fieldType)

    if origin is list and args:
        elemType = args[0]
        return _readListGroup(group, elemType)

    if origin is dict and args:
        valType = args[1]
        return _readDictGroup(group, valType)

    # Fallback: try as dict of arrays
    return _readDictGroup(group, None)


def _readVersionableGroup(group: h5py.Group) -> dict[str, Any]:
    """Read a nested Versionable subgroup as a dict with metadata."""
    metaGroup = group[_VERSIONABLE_GROUP]
    objectName = str(metaGroup.attrs.get("__OBJECT__", ""))

    result: dict[str, Any] = {
        "__OBJECT__": objectName,
        "__VERSION__": int(metaGroup.attrs.get("__VERSION__", 1)),
        "__HASH__": str(metaGroup.attrs.get("__HASH__", "")),
    }

    # Resolve field types from the class for nested reading
    cls = _resolveClass(objectName)
    nestedFieldTypes = _resolveFields(cls) if cls is not None else {}
    result.update(_readFields(group, nestedFieldTypes))
    return result


def _readListGroup(group: h5py.Group, elemType: Any) -> list[Any]:
    """Read a list from a group with integer-keyed children."""
    keys = sorted(group.keys(), key=int)

    concreteElem = typing.get_origin(elemType) or elemType
    isVersionableElem = isinstance(concreteElem, type) and issubclass(concreteElem, Versionable)

    result: list[Any] = []
    for key in keys:
        item = group[key]
        if isVersionableElem and isinstance(item, h5py.Group):
            result.append(_readVersionableGroup(item))
        elif isinstance(item, h5py.Dataset):
            result.append(item[()])
        elif isinstance(item, h5py.Group):
            result.append(_readGroup(item, elemType))
    return result


def _readDictGroup(group: h5py.Group, valType: Any) -> dict[str, Any]:
    """Read a dict from a group with named children."""
    concreteVal = typing.get_origin(valType) or valType if valType is not None else None
    isVersionableVal = isinstance(concreteVal, type) and issubclass(concreteVal, Versionable)

    result: dict[str, Any] = {}
    for key in group:
        if key == _VERSIONABLE_GROUP:
            continue
        item = group[key]
        if isVersionableVal and isinstance(item, h5py.Group):
            result[key] = _readVersionableGroup(item)
        elif isinstance(item, h5py.Dataset):
            result[key] = item[()]
        elif isinstance(item, h5py.Group):
            result[key] = _readGroup(item, valType)
    return result


def _readAttr(value: Any) -> Any:  # noqa: PLR0911 — type dispatch
    """Convert an HDF5 attribute value to a Python value."""
    # h5py.Empty → None
    if isinstance(value, h5py.Empty):
        return None

    # numpy scalars → Python primitives
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.bytes_):
        return value.decode("utf-8")
    if isinstance(value, bytes):
        return value.decode("utf-8")

    return value


# ---------------------------------------------------------------------------
# Type inspection helpers
# ---------------------------------------------------------------------------


def _resolveClass(objectName: str) -> type | None:
    """Look up a Versionable class by its serialization name."""
    from versionable._base import registeredClasses

    registry = registeredClasses()
    return registry.get(objectName)


def _isArrayField(fieldType: Any) -> bool:
    """Check if a field type is np.ndarray (not a list stored as dataset)."""
    if fieldType is None:
        return True  # unknown type, assume array (backward compat)
    origin = typing.get_origin(fieldType)
    if origin is not None:
        # e.g. npt.NDArray[np.float64] has origin np.ndarray
        return _isNdarrayType(fieldType)
    return fieldType is np.ndarray


def _isNdarrayType(fieldType: Any) -> bool:
    """Check if a type annotation refers to np.ndarray."""
    origin = typing.get_origin(fieldType)
    if origin is np.ndarray:
        return True
    if fieldType is np.ndarray:
        return True
    # Check for npt.NDArray which is Annotated[ndarray, ...]
    if origin is not None:
        args = typing.get_args(fieldType)
        if args and args[0] is np.ndarray:
            return True
    return False


def _isListField(fieldType: Any) -> bool:
    """Check if a field type is a list type."""
    if fieldType is None:
        return False
    origin = typing.get_origin(fieldType)
    return origin is list


def _dtypeForElementType(elemType: Any) -> np.dtype[Any] | Any:
    """Map a Python element type to a numpy dtype for empty datasets."""
    if elemType is int:
        return np.dtype(np.int64)
    if elemType is float:
        return np.dtype(np.float64)
    if elemType is bool:
        return np.dtype(np.bool_)
    if elemType is str:
        return h5py.string_dtype()
    return np.dtype(np.float64)  # fallback


# Register for .h5 and .hdf5 extensions
registerBackend([".h5", ".hdf5"], Hdf5Backend)
