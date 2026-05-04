"""TOML storage backend.

Stores Versionable objects as TOML files.  Metadata is stored in a
``[__versionable__]`` table to avoid conflicts with field names.

Nested Versionable objects use native TOML table syntax, with each
nested object's metadata under its own ``[<field>.__versionable__]``
sub-table::

    [__versionable__]
    object = "Config"
    version = 1

    name = "myapp"

    [point]
    x = 1.0
    y = 2.0

    [point.__versionable__]
    object = "Point"
    version = 1

Supports ``commentDefaults=True`` to comment out fields that are at
their default value, producing human-friendly config files.

TOML limitations compared to JSON:
- No ``None`` values — fields with ``None`` are omitted on save and
  restored from dataclass defaults on load.
- Keys are always strings.
- Better suited for config-style data than large arrays.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, ClassVar

try:
    import tomlkit
    from tomlkit.exceptions import TOMLKitError
except ImportError as e:
    raise ImportError("TOML backend requires tomlkit — install it with: `pip install tomlkit`") from e

from versionable._backend import Backend, registerBackend
from versionable._base import Versionable, _resolveFields
from versionable._types import serialize
from versionable.errors import BackendError


class TomlBackend(Backend):
    """TOML file backend for configuration data."""

    nativeTypes: ClassVar[set[type]] = set()

    def save(
        self,
        fields: dict[str, Any],
        meta: dict[str, Any],
        path: Path,
        *,
        cls: type,
        _rootId: int | None = None,
        **kwargs: Any,
    ) -> None:
        fieldTypes = _resolveFields(cls)
        # Seed cycle detection with the root object's id (when provided
        # by ``_api.save``) so a self-reference is reported at the
        # closing edge.  ``_path=k`` puts the top-level field name into
        # the error path.
        visited: set[int] = {_rootId} if _rootId is not None else set()
        fields = {
            k: serialize(v, fieldTypes[k], nativeTypes=self.nativeTypes, _visited=visited, _path=k)
            for k, v in fields.items()
        }
        commentDefaults: bool = kwargs.get("commentDefaults", False)

        data: dict[str, Any] = {
            "__versionable__": {
                "object": meta["name"],
                "version": meta["version"],
                "hash": meta["hash"],
            },
        }
        # TOML cannot represent None — omit those fields
        for key, value in fields.items():
            if value is None:
                continue
            data[key] = _toTomlSafe(value)

        try:
            content = _emitWithCommentedDefaults(data, cls) if commentDefaults else tomlkit.dumps(data)
            path.write_text(content, encoding="utf-8")
        except (OSError, TypeError) as e:
            raise BackendError(f"Failed to write TOML to {path}: {e}") from e

    def load(self, path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
        try:
            text = path.read_text(encoding="utf-8")
            data: dict[str, Any] = tomlkit.parse(text).unwrap()
        except (OSError, TOMLKitError) as e:
            raise BackendError(f"Failed to read TOML from {path}: {e}") from e

        if not isinstance(data, dict):
            raise BackendError(f"Expected TOML table in {path}, got {type(data).__name__}")

        metaTable = data.pop("__versionable__", {})
        if not isinstance(metaTable, dict):
            raise BackendError(f"Expected __versionable__ to be a table in {path}, got {type(metaTable).__name__}")
        fileFormat = metaTable.get("format", metaTable.get("__FORMAT__"))
        if fileFormat is not None:
            raise BackendError(
                f"File {path} uses versionable format {fileFormat!r}, but this version only supports "
                f"format-less files. Upgrade versionable to read this file."
            )

        meta = {
            "object": metaTable.get("object", metaTable.get("__OBJECT__", "")),
            "version": metaTable.get("version", metaTable.get("__VERSION__")),
            "hash": metaTable.get("hash", metaTable.get("__HASH__", "")),
        }

        # Unwrap __ver_json__ wrappers (ndarray blobs)
        data = _fromTomlSafe(data)

        return data, meta


# ---------------------------------------------------------------------------
# TOML value conversion helpers
# ---------------------------------------------------------------------------


def _toTomlSafe(value: Any) -> Any:
    """Convert a value to a TOML-safe representation.

    - Nested Versionable dicts (with ``object``) are kept as native
      TOML tables — they naturally become ``[field_name]`` sections.
    - ndarray dicts (with ``__ver_ndarray__``) are stored as JSON strings
      since they have no natural TOML representation.
    - ``None`` values are stripped (TOML cannot represent them).
    """
    if isinstance(value, dict):
        # ndarray blob — must use JSON wrapper
        if "__ver_ndarray__" in value:
            return {"__ver_json__": json.dumps(value, default=str)}
        # Nested Versionable or plain dict — keep as TOML table
        return {k: _toTomlSafe(v) for k, v in value.items() if v is not None}
    if isinstance(value, (list, tuple)):
        return [_toTomlSafe(v) for v in value]
    return value


def _fromTomlSafe(value: Any) -> Any:
    """Reverse of ``_toTomlSafe`` — unwrap ``__ver_json__`` wrappers.

    Also accepts the legacy ``__json__`` wrapper from 0.1.x files.
    """
    if isinstance(value, dict):
        if "__ver_json__" in value and len(value) == 1:
            return json.loads(value["__ver_json__"])
        if "__json__" in value and len(value) == 1:
            return json.loads(value["__json__"])
        return {k: _fromTomlSafe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_fromTomlSafe(v) for v in value]
    return value


# ---------------------------------------------------------------------------
# Comment-defaults logic
# ---------------------------------------------------------------------------


def _emitWithCommentedDefaults(data: dict[str, Any], cls: type) -> str:
    """Build TOML output where fields at their default value appear as comment lines.

    Walks `data` in parallel with the dataclass's defaults. For each scalar/list field
    whose serialized value matches the default, emit a ``# key = value`` comment instead
    of a key-value pair. Section headers and ``__versionable__`` envelope tables are
    never commented. Nested Versionable tables recurse with their own defaults.
    """
    doc = tomlkit.document()
    _addContainerWithDefaults(doc, data, cls)
    return tomlkit.dumps(doc)


def _addContainerWithDefaults(
    container: Any,  # tomlkit.TOMLDocument | tomlkit.items.Table
    data: dict[str, Any],
    cls: type | None,
) -> None:
    """Populate `container` with `data`; commented defaults driven by `cls`."""
    defaults = _classDefaultsToml(cls) if cls is not None else {}
    fieldTypes = _resolveFields(cls) if cls is not None else {}

    for key, value in data.items():
        if key == "__versionable__":
            sub = tomlkit.table()
            for metaKey, metaVal in value.items():
                sub[metaKey] = metaVal
            container[key] = sub
            continue

        if isinstance(value, dict):
            sub = tomlkit.table()
            nestedCls = _findVersionableType(fieldTypes.get(key))
            _addContainerWithDefaults(sub, value, nestedCls)
            container[key] = sub
            continue

        if key in defaults and value == defaults[key]:
            scratch = tomlkit.document()
            scratch[key] = value
            line = tomlkit.dumps(scratch).rstrip("\n")
            container.add(tomlkit.comment(line))
        else:
            container[key] = value


def _classDefaultsToml(cls: type) -> dict[str, Any]:
    """Compute serialized + TOML-safe defaults for each field of `cls`."""
    import dataclasses

    dcFields = {f.name: f for f in dataclasses.fields(cls)}
    resolvedFields = _resolveFields(cls)
    out: dict[str, Any] = {}
    for name, tp in resolvedFields.items():
        dcField = dcFields.get(name)
        if dcField is None:
            continue
        if dcField.default is not dataclasses.MISSING:
            val = serialize(dcField.default, tp)
            if val is not None:
                out[name] = _toTomlSafe(val)
        elif dcField.default_factory is not dataclasses.MISSING:
            val = serialize(dcField.default_factory(), tp)
            if val is not None:
                out[name] = _toTomlSafe(val)
    return out


def _findVersionableType(fieldType: Any) -> type | None:
    """If `fieldType` is a Versionable subclass, return it; else None.

    Parameterized types like ``list[Inner]`` are not unwrapped — commentDefaults
    does not recurse into list elements (matches existing behavior).
    """
    if isinstance(fieldType, type) and issubclass(fieldType, Versionable):
        return fieldType
    return None


# Register for .toml extension
registerBackend([".toml"], TomlBackend)
