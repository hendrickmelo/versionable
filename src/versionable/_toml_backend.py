"""TOML storage backend.

Stores Versionable objects as TOML files.  Metadata is stored in a
``[__versionable__]`` table to avoid conflicts with field names.

Nested Versionable objects use native TOML table syntax::

    [__versionable__]
    __OBJECT__ = "Config"
    __VERSION__ = 1

    name = "myapp"

    [point]
    __OBJECT__ = "Point"
    __VERSION__ = 1
    x = 1.0
    y = 2.0

Supports ``commentDefaults=True`` to comment out fields that are at
their default value, producing human-friendly config files.

TOML limitations compared to JSON:
- No ``None`` values — fields with ``None`` are omitted on save and
  restored from dataclass defaults on load.
- Keys are always strings.
- Better suited for config-style data than large arrays.
"""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any, ClassVar

try:
    import toml
except ImportError as e:
    raise ImportError("TOML backend requires toml — install it with: `pip install toml`") from e

from versionable._backend import Backend, registerBackend
from versionable._base import _resolveFields
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
        **kwargs: Any,
    ) -> None:
        fieldTypes = _resolveFields(cls)
        # Pass _path=k so cycle-detection error messages include the
        # top-level field name (e.g. "children[0]" rather than "[0]").
        fields = {k: serialize(v, fieldTypes[k], nativeTypes=self.nativeTypes, _path=k) for k, v in fields.items()}
        commentDefaults: bool = kwargs.get("commentDefaults", False)

        data: dict[str, Any] = {
            "__versionable__": {
                "__OBJECT__": meta["name"],
                "__VERSION__": meta["version"],
                "__HASH__": meta["hash"],
            },
        }
        # TOML cannot represent None — omit those fields
        for key, value in fields.items():
            if value is None:
                continue
            data[key] = _toTomlSafe(value)

        try:
            content = toml.dumps(data)
            if commentDefaults:
                content = _commentDefaultLines(content, fields, meta["name"])
            path.write_text(content, encoding="utf-8")
        except (OSError, TypeError) as e:
            raise BackendError(f"Failed to write TOML to {path}: {e}") from e

    def load(self, path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
        try:
            text = path.read_text(encoding="utf-8")
            data = toml.loads(text)
        except (OSError, toml.TomlDecodeError) as e:
            raise BackendError(f"Failed to read TOML from {path}: {e}") from e

        if not isinstance(data, dict):
            raise BackendError(f"Expected TOML table in {path}, got {type(data).__name__}")

        metaTable = data.pop("__versionable__", {})
        if not isinstance(metaTable, dict):
            raise BackendError(f"Expected __versionable__ to be a table in {path}, got {type(metaTable).__name__}")
        fileFormat = metaTable.get("__FORMAT__")
        if fileFormat is not None:
            raise BackendError(
                f"File {path} uses versionable format {fileFormat!r}, but this version only supports "
                f"format-less files. Upgrade versionable to read this file."
            )

        meta = {
            "__OBJECT__": metaTable.get("__OBJECT__", ""),
            "__VERSION__": metaTable.get("__VERSION__"),
            "__HASH__": metaTable.get("__HASH__", ""),
        }

        # Unwrap __json__ wrappers (ndarray blobs)
        data = _fromTomlSafe(data)

        return data, meta


# ---------------------------------------------------------------------------
# TOML value conversion helpers
# ---------------------------------------------------------------------------


def _toTomlSafe(value: Any) -> Any:
    """Convert a value to a TOML-safe representation.

    - Nested Versionable dicts (with ``__OBJECT__``) are kept as native
      TOML tables — they naturally become ``[field_name]`` sections.
    - ndarray dicts (with ``__ndarray__``) are stored as JSON strings
      since they have no natural TOML representation.
    - ``None`` values are stripped (TOML cannot represent them).
    """
    if isinstance(value, dict):
        # ndarray blob — must use JSON wrapper
        if "__ndarray__" in value:
            return {"__json__": json.dumps(value, default=str)}
        # Nested Versionable or plain dict — keep as TOML table
        return {k: _toTomlSafe(v) for k, v in value.items() if v is not None}
    if isinstance(value, (list, tuple)):
        return [_toTomlSafe(v) for v in value]
    return value


def _fromTomlSafe(value: Any) -> Any:
    """Reverse of ``_toTomlSafe`` — unwrap ``__json__`` wrappers."""
    if isinstance(value, dict):
        if "__json__" in value and len(value) == 1:
            return json.loads(value["__json__"])
        return {k: _fromTomlSafe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_fromTomlSafe(v) for v in value]
    return value


# ---------------------------------------------------------------------------
# Comment-defaults logic
# ---------------------------------------------------------------------------


def _commentDefaultLines(content: str, fields: dict[str, Any], objectName: str) -> str:
    """Comment out TOML value lines that match the class defaults.

    Section headers and ``__versionable__`` blocks are never commented — only
    individual field value lines.  This ensures nested Versionable
    sections remain structurally intact so users can uncomment just
    the fields they need.
    """
    import dataclasses

    from versionable._base import Versionable, _resolveFields, metadata, registeredClasses
    from versionable._types import serialize

    # Find the class — check registry first, then scan subclasses
    cls: type | None = registeredClasses().get(objectName)
    if cls is None:
        for sub in Versionable.__subclasses__():
            try:
                if metadata(sub).name == objectName:
                    cls = sub
                    break
            except Exception:
                continue
    if cls is None:
        return content

    # Build a set of TOML lines that represent the default values (per-field,
    # to handle classes with required fields that can't be default-constructed).
    dcFields = {f.name: f for f in dataclasses.fields(cls)}
    resolvedFields = _resolveFields(cls)
    defaultSerialized: dict[str, object] = {}
    for name, tp in resolvedFields.items():
        dcField = dcFields.get(name)
        if dcField is None:
            continue
        if dcField.default is not dataclasses.MISSING:
            val = serialize(dcField.default, tp)
            if val is not None:
                defaultSerialized[name] = _toTomlSafe(val)
        elif dcField.default_factory is not dataclasses.MISSING:
            val = serialize(dcField.default_factory(), tp)
            if val is not None:
                defaultSerialized[name] = _toTomlSafe(val)

    buf = io.StringIO()
    toml.dump(defaultSerialized, buf)
    defaultLineSet = set(buf.getvalue().splitlines())

    # Walk input lines
    lines = content.splitlines(keepends=True)
    result: list[str] = []
    inMetaSection = False

    for line in lines:
        stripped = line.rstrip("\n")

        # Section header — always keep uncommented
        if stripped.startswith("["):
            inMetaSection = "__versionable__" in stripped
            result.append(line)
            continue

        # Blank / comment line — pass through
        if not stripped or stripped.startswith("#"):
            result.append(line)
            continue

        # __versionable__ section — always keep uncommented
        if inMetaSection:
            result.append(line)
            continue

        # Metadata keys — always keep uncommented (even inside nested sections)
        key = stripped.split("=", 1)[0].strip()
        if key.startswith("__") and key.endswith("__"):
            result.append(line)
            continue

        # Value line — check against defaults
        if stripped in defaultLineSet:
            result.append("# " + line)
        else:
            result.append(line)

    return "".join(result)


# Register for .toml extension
registerBackend([".toml"], TomlBackend)
