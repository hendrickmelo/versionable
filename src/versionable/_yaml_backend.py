"""YAML storage backend.

Stores Versionable objects as YAML files.  Metadata is stored in a
``__versionable__`` mapping to avoid conflicts with field names, matching the
TOML backend convention::

    name: probe-A
    sampleRate_Hz: 120000
    channels:
    - 0
    - 1
    - 2
    __versionable__:
      object: SensorConfig
      version: 1
      hash: 9d6951

YAML handles ``None`` natively (as ``null``), so unlike TOML no fields
are lost on round-trip.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, ClassVar

try:
    import yaml
except ImportError as e:
    raise ImportError("YAML backend requires pyyaml — install it with: `pip install pyyaml`") from e

from versionable._backend import Backend, registerBackend
from versionable._base import _resolveFields
from versionable._types import serialize
from versionable.errors import BackendError


class YamlBackend(Backend):
    """YAML file backend for human-readable config and data files."""

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
        data: dict[str, Any] = {}
        for key, value in fields.items():
            data[key] = _toYamlSafe(value)

        data["__versionable__"] = {
            "object": meta["name"],
            "version": meta["version"],
            "hash": meta["hash"],
        }

        commentDefaults: bool = kwargs.get("commentDefaults", False)

        try:
            content = yaml.dump(data, default_flow_style=False, sort_keys=False)
            if commentDefaults:
                content = _commentDefaultLines(content, fields, meta["name"])
            path.write_text(content, encoding="utf-8")
        except (OSError, yaml.YAMLError) as e:
            raise BackendError(f"Failed to write YAML to {path}: {e}") from e

    def load(self, path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
        try:
            text = path.read_text(encoding="utf-8")
            data = yaml.safe_load(text)
        except (OSError, yaml.YAMLError) as e:
            raise BackendError(f"Failed to read YAML from {path}: {e}") from e

        if not isinstance(data, dict):
            raise BackendError(f"Expected YAML mapping in {path}, got {type(data).__name__}")

        metaTable = data.pop("__versionable__", {})
        if not isinstance(metaTable, dict):
            raise BackendError(f"Expected __versionable__ to be a mapping in {path}, got {type(metaTable).__name__}")
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

        try:
            data = _fromYamlSafe(data)
        except (ValueError, TypeError) as e:
            raise BackendError(f"Failed to decode embedded JSON in {path}: {e}") from e

        return data, meta


# ---------------------------------------------------------------------------
# YAML value conversion helpers
# ---------------------------------------------------------------------------


def _toYamlSafe(value: Any) -> Any:
    """Convert a value to a YAML-safe representation.

    ndarray dicts (with ``__ver_ndarray__``) are stored as JSON strings
    since they have no natural YAML representation.
    """
    if isinstance(value, dict):
        if "__ver_ndarray__" in value:
            return {"__ver_json__": json.dumps(value, default=str)}
        return {k: _toYamlSafe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_toYamlSafe(v) for v in value]
    return value


def _fromYamlSafe(value: Any) -> Any:
    """Reverse of ``_toYamlSafe`` — unwrap ``__ver_json__`` wrappers.

    Also accepts the legacy ``__json__`` wrapper from 0.1.x files.
    """
    if isinstance(value, dict):
        if "__ver_json__" in value and len(value) == 1:
            return json.loads(value["__ver_json__"])
        if "__json__" in value and len(value) == 1:
            return json.loads(value["__json__"])
        return {k: _fromYamlSafe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_fromYamlSafe(v) for v in value]
    return value


# ---------------------------------------------------------------------------
# Comment-defaults logic
# ---------------------------------------------------------------------------


def _findClass(objectName: str) -> type | None:
    """Find a Versionable subclass by its serialization name.

    Checks the registry first, then scans all Versionable subclasses
    (handles ``register=False`` classes used in tests).
    """
    from versionable._base import Versionable, metadata, registeredClasses

    registry = registeredClasses()
    if objectName in registry:
        return registry[objectName]

    # Fallback: scan all subclasses (covers register=False)
    for sub in Versionable.__subclasses__():
        try:
            if metadata(sub).name == objectName:
                return sub
        except Exception:
            continue
    return None


def _commentDefaultLines(content: str, fields: dict[str, Any], objectName: str) -> str:
    """Comment out YAML lines for fields whose values match the class defaults.

    The ``__versionable__`` block is always kept uncommented (required for
    deserialization).  Multi-line values (lists, nested mappings) are
    handled by commenting out the entire indented block under a top-level
    key.
    """
    import dataclasses

    from versionable._base import _resolveFields
    from versionable._types import serialize

    # Find the class — check registry first, then scan subclasses
    cls = _findClass(objectName)
    if cls is None:
        return content

    # Build YAML representation of default values (per-field, to handle
    # classes with required fields that can't be default-constructed).
    dcFields = {f.name: f for f in dataclasses.fields(cls)}
    resolvedFields = _resolveFields(cls)
    defaultSerialized: dict[str, object] = {}
    for name, tp in resolvedFields.items():
        dcField = dcFields.get(name)
        if dcField is None:
            continue
        if dcField.default is not dataclasses.MISSING:
            val = serialize(dcField.default, tp)
            defaultSerialized[name] = _toYamlSafe(val)
        elif dcField.default_factory is not dataclasses.MISSING:
            val = serialize(dcField.default_factory(), tp)
            defaultSerialized[name] = _toYamlSafe(val)

    defaultYaml = yaml.dump(defaultSerialized, default_flow_style=False, sort_keys=False)
    # Parse into blocks keyed by top-level field name
    defaultBlocks = _parseTopLevelBlocks(defaultYaml)

    # Parse original content into blocks
    originalBlocks = _parseTopLevelBlocks(content)

    # Walk lines, commenting out blocks that match defaults
    lines = content.splitlines(keepends=True)
    result: list[str] = []
    i = 0
    while i < len(lines):
        stripped = lines[i].rstrip("\n")

        # Detect top-level key (not indented, not blank, not comment)
        if stripped and not stripped[0].isspace() and ":" in stripped:
            key = stripped.split(":")[0]

            # Never comment out __versionable__
            if key == "__versionable__":
                # Pass through __versionable__ block
                result.append(lines[i])
                i += 1
                while i < len(lines) and lines[i].rstrip("\n") and lines[i][0].isspace():
                    result.append(lines[i])
                    i += 1
                continue

            # Check if this block matches the default
            if key in defaultBlocks and key in originalBlocks and defaultBlocks[key] == originalBlocks[key]:
                # Peek ahead: if the block contains a ``__versionable__:``
                # line, it's a nested Versionable — keep the key line and
                # the ``__versionable__:`` sub-block uncommented, only
                # comment data-field value lines (siblings of the wrapper).
                blockLines: list[str] = []
                j = i + 1
                while j < len(lines):
                    line = lines[j]
                    lineStripped = line.rstrip("\n")
                    if lineStripped and (lineStripped[0].isspace() or lineStripped.startswith("- ")):
                        blockLines.append(line)
                        j += 1
                    else:
                        break

                hasNestedMeta = any(bl.lstrip(" ").startswith("__versionable__:") for bl in blockLines)
                if hasNestedMeta:
                    # Keep key line uncommented; keep the ``__versionable__:``
                    # sub-block uncommented; comment data fields at the
                    # nested level (i.e. siblings of ``__versionable__:``).
                    result.append(lines[i])
                    metaIndent: int | None = None
                    for bl in blockLines:
                        blLstrip = bl.lstrip(" ")
                        blIndent = len(bl) - len(blLstrip)
                        if blLstrip.startswith("__versionable__:"):
                            metaIndent = blIndent
                            result.append(bl)
                        elif metaIndent is not None and blIndent > metaIndent:
                            # Inside the envelope sub-block
                            result.append(bl)
                        else:
                            # Sibling of __versionable__ (a data field)
                            metaIndent = None
                            result.append("# " + bl)
                else:
                    # Simple field — comment the whole block
                    result.append("# " + lines[i])
                    result.extend("# " + bl for bl in blockLines)
                i = j
                continue

        result.append(lines[i])
        i += 1

    return "".join(result)


def _parseTopLevelBlocks(content: str) -> dict[str, str]:
    """Parse YAML content into top-level blocks keyed by field name.

    Each block includes the key line and all indented continuation lines.
    """
    blocks: dict[str, str] = {}
    lines = content.splitlines(keepends=True)
    currentKey: str | None = None
    currentLines: list[str] = []

    for line in lines:
        stripped = line.rstrip("\n")

        # Top-level key
        if stripped and not stripped[0].isspace() and ":" in stripped:
            # Save previous block
            if currentKey is not None:
                blocks[currentKey] = "".join(currentLines)
            currentKey = stripped.split(":")[0]
            currentLines = [line]
        elif currentKey is not None:
            # Continuation (indented or bare list item)
            if stripped and (stripped[0].isspace() or stripped.startswith("- ")):
                currentLines.append(line)
            elif not stripped:
                # Blank line — include in block
                currentLines.append(line)
            else:
                # New non-indented line that isn't a key — save and reset
                blocks[currentKey] = "".join(currentLines)
                currentKey = None
                currentLines = []
        # else: skip lines before first key

    if currentKey is not None:
        blocks[currentKey] = "".join(currentLines)

    return blocks


# Register for .yaml and .yml extensions
registerBackend([".yaml", ".yml"], YamlBackend)
