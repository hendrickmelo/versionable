"""YAML storage backend.

Stores Versionable objects as YAML files.  Metadata is stored in a
``__meta__`` mapping to avoid conflicts with field names, matching the
TOML backend convention::

    name: probe-A
    sampleRate_Hz: 120000
    channels:
    - 0
    - 1
    - 2
    __meta__:
      __OBJECT__: SensorConfig
      __VERSION__: 1
      __HASH__: 9d6951

YAML handles ``None`` natively (as ``null``), so unlike TOML no fields
are lost on round-trip.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, ClassVar

import yaml

from versionable._backend import Backend, registerBackend
from versionable.errors import BackendError


class YamlBackend(Backend):
    """YAML file backend for human-readable config and data files."""

    nativeTypes: ClassVar[set[type]] = set()

    def save(
        self,
        fields: dict[str, Any],
        meta: dict[str, Any],
        path: Path,
        **kwargs: Any,
    ) -> None:
        data: dict[str, Any] = {}
        for key, value in fields.items():
            data[key] = _toYamlSafe(value)

        data["__meta__"] = {
            "__OBJECT__": meta["name"],
            "__VERSION__": meta["version"],
            "__HASH__": meta["hash"],
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

        metaTable = data.pop("__meta__", {})
        if not isinstance(metaTable, dict):
            raise BackendError(f"Expected __meta__ to be a mapping in {path}, got {type(metaTable).__name__}")
        meta = {
            "__OBJECT__": metaTable.get("__OBJECT__", ""),
            "__VERSION__": metaTable.get("__VERSION__"),
            "__HASH__": metaTable.get("__HASH__", ""),
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

    ndarray dicts (with ``__ndarray__``) are stored as JSON strings
    since they have no natural YAML representation.
    """
    if isinstance(value, dict):
        if "__ndarray__" in value:
            return {"__json__": json.dumps(value, default=str)}
        return {k: _toYamlSafe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_toYamlSafe(v) for v in value]
    return value


def _fromYamlSafe(value: Any) -> Any:
    """Reverse of ``_toYamlSafe`` — unwrap ``__json__`` wrappers."""
    if isinstance(value, dict):
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

    The ``__meta__`` block is always kept uncommented (required for
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

            # Never comment out __meta__
            if key == "__meta__":
                # Pass through __meta__ block
                result.append(lines[i])
                i += 1
                while i < len(lines) and lines[i].rstrip("\n") and lines[i][0].isspace():
                    result.append(lines[i])
                    i += 1
                continue

            # Check if this block matches the default
            if key in defaultBlocks and key in originalBlocks and defaultBlocks[key] == originalBlocks[key]:
                # Comment out this line and its continuation lines
                result.append("# " + lines[i])
                i += 1
                while i < len(lines):
                    line = lines[i]
                    lineStripped = line.rstrip("\n")
                    # Continuation: indented lines or bare list items
                    if lineStripped and (lineStripped[0].isspace() or lineStripped.startswith("- ")):
                        result.append("# " + line)
                        i += 1
                    else:
                        break
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
