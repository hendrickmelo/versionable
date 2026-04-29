"""JSON storage backend.

Stores Versionable objects as pretty-printed JSON with metadata
envelope.  Numpy arrays are serialized inline as base64-compressed npz.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, ClassVar

from versionable._backend import Backend, registerBackend
from versionable._base import _resolveFields
from versionable._types import serialize
from versionable.errors import BackendError


class JsonBackend(Backend):
    """JSON file backend for small/medium data."""

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
        # the error path.  ``_path``/``_visited``/``_rootId`` are
        # private helpers; custom backends can ignore ``_rootId``.
        visited: set[int] = {_rootId} if _rootId is not None else set()
        fields = {
            k: serialize(v, fieldTypes[k], nativeTypes=self.nativeTypes, _visited=visited, _path=k)
            for k, v in fields.items()
        }

        data = {
            "__versionable__": {
                "object": meta["name"],
                "version": meta["version"],
                "hash": meta["hash"],
            },
            **fields,
        }
        try:
            path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
        except (OSError, TypeError) as e:
            raise BackendError(f"Failed to write JSON to {path}: {e}") from e

    def load(self, path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
        try:
            text = path.read_text(encoding="utf-8")
            data = json.loads(text)
        except (OSError, json.JSONDecodeError) as e:
            raise BackendError(f"Failed to read JSON from {path}: {e}") from e

        if not isinstance(data, dict):
            raise BackendError(f"Expected JSON object in {path}, got {type(data).__name__}")

        metaTable = data.pop("__versionable__", {})
        if not isinstance(metaTable, dict):
            raise BackendError(f"Missing or invalid __versionable__ metadata in {path}")

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

        return data, meta


# Register for .json extension
registerBackend([".json"], JsonBackend)
