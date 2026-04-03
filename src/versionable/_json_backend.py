"""JSON storage backend.

Stores Versionable objects as pretty-printed JSON with metadata
envelope (``__VERSION__``, ``__HASH__``, ``__OBJECT__``).  Numpy arrays
are serialized inline as base64-compressed npz.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, ClassVar

from versionable._backend import Backend, registerBackend
from versionable.errors import BackendError


class JsonBackend(Backend):
    """JSON file backend for small/medium data."""

    nativeTypes: ClassVar[set[type]] = set()

    def save(
        self,
        fields: dict[str, Any],
        meta: dict[str, Any],
        path: Path,
        **kwargs: Any,
    ) -> None:
        data = {
            "__OBJECT__": meta["name"],
            "__VERSION__": meta["version"],
            "__HASH__": meta["hash"],
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

        meta = {
            "__OBJECT__": data.pop("__OBJECT__", ""),
            "__VERSION__": data.pop("__VERSION__", None),
            "__HASH__": data.pop("__HASH__", ""),
        }
        # Also remove __COMPAT__ if present
        data.pop("__COMPAT__", None)

        return data, meta


# Register for .json extension
registerBackend([".json"], JsonBackend)
