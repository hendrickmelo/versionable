"""Storage backend ABC and auto-detection by file extension."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, ClassVar

from versionable.errors import BackendError


class Backend(ABC):
    """Abstract base for storage backends."""

    nativeTypes: ClassVar[set[type]] = set()

    @abstractmethod
    def save(
        self,
        fields: dict[str, Any],
        meta: dict[str, Any],
        path: Path,
        **kwargs: Any,
    ) -> None:
        """Write serialized fields and metadata to *path*."""

    @abstractmethod
    def load(self, path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
        """Read fields and metadata from *path*.

        Returns:
            Tuple of (fields_dict, metadata_dict).
            metadata_dict should contain ``__VERSION__``, ``__HASH__``,
            ``__OBJECT__`` keys.
        """


# Extension → Backend class mapping
_BACKEND_REGISTRY: dict[str, type[Backend]] = {}


def registerBackend(extensions: list[str], backendCls: type[Backend]) -> None:
    """Register a backend class for the given file extensions."""
    for ext in extensions:
        _BACKEND_REGISTRY[ext.lower()] = backendCls


def getBackend(path: Path | str, explicit: type[Backend] | None = None) -> Backend:
    """Get a backend instance for *path*.

    Args:
        path: File path (extension used for auto-detection).
        explicit: If provided, use this backend class instead of auto-detecting.
    """
    if explicit is not None:
        return explicit()

    path = Path(path)
    ext = path.suffix.lower()
    if ext not in _BACKEND_REGISTRY:
        hint = ""
        if ext in (".h5", ".hdf5"):
            hint = " Install the HDF5 extra: `pip install versionable[hdf5]`"
        raise BackendError(
            f"No backend registered for extension {ext!r}.{hint} Known extensions: {sorted(_BACKEND_REGISTRY.keys())}"
        )
    return _BACKEND_REGISTRY[ext]()
