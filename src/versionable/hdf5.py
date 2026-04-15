"""HDF5 compression configuration, presets, and session API.

Usage::

    import versionable
    from versionable.hdf5 import Hdf5Compression, GZIP_DEFAULT
    versionable.save(obj, "out.h5", compression=GZIP_DEFAULT)

    # Save-as-you-go session
    import versionable.hdf5
    with versionable.hdf5.open(MyClass, "out.h5") as obj:
        obj.name = "foo"
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal, overload

from versionable._base import Versionable
from versionable._hdf5_compression import BLOSC_DEFAULT as BLOSC_DEFAULT
from versionable._hdf5_compression import DEFAULT_COMPRESSION as DEFAULT_COMPRESSION
from versionable._hdf5_compression import GZIP_DEFAULT as GZIP_DEFAULT
from versionable._hdf5_compression import LZF as LZF
from versionable._hdf5_compression import UNCOMPRESSED as UNCOMPRESSED
from versionable._hdf5_compression import ZSTD_BEST as ZSTD_BEST
from versionable._hdf5_compression import ZSTD_DEFAULT as ZSTD_DEFAULT
from versionable._hdf5_compression import ZSTD_FAST as ZSTD_FAST
from versionable._hdf5_compression import Hdf5Compression as Hdf5Compression
from versionable._hdf5_session import Hdf5Session

type SessionMode = Literal["create", "resume", "overwrite", "read"]


@overload
def open[T: Versionable](  # noqa: A001
    cls_or_instance: type[T],
    path: str | Path,
    *,
    mode: SessionMode = "create",
    compression: Hdf5Compression | None = None,
) -> Hdf5Session[T]: ...


@overload
def open[T: Versionable](  # noqa: A001
    cls_or_instance: T,
    path: str | Path,
    *,
    mode: SessionMode = "create",
    compression: Hdf5Compression | None = None,
) -> Hdf5Session[T]: ...


def open[T: Versionable](  # noqa: A001 — intentional; mirrors stdlib pattern (io.open)
    cls_or_instance: type[T] | T,
    path: str | Path,
    *,
    mode: SessionMode = "create",
    compression: Hdf5Compression | None = None,
) -> Hdf5Session[T]:
    """Open a file-backed Versionable instance for incremental writes.

    Args:
        cls_or_instance: A Versionable class (empty proxy) or an existing
            instance (all fields persisted on enter).
        path: HDF5 file path.
        mode: How to handle the target file:
            ``"create"`` (default) — new file, error if exists.
            ``"resume"`` — open existing file, restore state, continue.
            ``"overwrite"`` — delete existing file if present, create new.
            ``"read"`` — open existing file read-only, no mutations allowed.
        compression: Compression preset (default: DEFAULT_COMPRESSION, currently gzip level 4).

    Returns:
        Context manager yielding a file-backed instance of cls.
    """
    if isinstance(cls_or_instance, Versionable):
        instance = cls_or_instance
        cls = type(instance)
        return Hdf5Session(cls, path, mode=mode, compression=compression, instance=instance)
    return Hdf5Session(cls_or_instance, path, mode=mode, compression=compression)


__all__ = [
    "BLOSC_DEFAULT",
    "DEFAULT_COMPRESSION",
    "GZIP_DEFAULT",
    "LZF",
    "UNCOMPRESSED",
    "ZSTD_BEST",
    "ZSTD_DEFAULT",
    "ZSTD_FAST",
    "Hdf5Compression",
    "Hdf5Session",
    "SessionMode",
    "open",
]
