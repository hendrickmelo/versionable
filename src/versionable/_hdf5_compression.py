# hdf5plugin.Zstd/Blosc2 are public APIs with incomplete type stubs
# pyright: reportPrivateImportUsage=false
"""HDF5 compression configuration and presets."""

from __future__ import annotations

import types
from dataclasses import dataclass
from typing import Any, Literal


def _requireHdf5plugin() -> types.ModuleType:
    """Lazy-import hdf5plugin — only needed for zstd/blosc algorithms."""
    try:
        import hdf5plugin

        return hdf5plugin
    except ImportError as e:
        raise ImportError(
            "zstd/blosc compression requires hdf5plugin — install it with: `pip install hdf5plugin`"
        ) from e


type Hdf5CompressionAlgorithm = Literal["zstd", "gzip", "lzf", "blosc"]
type BloscCompressor = Literal["zstd", "blosclz", "lz4", "lz4hc", "zlib"]


@dataclass(frozen=True)
class Hdf5Compression:
    """Compression settings for HDF5 dataset creation.

    The default uses gzip (level 4) for maximum compatibility across tools
    (MATLAB, HDFView, h5py without plugins). Use zstd or blosc for better
    speed/ratio when all readers have hdf5plugin.

    Usage::

        from versionable.hdf5 import Hdf5Compression, GZIP_DEFAULT
        versionable.save(obj, "out.h5", compression=GZIP_DEFAULT)

    Or with custom settings::

        comp = Hdf5Compression(algorithm="zstd", level=9)
        versionable.save(obj, "out.h5", compression=comp)

    See `hdf5plugin docs <https://hdf5plugin.readthedocs.io/en/latest/usage.html>`_
    for details on filter parameters.
    """

    algorithm: Hdf5CompressionAlgorithm | None
    level: int | None
    shuffle: bool = True
    bloscCompressor: BloscCompressor = "zstd"

    def datasetKwargs(self) -> dict[str, Any]:
        """Build kwargs dict for ``h5py.Group.create_dataset()``."""
        if self.algorithm is None:
            return {}

        if self.algorithm == "zstd":
            hdf5plugin = _requireHdf5plugin()
            return dict(**hdf5plugin.Zstd(clevel=3 if self.level is None else self.level))

        if self.algorithm == "blosc":
            hdf5plugin = _requireHdf5plugin()
            shuffleFilter = hdf5plugin.Blosc2.SHUFFLE if self.shuffle else hdf5plugin.Blosc2.NOFILTER
            return dict(
                **hdf5plugin.Blosc2(
                    cname=self.bloscCompressor,
                    clevel=5 if self.level is None else self.level,
                    filters=shuffleFilter,
                )
            )

        # Built-in h5py filters: gzip, lzf
        kwargs: dict[str, Any] = {"compression": self.algorithm}
        if self.algorithm != "lzf" and self.level is not None:
            kwargs["compression_opts"] = self.level
        if self.shuffle:
            kwargs["shuffle"] = True
        return kwargs


# ---------------------------------------------------------------------------
# Presets
# ---------------------------------------------------------------------------

GZIP_DEFAULT = Hdf5Compression(algorithm="gzip", level=4)
ZSTD_DEFAULT = Hdf5Compression(algorithm="zstd", level=3)
ZSTD_FAST = Hdf5Compression(algorithm="zstd", level=1)
ZSTD_BEST = Hdf5Compression(algorithm="zstd", level=9)
BLOSC_DEFAULT = Hdf5Compression(algorithm="blosc", level=5, bloscCompressor="zstd")
LZF = Hdf5Compression(algorithm="lzf", level=None, shuffle=False)
UNCOMPRESSED = Hdf5Compression(algorithm=None, level=None, shuffle=False)

DEFAULT_COMPRESSION = GZIP_DEFAULT
