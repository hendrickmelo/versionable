"""HDF5 compression configuration and presets.

Usage::

    from versionable.hdf5 import Hdf5Compression, ZSTD_DEFAULT
    versionable.save(obj, "out.h5", compression=ZSTD_DEFAULT)
"""

from __future__ import annotations

from versionable._hdf5_compression import BLOSC_DEFAULT as BLOSC_DEFAULT
from versionable._hdf5_compression import GZIP_DEFAULT as GZIP_DEFAULT
from versionable._hdf5_compression import LZF as LZF
from versionable._hdf5_compression import UNCOMPRESSED as UNCOMPRESSED
from versionable._hdf5_compression import ZSTD_BEST as ZSTD_BEST
from versionable._hdf5_compression import ZSTD_DEFAULT as ZSTD_DEFAULT
from versionable._hdf5_compression import ZSTD_FAST as ZSTD_FAST
from versionable._hdf5_compression import Hdf5Compression as Hdf5Compression

__all__ = [
    "BLOSC_DEFAULT",
    "GZIP_DEFAULT",
    "LZF",
    "UNCOMPRESSED",
    "ZSTD_BEST",
    "ZSTD_DEFAULT",
    "ZSTD_FAST",
    "Hdf5Compression",
]
