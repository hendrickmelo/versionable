"""HDF5 compression filter plugin registration and error hinting.

Importing this module attempts to import ``hdf5plugin`` as a side effect, which
registers its compression filter plugins (zstd, blosc, etc.) with the HDF5
library. Consumers of the HDF5 backend then do not need to import
``hdf5plugin`` themselves to read files that use those filters.

If ``hdf5plugin`` is not installed, the backend still works for files that use
built-in filters (gzip, lzf, uncompressed). Attempts to read files that use a
plugin filter will raise an ``OSError`` from h5py; :func:`missingFilterHint`
turns that into a clear "install hdf5plugin" message.
"""

from __future__ import annotations

# hdf5plugin can fail with OSError (not just ImportError) when its bundled native
# shared libraries can't be loaded — e.g. dlopen "undefined symbol" against an
# incompatible libhdf5. Catch both so a broken install of the optional dependency
# doesn't crash the whole HDF5 backend.
HDF5PLUGIN_AVAILABLE = False
try:
    import hdf5plugin  # noqa: F401  — side-effect import: registers HDF5 filter plugins

    HDF5PLUGIN_AVAILABLE = True
except (ImportError, OSError):
    pass


_FILTER_ERROR_KEYWORDS = ("filter", "pipeline", "plugin")

_INSTALL_HINT = (
    " This file appears to use a compression filter (e.g. zstd, blosc) that requires "
    "the hdf5plugin package. Install it with: `pip install hdf5plugin`."
)


def missingFilterHint(exc: BaseException) -> str:
    """Return an install hint when *exc* looks like a missing HDF5 filter error.

    Returns an empty string when ``hdf5plugin`` is already installed (the error
    is not caused by a missing plugin) or when *exc*'s message does not match
    the filter-error keywords.
    """
    if HDF5PLUGIN_AVAILABLE:
        return ""
    msg = str(exc).lower()
    if any(kw in msg for kw in _FILTER_ERROR_KEYWORDS):
        return _INSTALL_HINT
    return ""
