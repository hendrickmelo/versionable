"""Numpy availability helpers.

Centralises the try/except for numpy so every module uses the same
pattern and error message.
"""

from __future__ import annotations

import types
from typing import Any

try:
    import numpy

    _np: types.ModuleType | None = numpy
except ImportError:
    _np = None


def hasNumpy() -> bool:
    """Return True if numpy is installed."""
    return _np is not None


def requireNumpy(feature: str = "This feature") -> Any:
    """Return the numpy module, or raise ImportError with a clear message."""
    if _np is None:
        raise ImportError(f"{feature} requires numpy. Install it with: pip install numpy")
    return _np
