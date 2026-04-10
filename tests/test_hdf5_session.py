# ruff: noqa: E402 — module-level imports must come after pytest.importorskip("h5py")
"""Tests for the HDF5 save-as-you-go session."""

from __future__ import annotations

import pytest

h5py = pytest.importorskip("h5py")

from dataclasses import dataclass
from typing import Annotated

import numpy as np
import numpy.typing as npt

import versionable
from versionable import Appendable, Versionable
from versionable._appendable import (
    _computeChunkSize,
    _getAppendable,
    _resolveAppendAxis,
)
from versionable._hash import computeHash
from versionable.errors import BackendError

# ---------------------------------------------------------------------------
# Phase 1 test classes
# ---------------------------------------------------------------------------

_h_with_appendable = computeHash({"name": str, "waveform": npt.NDArray[np.float64]})


@dataclass
class _WithAppendable(
    Versionable,
    version=1,
    hash=_h_with_appendable,
    register=False,
):
    name: str
    waveform: Annotated[npt.NDArray[np.float64], Appendable()]


_h_plain_array = computeHash({"name": str, "waveform": npt.NDArray[np.float64]})


@dataclass
class _WithPlainArray(
    Versionable,
    version=1,
    hash=_h_plain_array,
    register=False,
):
    name: str
    waveform: npt.NDArray[np.float64]


# ---------------------------------------------------------------------------
# Phase 1: Appendable annotation + TrackedArray
# ---------------------------------------------------------------------------


class TestAppendableHashCompat:
    """Appendable fields hash identically to plain np.ndarray."""

    def test_annotated_hashes_same_as_plain(self) -> None:
        """Annotated[np.ndarray, Appendable()] hashes the same as np.ndarray."""
        plain = computeHash({"name": str, "waveform": npt.NDArray[np.float64]})
        annotated = computeHash({"name": str, "waveform": Annotated[npt.NDArray[np.float64], Appendable()]})
        assert plain == annotated

    def test_class_with_appendable_uses_same_hash(self) -> None:
        """A class with Appendable field uses the same hash as one with plain ndarray."""
        assert _WithAppendable._serializer_meta_.hash == _WithPlainArray._serializer_meta_.hash


class TestAppendableSaveLoadPassthrough:
    """save()/load() ignores Appendable marker — field is a normal ndarray."""

    def test_save_load_roundtrip(self, tmp_path: object) -> None:
        path = f"{tmp_path}/test.h5"
        obj = _WithAppendable(name="test", waveform=np.arange(10, dtype=np.float64))
        versionable.save(obj, path)
        loaded = versionable.load(_WithAppendable, path)
        assert loaded.name == "test"
        np.testing.assert_array_equal(loaded.waveform, np.arange(10, dtype=np.float64))

    def test_save_load_json_roundtrip(self, tmp_path: object) -> None:
        """Non-HDF5 backends ignore Appendable entirely."""
        path = f"{tmp_path}/test.json"
        obj = _WithAppendable(name="test", waveform=np.arange(5, dtype=np.float64))
        versionable.save(obj, path)
        loaded = versionable.load(_WithAppendable, path)
        assert loaded.name == "test"
        np.testing.assert_array_equal(loaded.waveform, np.arange(5, dtype=np.float64))


class TestAxisInference:
    """Test _resolveAppendAxis logic."""

    def test_zero_axis_0(self) -> None:
        axis = _resolveAppendAxis((0, 1024), Appendable())
        assert axis == 0

    def test_zero_axis_1(self) -> None:
        axis = _resolveAppendAxis((16, 0), Appendable())
        assert axis == 1

    def test_multiple_zero_axes_error(self) -> None:
        with pytest.raises(BackendError, match="Ambiguous append axis"):
            _resolveAppendAxis((0, 0), Appendable())

    def test_explicit_axis_overrides(self) -> None:
        axis = _resolveAppendAxis((100, 200), Appendable(axis=1))
        assert axis == 1

    def test_no_zero_axis_defaults_to_0(self) -> None:
        axis = _resolveAppendAxis((100, 1024), Appendable())
        assert axis == 0

    def test_explicit_axis_ignores_zero_shape(self) -> None:
        """Explicit axis takes precedence even when a zero-size axis exists."""
        axis = _resolveAppendAxis((0, 1024), Appendable(axis=1))
        assert axis == 1


class TestChunkSize:
    """Test _computeChunkSize heuristic."""

    def test_1d(self) -> None:
        size = _computeChunkSize((0,), np.dtype(np.float64), 0)
        # 256 KB / 8 bytes = 32768
        assert size == 32768

    def test_2d_float64(self) -> None:
        size = _computeChunkSize((0, 1024), np.dtype(np.float64), 0)
        # 1024 * 8 = 8192 bytes/row → 256 KB / 8192 = 32
        assert size == 32

    def test_explicit_chunkRows_not_used_here(self) -> None:
        """_computeChunkSize doesn't know about chunkRows — that's applied in session."""
        # Just verify it returns a positive int for edge cases
        size = _computeChunkSize((0,), np.dtype(np.float32), 0)
        assert size > 0


class TestGetAppendable:
    """Test _getAppendable extraction."""

    def test_annotated_with_appendable(self) -> None:
        ft = Annotated[np.ndarray, Appendable(chunkRows=64)]
        result = _getAppendable(ft)
        assert result is not None
        assert result.chunkRows == 64

    def test_plain_ndarray(self) -> None:
        assert _getAppendable(np.ndarray) is None

    def test_annotated_without_appendable(self) -> None:
        ft = Annotated[np.ndarray, "some metadata"]
        assert _getAppendable(ft) is None

    def test_non_annotated_type(self) -> None:
        assert _getAppendable(int) is None
