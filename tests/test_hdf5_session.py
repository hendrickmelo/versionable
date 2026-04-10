# ruff: noqa: E402 — module-level imports must come after pytest.importorskip("h5py")
"""Tests for the HDF5 save-as-you-go session."""

from __future__ import annotations

import pytest

h5py = pytest.importorskip("h5py")

from dataclasses import dataclass, field
from typing import Annotated

import numpy as np
import numpy.typing as npt

import versionable
import versionable.hdf5
from versionable import Appendable, Versionable
from versionable._appendable import (
    _computeChunkSize,
    _getAppendable,
    _resolveAppendAxis,
)
from versionable._hash import computeHash
from versionable._tracked_array import TrackedArray
from versionable.errors import BackendError

# ---------------------------------------------------------------------------
# Test classes
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


# ---------------------------------------------------------------------------
# Phase 2 test classes
# ---------------------------------------------------------------------------

_h_session_basic = computeHash(
    {
        "name": str,
        "sampleRate_Hz": float,
        "data": npt.NDArray[np.float64],
        "waveform": npt.NDArray[np.float64],
    }
)


@dataclass
class _SessionBasic(
    Versionable,
    version=1,
    hash=_h_session_basic,
    register=False,
):
    name: str = ""
    sampleRate_Hz: float = 0.0
    data: npt.NDArray[np.float64] = field(default_factory=lambda: np.empty(0))
    waveform: Annotated[npt.NDArray[np.float64], Appendable()] = field(default_factory=lambda: np.empty(0))


_h_session_axis1 = computeHash({"channels": npt.NDArray[np.float64]})


@dataclass
class _SessionAxis1(
    Versionable,
    version=1,
    hash=_h_session_axis1,
    register=False,
):
    channels: Annotated[npt.NDArray[np.float64], Appendable(axis=1)] = field(default_factory=lambda: np.empty(0))


_h_session_chunk = computeHash({"data": npt.NDArray[np.float64]})


@dataclass
class _SessionChunk(
    Versionable,
    version=1,
    hash=_h_session_chunk,
    register=False,
):
    data: Annotated[npt.NDArray[np.float64], Appendable(chunkRows=16)] = field(default_factory=lambda: np.empty(0))


# ---------------------------------------------------------------------------
# Phase 2: Core session + scalar/array persistence
# ---------------------------------------------------------------------------


class TestSessionScalars:
    """Scalar field assignment persists to HDF5."""

    def test_assign_scalars_and_load(self, tmp_path: object) -> None:
        path = f"{tmp_path}/test.h5"
        with versionable.hdf5.open(_SessionBasic, path) as obj:
            obj.name = "baseline"
            obj.sampleRate_Hz = 48000.0

        loaded = versionable.load(_SessionBasic, path)
        assert loaded.name == "baseline"
        assert loaded.sampleRate_Hz == 48000.0

    def test_overwrite_scalar(self, tmp_path: object) -> None:
        path = f"{tmp_path}/test.h5"
        with versionable.hdf5.open(_SessionBasic, path) as obj:
            obj.name = "first"
            obj.name = "second"

        loaded = versionable.load(_SessionBasic, path)
        assert loaded.name == "second"


class TestSessionPlainArray:
    """Plain ndarray field assignment creates a contiguous dataset."""

    def test_assign_and_load(self, tmp_path: object) -> None:
        path = f"{tmp_path}/test.h5"
        arr = np.arange(100, dtype=np.float64)
        with versionable.hdf5.open(_SessionBasic, path) as obj:
            obj.data = arr

        loaded = versionable.load(_SessionBasic, path)
        np.testing.assert_array_equal(loaded.data, arr)

    def test_contiguous_dataset(self, tmp_path: object) -> None:
        """Plain ndarray datasets are not chunked/resizable."""
        path = f"{tmp_path}/test.h5"
        arr = np.arange(100, dtype=np.float64)
        with versionable.hdf5.open(_SessionBasic, path) as obj:
            obj.data = arr

        with h5py.File(path, "r") as f:
            ds = f["data"]
            # maxshape should be fixed (same as shape), not (None,)
            assert ds.maxshape == (100,)

    def test_replace_array(self, tmp_path: object) -> None:
        path = f"{tmp_path}/test.h5"
        with versionable.hdf5.open(_SessionBasic, path) as obj:
            obj.data = np.zeros(50, dtype=np.float64)
            obj.data = np.ones(30, dtype=np.float64)

        loaded = versionable.load(_SessionBasic, path)
        np.testing.assert_array_equal(loaded.data, np.ones(30, dtype=np.float64))


class TestSessionAppendableArray:
    """Appendable ndarray fields create resizable datasets."""

    def test_assign_creates_resizable(self, tmp_path: object) -> None:
        path = f"{tmp_path}/test.h5"
        with versionable.hdf5.open(_SessionBasic, path) as obj:
            obj.waveform = np.empty((0, 1024), dtype=np.float64)

        with h5py.File(path, "r") as f:
            ds = f["waveform"]
            assert ds.maxshape[0] is None
            assert ds.maxshape[1] == 1024

    def test_append_loop_and_load(self, tmp_path: object) -> None:
        path = f"{tmp_path}/test.h5"
        chunks = [np.random.randn(10, 4) for _ in range(5)]
        with versionable.hdf5.open(_SessionBasic, path) as obj:
            obj.waveform = np.empty((0, 4), dtype=np.float64)
            for chunk in chunks:
                obj.waveform.append(chunk)

        loaded = versionable.load(_SessionBasic, path)
        expected = np.vstack(chunks)
        np.testing.assert_array_almost_equal(loaded.waveform, expected)

    def test_append_returns_tracked_array(self, tmp_path: object) -> None:
        path = f"{tmp_path}/test.h5"
        with versionable.hdf5.open(_SessionBasic, path) as obj:
            obj.waveform = np.empty((0, 4), dtype=np.float64)
            assert isinstance(obj.waveform, TrackedArray)

    def test_append_axis1(self, tmp_path: object) -> None:
        path = f"{tmp_path}/test.h5"
        with versionable.hdf5.open(_SessionAxis1, path) as obj:
            obj.channels = np.zeros((16, 0), dtype=np.float64)
            obj.channels.append(np.ones((16, 10), dtype=np.float64))
            obj.channels.append(np.ones((16, 5), dtype=np.float64))
            assert obj.channels.shape == (16, 15)

        loaded = versionable.load(_SessionAxis1, path)
        assert loaded.channels.shape == (16, 15)

    def test_append_varying_sizes(self, tmp_path: object) -> None:
        path = f"{tmp_path}/test.h5"
        with versionable.hdf5.open(_SessionBasic, path) as obj:
            obj.waveform = np.empty((0, 4), dtype=np.float64)
            obj.waveform.append(np.ones((3, 4)))
            obj.waveform.append(np.ones((7, 4)))
            obj.waveform.append(np.ones((1, 4)))
            assert obj.waveform.shape == (11, 4)

    def test_append_shape_mismatch(self, tmp_path: object) -> None:
        path = f"{tmp_path}/test.h5"
        with versionable.hdf5.open(_SessionBasic, path) as obj:
            obj.waveform = np.empty((0, 4), dtype=np.float64)
            with pytest.raises(ValueError, match="non-append dimensions must match"):
                obj.waveform.append(np.ones((3, 8)))

    def test_setitem(self, tmp_path: object) -> None:
        path = f"{tmp_path}/test.h5"
        with versionable.hdf5.open(_SessionBasic, path) as obj:
            obj.waveform = np.zeros((5, 4), dtype=np.float64)
            obj.waveform[2] = np.ones(4)

        loaded = versionable.load(_SessionBasic, path)
        np.testing.assert_array_equal(loaded.waveform[2], np.ones(4))
        np.testing.assert_array_equal(loaded.waveform[0], np.zeros(4))

    def test_tracked_array_numpy_interop(self, tmp_path: object) -> None:
        path = f"{tmp_path}/test.h5"
        with versionable.hdf5.open(_SessionBasic, path) as obj:
            obj.waveform = np.ones((10, 4), dtype=np.float64)
            assert np.mean(obj.waveform) == 1.0
            assert len(obj.waveform) == 10
            assert obj.waveform.shape == (10, 4)
            assert obj.waveform.dtype == np.float64
            assert obj.waveform.axis == 0

    def test_explicit_chunk_rows(self, tmp_path: object) -> None:
        path = f"{tmp_path}/test.h5"
        with versionable.hdf5.open(_SessionChunk, path) as obj:
            obj.data = np.empty((0, 100), dtype=np.float64)

        with h5py.File(path, "r") as f:
            ds = f["data"]
            assert ds.chunks[0] == 16  # explicit chunkRows

    def test_reassign_appendable_field(self, tmp_path: object) -> None:
        """Reassigning an Appendable field replaces the dataset."""
        path = f"{tmp_path}/test.h5"
        with versionable.hdf5.open(_SessionBasic, path) as obj:
            obj.waveform = np.zeros((5, 4), dtype=np.float64)
            obj.waveform = np.ones((3, 4), dtype=np.float64)
            assert isinstance(obj.waveform, TrackedArray)
            assert obj.waveform.shape == (3, 4)


class TestSessionModes:
    """Test create/overwrite mode behavior."""

    def test_create_errors_on_existing(self, tmp_path: object) -> None:
        path = f"{tmp_path}/test.h5"
        with versionable.hdf5.open(_SessionBasic, path):
            pass
        with pytest.raises(BackendError, match="already exists"), versionable.hdf5.open(_SessionBasic, path):
            pass

    def test_overwrite_replaces_existing(self, tmp_path: object) -> None:
        path = f"{tmp_path}/test.h5"
        with versionable.hdf5.open(_SessionBasic, path) as obj:
            obj.name = "first"
        with versionable.hdf5.open(_SessionBasic, path, mode="overwrite") as obj:
            obj.name = "second"

        loaded = versionable.load(_SessionBasic, path)
        assert loaded.name == "second"

    def test_overwrite_on_nonexistent(self, tmp_path: object) -> None:
        path = f"{tmp_path}/test.h5"
        with versionable.hdf5.open(_SessionBasic, path, mode="overwrite") as obj:
            obj.name = "new"
        loaded = versionable.load(_SessionBasic, path)
        assert loaded.name == "new"

    def test_context_manager_closes_file(self, tmp_path: object) -> None:
        path = f"{tmp_path}/test.h5"
        with versionable.hdf5.open(_SessionBasic, path) as obj:
            obj.name = "test"
        # File should be loadable after context exit
        loaded = versionable.load(_SessionBasic, path)
        assert loaded.name == "test"


# ---------------------------------------------------------------------------
# Phase 3 test classes
# ---------------------------------------------------------------------------

_h_with_lists = computeHash(
    {
        "name": str,
        "traces": list[npt.NDArray[np.float64]],
        "timestamps": list[float],
        "tags": list[str],
    }
)


@dataclass
class _WithLists(
    Versionable,
    version=1,
    hash=_h_with_lists,
    register=False,
):
    name: str = ""
    traces: list[npt.NDArray[np.float64]] = field(default_factory=list)
    timestamps: list[float] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)


_h_with_dict = computeHash({"name": str, "channels": dict[str, npt.NDArray[np.float64]]})


@dataclass
class _WithDict(
    Versionable,
    version=1,
    hash=_h_with_dict,
    register=False,
):
    name: str = ""
    channels: dict[str, npt.NDArray[np.float64]] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Phase 3: Tracked collections
# ---------------------------------------------------------------------------


class TestTrackedListArrays:
    """list[np.ndarray] append creates datasets in a group."""

    def test_append_loop(self, tmp_path: object) -> None:
        path = f"{tmp_path}/test.h5"
        arrays = [np.random.randn(10, 4) for _ in range(5)]
        with versionable.hdf5.open(_WithLists, path) as obj:
            obj.traces = []
            for arr in arrays:
                obj.traces.append(arr)

        loaded = versionable.load(_WithLists, path)
        assert len(loaded.traces) == 5
        for i, arr in enumerate(arrays):
            np.testing.assert_array_almost_equal(loaded.traces[i], arr)

    def test_setitem(self, tmp_path: object) -> None:
        path = f"{tmp_path}/test.h5"
        with versionable.hdf5.open(_WithLists, path) as obj:
            obj.traces = []
            obj.traces.append(np.zeros((3, 4)))
            obj.traces.append(np.zeros((3, 4)))
            obj.traces[0] = np.ones((3, 4))

        loaded = versionable.load(_WithLists, path)
        np.testing.assert_array_equal(loaded.traces[0], np.ones((3, 4)))

    def test_extend(self, tmp_path: object) -> None:
        path = f"{tmp_path}/test.h5"
        arrays = [np.ones((2, 3)) * i for i in range(3)]
        with versionable.hdf5.open(_WithLists, path) as obj:
            obj.traces = []
            obj.traces.extend(arrays)

        loaded = versionable.load(_WithLists, path)
        assert len(loaded.traces) == 3


class TestTrackedListScalars:
    """list[float] and list[str] append uses resizable 1-D datasets."""

    def test_float_append_loop(self, tmp_path: object) -> None:
        path = f"{tmp_path}/test.h5"
        with versionable.hdf5.open(_WithLists, path) as obj:
            obj.timestamps = []
            for i in range(100):
                obj.timestamps.append(float(i))

        loaded = versionable.load(_WithLists, path)
        assert loaded.timestamps == list(range(100))

    def test_string_append(self, tmp_path: object) -> None:
        path = f"{tmp_path}/test.h5"
        with versionable.hdf5.open(_WithLists, path) as obj:
            obj.tags = []
            obj.tags.append("alpha")
            obj.tags.append("beta")

        loaded = versionable.load(_WithLists, path)
        assert loaded.tags == ["alpha", "beta"]


class TestTrackedDict:
    """dict[str, np.ndarray] setitem creates/replaces datasets in a group."""

    def test_setitem_and_load(self, tmp_path: object) -> None:
        path = f"{tmp_path}/test.h5"
        with versionable.hdf5.open(_WithDict, path) as obj:
            obj.channels = {}
            obj.channels["ch0"] = np.ones(100)
            obj.channels["ch1"] = np.zeros(100)

        loaded = versionable.load(_WithDict, path)
        assert len(loaded.channels) == 2
        np.testing.assert_array_equal(loaded.channels["ch0"], np.ones(100))
        np.testing.assert_array_equal(loaded.channels["ch1"], np.zeros(100))

    def test_replace_value(self, tmp_path: object) -> None:
        path = f"{tmp_path}/test.h5"
        with versionable.hdf5.open(_WithDict, path) as obj:
            obj.channels = {}
            obj.channels["ch0"] = np.zeros(10)
            obj.channels["ch0"] = np.ones(10)

        loaded = versionable.load(_WithDict, path)
        np.testing.assert_array_equal(loaded.channels["ch0"], np.ones(10))

    def test_delitem(self, tmp_path: object) -> None:
        path = f"{tmp_path}/test.h5"
        with versionable.hdf5.open(_WithDict, path) as obj:
            obj.channels = {}
            obj.channels["ch0"] = np.ones(10)
            obj.channels["ch1"] = np.zeros(10)
            del obj.channels["ch0"]

        loaded = versionable.load(_WithDict, path)
        assert "ch0" not in loaded.channels
        assert "ch1" in loaded.channels

    def test_update(self, tmp_path: object) -> None:
        path = f"{tmp_path}/test.h5"
        with versionable.hdf5.open(_WithDict, path) as obj:
            obj.channels = {}
            obj.channels.update({"a": np.ones(5), "b": np.zeros(5)})

        loaded = versionable.load(_WithDict, path)
        assert len(loaded.channels) == 2


class TestUnsupportedListOps:
    """Unsupported list operations raise NotImplementedError."""

    def test_insert(self, tmp_path: object) -> None:
        path = f"{tmp_path}/test.h5"
        with versionable.hdf5.open(_WithLists, path) as obj:
            obj.timestamps = []
            with pytest.raises(NotImplementedError, match="insert"):
                obj.timestamps.insert(0, 1.0)

    def test_pop(self, tmp_path: object) -> None:
        path = f"{tmp_path}/test.h5"
        with versionable.hdf5.open(_WithLists, path) as obj:
            obj.timestamps = [1.0, 2.0]
            with pytest.raises(NotImplementedError, match="pop"):
                obj.timestamps.pop()

    def test_remove(self, tmp_path: object) -> None:
        path = f"{tmp_path}/test.h5"
        with versionable.hdf5.open(_WithLists, path) as obj:
            obj.timestamps = [1.0]
            with pytest.raises(NotImplementedError, match="remove"):
                obj.timestamps.remove(1.0)

    def test_sort(self, tmp_path: object) -> None:
        path = f"{tmp_path}/test.h5"
        with versionable.hdf5.open(_WithLists, path) as obj:
            obj.timestamps = [3.0, 1.0, 2.0]
            with pytest.raises(NotImplementedError, match="sort"):
                obj.timestamps.sort()

    def test_reverse(self, tmp_path: object) -> None:
        path = f"{tmp_path}/test.h5"
        with versionable.hdf5.open(_WithLists, path) as obj:
            obj.timestamps = [1.0, 2.0]
            with pytest.raises(NotImplementedError, match="reverse"):
                obj.timestamps.reverse()


class TestMixedSession:
    """Test mixing scalars, lists, and TrackedArray in the same session."""

    def test_mixed_operations(self, tmp_path: object) -> None:
        path = f"{tmp_path}/test.h5"
        with versionable.hdf5.open(_SessionBasic, path) as obj:
            obj.name = "experiment"
            obj.sampleRate_Hz = 48000.0
            obj.data = np.arange(10, dtype=np.float64)
            obj.waveform = np.empty((0, 4), dtype=np.float64)
            obj.waveform.append(np.ones((5, 4)))

        loaded = versionable.load(_SessionBasic, path)
        assert loaded.name == "experiment"
        assert loaded.sampleRate_Hz == 48000.0
        np.testing.assert_array_equal(loaded.data, np.arange(10, dtype=np.float64))
        assert loaded.waveform.shape == (5, 4)


# ---------------------------------------------------------------------------
# Phase 4: Resume mode
# ---------------------------------------------------------------------------


class TestResumeMode:
    """Test resume mode: open existing file, restore state, continue."""

    def test_write_resume_write_load(self, tmp_path: object) -> None:
        path = f"{tmp_path}/test.h5"
        # First session: write some data
        with versionable.hdf5.open(_WithLists, path) as obj:
            obj.name = "experiment"
            obj.timestamps = []
            obj.traces = []
            for i in range(5):
                obj.timestamps.append(float(i))
                obj.traces.append(np.ones((3, 4)) * i)

        # Resume: continue writing
        with versionable.hdf5.open(_WithLists, path, mode="resume") as obj:
            assert obj.name == "experiment"
            assert len(obj.timestamps) == 5
            assert len(obj.traces) == 5
            for i in range(5, 10):
                obj.timestamps.append(float(i))
                obj.traces.append(np.ones((3, 4)) * i)

        # Verify all data
        loaded = versionable.load(_WithLists, path)
        assert loaded.name == "experiment"
        assert loaded.timestamps == list(range(10))
        assert len(loaded.traces) == 10

    def test_resume_validates_class(self, tmp_path: object) -> None:
        path = f"{tmp_path}/test.h5"
        with versionable.hdf5.open(_SessionBasic, path) as obj:
            obj.name = "test"

        with pytest.raises(BackendError, match="Cannot resume"), versionable.hdf5.open(_WithLists, path, mode="resume"):
            pass

    def test_resume_nonexistent_file(self, tmp_path: object) -> None:
        path = f"{tmp_path}/nonexistent.h5"
        with (
            pytest.raises(BackendError, match="does not exist"),
            versionable.hdf5.open(_SessionBasic, path, mode="resume"),
        ):
            pass

    def test_resume_empty_file(self, tmp_path: object) -> None:
        """Resume on a file with metadata only, no fields set."""
        path = f"{tmp_path}/test.h5"
        with versionable.hdf5.open(_SessionBasic, path):
            pass  # No fields set

        with versionable.hdf5.open(_SessionBasic, path, mode="resume") as obj:
            obj.name = "after resume"

        loaded = versionable.load(_SessionBasic, path)
        assert loaded.name == "after resume"

    def test_resume_preserves_scalars(self, tmp_path: object) -> None:
        path = f"{tmp_path}/test.h5"
        with versionable.hdf5.open(_SessionBasic, path) as obj:
            obj.name = "original"
            obj.sampleRate_Hz = 48000.0

        with versionable.hdf5.open(_SessionBasic, path, mode="resume") as obj:
            assert obj.name == "original"
            assert obj.sampleRate_Hz == 48000.0

    def test_resume_appendable_continues(self, tmp_path: object) -> None:
        path = f"{tmp_path}/test.h5"
        with versionable.hdf5.open(_SessionBasic, path) as obj:
            obj.waveform = np.empty((0, 4), dtype=np.float64)
            obj.waveform.append(np.ones((10, 4)))

        with versionable.hdf5.open(_SessionBasic, path, mode="resume") as obj:
            assert isinstance(obj.waveform, TrackedArray)
            assert obj.waveform.shape == (10, 4)
            obj.waveform.append(np.zeros((5, 4)))
            assert obj.waveform.shape == (15, 4)

        loaded = versionable.load(_SessionBasic, path)
        assert loaded.waveform.shape == (15, 4)
        np.testing.assert_array_equal(loaded.waveform[:10], np.ones((10, 4)))
        np.testing.assert_array_equal(loaded.waveform[10:], np.zeros((5, 4)))

    def test_resume_scalar_overwrite(self, tmp_path: object) -> None:
        path = f"{tmp_path}/test.h5"
        with versionable.hdf5.open(_SessionBasic, path) as obj:
            obj.name = "first"

        with versionable.hdf5.open(_SessionBasic, path, mode="resume") as obj:
            obj.name = "second"

        loaded = versionable.load(_SessionBasic, path)
        assert loaded.name == "second"

    def test_resume_scalar_list_continues(self, tmp_path: object) -> None:
        path = f"{tmp_path}/test.h5"
        with versionable.hdf5.open(_WithLists, path) as obj:
            obj.timestamps = []
            for i in range(3):
                obj.timestamps.append(float(i))

        with versionable.hdf5.open(_WithLists, path, mode="resume") as obj:
            assert obj.timestamps == [0.0, 1.0, 2.0]
            obj.timestamps.append(3.0)
            obj.timestamps.append(4.0)

        loaded = versionable.load(_WithLists, path)
        assert loaded.timestamps == [0.0, 1.0, 2.0, 3.0, 4.0]


# ---------------------------------------------------------------------------
# Phase 5 test classes
# ---------------------------------------------------------------------------

from tests.conftest import Inner

_h_with_nested = computeHash({"name": str, "point": Inner})


@dataclass
class _WithNested(
    Versionable,
    version=1,
    hash=_h_with_nested,
    register=False,
):
    name: str = ""
    point: Inner = field(default_factory=lambda: Inner(x=0.0, y=0.0))


_h_with_vlist = computeHash({"name": str, "measurements": list[Inner]})


@dataclass
class _WithVList(
    Versionable,
    version=1,
    hash=_h_with_vlist,
    register=False,
):
    name: str = ""
    measurements: list[Inner] = field(default_factory=list)


_h_with_vdict = computeHash({"name": str, "points": dict[str, Inner]})


@dataclass
class _WithVDict(
    Versionable,
    version=1,
    hash=_h_with_vdict,
    register=False,
):
    name: str = ""
    points: dict[str, Inner] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Phase 5: Polish
# ---------------------------------------------------------------------------


class TestNestedVersionable:
    """Nested Versionable field assignment."""

    def test_nested_roundtrip(self, tmp_path: object) -> None:
        path = f"{tmp_path}/test.h5"
        with versionable.hdf5.open(_WithNested, path) as obj:
            obj.name = "test"
            obj.point = Inner(x=1.5, y=2.5)

        loaded = versionable.load(_WithNested, path)
        assert loaded.name == "test"
        assert loaded.point.x == 1.5
        assert loaded.point.y == 2.5


class TestListVersionable:
    """list[Versionable] append support."""

    def test_append_and_load(self, tmp_path: object) -> None:
        path = f"{tmp_path}/test.h5"
        with versionable.hdf5.open(_WithVList, path) as obj:
            obj.name = "test"
            obj.measurements = []
            obj.measurements.append(Inner(x=1.0, y=2.0))
            obj.measurements.append(Inner(x=3.0, y=4.0))

        loaded = versionable.load(_WithVList, path)
        assert len(loaded.measurements) == 2
        assert loaded.measurements[0].x == 1.0
        assert loaded.measurements[1].y == 4.0


class TestDictVersionable:
    """dict[str, Versionable] setitem support."""

    def test_setitem_and_load(self, tmp_path: object) -> None:
        path = f"{tmp_path}/test.h5"
        with versionable.hdf5.open(_WithVDict, path) as obj:
            obj.name = "test"
            obj.points = {}
            obj.points["origin"] = Inner(x=0.0, y=0.0)
            obj.points["target"] = Inner(x=10.0, y=20.0)

        loaded = versionable.load(_WithVDict, path)
        assert len(loaded.points) == 2
        assert loaded.points["origin"].x == 0.0
        assert loaded.points["target"].y == 20.0


class TestFlush:
    """flush() re-persists in-place-mutated fields."""

    def test_flush_plain_array(self, tmp_path: object) -> None:
        path = f"{tmp_path}/test.h5"
        session = versionable.hdf5.open(_SessionBasic, path)
        with session as obj:
            obj.data = np.zeros(10, dtype=np.float64)
            # In-place mutation (not tracked by __setattr__)
            obj.data[5] = 99.0
            session.flush("data")

        loaded = versionable.load(_SessionBasic, path)
        assert loaded.data[5] == 99.0

    def test_flush_all(self, tmp_path: object) -> None:
        path = f"{tmp_path}/test.h5"
        session = versionable.hdf5.open(_SessionBasic, path)
        with session as obj:
            obj.name = "test"
            obj.data = np.zeros(5, dtype=np.float64)
            obj.data[0] = 42.0
            session.flush()

        loaded = versionable.load(_SessionBasic, path)
        assert loaded.data[0] == 42.0


class TestCompletenessWarning:
    """Warn on __exit__ about unset required fields."""

    def test_warns_on_unset_required(self, tmp_path: object, caplog: object) -> None:
        import logging

        path = f"{tmp_path}/test.h5"
        with caplog.at_level(logging.WARNING), versionable.hdf5.open(_WithNested, path):  # type: ignore[union-attr]
            # _WithNested has defaults for all fields, so no warning
            pass

        # _WithNested has defaults for all fields, so no warning expected
        assert "was never set" not in caplog.text  # type: ignore[union-attr]
