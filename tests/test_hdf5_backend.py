# ruff: noqa: E402 — module-level imports must come after pytest.importorskip("h5py")
# so the entire file is skipped when h5py is not installed.
"""Tests for the HDF5 backend with native type mapping."""

# Hash validation is tested separately in test_hash.py and test_base.py. Classes here
# omit the hash parameter to keep declarations concise.

from __future__ import annotations

import pytest

h5py = pytest.importorskip("h5py")

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt

import versionable
from versionable import Versionable
from versionable.errors import ArrayNotLoadedError, BackendError
from versionable.hdf5 import (
    UNCOMPRESSED,
    ZSTD_DEFAULT,
    Hdf5Compression,
)

from .conftest import Inner, SimpleConfig, WithNested


@dataclass
class WithArray(
    Versionable,
    version=1,
    hash="c0dc53",
    register=False,
):
    name: str
    data: npt.NDArray[np.float64]


# Module-level classes for tests that need type resolution across nested Versionables.
# Defining inside test functions causes forward-reference resolution failures with
# `from __future__ import annotations`.


@dataclass
class _Measurement(Versionable, version=1, name="Measurement", register=False):
    label: str
    data: npt.NDArray[np.float64]


@dataclass
class _WithTraces(Versionable, version=1, register=False):
    name: str
    traces: list[npt.NDArray[np.float64]]


@dataclass
class _WithChannels(Versionable, version=1, register=False):
    name: str
    channels: dict[str, npt.NDArray[np.float64]]


@dataclass
class _Experiment(Versionable, version=1, register=False):
    name: str
    measurements: list[_Measurement]


# Deeply nested: Sensor has list[ndarray], Lab has dict[str, Sensor]
@dataclass
class _Sensor(Versionable, version=1, name="Sensor", register=False):
    name: str
    traces: list[npt.NDArray[np.float64]]


@dataclass
class _Lab(Versionable, version=1, register=False):
    title: str
    sensors: dict[str, _Sensor]


class TestHdf5RoundTrip:
    def test_simpleScalars(self, tmp_path: Path) -> None:
        obj = SimpleConfig(name="test", debug=True, retries=5)
        p = tmp_path / "config.h5"
        versionable.save(obj, p)
        loaded = versionable.load(SimpleConfig, p)
        assert loaded.name == "test"
        assert loaded.debug is True
        assert loaded.retries == 5

    def test_largeArray(self, tmp_path: Path) -> None:
        rng = np.random.default_rng(42)
        arr = rng.random(10000).astype(np.float64)
        obj = WithArray(name="big", data=arr)
        p = tmp_path / "big.h5"
        versionable.save(obj, p)
        loaded = versionable.load(WithArray, p, preload="*")
        np.testing.assert_array_equal(loaded.data, arr)


class TestHdf5Metadata:
    def test_metadataInFile(self, tmp_path: Path) -> None:
        obj = SimpleConfig(name="test")
        p = tmp_path / "out.h5"
        versionable.save(obj, p)

        with h5py.File(p, "r") as f:
            meta = f["__versionable__"]
            assert meta.attrs["__OBJECT__"] == "SimpleConfig"
            assert meta.attrs["__VERSION__"] == 1


class TestHdf5Compression:
    """Compression algorithm round-trip and verification tests."""

    def _saveAndCheck(
        self,
        tmp_path: Path,
        comp: Hdf5Compression,
        filename: str = "out.h5",
    ) -> Path:
        arr = np.zeros(1000, dtype=np.float64)
        obj = WithArray(name="zeros", data=arr)
        p = tmp_path / filename
        versionable.save(obj, p, compression=comp)

        # Verify round-trip
        loaded = versionable.load(WithArray, p, preload="*")
        np.testing.assert_array_equal(loaded.data, arr)
        return p

    def test_zstdDefault(self, tmp_path: Path) -> None:
        p = self._saveAndCheck(tmp_path, ZSTD_DEFAULT)
        with h5py.File(p, "r") as f:
            ds = f["data"]
            assert isinstance(ds, h5py.Dataset)
            assert ds.compression is not None

    def test_uncompressed(self, tmp_path: Path) -> None:
        p = self._saveAndCheck(tmp_path, UNCOMPRESSED)
        with h5py.File(p, "r") as f:
            ds = f["data"]
            assert isinstance(ds, h5py.Dataset)
            assert ds.compression is None

    def test_defaultCompressionIsGzip(self, tmp_path: Path) -> None:
        """save() without explicit compression should use gzip."""
        arr = np.zeros(1000, dtype=np.float64)
        obj = WithArray(name="zeros", data=arr)
        p = tmp_path / "default.h5"
        versionable.save(obj, p)

        with h5py.File(p, "r") as f:
            ds = f["data"]
            assert isinstance(ds, h5py.Dataset)
            assert ds.compression == "gzip"
            assert ds.compression_opts == 4


class TestLazyLoading:
    def test_lazyByDefault(self, tmp_path: Path) -> None:
        """Array fields are lazy by default (not preloaded)."""
        arr = np.array([1.0, 2.0, 3.0])
        obj = WithArray(name="lazy", data=arr)
        p = tmp_path / "lazy.h5"
        versionable.save(obj, p)

        loaded = versionable.load(WithArray, p)
        assert loaded.name == "lazy"
        # Accessing the array triggers lazy load
        np.testing.assert_array_equal(loaded.data, arr)

    def test_preloadSpecific(self, tmp_path: Path) -> None:
        @dataclass
        class TwoArrays(Versionable, version=1, register=False):
            name: str
            a: npt.NDArray[np.float64]
            b: npt.NDArray[np.float64]

        obj = TwoArrays(name="test", a=np.array([1.0]), b=np.array([2.0]))
        p = tmp_path / "two.h5"
        versionable.save(obj, p)

        loaded = versionable.load(TwoArrays, p, preload=["a"])
        # 'a' is eagerly loaded, 'b' is lazy
        np.testing.assert_array_equal(loaded.a, np.array([1.0]))
        np.testing.assert_array_equal(loaded.b, np.array([2.0]))  # Triggers lazy load

    def test_preloadAll(self, tmp_path: Path) -> None:
        arr = np.array([1.0, 2.0])
        obj = WithArray(name="all", data=arr)
        p = tmp_path / "all.h5"
        versionable.save(obj, p)

        loaded = versionable.load(WithArray, p, preload="*")
        np.testing.assert_array_equal(loaded.data, arr)


class TestMetadataOnly:
    def test_metadataOnlyMode(self, tmp_path: Path) -> None:
        arr = np.array([1.0, 2.0, 3.0])
        obj = WithArray(name="meta", data=arr)
        p = tmp_path / "meta.h5"
        versionable.save(obj, p)

        loaded = versionable.load(WithArray, p, metadataOnly=True)
        assert loaded.name == "meta"

        with pytest.raises(ArrayNotLoadedError):
            _ = loaded.data

    def test_metadataOnlyScalarsAccessible(self, tmp_path: Path) -> None:
        obj = SimpleConfig(name="test", debug=True, retries=7)
        p = tmp_path / "scalars.h5"
        versionable.save(obj, p)

        loaded = versionable.load(SimpleConfig, p, metadataOnly=True)
        assert loaded.name == "test"
        assert loaded.debug is True
        assert loaded.retries == 7


class TestHdf5Errors:
    def test_missingFile(self, tmp_path: Path) -> None:
        with pytest.raises(BackendError):
            versionable.load(SimpleConfig, tmp_path / "nonexistent.h5")

    def test_futureFormatRaises(self, tmp_path: Path) -> None:
        p = tmp_path / "future.h5"
        with h5py.File(p, "w") as f:
            meta = f.create_group("__versionable__")
            meta.attrs["__OBJECT__"] = "SimpleConfig"
            meta.attrs["__VERSION__"] = 1
            meta.attrs["__HASH__"] = ""
            meta.attrs["__FORMAT__"] = 2

        with pytest.raises(BackendError, match="Upgrade versionable"):
            versionable.load(SimpleConfig, p)

    def test_unregisteredClassInLoad(self, tmp_path: Path) -> None:
        """load() raises when the file's __OBJECT__ isn't in the registry."""
        from versionable._hdf5_backend import Hdf5Backend

        # Write a file with a class name that won't be in the registry
        p = tmp_path / "unknown.h5"
        with h5py.File(p, "w") as f:
            meta = f.create_group("__versionable__")
            meta.attrs["__OBJECT__"] = "NonexistentClass"
            meta.attrs["__VERSION__"] = 1
            meta.attrs["__HASH__"] = "abc123"

        be = Hdf5Backend()
        with pytest.raises(BackendError, match="Unknown Versionable type"):
            be.load(p)

    def test_unregisteredClassInLoadLazy(self, tmp_path: Path) -> None:
        """loadLazy() raises when cls is None and __OBJECT__ isn't registered."""
        from versionable._hdf5_backend import Hdf5Backend

        p = tmp_path / "unknown_lazy.h5"
        with h5py.File(p, "w") as f:
            meta = f.create_group("__versionable__")
            meta.attrs["__OBJECT__"] = "NonexistentClass"
            meta.attrs["__VERSION__"] = 1
            meta.attrs["__HASH__"] = "abc123"

        be = Hdf5Backend()
        with pytest.raises(BackendError, match="Unknown Versionable type"):
            be.loadLazy(p)

    def test_dictKeyWithSlash(self, tmp_path: Path) -> None:
        """Dict keys containing '/' roundtrip correctly (percent-encoded in HDF5)."""

        @dataclass
        class WithSlashKey(Versionable, version=1, register=False):
            data: dict[str, int]

        obj = WithSlashKey(data={"valid": 1, "path/key": 2, "a/b/c": 3})
        p = tmp_path / "slash.h5"
        versionable.save(obj, p)
        loaded = versionable.load(WithSlashKey, p)
        assert loaded.data == {"valid": 1, "path/key": 2, "a/b/c": 3}

    def test_dictKeyWithLiteralPercent(self, tmp_path: Path) -> None:
        """Dict keys containing literal '%2F' don't collide with escaped '/'."""

        @dataclass
        class WithPercentKey(Versionable, version=1, register=False):
            data: dict[str, int]

        obj = WithPercentKey(data={"%2F": 1, "/": 2, "normal": 3})
        p = tmp_path / "percent.h5"
        versionable.save(obj, p)
        loaded = versionable.load(WithPercentKey, p)
        assert loaded.data == {"%2F": 1, "/": 2, "normal": 3}

    def test_dictKeyDot(self, tmp_path: Path) -> None:
        """Dict key '.' (HDF5 current-group alias) roundtrips correctly."""

        @dataclass
        class WithDotKey(Versionable, version=1, register=False):
            data: dict[str, int]

        obj = WithDotKey(data={".": 1, "..": 2, ".hidden": 3})
        p = tmp_path / "dot.h5"
        versionable.save(obj, p)
        loaded = versionable.load(WithDotKey, p)
        assert loaded.data == {".": 1, "..": 2, ".hidden": 3}


# ---------------------------------------------------------------------------
# Native type mapping tests
# ---------------------------------------------------------------------------


class TestNativeScalars:
    """Test that scalar fields are stored as native HDF5 attributes."""

    def test_nativeAttributeStorage(self, tmp_path: Path) -> None:
        """Verify scalars are HDF5 attributes, not JSON blobs."""
        obj = SimpleConfig(name="test", debug=True, retries=5)
        p = tmp_path / "native.h5"
        versionable.save(obj, p)

        with h5py.File(p, "r") as f:
            # Scalars should be native attributes, not in __scalars__
            assert "__scalars__" not in f.attrs
            assert f.attrs["name"] == "test"
            assert bool(f.attrs["debug"]) is True
            assert f.attrs["retries"] == 5


class TestNativeNone:
    """Test None handling with h5py.Empty."""

    def test_noneField(self, tmp_path: Path) -> None:
        @dataclass
        class WithOptional(Versionable, version=1, register=False):
            name: str
            label: str | None = None

        obj = WithOptional(name="test", label=None)
        p = tmp_path / "none.h5"
        versionable.save(obj, p)
        loaded = versionable.load(WithOptional, p)
        assert loaded.name == "test"
        assert loaded.label is None


class TestNativeListDatasets:
    """Test list[numeric], list[str], list[bool] as 1-D datasets."""

    @pytest.mark.parametrize(
        "values",
        [
            [1.0, 2.5, 3.7],
            [10, 20, 30],
            [True, False, True],
        ],
    )
    def test_numericList(self, tmp_path: Path, values: list[object]) -> None:
        @dataclass
        class WithNumericList(Versionable, version=1, register=False):
            name: str
            values: list[float] | list[int] | list[bool]

        obj = WithNumericList(name="test", values=values)
        p = tmp_path / "list.h5"
        versionable.save(obj, p)
        loaded = versionable.load(WithNumericList, p)
        assert loaded.values == values
        assert isinstance(loaded.values, list)

    def test_listStr(self, tmp_path: Path) -> None:
        @dataclass
        class WithListStr(Versionable, version=1, register=False):
            name: str
            tags: list[str]

        obj = WithListStr(name="test", tags=["alpha", "beta", "gamma"])
        p = tmp_path / "list_str.h5"
        versionable.save(obj, p)
        loaded = versionable.load(WithListStr, p)
        assert loaded.tags == ["alpha", "beta", "gamma"]

    def test_emptyList(self, tmp_path: Path) -> None:
        @dataclass
        class WithEmpty(Versionable, version=1, register=False):
            name: str
            values: list[float]

        obj = WithEmpty(name="test", values=[])
        p = tmp_path / "empty_list.h5"
        versionable.save(obj, p)
        loaded = versionable.load(WithEmpty, p)
        assert loaded.values == []


class TestNativeArrayCollections:
    """Test list[np.ndarray] and dict[str, np.ndarray] as groups."""

    def test_listNdarray(self, tmp_path: Path) -> None:
        @dataclass
        class WithTraces(Versionable, version=1, register=False):
            name: str
            traces: list[npt.NDArray[np.float64]]

        traces = [np.array([1.0, 2.0]), np.array([3.0, 4.0, 5.0])]
        obj = WithTraces(name="test", traces=traces)
        p = tmp_path / "list_ndarray.h5"
        versionable.save(obj, p)
        loaded = versionable.load(WithTraces, p, preload="*")
        assert len(loaded.traces) == 2
        np.testing.assert_array_equal(loaded.traces[0], traces[0])
        np.testing.assert_array_equal(loaded.traces[1], traces[1])

    def test_listNdarrayVaryingShapes(self, tmp_path: Path) -> None:
        @dataclass
        class VaryingShapes(Versionable, version=1, register=False):
            name: str
            data: list[npt.NDArray[np.float64]]

        arrays = [np.zeros((2, 3)), np.ones((4,)), np.array([[1, 2], [3, 4], [5, 6]])]
        obj = VaryingShapes(name="test", data=arrays)
        p = tmp_path / "varying.h5"
        versionable.save(obj, p)
        loaded = versionable.load(VaryingShapes, p, preload="*")
        for orig, loaded_arr in zip(arrays, loaded.data, strict=True):
            np.testing.assert_array_equal(loaded_arr, orig)

    def test_dictNdarray(self, tmp_path: Path) -> None:
        @dataclass
        class WithChannels(Versionable, version=1, register=False):
            name: str
            channels: dict[str, npt.NDArray[np.float64]]

        channels = {"ch0": np.array([1.0, 2.0]), "ch1": np.array([3.0, 4.0])}
        obj = WithChannels(name="test", channels=channels)
        p = tmp_path / "dict_ndarray.h5"
        versionable.save(obj, p)
        loaded = versionable.load(WithChannels, p, preload="*")
        assert set(loaded.channels.keys()) == {"ch0", "ch1"}
        np.testing.assert_array_equal(loaded.channels["ch0"], channels["ch0"])
        np.testing.assert_array_equal(loaded.channels["ch1"], channels["ch1"])

    def test_listNdarrayFileLayout(self, tmp_path: Path) -> None:
        """Verify list[ndarray] produces integer-keyed datasets in a group."""

        @dataclass
        class Traces(Versionable, version=1, register=False):
            name: str
            traces: list[npt.NDArray[np.float64]]

        obj = Traces(name="test", traces=[np.array([1.0]), np.array([2.0])])
        p = tmp_path / "layout.h5"
        versionable.save(obj, p)

        with h5py.File(p, "r") as f:
            assert "traces" in f
            tracesGroup = f["traces"]
            assert isinstance(tracesGroup, h5py.Group)
            assert sorted(tracesGroup.keys()) == ["0", "1"]

    def test_emptyListNdarray(self, tmp_path: Path) -> None:
        obj = _WithTraces(name="test", traces=[])
        p = tmp_path / "empty_traces.h5"
        versionable.save(obj, p)
        loaded = versionable.load(_WithTraces, p, preload="*")
        assert loaded.traces == []

    def test_emptyDictNdarray(self, tmp_path: Path) -> None:
        obj = _WithChannels(name="test", channels={})
        p = tmp_path / "empty_channels.h5"
        versionable.save(obj, p)
        loaded = versionable.load(_WithChannels, p, preload="*")
        assert loaded.channels == {}


class TestNativeNestedVersionable:
    """Test nested Versionable and list[Versionable]."""

    def test_nestedFileLayout(self, tmp_path: Path) -> None:
        """Verify nested Versionable uses __versionable__ child group."""
        obj = WithNested(name="origin", point=Inner(x=1.0, y=2.0))
        p = tmp_path / "nested_layout.h5"
        versionable.save(obj, p)

        with h5py.File(p, "r") as f:
            # Root has __versionable__
            assert "__versionable__" in f
            # Nested 'point' group has its own __versionable__
            assert "point" in f
            point = f["point"]
            assert isinstance(point, h5py.Group)
            assert "__versionable__" in point
            assert point["__versionable__"].attrs["__OBJECT__"] == "Inner"

    def test_listVersionable(self, tmp_path: Path) -> None:
        @dataclass
        class WithPointList(Versionable, version=1, register=False):
            name: str
            points: list[Inner]

        points = [Inner(x=1.0, y=2.0), Inner(x=3.0, y=4.0)]
        obj = WithPointList(name="test", points=points)
        p = tmp_path / "list_versionable.h5"
        versionable.save(obj, p)
        loaded = versionable.load(WithPointList, p)
        assert len(loaded.points) == 2
        assert loaded.points[0].x == 1.0
        assert loaded.points[0].y == 2.0
        assert loaded.points[1].x == 3.0
        assert loaded.points[1].y == 4.0

    def test_nestedWithArrays(self, tmp_path: Path) -> None:
        """Versionable containing array collections inside nested Versionable."""
        measurements = [
            _Measurement(label="a", data=np.array([1.0, 2.0])),
            _Measurement(label="b", data=np.array([3.0, 4.0, 5.0])),
        ]
        obj = _Experiment(name="exp1", measurements=measurements)
        p = tmp_path / "nested_arrays.h5"
        versionable.save(obj, p)
        loaded = versionable.load(_Experiment, p, preload="*")
        assert len(loaded.measurements) == 2
        assert loaded.measurements[0].label == "a"
        np.testing.assert_array_equal(loaded.measurements[0].data, np.array([1.0, 2.0]))
        assert loaded.measurements[1].label == "b"
        np.testing.assert_array_equal(loaded.measurements[1].data, np.array([3.0, 4.0, 5.0]))


class TestNativeNoJson:
    """Verify that no JSON appears anywhere in the file."""

    def test_noJsonInFile(self, tmp_path: Path) -> None:
        obj = WithNested(name="test", point=Inner(x=1.0, y=2.0))
        p = tmp_path / "no_json.h5"
        versionable.save(obj, p)

        with h5py.File(p, "r") as f:
            assert "__scalars__" not in f.attrs
            _assertNoJsonAttrs(f)


class TestLazyArrayCollections:
    """Test per-element lazy loading for list[ndarray] and dict[str, ndarray]."""

    def test_listNdarrayLazy(self, tmp_path: Path) -> None:
        """list[ndarray] is lazily loaded per-element by default."""
        from versionable._lazy import LazyArrayList

        traces = [np.array([1.0, 2.0]), np.array([3.0, 4.0, 5.0])]
        obj = _WithTraces(name="test", traces=traces)
        p = tmp_path / "lazy_list.h5"
        versionable.save(obj, p)

        loaded = versionable.load(_WithTraces, p)
        assert isinstance(loaded.traces, LazyArrayList)
        assert len(loaded.traces) == 2
        # Accessing an element loads it
        np.testing.assert_array_equal(loaded.traces[0], traces[0])
        np.testing.assert_array_equal(loaded.traces[1], traces[1])

    def test_listNdarrayPreload(self, tmp_path: Path) -> None:
        """list[ndarray] with preload returns a regular list."""
        traces = [np.array([1.0, 2.0]), np.array([3.0, 4.0, 5.0])]
        obj = _WithTraces(name="test", traces=traces)
        p = tmp_path / "preload_list.h5"
        versionable.save(obj, p)

        loaded = versionable.load(_WithTraces, p, preload=["traces"])
        assert isinstance(loaded.traces, list)
        assert len(loaded.traces) == 2
        np.testing.assert_array_equal(loaded.traces[0], traces[0])

    def test_listNdarrayMetadataOnly(self, tmp_path: Path) -> None:
        """list[ndarray] with metadataOnly raises on access."""
        traces = [np.array([1.0])]
        obj = _WithTraces(name="test", traces=traces)
        p = tmp_path / "meta_list.h5"
        versionable.save(obj, p)

        loaded = versionable.load(_WithTraces, p, metadataOnly=True)
        assert loaded.name == "test"
        with pytest.raises(ArrayNotLoadedError):
            _ = loaded.traces

    def test_dictNdarrayLazy(self, tmp_path: Path) -> None:
        """dict[str, ndarray] is lazily loaded per-element by default."""
        from versionable._lazy import LazyArrayDict

        channels = {"ch0": np.array([1.0, 2.0]), "ch1": np.array([3.0, 4.0])}
        obj = _WithChannels(name="test", channels=channels)
        p = tmp_path / "lazy_dict.h5"
        versionable.save(obj, p)

        loaded = versionable.load(_WithChannels, p)
        assert isinstance(loaded.channels, LazyArrayDict)
        assert len(loaded.channels) == 2
        assert "ch0" in loaded.channels
        np.testing.assert_array_equal(loaded.channels["ch0"], channels["ch0"])
        np.testing.assert_array_equal(loaded.channels["ch1"], channels["ch1"])

    def test_dictNdarrayPreload(self, tmp_path: Path) -> None:
        """dict[str, ndarray] with preload returns a regular dict."""
        channels = {"ch0": np.array([1.0, 2.0]), "ch1": np.array([3.0, 4.0])}
        obj = _WithChannels(name="test", channels=channels)
        p = tmp_path / "preload_dict.h5"
        versionable.save(obj, p)

        loaded = versionable.load(_WithChannels, p, preload=["channels"])
        assert isinstance(loaded.channels, dict)
        np.testing.assert_array_equal(loaded.channels["ch0"], channels["ch0"])

    def test_lazyListIteration(self, tmp_path: Path) -> None:
        """LazyArrayList supports iteration."""
        traces = [np.array([float(i)]) for i in range(5)]
        obj = _WithTraces(name="test", traces=traces)
        p = tmp_path / "iter_list.h5"
        versionable.save(obj, p)

        loaded = versionable.load(_WithTraces, p)
        for i, arr in enumerate(loaded.traces):
            np.testing.assert_array_equal(arr, traces[i])

    def test_lazyListSlice(self, tmp_path: Path) -> None:
        """LazyArrayList supports slicing."""
        traces = [np.array([float(i)]) for i in range(5)]
        obj = _WithTraces(name="test", traces=traces)
        p = tmp_path / "slice_list.h5"
        versionable.save(obj, p)

        loaded = versionable.load(_WithTraces, p)
        sliced = loaded.traces[1:3]
        assert len(sliced) == 2
        np.testing.assert_array_equal(sliced[0], traces[1])
        np.testing.assert_array_equal(sliced[1], traces[2])

    def test_lazyDictKeys(self, tmp_path: Path) -> None:
        """LazyArrayDict supports keys(), values(), items()."""
        channels = {"ch0": np.array([1.0]), "ch1": np.array([2.0])}
        obj = _WithChannels(name="test", channels=channels)
        p = tmp_path / "dict_keys.h5"
        versionable.save(obj, p)

        loaded = versionable.load(_WithChannels, p)
        assert set(loaded.channels.keys()) == {"ch0", "ch1"}

    def test_lazyDictWithEncodedKeys(self, tmp_path: Path) -> None:
        """LazyArrayDict decodes percent-encoded keys and loads on access."""
        from versionable._lazy import LazyArrayDict

        @dataclass
        class WithSlashChannels(Versionable, version=1, register=False):
            data: dict[str, npt.NDArray[np.float64]]

        obj = WithSlashChannels(data={"a/b": np.array([1.0]), "c": np.array([2.0])})
        p = tmp_path / "lazy_encoded.h5"
        versionable.save(obj, p)

        loaded = versionable.load(WithSlashChannels, p)
        raw = object.__getattribute__(loaded, "data")
        assert isinstance(raw, LazyArrayDict)

        # Keys should be decoded before any data is loaded
        assert "a/b" in raw
        assert "c" in raw
        assert len(raw._cache) == 0  # noqa: SLF001 — testing lazy internals

        # Accessing a key loads only that element
        np.testing.assert_array_equal(loaded.data["a/b"], np.array([1.0]))
        assert len(raw._cache) == 1  # noqa: SLF001

        np.testing.assert_array_equal(loaded.data["c"], np.array([2.0]))
        assert len(raw._cache) == 2  # noqa: SLF001

    def test_nestedVersionableLazyArrays(self, tmp_path: Path) -> None:
        """Arrays inside nested Versionables are lazily loaded."""
        from versionable._lazy import LazyArray

        measurements = [
            _Measurement(label="a", data=np.array([1.0, 2.0])),
            _Measurement(label="b", data=np.array([3.0, 4.0, 5.0])),
        ]
        obj = _Experiment(name="exp", measurements=measurements)
        p = tmp_path / "nested_lazy.h5"
        versionable.save(obj, p)

        loaded = versionable.load(_Experiment, p)

        # Scalars are eagerly available
        assert loaded.name == "exp"
        assert loaded.measurements[0].label == "a"
        assert loaded.measurements[1].label == "b"

        # Arrays inside nested Versionables should be lazy (not loaded yet)
        rawData0 = object.__getattribute__(loaded.measurements[0], "data")
        assert isinstance(rawData0, LazyArray), f"expected LazyArray, got {type(rawData0).__name__}"

        # Accessing the array triggers lazy load and returns correct data
        np.testing.assert_array_equal(loaded.measurements[0].data, np.array([1.0, 2.0]))
        np.testing.assert_array_equal(loaded.measurements[1].data, np.array([3.0, 4.0, 5.0]))

        # After access, it's cached as ndarray
        rawData0After = object.__getattribute__(loaded.measurements[0], "data")
        assert isinstance(rawData0After, np.ndarray)

    def test_nestedVersionablePreloadAll(self, tmp_path: Path) -> None:
        """preload='*' eagerly loads arrays inside nested Versionables."""
        measurements = [
            _Measurement(label="a", data=np.array([1.0, 2.0])),
        ]
        obj = _Experiment(name="exp", measurements=measurements)
        p = tmp_path / "nested_preload.h5"
        versionable.save(obj, p)

        loaded = versionable.load(_Experiment, p, preload="*")

        # With preload="*", arrays should be eagerly loaded
        rawData0 = object.__getattribute__(loaded.measurements[0], "data")
        assert isinstance(rawData0, np.ndarray), f"expected ndarray, got {type(rawData0).__name__}"
        np.testing.assert_array_equal(loaded.measurements[0].data, np.array([1.0, 2.0]))

    def test_deeplyNestedDictVersionableLazy(self, tmp_path: Path) -> None:
        """dict[str, Versionable] where each Versionable has list[ndarray] — all lazy."""
        from versionable._lazy import LazyArrayList

        sensors = {
            "accel": _Sensor(name="accel", traces=[np.array([1.0, 2.0]), np.array([3.0])]),
            "gyro": _Sensor(name="gyro", traces=[np.array([4.0, 5.0, 6.0])]),
        }
        obj = _Lab(title="lab-1", sensors=sensors)
        p = tmp_path / "deep_lazy.h5"
        versionable.save(obj, p)

        loaded = versionable.load(_Lab, p)

        # Top-level scalar is eager
        assert loaded.title == "lab-1"

        # dict[str, Versionable] — sensors are eagerly reconstructed (not lazy)
        assert isinstance(loaded.sensors, dict)
        assert set(loaded.sensors.keys()) == {"accel", "gyro"}

        # Nested Versionable scalars are eager
        assert loaded.sensors["accel"].name == "accel"
        assert loaded.sensors["gyro"].name == "gyro"

        # list[ndarray] inside each nested Versionable should be lazy
        rawTraces = object.__getattribute__(loaded.sensors["accel"], "traces")
        assert isinstance(rawTraces, LazyArrayList), f"expected LazyArrayList, got {type(rawTraces).__name__}"
        assert len(rawTraces._cache) == 0  # noqa: SLF001 — testing lazy internals

        # Accessing individual elements triggers lazy load
        np.testing.assert_array_equal(loaded.sensors["accel"].traces[0], np.array([1.0, 2.0]))
        assert len(rawTraces._cache) == 1  # noqa: SLF001

        np.testing.assert_array_equal(loaded.sensors["accel"].traces[1], np.array([3.0]))
        np.testing.assert_array_equal(loaded.sensors["gyro"].traces[0], np.array([4.0, 5.0, 6.0]))

    def test_deeplyNestedDictVersionablePreload(self, tmp_path: Path) -> None:
        """dict[str, Versionable] with preload='*' eagerly loads everything."""
        sensors = {
            "accel": _Sensor(name="accel", traces=[np.array([1.0, 2.0])]),
        }
        obj = _Lab(title="lab-1", sensors=sensors)
        p = tmp_path / "deep_preload.h5"
        versionable.save(obj, p)

        loaded = versionable.load(_Lab, p, preload="*")

        # With preload="*", traces should be a regular list (not LazyArrayList)
        assert isinstance(loaded.sensors["accel"].traces, list)
        rawTraces = object.__getattribute__(loaded.sensors["accel"], "traces")
        assert isinstance(rawTraces, list), f"expected list, got {type(rawTraces).__name__}"
        np.testing.assert_array_equal(loaded.sensors["accel"].traces[0], np.array([1.0, 2.0]))


class TestDtypeCoercion:
    """Verify that save/load preserves the original array dtype."""

    @pytest.mark.parametrize(
        ("srcDtype", "expectedDtype"),
        [
            (np.float64, np.float64),
            (np.float32, np.float32),
            (np.int32, np.int32),
            (np.uint16, np.uint16),
            (np.uint32, np.uint32),
        ],
    )
    def test_dtypePreserved(
        self,
        tmp_path: Path,
        srcDtype: type[np.generic],
        expectedDtype: type[np.generic],
    ) -> None:
        @dataclass
        class DtypeData(Versionable, version=1, register=False):
            data: npt.NDArray[np.generic]

        arr = np.arange(10, dtype=srcDtype)
        obj = DtypeData(data=arr)
        p = tmp_path / "dtype.h5"
        versionable.save(obj, p)
        loaded = versionable.load(DtypeData, p, preload="*")
        np.testing.assert_array_equal(loaded.data, arr)
        assert loaded.data.dtype == expectedDtype


class TestSkipDefaults:
    """skip_defaults=True omits default-valued fields from HDF5."""

    def test_defaultsOmitted(self, tmp_path: Path) -> None:
        @dataclass
        class WithDefaults(Versionable, version=1, skip_defaults=True, register=False):
            name: str
            count: int = 0
            tag: str = "default"

        obj = WithDefaults(name="test")
        p = tmp_path / "skip.h5"
        versionable.save(obj, p)

        with h5py.File(p, "r") as f:
            assert f.attrs["name"] == "test"
            assert "count" not in f.attrs
            assert "tag" not in f.attrs

        loaded = versionable.load(WithDefaults, p)
        assert loaded.name == "test"
        assert loaded.count == 0
        assert loaded.tag == "default"


class TestUnknownFieldHandling:
    """unknown='ignore' and unknown='error' modes for HDF5."""

    def test_ignoreUnknownFields(self, tmp_path: Path) -> None:
        @dataclass
        class V2(Versionable, version=1, name="UnkIgnore", register=True, unknown="ignore"):
            name: str

        # Write a file with an extra field
        p = tmp_path / "extra.h5"
        with h5py.File(p, "w") as f:
            meta = f.create_group("__versionable__")
            meta.attrs["__OBJECT__"] = "UnkIgnore"
            meta.attrs["__VERSION__"] = 1
            meta.attrs["__HASH__"] = ""
            f.attrs["name"] = "test"
            f.attrs["obsolete"] = 42

        loaded = versionable.load(V2, p)
        assert loaded.name == "test"
        assert not hasattr(loaded, "obsolete")

    def test_errorOnUnknownFields(self, tmp_path: Path) -> None:
        from versionable.errors import UnknownFieldError

        @dataclass
        class V2Strict(Versionable, version=1, name="UnkError", register=True, unknown="error"):
            name: str

        p = tmp_path / "extra.h5"
        with h5py.File(p, "w") as f:
            meta = f.create_group("__versionable__")
            meta.attrs["__OBJECT__"] = "UnkError"
            meta.attrs["__VERSION__"] = 1
            meta.attrs["__HASH__"] = ""
            f.attrs["name"] = "test"
            f.attrs["obsolete"] = 42

        with pytest.raises(UnknownFieldError, match="obsolete"):
            versionable.load(V2Strict, p)


class TestHdf5Migration:
    """Load a v1 HDF5 file with a v2 class that has a migration."""

    def test_migrationAddsField(self, tmp_path: Path) -> None:
        from versionable import Migration

        @dataclass
        class SensorV2(Versionable, version=2, name="MigSensor", register=True):
            name: str
            rate_Hz: float = 1000.0

            class Migrate:
                v1 = Migration().add("rate_Hz", default=1000.0)

        # Write a v1 HDF5 file manually
        p = tmp_path / "v1_sensor.h5"
        with h5py.File(p, "w") as f:
            meta = f.create_group("__versionable__")
            meta.attrs["__OBJECT__"] = "MigSensor"
            meta.attrs["__VERSION__"] = 1
            meta.attrs["__HASH__"] = ""
            f.attrs["name"] = "accelerometer"

        loaded = versionable.load(SensorV2, p)
        assert loaded.name == "accelerometer"
        assert loaded.rate_Hz == 1000.0


def _assertNoJsonAttrs(group: h5py.Group) -> None:
    """Recursively verify no attributes contain JSON strings."""
    for attrName in group.attrs:
        value = group.attrs[attrName]
        if isinstance(value, (str, bytes)):
            s = value if isinstance(value, str) else value.decode()
            assert not (s.startswith("{") or s.startswith("[")), (
                f"Found JSON-like attribute {attrName}={s!r} in {group.name}"
            )
    for key in group:
        item = group[key]
        if isinstance(item, h5py.Group):
            _assertNoJsonAttrs(item)
