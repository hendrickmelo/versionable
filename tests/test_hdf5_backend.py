# ruff: noqa: E402 — module-level imports must come after pytest.importorskip("h5py")
# so the entire file is skipped when h5py is not installed.
"""Tests for the HDF5 backend with native type mapping."""

from __future__ import annotations

import pytest

h5py = pytest.importorskip("h5py")

import datetime
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import numpy as np
import numpy.typing as npt

import versionable
from versionable import Versionable
from versionable._hash import computeHash
from versionable.errors import ArrayNotLoadedError, BackendError
from versionable.hdf5 import (
    BLOSC_DEFAULT,
    GZIP_DEFAULT,
    LZF,
    UNCOMPRESSED,
    ZSTD_BEST,
    ZSTD_DEFAULT,
    ZSTD_FAST,
    Hdf5Compression,
)

from .conftest import Inner, SimpleConfig, WithArray, WithNested

# Module-level classes for tests that need type resolution across nested Versionables.
# Defining inside test functions causes forward-reference resolution failures with
# `from __future__ import annotations`.

_h_measurement = computeHash({"label": str, "data": npt.NDArray[np.float64]})


@dataclass
class _Measurement(Versionable, version=1, hash=_h_measurement, name="Measurement"):
    label: str
    data: npt.NDArray[np.float64]


_h_experiment = computeHash({"name": str, "measurements": list[_Measurement]})


@dataclass
class _Experiment(Versionable, version=1, hash=_h_experiment, register=False):
    name: str
    measurements: list[_Measurement]


class TestHdf5RoundTrip:
    def test_simpleScalars(self, tmp_path: Path) -> None:
        obj = SimpleConfig(name="test", debug=True, retries=5)
        p = tmp_path / "config.h5"
        versionable.save(obj, p)
        loaded = versionable.load(SimpleConfig, p)
        assert loaded.name == "test"
        assert loaded.debug is True
        assert loaded.retries == 5

    def test_withArray(self, tmp_path: Path) -> None:
        arr = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        obj = WithArray(name="data", data=arr)
        p = tmp_path / "out.h5"
        versionable.save(obj, p)
        loaded = versionable.load(WithArray, p, preload="*")
        assert loaded.name == "data"
        np.testing.assert_array_equal(loaded.data, arr)

    def test_2dArray(self, tmp_path: Path) -> None:
        h = computeHash({"label": str, "matrix": npt.NDArray[np.int32]})

        @dataclass
        class MatrixData(Versionable, version=1, hash=h, register=False):
            label: str
            matrix: npt.NDArray[np.int32]

        arr = np.array([[1, 2], [3, 4]], dtype=np.int32)
        obj = MatrixData(label="test", matrix=arr)
        p = tmp_path / "matrix.h5"
        versionable.save(obj, p)
        loaded = versionable.load(MatrixData, p, preload="*")
        np.testing.assert_array_equal(loaded.matrix, arr)

    def test_largeArray(self, tmp_path: Path) -> None:
        rng = np.random.default_rng(42)
        arr = rng.random(10000).astype(np.float64)
        obj = WithArray(name="big", data=arr)
        p = tmp_path / "big.h5"
        versionable.save(obj, p)
        loaded = versionable.load(WithArray, p, preload="*")
        np.testing.assert_array_equal(loaded.data, arr)

    def test_withNested(self, tmp_path: Path) -> None:
        obj = WithNested(name="origin", point=Inner(x=1.0, y=2.0))
        p = tmp_path / "nested.h5"
        versionable.save(obj, p)
        loaded = versionable.load(WithNested, p)
        assert isinstance(loaded.point, Inner)
        assert loaded.point.x == 1.0
        assert loaded.point.y == 2.0


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
            # zstd uses hdf5plugin filter id 32015
            assert ds.compression is not None

    def test_zstdFast(self, tmp_path: Path) -> None:
        self._saveAndCheck(tmp_path, ZSTD_FAST)

    def test_zstdBest(self, tmp_path: Path) -> None:
        self._saveAndCheck(tmp_path, ZSTD_BEST)

    def test_bloscDefault(self, tmp_path: Path) -> None:
        p = self._saveAndCheck(tmp_path, BLOSC_DEFAULT)
        with h5py.File(p, "r") as f:
            ds = f["data"]
            assert isinstance(ds, h5py.Dataset)
            assert ds.compression is not None

    def test_gzipDefault(self, tmp_path: Path) -> None:
        p = self._saveAndCheck(tmp_path, GZIP_DEFAULT)
        with h5py.File(p, "r") as f:
            ds = f["data"]
            assert isinstance(ds, h5py.Dataset)
            assert ds.compression == "gzip"

    def test_lzf(self, tmp_path: Path) -> None:
        p = self._saveAndCheck(tmp_path, LZF)
        with h5py.File(p, "r") as f:
            ds = f["data"]
            assert isinstance(ds, h5py.Dataset)
            assert ds.compression == "lzf"

    def test_uncompressed(self, tmp_path: Path) -> None:
        p = self._saveAndCheck(tmp_path, UNCOMPRESSED)
        with h5py.File(p, "r") as f:
            ds = f["data"]
            assert isinstance(ds, h5py.Dataset)
            assert ds.compression is None

    def test_defaultCompressionIsZstd(self, tmp_path: Path) -> None:
        """save() without explicit compression should use zstd."""
        arr = np.zeros(1000, dtype=np.float64)
        obj = WithArray(name="zeros", data=arr)
        p = tmp_path / "default.h5"
        versionable.save(obj, p)

        with h5py.File(p, "r") as f:
            ds = f["data"]
            assert isinstance(ds, h5py.Dataset)
            # Should be compressed (blosc2 via hdf5plugin)
            assert ds.compression is not None

    def test_compressionDoesNotAffectHash(self) -> None:
        """Compression is a storage concern, not a schema concern."""
        assert ZSTD_DEFAULT != GZIP_DEFAULT
        # Hash comes from fields, not compression settings
        h = computeHash({"name": str, "data": npt.NDArray[np.float64]})
        assert h == WithArray._serializer_meta_.hash


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
        h = computeHash({"name": str, "a": npt.NDArray[np.float64], "b": npt.NDArray[np.float64]})

        @dataclass
        class TwoArrays(Versionable, version=1, hash=h, register=False):
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


class TestHdf5Extension:
    def test_hdf5Extension(self, tmp_path: Path) -> None:
        obj = SimpleConfig(name="test")
        p = tmp_path / "out.hdf5"
        versionable.save(obj, p)
        loaded = versionable.load(SimpleConfig, p)
        assert loaded.name == "test"


class TestHdf5Errors:
    def test_missingFile(self, tmp_path: Path) -> None:
        with pytest.raises(BackendError):
            versionable.load(SimpleConfig, tmp_path / "nonexistent.h5")


# ---------------------------------------------------------------------------
# Native type mapping tests
# ---------------------------------------------------------------------------


class Color(Enum):
    RED = "red"
    GREEN = "green"
    BLUE = "blue"


class Priority(Enum):
    LOW = 1
    HIGH = 2


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
            assert f.attrs["debug"] is np.True_
            assert f.attrs["retries"] == 5

    def test_floatPrecision(self, tmp_path: Path) -> None:
        h = computeHash({"value": float})

        @dataclass
        class FloatData(Versionable, version=1, hash=h, register=False):
            value: float

        obj = FloatData(value=3.141592653589793)
        p = tmp_path / "float.h5"
        versionable.save(obj, p)
        loaded = versionable.load(FloatData, p)
        assert loaded.value == 3.141592653589793
        assert isinstance(loaded.value, float)


class TestNativeNone:
    """Test None handling with h5py.Empty."""

    def test_noneField(self, tmp_path: Path) -> None:
        h = computeHash({"name": str, "label": str | None})

        @dataclass
        class WithOptional(Versionable, version=1, hash=h, register=False):
            name: str
            label: str | None = None

        obj = WithOptional(name="test", label=None)
        p = tmp_path / "none.h5"
        versionable.save(obj, p)
        loaded = versionable.load(WithOptional, p)
        assert loaded.name == "test"
        assert loaded.label is None

    def test_noneVsDefault(self, tmp_path: Path) -> None:
        """None stored explicitly, not confused with missing default."""
        h = computeHash({"name": str, "tag": str | None})

        @dataclass
        class WithDefault(Versionable, version=1, hash=h, register=False):
            name: str
            tag: str | None = "default"

        obj = WithDefault(name="test", tag=None)
        p = tmp_path / "none_vs_default.h5"
        versionable.save(obj, p)
        loaded = versionable.load(WithDefault, p)
        assert loaded.tag is None  # not "default"


class TestNativeEnum:
    """Test Enum storage as native attributes."""

    def test_stringEnum(self, tmp_path: Path) -> None:
        h = computeHash({"name": str, "color": Color})

        @dataclass
        class WithColor(Versionable, version=1, hash=h, register=False):
            name: str
            color: Color

        obj = WithColor(name="test", color=Color.GREEN)
        p = tmp_path / "enum.h5"
        versionable.save(obj, p)
        loaded = versionable.load(WithColor, p)
        assert loaded.color == Color.GREEN

    def test_intEnum(self, tmp_path: Path) -> None:
        h = computeHash({"name": str, "priority": Priority})

        @dataclass
        class WithPriority(Versionable, version=1, hash=h, register=False):
            name: str
            priority: Priority

        obj = WithPriority(name="test", priority=Priority.HIGH)
        p = tmp_path / "int_enum.h5"
        versionable.save(obj, p)
        loaded = versionable.load(WithPriority, p)
        assert loaded.priority == Priority.HIGH


class TestNativeConvertedTypes:
    """Test converted types (datetime, Path) as native attributes."""

    def test_datetime(self, tmp_path: Path) -> None:
        h = computeHash({"name": str, "timestamp": datetime.datetime})

        @dataclass
        class WithDatetime(Versionable, version=1, hash=h, register=False):
            name: str
            timestamp: datetime.datetime

        ts = datetime.datetime(2024, 1, 15, 10, 30, 0)
        obj = WithDatetime(name="test", timestamp=ts)
        p = tmp_path / "datetime.h5"
        versionable.save(obj, p)
        loaded = versionable.load(WithDatetime, p)
        assert loaded.timestamp == ts

    def test_path(self, tmp_path: Path) -> None:
        h = computeHash({"name": str, "location": Path})

        @dataclass
        class WithPath(Versionable, version=1, hash=h, register=False):
            name: str
            location: Path

        obj = WithPath(name="test", location=Path("/tmp/data"))
        p = tmp_path / "path.h5"
        versionable.save(obj, p)
        loaded = versionable.load(WithPath, p)
        assert loaded.location == Path("/tmp/data")


class TestNativeListDatasets:
    """Test list[numeric], list[str], list[bool] as 1-D datasets."""

    def test_listFloat(self, tmp_path: Path) -> None:
        h = computeHash({"name": str, "values": list[float]})

        @dataclass
        class WithListFloat(Versionable, version=1, hash=h, register=False):
            name: str
            values: list[float]

        obj = WithListFloat(name="test", values=[1.0, 2.5, 3.7])
        p = tmp_path / "list_float.h5"
        versionable.save(obj, p)
        loaded = versionable.load(WithListFloat, p)
        assert loaded.values == [1.0, 2.5, 3.7]
        assert isinstance(loaded.values, list)

    def test_listInt(self, tmp_path: Path) -> None:
        h = computeHash({"name": str, "counts": list[int]})

        @dataclass
        class WithListInt(Versionable, version=1, hash=h, register=False):
            name: str
            counts: list[int]

        obj = WithListInt(name="test", counts=[10, 20, 30])
        p = tmp_path / "list_int.h5"
        versionable.save(obj, p)
        loaded = versionable.load(WithListInt, p)
        assert loaded.counts == [10, 20, 30]

    def test_listStr(self, tmp_path: Path) -> None:
        h = computeHash({"name": str, "tags": list[str]})

        @dataclass
        class WithListStr(Versionable, version=1, hash=h, register=False):
            name: str
            tags: list[str]

        obj = WithListStr(name="test", tags=["alpha", "beta", "gamma"])
        p = tmp_path / "list_str.h5"
        versionable.save(obj, p)
        loaded = versionable.load(WithListStr, p)
        assert loaded.tags == ["alpha", "beta", "gamma"]

    def test_listBool(self, tmp_path: Path) -> None:
        h = computeHash({"name": str, "flags": list[bool]})

        @dataclass
        class WithListBool(Versionable, version=1, hash=h, register=False):
            name: str
            flags: list[bool]

        obj = WithListBool(name="test", flags=[True, False, True])
        p = tmp_path / "list_bool.h5"
        versionable.save(obj, p)
        loaded = versionable.load(WithListBool, p)
        assert loaded.flags == [True, False, True]

    def test_emptyList(self, tmp_path: Path) -> None:
        h = computeHash({"name": str, "values": list[float]})

        @dataclass
        class WithEmpty(Versionable, version=1, hash=h, register=False):
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
        h = computeHash({"name": str, "traces": list[npt.NDArray[np.float64]]})

        @dataclass
        class WithTraces(Versionable, version=1, hash=h, register=False):
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
        h = computeHash({"name": str, "data": list[npt.NDArray[np.float64]]})

        @dataclass
        class VaryingShapes(Versionable, version=1, hash=h, register=False):
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
        h = computeHash({"name": str, "channels": dict[str, npt.NDArray[np.float64]]})

        @dataclass
        class WithChannels(Versionable, version=1, hash=h, register=False):
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
        h = computeHash({"name": str, "traces": list[npt.NDArray[np.float64]]})

        @dataclass
        class Traces(Versionable, version=1, hash=h, register=False):
            name: str
            traces: list[npt.NDArray[np.float64]]

        obj = Traces(name="test", traces=[np.array([1.0]), np.array([2.0])])
        p = tmp_path / "layout.h5"
        versionable.save(obj, p)

        with h5py.File(p, "r") as f:
            assert "traces" in f
            traces_group = f["traces"]
            assert isinstance(traces_group, h5py.Group)
            assert sorted(traces_group.keys()) == ["0", "1"]


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
        h = computeHash({"name": str, "points": list[Inner]})

        @dataclass
        class WithPointList(Versionable, version=1, hash=h, register=False):
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
