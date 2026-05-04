"""Cross-backend roundtrip tests.

Saves objects through every backend and verifies that the loaded data
matches the original exactly — values, types, and structure.

Hash validation is tested separately in test_hash.py and test_base.py.
Classes here omit the hash parameter to keep declarations concise.
"""

from __future__ import annotations

import datetime
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path, PurePosixPath
from typing import Literal
from uuid import UUID

import pytest

try:
    import numpy as np
    import numpy.typing as npt

    _HAS_NUMPY = True
except ImportError:
    _HAS_NUMPY = False

import versionable
from versionable import Versionable

# ---------------------------------------------------------------------------
# Test classes — defined at module level for type-annotation resolution
# ---------------------------------------------------------------------------


class Color(Enum):
    RED = "red"
    GREEN = "green"
    BLUE = "blue"


class IntPriority(Enum):
    LOW = 1
    MEDIUM = 2
    HIGH = 3


@dataclass
class _Inner(Versionable, version=1, hash="e37514", name="CrossInner"):
    x: float
    y: float


@dataclass
class _KitchenSink(Versionable, version=1, hash="3b5a61", register=False):
    name: str
    count: int
    rate: float
    debug: bool
    color: Color
    intPriority: IntPriority
    createdAt: datetime.datetime
    createdAtTz: datetime.datetime
    today: datetime.date
    elapsed: datetime.timedelta
    filePath: Path
    posixPath: PurePosixPath
    uid: UUID
    mode: Literal["fast", "slow"]
    tags: list[str]
    scores: list[float]
    counts: list[int]
    flags: list[bool]
    uniqueTags: set[str]
    frozenIds: frozenset[int]
    coords: tuple[float, ...]
    metadata: dict[str, int]
    indexedLabels: dict[int, str]
    nested: _Inner
    points: list[_Inner]
    channelData: dict[str, list[float]]


@dataclass
class _WithOptional(Versionable, version=1, hash="c599a5", register=False):
    name: str
    label: str | None = None
    count: int | None = None


if _HAS_NUMPY:

    @dataclass
    class _WithArrays(Versionable, version=1, hash="e9fc06", register=False):
        name: str
        data: npt.NDArray[np.float64]
        matrix: npt.NDArray[np.int32]
        traces: list[npt.NDArray[np.float64]]
        channels: dict[str, npt.NDArray[np.float64]]


@dataclass
class _EmptyCollections(Versionable, version=1, hash="6c8e40", register=False):
    tags: list[str] = field(default_factory=list)
    scores: list[float] = field(default_factory=list)
    counts: list[int] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Backend extensions
# ---------------------------------------------------------------------------

_DICT_BACKENDS: list[str] = [".json"]

try:
    import tomlkit as _tomlkit  # noqa: F401

    _DICT_BACKENDS.append(".toml")
except ImportError:
    pass

try:
    import yaml as _yaml  # noqa: F401

    _DICT_BACKENDS.append(".yaml")
except ImportError:
    pass

try:
    import versionable._hdf5_backend

    _HDF5_BACKENDS: list[str] = [".h5"]
except ImportError:
    _HDF5_BACKENDS = []

_ALL_BACKENDS = [*_DICT_BACKENDS, *_HDF5_BACKENDS]
_NONE_SAFE_BACKENDS = [ext for ext in [".json", ".yaml", *_HDF5_BACKENDS] if ext in _ALL_BACKENDS]


def _roundtrip(cls: type, obj: Versionable, tmp_path: Path, ext: str) -> Versionable:
    """Save and reload through a specific backend."""
    p = tmp_path / f"roundtrip{ext}"
    versionable.save(obj, p)
    return versionable.load(cls, p, preload="*")


# ---------------------------------------------------------------------------
# Assertion helpers
# ---------------------------------------------------------------------------


def _assertFieldMatch(original: object, loaded: object, fieldName: str, ext: str) -> None:
    """Assert that a single field value and type match after roundtrip."""
    orig = getattr(original, fieldName)
    load = getattr(loaded, fieldName)

    if _HAS_NUMPY and isinstance(orig, np.ndarray):
        assert isinstance(load, np.ndarray), f"[{ext}] {fieldName}: expected np.ndarray, got {type(load).__name__}"
        np.testing.assert_array_equal(load, orig, err_msg=f"[{ext}] {fieldName}")
        assert load.dtype == orig.dtype, f"[{ext}] {fieldName}: dtype mismatch {load.dtype} != {orig.dtype}"
        return

    if _HAS_NUMPY and isinstance(orig, list) and orig and isinstance(orig[0], np.ndarray):
        assert isinstance(load, list), f"[{ext}] {fieldName}: expected list, got {type(load).__name__}"
        assert len(load) == len(orig), f"[{ext}] {fieldName}: length mismatch {len(load)} != {len(orig)}"
        for i, (origArr, loadArr) in enumerate(zip(orig, load, strict=True)):
            np.testing.assert_array_equal(loadArr, origArr, err_msg=f"[{ext}] {fieldName}[{i}]")
        return

    if _HAS_NUMPY and isinstance(orig, dict) and orig and isinstance(next(iter(orig.values())), np.ndarray):
        assert isinstance(load, dict), f"[{ext}] {fieldName}: expected dict, got {type(load).__name__}"
        assert set(load.keys()) == set(orig.keys()), f"[{ext}] {fieldName}: key mismatch"
        for k in orig:
            np.testing.assert_array_equal(load[k], orig[k], err_msg=f"[{ext}] {fieldName}[{k!r}]")
        return

    assert load == orig, f"[{ext}] {fieldName}: value mismatch {load!r} != {orig!r}"
    assert type(load) is type(orig), (
        f"[{ext}] {fieldName}: type mismatch {type(load).__name__} != {type(orig).__name__}"
    )


def _assertFullMatch(original: Versionable, loaded: Versionable, ext: str) -> None:
    """Assert all fields match between original and loaded."""
    from versionable._base import _resolveFields

    fields = _resolveFields(type(original))
    for fieldName in fields:
        orig = getattr(original, fieldName)
        load = getattr(loaded, fieldName)

        # Recurse into nested Versionables
        if isinstance(orig, Versionable):
            assert isinstance(load, type(orig)), (
                f"[{ext}] {fieldName}: expected {type(orig).__name__}, got {type(load).__name__}"
            )
            _assertFullMatch(orig, load, ext)
        elif isinstance(orig, list) and orig and isinstance(orig[0], Versionable):
            assert len(load) == len(orig), f"[{ext}] {fieldName}: length mismatch"
            for i, (origItem, loadItem) in enumerate(zip(orig, load, strict=True)):
                _assertFullMatch(origItem, loadItem, f"{ext}/{fieldName}[{i}]")
        else:
            _assertFieldMatch(original, loaded, fieldName, ext)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestKitchenSinkRoundtrip:
    """Every common type through every backend."""

    @pytest.fixture
    def obj(self) -> _KitchenSink:
        return _KitchenSink(
            name="experiment-1",
            count=42,
            rate=3.14159265358979,
            debug=True,
            color=Color.GREEN,
            intPriority=IntPriority.HIGH,
            createdAt=datetime.datetime(2024, 6, 15, 14, 30, 0),
            createdAtTz=datetime.datetime(2024, 6, 15, 14, 30, 0, tzinfo=datetime.UTC),
            today=datetime.date(2024, 6, 15),
            elapsed=datetime.timedelta(hours=2, minutes=30, seconds=15),
            filePath=Path("/data/results"),
            posixPath=PurePosixPath("/opt/config"),
            uid=UUID("12345678-1234-5678-1234-567812345678"),
            mode="slow",
            tags=["alpha", "beta", "gamma"],
            scores=[1.0, 2.5, 3.7],
            counts=[10, 20, 30],
            flags=[True, False, True],
            uniqueTags={"x", "y", "z"},
            frozenIds=frozenset({10, 20, 30}),
            coords=(1.0, 2.0, 3.0),
            metadata={"width": 100, "height": 200},
            indexedLabels={0: "first", 1: "second"},
            nested=_Inner(x=1.5, y=-2.5),
            points=[_Inner(x=0.0, y=0.0), _Inner(x=1.0, y=1.0)],
            channelData={"ch0": [1.0, 2.0, 3.0], "ch1": [4.0, 5.0]},
        )

    @pytest.mark.parametrize("ext", _ALL_BACKENDS)
    def test_allFieldsRoundtrip(self, obj: _KitchenSink, tmp_path: Path, ext: str) -> None:
        loaded = _roundtrip(_KitchenSink, obj, tmp_path, ext)
        _assertFullMatch(obj, loaded, ext)

    @pytest.mark.parametrize("ext", _ALL_BACKENDS)
    def test_exactTypes(self, obj: _KitchenSink, tmp_path: Path, ext: str) -> None:
        """Verify that deserialized types are exact Python types, not surrogates."""
        loaded = _roundtrip(_KitchenSink, obj, tmp_path, ext)

        assert type(loaded.name) is str, f"[{ext}] name type"
        assert type(loaded.count) is int, f"[{ext}] count type"
        assert type(loaded.rate) is float, f"[{ext}] rate type"
        assert type(loaded.debug) is bool, f"[{ext}] debug type"
        assert isinstance(loaded.color, Color), f"[{ext}] color type"
        assert isinstance(loaded.intPriority, IntPriority), f"[{ext}] intPriority type"
        assert isinstance(loaded.createdAt, datetime.datetime), f"[{ext}] createdAt type"
        assert isinstance(loaded.createdAtTz, datetime.datetime), f"[{ext}] createdAtTz type"
        assert loaded.createdAtTz.tzinfo is not None, f"[{ext}] createdAtTz should be timezone-aware"
        assert isinstance(loaded.today, datetime.date), f"[{ext}] today type"
        assert isinstance(loaded.elapsed, datetime.timedelta), f"[{ext}] elapsed type"
        assert isinstance(loaded.filePath, Path), f"[{ext}] filePath type"
        assert isinstance(loaded.posixPath, PurePosixPath), f"[{ext}] posixPath type"
        assert isinstance(loaded.uid, UUID), f"[{ext}] uid type"
        assert type(loaded.mode) is str, f"[{ext}] mode type"
        assert isinstance(loaded.tags, list), f"[{ext}] tags type"
        assert isinstance(loaded.scores, list), f"[{ext}] scores type"
        assert isinstance(loaded.counts, list), f"[{ext}] counts type"
        assert isinstance(loaded.flags, list), f"[{ext}] flags type"
        assert isinstance(loaded.uniqueTags, set), f"[{ext}] uniqueTags type"
        assert isinstance(loaded.frozenIds, frozenset), f"[{ext}] frozenIds type"
        assert isinstance(loaded.coords, tuple), f"[{ext}] coords type"
        assert isinstance(loaded.metadata, dict), f"[{ext}] metadata type"
        assert isinstance(loaded.indexedLabels, dict), f"[{ext}] indexedLabels type"
        assert isinstance(loaded.nested, _Inner), f"[{ext}] nested type"
        assert isinstance(loaded.points, list), f"[{ext}] points type"
        assert all(isinstance(p, _Inner) for p in loaded.points), f"[{ext}] points element type"
        assert isinstance(loaded.channelData, dict), f"[{ext}] channelData type"

    @pytest.mark.parametrize("ext", _ALL_BACKENDS)
    def test_listElementTypes(self, obj: _KitchenSink, tmp_path: Path, ext: str) -> None:
        """Verify that list elements have the correct Python types."""
        loaded = _roundtrip(_KitchenSink, obj, tmp_path, ext)

        for tag in loaded.tags:
            assert type(tag) is str, f"[{ext}] tag element type: {type(tag)}"
        for score in loaded.scores:
            assert type(score) is float, f"[{ext}] score element type: {type(score)}"
        for count in loaded.counts:
            assert type(count) is int, f"[{ext}] count element type: {type(count)}"
        for flag in loaded.flags:
            assert type(flag) is bool, f"[{ext}] flag element type: {type(flag)}"
        for tag in loaded.uniqueTags:
            assert type(tag) is str, f"[{ext}] uniqueTags element type: {type(tag)}"
        for val in loaded.frozenIds:
            assert type(val) is int, f"[{ext}] frozenIds element type: {type(val)}"
        for val in loaded.coords:
            assert type(val) is float, f"[{ext}] coords element type: {type(val)}"
        for k, v in loaded.metadata.items():
            assert type(k) is str, f"[{ext}] metadata key type: {type(k)}"
            assert type(v) is int, f"[{ext}] metadata value type: {type(v)}"
        for k, v in loaded.indexedLabels.items():
            assert type(k) is int, f"[{ext}] indexedLabels key type: {type(k)}"
            assert type(v) is str, f"[{ext}] indexedLabels value type: {type(v)}"
        for k, v in loaded.channelData.items():
            assert type(k) is str, f"[{ext}] channelData key type: {type(k)}"
            assert isinstance(v, list), f"[{ext}] channelData value type: {type(v)}"
            for elem in v:
                assert type(elem) is float, f"[{ext}] channelData element type: {type(elem)}"


class TestOptionalFieldsRoundtrip:
    """None handling across backends."""

    @pytest.mark.parametrize("ext", _NONE_SAFE_BACKENDS)
    def test_noneValues(self, tmp_path: Path, ext: str) -> None:
        """None fields roundtrip correctly."""
        obj = _WithOptional(name="test", label=None, count=None)
        loaded = _roundtrip(_WithOptional, obj, tmp_path, ext)
        assert loaded.label is None, f"[{ext}] label should be None"
        assert loaded.count is None, f"[{ext}] count should be None"

    @pytest.mark.parametrize("ext", _NONE_SAFE_BACKENDS)
    def test_noneVsNonNone(self, tmp_path: Path, ext: str) -> None:
        """Mix of None and non-None in optional fields."""
        obj = _WithOptional(name="test", label="hello", count=None)
        loaded = _roundtrip(_WithOptional, obj, tmp_path, ext)
        assert loaded.label == "hello", f"[{ext}] label value"
        assert type(loaded.label) is str, f"[{ext}] label type"
        assert loaded.count is None, f"[{ext}] count should be None"

    @pytest.mark.skipif(".toml" not in _DICT_BACKENDS, reason="toml not installed")
    def test_tomlOmitsNone(self, tmp_path: Path) -> None:
        """TOML omits None fields; they restore from defaults."""
        obj = _WithOptional(name="test", label=None, count=None)
        loaded = _roundtrip(_WithOptional, obj, tmp_path, ".toml")
        # TOML omits None → dataclass default (also None) kicks in
        assert loaded.label is None
        assert loaded.count is None


if _HAS_NUMPY:

    class TestArrayFieldsRoundtrip:
        """Array and array-collection fields across backends."""

        @pytest.fixture
        def obj(self) -> _WithArrays:
            return _WithArrays(
                name="arrays",
                data=np.array([1.0, 2.0, 3.0], dtype=np.float64),
                matrix=np.array([[1, 2], [3, 4]], dtype=np.int32),
                traces=[np.array([10.0, 20.0]), np.array([30.0, 40.0, 50.0])],
                channels={"ch0": np.array([1.0, 2.0]), "ch1": np.array([3.0, 4.0])},
            )

        @pytest.mark.skipif(not _HDF5_BACKENDS, reason="HDF5 backend not available")
        def test_hdf5Roundtrip(self, obj: _WithArrays, tmp_path: Path) -> None:
            loaded = _roundtrip(_WithArrays, obj, tmp_path, ".h5")
            _assertFullMatch(obj, loaded, ".h5")

        @pytest.mark.parametrize("ext", _DICT_BACKENDS)
        def test_dictBackendRoundtrip(self, obj: _WithArrays, tmp_path: Path, ext: str) -> None:
            """Dict backends serialize arrays as base64 npz and roundtrip correctly."""
            loaded = _roundtrip(_WithArrays, obj, tmp_path, ext)
            np.testing.assert_array_equal(loaded.data, obj.data)
            assert loaded.data.dtype == obj.data.dtype
            np.testing.assert_array_equal(loaded.matrix, obj.matrix)
            assert loaded.matrix.dtype == obj.matrix.dtype

        @pytest.mark.skipif(not _HDF5_BACKENDS, reason="HDF5 backend not available")
        def test_hdf5ArrayDtypes(self, obj: _WithArrays, tmp_path: Path) -> None:
            """HDF5 preserves exact dtypes."""
            loaded = _roundtrip(_WithArrays, obj, tmp_path, ".h5")
            assert loaded.data.dtype == np.float64
            assert loaded.matrix.dtype == np.int32

    class TestNumpyDtypeRoundtrip:
        """Verify that various numpy dtypes survive the roundtrip."""

        @pytest.mark.parametrize(
            "dtype",
            [
                np.float32,
                np.float64,
                np.int8,
                np.int16,
                np.int32,
                np.int64,
                np.uint8,
                np.uint16,
                np.uint32,
                np.uint64,
                np.bool_,
                np.complex64,
                np.complex128,
            ],
        )
        @pytest.mark.parametrize("ext", _ALL_BACKENDS)
        def test_scalarDtype(self, tmp_path: Path, dtype: type, ext: str) -> None:
            """1-D array of each dtype roundtrips with correct dtype."""

            @dataclass
            class DtypeTest(Versionable, version=1, register=False):
                arr: npt.NDArray[np.generic]

            arr = np.array([1, 2, 3], dtype=dtype)
            obj = DtypeTest(arr=arr)
            loaded = _roundtrip(DtypeTest, obj, tmp_path, ext)
            np.testing.assert_array_equal(loaded.arr, arr)
            assert loaded.arr.dtype == dtype, f"[{ext}] dtype: {loaded.arr.dtype} != {dtype}"

        @pytest.mark.parametrize("ext", _ALL_BACKENDS)
        def test_2dArray(self, tmp_path: Path, ext: str) -> None:
            """2-D array preserves shape and dtype."""

            @dataclass
            class Matrix(Versionable, version=1, register=False):
                matrix: npt.NDArray[np.float64]

            arr = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float64)
            obj = Matrix(matrix=arr)
            loaded = _roundtrip(Matrix, obj, tmp_path, ext)
            np.testing.assert_array_equal(loaded.matrix, arr)
            assert loaded.matrix.shape == (3, 2), f"[{ext}] shape: {loaded.matrix.shape}"

        @pytest.mark.parametrize("ext", _ALL_BACKENDS)
        def test_3dArray(self, tmp_path: Path, ext: str) -> None:
            """3-D array preserves shape."""

            @dataclass
            class Cube(Versionable, version=1, register=False):
                cube: npt.NDArray[np.float32]

            arr = np.ones((2, 3, 4), dtype=np.float32)
            obj = Cube(cube=arr)
            loaded = _roundtrip(Cube, obj, tmp_path, ext)
            np.testing.assert_array_equal(loaded.cube, arr)
            assert loaded.cube.shape == (2, 3, 4), f"[{ext}] shape: {loaded.cube.shape}"
            assert loaded.cube.dtype == np.float32, f"[{ext}] dtype: {loaded.cube.dtype}"

        @pytest.mark.parametrize("ext", _ALL_BACKENDS)
        def test_emptyArray(self, tmp_path: Path, ext: str) -> None:
            """Zero-length array preserves dtype."""

            @dataclass
            class EmptyArr(Versionable, version=1, register=False):
                arr: npt.NDArray[np.float64]

            arr = np.array([], dtype=np.float64)
            obj = EmptyArr(arr=arr)
            loaded = _roundtrip(EmptyArr, obj, tmp_path, ext)
            assert len(loaded.arr) == 0
            assert loaded.arr.dtype == np.float64, f"[{ext}] dtype: {loaded.arr.dtype}"


class TestEmptyCollectionsRoundtrip:
    """Empty lists across backends."""

    @pytest.mark.parametrize("ext", _ALL_BACKENDS)
    def test_emptyLists(self, tmp_path: Path, ext: str) -> None:
        obj = _EmptyCollections(tags=[], scores=[], counts=[])
        loaded = _roundtrip(_EmptyCollections, obj, tmp_path, ext)
        assert loaded.tags == []
        assert loaded.scores == []
        assert loaded.counts == []
        assert isinstance(loaded.tags, list)
        assert isinstance(loaded.scores, list)
        assert isinstance(loaded.counts, list)


@dataclass
class _WithUnion(Versionable, version=1, register=False):
    label: str
    value: int | str


class TestUnionTypeRoundtrip:
    """int | str union field across backends."""

    @pytest.mark.parametrize("ext", _DICT_BACKENDS)
    def test_strValue(self, tmp_path: Path, ext: str) -> None:
        obj = _WithUnion(label="test", value="hello")
        loaded = _roundtrip(_WithUnion, obj, tmp_path, ext)
        assert loaded.value == "hello"
        assert type(loaded.value) is str

    @pytest.mark.parametrize("ext", _DICT_BACKENDS)
    def test_intValue(self, tmp_path: Path, ext: str) -> None:
        obj = _WithUnion(label="test", value=42)
        loaded = _roundtrip(_WithUnion, obj, tmp_path, ext)
        assert loaded.value == 42
        assert type(loaded.value) is int
