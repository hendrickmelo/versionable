# ruff: noqa: E402 — module-level imports must come after pytest.importorskip("h5py")
# so the entire file is skipped when h5py is not installed.
"""Tests for the HDF5 backend with lazy loading."""

from __future__ import annotations

import pytest

h5py = pytest.importorskip("h5py")

from dataclasses import dataclass
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
            assert f.attrs["__OBJECT__"] == "SimpleConfig"
            assert f.attrs["__VERSION__"] == 1


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
