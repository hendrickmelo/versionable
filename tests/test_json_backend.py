"""Tests for the JSON backend and save/load API."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pytest

try:
    import numpy as np
    import numpy.typing as npt

    _HAS_NUMPY = True
except ImportError:
    _HAS_NUMPY = False

import versionable
from versionable._base import Versionable
from versionable.errors import BackendError, ConverterError, VersionError

from .conftest import (
    SimpleConfig,
    WithLiteral,
    WithSkipDefaults,
)

if _HAS_NUMPY:

    @dataclass
    class _JsonArrLegacy(Versionable, version=1, name="JsonArrLegacy", register=True):
        """Test class for back-compat ndarray reads in JSON files."""

        label: str
        data: npt.NDArray[np.float64]


class TestJsonMetadata:
    def test_metadataInFile(self, tmp_path: Path) -> None:
        obj = SimpleConfig(name="test")
        p = tmp_path / "out.json"
        versionable.save(obj, p)

        data = json.loads(p.read_text())
        assert "__versionable__" in data
        meta = data["__versionable__"]
        assert "object" in meta
        assert "version" in meta
        assert "hash" in meta
        assert meta["version"] == 1

    def test_prettyPrinted(self, tmp_path: Path) -> None:
        obj = SimpleConfig(name="test")
        p = tmp_path / "out.json"
        versionable.save(obj, p)

        text = p.read_text()
        assert "\n" in text  # Pretty-printed


class TestSkipDefaults:
    def test_defaultsOmitted(self, tmp_path: Path) -> None:
        obj = WithSkipDefaults(name="test")
        p = tmp_path / "out.json"
        versionable.save(obj, p)

        data = json.loads(p.read_text())
        assert "name" in data
        assert "debug" not in data  # default False omitted
        assert "count" not in data  # default 0 omitted

    def test_nonDefaultsKept(self, tmp_path: Path) -> None:
        obj = WithSkipDefaults(name="test", debug=True, count=5)
        p = tmp_path / "out.json"
        versionable.save(obj, p)

        data = json.loads(p.read_text())
        assert data["debug"] is True
        assert data["count"] == 5

    def test_roundtripWithDefaults(self, tmp_path: Path) -> None:
        obj = WithSkipDefaults(name="test")
        p = tmp_path / "out.json"
        versionable.save(obj, p)
        loaded = versionable.load(WithSkipDefaults, p)
        assert loaded.name == "test"
        assert loaded.debug is False  # Default restored
        assert loaded.count == 0


class TestUnknownFields:
    def test_ignoreByDefault(self, tmp_path: Path) -> None:
        """Unknown fields are silently ignored by default."""
        p = tmp_path / "out.json"
        p.write_text(
            json.dumps(
                {
                    "__versionable__": {
                        "object": "SimpleConfig",
                        "version": 1,
                        "hash": "",
                    },
                    "name": "test",
                    "debug": False,
                    "retries": 3,
                    "extra_field": "should be ignored",
                }
            )
        )
        loaded = versionable.load(SimpleConfig, p)
        assert loaded.name == "test"


class TestJsonLiteral:
    def test_literalRoundTrip(self, tmp_path: Path) -> None:
        obj = WithLiteral(name="test", mode="slow")
        p = tmp_path / "out.json"
        versionable.save(obj, p)
        loaded = versionable.load(WithLiteral, p)
        assert loaded.mode == "slow"

    def test_invalidLiteralRaises(self, tmp_path: Path) -> None:
        p = tmp_path / "out.json"
        p.write_text(
            json.dumps(
                {
                    "__versionable__": {
                        "object": "WithLiteral",
                        "version": 1,
                        "hash": "",
                    },
                    "name": "test",
                    "mode": "banana",
                }
            )
        )
        with pytest.raises(ConverterError, match="banana"):
            versionable.load(WithLiteral, p)

    def test_validateLiteralsOptOutViaLoad(self, tmp_path: Path) -> None:
        p = tmp_path / "out.json"
        p.write_text(
            json.dumps(
                {
                    "__versionable__": {
                        "object": "WithLiteral",
                        "version": 1,
                        "hash": "",
                    },
                    "name": "test",
                    "mode": "banana",
                }
            )
        )
        loaded = versionable.load(WithLiteral, p, validateLiterals=False)
        assert loaded.mode == "banana"


class TestJsonMissingVersion:
    def test_noVersionDefaultsToCurrentVersion(self, tmp_path: Path) -> None:
        """A JSON file with no metadata should load as the current version."""
        p = tmp_path / "plain.json"
        p.write_text(json.dumps({"name": "test", "debug": True, "retries": 5}))
        loaded = versionable.load(SimpleConfig, p)
        assert loaded.name == "test"
        assert loaded.debug is True
        assert loaded.retries == 5

    def test_noVersionLogsWarning(self, tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
        """A JSON file with no metadata should log a warning."""
        p = tmp_path / "plain.json"
        p.write_text(json.dumps({"name": "test", "debug": False, "retries": 3}))
        with caplog.at_level("WARNING"):
            versionable.load(SimpleConfig, p)
        assert "No version" in caplog.text
        assert "SimpleConfig" in caplog.text

    def test_assumeVersionOverride(self, tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
        """assumeVersion should suppress the warning and use the given version."""
        p = tmp_path / "plain.json"
        p.write_text(json.dumps({"name": "test", "debug": False, "retries": 3}))
        with caplog.at_level("WARNING"):
            loaded = versionable.load(SimpleConfig, p, assumeVersion=1)
        assert loaded.name == "test"
        assert "No version" not in caplog.text


class TestErrors:
    def test_missingFile(self, tmp_path: Path) -> None:
        with pytest.raises(BackendError):
            versionable.load(SimpleConfig, tmp_path / "nonexistent.json")

    def test_unknownExtension(self, tmp_path: Path) -> None:
        with pytest.raises(BackendError, match="No backend"):
            versionable.save(SimpleConfig(name="x"), tmp_path / "out.xyz")

    def test_newerVersionRaises(self, tmp_path: Path) -> None:
        p = tmp_path / "out.json"
        p.write_text(
            json.dumps(
                {
                    "__versionable__": {
                        "object": "SimpleConfig",
                        "version": 999,
                        "hash": "",
                    },
                    "name": "test",
                }
            )
        )
        with pytest.raises(VersionError, match="newer"):
            versionable.load(SimpleConfig, p)

    def test_futureFormatRaises(self, tmp_path: Path) -> None:
        p = tmp_path / "out.json"
        p.write_text(
            json.dumps(
                {
                    "__versionable__": {
                        "object": "SimpleConfig",
                        "version": 1,
                        "hash": "",
                        "format": 2,
                    },
                    "name": "test",
                }
            )
        )
        with pytest.raises(BackendError, match="Upgrade versionable"):
            versionable.load(SimpleConfig, p)


class TestJsonBackCompat:
    """Read-side compatibility for files written by versionable 0.1.x."""

    def test_loadOldFormatEnvelope(self, tmp_path: Path) -> None:
        """A file with the legacy __OBJECT__/__VERSION__/__HASH__ keys still loads."""
        p = tmp_path / "old.json"
        p.write_text(
            json.dumps(
                {
                    "__versionable__": {
                        "__OBJECT__": "SimpleConfig",
                        "__VERSION__": 1,
                        "__HASH__": "ed3a90",
                    },
                    "name": "legacy",
                    "debug": True,
                    "retries": 7,
                }
            )
        )
        loaded = versionable.load(SimpleConfig, p)
        assert loaded.name == "legacy"
        assert loaded.debug is True
        assert loaded.retries == 7

    def test_oldFormatFutureFormatRaises(self, tmp_path: Path) -> None:
        """The legacy __FORMAT__ key still triggers the upgrade-required error."""
        p = tmp_path / "old_future.json"
        p.write_text(
            json.dumps(
                {
                    "__versionable__": {
                        "__OBJECT__": "SimpleConfig",
                        "__VERSION__": 1,
                        "__HASH__": "",
                        "__FORMAT__": 2,
                    },
                    "name": "test",
                }
            )
        )
        with pytest.raises(BackendError, match="Upgrade versionable"):
            versionable.load(SimpleConfig, p)

    @pytest.mark.skipif(not _HAS_NUMPY, reason="numpy not installed")
    def test_loadOldFormatNdarray(self, tmp_path: Path) -> None:
        """A file with the legacy __ndarray__ sentinel still loads as a numpy array."""
        # Save a fresh file then rewrite the envelope + sentinel keys to legacy form.
        obj = _JsonArrLegacy(label="legacy-array", data=np.array([1.5, 2.5, 3.5], dtype=np.float64))
        p = tmp_path / "old_arr.json"
        versionable.save(obj, p)
        text = p.read_text()
        legacy = (
            text.replace('"object":', '"__OBJECT__":')
            .replace('"version":', '"__VERSION__":')
            .replace('"hash":', '"__HASH__":')
            .replace('"__ver_ndarray__":', '"__ndarray__":')
        )
        p.write_text(legacy)

        loaded = versionable.load(_JsonArrLegacy, p)
        assert loaded.label == "legacy-array"
        np.testing.assert_array_equal(loaded.data, np.array([1.5, 2.5, 3.5]))

    def test_newKeysPreferredWhenBothPresent(self, tmp_path: Path) -> None:
        """Mixed old + new envelope keys: new wins, no errors."""
        p = tmp_path / "mixed.json"
        p.write_text(
            json.dumps(
                {
                    "__versionable__": {
                        # Old keys point at a different class — must be ignored
                        "__OBJECT__": "WrongClass",
                        "__VERSION__": 99,
                        "__HASH__": "deadbe",
                        # New keys are correct — must win
                        "object": "SimpleConfig",
                        "version": 1,
                        "hash": "ed3a90",
                    },
                    "name": "winner",
                }
            )
        )
        loaded = versionable.load(SimpleConfig, p)
        assert loaded.name == "winner"
