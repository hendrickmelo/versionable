"""Tests for the JSON backend and save/load API."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest

import versionable
from versionable.errors import BackendError, ConverterError, VersionError

from .conftest import (
    Inner,
    Priority,
    SimpleConfig,
    WithArray,
    WithDatetime,
    WithEnum,
    WithList,
    WithLiteral,
    WithNested,
    WithOptional,
    WithSkipDefaults,
)


class TestJsonRoundTrip:
    def test_simpleConfig(self, tmp_path: Path) -> None:
        obj = SimpleConfig(name="test", debug=True, retries=5)
        p = tmp_path / "config.json"
        versionable.save(obj, p)
        loaded = versionable.load(SimpleConfig, p)
        assert loaded.name == "test"
        assert loaded.debug is True
        assert loaded.retries == 5

    def test_withDefaults(self, tmp_path: Path) -> None:
        obj = SimpleConfig(name="default")
        p = tmp_path / "config.json"
        versionable.save(obj, p)
        loaded = versionable.load(SimpleConfig, p)
        assert loaded.debug is False
        assert loaded.retries == 3

    def test_optional_present(self, tmp_path: Path) -> None:
        obj = WithOptional(label="test", description="hello")
        p = tmp_path / "out.json"
        versionable.save(obj, p)
        loaded = versionable.load(WithOptional, p)
        assert loaded.description == "hello"

    def test_optional_none(self, tmp_path: Path) -> None:
        obj = WithOptional(label="test")
        p = tmp_path / "out.json"
        versionable.save(obj, p)
        loaded = versionable.load(WithOptional, p)
        assert loaded.description is None

    def test_withArray(self, tmp_path: Path) -> None:
        arr = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        obj = WithArray(name="data", data=arr)
        p = tmp_path / "out.json"
        versionable.save(obj, p)
        loaded = versionable.load(WithArray, p)
        assert loaded.name == "data"
        np.testing.assert_array_equal(loaded.data, arr)

    def test_withEnum(self, tmp_path: Path) -> None:
        obj = WithEnum(title="task", priority=Priority.HIGH)
        p = tmp_path / "out.json"
        versionable.save(obj, p)
        loaded = versionable.load(WithEnum, p)
        assert loaded.priority is Priority.HIGH

    def test_withDatetime(self, tmp_path: Path) -> None:
        dt = datetime(2026, 3, 30, 12, 0, 0, tzinfo=UTC)
        obj = WithDatetime(label="event", createdAt=dt)
        p = tmp_path / "out.json"
        versionable.save(obj, p)
        loaded = versionable.load(WithDatetime, p)
        assert loaded.createdAt == dt

    def test_withNested(self, tmp_path: Path) -> None:
        obj = WithNested(name="origin", point=Inner(x=1.0, y=2.0))
        p = tmp_path / "out.json"
        versionable.save(obj, p)
        loaded = versionable.load(WithNested, p)
        assert isinstance(loaded.point, Inner)
        assert loaded.point.x == 1.0
        assert loaded.point.y == 2.0

    def test_withList(self, tmp_path: Path) -> None:
        obj = WithList(tags=["a", "b"], scores=[1.0, 2.0])
        p = tmp_path / "out.json"
        versionable.save(obj, p)
        loaded = versionable.load(WithList, p)
        assert loaded.tags == ["a", "b"]
        assert loaded.scores == [1.0, 2.0]

    def test_emptyList(self, tmp_path: Path) -> None:
        obj = WithList(tags=[], scores=[])
        p = tmp_path / "out.json"
        versionable.save(obj, p)
        loaded = versionable.load(WithList, p)
        assert loaded.tags == []


class TestJsonMetadata:
    def test_metadataInFile(self, tmp_path: Path) -> None:
        obj = SimpleConfig(name="test")
        p = tmp_path / "out.json"
        versionable.save(obj, p)

        data = json.loads(p.read_text())
        assert "__versionable__" in data
        meta = data["__versionable__"]
        assert "__OBJECT__" in meta
        assert "__VERSION__" in meta
        assert "__HASH__" in meta
        assert meta["__VERSION__"] == 1

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
                        "__OBJECT__": "SimpleConfig",
                        "__VERSION__": 1,
                        "__HASH__": "",
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
                        "__OBJECT__": "WithLiteral",
                        "__VERSION__": 1,
                        "__HASH__": "",
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
                        "__OBJECT__": "WithLiteral",
                        "__VERSION__": 1,
                        "__HASH__": "",
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
        assert "No __VERSION__" in caplog.text
        assert "SimpleConfig" in caplog.text

    def test_assumeVersionOverride(self, tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
        """assumeVersion should suppress the warning and use the given version."""
        p = tmp_path / "plain.json"
        p.write_text(json.dumps({"name": "test", "debug": False, "retries": 3}))
        with caplog.at_level("WARNING"):
            loaded = versionable.load(SimpleConfig, p, assumeVersion=1)
        assert loaded.name == "test"
        assert "No __VERSION__" not in caplog.text


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
                        "__OBJECT__": "SimpleConfig",
                        "__VERSION__": 999,
                        "__HASH__": "",
                    },
                    "name": "test",
                }
            )
        )
        with pytest.raises(VersionError, match="newer"):
            versionable.load(SimpleConfig, p)
