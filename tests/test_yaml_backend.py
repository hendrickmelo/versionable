"""Tests for the YAML backend."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest
import yaml

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
    WithLiteralFallback,
    WithLiteralNoValidation,
    WithNested,
    WithOptional,
    WithSkipDefaults,
)


class TestYamlRoundTrip:
    def test_simpleConfig(self, tmp_path: Path) -> None:
        obj = SimpleConfig(name="test", debug=True, retries=5)
        p = tmp_path / "config.yaml"
        versionable.save(obj, p)
        loaded = versionable.load(SimpleConfig, p)
        assert loaded.name == "test"
        assert loaded.debug is True
        assert loaded.retries == 5

    def test_withDefaults(self, tmp_path: Path) -> None:
        obj = SimpleConfig(name="default")
        p = tmp_path / "config.yaml"
        versionable.save(obj, p)
        loaded = versionable.load(SimpleConfig, p)
        assert loaded.debug is False
        assert loaded.retries == 3

    def test_optional_present(self, tmp_path: Path) -> None:
        obj = WithOptional(label="test", description="hello")
        p = tmp_path / "out.yaml"
        versionable.save(obj, p)
        loaded = versionable.load(WithOptional, p)
        assert loaded.description == "hello"

    def test_optional_none(self, tmp_path: Path) -> None:
        obj = WithOptional(label="test")
        p = tmp_path / "out.yaml"
        versionable.save(obj, p)
        loaded = versionable.load(WithOptional, p)
        assert loaded.description is None

    def test_withArray(self, tmp_path: Path) -> None:
        arr = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        obj = WithArray(name="data", data=arr)
        p = tmp_path / "out.yaml"
        versionable.save(obj, p)
        loaded = versionable.load(WithArray, p)
        assert loaded.name == "data"
        np.testing.assert_array_equal(loaded.data, arr)

    def test_withEnum(self, tmp_path: Path) -> None:
        obj = WithEnum(title="task", priority=Priority.HIGH)
        p = tmp_path / "out.yaml"
        versionable.save(obj, p)
        loaded = versionable.load(WithEnum, p)
        assert loaded.priority is Priority.HIGH

    def test_withDatetime(self, tmp_path: Path) -> None:
        dt = datetime(2026, 3, 30, 12, 0, 0, tzinfo=UTC)
        obj = WithDatetime(label="event", createdAt=dt)
        p = tmp_path / "out.yaml"
        versionable.save(obj, p)
        loaded = versionable.load(WithDatetime, p)
        assert loaded.createdAt == dt

    def test_withNested(self, tmp_path: Path) -> None:
        obj = WithNested(name="origin", point=Inner(x=1.0, y=2.0))
        p = tmp_path / "out.yaml"
        versionable.save(obj, p)
        loaded = versionable.load(WithNested, p)
        assert isinstance(loaded.point, Inner)
        assert loaded.point.x == 1.0
        assert loaded.point.y == 2.0

    def test_withList(self, tmp_path: Path) -> None:
        obj = WithList(tags=["a", "b"], scores=[1.0, 2.0])
        p = tmp_path / "out.yaml"
        versionable.save(obj, p)
        loaded = versionable.load(WithList, p)
        assert loaded.tags == ["a", "b"]
        assert loaded.scores == [1.0, 2.0]

    def test_emptyList(self, tmp_path: Path) -> None:
        obj = WithList(tags=[], scores=[])
        p = tmp_path / "out.yaml"
        versionable.save(obj, p)
        loaded = versionable.load(WithList, p)
        assert loaded.tags == []

    def test_ymlExtension(self, tmp_path: Path) -> None:
        """Both .yaml and .yml should work."""
        obj = SimpleConfig(name="test")
        p = tmp_path / "config.yml"
        versionable.save(obj, p)
        loaded = versionable.load(SimpleConfig, p)
        assert loaded.name == "test"


class TestYamlMetadata:
    def test_metadataInFile(self, tmp_path: Path) -> None:
        obj = SimpleConfig(name="test")
        p = tmp_path / "out.yaml"
        versionable.save(obj, p)

        data = yaml.safe_load(p.read_text())
        assert "__versionable__" in data
        meta = data["__versionable__"]
        assert meta["__OBJECT__"] == "SimpleConfig"
        assert meta["__VERSION__"] == 1
        assert "__HASH__" in meta

    def test_humanReadable(self, tmp_path: Path) -> None:
        obj = SimpleConfig(name="test")
        p = tmp_path / "out.yaml"
        versionable.save(obj, p)

        text = p.read_text()
        assert "name: test" in text


class TestYamlSkipDefaults:
    def test_defaultsOmitted(self, tmp_path: Path) -> None:
        obj = WithSkipDefaults(name="test")
        p = tmp_path / "out.yaml"
        versionable.save(obj, p)

        data = yaml.safe_load(p.read_text())
        assert "name" in data
        assert "debug" not in data
        assert "count" not in data

    def test_roundtripWithDefaults(self, tmp_path: Path) -> None:
        obj = WithSkipDefaults(name="test")
        p = tmp_path / "out.yaml"
        versionable.save(obj, p)
        loaded = versionable.load(WithSkipDefaults, p)
        assert loaded.name == "test"
        assert loaded.debug is False
        assert loaded.count == 0


class TestYamlCommentDefaults:
    def test_defaultsCommentedOut(self, tmp_path: Path) -> None:
        obj = SimpleConfig(name="test")
        p = tmp_path / "out.yaml"
        versionable.save(obj, p, commentDefaults=True)

        text = p.read_text()
        # "name" is not a default — should NOT be commented
        assert "\nname: test\n" in text or text.startswith("name: test\n")
        assert "# name:" not in text
        # "debug" and "retries" are at defaults — should be commented
        assert "# debug:" in text
        assert "# retries:" in text
        # __versionable__ should NOT be commented
        assert "\n__versionable__:\n" in text or text.startswith("__versionable__:\n")
        assert "# __versionable__:" not in text

    def test_nonDefaultsNotCommented(self, tmp_path: Path) -> None:
        obj = SimpleConfig(name="test", debug=True, retries=10)
        p = tmp_path / "out.yaml"
        versionable.save(obj, p, commentDefaults=True)

        text = p.read_text()
        assert "\ndebug: true\n" in text or text.startswith("debug: true\n")
        assert "\nretries: 10\n" in text or text.startswith("retries: 10\n")
        # None of these should be commented
        assert "# debug:" not in text
        assert "# retries:" not in text

    def test_commentedFileLoadsWithDefaults(self, tmp_path: Path) -> None:
        """Commented-out lines should be ignored by YAML parser, defaults restored."""
        obj = SimpleConfig(name="test")
        p = tmp_path / "out.yaml"
        versionable.save(obj, p, commentDefaults=True)
        loaded = versionable.load(SimpleConfig, p)
        assert loaded.name == "test"
        assert loaded.debug is False
        assert loaded.retries == 3

    def test_listDefaultCommentedOut(self, tmp_path: Path) -> None:
        """List fields at their default should be commented as a block."""
        obj = WithList(tags=["a", "b"])  # scores defaults to []
        p = tmp_path / "out.yaml"
        versionable.save(obj, p, commentDefaults=True)

        text = p.read_text()
        # tags is not default — should not be commented
        assert "# tags:" not in text
        # scores is default (empty list) — should be commented
        assert "# scores:" in text


class TestYamlMissingVersion:
    def test_noMetaDefaultsToCurrentVersion(self, tmp_path: Path) -> None:
        """A file with no __versionable__ should load as the current version (no migrations)."""
        p = tmp_path / "plain.yaml"
        p.write_text("name: test\ndebug: true\nretries: 5\n")
        loaded = versionable.load(SimpleConfig, p)
        assert loaded.name == "test"
        assert loaded.debug is True
        assert loaded.retries == 5

    def test_noMetaLogsWarning(self, tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
        """A file with no __versionable__ should log a warning."""
        p = tmp_path / "plain.yaml"
        p.write_text("name: test\ndebug: false\nretries: 3\n")
        with caplog.at_level("WARNING"):
            versionable.load(SimpleConfig, p)
        assert "No __VERSION__" in caplog.text
        assert "SimpleConfig" in caplog.text

    def test_assumeVersionOverride(self, tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
        """assumeVersion should suppress the warning and use the given version."""
        p = tmp_path / "plain.yaml"
        p.write_text("name: test\ndebug: false\nretries: 3\n")
        with caplog.at_level("WARNING"):
            loaded = versionable.load(SimpleConfig, p, assumeVersion=1)
        assert loaded.name == "test"
        assert "No __VERSION__" not in caplog.text


class TestYamlLiteral:
    def test_literalRoundTrip(self, tmp_path: Path) -> None:
        obj = WithLiteral(name="test", mode="slow")
        p = tmp_path / "out.yaml"
        versionable.save(obj, p)
        loaded = versionable.load(WithLiteral, p)
        assert loaded.mode == "slow"

    def test_invalidLiteralRaises(self, tmp_path: Path) -> None:
        p = tmp_path / "out.yaml"
        p.write_text(
            yaml.dump(
                {
                    "__versionable__": {"__OBJECT__": "WithLiteral", "__VERSION__": 1, "__HASH__": ""},
                    "name": "test",
                    "mode": "banana",
                }
            )
        )
        with pytest.raises(ConverterError, match="banana"):
            versionable.load(WithLiteral, p)

    def test_validateLiteralsOptOutViaLoad(self, tmp_path: Path) -> None:
        p = tmp_path / "out.yaml"
        p.write_text(
            yaml.dump(
                {
                    "__versionable__": {"__OBJECT__": "WithLiteral", "__VERSION__": 1, "__HASH__": ""},
                    "name": "test",
                    "mode": "banana",
                }
            )
        )
        loaded = versionable.load(WithLiteral, p, validateLiterals=False)
        assert loaded.mode == "banana"

    def test_validateLiteralsOptOutViaClass(self, tmp_path: Path) -> None:
        p = tmp_path / "out.yaml"
        p.write_text(
            yaml.dump(
                {
                    "__versionable__": {"__OBJECT__": "WithLiteralNoValidation", "__VERSION__": 1, "__HASH__": ""},
                    "name": "test",
                    "mode": "banana",
                }
            )
        )
        loaded = versionable.load(WithLiteralNoValidation, p)
        assert loaded.mode == "banana"

    def test_literalFallbackWarns(self, tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
        """literalFallback should warn and return the fallback value."""
        p = tmp_path / "out.yaml"
        p.write_text(
            yaml.dump(
                {
                    "__versionable__": {"__OBJECT__": "WithLiteralFallback", "__VERSION__": 1, "__HASH__": ""},
                    "name": "test",
                    "mode": "banana",
                }
            )
        )
        with caplog.at_level("WARNING"):
            loaded = versionable.load(WithLiteralFallback, p)
        assert loaded.mode == "fast"
        assert "banana" in caplog.text

    def test_literalFallbackRoundTrip(self, tmp_path: Path) -> None:
        """Valid values should round-trip normally with literalFallback."""
        obj = WithLiteralFallback(name="test", mode="slow")
        p = tmp_path / "out.yaml"
        versionable.save(obj, p)
        loaded = versionable.load(WithLiteralFallback, p)
        assert loaded.mode == "slow"


class TestYamlErrors:
    def test_missingFile(self, tmp_path: Path) -> None:
        with pytest.raises(BackendError):
            versionable.load(SimpleConfig, tmp_path / "nonexistent.yaml")

    def test_missingRequiredField(self, tmp_path: Path) -> None:
        """Missing required field should give a clear error with file path and field name."""
        p = tmp_path / "out.yaml"
        p.write_text(
            yaml.dump(
                {
                    "__versionable__": {
                        "__OBJECT__": "SimpleConfig",
                        "__VERSION__": 1,
                        "__HASH__": "",
                    },
                    "debug": True,
                    "retries": 5,
                    # "name" is missing — it has no default
                }
            )
        )
        with pytest.raises(BackendError, match="name") as exc_info:
            versionable.load(SimpleConfig, p)
        # Error should mention the file path and the class
        assert "out.yaml" in str(exc_info.value)
        assert "SimpleConfig" in str(exc_info.value)

    def test_newerVersionRaises(self, tmp_path: Path) -> None:
        p = tmp_path / "out.yaml"
        p.write_text(
            yaml.dump(
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

    def test_futureFormatRaises(self, tmp_path: Path) -> None:
        p = tmp_path / "out.yaml"
        p.write_text(
            yaml.dump(
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
