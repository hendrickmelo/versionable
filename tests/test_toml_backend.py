"""Tests for the TOML backend."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import versionable
from versionable.errors import BackendError, ConverterError

from .conftest import (
    Inner,
    Priority,
    SimpleConfig,
    WithArray,
    WithEnum,
    WithList,
    WithLiteral,
    WithNested,
    WithOptional,
)


class TestTomlRoundTrip:
    def test_simpleConfig(self, tmp_path: Path) -> None:
        obj = SimpleConfig(name="test", debug=True, retries=5)
        p = tmp_path / "config.toml"
        versionable.save(obj, p)
        loaded = versionable.load(SimpleConfig, p)
        assert loaded.name == "test"
        assert loaded.debug is True
        assert loaded.retries == 5

    def test_withDefaults(self, tmp_path: Path) -> None:
        obj = SimpleConfig(name="default")
        p = tmp_path / "config.toml"
        versionable.save(obj, p)
        loaded = versionable.load(SimpleConfig, p)
        assert loaded.debug is False
        assert loaded.retries == 3

    def test_optional_present(self, tmp_path: Path) -> None:
        obj = WithOptional(label="test", description="hello")
        p = tmp_path / "out.toml"
        versionable.save(obj, p)
        loaded = versionable.load(WithOptional, p)
        assert loaded.description == "hello"

    def test_optional_none(self, tmp_path: Path) -> None:
        """None values are omitted in TOML; restored from defaults on load."""
        obj = WithOptional(label="test")
        p = tmp_path / "out.toml"
        versionable.save(obj, p)
        loaded = versionable.load(WithOptional, p)
        assert loaded.description is None

    def test_withEnum(self, tmp_path: Path) -> None:
        obj = WithEnum(title="task", priority=Priority.HIGH)
        p = tmp_path / "out.toml"
        versionable.save(obj, p)
        loaded = versionable.load(WithEnum, p)
        assert loaded.priority is Priority.HIGH

    def test_withList(self, tmp_path: Path) -> None:
        obj = WithList(tags=["a", "b"], scores=[1.0, 2.0])
        p = tmp_path / "out.toml"
        versionable.save(obj, p)
        loaded = versionable.load(WithList, p)
        assert loaded.tags == ["a", "b"]
        assert loaded.scores == [1.0, 2.0]

    def test_withArray(self, tmp_path: Path) -> None:
        """Numpy arrays are stored as JSON-wrapped blobs in TOML."""
        arr = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        obj = WithArray(name="data", data=arr)
        p = tmp_path / "out.toml"
        versionable.save(obj, p)
        loaded = versionable.load(WithArray, p)
        assert loaded.name == "data"
        np.testing.assert_array_equal(loaded.data, arr)

    def test_withNested(self, tmp_path: Path) -> None:
        """Nested Versionable objects are stored as JSON-wrapped blobs."""
        obj = WithNested(name="origin", point=Inner(x=1.0, y=2.0))
        p = tmp_path / "out.toml"
        versionable.save(obj, p)
        loaded = versionable.load(WithNested, p)
        assert isinstance(loaded.point, Inner)
        assert loaded.point.x == 1.0
        assert loaded.point.y == 2.0


class TestTomlMetadata:
    def test_metaTableInFile(self, tmp_path: Path) -> None:
        import toml

        obj = SimpleConfig(name="test")
        p = tmp_path / "out.toml"
        versionable.save(obj, p)

        data = toml.loads(p.read_text())
        assert "__meta__" in data
        assert data["__meta__"]["__OBJECT__"] == "SimpleConfig"
        assert data["__meta__"]["__VERSION__"] == 1

    def test_nestedUsesNativeTable(self, tmp_path: Path) -> None:
        """Nested Versionable should use TOML table syntax, not JSON wrapper."""
        import toml

        obj = WithNested(name="origin", point=Inner(x=1.0, y=2.0))
        p = tmp_path / "out.toml"
        versionable.save(obj, p)

        data = toml.loads(p.read_text())
        # point should be a native TOML table, not a __json__ wrapper
        assert isinstance(data["point"], dict)
        assert "__json__" not in data["point"]
        assert data["point"]["__OBJECT__"] == "Inner"
        assert data["point"]["x"] == 1.0

        # Verify [point] section appears in the raw text
        text = p.read_text()
        assert "[point]" in text

    def test_humanReadable(self, tmp_path: Path) -> None:
        """TOML output should be readable text."""
        obj = SimpleConfig(name="myapp", debug=True, retries=10)
        p = tmp_path / "out.toml"
        versionable.save(obj, p)

        text = p.read_text()
        assert 'name = "myapp"' in text
        assert "debug = true" in text
        assert "retries = 10" in text


class TestTomlCommentDefaults:
    def test_defaultsCommentedOut(self, tmp_path: Path) -> None:
        obj = SimpleConfig(name="test")
        p = tmp_path / "out.toml"
        versionable.save(obj, p, commentDefaults=True)

        text = p.read_text()
        # "name" is not a default — should NOT be commented
        assert 'name = "test"' in text
        assert '# name = "test"' not in text
        # "debug" and "retries" are at defaults — should be commented
        assert "# debug = false" in text
        assert "# retries = 3" in text
        # __meta__ should NOT be commented
        assert "[__meta__]" in text
        assert "# [__meta__]" not in text

    def test_nonDefaultsNotCommented(self, tmp_path: Path) -> None:
        obj = SimpleConfig(name="test", debug=True, retries=10)
        p = tmp_path / "out.toml"
        versionable.save(obj, p, commentDefaults=True)

        text = p.read_text()
        assert "debug = true" in text
        assert "retries = 10" in text
        assert "# debug" not in text
        assert "# retries" not in text

    def test_commentedFileLoadsWithDefaults(self, tmp_path: Path) -> None:
        """Commented-out lines are ignored by TOML parser, defaults restored."""
        obj = SimpleConfig(name="test")
        p = tmp_path / "out.toml"
        versionable.save(obj, p, commentDefaults=True)
        loaded = versionable.load(SimpleConfig, p)
        assert loaded.name == "test"
        assert loaded.debug is False
        assert loaded.retries == 3


class TestTomlLiteral:
    def test_literalRoundTrip(self, tmp_path: Path) -> None:
        obj = WithLiteral(name="test", mode="slow")
        p = tmp_path / "out.toml"
        versionable.save(obj, p)
        loaded = versionable.load(WithLiteral, p)
        assert loaded.mode == "slow"

    def test_invalidLiteralRaises(self, tmp_path: Path) -> None:
        import toml

        p = tmp_path / "out.toml"
        p.write_text(
            toml.dumps(
                {
                    "__meta__": {"__OBJECT__": "WithLiteral", "__VERSION__": 1, "__HASH__": ""},
                    "name": "test",
                    "mode": "banana",
                }
            )
        )
        with pytest.raises(ConverterError, match="banana"):
            versionable.load(WithLiteral, p)

    def test_validateLiteralsOptOutViaLoad(self, tmp_path: Path) -> None:
        import toml

        p = tmp_path / "out.toml"
        p.write_text(
            toml.dumps(
                {
                    "__meta__": {"__OBJECT__": "WithLiteral", "__VERSION__": 1, "__HASH__": ""},
                    "name": "test",
                    "mode": "banana",
                }
            )
        )
        loaded = versionable.load(WithLiteral, p, validateLiterals=False)
        assert loaded.mode == "banana"


class TestTomlMissingVersion:
    def test_noVersionDefaultsToCurrentVersion(self, tmp_path: Path) -> None:
        """A TOML file with no __meta__ should load as the current version."""
        p = tmp_path / "plain.toml"
        p.write_text('name = "test"\ndebug = true\nretries = 5\n')
        loaded = versionable.load(SimpleConfig, p)
        assert loaded.name == "test"
        assert loaded.debug is True
        assert loaded.retries == 5

    def test_noVersionLogsWarning(self, tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
        """A TOML file with no __meta__ should log a warning."""
        p = tmp_path / "plain.toml"
        p.write_text('name = "test"\ndebug = false\nretries = 3\n')
        with caplog.at_level("WARNING"):
            versionable.load(SimpleConfig, p)
        assert "No __VERSION__" in caplog.text
        assert "SimpleConfig" in caplog.text

    def test_assumeVersionOverride(self, tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
        """assumeVersion should suppress the warning and use the given version."""
        p = tmp_path / "plain.toml"
        p.write_text('name = "test"\ndebug = false\nretries = 3\n')
        with caplog.at_level("WARNING"):
            loaded = versionable.load(SimpleConfig, p, assumeVersion=1)
        assert loaded.name == "test"
        assert "No __VERSION__" not in caplog.text


class TestTomlErrors:
    def test_missingFile(self, tmp_path: Path) -> None:
        with pytest.raises(BackendError):
            versionable.load(SimpleConfig, tmp_path / "nonexistent.toml")
