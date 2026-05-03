# ruff: noqa: E402 — module-level imports must come after pytest.importorskip("yaml")
# so the entire file is skipped when pyyaml is not installed.
"""Tests for the YAML backend."""

from __future__ import annotations

import pytest

yaml = pytest.importorskip("yaml")

from dataclasses import dataclass
from pathlib import Path

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
    WithList,
    WithLiteral,
    WithLiteralFallback,
    WithLiteralNoValidation,
    WithSkipDefaults,
)

if _HAS_NUMPY:

    @dataclass
    class _YamlArrLegacy(Versionable, version=1, name="YamlArrLegacy", register=True):
        """Test class for back-compat ndarray reads in YAML files."""

        label: str
        data: npt.NDArray[np.float64]


class TestYamlRoundTrip:
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
        assert meta["object"] == "SimpleConfig"
        assert meta["version"] == 1
        assert "hash" in meta

    def test_humanReadable(self, tmp_path: Path) -> None:
        obj = SimpleConfig(name="test")
        p = tmp_path / "out.yaml"
        versionable.save(obj, p)

        text = p.read_text()
        assert "name: test" in text

    def test_nestedHasWrappedEnvelope(self, tmp_path: Path) -> None:
        """Nested Versionable values get their own ``__versionable__`` envelope."""
        from .conftest import Inner, WithNested

        obj = WithNested(name="origin", point=Inner(x=1.0, y=2.0))
        p = tmp_path / "out.yaml"
        versionable.save(obj, p)

        data = yaml.safe_load(p.read_text())
        assert data["__versionable__"]["object"] == "WithNested"
        assert "__versionable__" in data["point"]
        assert data["point"]["__versionable__"]["object"] == "Inner"
        assert "object" not in data["point"]
        loaded = versionable.load(WithNested, p)
        assert loaded.point.x == 1.0
        assert loaded.point.y == 2.0


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
        assert "No version" in caplog.text
        assert "SimpleConfig" in caplog.text

    def test_assumeVersionOverride(self, tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
        """assumeVersion should suppress the warning and use the given version."""
        p = tmp_path / "plain.yaml"
        p.write_text("name: test\ndebug: false\nretries: 3\n")
        with caplog.at_level("WARNING"):
            loaded = versionable.load(SimpleConfig, p, assumeVersion=1)
        assert loaded.name == "test"
        assert "No version" not in caplog.text


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
                    "__versionable__": {"object": "WithLiteral", "version": 1, "hash": ""},
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
                    "__versionable__": {"object": "WithLiteral", "version": 1, "hash": ""},
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
                    "__versionable__": {"object": "WithLiteralNoValidation", "version": 1, "hash": ""},
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
                    "__versionable__": {"object": "WithLiteralFallback", "version": 1, "hash": ""},
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
                        "object": "SimpleConfig",
                        "version": 1,
                        "hash": "",
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
        p = tmp_path / "out.yaml"
        p.write_text(
            yaml.dump(
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


class TestYamlBackCompat:
    """Read-side compatibility for files written by versionable 0.1.x."""

    def test_loadOldFormatEnvelope(self, tmp_path: Path) -> None:
        """A file with the legacy __OBJECT__/__VERSION__/__HASH__ keys still loads."""
        p = tmp_path / "old.yaml"
        p.write_text(
            yaml.dump(
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
        p = tmp_path / "old_future.yaml"
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

    @pytest.mark.skipif(not _HAS_NUMPY, reason="numpy not installed")
    def test_loadOldFormatNdarray(self, tmp_path: Path) -> None:
        """A file with the legacy __json__ wrapper and __ndarray__ sentinel loads."""
        obj = _YamlArrLegacy(label="legacy-array", data=np.array([4.0, 5.0, 6.0], dtype=np.float64))
        p = tmp_path / "old_arr.yaml"
        versionable.save(obj, p)
        # Rewrite both the envelope keys and the YAML/ndarray sentinels to legacy form.
        text = p.read_text()
        legacy = (
            text.replace("object:", "__OBJECT__:")
            .replace("version:", "__VERSION__:")
            .replace("hash:", "__HASH__:")
            .replace("__ver_json__:", "__json__:")
            .replace("__ver_ndarray__", "__ndarray__")
        )
        p.write_text(legacy)

        loaded = versionable.load(_YamlArrLegacy, p)
        assert loaded.label == "legacy-array"
        np.testing.assert_array_equal(loaded.data, np.array([4.0, 5.0, 6.0]))
