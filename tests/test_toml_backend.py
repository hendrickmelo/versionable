# so the entire file is skipped when toml is not installed.
"""Tests for the TOML backend."""

from __future__ import annotations

import pytest

pytest.importorskip("tomlkit")

from dataclasses import dataclass, field
from pathlib import Path

try:
    import numpy as np
    import numpy.typing as npt

    _HAS_NUMPY = True
except ImportError:
    _HAS_NUMPY = False

import versionable
from versionable._base import Versionable
from versionable.errors import BackendError, ConverterError

from .conftest import (
    Inner,
    SimpleConfig,
    WithLiteral,
    WithNested,
)

if _HAS_NUMPY:

    @dataclass
    class _TomlArrLegacy(Versionable, version=1, name="TomlArrLegacy", register=True):
        """Test class for back-compat ndarray reads in TOML files."""

        label: str
        data: npt.NDArray[np.float64]


# Module-level helpers for nested commentDefaults tests.  Defining these
# at module scope (not inside a test method) lets `typing.get_type_hints`
# resolve the forward reference under `from __future__ import annotations`,
# so the nested class is recognized and its default-fields get commented.
# Hashes are pinned literals per project convention; if you change a field
# type below, expect a HashMismatchError telling you the new hash.
@dataclass
class _NestedHost(Versionable, version=1, hash="1c4d34", register=False):
    host: str = "localhost"
    port: int = 5432


@dataclass
class _NestedRoot(Versionable, version=1, hash="f4760d", register=False):
    label: str
    db: _NestedHost = field(default_factory=_NestedHost)


class TestTomlMetadata:
    def test_metaTableInFile(self, tmp_path: Path) -> None:
        import tomlkit

        obj = SimpleConfig(name="test")
        p = tmp_path / "out.toml"
        versionable.save(obj, p)

        data = tomlkit.parse(p.read_text()).unwrap()
        assert "__versionable__" in data
        assert data["__versionable__"]["object"] == "SimpleConfig"
        assert data["__versionable__"]["version"] == 1

    def test_nestedUsesNativeTable(self, tmp_path: Path) -> None:
        """Nested Versionable should use TOML table syntax, not JSON wrapper."""
        import tomlkit

        obj = WithNested(name="origin", point=Inner(x=1.0, y=2.0))
        p = tmp_path / "out.toml"
        versionable.save(obj, p)

        data = tomlkit.parse(p.read_text()).unwrap()
        # point should be a native TOML table, not a __ver_json__ wrapper
        assert isinstance(data["point"], dict)
        assert "__ver_json__" not in data["point"]
        assert data["point"]["__versionable__"]["object"] == "Inner"
        assert data["point"]["x"] == 1.0

        # Verify [point] and the wrapped envelope sub-table appear in the raw text
        text = p.read_text()
        assert "[point]" in text
        assert "[point.__versionable__]" in text

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
        # __versionable__ should NOT be commented
        assert "[__versionable__]" in text
        assert "# [__versionable__]" not in text

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

    def test_nestedSectionHeadersNotCommented(self, tmp_path: Path) -> None:
        """Nested Versionable section headers and their __versionable__ sub-tables stay uncommented."""
        obj = WithNested(name="test", point=Inner(x=1.0, y=2.0))
        p = tmp_path / "out.toml"
        versionable.save(obj, p, commentDefaults=True)

        text = p.read_text()
        # Section headers must never be commented
        assert "[point]" in text
        assert "# [point]" not in text
        # The nested envelope sub-table and its keys must never be commented
        assert "[point.__versionable__]" in text
        assert "# [point.__versionable__]" not in text
        assert 'object = "Inner"' in text
        assert "# object" not in text

    def test_commentDefaultsEmitsValidToml(self, tmp_path: Path) -> None:
        """commentDefaults output is parseable as valid TOML and round-trips."""
        import tomlkit

        obj = SimpleConfig(name="test")
        p = tmp_path / "out.toml"
        versionable.save(obj, p, commentDefaults=True)

        # Parsing should not raise — the file with commented defaults is valid TOML
        parsed = tomlkit.parse(p.read_text()).unwrap()
        assert parsed["__versionable__"]["object"] == "SimpleConfig"
        assert parsed["name"] == "test"
        # Commented defaults are not present as keys
        assert "debug" not in parsed
        assert "retries" not in parsed

    def test_uncommentingDefaultRoundTrips(self, tmp_path: Path) -> None:
        """A user uncommenting a default line must produce a valid override at the right path."""
        obj = SimpleConfig(name="test")
        p = tmp_path / "out.toml"
        versionable.save(obj, p, commentDefaults=True)

        # Simulate a user uncommenting "# debug = false" and changing the value
        text = p.read_text()
        assert "# debug = false" in text  # baseline
        modified = text.replace("# debug = false", "debug = true")
        p.write_text(modified)

        loaded = versionable.load(SimpleConfig, p)
        assert loaded.debug is True, (
            f"uncommented `debug = true` did not override default; file content was:\n{p.read_text()}"
        )

    def test_uncommentingNestedDefaultRoundTrips(self, tmp_path: Path) -> None:
        """Same as above but for a field inside a nested Versionable."""
        obj = _NestedRoot(label="x")
        p = tmp_path / "out.toml"
        versionable.save(obj, p, commentDefaults=True)

        text = p.read_text()
        assert "# host = " in text
        modified = text.replace('# host = "localhost"', 'host = "elsewhere"')
        p.write_text(modified)

        loaded = versionable.load(_NestedRoot, p)
        assert loaded.db.host == "elsewhere", (
            f"uncommented nested host did not override default; file content was:\n{p.read_text()}"
        )


class TestTomlLiteral:
    def test_literalRoundTrip(self, tmp_path: Path) -> None:
        obj = WithLiteral(name="test", mode="slow")
        p = tmp_path / "out.toml"
        versionable.save(obj, p)
        loaded = versionable.load(WithLiteral, p)
        assert loaded.mode == "slow"

    def test_invalidLiteralRaises(self, tmp_path: Path) -> None:
        import tomlkit

        p = tmp_path / "out.toml"
        p.write_text(
            tomlkit.dumps(
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
        import tomlkit

        p = tmp_path / "out.toml"
        p.write_text(
            tomlkit.dumps(
                {
                    "__versionable__": {"object": "WithLiteral", "version": 1, "hash": ""},
                    "name": "test",
                    "mode": "banana",
                }
            )
        )
        loaded = versionable.load(WithLiteral, p, validateLiterals=False)
        assert loaded.mode == "banana"


class TestTomlMissingVersion:
    def test_noVersionDefaultsToCurrentVersion(self, tmp_path: Path) -> None:
        """A TOML file with no __versionable__ should load as the current version."""
        p = tmp_path / "plain.toml"
        p.write_text('name = "test"\ndebug = true\nretries = 5\n')
        loaded = versionable.load(SimpleConfig, p)
        assert loaded.name == "test"
        assert loaded.debug is True
        assert loaded.retries == 5

    def test_noVersionLogsWarning(self, tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
        """A TOML file with no __versionable__ should log a warning."""
        p = tmp_path / "plain.toml"
        p.write_text('name = "test"\ndebug = false\nretries = 3\n')
        with caplog.at_level("WARNING"):
            versionable.load(SimpleConfig, p)
        assert "No version" in caplog.text
        assert "SimpleConfig" in caplog.text

    def test_assumeVersionOverride(self, tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
        """assumeVersion should suppress the warning and use the given version."""
        p = tmp_path / "plain.toml"
        p.write_text('name = "test"\ndebug = false\nretries = 3\n')
        with caplog.at_level("WARNING"):
            loaded = versionable.load(SimpleConfig, p, assumeVersion=1)
        assert loaded.name == "test"
        assert "No version" not in caplog.text


class TestTomlErrors:
    def test_missingFile(self, tmp_path: Path) -> None:
        with pytest.raises(BackendError):
            versionable.load(SimpleConfig, tmp_path / "nonexistent.toml")

    def test_futureFormatRaises(self, tmp_path: Path) -> None:
        p = tmp_path / "out.toml"
        p.write_text(
            '[__versionable__]\nobject = "SimpleConfig"\nversion = 1\nhash = ""\nformat = 2\n\nname = "test"\n'
        )
        with pytest.raises(BackendError, match="Upgrade versionable"):
            versionable.load(SimpleConfig, p)


class TestTomlBackCompat:
    """Read-side compatibility for files written by versionable 0.1.x."""

    def test_loadOldFormatEnvelope(self, tmp_path: Path) -> None:
        """A file with the legacy __OBJECT__/__VERSION__/__HASH__ keys still loads."""
        p = tmp_path / "old.toml"
        p.write_text(
            'name = "legacy"\ndebug = true\nretries = 7\n\n'
            '[__versionable__]\n__OBJECT__ = "SimpleConfig"\n__VERSION__ = 1\n__HASH__ = "ed3a90"\n'
        )
        loaded = versionable.load(SimpleConfig, p)
        assert loaded.name == "legacy"
        assert loaded.debug is True
        assert loaded.retries == 7

    def test_oldFormatFutureFormatRaises(self, tmp_path: Path) -> None:
        """The legacy __FORMAT__ key still triggers the upgrade-required error."""
        p = tmp_path / "old_future.toml"
        p.write_text(
            '[__versionable__]\n__OBJECT__ = "SimpleConfig"\n'
            '__VERSION__ = 1\n__HASH__ = ""\n__FORMAT__ = 2\n\n'
            'name = "test"\n'
        )
        with pytest.raises(BackendError, match="Upgrade versionable"):
            versionable.load(SimpleConfig, p)

    @pytest.mark.skipif(not _HAS_NUMPY, reason="numpy not installed")
    def test_loadOldFormatNdarray(self, tmp_path: Path) -> None:
        """A file with the legacy __json__ wrapper and __ndarray__ sentinel loads."""
        obj = _TomlArrLegacy(label="legacy-array", data=np.array([7.0, 8.0, 9.0], dtype=np.float64))
        p = tmp_path / "old_arr.toml"
        versionable.save(obj, p)
        text = p.read_text()
        legacy = (
            text.replace("object =", "__OBJECT__ =")
            .replace("version =", "__VERSION__ =")
            .replace("hash =", "__HASH__ =")
            .replace("__ver_json__ =", "__json__ =")
            .replace("__ver_ndarray__", "__ndarray__")
        )
        p.write_text(legacy)

        loaded = versionable.load(_TomlArrLegacy, p)
        assert loaded.label == "legacy-array"
        np.testing.assert_array_equal(loaded.data, np.array([7.0, 8.0, 9.0]))


def test_dependencyImport() -> None:
    """tomlkit imports cleanly — guardrail against accidental dep removal."""
    import tomlkit

    assert hasattr(tomlkit, "parse")
    assert hasattr(tomlkit, "dumps")
