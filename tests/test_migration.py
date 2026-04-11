"""Tests for the migration system."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pytest

import versionable
from versionable import Migration, MigrationContext, Versionable, migration
from versionable._hash import computeHash
from versionable._migration import applyMigrations, resolveMigrations
from versionable.errors import MigrationError, UpgradeRequiredError


def _has_toml() -> bool:
    try:
        import toml  # noqa: F401

        return True
    except ImportError:
        return False


class TestDeclarativeOperations:
    def test_rename(self) -> None:
        mig = Migration().rename("old", "new")
        result = applyMigrations({"old": 42}, [mig])
        assert result == {"new": 42}

    def test_add(self) -> None:
        mig = Migration().add("extra", default=99)
        result = applyMigrations({"name": "test"}, [mig])
        assert result == {"name": "test", "extra": 99}

    def test_addDoesNotOverwrite(self) -> None:
        mig = Migration().add("name", default="default")
        result = applyMigrations({"name": "existing"}, [mig])
        assert result == {"name": "existing"}

    def test_addWithCallableDefault(self) -> None:
        mig = Migration().add("items", default=list)
        result = applyMigrations({}, [mig])
        assert result == {"items": []}

    def test_drop(self) -> None:
        mig = Migration().drop("old")
        result = applyMigrations({"old": 1, "keep": 2}, [mig])
        assert result == {"keep": 2}

    def test_dropMissingField(self) -> None:
        """Dropping a non-existent field is a no-op."""
        mig = Migration().drop("nonexistent")
        result = applyMigrations({"keep": 1}, [mig])
        assert result == {"keep": 1}

    def test_convert(self) -> None:
        mig = Migration().convert("temp", via=lambda c: c * 9 / 5 + 32)
        result = applyMigrations({"temp": 100}, [mig])
        assert result["temp"] == 212.0

    def test_derive(self) -> None:
        mig = Migration().derive("doubled", from_="value", via=lambda v: v * 2)
        result = applyMigrations({"value": 5}, [mig])
        assert result == {"value": 5, "doubled": 10}

    def test_split(self) -> None:
        mig = Migration().split(
            "full_name",
            into={
                "first": lambda n: n.split()[0],
                "last": lambda n: n.split()[1],
            },
        )
        result = applyMigrations({"full_name": "John Doe"}, [mig])
        assert result == {"first": "John", "last": "Doe"}

    def test_merge(self) -> None:
        mig = Migration().merge(["first", "last"], into="full", via=lambda first, last: f"{first} {last}")
        result = applyMigrations({"first": "John", "last": "Doe"}, [mig])
        assert result == {"full": "John Doe"}

    def test_chainedOps(self) -> None:
        mig = Migration().rename("title", "name").add("version", default=1).drop("old")
        result = applyMigrations({"title": "Test", "old": "junk"}, [mig])
        assert result == {"name": "Test", "version": 1}


class TestMigrationChaining:
    def test_then(self) -> None:
        m1 = Migration().rename("a", "b")
        m2 = Migration().rename("b", "c")
        combined = m1.then(m2)
        result = applyMigrations({"a": 1}, [combined])
        assert result == {"c": 1}

    def test_multiStepChain(self) -> None:
        """v1 -> v2 -> v3 chain."""
        v1_to_v2 = Migration().rename("title", "name")
        v2_to_v3 = Migration().add("count", default=0)

        result = applyMigrations({"title": "Test"}, [v1_to_v2, v2_to_v3])
        assert result == {"name": "Test", "count": 0}


class TestImperativeMigration:
    def test_basic(self) -> None:
        @migration(fromVersion=1)
        def from_v1(ctx: MigrationContext) -> None:
            ctx["new_field"] = ctx.pop("old_field") * 2

        result = applyMigrations({"old_field": 5}, [from_v1])
        assert result == {"new_field": 10}

    def test_conditionalLogic(self) -> None:
        @migration(fromVersion=1)
        def from_v1(ctx: MigrationContext) -> None:
            if "mode" in ctx and ctx["mode"] == "legacy":
                ctx["value"] = ctx["value"] * 1000
            ctx.drop("mode")

        result = applyMigrations({"mode": "legacy", "value": 5}, [from_v1])
        assert result == {"value": 5000}


class TestRequiresUpgrade:
    def test_raisesWithoutFlag(self) -> None:
        mig = Migration().requiresUpgrade()
        with pytest.raises(UpgradeRequiredError):
            applyMigrations({}, [mig])

    def test_allowsWithFlag(self) -> None:
        mig = Migration().requiresUpgrade().add("x", default=1)
        result = applyMigrations({}, [mig], upgradeInPlace=True)
        assert result == {"x": 1}


class TestResolveMigrations:
    def test_resolvesFromMigrateClass(self) -> None:
        @dataclass
        class Sample(Versionable, version=3, register=False):
            name: str
            count: int

            class Migrate:
                v1 = Migration().rename("title", "name")
                v2 = Migration().add("count", default=0)

        chain = resolveMigrations(Sample, fromVersion=1, toVersion=3)
        assert len(chain) == 2

    def test_missingMigrationRaises(self) -> None:
        @dataclass
        class Sample(Versionable, version=3, register=False):
            name: str

            class Migrate:
                v1 = Migration().rename("old", "name")
                # Missing v2!

        with pytest.raises(MigrationError, match="No migration from v2"):
            resolveMigrations(Sample, fromVersion=1, toVersion=3)

    def test_noMigrateClassRaises(self) -> None:
        @dataclass
        class Sample(Versionable, version=2, register=False):
            name: str

        with pytest.raises(MigrationError, match="no Migrate class"):
            resolveMigrations(Sample, fromVersion=1, toVersion=2)


class TestEndToEndMigration:
    def test_loadOldVersionJson(self, tmp_path: Path) -> None:
        """Load a v1 file with a v2 class that has a migration."""
        h = computeHash({"name": str, "count": int})

        @dataclass
        class Config(Versionable, version=2, hash=h, name="MigConfig", register=True):
            name: str
            count: int = 0

            class Migrate:
                v1 = Migration().add("count", default=42)

        # Write a v1 file manually
        p = tmp_path / "old.json"
        p.write_text(
            json.dumps(
                {
                    "__versionable__": {
                        "__OBJECT__": "MigConfig",
                        "__VERSION__": 1,
                        "__HASH__": "",
                    },
                    "name": "old-config",
                }
            )
        )

        loaded = versionable.load(Config, p)
        assert loaded.name == "old-config"
        assert loaded.count == 42  # Filled by migration

    def test_loadWithRename(self, tmp_path: Path) -> None:
        h = computeHash({"name": str})

        @dataclass
        class Doc(Versionable, version=2, hash=h, name="MigDoc", register=True):
            name: str

            class Migrate:
                v1 = Migration().rename("title", "name")

        p = tmp_path / "old.json"
        p.write_text(
            json.dumps(
                {
                    "__versionable__": {
                        "__OBJECT__": "MigDoc",
                        "__VERSION__": 1,
                        "__HASH__": "",
                    },
                    "title": "My Document",
                }
            )
        )

        loaded = versionable.load(Doc, p)
        assert loaded.name == "My Document"

    def test_multiVersionMigration(self, tmp_path: Path) -> None:
        h = computeHash({"name": str, "retries": int})

        @dataclass
        class AppConfig(Versionable, version=3, hash=h, name="MigApp", register=True):
            name: str
            retries: int = 3

            class Migrate:
                v1 = Migration().rename("title", "name")
                v2 = Migration().add("retries", default=3)

        p = tmp_path / "v1.json"
        p.write_text(
            json.dumps(
                {
                    "__versionable__": {
                        "__OBJECT__": "MigApp",
                        "__VERSION__": 1,
                        "__HASH__": "",
                    },
                    "title": "Old App",
                }
            )
        )

        loaded = versionable.load(AppConfig, p)
        assert loaded.name == "Old App"
        assert loaded.retries == 3


@pytest.mark.skipif(
    not _has_toml(),
    reason="toml not installed",
)
class TestAddDefaultBehavior:
    """Verify that Migration().add() injects the default only when the field is absent.

    Based on the WorkerConfig v4 example in the migrations docs.
    """

    @pytest.fixture(scope="class")
    def worker_config_v4(self) -> type:
        @dataclass
        class WorkerConfig(
            Versionable,
            version=4,
            hash="ea7fc2",
            name="WCAddDefault",
            register=True,
        ):
            name: str
            retries: int = 3
            timeout_s: float = 30.0

            class Migrate:
                v1 = Migration().rename("title", "name")
                v2 = Migration().drop("debug")
                v3 = Migration().add("timeout_s", default=0.0)

        return WorkerConfig

    def _write_toml(self, path: Path, content: str) -> None:
        path.write_text(content)

    def test_addInjectsDefaultForV1File(self, worker_config_v4: type, tmp_path: Path) -> None:
        """v1 file has no timeout_s — migration injects 0.0."""
        p = tmp_path / "v1.toml"
        self._write_toml(
            p,
            'title = "batch-processor"\ndebug = false\nretries = 5\n\n'
            '[__versionable__]\n__OBJECT__ = "WCAddDefault"\n__VERSION__ = 1\n__HASH__ = "5556c8"\n',
        )
        result = versionable.load(worker_config_v4, p)
        assert result.timeout_s == 0.0

    def test_addInjectsDefaultForV2File(self, worker_config_v4: type, tmp_path: Path) -> None:
        """v2 file also has no timeout_s — migration injects 0.0."""
        p = tmp_path / "v2.toml"
        self._write_toml(
            p,
            'name = "batch-processor"\ndebug = false\nretries = 5\n\n'
            '[__versionable__]\n__OBJECT__ = "WCAddDefault"\n__VERSION__ = 2\n__HASH__ = "ed3a90"\n',
        )
        result = versionable.load(worker_config_v4, p)
        assert result.timeout_s == 0.0

    def test_addDoesNotOverwriteExistingValue(self, worker_config_v4: type, tmp_path: Path) -> None:
        """v3 file already has timeout_s=7.5 — add() must not overwrite it."""
        p = tmp_path / "v3.toml"
        self._write_toml(
            p,
            'name = "batch-processor"\nretries = 5\ntimeout_s = 7.5\n\n'
            '[__versionable__]\n__OBJECT__ = "WCAddDefault"\n__VERSION__ = 3\n__HASH__ = "ea7fc2"\n',
        )
        result = versionable.load(worker_config_v4, p)
        assert result.timeout_s == 7.5
