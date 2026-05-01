"""Tests for nested migration recursion, polymorphism, and related behaviors.

Each test isolates its registry mutations via the autouse fixture below — the
nested-migration test pattern routinely redefines the same class name at v1
then v2 within a single test, which would collide with the global `_REGISTRY`
otherwise.

Hashes are omitted (default ``""``) on test classes, matching the existing
`test_migration.py` convention; the hash check is skipped when hash is empty.

Note: this file intentionally does not use ``from __future__ import
annotations``. Tests define Versionable classes inside functions and reference
each other by name in type annotations (e.g., ``items: list[_BV2]``).
With deferred annotations active, ``typing.get_type_hints`` would try to
resolve these class names against the module's globals (where they don't exist
— they're function-local) and fall back to raw string annotations, which
``_deserializeConcrete`` can't dispatch through. Evaluating annotations
eagerly captures the class objects at definition time, when they're in scope.
"""

import json
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import pytest

import versionable
from versionable import Migration, MigrationContext, Versionable, migration
from versionable._base import _REGISTRY
from versionable.errors import (
    BackendError,
    ConverterError,
    UnknownFieldError,
    VersionError,
)


def _has_toml() -> bool:
    try:
        import toml  # noqa: F401

        return True
    except ImportError:
        return False


def _has_yaml() -> bool:
    try:
        import yaml  # noqa: F401

        return True
    except ImportError:
        return False


def _has_h5py() -> bool:
    try:
        import h5py  # noqa: F401

        return True
    except ImportError:
        return False


@pytest.fixture(autouse=True)
def _isolated_registry() -> Iterator[None]:
    """Snapshot _REGISTRY before each test, restore after.

    Allows tests to redefine the same class name at different versions (e.g., B
    at v1 to write a file, then B at v2 to load it) without polluting other
    tests' state.
    """
    before = dict(_REGISTRY)
    yield
    _REGISTRY.clear()
    _REGISTRY.update(before)


# ---------------------------------------------------------------------------
# Migration recursion
# ---------------------------------------------------------------------------


class TestSingleNestedMigration:
    """Test 1: single-field nested migration — A.point: B, save B-v1, load B-v2.

    Covered per backend to confirm each delivers the nested envelope in a shape
    the recursion can read.
    """

    def _save_v1_then_define_v2(self, p: Path, backend_save: callable) -> type:
        """Save a v1 file with B-v1, then redefine B at v2 with a rename migration.

        Returns the v2 container class for loading.
        """

        @dataclass
        class _BV1(Versionable, version=1, name="NMig1_B", register=True):
            title: str

        @dataclass
        class _AV1(Versionable, version=1, name="NMig1_A", register=True):
            point: _BV1

        backend_save(_AV1(point=_BV1(title="hello")), p)

        _REGISTRY.pop("NMig1_B", None)
        _REGISTRY.pop("NMig1_A", None)

        @dataclass
        class _BV2(Versionable, version=2, name="NMig1_B", register=True):
            name: str

            class Migrate:
                v1 = Migration().rename("title", "name")

        @dataclass
        class _AV1WithV2B(Versionable, version=1, name="NMig1_A", register=True):
            point: _BV2

        return _AV1WithV2B

    def test_json(self, tmp_path: Path) -> None:
        p = tmp_path / "data.json"
        a_cls = self._save_v1_then_define_v2(p, versionable.save)
        loaded = versionable.load(a_cls, p)
        assert loaded.point.name == "hello"

    @pytest.mark.skipif(not _has_yaml(), reason="yaml not installed")
    def test_yaml(self, tmp_path: Path) -> None:
        p = tmp_path / "data.yaml"
        a_cls = self._save_v1_then_define_v2(p, versionable.save)
        loaded = versionable.load(a_cls, p)
        assert loaded.point.name == "hello"

    @pytest.mark.skipif(not _has_toml(), reason="toml not installed")
    def test_toml(self, tmp_path: Path) -> None:
        p = tmp_path / "data.toml"
        a_cls = self._save_v1_then_define_v2(p, versionable.save)
        loaded = versionable.load(a_cls, p)
        assert loaded.point.name == "hello"

    @pytest.mark.skipif(not _has_h5py(), reason="h5py not installed")
    def test_hdf5(self, tmp_path: Path) -> None:
        p = tmp_path / "data.h5"
        a_cls = self._save_v1_then_define_v2(p, versionable.save)
        loaded = versionable.load(a_cls, p)
        assert loaded.point.name == "hello"


def test_list_element_migration(tmp_path: Path) -> None:
    """Test 2: each element of list[B] gets migrated."""
    p = tmp_path / "data.json"

    @dataclass
    class _BV1(Versionable, version=1, name="NMig2_B", register=True):
        title: str

    @dataclass
    class _AV1(Versionable, version=1, name="NMig2_A", register=True):
        items: list[_BV1]

    versionable.save(_AV1(items=[_BV1(title="a"), _BV1(title="b")]), p)

    _REGISTRY.pop("NMig2_B", None)
    _REGISTRY.pop("NMig2_A", None)

    @dataclass
    class _BV2(Versionable, version=2, name="NMig2_B", register=True):
        name: str

        class Migrate:
            v1 = Migration().rename("title", "name")

    @dataclass
    class _AV1WithV2B(Versionable, version=1, name="NMig2_A", register=True):
        items: list[_BV2]

    loaded = versionable.load(_AV1WithV2B, p)
    assert [b.name for b in loaded.items] == ["a", "b"]


def test_dict_value_migration(tmp_path: Path) -> None:
    """Test 3: each value of dict[str, B] gets migrated."""
    p = tmp_path / "data.json"

    @dataclass
    class _BV1(Versionable, version=1, name="NMig3_B", register=True):
        title: str

    @dataclass
    class _AV1(Versionable, version=1, name="NMig3_A", register=True):
        by_key: dict[str, _BV1]

    versionable.save(_AV1(by_key={"x": _BV1(title="hello")}), p)

    _REGISTRY.pop("NMig3_B", None)
    _REGISTRY.pop("NMig3_A", None)

    @dataclass
    class _BV2(Versionable, version=2, name="NMig3_B", register=True):
        name: str

        class Migrate:
            v1 = Migration().rename("title", "name")

    @dataclass
    class _AV1WithV2B(Versionable, version=1, name="NMig3_A", register=True):
        by_key: dict[str, _BV2]

    loaded = versionable.load(_AV1WithV2B, p)
    assert loaded.by_key["x"].name == "hello"


def test_tuple_element_migration(tmp_path: Path) -> None:
    """Test 4: each element of tuple[B, ...] gets migrated."""
    p = tmp_path / "data.json"

    @dataclass
    class _BV1(Versionable, version=1, name="NMig4_B", register=True):
        title: str

    @dataclass
    class _AV1(Versionable, version=1, name="NMig4_A", register=True):
        items: tuple[_BV1, ...]

    versionable.save(_AV1(items=(_BV1(title="a"), _BV1(title="b"))), p)

    _REGISTRY.pop("NMig4_B", None)
    _REGISTRY.pop("NMig4_A", None)

    @dataclass
    class _BV2(Versionable, version=2, name="NMig4_B", register=True):
        name: str

        class Migrate:
            v1 = Migration().rename("title", "name")

    @dataclass
    class _AV1WithV2B(Versionable, version=1, name="NMig4_A", register=True):
        items: tuple[_BV2, ...]

    loaded = versionable.load(_AV1WithV2B, p)
    assert isinstance(loaded.items, tuple)
    assert [b.name for b in loaded.items] == ["a", "b"]


def test_set_element_migration(tmp_path: Path) -> None:
    """Test 5: each element of set[B] gets migrated. B must be hashable."""
    p = tmp_path / "data.json"

    @dataclass(frozen=True)
    class _BV1(Versionable, version=1, name="NMig5_B", register=True):
        title: str

    @dataclass
    class _AV1(Versionable, version=1, name="NMig5_A", register=True):
        items: set[_BV1]

    versionable.save(_AV1(items={_BV1(title="a"), _BV1(title="b")}), p)

    _REGISTRY.pop("NMig5_B", None)
    _REGISTRY.pop("NMig5_A", None)

    @dataclass(frozen=True)
    class _BV2(Versionable, version=2, name="NMig5_B", register=True):
        name: str

        class Migrate:
            v1 = Migration().rename("title", "name")

    @dataclass
    class _AV1WithV2B(Versionable, version=1, name="NMig5_A", register=True):
        items: set[_BV2]

    loaded = versionable.load(_AV1WithV2B, p)
    assert {b.name for b in loaded.items} == {"a", "b"}


def test_multistep_chain_at_nested_level(tmp_path: Path) -> None:
    """Test 6: B v1 → v2 → v3 chain applied to a nested element."""
    p = tmp_path / "data.json"

    @dataclass
    class _BV1(Versionable, version=1, name="NMig6_B", register=True):
        title: str

    @dataclass
    class _AV1(Versionable, version=1, name="NMig6_A", register=True):
        b: _BV1

    versionable.save(_AV1(b=_BV1(title="hello")), p)

    _REGISTRY.pop("NMig6_B", None)
    _REGISTRY.pop("NMig6_A", None)

    @dataclass
    class _BV3(Versionable, version=3, name="NMig6_B", register=True):
        name: str
        count: int = 0

        class Migrate:
            v1 = Migration().rename("title", "name")
            v2 = Migration().add("count", default=42)

    @dataclass
    class _AV1WithV3B(Versionable, version=1, name="NMig6_A", register=True):
        b: _BV3

    loaded = versionable.load(_AV1WithV3B, p)
    assert loaded.b.name == "hello"
    assert loaded.b.count == 42


def test_parent_and_child_both_migrated(tmp_path: Path) -> None:
    """Test 7: A-v1 → A-v2 plus B-v1 → B-v2 applied at appropriate levels."""
    p = tmp_path / "data.json"

    @dataclass
    class _BV1(Versionable, version=1, name="NMig7_B", register=True):
        title: str

    @dataclass
    class _AV1(Versionable, version=1, name="NMig7_A", register=True):
        label: str
        b: _BV1

    versionable.save(_AV1(label="parent", b=_BV1(title="child")), p)

    _REGISTRY.pop("NMig7_B", None)
    _REGISTRY.pop("NMig7_A", None)

    @dataclass
    class _BV2(Versionable, version=2, name="NMig7_B", register=True):
        name: str

        class Migrate:
            v1 = Migration().rename("title", "name")

    @dataclass
    class _AV2(Versionable, version=2, name="NMig7_A", register=True):
        title: str  # renamed from "label"
        b: _BV2

        class Migrate:
            v1 = Migration().rename("label", "title")

    loaded = versionable.load(_AV2, p)
    assert loaded.title == "parent"
    assert loaded.b.name == "child"


def test_newer_nested_version_raises(tmp_path: Path) -> None:
    """Test 8: nested file version > class version raises VersionError naming the type."""
    p = tmp_path / "data.json"

    @dataclass
    class _BV2(Versionable, version=2, name="NMig8_B", register=True):
        name: str

    @dataclass
    class _AV1(Versionable, version=1, name="NMig8_A", register=True):
        b: _BV2

    versionable.save(_AV1(b=_BV2(name="x")), p)

    _REGISTRY.pop("NMig8_B", None)
    _REGISTRY.pop("NMig8_A", None)

    @dataclass
    class _BV1Only(Versionable, version=1, name="NMig8_B", register=True):
        name: str

    @dataclass
    class _AV1WithOldB(Versionable, version=1, name="NMig8_A", register=True):
        b: _BV1Only

    with pytest.raises(VersionError, match="NMig8_B"):
        versionable.load(_AV1WithOldB, p)


def test_missing_nested_envelope_warns(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    """Test 9: nested data without an envelope logs a warning and assumes current version."""
    p = tmp_path / "data.json"

    @dataclass
    class _B(Versionable, version=1, name="NMig9_B", register=True):
        name: str

    @dataclass
    class _A(Versionable, version=1, name="NMig9_A", register=True):
        b: _B

    # Hand-craft a file with no nested envelope at all (no __versionable__ wrapper,
    # no flat envelope keys).
    p.write_text(
        json.dumps(
            {
                "__versionable__": {"object": "NMig9_A", "version": 1, "hash": ""},
                "b": {"name": "x"},
            }
        )
    )
    with caplog.at_level("WARNING"):
        loaded = versionable.load(_A, p)
    assert loaded.b.name == "x"
    assert any("NMig9_B" in m and "no version" in m for m in caplog.messages)


def test_imperative_migration_on_nested_type(tmp_path: Path) -> None:
    """Test 10: imperative @migration decorator works at nested level."""
    p = tmp_path / "data.json"

    @dataclass
    class _BV1(Versionable, version=1, name="NMig10_B", register=True):
        old_field: int

    @dataclass
    class _AV1(Versionable, version=1, name="NMig10_A", register=True):
        b: _BV1

    versionable.save(_AV1(b=_BV1(old_field=5)), p)

    _REGISTRY.pop("NMig10_B", None)
    _REGISTRY.pop("NMig10_A", None)

    @dataclass
    class _BV2(Versionable, version=2, name="NMig10_B", register=True):
        new_field: int

        class Migrate:
            @migration(fromVersion=1)
            def from_v1(ctx: MigrationContext) -> None:  # noqa: N805  # @migration wraps this; not a regular method
                ctx["new_field"] = ctx.pop("old_field") * 2

    @dataclass
    class _AV1WithV2B(Versionable, version=1, name="NMig10_A", register=True):
        b: _BV2

    loaded = versionable.load(_AV1WithV2B, p)
    assert loaded.b.new_field == 10


def test_three_level_deep_migration(tmp_path: Path) -> None:
    """Test 11: A.b: B, B.c: C — all three classes migrate v1 → v2."""
    p = tmp_path / "data.json"

    @dataclass
    class _CV1(Versionable, version=1, name="NMig11_C", register=True):
        c_old: str

    @dataclass
    class _BV1(Versionable, version=1, name="NMig11_B", register=True):
        b_old: str
        c: _CV1

    @dataclass
    class _AV1(Versionable, version=1, name="NMig11_A", register=True):
        a_old: str
        b: _BV1

    versionable.save(_AV1(a_old="A", b=_BV1(b_old="B", c=_CV1(c_old="C"))), p)

    _REGISTRY.pop("NMig11_C", None)
    _REGISTRY.pop("NMig11_B", None)
    _REGISTRY.pop("NMig11_A", None)

    @dataclass
    class _CV2(Versionable, version=2, name="NMig11_C", register=True):
        c_new: str

        class Migrate:
            v1 = Migration().rename("c_old", "c_new")

    @dataclass
    class _BV2(Versionable, version=2, name="NMig11_B", register=True):
        b_new: str
        c: _CV2

        class Migrate:
            v1 = Migration().rename("b_old", "b_new")

    @dataclass
    class _AV2(Versionable, version=2, name="NMig11_A", register=True):
        a_new: str
        b: _BV2

        class Migrate:
            v1 = Migration().rename("a_old", "a_new")

    loaded = versionable.load(_AV2, p)
    assert loaded.a_new == "A"
    assert loaded.b.b_new == "B"
    assert loaded.b.c.c_new == "C"


@pytest.mark.skipif(not _has_yaml() or not _has_toml(), reason="yaml or toml not installed")
def test_cross_backend_yaml_to_toml(tmp_path: Path) -> None:
    """Test 12: save in YAML at v1, load in TOML at v2 — migrations apply regardless of backend."""
    yaml_path = tmp_path / "v1.yaml"
    toml_path = tmp_path / "v1.toml"

    @dataclass
    class _BV1(Versionable, version=1, name="NMig12_B", register=True):
        title: str

    @dataclass
    class _AV1(Versionable, version=1, name="NMig12_A", register=True):
        b: _BV1

    obj = _AV1(b=_BV1(title="hello"))
    versionable.save(obj, yaml_path)

    # Re-read YAML, re-write as TOML to exercise cross-backend migration.
    import yaml as yaml_lib

    yaml_data = yaml_lib.safe_load(yaml_path.read_text())
    import toml as toml_lib

    toml_path.write_text(toml_lib.dumps(yaml_data))

    _REGISTRY.pop("NMig12_B", None)
    _REGISTRY.pop("NMig12_A", None)

    @dataclass
    class _BV2(Versionable, version=2, name="NMig12_B", register=True):
        name: str

        class Migrate:
            v1 = Migration().rename("title", "name")

    @dataclass
    class _AV1WithV2B(Versionable, version=1, name="NMig12_A", register=True):
        b: _BV2

    loaded = versionable.load(_AV1WithV2B, toml_path)
    assert loaded.b.name == "hello"


@pytest.mark.skipif(not _has_h5py(), reason="h5py not installed")
def test_hdf5_nested_migration_via_load_lazy(tmp_path: Path) -> None:
    """Test 13: nested migration through HDF5's loadLazy reader path."""
    p = tmp_path / "data.h5"

    @dataclass
    class _BV1(Versionable, version=1, name="NMig13_B", register=True):
        title: str

    @dataclass
    class _AV1(Versionable, version=1, name="NMig13_A", register=True):
        items: list[_BV1]

    versionable.save(_AV1(items=[_BV1(title="a"), _BV1(title="b")]), p)

    _REGISTRY.pop("NMig13_B", None)
    _REGISTRY.pop("NMig13_A", None)

    @dataclass
    class _BV2(Versionable, version=2, name="NMig13_B", register=True):
        name: str

        class Migrate:
            v1 = Migration().rename("title", "name")

    @dataclass
    class _AV1WithV2B(Versionable, version=1, name="NMig13_A", register=True):
        items: list[_BV2]

    loaded = versionable.load(_AV1WithV2B, p)
    assert [b.name for b in loaded.items] == ["a", "b"]


# ---------------------------------------------------------------------------
# Polymorphism
# ---------------------------------------------------------------------------


class TestPolymorphism:
    """Tests 14, 16, 17: polymorphic resolution, errors, and runtime preservation."""

    def test_subclass_identity_preserved_json(self, tmp_path: Path) -> None:
        p = tmp_path / "data.json"

        @dataclass
        class _Animal(Versionable, version=1, name="NPoly14_Animal", register=True):
            name: str

        @dataclass
        class _Dog(_Animal, version=1, name="NPoly14_Dog", register=True):
            breed: str = "mutt"

        @dataclass
        class _Cat(_Animal, version=1, name="NPoly14_Cat", register=True):
            indoor: bool = True

        @dataclass
        class _Zoo(Versionable, version=1, name="NPoly14_Zoo", register=True):
            animals: list[_Animal] = field(default_factory=list)

        zoo = _Zoo(animals=[_Dog(name="Rex", breed="lab"), _Cat(name="Whiskers", indoor=False)])
        versionable.save(zoo, p)

        loaded = versionable.load(_Zoo, p)
        assert isinstance(loaded.animals[0], _Dog)
        assert isinstance(loaded.animals[1], _Cat)
        assert loaded.animals[0].breed == "lab"
        assert loaded.animals[1].indoor is False

    @pytest.mark.skipif(not _has_h5py(), reason="h5py not installed")
    def test_subclass_identity_preserved_hdf5(self, tmp_path: Path) -> None:
        p = tmp_path / "data.h5"

        @dataclass
        class _Animal(Versionable, version=1, name="NPolyH_Animal", register=True):
            name: str

        @dataclass
        class _Dog(_Animal, version=1, name="NPolyH_Dog", register=True):
            breed: str = "mutt"

        @dataclass
        class _Cat(_Animal, version=1, name="NPolyH_Cat", register=True):
            indoor: bool = True

        @dataclass
        class _Zoo(Versionable, version=1, name="NPolyH_Zoo", register=True):
            animals: list[_Animal] = field(default_factory=list)

        zoo = _Zoo(animals=[_Dog(name="Rex", breed="poodle"), _Cat(name="Whiskers", indoor=True)])
        versionable.save(zoo, p)

        loaded = versionable.load(_Zoo, p)
        assert isinstance(loaded.animals[0], _Dog)
        assert isinstance(loaded.animals[1], _Cat)
        assert loaded.animals[0].breed == "poodle"
        assert loaded.animals[1].indoor is True

    def test_unknown_object_name_raises(self, tmp_path: Path) -> None:
        """Test 16: file references a class name not in the registry."""
        p = tmp_path / "data.json"

        @dataclass
        class _Animal(Versionable, version=1, name="NPoly16_Animal", register=True):
            name: str

        @dataclass
        class _Zoo(Versionable, version=1, name="NPoly16_Zoo", register=True):
            animals: list[_Animal] = field(default_factory=list)

        # Hand-craft a file that names a class not in the registry.
        p.write_text(
            json.dumps(
                {
                    "__versionable__": {"object": "NPoly16_Zoo", "version": 1, "hash": ""},
                    "animals": [
                        {
                            "__versionable__": {"object": "NotARealClass", "version": 1, "hash": ""},
                            "name": "Rex",
                        }
                    ],
                }
            )
        )

        with pytest.raises(BackendError, match="NotARealClass"):
            versionable.load(_Zoo, p)

    def test_resolved_class_not_subclass_raises(self, tmp_path: Path) -> None:
        """Test 17: file's object resolves to a class that's not a subclass of declared type."""
        p = tmp_path / "data.json"

        @dataclass
        class _Animal(Versionable, version=1, name="NPoly17_Animal", register=True):
            name: str

        @dataclass
        class _Vehicle(Versionable, version=1, name="NPoly17_Vehicle", register=True):
            wheels: int

        @dataclass
        class _Zoo(Versionable, version=1, name="NPoly17_Zoo", register=True):
            animals: list[_Animal] = field(default_factory=list)

        # Hand-craft a file where the nested element's object resolves to Vehicle
        # (registered) which is NOT a subclass of Animal.
        p.write_text(
            json.dumps(
                {
                    "__versionable__": {"object": "NPoly17_Zoo", "version": 1, "hash": ""},
                    "animals": [
                        {
                            "__versionable__": {"object": "NPoly17_Vehicle", "version": 1, "hash": ""},
                            "wheels": 4,
                        }
                    ],
                }
            )
        )

        with pytest.raises(BackendError, match="not a subclass"):
            versionable.load(_Zoo, p)


def test_polymorphism_with_migration(tmp_path: Path) -> None:
    """Test 15: per-subclass migrations applied to each element by its actual class."""
    p = tmp_path / "data.json"

    @dataclass
    class _Animal(Versionable, version=1, name="NPoly15_Animal", register=True):
        name: str

    @dataclass
    class _DogV1(_Animal, version=1, name="NPoly15_Dog", register=True):
        bark: str = "woof"

    @dataclass
    class _CatV1(_Animal, version=1, name="NPoly15_Cat", register=True):
        meow: str = "meow"

    @dataclass
    class _ZooV1(Versionable, version=1, name="NPoly15_Zoo", register=True):
        animals: list[_Animal] = field(default_factory=list)

    zoo = _ZooV1(
        animals=[
            _DogV1(name="Rex", bark="WOOF"),
            _CatV1(name="Whiskers", meow="MEOW"),
        ]
    )
    versionable.save(zoo, p)

    _REGISTRY.pop("NPoly15_Zoo", None)
    _REGISTRY.pop("NPoly15_Cat", None)
    _REGISTRY.pop("NPoly15_Dog", None)
    _REGISTRY.pop("NPoly15_Animal", None)

    @dataclass
    class _AnimalV1(Versionable, version=1, name="NPoly15_Animal", register=True):
        name: str

    @dataclass
    class _DogV2(_AnimalV1, version=2, name="NPoly15_Dog", register=True):
        sound: str = "woof"  # renamed from "bark"

        class Migrate:
            v1 = Migration().rename("bark", "sound")

    @dataclass
    class _CatV2(_AnimalV1, version=2, name="NPoly15_Cat", register=True):
        sound: str = "meow"  # renamed from "meow"

        class Migrate:
            v1 = Migration().rename("meow", "sound")

    @dataclass
    class _ZooV1WithV2(Versionable, version=1, name="NPoly15_Zoo", register=True):
        animals: list[_AnimalV1] = field(default_factory=list)

    loaded = versionable.load(_ZooV1WithV2, p)
    assert isinstance(loaded.animals[0], _DogV2)
    assert isinstance(loaded.animals[1], _CatV2)
    assert loaded.animals[0].sound == "WOOF"
    assert loaded.animals[1].sound == "MEOW"


def test_polymorphism_old_names_rename(tmp_path: Path) -> None:
    """Test 18: file says object='OldDog'; current class registers name='Dog', old_names=['OldDog']."""
    p = tmp_path / "data.json"

    @dataclass
    class _Animal(Versionable, version=1, name="NPoly18_Animal", register=True):
        name: str

    @dataclass
    class _DogRenamed(_Animal, version=1, name="NPoly18_Dog", old_names=["NPoly18_OldDog"], register=True):
        breed: str = "mutt"

    @dataclass
    class _Zoo(Versionable, version=1, name="NPoly18_Zoo", register=True):
        animals: list[_Animal] = field(default_factory=list)

    # Hand-craft a file using the old name.
    p.write_text(
        json.dumps(
            {
                "__versionable__": {"object": "NPoly18_Zoo", "version": 1, "hash": ""},
                "animals": [
                    {
                        "__versionable__": {"object": "NPoly18_OldDog", "version": 1, "hash": ""},
                        "name": "Rex",
                        "breed": "lab",
                    }
                ],
            }
        )
    )

    loaded = versionable.load(_Zoo, p)
    assert isinstance(loaded.animals[0], _DogRenamed)
    assert loaded.animals[0].name == "Rex"
    assert loaded.animals[0].breed == "lab"


# ---------------------------------------------------------------------------
# Save-side guard
# ---------------------------------------------------------------------------


def test_save_dict_with_versionable_keys_raises(tmp_path: Path) -> None:
    """Test 19: saving dict[Versionable, X] raises ConverterError, identifying the field path."""
    p = tmp_path / "data.json"

    @dataclass(frozen=True)
    class _K(Versionable, version=1, name="NSave19_K", register=True):
        id: int

    @dataclass
    class _A(Versionable, version=1, name="NSave19_A", register=True):
        by_obj: dict[_K, str] = field(default_factory=dict)

    obj = _A(by_obj={_K(id=1): "x"})
    with pytest.raises(ConverterError, match="Dict keys cannot be Versionable"):
        versionable.save(obj, p)


# ---------------------------------------------------------------------------
# validateLiterals propagation
# ---------------------------------------------------------------------------


def test_validate_literals_false_propagates_to_nested(tmp_path: Path) -> None:
    """Test 20: load(..., validateLiterals=False) skips Literal validation in nested Versionables."""
    p = tmp_path / "data.json"

    @dataclass
    class _Inner(Versionable, version=1, name="NLit20_Inner", register=True):
        mode: Literal["fast", "slow"] = "fast"

    @dataclass
    class _Outer(Versionable, version=1, name="NLit20_Outer", register=True):
        inner: _Inner

    # Hand-craft a file where the nested Inner has an invalid Literal value.
    p.write_text(
        json.dumps(
            {
                "__versionable__": {"object": "NLit20_Outer", "version": 1, "hash": ""},
                "inner": {
                    "__versionable__": {"object": "NLit20_Inner", "version": 1, "hash": ""},
                    "mode": "banana",  # Not a valid Literal option
                },
            }
        )
    )

    # Without override → ConverterError at nested level.
    with pytest.raises(ConverterError, match="banana"):
        versionable.load(_Outer, p)

    # With load-level override → nested Literal validation skipped, "banana" passes through.
    loaded = versionable.load(_Outer, p, validateLiterals=False)
    assert loaded.inner.mode == "banana"


def test_nested_class_validate_literals_setting_honored(tmp_path: Path) -> None:
    """Nested class's own validate_literals=False is honored when no load-level override is set."""
    p = tmp_path / "data.json"

    @dataclass
    class _InnerNoVal(
        Versionable,
        version=1,
        name="NLit20b_Inner",
        register=True,
        validate_literals=False,
    ):
        mode: Literal["fast", "slow"] = "fast"

    @dataclass
    class _Outer(Versionable, version=1, name="NLit20b_Outer", register=True):
        inner: _InnerNoVal

    p.write_text(
        json.dumps(
            {
                "__versionable__": {"object": "NLit20b_Outer", "version": 1, "hash": ""},
                "inner": {
                    "__versionable__": {"object": "NLit20b_Inner", "version": 1, "hash": ""},
                    "mode": "banana",
                },
            }
        )
    )

    # Outer is default validate_literals=True, but Inner's own setting wins for Inner's fields.
    loaded = versionable.load(_Outer, p)
    assert loaded.inner.mode == "banana"


# ---------------------------------------------------------------------------
# Unknown-field handling at nested
# ---------------------------------------------------------------------------


def test_nested_unknown_error_raises(tmp_path: Path) -> None:
    """Test 21: nested class with unknown='error' raises UnknownFieldError on extra field."""
    p = tmp_path / "data.json"

    @dataclass
    class _Inner(Versionable, version=1, name="NUnk21_Inner", register=True, unknown="error"):
        name: str

    @dataclass
    class _Outer(Versionable, version=1, name="NUnk21_Outer", register=True):
        inner: _Inner

    # Hand-craft a file with an unknown field on the nested Inner.
    p.write_text(
        json.dumps(
            {
                "__versionable__": {"object": "NUnk21_Outer", "version": 1, "hash": ""},
                "inner": {
                    "__versionable__": {"object": "NUnk21_Inner", "version": 1, "hash": ""},
                    "name": "x",
                    "extra": "should not be here",
                },
            }
        )
    )

    with pytest.raises(UnknownFieldError, match="NUnk21_Inner"):
        versionable.load(_Outer, p)


def test_nested_unknown_ignore_drops_silently(tmp_path: Path) -> None:
    """Test 22: nested class with unknown='ignore' silently drops extra fields."""
    p = tmp_path / "data.json"

    @dataclass
    class _Inner(Versionable, version=1, name="NUnk22_Inner", register=True, unknown="ignore"):
        name: str

    @dataclass
    class _Outer(Versionable, version=1, name="NUnk22_Outer", register=True):
        inner: _Inner

    p.write_text(
        json.dumps(
            {
                "__versionable__": {"object": "NUnk22_Outer", "version": 1, "hash": ""},
                "inner": {
                    "__versionable__": {"object": "NUnk22_Inner", "version": 1, "hash": ""},
                    "name": "x",
                    "extra": "drop me",
                },
            }
        )
    )

    loaded = versionable.load(_Outer, p)
    assert loaded.inner.name == "x"
    assert not hasattr(loaded.inner, "extra")
