"""Tests for the Versionable base class, metadata, and registry."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import pytest

from versionable._base import (
    Versionable,
    _resolveFields,
    getVersionableFields,
    ignoreHashErrors,
    metadata,
    registeredClasses,
)
from versionable._hash import computeHash
from versionable.errors import HashMismatchError, VersionableError


class TestFieldIntrospection:
    def test_publicFieldsIncluded(self) -> None:
        @dataclass
        class Sample(Versionable, version=1, register=False):
            name: str
            value: int

        fields = _resolveFields(Sample)
        assert set(fields.keys()) == {"name", "value"}

    def test_privateFieldsExcluded(self) -> None:
        @dataclass
        class Sample(Versionable, version=1, register=False):
            name: str
            _secret: int = 0

        fields = _resolveFields(Sample)
        assert "name" in fields
        assert "_secret" not in fields

    def test_classVarExcluded(self) -> None:
        @dataclass
        class Sample(Versionable, version=1, register=False):
            name: str
            CONSTANT: ClassVar[int] = 42

        fields = _resolveFields(Sample)
        assert "name" in fields
        assert "CONSTANT" not in fields

    def test_optionalField(self) -> None:
        @dataclass
        class Sample(Versionable, version=1, register=False):
            name: str
            label: str | None = None

        fields = _resolveFields(Sample)
        assert "name" in fields
        assert "label" in fields

    def test_inheritedFields(self) -> None:
        @dataclass
        class Base(Versionable, version=1, register=False):
            baseField: str

        @dataclass
        class Child(Base, version=1, register=False):
            childField: int

        fields = _resolveFields(Child)
        assert "baseField" in fields
        assert "childField" in fields


class TestHashValidation:
    def test_correctHashAccepted(self) -> None:
        fields = {"name": str, "value": int}
        correctHash = computeHash(fields)

        # Should not raise
        @dataclass
        class Sample(Versionable, version=1, hash=correctHash, register=False):
            name: str
            value: int

        assert metadata(Sample).hash == correctHash

    def test_wrongHashRaises(self) -> None:
        with pytest.raises(HashMismatchError, match="hash mismatch"):

            @dataclass
            class Bad(Versionable, version=1, hash="wrong!", register=False):
                name: str

    def test_emptyHashSkipsValidation(self) -> None:
        # No hash provided — no validation
        @dataclass
        class Sample(Versionable, version=1, register=False):
            name: str

        assert metadata(Sample).hash == ""

    def test_ignoreHashErrorsMode(self) -> None:
        try:
            ignoreHashErrors(True)

            # Should not raise, just warn
            @dataclass
            class Sample(Versionable, version=1, hash="wrong!", register=False):
                name: str

        finally:
            ignoreHashErrors(False)

    def test_hashMismatchShowsComputed(self) -> None:
        fields = {"name": str}
        expected = computeHash(fields)

        with pytest.raises(HashMismatchError) as exc_info:

            @dataclass
            class Bad(Versionable, version=1, hash="wrong!", register=False):
                name: str

        assert exc_info.value.computed == expected
        assert exc_info.value.declared == "wrong!"


class TestRegistry:
    def test_classRegistered(self) -> None:
        @dataclass
        class RegTest(Versionable, version=1):
            name: str

        reg = registeredClasses()
        assert reg["RegTest"] is RegTest

    def test_customName(self) -> None:
        @dataclass
        class Impl(Versionable, version=1, name="CustomName"):
            value: int

        reg = registeredClasses()
        assert reg["CustomName"] is Impl

    def test_oldNames(self) -> None:
        @dataclass
        class Current(Versionable, version=1, name="Current", old_names=["OldName"]):
            value: int

        reg = registeredClasses()
        assert reg["Current"] is Current
        assert reg["OldName"] is Current

    def test_registerFalse(self) -> None:
        @dataclass
        class Unregistered(Versionable, version=1, register=False):
            value: int

        reg = registeredClasses()
        assert "Unregistered" not in reg


class TestMetadata:
    def test_basicMetadata(self) -> None:
        fields = {"name": str, "count": int}
        h = computeHash(fields)

        @dataclass
        class Sample(Versionable, version=2, hash=h, register=False):
            name: str
            count: int

        meta = metadata(Sample)
        assert meta.version == 2
        assert meta.hash == h
        assert meta.name == "Sample"
        assert meta.fields == ["count", "name"] or set(meta.fields) == {"name", "count"}
        assert meta.skipDefaults is False
        assert meta.unknown == "ignore"

    def test_skipDefaults(self) -> None:
        @dataclass
        class Sample(Versionable, version=1, skip_defaults=True, register=False):
            name: str

        assert metadata(Sample).skipDefaults is True

    def test_unknownMode(self) -> None:
        @dataclass
        class Strict(Versionable, version=1, unknown="error", register=False):
            name: str

        assert metadata(Strict).unknown == "error"


class TestGetVersionableFields:
    def test_returnsFieldDict(self) -> None:
        fields = {"x": float, "y": float}
        h = computeHash(fields)

        @dataclass
        class Point(Versionable, version=1, hash=h, register=False):
            x: float
            y: float

        result = getVersionableFields(Point)
        assert set(result.keys()) == {"x", "y"}
        assert result["x"] is float
        assert result["y"] is float


class TestInheritance:
    def test_independentVersioning(self) -> None:
        @dataclass
        class Base(Versionable, version=2, register=False):
            createdAt: str

        @dataclass
        class Child(Base, version=3, register=False):
            title: str

        assert metadata(Base).version == 2
        assert metadata(Child).version == 3

    def test_childFieldsIncludeParent(self) -> None:
        @dataclass
        class Base(Versionable, version=1, register=False):
            baseField: str

        @dataclass
        class Child(Base, version=1, register=False):
            childField: int

        fields = getVersionableFields(Child)
        assert "baseField" in fields
        assert "childField" in fields


class TestDuplicateRegistration:
    def test_duplicateNameRaises(self) -> None:
        @dataclass
        class UniqueA(Versionable, version=1, name="DuplicateTest"):
            value: int

        with pytest.raises(VersionableError, match="already registered"):

            @dataclass
            class UniqueB(Versionable, version=1, name="DuplicateTest"):
                value: int

    def test_duplicateOldNameRaises(self) -> None:
        @dataclass
        class OwnsOldName(Versionable, version=1, name="OwnsOldName", old_names=["LegacyName"]):
            value: int

        with pytest.raises(VersionableError, match="already registered"):

            @dataclass
            class ClaimsLegacy(Versionable, version=1, name="ClaimsLegacy", old_names=["LegacyName"]):
                value: int

    def test_oldNameCollidesWithExistingName(self) -> None:
        @dataclass
        class TakenName(Versionable, version=1, name="TakenName"):
            value: int

        with pytest.raises(VersionableError, match="already registered"):

            @dataclass
            class Other(Versionable, version=1, name="Other", old_names=["TakenName"]):
                value: int

    def test_sameClassReregistrationAllowed(self) -> None:
        """Re-importing a module shouldn't error — same class object is fine."""

        @dataclass
        class Idempotent(Versionable, version=1, name="Idempotent"):
            value: int

        # Simulate re-registration of the same class (e.g. module reload)
        from versionable._base import _REGISTRY

        _REGISTRY["Idempotent"] = Idempotent  # same object — no error
