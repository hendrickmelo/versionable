"""Tests for the type converter system."""

from __future__ import annotations

import datetime
import re
import uuid
from dataclasses import dataclass
from decimal import Decimal
from enum import Enum, auto
from pathlib import Path
from typing import Optional

import pytest

from versionable._base import Versionable
from versionable._hash import computeHash
from versionable._types import (
    deserialize,
    registerConverter,
    serialize,
)
from versionable.errors import ConverterError


class TestPrimitives:
    def test_int(self) -> None:
        assert serialize(42, int) == 42
        assert deserialize(42, int) == 42

    def test_float(self) -> None:
        assert serialize(3.14, float) == 3.14
        assert deserialize(3.14, float) == 3.14

    def test_str(self) -> None:
        assert serialize("hello", str) == "hello"
        assert deserialize("hello", str) == "hello"

    def test_bool(self) -> None:
        assert serialize(True, bool) is True
        assert deserialize(True, bool) is True

    def test_none(self) -> None:
        # Intentionally testing old-style Optional[str] handling
        assert serialize(None, Optional[str]) is None  # noqa: UP045
        assert deserialize(None, Optional[str]) is None  # noqa: UP045


class TestDatetime:
    def test_datetime_roundtrip(self) -> None:
        dt = datetime.datetime(2026, 3, 30, 12, 0, 0, tzinfo=datetime.UTC)
        serialized = serialize(dt, datetime.datetime)
        assert isinstance(serialized, str)
        result = deserialize(serialized, datetime.datetime)
        assert result == dt

    def test_date_roundtrip(self) -> None:
        d = datetime.date(2026, 3, 30)
        serialized = serialize(d, datetime.date)
        result = deserialize(serialized, datetime.date)
        assert result == d

    def test_time_roundtrip(self) -> None:
        t = datetime.time(12, 30, 45)
        serialized = serialize(t, datetime.time)
        result = deserialize(serialized, datetime.time)
        assert result == t

    def test_timedelta_roundtrip(self) -> None:
        td = datetime.timedelta(hours=1, minutes=30, seconds=15)
        serialized = serialize(td, datetime.timedelta)
        assert isinstance(serialized, float)
        result = deserialize(serialized, datetime.timedelta)
        assert result == td


class TestPath:
    def test_path_roundtrip(self) -> None:
        p = Path("/some/path/to/file.txt")
        serialized = serialize(p, Path)
        assert isinstance(serialized, str)
        result = deserialize(serialized, Path)
        assert result == p


class TestUUID:
    def test_roundtrip(self) -> None:
        u = uuid.uuid4()
        serialized = serialize(u, uuid.UUID)
        assert isinstance(serialized, str)
        result = deserialize(serialized, uuid.UUID)
        assert result == u


class TestDecimal:
    def test_roundtrip(self) -> None:
        d = Decimal("3.14159265358979")
        serialized = serialize(d, Decimal)
        assert isinstance(serialized, str)
        result = deserialize(serialized, Decimal)
        assert result == d


class TestBytes:
    def test_roundtrip(self) -> None:
        b = b"\x00\x01\x02\xff"
        serialized = serialize(b, bytes)
        assert isinstance(serialized, str)
        result = deserialize(serialized, bytes)
        assert result == b


class TestComplex:
    def test_roundtrip(self) -> None:
        c = complex(1.5, -2.3)
        serialized = serialize(c, complex)
        assert serialized == [1.5, -2.3]
        result = deserialize(serialized, complex)
        assert result == c


class TestRegex:
    def test_roundtrip(self) -> None:
        pattern = re.compile(r"\d+\.\d+")
        serialized = serialize(pattern, re.Pattern)
        assert isinstance(serialized, str)
        result = deserialize(serialized, re.Pattern)
        assert result.pattern == pattern.pattern


class TestEnum:
    def test_roundtrip(self) -> None:
        class Color(Enum):
            RED = "red"
            BLUE = "blue"

        serialized = serialize(Color.RED, Color)
        assert serialized == "red"
        result = deserialize("red", Color)
        assert result is Color.RED

    def test_fallback(self) -> None:
        class Status(Enum):
            ACTIVE = "active"
            UNKNOWN = "unknown"

        Status.VERSIONABLE_FALLBACK = Status.UNKNOWN  # type: ignore[attr-defined]

        result = deserialize("deleted", Status)
        assert result is Status.UNKNOWN

    def test_noFallbackRaises(self) -> None:
        class Strict(Enum):
            A = "a"

        with pytest.raises(ConverterError, match="Unknown"):
            deserialize("nonexistent", Strict)

    def test_autoEnum(self) -> None:
        class Mode(Enum):
            FAST = auto()
            SLOW = auto()

        serialized = serialize(Mode.FAST, Mode)
        result = deserialize(serialized, Mode)
        assert result is Mode.FAST


class TestCollections:
    def test_list(self) -> None:
        data = [1, 2, 3]
        serialized = serialize(data, list[int])
        assert serialized == [1, 2, 3]
        result = deserialize(serialized, list[int])
        assert result == [1, 2, 3]

    def test_dict(self) -> None:
        data = {"a": 1, "b": 2}
        serialized = serialize(data, dict[str, int])
        assert serialized == {"a": 1, "b": 2}
        result = deserialize(serialized, dict[str, int])
        assert result == {"a": 1, "b": 2}

    def test_set(self) -> None:
        data = {3, 1, 2}
        serialized = serialize(data, set[int])
        assert isinstance(serialized, list)
        result = deserialize(serialized, set[int])
        assert result == {1, 2, 3}

    def test_frozenset(self) -> None:
        data = frozenset([1, 2])
        serialized = serialize(data, frozenset[int])
        assert isinstance(serialized, list)
        result = deserialize(serialized, frozenset[int])
        assert result == frozenset([1, 2])

    def test_tuple(self) -> None:
        data = (1, 2, 3)
        serialized = serialize(data, tuple[int, ...])
        assert serialized == [1, 2, 3]
        result = deserialize(serialized, tuple[int, ...])
        assert result == (1, 2, 3)

    def test_optional(self) -> None:
        # Intentionally testing old-style Optional[str] handling
        assert serialize("hello", Optional[str]) == "hello"  # noqa: UP045
        assert deserialize("hello", Optional[str]) == "hello"  # noqa: UP045
        assert serialize(None, Optional[str]) is None  # noqa: UP045
        assert deserialize(None, Optional[str]) is None  # noqa: UP045

    def test_nestedCollections(self) -> None:
        data = {"key": [1, 2, 3]}
        serialized = serialize(data, dict[str, list[int]])
        result = deserialize(serialized, dict[str, list[int]])
        assert result == data

    def test_listOfDatetimes(self) -> None:
        dates = [datetime.date(2026, 1, 1), datetime.date(2026, 6, 15)]
        serialized = serialize(dates, list[datetime.date])
        assert all(isinstance(s, str) for s in serialized)
        result = deserialize(serialized, list[datetime.date])
        assert result == dates


class TestNdarray:
    np = pytest.importorskip("numpy")

    def test_roundtrip(self) -> None:
        np = self.np
        arr = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        serialized = serialize(arr, np.ndarray)
        assert isinstance(serialized, dict)
        assert serialized["__ver_ndarray__"] is True
        result = deserialize(serialized, np.ndarray)
        np.testing.assert_array_equal(result, arr)

    def test_2dArray(self) -> None:
        np = self.np
        arr = np.array([[1, 2], [3, 4]], dtype=np.int32)
        serialized = serialize(arr, np.ndarray)
        result = deserialize(serialized, np.ndarray)
        np.testing.assert_array_equal(result, arr)

    def test_nativeBypass(self) -> None:
        """If ndarray is in nativeTypes, it passes through."""
        np = self.np
        arr = np.array([1.0, 2.0])
        result = serialize(arr, np.ndarray, nativeTypes={np.ndarray})
        assert isinstance(result, np.ndarray)

    def test_fromList(self) -> None:
        np = self.np
        result = deserialize([1, 2, 3], np.ndarray)
        np.testing.assert_array_equal(result, np.array([1, 2, 3]))


class TestNestedVersionable:
    def test_roundtrip(self) -> None:
        fields = {"x": float, "y": float}
        h = computeHash(fields)

        @dataclass
        class Point(Versionable, version=1, hash=h, register=False):
            x: float
            y: float

        pt = Point(x=1.0, y=2.0)
        serialized = serialize(pt, Point)
        assert isinstance(serialized, dict)
        assert serialized["__versionable__"]["object"] == "Point"
        assert serialized["x"] == 1.0

        result = deserialize(serialized, Point)
        assert isinstance(result, Point)
        assert result.x == 1.0
        assert result.y == 2.0


class TestVersionableValueProtocol:
    def test_roundtrip(self) -> None:
        class UserId:
            def __init__(self, value: str) -> None:
                self._value = value

            def toValue(self) -> str:
                return self._value

            @classmethod
            def fromValue(cls, value: str) -> UserId:
                return cls(value)

            def __eq__(self, other: object) -> bool:
                return isinstance(other, UserId) and self._value == other._value

        uid = UserId("user-123")
        serialized = serialize(uid, UserId)
        assert serialized == "user-123"
        result = deserialize(serialized, UserId)
        assert isinstance(result, UserId)
        assert result == uid


class TestCustomConverter:
    def test_registered(self) -> None:
        class Coord:
            def __init__(self, lat: float, lon: float) -> None:
                self.lat = lat
                self.lon = lon

            def __eq__(self, other: object) -> bool:
                return isinstance(other, Coord) and self.lat == other.lat and self.lon == other.lon

        registerConverter(
            Coord,
            serialize=lambda v: {"lat": v.lat, "lon": v.lon},
            deserialize=lambda v, _cls: Coord(v["lat"], v["lon"]),
        )

        coord = Coord(40.7, -74.0)
        serialized = serialize(coord, Coord)
        assert serialized == {"lat": 40.7, "lon": -74.0}
        result = deserialize(serialized, Coord)
        assert result == coord
