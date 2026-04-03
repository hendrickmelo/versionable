"""Tests for schema hash computation."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Optional

import numpy as np
import numpy.typing as npt

from versionable._hash import canonicalTypeName, computeHash


class TestComputeHash:
    def test_deterministic(self) -> None:
        fields: dict[str, type] = {"name": str, "value": int}
        assert computeHash(fields) == computeHash(fields)

    def test_orderIndependent(self) -> None:
        a = computeHash({"x": int, "y": str})
        b = computeHash({"y": str, "x": int})
        assert a == b

    def test_sixCharHex(self) -> None:
        result = computeHash({"a": int})
        assert len(result) == 6
        assert all(c in "0123456789abcdef" for c in result)

    def test_changesOnFieldAdd(self) -> None:
        h1 = computeHash({"name": str})
        h2 = computeHash({"name": str, "age": int})
        assert h1 != h2

    def test_changesOnFieldRemove(self) -> None:
        h1 = computeHash({"name": str, "age": int})
        h2 = computeHash({"name": str})
        assert h1 != h2

    def test_changesOnFieldRename(self) -> None:
        h1 = computeHash({"name": str})
        h2 = computeHash({"title": str})
        assert h1 != h2

    def test_changesOnTypeChange(self) -> None:
        h1 = computeHash({"value": int})
        h2 = computeHash({"value": float})
        assert h1 != h2

    def test_emptyFields(self) -> None:
        result = computeHash({})
        assert len(result) == 6


class TestCanonicalTypeName:
    def test_primitives(self) -> None:
        assert canonicalTypeName(int) == "int"
        assert canonicalTypeName(str) == "str"
        assert canonicalTypeName(float) == "float"
        assert canonicalTypeName(bool) == "bool"
        assert canonicalTypeName(bytes) == "bytes"

    def test_none(self) -> None:
        assert canonicalTypeName(type(None)) == "None"

    def test_genericList(self) -> None:
        assert canonicalTypeName(list[int]) == "list[int]"

    def test_genericDict(self) -> None:
        assert canonicalTypeName(dict[str, int]) == "dict[str, int]"

    def test_optional(self) -> None:
        result = canonicalTypeName(Optional[str])  # noqa: UP045 — testing old-style Optional handling
        # Optional[str] is Union[str, None]
        assert "Union" in result
        assert "str" in result
        assert "None" in result

    def test_union(self) -> None:
        result = canonicalTypeName(int | str)
        assert "Union" in result
        assert "int" in result
        assert "str" in result

    def test_unionSorted(self) -> None:
        """Union members are sorted for determinism."""
        a = canonicalTypeName(int | str)
        b = canonicalTypeName(str | int)
        assert a == b

    def test_nestedGeneric(self) -> None:
        result = canonicalTypeName(list[dict[str, int]])
        assert result == "list[dict[str, int]]"

    def test_numpyNdarray(self) -> None:
        result = canonicalTypeName(np.ndarray)
        assert result == "ndarray"

    def test_numpyTypedArray(self) -> None:
        result = canonicalTypeName(npt.NDArray[np.float64])
        assert "ndarray" in result

    def test_enum(self) -> None:
        class Color(Enum):
            RED = 1

        result = canonicalTypeName(Color)
        assert "Color" in result

    def test_datetime(self) -> None:
        result = canonicalTypeName(datetime)
        assert "datetime" in result

    def test_tuple(self) -> None:
        result = canonicalTypeName(tuple[int, str])
        assert result == "tuple[int, str]"

    def test_set(self) -> None:
        result = canonicalTypeName(set[int])
        assert result == "set[int]"

    def test_frozenset(self) -> None:
        result = canonicalTypeName(frozenset[str])
        assert result == "frozenset[str]"
