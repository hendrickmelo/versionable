# ruff: noqa: E402 — module-level imports must come after pytest.importorskip("numpy")
# so the entire file is skipped when numpy is not installed.
"""Tests for the canonical type grammar cleanup (ADR-0001, ADR-0002).

Covers the breaking changes shipped for cross-language hash parity: bare
serialization names (unique across a schema, never parameterized),
``ndarray[<dtype>]`` over a closed dtype token table with runtime validation,
and ``Literal`` value quoting over a closed set of option kinds.
"""

from __future__ import annotations

import datetime
import re
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum, StrEnum
from pathlib import Path, PurePosixPath
from typing import Annotated, Any, Literal

import pytest

np = pytest.importorskip("numpy")
npt = pytest.importorskip("numpy.typing")

import versionable
from versionable import (
    DtypeMismatchError,
    Hdf5FieldInfo,
    UnsupportedTypeError,
    Versionable,
    VersionableError,
    registerConverter,
    setSerializationName,
)
from versionable._hash import _SERIALIZATION_NAMES, canonicalTypeName, computeHash

# --- Module-level sample types (module scope keeps names free of test-local paths) ---


class Status(Enum):
    ACTIVE = "active"
    RETIRED = "retired"


class Mode(Enum):
    FAST = "fast"


# Declared after the body: an assignment inside it would become an enum member.
Mode.VERSIONABLE_NAME = "Speed"


class _Marker(Enum):
    """Member-less base used to check that a serialization name is not inherited."""


_Marker.VERSIONABLE_NAME = "Marker"


class DerivedMarker(_Marker):
    ONLY = 1


class ColourCode(StrEnum):
    """String-valued enum: a Literal of one of its members must not render the value."""

    RED = "red"


class Temperature:
    """Custom type registered with an explicit serialization name."""

    def __init__(self, kelvin: float) -> None:
        self.kelvin = kelvin


class Weird:
    """Custom type whose serialization name is pinned imperatively."""


class MyBox[T]:
    """Generic non-container: its type parameter is dropped by the grammar."""


class Ns1:
    """Namespace holding a type whose bare name collides with Ns2's."""

    class Kind(Enum):
        A = 1


class Ns2:
    class Kind(Enum):
        B = 2


class NsRenamed:
    class Kind(Enum):
        C = 3


NsRenamed.Kind.VERSIONABLE_NAME = "OtherKind"


@dataclass
class CollisionLeaf(Versionable, version=1, hash="47a9d1", name="CollisionLeaf", register=False):
    """Nested class used to prove the name walk reaches through a Versionable field."""

    kind: Ns1.Kind


@dataclass
class GrammarNode(Versionable, version=1, hash="aade9d", name="GrammarNode", register=False):
    """Self-referential schema: the reachable-name walk must not recurse forever."""

    children: list[GrammarNode] = field(default_factory=list)


@contextmanager
def _restoredNames() -> Iterator[None]:
    """Restore the global serialization-name registry after a test mutates it.

    Only the name registry needs restoring: it feeds every canonical name, so a
    leaked entry would change other schemas' hashes. A leaked converter entry is
    keyed by a type this module owns and affects nothing else.
    """
    saved = dict(_SERIALIZATION_NAMES)
    try:
        yield
    finally:
        _SERIALIZATION_NAMES.clear()
        _SERIALIZATION_NAMES.update(saved)


@pytest.fixture
def registered_temperature() -> Iterator[type[Temperature]]:
    """Register a named converter for Temperature, then drop the name again."""
    with _restoredNames():
        registerConverter(
            Temperature,
            serialize=lambda v: v.kelvin,
            deserialize=lambda v, _tp: Temperature(v),
            name="temperature",
        )
        yield Temperature


@pytest.fixture
def named_weird() -> Iterator[type[Weird]]:
    """Pin a serialization name on Weird, then restore the registry."""
    with _restoredNames():
        setSerializationName(Weird, "Odd")
        yield Weird


class TestBareSerializationNames:
    def test_enum_uses_bare_class_name(self) -> None:
        assert canonicalTypeName(Status) == "Status"

    def test_locally_defined_enum_has_no_module_or_locals_path(self) -> None:
        class Inline(Enum):
            X = 1

        assert canonicalTypeName(Inline) == "Inline"

    def test_enum_name_override(self) -> None:
        assert canonicalTypeName(Mode) == "Speed"

    def test_enum_name_override_is_not_inherited(self) -> None:
        assert canonicalTypeName(DerivedMarker) == "DerivedMarker"

    def test_converter_types_use_bare_class_names(self) -> None:
        assert canonicalTypeName(datetime.datetime) == "datetime"
        assert canonicalTypeName(datetime.date) == "date"
        assert canonicalTypeName(datetime.time) == "time"
        assert canonicalTypeName(datetime.timedelta) == "timedelta"
        assert canonicalTypeName(Path) == "Path"
        assert canonicalTypeName(PurePosixPath) == "PurePosixPath"
        assert canonicalTypeName(Decimal) == "Decimal"
        assert canonicalTypeName(uuid.UUID) == "UUID"
        assert canonicalTypeName(re.Pattern) == "Pattern"

    def test_converter_registration_name_override(self, registered_temperature: type[Temperature]) -> None:
        assert canonicalTypeName(registered_temperature) == "temperature"

    def test_converter_name_override_is_undone_by_the_fixture(self) -> None:
        assert canonicalTypeName(Temperature) == "Temperature"

    def test_set_serialization_name_override(self, named_weird: type[Weird]) -> None:
        assert canonicalTypeName(named_weird) == "Odd"

    def test_unregistered_type_uses_bare_class_name(self) -> None:
        class Unregistered:
            pass

        assert canonicalTypeName(Unregistered) == "Unregistered"

    def test_versionable_declared_name_still_wins(self) -> None:
        @dataclass
        class Point(Versionable, version=1, hash="04b128", name="Coord", register=False):
            x: float

        assert canonicalTypeName(Point) == "Coord"

    def test_enum_inside_unions_and_containers(self) -> None:
        assert canonicalTypeName(Status | None) == "Union[None, Status]"
        assert canonicalTypeName(list[Status]) == "list[Status]"
        assert canonicalTypeName(dict[str, Mode]) == "dict[str, Speed]"

    def test_enum_field_hash_is_module_independent(self) -> None:
        assert computeHash({"status": Status}) == "1c5587"

    def test_converter_field_hash(self) -> None:
        assert computeHash({"created": datetime.datetime}) == "00bed0"


class TestSetSerializationNameGuards:
    def test_rejects_versionable_subclass(self) -> None:
        @dataclass
        class Owned(Versionable, version=1, hash="04b128", register=False):
            x: float

        with pytest.raises(VersionableError, match="name="):
            setSerializationName(Owned, "Renamed")

    def test_rejects_builtin(self) -> None:
        with pytest.raises(VersionableError, match="fixed by the type grammar"):
            setSerializationName(int, "integer")
        assert canonicalTypeName(int) == "int"


class TestNonContainerParameters:
    """Type parameters are dropped outside the closed container set."""

    def test_parameterized_converter_type(self) -> None:
        assert canonicalTypeName(re.Pattern[str]) == "Pattern"

    def test_parameterized_user_generic(self) -> None:
        assert canonicalTypeName(MyBox[int]) == "MyBox"
        assert canonicalTypeName(MyBox[str]) == canonicalTypeName(MyBox[int])

    def test_conformance_vector_parameterized_non_container(self) -> None:
        assert computeHash({"rx": re.Pattern[str], "box": MyBox[int]}) == "25ae4b"

    def test_containers_keep_their_parameters(self) -> None:
        assert canonicalTypeName(list[int]) == "list[int]"
        assert canonicalTypeName(dict[str, int]) == "dict[str, int]"
        assert canonicalTypeName(set[int]) == "set[int]"
        assert canonicalTypeName(frozenset[str]) == "frozenset[str]"
        assert canonicalTypeName(tuple[int, str]) == "tuple[int, str]"


class TestVariadicTuple:
    def test_ellipsis_renders_as_three_dots(self) -> None:
        assert canonicalTypeName(tuple[int, ...]) == "tuple[int, ...]"

    def test_conformance_vector_tuple_variadic(self) -> None:
        assert computeHash({"row": tuple[int, ...]}) == "177e64"

    def test_variadic_differs_from_fixed(self) -> None:
        assert canonicalTypeName(tuple[int, ...]) != canonicalTypeName(tuple[int])


class TestSerializationNameUniqueness:
    """Names are flattened into one namespace, so collisions must be rejected."""

    def test_two_enums_with_the_same_name_collide(self) -> None:
        with pytest.raises(VersionableError, match="Serialization name 'Kind'"):

            @dataclass
            class Clash(Versionable, version=1, register=False):
                left: Ns1.Kind
                right: Ns2.Kind

    def test_collision_through_a_nested_versionable(self) -> None:
        with pytest.raises(VersionableError, match="Serialization name 'Kind'"):

            @dataclass
            class Root(Versionable, version=1, register=False):
                leaves: list[CollisionLeaf]
                other: Ns2.Kind

    def test_collision_through_a_literal_enum_member(self) -> None:
        with pytest.raises(VersionableError, match="Serialization name 'Kind'"):

            @dataclass
            class Clash(Versionable, version=1, register=False):
                pinned: Literal[Ns1.Kind.A]
                other: Ns2.Kind

    def test_error_names_both_types_and_the_override(self) -> None:
        with pytest.raises(VersionableError) as excinfo:

            @dataclass
            class Clash(Versionable, version=1, register=False):
                left: Ns1.Kind
                right: Ns2.Kind

        message = str(excinfo.value)
        assert "Ns1.Kind" in message
        assert "Ns2.Kind" in message
        assert "VERSIONABLE_NAME" in message

    def test_same_type_used_twice_is_not_a_collision(self) -> None:
        @dataclass
        class Twice(Versionable, version=1, hash="03f3a2", register=False):
            first: Status
            second: Status

        assert Twice.hash() == "03f3a2"

    def test_override_resolves_the_collision(self) -> None:
        @dataclass
        class Resolved(Versionable, version=1, hash="5b574b", register=False):
            left: Ns1.Kind
            right: NsRenamed.Kind

        assert canonicalTypeName(NsRenamed.Kind) == "OtherKind"
        assert Resolved.hash() == "5b574b"

    def test_self_referential_schema_terminates(self) -> None:
        assert GrammarNode.hash() == "aade9d"


class TestNdarrayGrammar:
    def test_bare_ndarray(self) -> None:
        assert canonicalTypeName(np.ndarray) == "ndarray"

    def test_dtype_is_rendered(self) -> None:
        assert canonicalTypeName(npt.NDArray[np.float64]) == "ndarray[float64]"
        assert canonicalTypeName(npt.NDArray[np.int32]) == "ndarray[int32]"
        assert canonicalTypeName(npt.NDArray[np.complex128]) == "ndarray[complex128]"
        assert canonicalTypeName(npt.NDArray[np.bool_]) == "ndarray[bool]"
        assert canonicalTypeName(npt.NDArray[np.uint8]) == "ndarray[uint8]"

    def test_every_token_in_the_closed_table(self) -> None:
        expected = {
            np.bool_: "bool",
            np.int8: "int8",
            np.int16: "int16",
            np.int32: "int32",
            np.int64: "int64",
            np.uint8: "uint8",
            np.uint16: "uint16",
            np.uint32: "uint32",
            np.uint64: "uint64",
            np.float16: "float16",
            np.float32: "float32",
            np.float64: "float64",
            np.complex64: "complex64",
            np.complex128: "complex128",
        }
        for scalar, token in expected.items():
            assert canonicalTypeName(npt.NDArray[scalar]) == f"ndarray[{token}]"

    def test_dtype_outside_the_table_is_rejected(self) -> None:
        for scalar in (np.str_, np.datetime64, np.timedelta64, np.object_):
            with pytest.raises(UnsupportedTypeError, match="no canonical grammar token"):
                canonicalTypeName(npt.NDArray[scalar])

    def test_rejected_dtype_raises_at_class_definition(self) -> None:
        with pytest.raises(UnsupportedTypeError, match="no canonical grammar token"):

            @dataclass
            class Labels(Versionable, version=1, register=False):
                labels: npt.NDArray[np.str_]

    def test_shape_is_erased(self) -> None:
        explicit = np.ndarray[tuple[int, int], np.dtype[np.float64]]
        assert canonicalTypeName(explicit) == "ndarray[float64]"
        assert canonicalTypeName(explicit) == canonicalTypeName(npt.NDArray[np.float64])

    def test_unparametrized_alias_has_no_dtype(self) -> None:
        assert canonicalTypeName(npt.NDArray) == "ndarray"
        assert canonicalTypeName(npt.NDArray[Any]) == "ndarray"

    def test_abstract_scalar_type_has_no_dtype(self) -> None:
        assert canonicalTypeName(npt.NDArray[np.floating]) == "ndarray"

    def test_optional_array_keeps_dtype(self) -> None:
        assert canonicalTypeName(npt.NDArray[np.float64] | None) == "Union[None, ndarray[float64]]"

    def test_arrays_in_containers(self) -> None:
        assert canonicalTypeName(list[npt.NDArray[np.float32]]) == "list[ndarray[float32]]"
        assert canonicalTypeName(dict[str, npt.NDArray[np.int8]]) == "dict[str, ndarray[int8]]"

    def test_annotated_array_keeps_dtype(self) -> None:
        assert canonicalTypeName(Annotated[npt.NDArray[np.float64], "chunked"]) == "ndarray[float64]"

    def test_dtype_change_changes_the_hash(self) -> None:
        wide = computeHash({"data": npt.NDArray[np.float64]})
        narrow = computeHash({"data": npt.NDArray[np.float32]})
        bare = computeHash({"data": np.ndarray})
        assert wide == "9ffa65"
        assert narrow == "3023aa"
        assert bare == "6062ef"
        assert len({wide, narrow, bare}) == 3


class TestLiteralGrammar:
    def test_strings_are_quoted_without_typing_prefix(self) -> None:
        assert canonicalTypeName(Literal["fast", "slow"]) == "Literal['fast', 'slow']"

    def test_int_and_string_options_are_distinct(self) -> None:
        assert canonicalTypeName(Literal[1]) == "Literal[1]"
        assert canonicalTypeName(Literal["1"]) == "Literal['1']"
        assert computeHash({"v": Literal[1]}) != computeHash({"v": Literal["1"]})

    def test_order_is_significant(self) -> None:
        assert canonicalTypeName(Literal["a", "b"]) != canonicalTypeName(Literal["b", "a"])

    def test_bool_and_none_options(self) -> None:
        assert canonicalTypeName(Literal[True, False, None]) == "Literal[True, False, None]"

    def test_bool_is_not_rendered_as_int(self) -> None:
        assert canonicalTypeName(Literal[True]) != canonicalTypeName(Literal[1])

    def test_negative_ints(self) -> None:
        assert canonicalTypeName(Literal[-1, 0, 1]) == "Literal[-1, 0, 1]"

    def test_quotes_and_backslashes_are_escaped(self) -> None:
        assert canonicalTypeName(Literal["it's"]) == "Literal['it\\'s']"
        assert canonicalTypeName(Literal["a\\b"]) == "Literal['a\\\\b']"

    def test_literal_in_containers(self) -> None:
        assert canonicalTypeName(dict[str, Literal["x", 1]]) == "dict[str, Literal['x', 1]]"
        assert canonicalTypeName(list[Literal["x"]] | None) == "Union[None, list[Literal['x']]]"

    def test_enum_member_option_uses_bare_enum_name(self) -> None:
        assert canonicalTypeName(Literal[Status.ACTIVE]) == "Literal[Status.ACTIVE]"

    def test_enum_member_wins_over_its_value_type(self) -> None:
        """A str-mixin enum member renders as the member, never as its value."""
        assert canonicalTypeName(Literal[ColourCode.RED]) == "Literal[ColourCode.RED]"
        assert canonicalTypeName(Literal[ColourCode.RED]) != canonicalTypeName(Literal["red"])

    def test_unsupported_option_kinds_are_rejected(self) -> None:
        for bad in (Literal[1.5], Literal[b"raw"]):
            with pytest.raises(UnsupportedTypeError, match="Literal option"):
                canonicalTypeName(bad)

    def test_unsupported_option_raises_at_class_definition(self) -> None:
        with pytest.raises(UnsupportedTypeError, match="Literal option"):

            @dataclass
            class Gain(Versionable, version=1, register=False):
                gain: Literal[1.5, 3.0]

    def test_literal_field_hash(self) -> None:
        assert computeHash({"mode": Literal["fast", "slow"]}) == "08e2ae"
        assert computeHash({"a": Literal[1], "b": Literal["1"]}) == "128a4c"


# --- Runtime dtype validation ---


@dataclass
class WideArray(Versionable, version=1, hash="9ffa65", register=False):
    data: npt.NDArray[np.float64]


@dataclass
class NarrowArray(Versionable, version=1, hash="3023aa", register=False):
    data: npt.NDArray[np.float32]


@dataclass
class UntypedArray(Versionable, version=1, hash="6062ef", register=False):
    data: np.ndarray


@dataclass
class OptionalArray(Versionable, version=1, hash="ff3e55", register=False):
    data: npt.NDArray[np.float32] | None = None


@dataclass
class ArrayList(Versionable, version=1, hash="dcb1ab", register=False):
    items: list[npt.NDArray[np.float32]]


@dataclass
class ArrayDict(Versionable, version=1, hash="d594ce", register=False):
    arrays: dict[str, npt.NDArray[np.float32]]


@dataclass
class AnnotatedArray(Versionable, version=1, hash="3023aa", register=False):
    data: Annotated[npt.NDArray[np.float32], "chunked"]


@dataclass
class UnionArray(Versionable, version=1, hash="2f44fb", register=False):
    """Multi-member union: a dtype violation must not be swallowed by member probing."""

    data: npt.NDArray[np.float32] | list[str]


# Nested pair sharing one serialization name, so a file written by the wide
# variant loads back into the narrow one (schema drift across a release).
@dataclass
class WideChild(Versionable, version=1, hash="9ffa65", name="Child", register=False):
    data: npt.NDArray[np.float64]


@dataclass
class NarrowChild(Versionable, version=1, hash="3023aa", name="Child", register=False):
    data: npt.NDArray[np.float32]


@dataclass
class WideHolder(Versionable, version=1, hash="c57bc7", register=False):
    child: WideChild


@dataclass
class NarrowHolder(Versionable, version=1, hash="c57bc7", register=False):
    child: NarrowChild


@dataclass
class StringData(Versionable, version=1, hash="1cad73", register=False):
    """String arrays stay legal under a bare, undeclared ndarray annotation."""

    labels: np.ndarray


@dataclass
class UntypedList(Versionable, version=1, hash="119c30", register=False):
    """Bare element annotation: writes each element with whatever dtype it has."""

    items: list[np.ndarray]


@dataclass
class UntypedDict(Versionable, version=1, hash="27fe67", register=False):
    arrays: dict[str, np.ndarray]


@dataclass
class WideList(Versionable, version=1, hash="13fc7e", name="Listy", register=False):
    items: list[npt.NDArray[np.float64]]


@dataclass
class NarrowList(Versionable, version=1, hash="dcb1ab", name="Listy", register=False):
    items: list[npt.NDArray[np.float32]]


@dataclass
class WideDict(Versionable, version=1, hash="f7e39f", register=False):
    arrays: dict[str, npt.NDArray[np.float64]]


def _writeMixedDtypeList(path: Path) -> None:
    """Write a list whose elements are stored with *different* dtypes."""
    versionable.save(
        UntypedList(items=[np.array([1.5], dtype=np.float32), np.array([2.5], dtype=np.float64)]),
        path,
    )


def _writeMixedDtypeDict(path: Path) -> None:
    versionable.save(
        UntypedDict(arrays={"a": np.array([1.5], dtype=np.float32), "b": np.array([2.5], dtype=np.float64)}),
        path,
    )


class TestDtypeValidationOnSave:
    def test_safe_cast_is_applied_silently(self, tmp_path: Path) -> None:
        obj = NarrowArray(data=np.array([1, 2, 3], dtype=np.float16))
        path = tmp_path / "safe.json"
        versionable.save(obj, path)
        assert versionable.load(NarrowArray, path).data.dtype == np.float32

    def test_unsafe_cast_raises(self, tmp_path: Path) -> None:
        obj = NarrowArray(data=np.array([1.5, 2.5], dtype=np.float64))
        with pytest.raises(DtypeMismatchError) as excinfo:
            versionable.save(obj, tmp_path / "unsafe.json")
        error = excinfo.value
        assert error.declared == "float32"
        assert error.actual == "float64"
        assert error.fieldPath == "data"
        assert error.context == "save"

    def test_error_is_a_versionable_error(self) -> None:
        assert issubclass(DtypeMismatchError, versionable.ConverterError)
        assert issubclass(DtypeMismatchError, versionable.VersionableError)

    def test_bare_ndarray_field_is_not_validated(self, tmp_path: Path) -> None:
        obj = UntypedArray(data=np.array([1, 2], dtype=np.int16))
        path = tmp_path / "bare.json"
        versionable.save(obj, path)
        assert versionable.load(UntypedArray, path).data.dtype == np.int16

    def test_optional_array_accepts_none(self, tmp_path: Path) -> None:
        path = tmp_path / "none.json"
        versionable.save(OptionalArray(), path)
        assert versionable.load(OptionalArray, path).data is None

    def test_optional_array_still_validates_the_dtype(self, tmp_path: Path) -> None:
        obj = OptionalArray(data=np.array([1.0], dtype=np.float64))
        with pytest.raises(DtypeMismatchError):
            versionable.save(obj, tmp_path / "optional.json")

    def test_array_inside_a_list_is_validated(self, tmp_path: Path) -> None:
        obj = ArrayList(items=[np.array([1.0], dtype=np.float64)])
        with pytest.raises(DtypeMismatchError) as excinfo:
            versionable.save(obj, tmp_path / "list.json")
        assert excinfo.value.fieldPath == "items[0]"

    def test_array_inside_a_dict_is_validated(self, tmp_path: Path) -> None:
        obj = ArrayDict(arrays={"a": np.array([1.0], dtype=np.float64)})
        with pytest.raises(DtypeMismatchError) as excinfo:
            versionable.save(obj, tmp_path / "dict.json")
        assert excinfo.value.fieldPath == "arrays['a']"

    def test_annotated_array_is_validated(self, tmp_path: Path) -> None:
        obj = AnnotatedArray(data=np.array([1.0], dtype=np.float64))
        with pytest.raises(DtypeMismatchError):
            versionable.save(obj, tmp_path / "annotated.json")

    def test_annotated_array_safe_cast_round_trips(self, tmp_path: Path) -> None:
        path = tmp_path / "annotated.json"
        versionable.save(AnnotatedArray(data=np.array([1.0], dtype=np.float16)), path)
        assert versionable.load(AnnotatedArray, path).data.dtype == np.float32

    def test_nested_field_path_is_reported(self, tmp_path: Path) -> None:
        obj = NarrowHolder(child=NarrowChild(data=np.array([1.5], dtype=np.float64)))
        with pytest.raises(DtypeMismatchError) as excinfo:
            versionable.save(obj, tmp_path / "nested.json")
        assert excinfo.value.fieldPath == "child.data"

    def test_string_arrays_round_trip_under_a_bare_annotation(self, tmp_path: Path) -> None:
        path = tmp_path / "labels.json"
        versionable.save(StringData(labels=np.array(["alpha", "beta"])), path)
        assert list(versionable.load(StringData, path).labels) == ["alpha", "beta"]


class TestDtypeValidationOnLoad:
    def test_unsafe_stored_dtype_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "drift.json"
        versionable.save(WideArray(data=np.array([1.5, 2.5], dtype=np.float64)), path)
        with pytest.raises(DtypeMismatchError) as excinfo:
            versionable.load(NarrowArray, path)
        assert excinfo.value.context == "load"
        assert excinfo.value.actual == "float64"
        assert excinfo.value.fieldPath == "data"

    def test_safe_stored_dtype_is_cast(self, tmp_path: Path) -> None:
        path = tmp_path / "widen.json"
        versionable.save(NarrowArray(data=np.array([1.5], dtype=np.float32)), path)
        loaded = versionable.load(WideArray, path)
        assert loaded.data.dtype == np.float64

    def test_nested_field_path_is_reported(self, tmp_path: Path) -> None:
        path = tmp_path / "nested.json"
        versionable.save(WideHolder(child=WideChild(data=np.array([1.5], dtype=np.float64))), path)
        with pytest.raises(DtypeMismatchError) as excinfo:
            versionable.load(NarrowHolder, path)
        assert excinfo.value.fieldPath == "child.data"
        assert excinfo.value.context == "load"

    def test_multi_member_union_does_not_swallow_the_error(self, tmp_path: Path) -> None:
        """Union member probing catches ConverterError; a dtype violation must survive it."""
        path = tmp_path / "union.json"
        versionable.save(WideArray(data=np.array([1.5], dtype=np.float64)), path)
        with pytest.raises(DtypeMismatchError):
            versionable.load(UnionArray, path)

    def test_yaml_backend(self, tmp_path: Path) -> None:
        pytest.importorskip("yaml")
        path = tmp_path / "drift.yaml"
        versionable.save(WideArray(data=np.array([1.5], dtype=np.float64)), path)
        with pytest.raises(DtypeMismatchError):
            versionable.load(NarrowArray, path)


class TestDtypeValidationHdf5:
    def test_safe_cast_on_save(self, tmp_path: Path) -> None:
        pytest.importorskip("h5py")
        path = tmp_path / "safe.h5"
        versionable.save(NarrowArray(data=np.array([1, 2], dtype=np.float16)), path)
        assert versionable.load(NarrowArray, path, preload="*").data.dtype == np.float32

    def test_unsafe_cast_on_save_raises(self, tmp_path: Path) -> None:
        pytest.importorskip("h5py")
        obj = NarrowArray(data=np.array([1.5], dtype=np.float64))
        with pytest.raises(DtypeMismatchError):
            versionable.save(obj, tmp_path / "unsafe.h5")

    def test_unsafe_stored_dtype_raises_on_eager_load(self, tmp_path: Path) -> None:
        pytest.importorskip("h5py")
        path = tmp_path / "drift.h5"
        versionable.save(WideArray(data=np.array([1.5], dtype=np.float64)), path)
        with pytest.raises(DtypeMismatchError):
            versionable.load(NarrowArray, path, preload="*")


class TestDtypeValidationHdf5Lazy:
    """The default HDF5 load path returns sentinels; dtype drift must still be caught."""

    def test_unsafe_stored_dtype_raises_at_load_not_on_access(self, tmp_path: Path) -> None:
        pytest.importorskip("h5py")
        path = tmp_path / "drift.h5"
        versionable.save(WideArray(data=np.array([1.5], dtype=np.float64)), path)
        with pytest.raises(DtypeMismatchError) as excinfo:
            versionable.load(NarrowArray, path)
        assert excinfo.value.context == "load"
        assert excinfo.value.fieldPath == "data"

    def test_safe_stored_dtype_is_cast_on_materialization(self, tmp_path: Path) -> None:
        pytest.importorskip("h5py")
        path = tmp_path / "widen.h5"
        versionable.save(NarrowArray(data=np.array([1.5], dtype=np.float32)), path)
        loaded = versionable.load(WideArray, path)
        assert loaded.data.dtype == np.float64

    def test_matching_dtype_is_untouched(self, tmp_path: Path) -> None:
        pytest.importorskip("h5py")
        path = tmp_path / "match.h5"
        versionable.save(NarrowArray(data=np.array([1.5], dtype=np.float32)), path)
        loaded = versionable.load(NarrowArray, path)
        assert loaded.data.dtype == np.float32
        assert loaded.data[0] == np.float32(1.5)

    def test_bare_annotation_keeps_the_stored_dtype(self, tmp_path: Path) -> None:
        pytest.importorskip("h5py")
        path = tmp_path / "bare.h5"
        versionable.save(UntypedArray(data=np.array([1, 2], dtype=np.int16)), path)
        assert versionable.load(UntypedArray, path).data.dtype == np.int16

    def test_mixed_dtype_list_casts_every_element(self, tmp_path: Path) -> None:
        """A safe cast must not be dropped just because a later element matches."""
        pytest.importorskip("h5py")
        path = tmp_path / "mixed_list.h5"
        _writeMixedDtypeList(path)

        lazy = versionable.load(WideList, path)
        eager = versionable.load(WideList, path, preload="*")
        assert [a.dtype for a in lazy.items] == [np.float64, np.float64]
        assert [a.dtype for a in lazy.items] == [a.dtype for a in eager.items]

    def test_mixed_dtype_dict_casts_every_value(self, tmp_path: Path) -> None:
        pytest.importorskip("h5py")
        path = tmp_path / "mixed_dict.h5"
        _writeMixedDtypeDict(path)

        lazy = versionable.load(WideDict, path)
        eager = versionable.load(WideDict, path, preload="*")
        assert [lazy.arrays[k].dtype for k in ("a", "b")] == [np.float64, np.float64]
        assert [lazy.arrays[k].dtype for k in ("a", "b")] == [eager.arrays[k].dtype for k in ("a", "b")]

    def test_lazy_collection_elements_are_validated(self, tmp_path: Path) -> None:
        pytest.importorskip("h5py")
        path = tmp_path / "list.h5"
        versionable.save(WideList(items=[np.array([1.5], dtype=np.float64)]), path)
        with pytest.raises(DtypeMismatchError):
            versionable.load(NarrowList, path)


class TestSessionDtypeValidation:
    def test_append_validates_against_the_declared_dtype(self, tmp_path: Path) -> None:
        pytest.importorskip("h5py")
        import versionable.hdf5

        @dataclass
        class Recording(Versionable, version=1, hash="790832", register=False):
            waveform: Annotated[npt.NDArray[np.float32], Hdf5FieldInfo()] = field(
                default_factory=lambda: np.empty(0, dtype=np.float32)
            )

        path = tmp_path / "session.h5"
        with versionable.hdf5.open(Recording, path) as obj:
            obj.waveform = np.zeros(4, dtype=np.float32)
            obj.waveform.append(np.ones(2, dtype=np.float16))  # safe widening
            with pytest.raises(DtypeMismatchError):
                obj.waveform.append(np.ones(2, dtype=np.float64))

        loaded = versionable.load(Recording, path, preload="*")
        assert loaded.waveform.dtype == np.float32
        assert len(loaded.waveform) == 6

    def test_assignment_rejects_an_unsafe_dtype(self, tmp_path: Path) -> None:
        pytest.importorskip("h5py")
        import versionable.hdf5

        @dataclass
        class Narrow(Versionable, version=1, hash="3023aa", register=False):
            data: npt.NDArray[np.float32] = field(default_factory=lambda: np.empty(0, dtype=np.float32))

        path = tmp_path / "session.h5"
        with versionable.hdf5.open(Narrow, path) as obj, pytest.raises(DtypeMismatchError):
            obj.data = np.ones(3, dtype=np.float64)
