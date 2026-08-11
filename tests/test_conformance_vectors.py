"""Conformance: the Python renderer must reproduce every vector in `conformance/hash-vectors.json`.

The vectors are the language-neutral contract described by `conformance/GRAMMAR.md`. A vector
is passed by declaring a Python type whose fields *render* to the vector's canonical strings,
rebuilding the payload from that rendering, and reproducing the hash — checking only
`payload -> hash` would test `sha256`, not the grammar.

Every vector must appear in `_VECTOR_FIXTURES` below; an unmapped vector is a failure, so a
newly-added vector cannot be silently ignored by the Python side.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta
from decimal import Decimal
from enum import Enum
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Annotated, Any, Literal
from uuid import UUID

import pytest

from versionable import Versionable
from versionable._base import _resolveFields
from versionable._hash import canonicalTypeName, computeHash

try:
    import numpy as np
    import numpy.typing as npt
except ImportError:  # pragma: no cover - exercised only in the minimal env
    np = None  # type: ignore[assignment]  # numpy is optional; array vectors are skipped without it

VECTORS_PATH = Path(__file__).resolve().parent.parent / "conformance" / "hash-vectors.json"


# ---------------------------------------------------------------------------
# Fixture types referenced by the vectors' canonical strings
# ---------------------------------------------------------------------------


class Status(Enum):
    ACTIVE = "active"
    IDLE = "idle"


class Colour(str, Enum):  # noqa: UP042 — the classic str-mixin form is what the vector describes; StrEnum would change the class under test
    """A str-mixin enum: its members are strings, so Literal rendering order matters."""

    RED = "red"
    GREEN = "green"


def _make_renamed_enum() -> type[Enum]:
    """An enum whose class name is `Status` but whose serialization name is `DeviceStatus`."""

    class Status(Enum):
        ONLINE = "online"
        OFFLINE = "offline"

    Status.VERSIONABLE_NAME = "DeviceStatus"  # type: ignore[attr-defined]  # set after the body so it is not a member
    return Status


DeviceStatusEnum = _make_renamed_enum()


class MyBox[T]:
    """A generic non-container: its type parameter is dropped (GRAMMAR section 9)."""


@dataclass
class Leaf(Versionable, version=1, register=False):
    value: int = 0


@dataclass
class Node(Versionable, version=1, register=False, hash="f1fda1"):
    children: list[Node] = field(default_factory=list)
    parent: Node | None = None


@dataclass
class DeviceConfig(Versionable, version=1, register=False):
    name: str = ""


@dataclass
class Calibration(Versionable, version=1, register=False):
    offset: float = 0.0


@dataclass
class ChannelConfig(Versionable, version=1, register=False):
    index: int = 0


# ---------------------------------------------------------------------------
# Vector name -> {wire name: Python annotation}
# ---------------------------------------------------------------------------


def _vector_fixtures() -> dict[str, dict[str, Any]]:
    """Return the Python annotation mapping for every vector.

    Built lazily inside a function so the numpy-typed entries can be omitted when numpy is
    absent (the minimal environment) rather than failing at import.
    """
    fixtures: dict[str, dict[str, Any]] = {
        "no-fields": {},
        "single-field": {"value": int},
        "simple-scalars": {"count": int, "label": str},
        "field-name-sorting": {"zeta": int, "alpha": str, "Mid": bool},
        "field-name-sorting-case": {"b": int, "B": int, "a": int, "A": int},
        "worked-example": {"label": str, "count": int, "tags": set[str]},
        "all-scalars": {
            "a": int,
            "b": float,
            "c": str,
            "d": bool,
            "e": bytes,
            "f": complex,
            "g": type(None),
        },
        "bool-is-not-int": {"flag": bool},
        "scalar-width-erased": {"big": int, "small": int, "precise": float, "rough": float},
        "complex-and-bytes": {"z": complex, "blob": bytes},
        "list-of-int": {"items": list[int]},
        "dict-str-float": {"table": dict[str, float]},
        "set-of-str": {"tags": set[str]},
        "frozenset-of-int": {"ids": frozenset[int]},
        "tuple-mixed": {"row": tuple[int, str, float]},
        "tuple-mixed-reordered": {"row": tuple[str, int, float]},
        "tuple-variadic": {"row": tuple[int, ...]},
        "tuple-single": {"pair": tuple[int]},
        "nested-containers": {"matrix": list[list[float]], "index": dict[str, list[int]]},
        "deep-nesting": {"grid": list[dict[str, tuple[int, float]]]},
        "container-of-serialization-name": {
            "nodes": list[Node],
            "flags": set[Status],
            "byStatus": dict[Status, int],
        },
        "optional-int": {"value": int | None},
        "union-sorting": {"value": str | int | bool},
        "union-none-not-first": {"value": Decimal | None},
        "union-none-middle": {"value": Decimal | None | Path},
        "union-nested": {"value": list[int] | str | None},
        "union-in-container": {"lookup": dict[str, float | None]},
        "union-of-serialization-names": {"node": Leaf | Node | None},
        "enum-bare-name": {"status": Status},
        "enum-explicit-name": {"status": DeviceStatusEnum},
        "versionable-name": {"config": DeviceConfig},
        # A self-referential schema: the annotations are forward references in
        # source, and must render as the referenced Serialization Name.
        "versionable-forward-ref": _resolveFields(Node),
        "datetime-family": {
            "created": datetime,
            "day": date,
            "clock": time,
            "elapsed": timedelta,
        },
        "stdlib-converters": {
            "path": Path,
            "amount": Decimal,
            "id": UUID,
            "pattern": re.Pattern,
        },
        "pure-paths": {"posix": PurePosixPath, "windows": PureWindowsPath},
        "literal-strings": {"mode": Literal["fast", "slow"]},
        "literal-strings-reordered": {"mode": Literal["slow", "fast"]},
        "literal-ints": {"level": Literal[1, 2, 3]},
        "literal-str-one": {"value": Literal["1"]},
        "literal-int-one": {"value": Literal[1]},
        "literal-mixed": {"tag": Literal["auto", 0, "off", 1]},
        "literal-negative-int": {"offset": Literal[-1, 0, 1]},
        "literal-single-member": {"kind": Literal["only"]},
        "literal-optional": {"mode": Literal["fast", "slow"] | None},
        "literal-bool": {"flag": Literal[True, False]},
        "literal-bool-vs-int": {"flag": Literal[1, 0]},
        "literal-none-member": {"opt": Literal["a", None]},
        "literal-enum-member": {"mode": Literal[Status.ACTIVE, Status.IDLE]},
        "literal-mixin-enum-member": {"colour": Literal[Colour.RED, Colour.GREEN]},
        "literal-escapes": {"quote": Literal["it's"], "backslash": Literal["a\\b"]},
        "annotated-unwrapped": {
            "gain": Annotated[float, "dB"],
            "note": Annotated[str, "free text"],
        },
        "annotated-nested": {"samples": list[Annotated[float, "volts"]]},
        "sort-by-name-not-pair": {"a": str, "a1": int},
        "parameterized-non-container": {"rx": re.Pattern[str], "box": MyBox[int]},
        "non-ascii-utf8": {"größe": int, "label": Literal["café", "日本語"]},
    }

    if np is None:
        return fixtures

    fixtures.update(
        {
            "ndarray-bare": {"data": np.ndarray},
            "ndarray-float64": {"data": npt.NDArray[np.float64]},
            "ndarray-float32": {"data": npt.NDArray[np.float32]},
            "ndarray-int32": {"counts": npt.NDArray[np.int32]},
            "ndarray-int64": {"counts": npt.NDArray[np.int64]},
            "ndarray-uint8": {"image": npt.NDArray[np.uint8]},
            "ndarray-bool": {"mask": npt.NDArray[np.bool_]},
            "ndarray-complex128": {"spectrum": npt.NDArray[np.complex128]},
            "ndarray-float16": {"weights": npt.NDArray[np.float16]},
            # Declared 3-D: shape is erased, so this matches ndarray-float64 exactly.
            "ndarray-shape-erased": {"data": np.ndarray[tuple[int, int, int], np.dtype[np.float64]]},
            "ndarray-in-container": {
                "frames": list[npt.NDArray[np.uint8]],
                "byName": dict[str, npt.NDArray[np.float64]],
            },
            "ndarray-optional": {"data": npt.NDArray[np.float64] | None},
            "realistic-measurement": {
                "name": str,
                "sampleRate_Hz": float,
                "samples": npt.NDArray[np.float64],
                "acquiredAt": datetime,
                "status": Status,
                "mode": Literal["fast", "slow"],
                "tags": set[str],
                "operator": str | None,
                "calibration": Calibration | None,
            },
            "realistic-device-config": {
                "deviceId": UUID,
                "firmwarePath": Path,
                "price": Decimal,
                "timeout": timedelta,
                "serialPattern": re.Pattern,
                "raw": bytes,
                "channels": list[ChannelConfig],
                "limits": dict[str, tuple[float, float]],
            },
        }
    )
    return fixtures


VECTOR_FIXTURES = _vector_fixtures()


def _load_vectors() -> list[dict[str, Any]]:
    with VECTORS_PATH.open(encoding="utf-8") as handle:
        return list(json.load(handle)["vectors"])


VECTORS = _load_vectors()
VECTORS_BY_NAME = {v["name"]: v for v in VECTORS}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestVectorFile:
    """Structural invariants of hash-vectors.json itself."""

    def test_grammar_version_is_one(self) -> None:
        with VECTORS_PATH.open(encoding="utf-8") as handle:
            assert json.load(handle)["grammarVersion"] == 1

    def test_vector_names_are_unique(self) -> None:
        names = [v["name"] for v in VECTORS]
        assert len(names) == len(set(names))

    def test_every_vector_has_a_python_fixture(self) -> None:
        """A new vector must be mapped here, or the Python side silently stops covering it."""
        if np is None:
            pytest.skip("numpy absent — array vectors are unmappable")
        missing = sorted(v["name"] for v in VECTORS if v["name"] not in VECTOR_FIXTURES)
        assert missing == []

    def test_no_stale_fixtures(self) -> None:
        stale = sorted(name for name in VECTOR_FIXTURES if name not in VECTORS_BY_NAME)
        assert stale == []


@pytest.mark.parametrize("vector", VECTORS, ids=lambda v: str(v["name"]))
class TestVectors:
    """Each vector, rendered from a Python type and re-hashed."""

    def _fields(self, vector: dict[str, Any]) -> dict[str, Any]:
        name = vector["name"]
        if name not in VECTOR_FIXTURES:
            if np is None:
                pytest.skip("numpy absent")
            pytest.fail(f"vector {name!r} has no Python fixture")
        return VECTOR_FIXTURES[name]

    def test_each_field_renders_to_the_canonical_string(self, vector: dict[str, Any]) -> None:
        fields = self._fields(vector)
        rendered = {name: canonicalTypeName(ann) for name, ann in fields.items()}
        assert rendered == vector["fields"]

    def test_payload_is_rebuilt_from_the_rendering(self, vector: dict[str, Any]) -> None:
        fields = self._fields(vector)
        payload = ",".join(f"{name}:{canonicalTypeName(fields[name])}" for name in sorted(fields))
        assert payload == vector["payload"]

    def test_hash_matches(self, vector: dict[str, Any]) -> None:
        assert computeHash(self._fields(vector)) == vector["hash"]

    def test_declared_hash_is_the_sha256_of_the_declared_payload(self, vector: dict[str, Any]) -> None:
        """Guards against an arithmetically wrong vector (hand-written hash typo)."""
        digest = hashlib.sha256(vector["payload"].encode("utf-8")).hexdigest()[:6]
        assert digest == vector["hash"]

    def test_must_match_relations_hold(self, vector: dict[str, Any]) -> None:
        for other in vector.get("mustMatch", []):
            assert other in VECTORS_BY_NAME
            assert VECTORS_BY_NAME[other]["hash"] == vector["hash"]
            assert vector["name"] in VECTORS_BY_NAME[other].get("mustMatch", [])

    def test_must_differ_relations_hold(self, vector: dict[str, Any]) -> None:
        for other in vector.get("mustDiffer", []):
            assert other in VECTORS_BY_NAME
            assert VECTORS_BY_NAME[other]["hash"] != vector["hash"]
            assert vector["name"] in VECTORS_BY_NAME[other].get("mustDiffer", [])


class TestParallelFixtureClasses:
    """Real Versionable classes carrying the vectors' hashes as declared literals.

    These are the Python half of the cross-language mirror: the C# suite declares the same
    schemas with the same hardcoded hash, and both fail at class-definition/compile time if
    either renderer drifts.
    """

    def test_worked_example_class(self) -> None:
        @dataclass
        class WorkedExample(Versionable, version=1, register=False, hash="fe82a7"):
            label: str = ""
            count: int = 0
            tags: set[str] = field(default_factory=set)

        assert WorkedExample.hash() == "fe82a7"

    def test_self_referential_class(self) -> None:
        assert Node.hash() == "f1fda1"


class TestSelfReferenceResolvesAtDefinitionTime:
    """A self-referential schema must hash conformantly *while the class is being created*.

    The class's own name is not bound in its module until `class` finishes executing, so
    `typing.get_type_hints` fails inside `__init_subclass__` and `_resolveFields` retries with
    the class supplied as a local name. Without that retry the annotations reach
    `canonicalTypeName` as raw source strings and render verbatim — `Tree | None` instead of
    `Union[None, Tree]` — producing a hash no other implementation can reproduce. Because the
    declared hash is validated at definition time, the bug is only observable there: calling
    `.hash()` afterwards resolves fine and hides it.

    `Node` above covers this incidentally at module import. This test states the invariant
    outright so it cannot be weakened by accident.
    """

    def test_declaring_the_hash_inline_succeeds(self) -> None:
        """Defining the class *is* the assertion: a wrong hash raises HashMismatchError here."""

        @dataclass
        class Tree(Versionable, version=1, register=False, hash="d5dc54"):
            label: str = ""
            parent: Tree | None = None
            children: list[Tree] = field(default_factory=list)

        assert Tree.hash() == "d5dc54"

    def test_optional_self_reference_renders_as_a_sorted_union(self) -> None:
        """The canonical form, spelled out: not the raw `Tree | None` source text."""

        @dataclass
        class Branch(Versionable, version=1, register=False):
            parent: Branch | None = None

        rendered = canonicalTypeName(_resolveFields(Branch)["parent"])
        assert rendered == "Union[Branch, None]"

    def test_definition_time_and_post_definition_hashes_agree(self) -> None:
        """The two resolution paths must not disagree — that divergence was the original bug."""

        @dataclass
        class Ring(Versionable, version=1, register=False):
            peer: Ring | None = None

        # `.hash()` re-resolves from module globals, where `Ring` is now bound.
        assert Ring.hash() == computeHash(_resolveFields(Ring))
        assert Ring.hash() == computeHash({"peer": Ring | None})

    def test_non_ascii_field_name_hashes_as_utf8(self) -> None:
        fields: dict[str, Any] = {"größe": int, "label": Literal["café", "日本語"]}
        assert computeHash(fields) == "993fd4"

    @pytest.mark.skipif(np is None, reason="numpy not installed")
    def test_realistic_measurement_class(self) -> None:
        @dataclass
        class Measurement(Versionable, version=1, register=False, hash="a9eaff"):
            name: str = ""
            sampleRate_Hz: float = 0.0
            samples: npt.NDArray[np.float64] = field(default_factory=lambda: np.empty(0, dtype=np.float64))
            acquiredAt: datetime = datetime(2026, 1, 1)
            status: Status = Status.ACTIVE
            mode: Literal["fast", "slow"] = "fast"
            tags: set[str] = field(default_factory=set)
            operator: str | None = None
            calibration: Calibration | None = None

        assert Measurement.hash() == "a9eaff"

    def test_realistic_device_config_class(self) -> None:
        @dataclass
        class Device(Versionable, version=1, register=False, hash="6d37e3"):
            deviceId: UUID = field(default_factory=lambda: UUID(int=0))
            firmwarePath: Path = field(default_factory=Path)
            price: Decimal = Decimal(0)
            timeout: timedelta = timedelta()
            serialPattern: re.Pattern = field(default_factory=lambda: re.compile(""))
            raw: bytes = b""
            channels: list[ChannelConfig] = field(default_factory=list)
            limits: dict[str, tuple[float, float]] = field(default_factory=dict)

        assert Device.hash() == "6d37e3"
