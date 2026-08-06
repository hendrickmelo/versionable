"""Regenerate the checked-in golden corpus under `conformance/golden/`.

Run from the repository root::

    pixi run -- python conformance/generate_golden.py

Each fixture in `golden_schemas.py` is written once per backend (JSON, YAML, TOML, HDF5)
together with a `manifest.json` recording the schema hash, the schema version, and every
field value in a language-neutral comparable form.  The **checked-in bytes are the contract**:
the C# suite reads them and asserts against the same manifests, and `tests/test_golden_corpus.py`
does the Python half.  Regenerating is therefore a deliberate act — if a diff appears in
`conformance/golden/` that you did not intend, a serializer changed behaviour.

HDF5 files use the library default compression (gzip level 4 + shuffle) so the corpus
exercises the real default; arrays are a handful of elements so the bytes stay small.

Comparable form
---------------

JSON cannot express most of the type system, so values are tagged.  Anything not listed
round-trips as its plain JSON counterpart (null, bool, int, float, str, list).

===================  ==========================================================
Tag                  Payload
===================  ==========================================================
``$bytes``           base64 string
``$complex``         ``[real, imag]``
``$decimal``         decimal string
``$datetime``        ISO 8601 (offset included when timezone-aware)
``$date``            ISO 8601 date
``$time``            ISO 8601 time
``$timedeltaSeconds``  total seconds as a number
``$path``            ``{"kind": <class name>, "value": <str(path)>}``
``$uuid``            lowercase hyphenated string
``$pattern``         pattern source string
``$enum``            ``{"type": <serialization name>, "member": ..., "value": ...}``
``$set``             array, ordered by the encoded element's JSON text
``$tuple``           array, in order
``$ndarray``         ``{"dtype": ..., "shape": [...], "data": <nested arrays>}``
``$dict``            ``[[key, value], ...]``, ordered by the encoded key's JSON text
``$object``          ``{"name": ..., "version": ..., "hash": ..., "fields": {...}}``
===================  ==========================================================

`$set` and `$dict` are ordered deterministically so the manifest is byte-stable across runs;
that ordering is a manifest detail and says nothing about the on-disk file format.
"""

from __future__ import annotations

import base64
import datetime
import json
import re
import shutil
import sys
from dataclasses import dataclass
from decimal import Decimal
from enum import Enum
from pathlib import Path, PurePath, PurePosixPath, PureWindowsPath
from typing import Any
from uuid import UUID

import numpy as np

CONFORMANCE_DIR = Path(__file__).resolve().parent
if str(CONFORMANCE_DIR) not in sys.path:
    sys.path.insert(0, str(CONFORMANCE_DIR))

from golden_schemas import (  # noqa: E402
    GoldenArrays,
    GoldenCircle,
    GoldenColour,
    GoldenContainers,
    GoldenEnums,
    GoldenInner,
    GoldenLiterals,
    GoldenNested,
    GoldenOptionals,
    GoldenPolymorphic,
    GoldenPriority,
    GoldenScalars,
    GoldenSquare,
    GoldenStdlib,
    GoldenTemporal,
    GoldenWorker,
    GoldenWorkerV1,
    GoldenWorkerV2,
)

import versionable  # noqa: E402 — after the sys.path insert above
from versionable import Versionable  # noqa: E402
from versionable._base import metadata as getMetadata  # noqa: E402

GOLDEN_ROOT = CONFORMANCE_DIR / "golden"

# Backend key -> file extension.  The key is what the manifest and both test suites use.
BACKENDS: dict[str, str] = {"json": ".json", "yaml": ".yaml", "toml": ".toml", "hdf5": ".h5"}

# Constructs the grammar defines but the corpus deliberately does not cover, and why.  These
# ship in index.json because the corpus — not any task report — is the contract the C#
# implementation reads.  Follow-up issues are filed in phase 5
# (see docs/plans/csharp-port.md).
KNOWN_GAPS: list[dict[str, Any]] = [
    {
        "construct": "heterogeneous fixed tuple",
        "example": "tuple[int, str, float]",
        "status": "specified, hashable, NOT loadable",
        "detail": (
            "GRAMMAR.md section 5 defines fixed tuples as order-significant with per-position types, and "
            "vectors 'tuple-mixed' and 'tuple-mixed-reordered' lock their canonical strings and hashes. "
            "Rendering and hashing are therefore in scope and must be implemented. Round-tripping is not: "
            "no Python backend can currently LOAD a heterogeneous fixed tuple. The deserializer applies the "
            "first tuple argument to every element, so JSON/YAML/TOML raise ValueError ('invalid literal for "
            "int() with base 10') and HDF5 raises TypeError ('No conversion path for dtype'). This is a "
            "pre-existing Python bug, not a grammar question. Homogeneous fixed tuples (tuple[int, int]) and "
            "variadic tuples (tuple[float, ...]) do round-trip and are covered by the 'containers' fixture."
        ),
        "followUp": "issue to be filed at phase 5",
    },
    {
        "construct": "bare (un-parameterized) ndarray",
        "example": "ndarray",
        "status": "specified, Python-only, omitted from the corpus",
        "detail": (
            "GRAMMAR.md section 7 defines bare 'ndarray' as the dynamic-dtype escape hatch. C# Tensor<T> is "
            "always typed, so a bare ndarray field cannot be mirrored in C# and no golden file could be read "
            "by both implementations. Vector 'ndarray-bare' is marked pythonOnly for the same reason. The "
            "'arrays' fixture covers only declared, hash-significant dtypes."
        ),
        "followUp": "none — by design",
    },
    {
        "construct": "fixtures whose C# schema hash differs from the manifest's",
        "example": "containers (846dc8 -> 51917e), stdlib (c3ff88 -> b38e82)",
        "status": "specified, loadable in both languages, NOT hash-identical",
        "detail": (
            "Two fixtures declare Python constructs C# has no spelling for, so the C# mirror renders a "
            "different canonical payload and therefore a different schema hash. 'containers' declares "
            "samples: tuple[float, ...]; GRAMMAR.md section 5 lists the variadic tuple as having no C# "
            "equivalent, so C# declares double[] and renders list[float], hashing 51917e instead of 846dc8. "
            "'stdlib' declares posixPath: PurePosixPath and windowsPath: PureWindowsPath; GRAMMAR.md "
            "section 9 lists both as '(none)' in C#, so all three path fields are FilePath and render Path, "
            "hashing b38e82 instead of c3ff88. Neither divergence affects the wire form: both fixtures load "
            "in C# and every value compares equal to this manifest, which is why they stay in the corpus. "
            "FORWARD CONSTRAINT: the envelope hash therefore CANNOT become a load-time cross-language gate. "
            "A reader that rejected a file whose stored hash disagreed with the loading type's would refuse "
            "these two fixtures in C# while accepting them in Python. Any future load-time hash validation "
            "must exclude or special-case them — or the two constructs must first gain C# spellings, at "
            "which point the C# suite's "
            "'a_fixture_csharp_cannot_spell_mirrors_everything_but_its_hash' theory fails and says so. The "
            "hash remains what it has always been: a build-time/definition-time schema-drift check, "
            "validated by the analyzer in C# and at class definition in Python."
        ),
        "followUp": "revisit if load-time hash validation is ever proposed",
    },
]


# ---------------------------------------------------------------------------
# Comparable-form encoding
# ---------------------------------------------------------------------------


def toComparable(value: Any) -> Any:
    """Encode *value* as language-neutral JSON (see the module docstring)."""
    if value is None or isinstance(value, bool | int | float | str):
        return value
    if isinstance(value, bytes):
        return {"$bytes": base64.b64encode(value).decode("ascii")}
    if isinstance(value, complex):
        return {"$complex": [value.real, value.imag]}
    if isinstance(value, Decimal):
        return {"$decimal": str(value)}
    if isinstance(value, datetime.datetime):
        return {"$datetime": value.isoformat()}
    if isinstance(value, datetime.date):
        return {"$date": value.isoformat()}
    if isinstance(value, datetime.time):
        return {"$time": value.isoformat()}
    if isinstance(value, datetime.timedelta):
        return {"$timedeltaSeconds": value.total_seconds()}
    if isinstance(value, PurePath):
        return {"$path": {"kind": type(value).__name__, "value": str(value)}}
    if isinstance(value, UUID):
        return {"$uuid": str(value)}
    if isinstance(value, re.Pattern):
        return {"$pattern": value.pattern}
    if isinstance(value, Enum):
        return {
            "$enum": {
                "type": type(value).__name__,
                "member": value.name,
                "value": toComparable(value.value),
            }
        }
    if isinstance(value, np.ndarray):
        return {"$ndarray": {"dtype": value.dtype.name, "shape": list(value.shape), "data": value.tolist()}}
    if isinstance(value, np.generic):
        return toComparable(value.item())
    if isinstance(value, Versionable):
        meta = getMetadata(type(value))
        return {
            "$object": {
                "name": meta.name,
                "version": meta.version,
                "hash": meta.hash,
                "fields": {n: toComparable(getattr(value, n)) for n in meta.fields},
            }
        }
    if isinstance(value, set | frozenset):
        return {"$set": _sortedEncoded(value)}
    if isinstance(value, tuple):
        return {"$tuple": [toComparable(v) for v in value]}
    if isinstance(value, list):
        return [toComparable(v) for v in value]
    if isinstance(value, dict):
        pairs = [(toComparable(k), toComparable(v)) for k, v in value.items()]
        pairs.sort(key=lambda kv: _stableText(kv[0]))
        return {"$dict": [[k, v] for k, v in pairs]}
    raise TypeError(f"no comparable encoding for {type(value).__name__}")


def _sortedEncoded(values: set[Any] | frozenset[Any]) -> list[Any]:
    """Encode an unordered collection into a deterministically ordered list."""
    return sorted((toComparable(v) for v in values), key=_stableText)


def _stableText(encoded: Any) -> str:
    return json.dumps(encoded, sort_keys=True, ensure_ascii=False)


def objectValues(obj: Versionable) -> dict[str, Any]:
    """Encode every declared field of *obj*."""
    return {name: toComparable(getattr(obj, name)) for name in getMetadata(type(obj)).fields}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MigrationSource:
    """An older-version file that a reader must migrate forward on load."""

    fromVersion: int
    writer: Versionable
    expected: Versionable


@dataclass(frozen=True)
class Fixture:
    """One corpus entry: an object, its note, and any older-version files beside it."""

    name: str
    obj: Versionable
    note: str
    migrations: tuple[MigrationSource, ...] = ()


def buildFixtures() -> list[Fixture]:
    """Return every fixture instance written to the corpus."""
    return [
        Fixture(
            name="scalars",
            obj=GoldenScalars(
                text="probe-A",
                count=-42,
                ratio=9.8125,
                enabled=True,
                phase=complex(1.5, -2.25),
                blob=b"\x00\x01\xfe\xff",
            ),
            note="Every scalar token with a C# counterpart. 'ratio' and 'phase' use exactly "
            "representable binary fractions so no backend's float formatting can round-trip lossily.",
        ),
        Fixture(
            name="containers",
            obj=GoldenContainers(
                names=["alpha", "beta"],
                readings=[1.5, -2.25, 0.0],
                counts=[1, 2, 3],
                flags=[True, False, True],
                lookup={"a": 1, "b": 2},
                byIndex={1: "one", 2: "two"},
                tags={"red", "green"},
                ids=frozenset({7, 11}),
                pair=(3, 4),
                samples=(0.5, 1.5, 2.5),
                matrix=[[1.0, 2.0], [3.0, 4.0]],
                grouped={"ch0": [0.5, 1.5], "ch1": [2.5]},
            ),
            note="All six container forms plus nesting and a non-string dict key. Sets are "
            "unordered on the wire; the manifest orders them for comparison only.",
        ),
        Fixture(
            name="optionals",
            obj=GoldenOptionals(
                present="filled",
                absent=None,
                maybeCount=5,
                money=Decimal("12.50"),
                either="text",
            ),
            note="TOML has no null literal, so 'absent' is omitted from the TOML file entirely and "
            "is refilled from the dataclass default on load. 'either' exercises a union of two "
            "non-null members.",
        ),
        Fixture(
            name="enums",
            obj=GoldenEnums(
                colour=GoldenColour.GREEN,
                priority=GoldenPriority.HIGH,
                palette=[GoldenColour.RED, GoldenColour.BLUE],
                byName={"primary": GoldenColour.RED, "accent": GoldenColour.BLUE},
            ),
            note="Enums hash by bare Serialization Name but wire as their member value, so a "
            "string-valued and an integer-valued enum look different on disk.",
        ),
        Fixture(
            name="literals",
            obj=GoldenLiterals(mode="slow", level=3, flag=False, tag=1),
            note="Literal member kinds: string, integer, boolean, and a mixed set where the "
            "chosen value is the integer 1 rather than the string 'off'.",
        ),
        Fixture(
            name="temporal",
            obj=GoldenTemporal(
                naive=datetime.datetime(2026, 8, 5, 14, 30, 15, 123456),
                aware=datetime.datetime(
                    2026, 8, 5, 14, 30, 15, 123456, tzinfo=datetime.timezone(datetime.timedelta(hours=-5))
                ),
                day=datetime.date(2026, 8, 5),
                clock=datetime.time(23, 59, 58, 500000),
                elapsed=datetime.timedelta(days=1, seconds=3661, microseconds=500000),
            ),
            note="Naive and offset-aware datetimes, microsecond precision, and a timedelta wired as total seconds.",
        ),
        Fixture(
            name="stdlib",
            obj=GoldenStdlib(
                filePath=Path("data/run-01.h5"),
                posixPath=PurePosixPath("/var/log/device.log"),
                windowsPath=PureWindowsPath(r"C:\Devices\probe.cfg"),
                amount=Decimal("1234.5678"),
                deviceId=UUID("6ba7b810-9dad-11d1-80b4-00c04fd430c8"),
                serialPattern=re.compile(r"^SN-\d{6}$"),
            ),
            note="Registered converter types. Decimal wires as a string so no float precision is "
            "lost; the regex wires as its pattern source with flags dropped.",
        ),
        Fixture(
            name="arrays",
            obj=GoldenArrays(
                signal=np.array([0.5, -1.25, 2.0, 3.75], dtype=np.float64),
                weights=np.array([0.5, 0.25, 0.125], dtype=np.float32),
                counts=np.array([-2, 0, 7], dtype=np.int32),
                image=np.array([0, 128, 255], dtype=np.uint8),
                mask=np.array([True, False, True], dtype=np.bool_),
                matrix=np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float64),
                traces=[
                    np.array([1.0, 2.0], dtype=np.float64),
                    np.array([3.0, 4.0, 5.0], dtype=np.float64),
                ],
                channels={
                    "ch0": np.array([0.25, 0.5], dtype=np.float64),
                    "ch1": np.array([0.75], dtype=np.float64),
                },
            ),
            note="Five hash-significant dtypes, a 2-D array (shape is erased from the hash but "
            "preserved on disk), and arrays nested in a list and a dict. HDF5 uses the default "
            "gzip level 4 + shuffle.",
        ),
        Fixture(
            name="nested",
            obj=GoldenNested(
                label="assembly",
                inner=GoldenInner(x=1.5, y=-2.5),
                points=[GoldenInner(x=0.0, y=0.0), GoldenInner(x=1.0, y=1.0)],
                byName={"origin": GoldenInner(x=0.0, y=0.0), "corner": GoldenInner(x=2.0, y=3.0)},
                optionalInner=GoldenInner(x=9.0, y=9.0),
            ),
            note="A nested Versionable standalone, in a list, in a dict, and as an Optional. Each "
            "nested object carries its own __versionable__ envelope.",
        ),
        Fixture(
            name="polymorphic",
            obj=GoldenPolymorphic(
                shapes=[
                    GoldenCircle(label="c1", radius=2.5),
                    GoldenSquare(label="s1", side=4.0),
                    GoldenCircle(label="c2", radius=0.5),
                ]
            ),
            note="A list declared as the base class holding two different subclasses. The reader "
            "must resolve each element's concrete class from its envelope, not from the annotation.",
        ),
        Fixture(
            name="migration-chain",
            obj=GoldenWorker(name="batch-processor", retries=5, timeout_ms=15000),
            note="Current-version file plus v1 and v2 files that a reader must migrate forward. "
            "v1 -> v2 renames 'title' to 'name' and drops 'debug'; v2 -> v3 adds 'timeout_ms' "
            "defaulting old files to 0, not to the dataclass default of 30000.",
            migrations=(
                MigrationSource(
                    fromVersion=1,
                    writer=GoldenWorkerV1(title="legacy-worker", debug=True, retries=9),
                    expected=GoldenWorker(name="legacy-worker", retries=9, timeout_ms=0),
                ),
                MigrationSource(
                    fromVersion=2,
                    writer=GoldenWorkerV2(name="interim-worker", retries=2),
                    expected=GoldenWorker(name="interim-worker", retries=2, timeout_ms=0),
                ),
            ),
        ),
    ]


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------


def writeAllBackends(obj: Versionable, directory: Path, stem: str) -> dict[str, str]:
    """Save *obj* once per backend; return ``{backend key: file name}``."""
    written: dict[str, str] = {}
    for key, extension in BACKENDS.items():
        fileName = f"{stem}{extension}"
        versionable.save(obj, directory / fileName)
        written[key] = fileName
    return written


def buildManifest(fixture: Fixture) -> dict[str, Any]:
    """Build the manifest for *fixture*, writing its files as a side effect."""
    directory = GOLDEN_ROOT / fixture.name
    directory.mkdir(parents=True, exist_ok=True)

    meta = getMetadata(type(fixture.obj))
    manifest: dict[str, Any] = {
        "fixture": fixture.name,
        "serializationName": meta.name,
        "schemaHash": meta.hash,
        "version": meta.version,
        "note": fixture.note,
        "files": writeAllBackends(fixture.obj, directory, fixture.name),
        "values": objectValues(fixture.obj),
    }

    if fixture.migrations:
        sources = []
        for source in fixture.migrations:
            sourceMeta = getMetadata(type(source.writer))
            sources.append(
                {
                    "fromVersion": source.fromVersion,
                    "schemaHash": sourceMeta.hash,
                    "files": writeAllBackends(source.writer, directory, f"{fixture.name}-v{source.fromVersion}"),
                    "expected": objectValues(source.expected),
                }
            )
        manifest["migrationSources"] = sources

    return manifest


def main() -> None:
    """Regenerate the whole corpus from scratch."""
    if GOLDEN_ROOT.exists():
        shutil.rmtree(GOLDEN_ROOT)
    GOLDEN_ROOT.mkdir(parents=True)

    fixtures = buildFixtures()
    index: list[dict[str, Any]] = []
    for fixture in fixtures:
        manifest = buildManifest(fixture)
        manifestPath = GOLDEN_ROOT / fixture.name / "manifest.json"
        manifestPath.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=False) + "\n",
            encoding="utf-8",
        )
        index.append(
            {
                "fixture": fixture.name,
                "serializationName": manifest["serializationName"],
                "schemaHash": manifest["schemaHash"],
                "version": manifest["version"],
                "migrationSources": [s["fromVersion"] for s in manifest.get("migrationSources", [])],
            }
        )
        print(f"wrote {fixture.name}: {len(manifest['files'])} backends")

    indexPath = GOLDEN_ROOT / "index.json"
    indexPath.write_text(
        json.dumps(
            {"backends": BACKENDS, "fixtures": index, "knownGaps": KNOWN_GAPS},
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        f"wrote {indexPath.relative_to(CONFORMANCE_DIR.parent)} ({len(index)} fixtures, {len(KNOWN_GAPS)} known gaps)"
    )


if __name__ == "__main__":
    main()
