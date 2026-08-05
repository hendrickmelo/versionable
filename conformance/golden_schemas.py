"""Schema definitions for the checked-in golden corpus.

These classes are the cross-language contract: the C# suite declares the same schemas, with
the same Serialization Names and the same hardcoded schema hashes, and reads the bytes under
`conformance/golden/` that this module's schemas produced.  Both directions matter — Python
writes and C# reads, C# writes and Python reads — so nothing here may change without
regenerating the corpus (`python conformance/generate_golden.py`) and updating the C# mirror.

Hashes are hardcoded string literals on purpose: a rendering drift in either implementation
then fails at class-definition time (Python) or compile time (C#) rather than at load time.

Kept deliberately small: a few elements per array, so the checked-in HDF5 bytes stay tiny
while still exercising the real default compression (gzip level 4 + shuffle).
"""

from __future__ import annotations

import datetime
import re
from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Literal
from uuid import UUID

import numpy as np
import numpy.typing as npt

from versionable import Migration, Versionable

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class GoldenColour(Enum):
    """String-valued enum; hashes by bare name only (GRAMMAR.md section 9)."""

    RED = "red"
    GREEN = "green"
    BLUE = "blue"


class GoldenPriority(Enum):
    """Integer-valued enum; the member *values* are not part of the hash."""

    LOW = 1
    MEDIUM = 2
    HIGH = 3


# ---------------------------------------------------------------------------
# Fixture schemas
# ---------------------------------------------------------------------------


@dataclass
class GoldenScalars(Versionable, version=1, hash="c51ec9", name="GoldenScalars"):
    """Every scalar token that has a C# counterpart."""

    text: str = ""
    count: int = 0
    ratio: float = 0.0
    enabled: bool = False
    phase: complex = 0j
    blob: bytes = b""


@dataclass
class GoldenContainers(Versionable, version=1, hash="846dc8", name="GoldenContainers"):
    """The six container forms, including nesting and a non-string dict key.

    Known gap — heterogeneous fixed tuples (`tuple[int, str, float]`) are deliberately absent.
    GRAMMAR.md section 5 defines them and vectors `tuple-mixed` / `tuple-mixed-reordered` lock
    their hashes, so *rendering* them is in scope, but no Python backend can currently **load**
    one: the deserializer applies the first tuple argument to every element, so JSON/YAML/TOML
    raise `ValueError` and HDF5 raises `TypeError`.  Pre-existing Python bug; follow-up issue at
    phase 5 (see `docs/plans/csharp-port.md`).  `pair` and `samples` below cover the homogeneous
    fixed and variadic forms, which do round-trip.  Recorded machine-readably under `knownGaps`
    in `golden/index.json`.
    """

    names: list[str] = field(default_factory=list)
    readings: list[float] = field(default_factory=list)
    counts: list[int] = field(default_factory=list)
    flags: list[bool] = field(default_factory=list)
    lookup: dict[str, int] = field(default_factory=dict)
    byIndex: dict[int, str] = field(default_factory=dict)
    tags: set[str] = field(default_factory=set)
    ids: frozenset[int] = field(default_factory=frozenset)
    pair: tuple[int, int] = (0, 0)
    samples: tuple[float, ...] = ()
    matrix: list[list[float]] = field(default_factory=list)
    grouped: dict[str, list[float]] = field(default_factory=dict)


@dataclass
class GoldenOptionals(Versionable, version=1, hash="70c93b", name="GoldenOptionals"):
    """Optionals both populated and null, plus a multi-member union.

    TOML has no null literal, so a null field is *omitted* from the TOML file and is refilled
    from the dataclass default on load.  That makes `absent` a round-trip only because its
    default is `None` — see the fixture note in the manifest.
    """

    present: str | None = None
    absent: str | None = None
    maybeCount: int | None = None
    money: Decimal | None = None
    either: int | str = 0


@dataclass
class GoldenEnums(Versionable, version=1, hash="660511", name="GoldenEnums"):
    """Enums standalone and inside containers; they wire as their member *value*."""

    colour: GoldenColour = GoldenColour.RED
    priority: GoldenPriority = GoldenPriority.LOW
    palette: list[GoldenColour] = field(default_factory=list)
    byName: dict[str, GoldenColour] = field(default_factory=dict)


@dataclass
class GoldenLiterals(Versionable, version=1, hash="2085b2", name="GoldenLiterals"):
    """Literal member kinds: string, integer, boolean, and a mixed set."""

    mode: Literal["fast", "slow"] = "fast"
    level: Literal[1, 2, 3] = 1
    flag: Literal[True, False] = True
    tag: Literal["auto", 0, "off", 1] = "auto"


@dataclass
class GoldenTemporal(Versionable, version=1, hash="3856a2", name="GoldenTemporal"):
    """The datetime family, naive and timezone-aware."""

    naive: datetime.datetime = datetime.datetime(2026, 1, 1)
    aware: datetime.datetime = datetime.datetime(2026, 1, 1, tzinfo=datetime.UTC)
    day: datetime.date = datetime.date(2026, 1, 1)
    clock: datetime.time = datetime.time()
    elapsed: datetime.timedelta = datetime.timedelta()


@dataclass
class GoldenStdlib(Versionable, version=1, hash="c3ff88", name="GoldenStdlib"):
    """Registered converter types (GRAMMAR.md section 9)."""

    filePath: Path = field(default_factory=Path)
    posixPath: PurePosixPath = field(default_factory=PurePosixPath)
    windowsPath: PureWindowsPath = field(default_factory=PureWindowsPath)
    amount: Decimal = Decimal(0)
    deviceId: UUID = field(default_factory=lambda: UUID(int=0))
    serialPattern: re.Pattern = field(default_factory=lambda: re.compile(""))


@dataclass
class GoldenArrays(Versionable, version=1, hash="b76a00", name="GoldenArrays"):
    """Five dtypes plus a 2-D array and arrays inside containers.

    Every dtype here is declared, so it is hash-significant (ADR-0002); bare `ndarray` is
    deliberately absent because C# `Tensor<T>` is always typed and could not mirror it.  Also
    recorded under `knownGaps` in `golden/index.json`.
    """

    signal: npt.NDArray[np.float64] = field(default_factory=lambda: np.empty(0, dtype=np.float64))
    weights: npt.NDArray[np.float32] = field(default_factory=lambda: np.empty(0, dtype=np.float32))
    counts: npt.NDArray[np.int32] = field(default_factory=lambda: np.empty(0, dtype=np.int32))
    image: npt.NDArray[np.uint8] = field(default_factory=lambda: np.empty(0, dtype=np.uint8))
    mask: npt.NDArray[np.bool_] = field(default_factory=lambda: np.empty(0, dtype=np.bool_))
    matrix: npt.NDArray[np.float64] = field(default_factory=lambda: np.empty((0, 0), dtype=np.float64))
    traces: list[npt.NDArray[np.float64]] = field(default_factory=list)
    channels: dict[str, npt.NDArray[np.float64]] = field(default_factory=dict)


@dataclass
class GoldenInner(Versionable, version=1, hash="e37514", name="GoldenInner"):
    """Leaf object used by the nested fixture."""

    x: float = 0.0
    y: float = 0.0


@dataclass
class GoldenNested(Versionable, version=1, hash="705e61", name="GoldenNested"):
    """A nested Versionable standalone, in a list, in a dict, and as an Optional."""

    label: str = ""
    inner: GoldenInner = field(default_factory=GoldenInner)
    points: list[GoldenInner] = field(default_factory=list)
    byName: dict[str, GoldenInner] = field(default_factory=dict)
    optionalInner: GoldenInner | None = None


@dataclass
class GoldenShape(Versionable, version=1, hash="357f27", name="GoldenShape"):
    """Polymorphic base: `list[GoldenShape]` may hold any registered subclass."""

    label: str = ""


@dataclass
class GoldenCircle(GoldenShape, version=1, hash="8e5e7c", name="GoldenCircle"):
    radius: float = 0.0


@dataclass
class GoldenSquare(GoldenShape, version=1, hash="42fff3", name="GoldenSquare"):
    side: float = 0.0


@dataclass
class GoldenPolymorphic(Versionable, version=1, hash="cfb9bf", name="GoldenPolymorphic"):
    """A collection declared as the base type but holding concrete subclasses.

    Each element carries its own `__versionable__` envelope naming the concrete class, so a
    reader resolves the subclass from the file rather than from the annotation.
    """

    shapes: list[GoldenShape] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Migration chain: v1 -> v2 -> v3
# ---------------------------------------------------------------------------


@dataclass
class GoldenWorker(Versionable, version=3, hash="aac8a2", name="GoldenWorker"):
    """Current schema. Files exist on disk at v1 and v2 and must migrate forward on load.

    v1 -> v2 renames `title` to `name` and drops `debug`.
    v2 -> v3 adds `timeout_ms`, defaulting old files to 0 (they predate the timeout concept)
    rather than to the dataclass default of 30000.
    """

    name: str = ""
    retries: int = 3
    timeout_ms: int = 30000

    class Migrate:
        v1 = Migration().rename("title", "name").drop("debug")
        v2 = Migration().add("timeout_ms", default=0)


@dataclass
class GoldenWorkerV1(Versionable, version=1, hash="5556c8", name="GoldenWorker", register=False):
    """Writer-only mirror of `GoldenWorker` as it was at v1. Never loaded directly."""

    title: str = ""
    debug: bool = False
    retries: int = 3


@dataclass
class GoldenWorkerV2(Versionable, version=2, hash="beb912", name="GoldenWorker", register=False):
    """Writer-only mirror of `GoldenWorker` as it was at v2. Never loaded directly."""

    name: str = ""
    retries: int = 3
