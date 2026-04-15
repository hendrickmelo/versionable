"""Shared fixtures for versionable tests."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Literal

import numpy as np  # noqa: TC002 — needed at runtime for NDArray type resolution
import numpy.typing as npt  # noqa: TC002

from versionable._base import Versionable
from versionable._types import literalFallback

# --- Sample classes used across tests ---


@dataclass
class SimpleConfig(Versionable, version=1, hash="ed3a90", register=False):
    name: str
    debug: bool = False
    retries: int = 3


@dataclass
class WithOptional(
    Versionable,
    version=1,
    hash="36e64d",
    register=False,
):
    label: str
    description: str | None = None


@dataclass
class WithArray(
    Versionable,
    version=1,
    hash="c0dc53",
    register=False,
):
    name: str
    data: npt.NDArray[np.float64]


class Priority(Enum):
    LOW = "low"
    HIGH = "high"


@dataclass
class WithEnum(
    Versionable,
    version=1,
    hash="139030",
    register=False,
):
    title: str
    priority: Priority


@dataclass
class Inner(
    Versionable,
    version=1,
    hash="e37514",
    name="Inner",
):
    x: float
    y: float


@dataclass
class WithNested(
    Versionable,
    version=1,
    hash="db925f",
    register=False,
):
    name: str
    point: Inner


@dataclass
class WithDatetime(
    Versionable,
    version=1,
    hash="221aca",
    register=False,
):
    label: str
    createdAt: datetime


@dataclass
class WithList(
    Versionable,
    version=1,
    hash="2086ae",
    register=False,
):
    tags: list[str]
    scores: list[float] = field(default_factory=list)


@dataclass
class WithSkipDefaults(
    Versionable,
    version=1,
    hash="2bb7c4",
    skip_defaults=True,
    register=False,
):
    name: str
    debug: bool = False
    count: int = 0


@dataclass
class WithLiteral(
    Versionable,
    version=1,
    hash="3c2d1b",
    register=False,
):
    name: str
    mode: Literal["fast", "slow"] = "fast"


@dataclass
class WithLiteralFallback(
    Versionable,
    version=1,
    hash="3c2d1b",
    register=False,
):
    name: str
    mode: Literal["fast", "slow"] = literalFallback("fast")


@dataclass
class WithLiteralNoValidation(
    Versionable,
    version=1,
    hash="3c2d1b",
    register=False,
    validate_literals=False,
):
    name: str
    mode: Literal["fast", "slow"] = "fast"
