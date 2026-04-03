"""Shared fixtures for versionable tests."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Literal, Optional

import numpy as np
import numpy.typing as npt

from versionable._base import Versionable
from versionable._hash import computeHash
from versionable._types import literalFallback

# --- Sample classes used across tests ---


def _hash(**fields: type) -> str:
    return computeHash(fields)


@dataclass
class SimpleConfig(
    Versionable, version=1, hash=computeHash({"name": str, "debug": bool, "retries": int}), register=False
):
    name: str
    debug: bool = False
    retries: int = 3


@dataclass
class WithOptional(
    Versionable,
    version=1,
    hash=computeHash({"label": str, "description": Optional[str]}),  # noqa: UP045 — hash depends on Optional form
    register=False,
):
    label: str
    description: str | None = None


@dataclass
class WithArray(
    Versionable,
    version=1,
    hash=computeHash({"name": str, "data": npt.NDArray[np.float64]}),
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
    hash=computeHash({"title": str, "priority": Priority}),
    register=False,
):
    title: str
    priority: Priority


@dataclass
class Inner(
    Versionable,
    version=1,
    hash=computeHash({"x": float, "y": float}),
    name="Inner",
):
    x: float
    y: float


@dataclass
class WithNested(
    Versionable,
    version=1,
    hash=computeHash({"name": str, "point": Inner}),
    register=False,
):
    name: str
    point: Inner


@dataclass
class WithDatetime(
    Versionable,
    version=1,
    hash=computeHash({"label": str, "createdAt": datetime}),
    register=False,
):
    label: str
    createdAt: datetime


@dataclass
class WithList(
    Versionable,
    version=1,
    hash=computeHash({"tags": list[str], "scores": list[float]}),
    register=False,
):
    tags: list[str]
    scores: list[float] = field(default_factory=list)


@dataclass
class WithSkipDefaults(
    Versionable,
    version=1,
    hash=computeHash({"name": str, "debug": bool, "count": int}),
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
    hash=computeHash({"name": str, "mode": Literal["fast", "slow"]}),
    register=False,
):
    name: str
    mode: Literal["fast", "slow"] = "fast"


@dataclass
class WithLiteralFallback(
    Versionable,
    version=1,
    hash=computeHash({"name": str, "mode": Literal["fast", "slow"]}),
    register=False,
):
    name: str
    mode: Literal["fast", "slow"] = literalFallback("fast")


@dataclass
class WithLiteralNoValidation(
    Versionable,
    version=1,
    hash=computeHash({"name": str, "mode": Literal["fast", "slow"]}),
    register=False,
    validate_literals=False,
):
    name: str
    mode: Literal["fast", "slow"] = "fast"
