"""Cycle-detection tests for ``serialize()`` and the HDF5 writer.

PR 1 (lands before 0.2.0) replaces ``RecursionError`` on cyclic object
graphs with a clear :class:`CircularReferenceError` carrying the field
path of the revisit.  These tests pin the four cycle shapes covered by
the design, plus the diamond non-cycle that must NOT be flagged.

Lossless support for shared references (and therefore cycles, on
opt-in) is the subject of PR 2 / 0.3.0.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pytest

import versionable
from versionable import CircularReferenceError, Versionable

# ---------------------------------------------------------------------------
# Backend extensions (skip unavailable optional deps)
# ---------------------------------------------------------------------------

_DICT_BACKENDS: list[str] = [".json"]

try:
    import toml as _toml  # noqa: F401

    _DICT_BACKENDS.append(".toml")
except ImportError:
    pass

try:
    import yaml as _yaml  # noqa: F401

    _DICT_BACKENDS.append(".yaml")
except ImportError:
    pass

try:
    import versionable._hdf5_backend

    _HDF5_BACKENDS: list[str] = [".h5"]
except ImportError:
    _HDF5_BACKENDS = []

_ALL_BACKENDS = [*_DICT_BACKENDS, *_HDF5_BACKENDS]


# ---------------------------------------------------------------------------
# Test fixtures — module-level so type annotations resolve
# ---------------------------------------------------------------------------


@dataclass
class _Node(Versionable, version=1, hash="b54495", register=False):
    name: str
    children: list[_Node] = field(default_factory=list)


@dataclass
class _Person(Versionable, version=1, hash="130ace", register=False):
    name: str
    partner: _Person | None = None


@dataclass
class _PersonWithPartners(Versionable, version=1, hash="3a6a3c", register=False):
    name: str
    partners: dict[str, _PersonWithPartners] = field(default_factory=dict)


@dataclass
class _DiamondInner(Versionable, version=1, hash="da39fe", register=False):
    value: int


@dataclass
class _DiamondOuter(Versionable, version=1, hash="643c51", register=False):
    left: _DiamondInner
    right: _DiamondInner


# ---------------------------------------------------------------------------
# Self-cycle (the simplest case): n.children = [n]
# ---------------------------------------------------------------------------


class TestSelfCycle:
    """A Versionable whose own field references itself directly."""

    @pytest.mark.parametrize("ext", _ALL_BACKENDS)
    def test_raises_circular_reference_error(self, tmp_path: Path, ext: str) -> None:
        n = _Node(name="root")
        n.children.append(n)
        with pytest.raises(CircularReferenceError, match=r"children\[0\]"):
            versionable.save(n, tmp_path / f"out{ext}")

    def test_error_includes_type_and_id(self, tmp_path: Path) -> None:
        n = _Node(name="root")
        n.children.append(n)
        with pytest.raises(CircularReferenceError) as excinfo:
            versionable.save(n, tmp_path / "out.json")
        msg = str(excinfo.value)
        assert "_Node" in msg
        assert f"@{id(n):x}" in msg


# ---------------------------------------------------------------------------
# Mutual cycle: a.partner = b, b.partner = a
# ---------------------------------------------------------------------------


class TestMutualCycle:
    """Two Versionable instances that reference each other."""

    @pytest.mark.parametrize("ext", _ALL_BACKENDS)
    def test_raises_circular_reference_error(self, tmp_path: Path, ext: str) -> None:
        alice = _Person(name="alice")
        bob = _Person(name="bob")
        alice.partner = bob
        bob.partner = alice
        with pytest.raises(CircularReferenceError, match=r"partner"):
            versionable.save(alice, tmp_path / f"out{ext}")


# ---------------------------------------------------------------------------
# Three-way cycle: a → b → c → a
# ---------------------------------------------------------------------------


class TestThreeWayCycle:
    """Three Versionable instances forming a cycle through ``partner``."""

    @pytest.mark.parametrize("ext", _ALL_BACKENDS)
    def test_raises_circular_reference_error(self, tmp_path: Path, ext: str) -> None:
        a = _Person(name="a")
        b = _Person(name="b")
        c = _Person(name="c")
        a.partner = b
        b.partner = c
        c.partner = a
        with pytest.raises(CircularReferenceError, match=r"partner\.partner\.partner"):
            versionable.save(a, tmp_path / f"out{ext}")


# ---------------------------------------------------------------------------
# Cycle through a list of Versionables (not just a self-cycle)
# ---------------------------------------------------------------------------


class TestCycleThroughList:
    """Two Nodes referencing each other through their ``children`` lists."""

    @pytest.mark.parametrize("ext", _ALL_BACKENDS)
    def test_raises_circular_reference_error(self, tmp_path: Path, ext: str) -> None:
        a = _Node(name="a")
        b = _Node(name="b")
        a.children.append(b)
        b.children.append(a)
        with pytest.raises(CircularReferenceError, match=r"children\[0\]\.children\[0\]"):
            versionable.save(a, tmp_path / f"out{ext}")


# ---------------------------------------------------------------------------
# Cycle through a dict
# ---------------------------------------------------------------------------


class TestCycleThroughDict:
    """A Person referencing itself through its ``partners`` dict."""

    @pytest.mark.parametrize("ext", _ALL_BACKENDS)
    def test_raises_circular_reference_error(self, tmp_path: Path, ext: str) -> None:
        alice = _PersonWithPartners(name="alice")
        alice.partners["self"] = alice
        with pytest.raises(CircularReferenceError, match=r"partners\['self'\]"):
            versionable.save(alice, tmp_path / f"out{ext}")


# ---------------------------------------------------------------------------
# Diamond non-cycle: same instance referenced twice in unrelated branches.
#
# This is the regression guard for the try/finally discard.  Without
# discarding ``id(obj)`` on the way back up, the second leg of the
# diamond would falsely look like a cycle.  PR 1 still duplicates the
# shared instance on disk — that is what PR 2 fixes — but it must not
# raise.
# ---------------------------------------------------------------------------


class TestDiamondNotFlagged:
    """Two fields pointing at the same Versionable instance is NOT a cycle."""

    @pytest.mark.parametrize("ext", _ALL_BACKENDS)
    def test_save_succeeds(self, tmp_path: Path, ext: str) -> None:
        shared = _DiamondInner(value=42)
        parent = _DiamondOuter(left=shared, right=shared)
        path = tmp_path / f"out{ext}"
        # Must not raise — diamonds are still duplicated in 0.2.x but not flagged.
        versionable.save(parent, path)

    @pytest.mark.parametrize("ext", _ALL_BACKENDS)
    def test_load_yields_distinct_instances(self, tmp_path: Path, ext: str) -> None:
        # Round-trip works (with duplication — PR 2 / 0.3.0 will preserve
        # identity, but for 0.2.x we just want it to succeed).
        shared = _DiamondInner(value=42)
        parent = _DiamondOuter(left=shared, right=shared)
        path = tmp_path / f"out{ext}"
        versionable.save(parent, path)
        loaded = versionable.load(_DiamondOuter, path)
        assert loaded.left.value == 42
        assert loaded.right.value == 42
        # Identity is NOT preserved in 0.2.x — two distinct instances on load.
        assert loaded.left is not loaded.right


# ---------------------------------------------------------------------------
# Repeated references in a list (not a cycle)
# ---------------------------------------------------------------------------


class TestRepeatedReferenceInList:
    """The same Versionable appearing twice in a sibling list is NOT a cycle."""

    @pytest.mark.parametrize("ext", _ALL_BACKENDS)
    def test_save_succeeds(self, tmp_path: Path, ext: str) -> None:
        shared = _Node(name="shared")
        parent = _Node(name="parent", children=[shared, shared])
        path = tmp_path / f"out{ext}"
        # Must not raise — two list slots holding the same instance is a
        # diamond, not a cycle.
        versionable.save(parent, path)
