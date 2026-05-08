"""Regression: nested Versionable fields with envelope-like names round-trip cleanly.

Before the fix, ``_stripEnvelope`` silently dropped any user field whose name matched a
lowercase entry in ``_ENVELOPE_KEYS`` (``object``, ``version``, ``format``, ``format_be``,
``shared_refs``). Loading then failed with a ``TypeError`` on missing required arguments.

The flat-lowercase nested envelope layout was a transient dev-only format that never
shipped in any released file format, so the entries were dead weight. The wrapped layout
under ``__versionable__`` is what every backend writes and reads.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from versionable import Versionable, load, save

# ---------------------------------------------------------------------------
# Backend extensions — gate on availability so the minimal env (JSON only)
# can still run this file.
# ---------------------------------------------------------------------------

_DICT_BACKENDS: list[str] = [".json"]

try:
    import tomlkit as _tomlkit  # noqa: F401

    _DICT_BACKENDS.append(".toml")
except ImportError:
    pass

try:
    import yaml as _yaml  # noqa: F401

    _DICT_BACKENDS.append(".yaml")
except ImportError:
    pass

try:
    import versionable._hdf5_backend  # noqa: F401

    _HDF5_BACKENDS: list[str] = [".h5"]
except ImportError:
    _HDF5_BACKENDS = []

_ALL_BACKENDS = [*_DICT_BACKENDS, *_HDF5_BACKENDS]

# ---------------------------------------------------------------------------
# Inner/outer pairs — one per envelope-like field name. Each inner has a
# ``payload`` field (control) plus the envelope-named field.
# ---------------------------------------------------------------------------


@dataclass
class _InnerObject(Versionable, version=1, hash="3dbe5a", register=False):
    payload: str
    object: str


@dataclass
class _OuterObject(Versionable, version=1, hash="4cf033", register=False):
    inner: _InnerObject


@dataclass
class _InnerVersion(Versionable, version=1, hash="82da00", register=False):
    payload: str
    version: str


@dataclass
class _OuterVersion(Versionable, version=1, hash="d36117", register=False):
    inner: _InnerVersion


@dataclass
class _InnerFormat(Versionable, version=1, hash="11a1f2", register=False):
    payload: str
    format: str


@dataclass
class _OuterFormat(Versionable, version=1, hash="f29f62", register=False):
    inner: _InnerFormat


@dataclass
class _InnerFormatBe(Versionable, version=1, hash="186900", register=False):
    payload: str
    format_be: str


@dataclass
class _OuterFormatBe(Versionable, version=1, hash="421642", register=False):
    inner: _InnerFormatBe


@dataclass
class _InnerSharedRefs(Versionable, version=1, hash="5139c8", register=False):
    payload: str
    shared_refs: str


@dataclass
class _OuterSharedRefs(Versionable, version=1, hash="9330f1", register=False):
    inner: _InnerSharedRefs


# ---------------------------------------------------------------------------
# Parametrized round-trip tests across all backends
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("ext", _ALL_BACKENDS)
def test_nested_field_named_object_roundtrips(tmp_path: Path, ext: str) -> None:
    inst = _OuterObject(inner=_InnerObject(payload="p", object="o-value"))
    p = tmp_path / f"out{ext}"
    save(inst, p)
    loaded = load(_OuterObject, p)
    assert loaded.inner.object == "o-value"
    assert loaded.inner.payload == "p"


@pytest.mark.parametrize("ext", _ALL_BACKENDS)
def test_nested_field_named_version_roundtrips(tmp_path: Path, ext: str) -> None:
    inst = _OuterVersion(inner=_InnerVersion(payload="p", version="v-value"))
    p = tmp_path / f"out{ext}"
    save(inst, p)
    loaded = load(_OuterVersion, p)
    assert loaded.inner.version == "v-value"
    assert loaded.inner.payload == "p"


@pytest.mark.parametrize("ext", _ALL_BACKENDS)
def test_nested_field_named_format_roundtrips(tmp_path: Path, ext: str) -> None:
    inst = _OuterFormat(inner=_InnerFormat(payload="p", format="f-value"))
    p = tmp_path / f"out{ext}"
    save(inst, p)
    loaded = load(_OuterFormat, p)
    assert loaded.inner.format == "f-value"
    assert loaded.inner.payload == "p"


@pytest.mark.parametrize("ext", _ALL_BACKENDS)
def test_nested_field_named_format_be_roundtrips(tmp_path: Path, ext: str) -> None:
    inst = _OuterFormatBe(inner=_InnerFormatBe(payload="p", format_be="fbe-value"))
    p = tmp_path / f"out{ext}"
    save(inst, p)
    loaded = load(_OuterFormatBe, p)
    assert loaded.inner.format_be == "fbe-value"
    assert loaded.inner.payload == "p"


@pytest.mark.parametrize("ext", _ALL_BACKENDS)
def test_nested_field_named_shared_refs_roundtrips(tmp_path: Path, ext: str) -> None:
    inst = _OuterSharedRefs(inner=_InnerSharedRefs(payload="p", shared_refs="sr-value"))
    p = tmp_path / f"out{ext}"
    save(inst, p)
    loaded = load(_OuterSharedRefs, p)
    assert loaded.inner.shared_refs == "sr-value"
    assert loaded.inner.payload == "p"


# ---------------------------------------------------------------------------
# Combined-fields case: a single inner with all five envelope-named fields,
# verifying that multiple collisions on the same object don't interact badly.
# ---------------------------------------------------------------------------


@dataclass
class _InnerMultiple(Versionable, version=1, hash="491ce7", register=False):
    payload: str
    object: str
    version: str
    format: str
    format_be: str
    shared_refs: str


@dataclass
class _OuterMultiple(Versionable, version=1, hash="b3a35c", register=False):
    inner: _InnerMultiple


@pytest.mark.parametrize("ext", _ALL_BACKENDS)
def test_multiple_envelope_named_fields_roundtrip(tmp_path: Path, ext: str) -> None:
    """All five envelope-named fields on one nested object round-trip together."""
    inst = _OuterMultiple(
        inner=_InnerMultiple(
            payload="p",
            object="o",
            version="v",
            format="f",
            format_be="fbe",
            shared_refs="sr",
        )
    )
    p = tmp_path / f"out{ext}"
    save(inst, p)
    loaded = load(_OuterMultiple, p)
    assert loaded.inner.payload == "p"
    assert loaded.inner.object == "o"
    assert loaded.inner.version == "v"
    assert loaded.inner.format == "f"
    assert loaded.inner.format_be == "fbe"
    assert loaded.inner.shared_refs == "sr"
