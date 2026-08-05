"""The Python half of the golden-corpus contract.

`conformance/golden/` holds checked-in files written by every backend, plus a `manifest.json`
per fixture recording the schema hash, version, and field values in a language-neutral
comparable form. The C# suite reads the same bytes and asserts against the same manifests.

These tests never regenerate anything: they load the committed bytes and compare. A failure
means either a serializer changed behaviour (regenerate deliberately with
`pixi run -- python conformance/generate_golden.py` and review the diff) or a reader broke.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("numpy")
pytest.importorskip("yaml")
pytest.importorskip("tomlkit")
pytest.importorskip("h5py")

import versionable
from versionable import Versionable
from versionable._base import metadata as get_metadata

_CONFORMANCE_DIR = Path(__file__).resolve().parent.parent / "conformance"
if str(_CONFORMANCE_DIR) not in sys.path:
    sys.path.insert(0, str(_CONFORMANCE_DIR))

import generate_golden  # noqa: E402  # the comparable encoder is part of the contract
import golden_schemas  # noqa: E402
from generate_golden import BACKENDS  # noqa: E402

GOLDEN_ROOT = _CONFORMANCE_DIR / "golden"


def _load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


INDEX = _load_json(GOLDEN_ROOT / "index.json")
FIXTURE_NAMES = [entry["fixture"] for entry in INDEX["fixtures"]]


def _manifest(fixture: str) -> dict[str, Any]:
    return dict(_load_json(GOLDEN_ROOT / fixture / "manifest.json"))


def _load(cls: type[Versionable], path: Path) -> Versionable:
    """Load *path*, eagerly materializing HDF5 arrays so values can be compared.

    HDF5 loads are lazy by default: array fields come back as sentinels that read from disk on
    access. `preload='*'` forces them, which is what a value comparison needs. The lazy path
    is covered separately by `TestLazyHdf5Load`.
    """
    if path.suffix == ".h5":
        return versionable.load(cls, path, preload="*")
    return versionable.load(cls, path)


def _encode(obj: Versionable) -> dict[str, Any]:
    return {name: generate_golden.toComparable(getattr(obj, name)) for name in get_metadata(type(obj)).fields}


def _schema_class(serialization_name: str) -> type[Versionable]:
    """Resolve a Serialization Name to its class in `golden_schemas`."""
    for candidate in vars(golden_schemas).values():
        if (
            isinstance(candidate, type)
            and issubclass(candidate, Versionable)
            and candidate is not Versionable
            and get_metadata(candidate).name == serialization_name
            and get_metadata(candidate).version == _current_version(serialization_name)
        ):
            return candidate
    raise LookupError(f"no class in golden_schemas with serialization name {serialization_name!r}")


def _current_version(serialization_name: str) -> int:
    """The version the corpus treats as current for a Serialization Name."""
    for entry in INDEX["fixtures"]:
        if entry["serializationName"] == serialization_name:
            return int(entry["version"])
    raise LookupError(serialization_name)


# ---------------------------------------------------------------------------
# Corpus shape
# ---------------------------------------------------------------------------


class TestCorpusShape:
    """The corpus is complete and self-consistent before anything is loaded."""

    def test_index_lists_the_four_backends(self) -> None:
        assert set(INDEX["backends"]) == {"json", "yaml", "toml", "hdf5"}

    def test_index_matches_the_directories_on_disk(self) -> None:
        on_disk = sorted(p.name for p in GOLDEN_ROOT.iterdir() if p.is_dir())
        assert on_disk == sorted(FIXTURE_NAMES)

    def test_corpus_is_not_empty(self) -> None:
        assert len(FIXTURE_NAMES) >= 10

    def test_known_gaps_are_published_in_the_index(self) -> None:
        """The corpus is the contract the C# side reads — gaps belong here, not in a task report."""
        gaps = INDEX["knownGaps"]
        assert gaps, "knownGaps must document what the grammar defines but the corpus omits"
        for gap in gaps:
            assert set(gap) == {"construct", "example", "status", "detail", "followUp"}
        constructs = {gap["construct"] for gap in gaps}
        assert "heterogeneous fixed tuple" in constructs
        assert "bare (un-parameterized) ndarray" in constructs

    def test_heterogeneous_fixed_tuple_gap_is_still_real(self) -> None:
        """If this starts passing, the gap is fixed — delete the entry instead of the test."""
        import tempfile
        from dataclasses import dataclass

        @dataclass
        class MixedTuple(Versionable, version=1, register=False):
            row: tuple[int, str, float] = (1, "x", 2.0)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "mixed.json"
            versionable.save(MixedTuple(), path)
            with pytest.raises((ValueError, TypeError)):
                versionable.load(MixedTuple, path)

    def test_at_least_one_migration_fixture_exists(self) -> None:
        with_migrations = [e for e in INDEX["fixtures"] if e["migrationSources"]]
        assert with_migrations, "the corpus must exercise migrating an old file forward"
        assert any(len(e["migrationSources"]) >= 2 for e in with_migrations), (
            "at least one fixture must chain two migrations (v1 -> v2 -> v3)"
        )

    @pytest.mark.parametrize("fixture", FIXTURE_NAMES)
    def test_every_backend_file_exists(self, fixture: str) -> None:
        manifest = _manifest(fixture)
        assert set(manifest["files"]) == set(BACKENDS)
        for file_name in manifest["files"].values():
            assert (GOLDEN_ROOT / fixture / file_name).is_file()

    @pytest.mark.parametrize("fixture", FIXTURE_NAMES)
    def test_declared_hash_matches_the_manifest(self, fixture: str) -> None:
        """The schema literal in `golden_schemas` and the corpus cannot drift apart."""
        manifest = _manifest(fixture)
        cls = _schema_class(manifest["serializationName"])
        meta = get_metadata(cls)
        assert meta.hash == manifest["schemaHash"]
        assert meta.version == manifest["version"]


# ---------------------------------------------------------------------------
# Round-trip
# ---------------------------------------------------------------------------


def _current_cases() -> list[tuple[str, str]]:
    return [(fixture, backend) for fixture in FIXTURE_NAMES for backend in BACKENDS]


class TestCurrentVersionFiles:
    """Every committed file loads into the same values the manifest records."""

    @pytest.mark.parametrize(("fixture", "backend"), _current_cases())
    def test_values_match_the_manifest(self, fixture: str, backend: str) -> None:
        manifest = _manifest(fixture)
        cls = _schema_class(manifest["serializationName"])
        path = GOLDEN_ROOT / fixture / manifest["files"][backend]

        assert _encode(_load(cls, path)) == manifest["values"]

    @pytest.mark.parametrize("fixture", FIXTURE_NAMES)
    def test_all_backends_agree_with_each_other(self, fixture: str) -> None:
        """A value that survives JSON but not TOML would otherwise hide behind the manifest."""
        manifest = _manifest(fixture)
        cls = _schema_class(manifest["serializationName"])
        encoded = {
            backend: _encode(_load(cls, GOLDEN_ROOT / fixture / file_name))
            for backend, file_name in manifest["files"].items()
        }
        reference = encoded["json"]
        for backend, values in encoded.items():
            assert values == reference, f"{backend} disagrees with json for fixture {fixture!r}"


# ---------------------------------------------------------------------------
# Migrations
# ---------------------------------------------------------------------------


def _migration_cases() -> list[tuple[str, int, str]]:
    cases: list[tuple[str, int, str]] = []
    for fixture in FIXTURE_NAMES:
        for source in _manifest(fixture).get("migrationSources", []):
            cases.extend((fixture, int(source["fromVersion"]), backend) for backend in source["files"])
    return cases


class TestMigrationSources:
    """Old-version files migrate forward to the current schema on load."""

    @pytest.mark.parametrize(("fixture", "from_version", "backend"), _migration_cases())
    def test_old_file_migrates_to_expected_values(self, fixture: str, from_version: int, backend: str) -> None:
        manifest = _manifest(fixture)
        cls = _schema_class(manifest["serializationName"])
        source = next(s for s in manifest["migrationSources"] if s["fromVersion"] == from_version)
        path = GOLDEN_ROOT / fixture / source["files"][backend]

        assert _encode(_load(cls, path)) == source["expected"]

    @pytest.mark.parametrize("fixture", FIXTURE_NAMES)
    def test_old_files_declare_the_old_version_and_hash(self, fixture: str) -> None:
        """The files really are old — otherwise the migration path is never exercised."""
        manifest = _manifest(fixture)
        for source in manifest.get("migrationSources", []):
            envelope = _load_json(GOLDEN_ROOT / fixture / source["files"]["json"])["__versionable__"]
            assert envelope["version"] == source["fromVersion"]
            assert envelope["hash"] == source["schemaHash"]
            assert envelope["hash"] != manifest["schemaHash"]


# ---------------------------------------------------------------------------
# Wire-format details the C# side has to reproduce
# ---------------------------------------------------------------------------


class TestLazyHdf5Load:
    """The default (lazy) HDF5 read must materialize to the same values as the eager one."""

    def test_lazy_arrays_match_the_manifest(self) -> None:
        import numpy as np

        manifest = _manifest("arrays")
        cls = _schema_class(manifest["serializationName"])
        lazy = versionable.load(cls, GOLDEN_ROOT / "arrays" / manifest["files"]["hdf5"])

        for name in ("signal", "weights", "counts", "image", "mask", "matrix"):
            expected = manifest["values"][name]["$ndarray"]
            actual = np.asarray(getattr(lazy, name))
            assert actual.dtype.name == expected["dtype"]
            assert list(actual.shape) == expected["shape"]
            assert actual.tolist() == expected["data"]

        for index, element in enumerate(lazy.traces):
            assert np.asarray(element).tolist() == manifest["values"]["traces"][index]["$ndarray"]["data"]

        by_key = dict(manifest["values"]["channels"]["$dict"])
        for key, element in lazy.channels.items():
            assert np.asarray(element).tolist() == by_key[key]["$ndarray"]["data"]


class TestWireFormat:
    """Spot-checks on the committed bytes themselves, not just the loaded values."""

    def test_json_envelope_carries_name_version_and_hash(self) -> None:
        envelope = _load_json(GOLDEN_ROOT / "scalars" / "scalars.json")["__versionable__"]
        assert envelope == {"object": "GoldenScalars", "version": 1, "hash": "c51ec9"}

    def test_polymorphic_elements_carry_their_concrete_class(self) -> None:
        shapes = _load_json(GOLDEN_ROOT / "polymorphic" / "polymorphic.json")["shapes"]
        assert [s["__versionable__"]["object"] for s in shapes] == [
            "GoldenCircle",
            "GoldenSquare",
            "GoldenCircle",
        ]

    def test_toml_omits_null_fields(self) -> None:
        """TOML has no null literal; the reader refills the field from the dataclass default."""
        text = (GOLDEN_ROOT / "optionals" / "optionals.toml").read_text(encoding="utf-8")
        assert "absent" not in text
        assert "present" in text

    def test_hdf5_arrays_use_the_default_compression(self) -> None:
        import h5py

        with h5py.File(GOLDEN_ROOT / "arrays" / "arrays.h5", "r") as handle:
            dataset = handle["signal"]
            assert dataset.compression == "gzip"
            assert dataset.compression_opts == 4
            assert dataset.shuffle is True

    def test_hdf5_dtypes_are_preserved_on_disk(self) -> None:
        import h5py
        import numpy as np

        expected = {
            "signal": np.float64,
            "weights": np.float32,
            "counts": np.int32,
            "image": np.uint8,
            "mask": np.bool_,
        }
        with h5py.File(GOLDEN_ROOT / "arrays" / "arrays.h5", "r") as handle:
            for name, dtype in expected.items():
                assert handle[name].dtype == dtype
