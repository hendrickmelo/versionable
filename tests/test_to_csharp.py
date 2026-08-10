"""Tests for the ``python -m versionable.tools.to_csharp`` scaffolder.

Most of these read the emitted C# as text, which is the tool's actual contract: a human
reviews the file, so what it *says* is what matters.

One test does not read text at all.  ``test_scaffolded_csharp_compiles_and_the_analyzer_agrees``
writes a scaffolded file into a throwaway csproj that references the real runtime and the
real Roslyn analyzer, and builds it.  That is the cross-language round trip end to end: the
hash literal in the emitted file came from a Python class definition, and the analyzer
recomputes it from C# symbols and agrees.  It is skipped when dotnet is absent.
"""

from __future__ import annotations

import datetime
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum
from pathlib import Path, PurePosixPath
from typing import Any, Literal
from uuid import UUID

import pytest

from versionable import Migration, MigrationContext, Versionable, literalFallback, migration
from versionable.tools import to_csharp
from versionable.tools._csharp_types import mapType

np = pytest.importorskip("numpy")
npt = pytest.importorskip("numpy.typing")

REPO_ROOT = Path(__file__).resolve().parent.parent
DOTNET_SRC = REPO_ROOT / "dotnet" / "src"


# ---------------------------------------------------------------------------
# Fixture schemas
# ---------------------------------------------------------------------------
#
# All register=False: these mirror shapes that other fixtures in the suite already claim, and
# the registry is process-wide.


class ScaffoldColour(Enum):
    """A string-valued enum with a fallback member."""

    RED = "red"
    GREEN = "green"
    UNKNOWN = "unknown"


ScaffoldColour.VERSIONABLE_FALLBACK = ScaffoldColour.UNKNOWN


class ScaffoldPriority(Enum):
    """An integer-valued enum, whose values C# carries as member values."""

    LOW = 1
    HIGH = 7


@dataclass
class ScaffoldScalars(Versionable, version=1, hash="c51ec9", register=False):
    """Every scalar token with a C# counterpart."""

    text: str
    count: int
    ratio: float
    enabled: bool
    phase: complex
    blob: bytes


@dataclass
class ScaffoldContainers(Versionable, version=1, hash="a7ac49", register=False):
    """The container forms that map onto a C# generic."""

    names: list[str]
    lookup: dict[str, int]
    tags: set[str]
    ids: frozenset[int]
    pair: tuple[int, str]
    matrix: list[list[float]]


@dataclass
class ScaffoldOptionals(Versionable, version=1, hash="3e5e08", register=False):
    """An optional, which C# spells ``T?``, and an n-ary union, which it cannot spell."""

    maybe: str | None
    count: int | None
    either: int | str


@dataclass
class ScaffoldEnums(Versionable, version=1, hash="8ad927", register=False):
    """Enums standalone and in a container."""

    colour: ScaffoldColour
    priority: ScaffoldPriority
    palette: list[ScaffoldColour]


@dataclass
class ScaffoldLiterals(Versionable, version=1, hash="4742a4", register=False):
    """Literal option sets, one of them with a fallback."""

    level: Literal[1, 2, 3]
    mixed: Literal["auto", 0]
    mode: Literal["fast", "slow"] = literalFallback("fast")


@dataclass
class ScaffoldArrays(Versionable, version=1, hash="bb050b", register=False):
    """A dtype-declared array, which maps, and a bare one, which cannot."""

    signal: npt.NDArray[np.float64]
    counts: npt.NDArray[np.int32]
    dynamic: np.ndarray


@dataclass
class ScaffoldLeaf(Versionable, version=1, hash="e37514", register=False):
    """Leaf of the nested fixture."""

    x: float
    y: float


@dataclass
class ScaffoldNested(Versionable, version=1, hash="753b3d", register=False):
    """A nested Versionable standalone, in a list, and as an optional."""

    inner: ScaffoldLeaf
    points: list[ScaffoldLeaf]
    maybeInner: ScaffoldLeaf | None


@dataclass
class ScaffoldAwkward(Versionable, version=1, hash="09c091", register=False):
    """Constructs the grammar defines but C# cannot declare."""

    samples: tuple[float, ...]
    posix: PurePosixPath


@dataclass
class ScaffoldDefaults(Versionable, version=1, hash="86905a", register=False):
    """Defaults that convert, and one factory that does not."""

    names: list[str] = field(default_factory=list)
    lookup: dict[str, int] = field(default_factory=dict)
    label: str = "unset"
    amount: Decimal = Decimal("1.5")
    stamp: datetime.datetime = datetime.datetime(2026, 1, 1, 12, 30)
    deviceId: UUID = field(default_factory=lambda: UUID(int=7))


@dataclass
class ScaffoldRenamed(Versionable, version=4, hash="ed3a90", name="OnDiskName", register=False):
    """A class whose Serialization Name differs from its class name."""

    name: str
    debug: bool = False
    retries: int = 3


@dataclass
class ScaffoldBase(Versionable, version=1, hash="357f27", register=False):
    """Polymorphic base, which must not be emitted sealed."""

    label: str = ""


@dataclass
class ScaffoldDerived(ScaffoldBase, version=1, hash="8e5e7c", register=False):
    """Concrete subclass; only its own field is re-declared in C#."""

    radius: float = 0.0


@dataclass
class ScaffoldMigratingBase(Versionable, version=2, hash="357f27", register=False):
    """A base whose ``Migrate`` class its subclasses inherit through the MRO."""

    label: str = ""

    class Migrate:
        v1 = Migration().rename("title", "label")


@dataclass
class ScaffoldMigratingDerived(ScaffoldMigratingBase, version=2, hash="8e5e7c", register=False):
    """Inherits ``Migrate``, so the C# mirror has to re-declare it with ``new``."""

    radius: float = 0.0


@dataclass
class ScaffoldColliding(Versionable, version=1, hash="83c93d", register=False):
    """Two distinct wire names that PascalCase to one C# property."""

    timeout_ms: int = 0
    timeoutMs: int = 0


@dataclass
class ScaffoldPreciseTemporal(Versionable, version=1, hash="d2349d", register=False):
    """Temporal defaults carrying sub-second detail the obvious C# spellings would drop."""

    stamp: datetime.datetime = datetime.datetime(2026, 1, 2, 3, 4, 5, 678901)
    clock: datetime.time = datetime.time(3, 4, 5, 678901)
    elapsed: datetime.timedelta = datetime.timedelta(days=1, seconds=7384, microseconds=5006)


@dataclass
class ScaffoldDeclarativeMigrations(Versionable, version=3, hash="aac8a2", register=False):
    """A migration chain built entirely from ops with no callables."""

    name: str = ""
    retries: int = 3
    timeout_ms: int = 30000

    class Migrate:
        v1 = Migration().rename("title", "name").drop("debug")
        v2 = Migration().add("timeout_ms", default=0).requiresUpgrade()


@dataclass
class ScaffoldLambdaMigrations(Versionable, version=2, hash="25cd83", register=False):
    """A migration whose ops carry Python callables C# cannot receive."""

    celsius: float = 0.0
    total: float = 0.0

    class Migrate:
        v1 = (
            Migration()
            .convert("celsius", via=lambda value: (value - 32) * 5 / 9)
            .derive("total", from_="celsius", via=lambda value: value * 2)
            .merge(["a", "b"], into="total", via=lambda a, b: a + b)
            .split("pair", into={"left": lambda v: v[0], "right": lambda v: v[1]})
        )


@dataclass
class ScaffoldImperativeMigrations(Versionable, version=2, hash="ece5e8", register=False):
    """A migration written as a function, which only a human can transcribe."""

    doubled: int = 0

    class Migrate:
        @migration(fromVersion=1)
        def from_v1(ctx: MigrationContext) -> None:  # noqa: N805  # @migration wraps this; not a regular method
            ctx["doubled"] = ctx.pop("single") * 2


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _emit(*types: type, namespace: str = "Scaffold.Tests") -> str:
    """Scaffold *types* into one file and return its text."""
    return to_csharp.convertTypes(list(types), moduleName=__name__, namespace=namespace).text


def _todos_for(types: list[type]) -> list[str]:
    """Return the TODO messages the scaffolder produced for *types*."""
    return [item.message for item in to_csharp.convertTypes(types, moduleName=__name__).todos]


# ---------------------------------------------------------------------------
# Type mapping
# ---------------------------------------------------------------------------


class TestScalars:
    def test_widths_are_the_widest_python_could_hold(self):
        text = _emit(ScaffoldScalars)
        assert "public required string Text" in text
        # int is unbounded in Python; long is the widest C# integral, and width is erased
        # from the hash, so the choice costs nothing.
        assert "public required long Count" in text
        assert "public required double Ratio" in text
        assert "public required bool Enabled" in text

    def test_bytes_is_a_byte_array_not_a_list_of_int(self):
        # GRAMMAR section 7: byte[] is the carve-out that renders `bytes`.
        assert "public required byte[] Blob" in _emit(ScaffoldScalars)

    def test_complex_maps_to_system_numerics(self):
        text = _emit(ScaffoldScalars)
        assert "public required Complex Phase" in text
        assert "using System.Numerics;" in text

    def test_the_declared_hash_is_carried_over_verbatim(self):
        # The whole point: a canonical payload is language-neutral, so the literal is the
        # same one on both sides and the analyzer re-derives it.
        assert '[Versionable(Version = 1, Hash = "c51ec9", Register = false)]' in _emit(ScaffoldScalars)


class TestContainers:
    def test_each_container_maps_to_its_csharp_generic(self):
        text = _emit(ScaffoldContainers)
        assert "List<string> Names" in text
        assert "Dictionary<string, long> Lookup" in text
        assert "HashSet<string> Tags" in text
        assert "FrozenSet<long> Ids" in text
        assert "using System.Collections.Frozen;" in text

    def test_a_fixed_tuple_becomes_a_value_tuple_in_declaration_order(self):
        # Order is hash-significant for a fixed tuple (GRAMMAR section 5).
        assert "(long, string) Pair" in _emit(ScaffoldContainers)

    def test_containers_nest_without_special_casing(self):
        assert "List<List<double>> Matrix" in _emit(ScaffoldContainers)


class TestOptionalsAndUnions:
    def test_optional_becomes_a_nullable(self):
        text = _emit(ScaffoldOptionals)
        assert "string? Maybe" in text
        assert "long? Count" in text

    def test_an_n_ary_union_is_a_todo_citing_the_grammar(self):
        todos = _todos_for([ScaffoldOptionals])
        assert len(todos) == 1
        assert "Union[int, str]" in todos[0]
        assert "csharpDeclarable" in todos[0]

    def test_the_n_ary_union_todo_is_emitted_above_the_property(self):
        text = _emit(ScaffoldOptionals)
        todo_line = next(line for line in text.splitlines() if "TODO (either)" in line)
        assert text.index(todo_line) < text.index("Either { get; init; }")


class TestEnums:
    def test_string_valued_members_carry_their_wire_value(self):
        text = _emit(ScaffoldEnums)
        assert '[EnumValue("red")]' in text
        assert "Red," in text

    def test_the_python_fallback_member_becomes_the_attribute(self):
        text = _emit(ScaffoldEnums)
        unknown = text[text.index('[EnumValue("unknown")]') :]
        assert unknown.startswith('[EnumValue("unknown")]\n    [EnumFallback]\n    Unknown,')

    def test_integer_valued_members_keep_their_numbers(self):
        text = _emit(ScaffoldEnums)
        assert "Low = 1," in text
        assert "High = 7," in text

    def test_a_referenced_enum_is_pulled_into_the_same_file(self):
        # The class does not compile without it, and both live in the same Python module.
        text = _emit(ScaffoldEnums)
        assert "public enum ScaffoldColour" in text
        assert text.index("public enum ScaffoldColour") < text.index("class ScaffoldEnums")

    def test_screaming_snake_members_are_pascal_cased(self):
        assert "RED," not in _emit(ScaffoldEnums)


class TestLiterals:
    def test_options_keep_their_declaration_order_and_kind(self):
        # Order is hash-significant and strings are distinguishable from ints (section 8).
        text = _emit(ScaffoldLiterals)
        assert "[LiteralValues(1, 2, 3)]" in text
        assert "public required int Level" in text

    def test_a_mixed_option_set_is_typed_object(self):
        text = _emit(ScaffoldLiterals)
        assert '[LiteralValues("auto", 0)]' in text
        assert "public required object Mixed" in text

    def test_literal_fallback_becomes_the_attribute_argument(self):
        text = _emit(ScaffoldLiterals)
        assert '[LiteralValues("fast", "slow", Fallback = "fast")]' in text
        assert 'public string Mode { get; init; } = "fast";' in text


class TestArrays:
    def test_a_declared_dtype_becomes_the_tensor_element_type(self):
        text = _emit(ScaffoldArrays)
        assert "Tensor<double> Signal" in text
        assert "Tensor<int> Counts" in text
        assert "using System.Numerics.Tensors;" in text

    def test_a_bare_ndarray_is_a_todo_because_tensor_is_always_typed(self):
        todos = _todos_for([ScaffoldArrays])
        assert len(todos) == 1
        assert "bare `ndarray`" in todos[0]
        assert "section 7" in todos[0]


class TestNested:
    def test_a_nested_versionable_is_referenced_by_its_class_name(self):
        text = _emit(ScaffoldLeaf, ScaffoldNested)
        assert "public required ScaffoldLeaf Inner" in text
        assert "List<ScaffoldLeaf> Points" in text
        assert "ScaffoldLeaf? MaybeInner" in text

    def test_a_type_from_another_module_is_named_in_the_header(self):
        emitted = to_csharp.convertTypes([ScaffoldNested], moduleName="other.module")
        assert "Referenced by these declarations but not emitted in this run" in emitted.text
        assert "ScaffoldLeaf" in emitted.text.split("namespace")[0]


class TestUnconvertibleConstructs:
    def test_a_variadic_tuple_says_the_hash_will_not_match(self):
        todos = _todos_for([ScaffoldAwkward])
        variadic = next(item for item in todos if "variadic tuple" in item)
        assert "section 5" in variadic
        assert "will NOT match" in variadic

    def test_a_pure_path_says_it_has_no_csharp_counterpart(self):
        todos = _todos_for([ScaffoldAwkward])
        pure = next(item for item in todos if "PurePosixPath" in item)
        assert "section 9" in pure
        assert "will NOT match" in pure


class TestDefaults:
    def test_an_empty_container_factory_converts_rather_than_stalling(self):
        # `field(default_factory=list)` is how a dataclass spells "starts empty"; a TODO on
        # it would land on nearly every real schema.
        text = _emit(ScaffoldDefaults)
        assert "List<string> Names { get; init; } = [];" in text
        assert "Dictionary<string, long> Lookup { get; init; } = new();" in text

    def test_scalar_and_converter_defaults_convert(self):
        text = _emit(ScaffoldDefaults)
        assert 'public string Label { get; init; } = "unset";' in text
        assert "public decimal Amount { get; init; } = 1.5m;" in text
        assert "new DateTime(2026, 1, 1, 12, 30, 0);" in text

    def test_a_callable_factory_is_a_todo_and_the_property_becomes_required(self):
        text = _emit(ScaffoldDefaults)
        assert "public required Guid DeviceId { get; init; }" in text
        todo = next(item for item in _todos_for([ScaffoldDefaults]) if "`default_factory`" in item)
        assert "required" in todo
        # The Python source is commented under the TODO so the reader can port it.
        assert "//     deviceId: UUID = field(default_factory=lambda: UUID(int=7))" in text

    def test_a_class_used_as_a_factory_is_named_not_pasted(self):
        # inspect.getsource(Path) is several hundred lines of standard library.
        @dataclass
        class WithPathFactory(Versionable, version=1, hash="0dc237", register=False):
            where: Path = field(default_factory=Path)

        text = _emit(WithPathFactory)
        assert "//     pathlib" in text
        assert "PurePath subclass" not in text

    def test_sub_second_precision_survives_every_temporal_default(self):
        # TimeSpan.FromSeconds rounds to the nearest millisecond and the short DateTime
        # constructors stop at seconds; .NET 7 added microsecond overloads for all of them.
        text = _emit(ScaffoldPreciseTemporal)
        assert "new DateTime(2026, 1, 2, 3, 4, 5, 678, 901);" in text
        assert "new TimeOnly(3, 4, 5, 678, 901);" in text
        assert "new TimeSpan(1, 2, 3, 4, 5, 6);" in text
        assert _todos_for([ScaffoldPreciseTemporal]) == []

    def test_a_whole_second_temporal_default_keeps_the_short_form(self):
        text = _emit(ScaffoldDefaults)
        assert "new DateTime(2026, 1, 1, 12, 30, 0);" in text

    def test_an_offset_that_is_not_whole_minutes_is_a_todo_rather_than_a_rounded_value(self):
        # DateTimeOffset rejects a sub-minute offset, and rounding one would move the instant.
        odd = datetime.timezone(datetime.timedelta(seconds=90))

        @dataclass
        class WithOddOffset(Versionable, version=1, hash="6e6064", register=False):
            when: datetime.datetime = datetime.datetime(2026, 1, 1, tzinfo=odd)

        text = _emit(WithOddOffset)
        assert "public required DateTimeOffset When { get; init; }" in text
        assert "no C# literal form" in _todos_for([WithOddOffset])[0]

    def test_an_aware_datetime_default_upgrades_the_property_to_datetimeoffset(self):
        @dataclass
        class WithAware(Versionable, version=1, hash="6e6064", register=False):
            when: datetime.datetime = datetime.datetime(2026, 1, 1, tzinfo=datetime.UTC)

        text = _emit(WithAware)
        assert "public DateTimeOffset When" in text
        assert "new DateTimeOffset(2026, 1, 1, 0, 0, 0, TimeSpan.FromMinutes(0));" in text


class TestClassShape:
    def test_a_name_override_becomes_the_serialization_name_attribute(self):
        text = _emit(ScaffoldRenamed)
        assert '[SerializationName("OnDiskName")]' in text
        # The C# type keeps the Python class name; only the wire identity is pinned.
        assert "class ScaffoldRenamed" in text

    def test_register_false_is_carried_across(self):
        assert 'Hash = "ed3a90", Register = false)]' in _emit(ScaffoldRenamed)

    def test_wire_names_survive_the_pascal_case_rename(self):
        text = _emit(ScaffoldDeclarativeMigrations)
        assert '[VersionableField("timeout_ms")]' in text
        assert "long TimeoutMs" in text

    def test_a_subclassed_base_is_not_sealed_and_the_derived_type_inherits(self):
        text = _emit(ScaffoldBase, ScaffoldDerived)
        assert "public partial class ScaffoldBase" in text
        assert "public sealed partial class ScaffoldDerived : ScaffoldBase" in text

    def test_inherited_fields_are_not_re_declared(self):
        # C# folds base members into the field set exactly as Python's MRO walk does.
        text = _emit(ScaffoldBase, ScaffoldDerived)
        assert text.count("Label { get; init; }") == 1

    def test_sibling_wire_names_that_pascal_case_alike_are_a_todo(self):
        # `timeout_ms` and `timeoutMs` are distinct wire names and distinct schemas, but one
        # C# property name — CS0102. The scaffolder names both rather than picking a winner,
        # because either choice silently renames a hash-significant wire field.
        text = _emit(ScaffoldColliding)
        todos = _todos_for([ScaffoldColliding])
        assert len(todos) == 2
        assert all("`timeout_ms`, `timeoutMs`" in item and "TimeoutMs" in item for item in todos)
        assert "CS0102" in todos[0]
        assert text.count("TODO") >= 2

    def test_names_that_merely_look_alike_do_not_collide(self):
        # `retries` and `timeout_ms` share nothing; only an exact PascalCase match is a clash.
        assert _todos_for([ScaffoldDeclarativeMigrations]) == []

    def test_types_are_partial_because_the_generator_extends_them(self):
        assert "sealed partial class ScaffoldScalars" in _emit(ScaffoldScalars)

    def test_no_auto_generated_marker_is_emitted(self):
        # The analyzer configures GeneratedCodeAnalysisFlags.None, so the marker would make
        # Roslyn skip every hash check in the file.
        assert "auto-generated" not in _emit(ScaffoldScalars)


class TestMigrations:
    def test_declarative_ops_become_the_csharp_builder_chain(self):
        text = _emit(ScaffoldDeclarativeMigrations)
        # One op per line: a migration is the part of a scaffold most likely to be wrong, and
        # a one-line chain of calls is the shape a reviewer skims past.
        assert "public static readonly Migration V1 = new Migration()" in text
        assert '\n            .Rename("title", "name")\n            .Drop("debug");' in text
        assert '\n            .Add("timeout_ms", 0)\n            .RequiresUpgrade();' in text

    def test_the_migrate_class_is_nested_and_static(self):
        text = _emit(ScaffoldDeclarativeMigrations)
        assert "public static class Migrate" in text
        assert "using Versionable.Migrations;" in text

    def test_a_declarative_chain_with_no_callables_produces_no_todos(self):
        assert _todos_for([ScaffoldDeclarativeMigrations]) == []

    def test_callable_carrying_ops_emit_throwing_lambdas_and_todos(self):
        text = _emit(ScaffoldLambdaMigrations)
        assert '.Convert("celsius", value => throw new NotImplementedException(' in text
        assert '.Derive("total", "celsius", value => throw new NotImplementedException(' in text
        assert '.Merge(new[] { "a", "b" }, "total", values => throw new NotImplementedException(' in text
        todos = _todos_for([ScaffoldLambdaMigrations])
        assert len(todos) == 4
        assert all("cannot be converted" in item for item in todos)

    def test_the_python_source_of_a_migration_lambda_is_commented_in(self):
        text = _emit(ScaffoldLambdaMigrations)
        assert '//     .convert("celsius", via=lambda value: (value - 32) * 5 / 9)' in text

    def test_a_split_quotes_its_shared_source_once_not_once_per_target(self):
        # Every target's lambda is written in the one statement, so getsource returns the
        # same block for each; quoting it per target buried the TODO in repeats.
        text = _emit(ScaffoldLambdaMigrations)
        split_todo = next(item for item in _todos_for([ScaffoldLambdaMigrations]) if "Split(" in item)
        assert "cannot be converted" in split_todo
        assert text.count('//     .split("pair", into={"left": lambda v: v[0], "right": lambda v: v[1]})') == 1
        # Both targets still get their own throwing lambda.
        assert '.Split("pair", new SplitTarget("left", value => throw' in text
        assert 'new SplitTarget("right", value => throw' in text

    def test_a_migrate_class_inherited_through_the_mro_is_re_declared_with_new(self):
        # Python's resolveMigrations finds the base's Migrate through the MRO, but Roslyn's
        # GetTypeMembers does not see a base type's nested types — so the chain has to be
        # re-declared here, and re-declaring a name the base has needs `new`, or CS0108 fires
        # and TreatWarningsAsErrors turns it into a failed build.
        text = _emit(ScaffoldMigratingBase, ScaffoldMigratingDerived)
        assert text.count("public static class Migrate") == 1
        assert text.count("public new static class Migrate") == 1
        assert text.count('.Rename("title", "label");') == 2

    def test_the_re_declaration_says_why_it_is_there(self):
        # The summary is wrapped across doc-comment lines, so match its two halves.
        text = _emit(ScaffoldMigratingBase, ScaffoldMigratingDerived)
        assert "Inherited from" in text
        assert "`ScaffoldMigratingBase` in Python, and re-declared here because a nested type" in text

    def test_a_class_owning_its_migrate_does_not_get_the_new_modifier(self):
        assert "new static class Migrate" not in _emit(ScaffoldDeclarativeMigrations)

    def test_an_imperative_migration_becomes_a_throwing_stub_that_keeps_the_chain_contiguous(self):
        text = _emit(ScaffoldImperativeMigrations)
        assert "[Migration(FromVersion = 1)]" in text
        assert "public static void FromV1(MigrationContext ctx) =>" in text
        assert "throw new NotImplementedException" in text
        todo = _todos_for([ScaffoldImperativeMigrations])[0]
        assert "imperative migration from v1" in todo

    def test_the_imperative_body_is_commented_in_so_it_can_be_transcribed(self):
        text = _emit(ScaffoldImperativeMigrations)
        # Commented under the TODO, body indentation and all, inside the nested Migrate class.
        assert '//         ctx["doubled"] = ctx.pop("single") * 2' in text


class TestFileShape:
    def test_the_namespace_defaults_to_the_pascal_cased_module_path(self):
        emitted = to_csharp.convertTypes([ScaffoldScalars], moduleName="my_pkg.schema_defs")
        assert emitted.namespace == "MyPkg.SchemaDefs"
        assert "namespace MyPkg.SchemaDefs;" in emitted.text

    def test_the_file_is_named_after_the_source_module(self):
        emitted = to_csharp.convertTypes([ScaffoldScalars], moduleName="my_pkg.schema_defs")
        assert emitted.fileName == "SchemaDefs.cs"

    def test_the_header_counts_the_todos(self):
        emitted = to_csharp.convertTypes([ScaffoldAwkward], moduleName=__name__)
        assert f"// {len(emitted.todos)} TODO(s) below need a human." in emitted.text

    def test_a_clean_file_has_no_todo_banner(self):
        assert "TODO" not in _emit(ScaffoldScalars)

    def test_usings_are_sorted_and_versionable_is_always_present(self):
        text = _emit(ScaffoldArrays)
        usings = [line for line in text.splitlines() if line.startswith("using ")]
        assert usings == sorted(usings)
        assert "using Versionable;" in usings


class TestTargetResolution:
    def test_a_module_target_takes_every_versionable_it_defines(self):
        files = to_csharp.convert([__name__])
        assert len(files) == 1
        assert "class ScaffoldScalars" in files[0].text
        assert "class ScaffoldNested" in files[0].text

    def test_a_class_target_takes_only_that_class(self):
        files = to_csharp.convert([f"{__name__}:ScaffoldScalars"])
        assert "class ScaffoldScalars" in files[0].text
        assert "class ScaffoldNested" not in files[0].text

    def test_an_unknown_attribute_is_a_lookup_error(self):
        with pytest.raises(LookupError, match="Nope"):
            to_csharp.convert([f"{__name__}:Nope"])

    def test_a_non_versionable_attribute_is_a_type_error(self):
        with pytest.raises(TypeError, match="not a Versionable"):
            to_csharp.convert([f"{__name__}:REPO_ROOT"])


class TestCommandLine:
    def test_it_writes_one_file_per_module(self, tmp_path):
        code = to_csharp.main([f"{__name__}:ScaffoldScalars", "--out", str(tmp_path), "--namespace", "Acme"])
        assert code == 0
        written = list(tmp_path.glob("*.cs"))
        assert len(written) == 1
        assert "namespace Acme;" in written[0].read_text(encoding="utf-8")

    def test_it_prints_to_stdout_by_default(self, capsys):
        assert to_csharp.main([f"{__name__}:ScaffoldScalars"]) == 0
        assert "class ScaffoldScalars" in capsys.readouterr().out

    def test_it_reports_the_outstanding_todo_count_on_stderr(self, capsys):
        to_csharp.main([f"{__name__}:ScaffoldAwkward"])
        assert "TODO(s) need a human" in capsys.readouterr().err


class TestMapTypeDirectly:
    @pytest.mark.parametrize(
        ("annotation", "expected"),
        [
            (str, "string"),
            (int, "long"),
            (float, "double"),
            (bool, "bool"),
            (bytes, "byte[]"),
            (Decimal, "decimal"),
            (UUID, "Guid"),
            (Path, "FilePath"),
            (datetime.date, "DateOnly"),
            (datetime.time, "TimeOnly"),
            (datetime.timedelta, "TimeSpan"),
            (list[int], "List<long>"),
            (dict[str, list[bool]], "Dictionary<string, List<bool>>"),
            (tuple[int, str, float], "(long, string, double)"),
            (str | None, "string?"),
            (list[str] | None, "List<string>?"),
        ],
    )
    def test_annotations_map_to_their_csharp_spelling(self, annotation: Any, expected: str):
        assert mapType(annotation).text == expected

    def test_annotated_metadata_is_ignored_at_any_depth(self):
        from typing import Annotated

        assert mapType(Annotated[float, "dB"]).text == "double"
        assert mapType(list[Annotated[float, "volts"]]).text == "List<double>"

    def test_a_standalone_none_field_is_python_only(self):
        mapped = mapType(type(None))
        assert len(mapped.todos) == 1
        assert "section 4" in mapped.todos[0].message

    def test_an_unregistered_type_renders_its_bare_serialization_name(self):
        class Widget:
            pass

        mapped = mapType(Widget)
        assert mapped.text == "Widget"
        assert "section 9" in mapped.todos[0].message


# ---------------------------------------------------------------------------
# The golden proof: the emitted C# compiles, and the analyzer validates the hash
# ---------------------------------------------------------------------------


@dataclass
class CompileProofLeaf(Versionable, version=1, hash="e37514", register=False):
    """Leaf schema for the compile proof."""

    x: float
    y: float


@dataclass
class CompileProofWide(Versionable, version=2, hash="0db231", register=False):
    """A schema spanning every mapping that has a C# spelling, so the whole table compiles."""

    text: str
    count: int
    ratio: float
    enabled: bool
    phase: complex
    blob: bytes
    names: list[str]
    lookup: dict[str, int]
    tags: set[str]
    ids: frozenset[int]
    pair: tuple[int, int]
    maybe: str | None
    when: datetime.datetime
    day: datetime.date
    elapsed: datetime.timedelta
    amount: Decimal
    deviceId: UUID
    signal: npt.NDArray[np.float64]
    inner: CompileProofLeaf
    colour: ScaffoldColour
    mode: Literal["fast", "slow"] = literalFallback("fast")

    class Migrate:
        v1 = Migration().rename("old_text", "text").add("count", default=0)


@dataclass
class CompileProofBase(Versionable, version=2, hash="357f27", register=False):
    """Polymorphic base owning the migration chain its subclass inherits."""

    label: str = ""

    class Migrate:
        v1 = Migration().rename("title", "label")


@dataclass
class CompileProofDerived(CompileProofBase, version=2, hash="8e5e7c", register=False):
    """Inherits ``Migrate`` through the MRO; its C# mirror must re-declare it with ``new``.

    Present in the compile proof rather than only in an emission assertion because ``new`` is
    exactly the kind of modifier a string test can claim and a compiler disproves: without it
    CS0108 fires, and ``TreatWarningsAsErrors`` turns that into a failed build.
    """

    radius: float = 0.0


_SCRATCH_CSPROJ = """<Project Sdk="Microsoft.NET.Sdk">
  <PropertyGroup>
    <TargetFramework>net8.0</TargetFramework>
    <Nullable>enable</Nullable>
    <ImplicitUsings>enable</ImplicitUsings>
    <LangVersion>12.0</LangVersion>
  </PropertyGroup>
  <ItemGroup>
    <ProjectReference Include="{versionable}" />
    <!-- As an analyzer, and only as an analyzer: analyzer references do not flow through a
         ProjectReference, so referencing the runtime alone would compile the file without ever
         checking a hash — which is the one thing this test is for. -->
    <ProjectReference Include="{analyzers}" OutputItemType="Analyzer" ReferenceOutputAssembly="false" />
  </ItemGroup>
</Project>
"""


def _dotnet() -> str | None:
    """Return a usable dotnet executable, or None when the SDK is not installed."""
    found = shutil.which("dotnet")
    if found:
        return found
    fallback = Path.home() / ".dotnet" / "dotnet"
    return str(fallback) if fallback.exists() else None


def _build(project_dir: Path) -> subprocess.CompletedProcess[str]:
    """Run ``dotnet build`` in *project_dir*."""
    executable = _dotnet()
    assert executable is not None  # guaranteed by the skipif on the calling test
    environment = {**os.environ, "DOTNET_ROOT": str(Path(executable).parent), "DOTNET_CLI_TELEMETRY_OPTOUT": "1"}
    return subprocess.run(
        [executable, "build", "-v", "quiet", "--nologo"],
        cwd=project_dir,
        capture_output=True,
        text=True,
        env=environment,
        check=False,
        timeout=600,
    )


@pytest.mark.skipif(_dotnet() is None, reason="the .NET SDK is not installed")
@pytest.mark.skipif(not DOTNET_SRC.exists(), reason="the C# sources are not in this checkout")
def test_scaffolded_csharp_compiles_and_the_analyzer_agrees(tmp_path):
    """The cross-language round trip, proved by a compiler rather than by string matching.

    The hash literals in the emitted file were computed by Python from Python annotations.
    The Roslyn analyzer recomputes them from C# symbols and fails the build on any
    disagreement, so a green build here is both languages agreeing on the same canonical
    payload — through a file no human wrote.
    """
    emitted = to_csharp.convertTypes(
        [ScaffoldColour, CompileProofLeaf, CompileProofWide, CompileProofBase, CompileProofDerived],
        moduleName="compile_proof",
        namespace="Versionable.Scaffold.Proof",
    )
    assert emitted.todos == [], f"the compile fixture must scaffold cleanly, got: {emitted.todos}"
    assert '[Versionable(Version = 2, Hash = "0db231", Register = false)]' in emitted.text
    # The one modifier a string assertion cannot prove: without it this build fails CS0108.
    assert "public new static class Migrate" in emitted.text

    (tmp_path / "Proof.csproj").write_text(
        _SCRATCH_CSPROJ.format(
            versionable=DOTNET_SRC / "Versionable" / "Versionable.csproj",
            analyzers=DOTNET_SRC / "Versionable.Analyzers" / "Versionable.Analyzers.csproj",
        ),
        encoding="utf-8",
    )
    source = tmp_path / emitted.fileName
    source.write_text(emitted.text, encoding="utf-8")

    result = _build(tmp_path)
    assert result.returncode == 0, f"scaffolded C# did not compile:\n{result.stdout}\n{result.stderr}"

    # And the check is not vacuous: corrupt one hash and the same build must fail on it.
    source.write_text(emitted.text.replace('Hash = "0db231"', 'Hash = "000000"'), encoding="utf-8")
    tampered = _build(tmp_path)
    assert tampered.returncode != 0
    assert "VSN0001" in tampered.stdout
    assert "0db231" in tampered.stdout, "the analyzer should report the hash the scaffolder emitted"


if __name__ == "__main__":  # pragma: no cover — convenience for computing fixture hashes
    for name, value in sorted(vars(sys.modules[__name__]).items()):
        if isinstance(value, type) and issubclass(value, Versionable) and value is not Versionable:
            print(f"{name}: {value.hash()}")
