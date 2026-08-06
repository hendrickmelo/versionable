using System.Collections.Immutable;
using Microsoft.CodeAnalysis;
using Xunit;

namespace Versionable.Tests;

/// <summary>
/// Runs <c>SchemaHashAnalyzer</c> over fixture compilations and asserts on what it reports.
/// </summary>
/// <remarks>
/// Python's equivalent is the import-time <c>HashMismatchError</c> raised from
/// <c>Versionable.__init_subclass__</c>; per ADR-0003 the C# form is a build error, so these
/// are the parity tests for that tripwire plus the checks that only a compiler can make.
/// </remarks>
public class AnalyzerDiagnosticTests
{
    [Fact]
    public void a_correct_hash_is_silent()
    {
        ImmutableArray<Diagnostic> diagnostics = GrammarTestHarness.Analyze(Fixture(
            """[Versionable(Version = 1, Hash = "527d04")]""",
            "public int Amount { get; init; } public double Ratio { get; init; }"));

        Assert.Empty(diagnostics);
    }

    [Fact]
    public void a_wrong_hash_reports_the_payload_and_the_computed_hash()
    {
        // ADR-0003: the payload is the actionable half of the message. "expected X, got Y"
        // alone does not say which field drifted.
        Diagnostic diagnostic = Assert.Single(GrammarTestHarness.Analyze(Fixture(
            """[Versionable(Version = 1, Hash = "deadbe")]""",
            "public int Amount { get; init; } public double Ratio { get; init; }")));

        Assert.Equal("VSN0001", diagnostic.Id);
        Assert.Equal(DiagnosticSeverity.Error, diagnostic.DefaultSeverity);

        string message = diagnostic.GetMessage();
        Assert.Contains("Amount:int,Ratio:float", message, StringComparison.Ordinal);
        Assert.Contains("527d04", message, StringComparison.Ordinal);
        Assert.Contains("deadbe", message, StringComparison.Ordinal);
    }

    [Fact]
    public void a_missing_hash_is_reported_with_the_hash_to_paste_in()
    {
        Diagnostic diagnostic = Assert.Single(GrammarTestHarness.Analyze(Fixture(
            "[Versionable(Version = 1)]",
            "public int Amount { get; init; } public double Ratio { get; init; }")));

        Assert.Equal("VSN0001", diagnostic.Id);
        Assert.Contains("527d04", diagnostic.GetMessage(), StringComparison.Ordinal);
    }

    [Fact]
    public void a_non_partial_versionable_type_is_reported()
    {
        // The generator implements IVersionableMetadataProvider's static abstract member from
        // a second part of the type, which a non-partial type cannot have.
        string source = $$"""
            {{GrammarTestHarness.Preamble}}

            [Versionable(Version = 1, Hash = "6d52fd")]
            public sealed class Fixture
            {
                public int Value { get; init; }
            }
            """;

        Diagnostic diagnostic = Assert.Single(GrammarTestHarness.Analyze(source));

        Assert.Equal("VSN0005", diagnostic.Id);
        Assert.Contains("partial", diagnostic.GetMessage(), StringComparison.Ordinal);
    }

    [Fact]
    public void a_partial_type_inside_a_non_partial_type_is_reported()
    {
        string source = $$"""
            {{GrammarTestHarness.Preamble}}

            public static class Holder
            {
                [Versionable(Version = 1, Hash = "6d52fd")]
                public sealed partial class Fixture
                {
                    public int Value { get; init; }
                }
            }
            """;

        Diagnostic diagnostic = Assert.Single(GrammarTestHarness.Analyze(source));

        Assert.Equal("VSN0005", diagnostic.Id);
        Assert.Contains("enclosing type", diagnostic.GetMessage(), StringComparison.Ordinal);
    }

    [Fact]
    public void two_types_claiming_one_serialization_name_are_both_reported()
    {
        // GRAMMAR §9: dropping namespaces flattens every type into one space, so a collision
        // means two different schemas hash identically.
        string source = $$"""
            {{GrammarTestHarness.Preamble}}

            namespace First
            {
                [Versionable(Version = 1, Hash = "6d52fd")]
                public sealed partial class Config
                {
                    public int Value { get; init; }
                }
            }

            namespace Second
            {
                [Versionable(Version = 1, Hash = "6d52fd")]
                public sealed partial class Config
                {
                    public int Value { get; init; }
                }
            }
            """;

        ImmutableArray<Diagnostic> diagnostics = GrammarTestHarness.Analyze(source);

        Assert.Equal(2, diagnostics.Length);
        Assert.All(diagnostics, diagnostic => Assert.Equal("VSN0004", diagnostic.Id));
        Assert.All(diagnostics, diagnostic =>
        {
            Assert.Contains("First.Config", diagnostic.GetMessage(), StringComparison.Ordinal);
            Assert.Contains("Second.Config", diagnostic.GetMessage(), StringComparison.Ordinal);
            Assert.Contains("SerializationName", diagnostic.GetMessage(), StringComparison.Ordinal);
        });
    }

    [Fact]
    public void a_collision_between_a_field_type_and_a_versionable_type_is_reported()
    {
        // The claim comes from a referenced type, not from a [Versionable] declaration: the
        // uniqueness rule covers every type reachable from a schema.
        string source = $$"""
            {{GrammarTestHarness.Preamble}}

            namespace Other
            {
                public sealed class Node { }
            }

            namespace Root
            {
                [Versionable(Version = 1, Hash = "aaaaaa")]
                public sealed partial class Fixture
                {
                    public global::Node First { get; init; } = new();

                    public global::Other.Node Second { get; init; } = new();
                }
            }
            """;

        Assert.Contains(
            GrammarTestHarness.Analyze(source),
            diagnostic => diagnostic.Id == "VSN0004"
                && diagnostic.GetMessage().Contains("Other.Node", StringComparison.Ordinal));
    }

    [Theory]
    [InlineData("[LiteralValues(1.5)] public double Value { get; init; }", "VSN0006")]
    [InlineData("[LiteralValues(Status.Active)] public Status Value { get; init; }", "VSN0006")]
    [InlineData("public Tensor<decimal> Value { get; init; } = null!;", "VSN0002")]
    [InlineData("public int[,] Value { get; init; } = null!;", "VSN0002")]
    [InlineData("""[VersionableField("a:b")] public int Value { get; init; }""", "VSN0007")]
    [InlineData("""[VersionableField("a,b")] public int Value { get; init; }""", "VSN0007")]
    [InlineData("""[VersionableField("a[0]")] public int Value { get; init; }""", "VSN0007")]
    public void a_rejected_construct_is_reported(string member, string expectedId)
    {
        ImmutableArray<Diagnostic> diagnostics = GrammarTestHarness.Analyze(Fixture(
            "[Versionable(Version = 1)]",
            member));

        Assert.Contains(diagnostics, diagnostic => diagnostic.Id == expectedId);
    }

    [Fact]
    public void a_char_literal_option_erases_to_a_one_character_string()
    {
        // Same erasure as the `char` scalar: Literal['a'] is one schema however C# spells the
        // option, so the two spellings must not produce two hashes.
        ImmutableArray<Diagnostic> fromChars = GrammarTestHarness.Analyze(Fixture(
            """[Versionable(Version = 1, Hash = "120b85")]""",
            "[LiteralValues('a', 'b')] public char Value { get; init; }"));
        ImmutableArray<Diagnostic> fromStrings = GrammarTestHarness.Analyze(Fixture(
            """[Versionable(Version = 1, Hash = "120b85")]""",
            """[LiteralValues("a", "b")] public string Value { get; init; } = "a";"""));

        Assert.Empty(fromChars);
        Assert.Empty(fromStrings);
    }

    [Fact]
    public void a_field_type_that_names_nothing_buildable_is_reported()
    {
        ImmutableArray<Diagnostic> diagnostics = GrammarTestHarness.Analyze(Fixture(
            "[Versionable(Version = 1)]",
            "public IReadOnlyList<int> Values { get; init; } = null!;"));

        Assert.Contains(diagnostics, diagnostic => diagnostic.Id == "VSN0002");
    }

    [Fact]
    public void a_literal_property_typed_object_is_still_accepted()
    {
        // The rejection sits on the field-type render path. A [LiteralValues] property never
        // renders its declared type — the option list is the schema — so `object` stays legal
        // there, which is what a mixed string/int literal needs.
        ImmutableArray<Diagnostic> diagnostics = GrammarTestHarness.Analyze(Fixture(
            """[Versionable(Version = 1, Hash = "224f43")]""",
            """[LiteralValues("auto", 0)] public object Tag { get; init; } = "auto";"""));

        Assert.Empty(diagnostics);
    }

    [Fact]
    public void a_constructor_parameter_matching_two_members_by_case_is_reported()
    {
        // C# is case-sensitive, so a type may declare both Value and value. Folding case and
        // taking the first would silently wire the argument to the wrong member.
        ImmutableArray<Diagnostic> diagnostics = GrammarTestHarness.Analyze($$"""
            {{GrammarTestHarness.Preamble}}

            [Versionable(Version = 1, Hash = "39b73a")]
            public sealed partial class Fixture
            {
                public Fixture(int VALUE) => Value = VALUE;

                public int Value { get; }

                public int value { get; }
            }
            """);

        Diagnostic diagnostic = Assert.Single(diagnostics, candidate => candidate.Id == "VSN0010");
        Assert.Contains("case is folded", diagnostic.GetMessage(), StringComparison.Ordinal);
    }

    [Fact]
    public void a_type_that_cannot_self_register_is_warned_about()
    {
        // The CLR only accepts a module initializer reachable from module scope, so a type
        // nested in a private container gets metadata but never enters the registry.
        ImmutableArray<Diagnostic> diagnostics = GrammarTestHarness.Analyze($$"""
            {{GrammarTestHarness.Preamble}}

            public partial class Holder
            {
                private partial class Inner
                {
                    [Versionable(Version = 1, Hash = "6d52fd")]
                    public sealed partial class Fixture
                    {
                        public int Value { get; init; }
                    }
                }
            }
            """);

        Diagnostic diagnostic = Assert.Single(diagnostics, candidate => candidate.Id == "VSN0011");
        Assert.Equal(DiagnosticSeverity.Warning, diagnostic.DefaultSeverity);
        Assert.Contains("module scope", diagnostic.GetMessage(), StringComparison.Ordinal);
    }

    [Fact]
    public void an_ordinary_type_is_not_warned_about_registration()
    {
        ImmutableArray<Diagnostic> diagnostics = GrammarTestHarness.Analyze(Fixture(
            """[Versionable(Version = 1, Hash = "6d52fd")]""",
            "public int Value { get; init; }"));

        Assert.Empty(diagnostics);
    }

    [Fact]
    public void two_members_claiming_one_wire_name_are_reported()
    {
        ImmutableArray<Diagnostic> diagnostics = GrammarTestHarness.Analyze(Fixture(
            "[Versionable(Version = 1)]",
            """
            [VersionableField("value")] public int First { get; init; }
            [VersionableField("value")] public int Second { get; init; }
            """));

        Assert.Contains(diagnostics, diagnostic => diagnostic.Id == "VSN0008");
    }

    [Fact]
    public void a_type_with_no_way_to_rebuild_it_is_reported()
    {
        // A get-only auto property has no setter, so it must arrive through a constructor
        // parameter of the same name; otherwise the generated Factory could never write it.
        ImmutableArray<Diagnostic> diagnostics = GrammarTestHarness.Analyze(Fixture(
            """[Versionable(Version = 1, Hash = "6d52fd")]""",
            "public int Value { get; } = 3;"));

        Diagnostic diagnostic = Assert.Single(diagnostics);
        Assert.Equal("VSN0010", diagnostic.Id);
    }

    [Fact]
    public void a_disabled_nullable_context_is_a_warning_not_an_error()
    {
        // Without NRT the compiler reports no annotation at all, so an optional field
        // silently hashes as required. Worth a warning; not worth failing a build that may
        // predate the nullable rollout.
        ImmutableArray<Diagnostic> diagnostics = GrammarTestHarness.Analyze(
            Fixture("[Versionable(Version = 1)]", "public string Name { get; init; }"),
            Microsoft.CodeAnalysis.NullableContextOptions.Disable);

        Diagnostic diagnostic = Assert.Single(diagnostics, candidate => candidate.Id == "VSN0009");
        Assert.Equal(DiagnosticSeverity.Warning, diagnostic.DefaultSeverity);
    }

    [Fact]
    public void a_gap_above_the_oldest_migration_is_reported()
    {
        ImmutableArray<Diagnostic> diagnostics = GrammarTestHarness.Analyze(Fixture(
            """[Versionable(Version = 4, Hash = "6d52fd")]""",
            """
            public int Value { get; init; }

            public static class Migrate
            {
                public static readonly int V1 = 0;
                public static readonly int V3 = 0;
            }
            """));

        Diagnostic diagnostic = Assert.Single(diagnostics, candidate => candidate.Id == "VSN0003");
        Assert.Contains("version 2", diagnostic.GetMessage(), StringComparison.Ordinal);
    }

    [Fact]
    public void a_gap_below_the_oldest_migration_is_legitimate()
    {
        // Files older than the oldest migration are simply unsupported — that is what
        // MinReversibleVersion reports, not an error.
        ImmutableArray<Diagnostic> diagnostics = GrammarTestHarness.Analyze(Fixture(
            """[Versionable(Version = 4, Hash = "6d52fd")]""",
            """
            public int Value { get; init; }

            public static class Migrate
            {
                public static readonly int V2 = 0;
                public static readonly int V3 = 0;
            }
            """));

        Assert.Empty(diagnostics);
    }

    [Fact]
    public void an_imperative_migration_joins_the_same_chain()
    {
        ImmutableArray<Diagnostic> diagnostics = GrammarTestHarness.Analyze(Fixture(
            """[Versionable(Version = 4, Hash = "6d52fd")]""",
            """
            public int Value { get; init; }

            public static class Migrate
            {
                public static readonly int V1 = 0;

                [Migration(FromVersion = 3)]
                public static void Third() { }
            }
            """,
            extraUsings: "using Versionable.Migrations;"));

        Diagnostic diagnostic = Assert.Single(diagnostics, candidate => candidate.Id == "VSN0003");
        Assert.Contains("version 2", diagnostic.GetMessage(), StringComparison.Ordinal);
    }

    [Fact]
    public void a_type_without_the_attribute_is_not_analyzed()
    {
        ImmutableArray<Diagnostic> diagnostics = GrammarTestHarness.Analyze($$"""
            {{GrammarTestHarness.Preamble}}

            public sealed class NotVersionable
            {
                public int[,] Grid { get; init; } = null!;
            }
            """);

        Assert.Empty(diagnostics);
    }

    private static string Fixture(string attribute, string members, string extraUsings = "") => $$"""
        {{extraUsings}}
        {{GrammarTestHarness.Preamble}}

        {{attribute}}
        public sealed partial class Fixture
        {
        {{members}}
        }
        """;
}
