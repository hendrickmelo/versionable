using Microsoft.CodeAnalysis;
using Xunit;

namespace Versionable.Tests;

/// <summary>
/// Asserts on what the source generator emits, and that what it emits compiles.
/// </summary>
/// <remarks>
/// The compile check is the load-bearing one: generated code is never reviewed, so the only
/// thing standing between a bad emitter and a broken consumer build is a compilation of its
/// output.
/// </remarks>
public class GeneratorEmissionTests
{
    [Fact]
    public void generated_code_compiles_and_implements_the_metadata_provider()
    {
        (Compilation compilation, IReadOnlyList<(string HintName, string Text)> generated) =
            GrammarTestHarness.Generate(Fixture());

        Assert.Empty(compilation.GetDiagnostics().Where(diagnostic => diagnostic.Severity == DiagnosticSeverity.Error));

        string emitted = Source(generated, "Fixtures.Config");
        Assert.Contains("partial class Config : global::Versionable.IVersionableMetadataProvider", emitted, StringComparison.Ordinal);
        Assert.Contains("public static global::Versionable.VersionableMetadata VersionableMetadata", emitted, StringComparison.Ordinal);
        Assert.Contains("""Name = "Config",""", emitted, StringComparison.Ordinal);
        Assert.Contains("""Hash = "5e2abc",""", emitted, StringComparison.Ordinal);
        Assert.Contains("""OldNames = new string[] { "OldConfig" },""", emitted, StringComparison.Ordinal);
    }

    [Fact]
    public void registration_runs_from_a_module_initializer()
    {
        string emitted = Source(GrammarTestHarness.Generate(Fixture()).Generated, "Fixtures.Config");

        Assert.Contains("[global::System.Runtime.CompilerServices.ModuleInitializer]", emitted, StringComparison.Ordinal);
        Assert.Contains("global::Versionable.VersionableRegistry.Register(", emitted, StringComparison.Ordinal);
        Assert.DoesNotContain("RegisterTypeOnly", emitted, StringComparison.Ordinal);
    }

    [Fact]
    public void register_false_claims_no_serialization_name()
    {
        // The documented seam: metadata is still generated and still reachable by CLR type,
        // so the type stays serializable as a nested value; it just never resolves from an
        // envelope name.
        string emitted = Source(
            GrammarTestHarness.Generate($$"""
                {{GrammarTestHarness.Preamble}}

                [Versionable(Version = 1, Hash = "6d52fd", Register = false)]
                public sealed partial class Hidden
                {
                    public int Value { get; init; }
                }
                """).Generated,
            "Hidden");

        Assert.Contains("global::Versionable.VersionableRegistry.RegisterTypeOnly(", emitted, StringComparison.Ordinal);
    }

    [Fact]
    public void a_non_partial_type_gets_no_generated_source()
    {
        // Nothing can be emitted for it, and the analyzer already says why (VSN0005). Emitting
        // a broken partial would bury that message under compiler errors.
        IReadOnlyList<(string HintName, string Text)> generated = GrammarTestHarness.Generate($$"""
            {{GrammarTestHarness.Preamble}}

            [Versionable(Version = 1, Hash = "6d52fd")]
            public sealed class Fixture
            {
                public int Value { get; init; }
            }
            """).Generated;

        Assert.False(Emitted(generated, "Fixture"));
    }

    [Fact]
    public void a_settable_member_gets_a_setter_and_an_init_only_member_does_not()
    {
        string emitted = Source(
            GrammarTestHarness.Generate($$"""
                {{GrammarTestHarness.Preamble}}

                [Versionable(Version = 1, Hash = "527d04")]
                public sealed partial class Fixture
                {
                    public int Amount { get; init; }

                    public double Ratio { get; set; }
                }
                """).Generated,
            "Fixture");

        Assert.Contains("Setter = static (instance, value) => ((global::Fixture)instance).Ratio", emitted, StringComparison.Ordinal);
        Assert.DoesNotContain(".Amount = (int)value!", emitted, StringComparison.Ordinal);
    }

    [Fact]
    public void a_literal_member_carries_its_options_in_declaration_order()
    {
        string emitted = Source(
            GrammarTestHarness.Generate($$"""
                {{GrammarTestHarness.Preamble}}

                [Versionable(Version = 1, Hash = "d5c581")]
                public sealed partial class Fixture
                {
                    [LiteralValues("fast", "slow", Fallback = "fast")]
                    public string Mode { get; init; } = "fast";
                }
                """).Generated,
            "Fixture");

        Assert.Contains("""CanonicalType = "Literal['fast', 'slow']",""", emitted, StringComparison.Ordinal);
        Assert.Contains("""LiteralOptions = new object?[] { "fast", "slow" },""", emitted, StringComparison.Ordinal);
        Assert.Contains("HasLiteralFallback = true,", emitted, StringComparison.Ordinal);
        Assert.Contains("""LiteralFallback = "fast",""", emitted, StringComparison.Ordinal);
        Assert.Contains("""DefaultFactory = static () => (string)("fast"),""", emitted, StringComparison.Ordinal);
    }

    [Theory]
    // Every shape WireValues.Read refuses, because building it means Activator.CreateInstance
    // over a type the trimmer cannot see (ADR-0003). Element handling always goes back to the
    // engine; only the construction is generated.
    [InlineData("public List<int> Field { get; init; } = new();", "Enumerable.ToList(")]
    [InlineData("public int[] Field { get; init; } = [];", "Enumerable.ToArray(")]
    [InlineData("public HashSet<string> Field { get; init; } = new();", "HashSet<string>(")]
    [InlineData("public FrozenSet<int> Field { get; init; } = null!;", "FrozenSet.ToFrozenSet(")]
    [InlineData(
        "public System.Collections.Immutable.ImmutableHashSet<int> Field { get; init; } = null!;",
        "ImmutableHashSet.CreateRange(")]
    [InlineData("public Dictionary<string, int> Field { get; init; } = new();", "Enumerable.ToDictionary(")]
    [InlineData("public Dictionary<Status, int> Field { get; init; } = new();", "typeof(global::Status)")]
    [InlineData("public (int, string) Field { get; init; }", "AsList(wire) switch { var items0 =>")]
    [InlineData("public ValueTuple<int> Field { get; init; }", "new global::System.ValueTuple<int>(")]
    [InlineData("public List<Status> Field { get; init; } = new();", "typeof(global::Status)")]
    // A nullable container: null must survive rather than reach AsList. Both branches carry an
    // explicit cast, which is what makes the same shape legal in a nested position too.
    [InlineData(
        "public List<int>? Field { get; init; }",
        "wire is null ? (global::System.Collections.Generic.List<int>?)null :")]
    // Nullable VALUE-type containers nested one level down: no target type exists inside a
    // Select projection, a ToDictionary selector, or a tuple element, so an untyped null branch
    // emitted code that failed in the consumer's build (CS0411 / CS1660 / CS8135+CS1662).
    [InlineData("public List<(int, string)?> Field { get; init; } = new();", "item0 is null ? ((int, string)?)null")]
    [InlineData(
        "public Dictionary<string, (int, int)?> Field { get; init; } = new();",
        "entry0.Value is null ? ((int, int)?)null")]
    [InlineData("public (int, (int, string)?) Field { get; init; }", "items0[1] is null ? ((int, string)?)null")]
    [InlineData("public List<List<int>?> Field { get; init; } = new();",
        "item0 is null ? (global::System.Collections.Generic.List<int>?)null")]
    // Nesting recurses into one expression rather than bottoming out.
    [InlineData("public List<List<int>> Field { get; init; } = new();", "AsList(item0)")]
    [InlineData(
        "public Dictionary<string, List<int>> Field { get; init; } = new();",
        "AsList(entry0.Value)")]
    public void every_shape_the_engine_cannot_build_gets_a_reader(string member, string expected)
    {
        (Compilation compilation, IReadOnlyList<(string HintName, string Text)> generated) =
            GrammarTestHarness.Generate($$"""
                {{GrammarTestHarness.Preamble}}

                [Versionable(Version = 1)]
                public sealed partial class Fixture
                {
                    {{member}}
                }
                """);

        // Asserted per row, not once: generated code is never read by a human, so "it compiles"
        // is the only check that covers the shapes nobody thought to write an assertion for.
        // Its absence is exactly how the untyped null branch reached a review.
        Assert.Empty(compilation.GetDiagnostics().Where(diagnostic => diagnostic.Severity == DiagnosticSeverity.Error));

        string emitted = Source(generated, "Fixture");
        Assert.Contains("WireReader = static wire =>", emitted, StringComparison.Ordinal);
        Assert.Contains(expected, emitted, StringComparison.Ordinal);
    }

    [Fact]
    public void a_null_inside_a_value_type_container_is_rejected_with_a_readable_message()
    {
        // WireValues.Read returns null for a null wire value whatever the declared type, so an
        // unboxing cast would turn a bad file into a bare NullReferenceException naming nothing.
        string emitted = Source(
            GrammarTestHarness.Generate($$"""
                {{GrammarTestHarness.Preamble}}

                [Versionable(Version = 1)]
                public sealed partial class Fixture
                {
                    public List<int> Counts { get; init; } = new();

                    public List<string> Names { get; init; } = new();
                }
                """).Generated,
            "Fixture");

        Assert.Contains(
            "?? throw new global::Versionable.Errors.ConverterException(\"Expected a value of type 'int'",
            emitted,
            StringComparison.Ordinal);

        // A reference element passes null through, as Python's deserialize does; refusing it
        // here alone would fail loads that Python round-trips.
        Assert.Contains(
            "(string)global::Versionable.Engine.WireValues.Read(item0, typeof(string))!",
            emitted,
            StringComparison.Ordinal);
    }

    [Fact]
    public void a_nested_versionable_element_skips_the_registry_entirely()
    {
        // The reflection-free path: the element's own generated metadata is named directly, so
        // materializing it needs no VersionableRegistry lookup at all.
        string emitted = Source(
            GrammarTestHarness.Generate($$"""
                {{GrammarTestHarness.Preamble}}

                [Versionable(Version = 1)]
                public sealed partial class Fixture
                {
                    public List<NestedLeaf> Leaves { get; init; } = new();
                }
                """).Generated,
            "Fixture");

        Assert.Contains(
            "WireValues.ReadVersionable(item0, global::NestedLeaf.VersionableMetadata)",
            emitted,
            StringComparison.Ordinal);
    }

    [Fact]
    public void a_type_the_engine_can_read_on_its_own_gets_no_reader()
    {
        // Scalars, converter types, enums, and nested objects all resolve from the declared
        // type alone; emitting a reader for them would be duplication, not a seam.
        string emitted = Source(
            GrammarTestHarness.Generate($$"""
                {{GrammarTestHarness.Preamble}}

                [Versionable(Version = 1)]
                public sealed partial class Fixture
                {
                    public int Count { get; init; }

                    public byte[] Blob { get; init; } = [];

                    public Tensor<double>? Samples { get; init; }

                    public Status State { get; init; }

                    public NestedLeaf? Leaf { get; init; }
                }
                """).Generated,
            "Fixture");

        Assert.DoesNotContain("WireReader", emitted, StringComparison.Ordinal);
    }

    [Fact]
    public void no_wire_writer_is_ever_emitted()
    {
        // Lowering constructs nothing — the engine walks any collection on its runtime type —
        // and enum wire mapping belongs to EnumConverter, which the write path routes through.
        string emitted = Source(
            GrammarTestHarness.Generate($$"""
                {{GrammarTestHarness.Preamble}}

                [Versionable(Version = 1)]
                public sealed partial class Fixture
                {
                    public Dictionary<Status, List<NestedLeaf>> Graph { get; init; } = new();
                }
                """).Generated,
            "Fixture");

        Assert.DoesNotContain("WireWriter", emitted, StringComparison.Ordinal);
    }

    [Fact]
    public void an_abstract_versionable_base_gets_metadata_but_no_constructor()
    {
        // The polymorphic base case: a load resolves the concrete type from the envelope, so
        // the base is never constructed and `new` on it would not even compile.
        (Compilation compilation, IReadOnlyList<(string HintName, string Text)> generated) =
            GrammarTestHarness.Generate($$"""
                {{GrammarTestHarness.Preamble}}

                [Versionable(Version = 1, Hash = "e3b0c4")]
                public abstract partial class Fixture
                {
                }
                """);

        Assert.Empty(compilation.GetDiagnostics().Where(diagnostic => diagnostic.Severity == DiagnosticSeverity.Error));

        string emitted = Source(generated, "Fixture");
        Assert.Contains("Factory = static values => throw new", emitted, StringComparison.Ordinal);
        Assert.Contains("is abstract and is never constructed directly", emitted, StringComparison.Ordinal);
    }

    [Fact]
    public void an_empty_collection_initializer_becomes_a_default_factory()
    {
        // The static lambda only excludes initializers that close over state; `new()` and `[]`
        // close over nothing, and rebuilding per call is why the contract makes this a delegate.
        string emitted = Source(
            GrammarTestHarness.Generate($$"""
                {{GrammarTestHarness.Preamble}}

                [Versionable(Version = 1)]
                public sealed partial class Fixture
                {
                    public List<int> FromNew { get; init; } = new();

                    public List<string> FromCollection { get; init; } = [];

                    public int[] FromArray { get; init; } = [];

                    public Dictionary<string, int> FromExplicit { get; init; } = new Dictionary<string, int>();
                }
                """).Generated,
            "Fixture");

        Assert.Contains(
            "DefaultFactory = static () => (global::System.Collections.Generic.List<int>)"
                + "(new global::System.Collections.Generic.List<int>()),",
            emitted,
            StringComparison.Ordinal);
        Assert.Contains("global::System.Array.Empty<int>()", emitted, StringComparison.Ordinal);
        Assert.Equal(4, emitted.Split("DefaultFactory").Length - 1);
    }

    [Fact]
    public void a_generic_versionable_type_is_not_generated()
    {
        IReadOnlyList<(string HintName, string Text)> generated = GrammarTestHarness.Generate($$"""
            {{GrammarTestHarness.Preamble}}

            [Versionable(Version = 1, Hash = "6d52fd")]
            public sealed partial class Fixture<T>
            {
                public int Value { get; init; }
            }
            """).Generated;

        Assert.False(Emitted(generated, "Fixture"));
    }

    [Fact]
    public void a_nested_type_is_generated_inside_its_enclosing_partial_type()
    {
        (Compilation compilation, IReadOnlyList<(string HintName, string Text)> generated) =
            GrammarTestHarness.Generate($$"""
                {{GrammarTestHarness.Preamble}}

                public partial class Holder
                {
                    [Versionable(Version = 1, Hash = "6d52fd")]
                    public sealed partial class Fixture
                    {
                        public int Value { get; init; }
                    }
                }
                """);

        Assert.Empty(compilation.GetDiagnostics().Where(diagnostic => diagnostic.Severity == DiagnosticSeverity.Error));
        Assert.Contains("public partial class Holder", Source(generated, "Holder.Fixture"), StringComparison.Ordinal);
    }

    [Fact]
    public void a_versionable_type_deriving_from_another_hides_the_base_members_deliberately()
    {
        // The polymorphic declaration GRAMMAR §9 blesses. Each type needs its own metadata, so
        // the derived members hide the base's — which is CS0108, an error under the
        // TreatWarningsAsErrors this repo and its consumers use, unless `new` says it was meant.
        (Compilation compilation, IReadOnlyList<(string HintName, string Text)> generated) =
            GrammarTestHarness.Generate($$"""
                {{GrammarTestHarness.Preamble}}

                [Versionable(Version = 1, Hash = "6186d4")]
                public partial class Shape
                {
                    public string Label { get; init; } = "";
                }

                [Versionable(Version = 1, Hash = "04fc60")]
                public sealed partial class Circle : Shape
                {
                    public double Radius { get; init; }
                }
                """);

        Assert.Empty(compilation.GetDiagnostics().Where(diagnostic => diagnostic.Severity == DiagnosticSeverity.Error));

        string baseSource = Source(generated, "Shape");
        string derived = Source(generated, "Circle");

        Assert.Contains(
            "public static global::Versionable.VersionableMetadata VersionableMetadata",
            baseSource,
            StringComparison.Ordinal);
        Assert.Contains("internal static class VersionableRegistration", baseSource, StringComparison.Ordinal);

        Assert.Contains(
            "public static new global::Versionable.VersionableMetadata VersionableMetadata",
            derived,
            StringComparison.Ordinal);
        Assert.Contains("internal static new class VersionableRegistration", derived, StringComparison.Ordinal);
    }

    [Fact]
    public void a_nested_migrate_that_is_a_chain_becomes_the_metadata_migrations()
    {
        (Compilation compilation, IReadOnlyList<(string HintName, string Text)> generated) =
            GrammarTestHarness.Generate($$"""
                {{GrammarTestHarness.Preamble}}

                [Versionable(Version = 2, Hash = "fa9c8c")]
                public sealed partial class Job
                {
                    public string Name { get; init; } = "";

                    internal sealed class Migrate : Versionable.Migrations.IMigrationChain
                    {
                        public IReadOnlyList<int> FromVersions { get; } = new[] { 1 };

                        public int? MinReversibleVersion => null;
                    }
                }
                """);

        Assert.Empty(compilation.GetDiagnostics().Where(diagnostic => diagnostic.Severity == DiagnosticSeverity.Error));
        Assert.Contains("Migrations = new global::Job.Migrate(),", Source(generated, "Job"), StringComparison.Ordinal);
    }

    [Fact]
    public void a_declarative_migrate_class_gets_no_migrations_member_yet()
    {
        // The form IMigrationChain documents — static V1/V2 members — needs the phase-4
        // `Migration` builder to compose a chain from. Until then the generator must leave
        // Migrations null rather than guess, and a static class can implement nothing anyway.
        string emitted = Source(
            GrammarTestHarness.Generate($$"""
                {{GrammarTestHarness.Preamble}}

                [Versionable(Version = 2, Hash = "fa9c8c")]
                public sealed partial class Job
                {
                    public string Name { get; init; } = "";

                    public static class Migrate
                    {
                        public static readonly string V1 = "rename";
                    }
                }
                """).Generated,
            "Job");

        Assert.DoesNotContain("Migrations =", emitted, StringComparison.Ordinal);
    }

    private static string Fixture() => $$"""
        {{GrammarTestHarness.Preamble}}

        namespace Fixtures
        {
            [Versionable(Version = 2, Hash = "5e2abc", OldNames = new[] { "OldConfig" })]
            [SerializationName("Config")]
            public sealed partial class Config
            {
                public string Name { get; init; } = "";

                public int Retries { get; set; }

                [VersionableField("tag_list")]
                public List<string> Tags { get; init; } = new();
            }
        }
        """;

    /// <summary>
    /// The source generated for one fixture type. Named rather than assumed-single: the shared
    /// preamble declares [Versionable] types of its own, so every run produces several files.
    /// </summary>
    private static string Source(IReadOnlyList<(string HintName, string Text)> generated, string typeName) =>
        generated.Single(source => source.HintName == typeName + ".Versionable.g.cs").Text;

    private static bool Emitted(IReadOnlyList<(string HintName, string Text)> generated, string typeName) =>
        generated.Any(source => source.HintName == typeName + ".Versionable.g.cs");
}
