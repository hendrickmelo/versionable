using Versionable.Analyzers.Grammar;
using Xunit;

namespace Versionable.Tests;

/// <summary>
/// Rendering cases the conformance vectors do not reach, because they are about the C# side
/// of the mapping rather than about a canonical string Python also produces.
/// </summary>
public class GrammarRenderingTests
{
    [Theory]
    // Width erasure, in both directions (GRAMMAR §4).
    [InlineData("public sbyte x;", "int")]
    [InlineData("public ulong x;", "int")]
    [InlineData("public nint x;", "int")]
    [InlineData("public nuint x;", "int")]
    [InlineData("public float x;", "float")]
    // Half erases like every other binary float (§4); only an array dtype keeps its width, and
    // Tensor<Half> renders ndarray[float16] in the vector suite.
    [InlineData("public Half x;", "float")]
    // char is a one-character string on the wire; Python has no character type.
    [InlineData("public char x;", "str")]
    [InlineData("public char? x;", "Union[None, str]")]
    [InlineData("public char[] x;", "list[str]")]
    // decimal is a converter type, not a numeric width.
    [InlineData("public decimal x;", "Decimal")]
    // byte[] is the carve-out; every other array is a list.
    [InlineData("public byte[] x;", "bytes")]
    [InlineData("public sbyte[] x;", "list[int]")]
    [InlineData("public byte?[] x;", "list[Union[None, int]]")]
    [InlineData("public byte[]? x;", "Union[None, bytes]")]
    [InlineData("public int[][] x;", "list[list[int]]")]
    // Nullable value types and annotated reference types are the same construct (§6).
    [InlineData("public string? x;", "Union[None, str]")]
    [InlineData("public int? x;", "Union[None, int]")]
    [InlineData("public Node? x;", "Union[Node, None]")]
    // Both immutable set spellings land on the same token.
    [InlineData("public System.Collections.Immutable.ImmutableHashSet<int> x;", "frozenset[int]")]
    [InlineData("public FrozenSet<int> x;", "frozenset[int]")]
    // Type parameters outside the closed container set are dropped (§9).
    [InlineData("public MyBox<int> x;", "MyBox")]
    [InlineData("public Regex x;", "Pattern")]
    [InlineData("public (int, string) x;", "tuple[int, str]")]
    [InlineData("public (int?, string?) x;", "tuple[Union[None, int], Union[None, str]]")]
    [InlineData("public DateTimeOffset x;", "datetime")]
    public void a_declared_type_renders_to_its_canonical_string(string declaration, string canonical) =>
        Assert.Equal(canonical, GrammarTestHarness.Render(declaration));

    [Fact]
    public void a_nested_type_renders_only_its_own_name()
    {
        // Serialization Names carry no namespace, assembly, or enclosing-type qualification
        // (GRAMMAR §9), which is what makes a hash survive a file move.
        SchemaModel schema = GrammarTestHarness.Schema("public Outer.Inner x;");

        Assert.Equal("Inner", schema.Fields[0].CanonicalType);
    }

    [Fact]
    public void an_explicit_serialization_name_beats_the_type_name() =>
        Assert.Equal("Renamed", GrammarTestHarness.Render("public NeedsRenaming x;"));

    [Fact]
    public void a_wire_name_override_changes_the_payload_and_the_hash()
    {
        SchemaModel plain = GrammarTestHarness.Schema("public int Amount; public double Ratio;");
        SchemaModel renamed = GrammarTestHarness.Schema(
            """[VersionableField("amount")] public int Amount; public double Ratio;""");

        Assert.Equal("Amount:int,Ratio:float", plain.Payload);
        Assert.Equal("527d04", plain.ComputedHash);
        Assert.Equal("Ratio:float,amount:int", renamed.Payload);
        Assert.NotEqual(plain.ComputedHash, renamed.ComputedHash);
    }

    [Fact]
    public void a_computed_property_is_not_a_field()
    {
        // A get-only property with a body is a derived value, not state. Including it would
        // change the hash and leave the generated factory with nothing to write it through.
        SchemaModel schema = GrammarTestHarness.Schema("""
            public List<int> Items { get; init; } = new();
            public string Name { get; init; } = "";
            public int Count => Items.Count;
            public string Label { get { return Name; } }
            """);

        Assert.Equal(new[] { "Items", "Name" }, schema.Fields.Select(field => field.ClrName));
        Assert.Equal("Items:list[int],Name:str", schema.Payload);
        Assert.Equal("fb5474", schema.ComputedHash);
    }

    [Fact]
    public void a_record_hashes_its_positional_members_in_declaration_order()
    {
        SchemaModel schema = GrammarTestHarness.Schema(
            GrammarTestHarness.Compile($$"""
                {{GrammarTestHarness.Preamble}}

                [Versionable(Version = 1, Hash = "527d04")]
                public sealed partial record Fixture(int Amount, double Ratio);
                """),
            "Fixture");

        Assert.Equal(new[] { "Amount", "Ratio" }, schema.Fields.Select(field => field.ClrName));
        Assert.Equal("527d04", schema.ComputedHash);
        Assert.Empty(schema.Problems);
        Assert.NotNull(schema.Factory);
    }

    [Fact]
    public void inherited_members_are_fields_of_the_derived_schema()
    {
        // Python resolves dataclass fields across the MRO; the C# equivalent is to walk the
        // base chain, base declarations first, so declaration order matches the factory's.
        SchemaModel schema = GrammarTestHarness.Schema(
            GrammarTestHarness.Compile($$"""
                {{GrammarTestHarness.Preamble}}

                public abstract class Shape
                {
                    public int Shared { get; init; }
                }

                [Versionable(Version = 1, Hash = "424e5f")]
                public sealed partial class Fixture : Shape
                {
                    public string Own { get; init; } = "";
                }
                """),
            "Fixture");

        Assert.Equal(new[] { "Shared", "Own" }, schema.Fields.Select(field => field.ClrName));
        Assert.Equal("Own:str,Shared:int", schema.Payload);
        Assert.Equal("424e5f", schema.ComputedHash);
        Assert.Empty(schema.Problems);
    }

    [Fact]
    public void a_base_members_private_setter_leaves_the_type_unbuildable()
    {
        // The member is still state, so it still hashes — but a private setter on a base type
        // is invisible from the derived type's generated partial, so nothing can write it and
        // the type reports VSN0010 rather than generating code that would not compile.
        SchemaModel schema = GrammarTestHarness.Schema(
            GrammarTestHarness.Compile($$"""
                {{GrammarTestHarness.Preamble}}

                public abstract class Shape
                {
                    public int Hidden { get; private set; }
                }

                [Versionable(Version = 1, Hash = "aaaaaa")]
                public sealed partial class Fixture : Shape
                {
                    public string Own { get; init; } = "";
                }
                """),
            "Fixture");

        Assert.Equal("Hidden:int,Own:str", schema.Payload);
        Assert.False(schema.Fields[0].IsInitializable);
        Assert.False(schema.Fields[0].IsAssignable);
        Assert.Null(schema.Factory);
        Assert.Contains(schema.Problems, problem => problem.Id == "VSN0010");
    }

    [Fact]
    public void only_a_literal_initializer_becomes_a_default()
    {
        // The generated DefaultFactory is a static lambda, so an initializer that closes over
        // instance state or a primary-constructor parameter cannot be re-emitted into it.
        SchemaModel schema = GrammarTestHarness.Schema("""
            public int Count { get; init; } = -7;
            public string Label { get; init; } = string.Empty;
            """);

        Assert.Equal("Count:int,Label:str", schema.Payload);
        Assert.Equal("84f4bb", schema.ComputedHash);
        Assert.Equal("-7", schema.Fields[0].DefaultExpression);
        Assert.Null(schema.Fields[1].DefaultExpression);
    }

    [Theory]
    // A hash the runtime could never honor is worse than a build error: §9's bare-name
    // fall-through stops short of types that can never *be* anything.
    [InlineData("public object x;", "'object' names no schema")]
    [InlineData("public IReadOnlyList<int> x;", "cannot construct")]
    [InlineData("public List<object> x;", "'object' names no schema")]
    [InlineData("public AbstractThing x;", "not [Versionable]")]
    [InlineData("public Action x;", "no wire form")]
    [InlineData("public dynamic x;", "'dynamic' has no canonical form")]
    public void a_type_that_names_nothing_buildable_is_rejected(string declaration, string reason)
    {
        SchemaModel schema = GrammarTestHarness.Schema("    " + declaration);

        SchemaProblem problem = Assert.Single(
            schema.Problems,
            candidate => candidate.Id == "VSN0002");
        Assert.Contains(
            reason,
            string.Join(" ", problem.MessageArguments.Select(argument => argument?.ToString())),
            StringComparison.Ordinal);
    }

    [Fact]
    public void an_abstract_versionable_base_is_a_legitimate_field_type()
    {
        // The polymorphic case: the envelope names the concrete type, so an abstract base is
        // exactly what a polymorphic field should declare.
        SchemaModel schema = GrammarTestHarness.Schema("public AbstractShape x;");

        Assert.Equal("AbstractShape", schema.Fields[0].CanonicalType);
        Assert.Empty(schema.Problems.Where(problem => problem.Id == "VSN0002"));
    }

    [Fact]
    public void an_ignored_member_leaves_the_payload_and_the_descriptors_together()
    {
        // Python needs no counterpart: an unannotated attribute is simply not a dataclass
        // field. C# cannot tell a persistable member from an incidental one by shape, so the
        // opt-out is explicit — and removing a field changes the hash, as it must.
        SchemaModel kept = GrammarTestHarness.Schema("""
            public int Amount { get; init; }
            public double Ratio { get; init; }
            """);
        SchemaModel ignored = GrammarTestHarness.Schema("""
            public int Amount { get; init; }
            public double Ratio { get; init; }

            [VersionableIgnore]
            public string Scratch { get; set; } = "";
            """);

        Assert.Equal(new[] { "Amount", "Ratio" }, ignored.Fields.Select(field => field.ClrName));
        Assert.Equal(kept.Payload, ignored.Payload);
        Assert.Equal("527d04", ignored.ComputedHash);
    }

    [Fact]
    public void an_ignored_member_of_an_otherwise_unrenderable_type_is_not_rejected()
    {
        // The point of the opt-out: a handle or a cache has no wire form, and saying so should
        // silence the grammar rather than fail the build.
        SchemaModel schema = GrammarTestHarness.Schema("""
            public int Amount { get; init; }

            [VersionableIgnore]
            public Action? OnChanged { get; set; }
            """);

        Assert.Empty(schema.Problems.Where(problem => problem.Id == "VSN0002"));
        Assert.Equal("Amount:int", schema.Payload);
    }

    [Fact]
    public void an_empty_schema_hashes_the_empty_payload()
    {
        SchemaModel schema = GrammarTestHarness.Schema(string.Empty);

        Assert.Equal(string.Empty, schema.Payload);
        Assert.Equal("e3b0c4", schema.ComputedHash);
    }

    [Fact]
    public void field_names_sort_ordinally_and_by_name_not_by_pair()
    {
        // Two traps in one payload: 'Mid' precedes 'alpha' only under ordinal comparison, and
        // 'a' precedes 'a1' only when the sort key is the name rather than the assembled pair.
        SchemaModel schema = GrammarTestHarness.Schema(
            "public int zeta; public string a1; public string alpha; public bool Mid; public string a;");

        Assert.Equal("Mid:bool,a:str,a1:str,alpha:str,zeta:int", schema.Payload);
    }
}
