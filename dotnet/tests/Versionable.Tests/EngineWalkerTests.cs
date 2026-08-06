using System.Collections.Frozen;
using Versionable.Engine;
using Versionable.Errors;
using Xunit;

namespace Versionable.Tests;

/// <summary>
/// The serialization walker: values down to the wire and back, without a file in the way.
/// </summary>
[Collection(RegistryCollection.Name)]
public class EngineWalkerTests
{
    public EngineWalkerTests() => GoldenSchemas.EnsureRegistered();

    [Fact]
    public void primitives_pass_through_untouched()
    {
        Assert.Equal("text", WireValues.Write("text"));
        Assert.Equal(42, WireValues.Write(42));
        Assert.Equal(9.8125, WireValues.Write(9.8125));
        Assert.Equal(true, WireValues.Write(true));
        Assert.Null(WireValues.Write(null));
    }

    [Fact]
    public void a_nested_object_is_written_with_its_own_envelope()
    {
        EngineLeaf leaf = new("tip", 1.5);

        IReadOnlyDictionary<string, object?> wire = WireValues.AsMap(WireValues.Write(leaf));

        EnvelopeMetadata envelope = EnvelopeCodec.Read(wire);
        Assert.Equal("EngineLeaf", envelope.ObjectName);
        Assert.Equal(1, envelope.Version);
        Assert.Equal("aaaaaa", envelope.Hash);
        Assert.Equal("tip", wire["name"]);
        Assert.Equal(1.5, wire["weight"]);
    }

    [Fact]
    public void containers_map_element_wise_and_sets_are_written_in_a_stable_order()
    {
        EngineNode node = NodeSample();

        IReadOnlyDictionary<string, object?> wire = WireValues.AsMap(WireValues.Write(node));

        Assert.Equal(
            ["left", "right"],
            WireValues.AsList(wire["leaves"]).Select(item => WireValues.AsMap(item)["name"]));
        Assert.Equal(["primary"], WireValues.AsMap(wire["byName"]).Keys);

        // Written twice from differently ordered sets, the file has to come out the same.
        Assert.Equal(["alpha", "beta"], WireValues.AsList(wire["tags"]));
        EngineNode reordered = NodeSample(tags: ["beta", "alpha"]);
        Assert.Equal(["alpha", "beta"], WireValues.AsList(WireValues.AsMap(WireValues.Write(reordered))["tags"]));
    }

    [Fact]
    public void a_null_optional_is_written_as_null_and_read_back_as_null()
    {
        EngineNode node = NodeSample();

        IReadOnlyDictionary<string, object?> wire = WireValues.AsMap(WireValues.Write(node));
        Assert.Null(wire["optionalLeaf"]);

        EngineNode restored = (EngineNode)WireValues.ReadVersionable(wire, EngineNode.Metadata);
        Assert.Null(restored.OptionalLeaf);
    }

    [Fact]
    public void the_walker_round_trips_a_whole_object_graph()
    {
        EngineNode node = NodeSample(optionalLeaf: new EngineLeaf("maybe", 0.25));

        EngineNode restored = (EngineNode)WireValues.ReadVersionable(WireValues.Write(node), EngineNode.Metadata);

        Assert.Equal("root", restored.Label);
        Assert.Equal("centre", restored.Leaf.Name);
        Assert.Equal(2, restored.Leaves.Count);
        Assert.Equal(["left", "right"], restored.Leaves.Select(leaf => leaf.Name));
        Assert.Equal(3.5, restored.ByName["primary"].Weight);
        Assert.Equal(["alpha", "beta"], restored.Tags.Order(StringComparer.Ordinal));
        Assert.Equal("maybe", restored.OptionalLeaf?.Name);
    }

    [Fact]
    public void a_reference_cycle_is_reported_rather_than_recursed()
    {
        EngineChain first = new("first");
        EngineChain second = new("second");
        first.Next = second;
        second.Next = first;

        CircularReferenceException error = Assert.Throws<CircularReferenceException>(() => WireValues.Write(first));

        Assert.Equal(typeof(EngineChain), error.ObjectType);
        Assert.Equal("next.next", error.FieldPath);
    }

    [Fact]
    public void a_diamond_is_duplicated_rather_than_reported_as_a_cycle()
    {
        // The same instance reached twice down different branches is not a cycle; 0.2.x has no
        // shared-reference table, so it is written twice, exactly as Python writes it.
        EngineLeaf shared = new("shared", 1.0);
        EngineNode node = NodeSample(leaf: shared, optionalLeaf: shared);

        IReadOnlyDictionary<string, object?> wire = WireValues.AsMap(WireValues.Write(node));

        Assert.Equal("shared", WireValues.AsMap(wire["leaf"])["name"]);
        Assert.Equal("shared", WireValues.AsMap(wire["optionalLeaf"])["name"]);
    }

    [Fact]
    public void a_type_nothing_can_lower_is_rejected()
    {
        UnsupportedTypeException error = Assert.Throws<UnsupportedTypeException>(
            () => WireValues.Write(new UnknownToTheWalker()));

        Assert.Contains(nameof(UnknownToTheWalker), error.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void backend_native_types_are_not_run_through_the_walker()
    {
        UnknownToTheWalker native = new();

        object? wire = WireValues.Write(native, new[] { typeof(UnknownToTheWalker) }.ToFrozenSet());

        Assert.Same(native, wire);
    }

    [Fact]
    public void dictionary_keys_are_stringified_the_way_python_stringifies_them()
    {
        Dictionary<int, string> byIndex = new() { [1] = "one", [2] = "two" };

        IReadOnlyDictionary<string, object?> wire = WireValues.AsMap(WireValues.Write(byIndex));

        Assert.Equal(["1", "2"], wire.Keys.Order(StringComparer.Ordinal));
    }

    [Fact]
    public void a_versionable_dictionary_key_is_refused()
    {
        Dictionary<EngineLeaf, string> byLeaf = new() { [new EngineLeaf("k", 0)] = "value" };

        ConverterException error = Assert.Throws<ConverterException>(() => WireValues.Write(byLeaf));

        Assert.Contains("keys cannot be", error.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void an_integer_too_wide_for_its_field_is_an_error_rather_than_a_truncation()
    {
        ConverterException error = Assert.Throws<ConverterException>(
            () => WireValues.Read(long.MaxValue, typeof(int)));

        Assert.Contains("Int32", error.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void a_union_takes_the_first_member_that_accepts_the_value()
    {
        Func<object?, object?>[] members =
        [
            value => WireValues.Read(value, typeof(int)),
            value => WireValues.Read(value, typeof(string)),
        ];

        Assert.Equal(5, WireValues.ReadUnion(5L, members));
        Assert.Equal("text", WireValues.ReadUnion("text", members));
        Assert.Null(WireValues.ReadUnion(null, members));
    }

    [Fact]
    public void a_union_rethrows_a_dtype_mismatch_instead_of_trying_the_next_member()
    {
        // A declared-dtype violation is schema drift. Swallowing it would fall through to a member
        // that happens to accept the value and silently change the field's type.
        bool secondMemberRan = false;

        Assert.Throws<DtypeMismatchException>(() => WireValues.ReadUnion(
            "anything",
            _ => throw new DtypeMismatchException("float32", "float64", "samples", DtypeContext.Load),
            _ =>
            {
                secondMemberRan = true;
                return null;
            }));

        Assert.False(secondMemberRan);
    }

    [Fact]
    public void an_unmatched_union_hands_the_value_back_unchanged()
    {
        object? result = WireValues.ReadUnion(
            "text",
            value => WireValues.Read(value, typeof(int)),
            value => WireValues.Read(value, typeof(long)));

        Assert.Equal("text", result);
    }

    [Fact]
    public void a_union_with_one_member_propagates_the_failure_instead_of_swallowing_it()
    {
        // Python short-circuits `Optional[T]` straight into deserialize, outside its try/except.
        // With nothing to fall through to, returning the raw value would hand the factory a string
        // where an int belongs and surface as a cast failure with no clue where it came from.
        ConverterException error = Assert.Throws<ConverterException>(
            () => WireValues.ReadUnion("text", value => WireValues.Read(value, typeof(int))));

        Assert.Contains("Int32", error.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void a_numeric_enum_needs_no_generated_reader()
    {
        Assert.Equal(GoldenPriority.High, WireValues.Read(3L, typeof(GoldenPriority)));
        Assert.Equal(3, WireValues.Write(GoldenPriority.High));
    }

    [Fact]
    public void an_enum_value_the_build_does_not_define_is_rejected()
    {
        ConverterException error = Assert.Throws<ConverterException>(
            () => WireValues.Read(99L, typeof(GoldenPriority)));

        Assert.Contains("Unknown GoldenPriority value", error.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void a_string_valued_enum_writes_its_declared_value_not_its_ordinal()
    {
        // The whole reason the engine routes enums through EnumConverter: writing the ordinal here
        // would put 1 in a file where Python puts "slow", and neither side could read the other.
        Assert.Equal("slow", WireValues.Write(EngineMode.Slow));
        Assert.Equal(EngineMode.Slow, WireValues.Read("slow", typeof(EngineMode)));
    }

    [Fact]
    public void a_string_valued_enum_inside_a_container_goes_through_the_same_path()
    {
        EngineEnumHolder holder = new(
            EngineStatus.Retired, EngineMode.Fast, [EngineStatus.Active, EngineStatus.Unknown]);

        IReadOnlyDictionary<string, object?> wire = WireValues.AsMap(WireValues.Write(holder));

        Assert.Equal("retired", wire["status"]);
        Assert.Equal("fast", wire["mode"]);
        Assert.Equal(["active", "unknown"], WireValues.AsList(wire["history"]));

        EngineEnumHolder restored = (EngineEnumHolder)WireValues.ReadVersionable(wire, EngineEnumHolder.Metadata);
        Assert.Equal(EngineStatus.Retired, restored.Status);
        Assert.Equal([EngineStatus.Active, EngineStatus.Unknown], restored.History);
    }

    [Fact]
    public void an_unknown_value_lands_on_the_fallback_member_when_the_enum_names_one()
    {
        // Version skew, not corruption: enum member values are not hash-significant, so a newer
        // schema can add a member without the hash changing.
        Assert.Equal(EngineStatus.Unknown, WireValues.Read("decommissioned", typeof(EngineStatus)));
    }

    [Fact]
    public void an_unknown_value_is_an_error_when_the_enum_names_no_fallback()
    {
        ConverterException error = Assert.Throws<ConverterException>(
            () => WireValues.Read("glacial", typeof(EngineMode)));

        Assert.Contains("Unknown EngineMode value", error.Message, StringComparison.Ordinal);
        Assert.Contains("EnumFallback", error.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void a_container_of_literals_is_validated_element_wise_by_its_generated_reader()
    {
        // The seam documented on ObjectMaterializer.ValidateLiteral: the descriptor leaves
        // LiteralOptions null — whole-field validation would compare the list itself against
        // 'fast' and reject every file — so the reader checks each element instead.
        FieldDescriptor modes = EngineLiteralList.Metadata.Fields[0];
        Assert.Null(modes.LiteralOptions);

        Assert.Equal(["fast", "slow"], (List<string>)modes.WireReader!(new List<object?> { "fast", "slow" })!);

        ConverterException error = Assert.Throws<ConverterException>(
            () => modes.WireReader!(new List<object?> { "fast", "glacial" }));
        Assert.Contains("not a valid Literal option", error.Message, StringComparison.Ordinal);
    }

    private static EngineNode NodeSample(
        EngineLeaf? leaf = null,
        IEnumerable<string>? tags = null,
        EngineLeaf? optionalLeaf = null) =>
        new(
            "root",
            leaf ?? new EngineLeaf("centre", 2.5),
            [new EngineLeaf("left", 1.0), new EngineLeaf("right", 2.0)],
            new Dictionary<string, EngineLeaf>(StringComparer.Ordinal) { ["primary"] = new("primary", 3.5) },
            new HashSet<string>(tags ?? ["alpha", "beta"], StringComparer.Ordinal),
            optionalLeaf);

    private sealed class UnknownToTheWalker;
}
