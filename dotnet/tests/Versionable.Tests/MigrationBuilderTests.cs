using Versionable.Errors;
using Versionable.Migrations;
using Xunit;

namespace Versionable.Tests;

/// <summary>
/// The declarative builder, op by op.
/// </summary>
/// <remarks>
/// Ported from <c>TestDeclarativeOperations</c>, <c>TestMigrationChaining</c>, and
/// <c>TestRequiresUpgrade</c> in <c>tests/test_migration.py</c>: the same inputs and the same
/// expected dictionaries, because a migration that behaves differently here would silently
/// re-interpret files the Python implementation wrote.
/// </remarks>
public class MigrationBuilderTests
{
    [Fact]
    public void rename_moves_the_value_to_the_new_name()
    {
        AssertFields(Apply(new Migration().Rename("old", "new"), ("old", 42)), ("new", 42));
    }

    [Fact]
    public void rename_of_an_absent_field_does_nothing()
    {
        // A file that never carried the old name is already shaped for the new schema; inventing
        // a null under the new name would be a value the schema never had.
        AssertFields(Apply(new Migration().Rename("old", "new"), ("keep", 1)), ("keep", 1));
    }

    [Fact]
    public void add_introduces_a_field_the_old_file_lacks()
    {
        AssertFields(Apply(new Migration().Add("extra", 99), ("name", "test")), ("name", "test"), ("extra", 99));
    }

    [Fact]
    public void add_never_overwrites_a_value_the_file_carries()
    {
        AssertFields(Apply(new Migration().Add("name", "default"), ("name", "existing")), ("name", "existing"));
    }

    [Fact]
    public void add_computed_calls_the_factory_per_application()
    {
        Migration migration = new Migration().AddComputed("items", () => new List<int>());

        IDictionary<string, object?> first = Apply(migration);
        IDictionary<string, object?> second = Apply(migration);

        // Python's `add(field, default=list)` calls the callable rather than storing it, so two
        // files migrated by the same declaration do not end up sharing one list.
        Assert.NotSame(first["items"], second["items"]);
        Assert.Empty(Assert.IsType<List<int>>(first["items"]));
    }

    [Fact]
    public void drop_removes_the_field_and_ignores_its_absence()
    {
        AssertFields(Apply(new Migration().Drop("old").Drop("nonexistent"), ("old", 1), ("keep", 2)), ("keep", 2));
    }

    [Fact]
    public void convert_rewrites_the_value_in_place()
    {
        IDictionary<string, object?> result = Apply(
            new Migration().Convert("temp", celsius => ((int)celsius!) * 9.0 / 5.0 + 32.0),
            ("temp", 100));

        Assert.Equal(212.0, result["temp"]);
    }

    [Fact]
    public void convert_records_a_reverse_without_running_it()
    {
        bool reversed = false;
        Migration migration = new Migration().Convert(
            "value",
            value => (int)value! + 1,
            value =>
            {
                reversed = true;
                return value;
            });

        IDictionary<string, object?> result = Apply(migration, ("value", 1));

        Assert.Equal(2, result["value"]);
        Assert.False(reversed);
        Assert.NotNull(Assert.IsType<ConvertOp>(Assert.Single(migration.Ops)).Reverse);
    }

    [Fact]
    public void derive_computes_a_new_field_and_keeps_the_source()
    {
        AssertFields(
            Apply(new Migration().Derive("doubled", "value", value => (int)value! * 2), ("value", 5)),
            ("value", 5),
            ("doubled", 10));
    }

    [Fact]
    public void derive_from_an_absent_source_writes_nothing()
    {
        AssertFields(
            Apply(new Migration().Derive("doubled", "value", value => (int)value! * 2), ("other", 1)),
            ("other", 1));
    }

    [Fact]
    public void split_consumes_the_source_and_writes_every_target()
    {
        Migration migration = new Migration().Split(
            "full_name",
            new SplitTarget("first", name => ((string)name!).Split(' ')[0]),
            new SplitTarget("last", name => ((string)name!).Split(' ')[1]));

        AssertFields(Apply(migration, ("full_name", "John Doe")), ("first", "John"), ("last", "Doe"));
    }

    [Fact]
    public void merge_consumes_every_present_field()
    {
        Migration migration = new Migration().Merge(
            ["first", "last"],
            "full",
            values => string.Join(" ", values));

        AssertFields(Apply(migration, ("first", "John"), ("last", "Doe")), ("full", "John Doe"));
    }

    [Fact]
    public void merge_passes_only_the_fields_the_file_carried()
    {
        // Python collects `[result.pop(f) for f in op.fields if f in result]`, so a file missing
        // one input merges the rest rather than failing.
        Migration migration = new Migration().Merge(
            ["first", "last"],
            "full",
            values => string.Join(" ", values));

        AssertFields(Apply(migration, ("first", "John")), ("full", "John"));
    }

    [Fact]
    public void merge_of_nothing_present_writes_nothing()
    {
        Migration migration = new Migration().Merge(["first", "last"], "full", values => values.Length);

        AssertFields(Apply(migration, ("other", 1)), ("other", 1));
    }

    [Fact]
    public void ops_run_in_declaration_order()
    {
        Migration migration = new Migration().Rename("title", "name").Add("version", 1).Drop("old");

        AssertFields(Apply(migration, ("title", "Test"), ("old", "junk")), ("name", "Test"), ("version", 1));
    }

    [Fact]
    public void then_concatenates_two_migrations()
    {
        Migration combined = new Migration().Rename("a", "b").Then(new Migration().Rename("b", "c"));

        AssertFields(Apply(combined, ("a", 1)), ("c", 1));
    }

    [Fact]
    public void then_leaves_both_operands_alone()
    {
        // The C# builder is immutable where Python's mutates in place: a migration published as a
        // static readonly member must not be extendable by whoever chains onto it.
        Migration first = new Migration().Rename("a", "b");
        Migration second = new Migration().Rename("b", "c");

        _ = first.Then(second);

        Assert.Single(first.Ops);
        Assert.Single(second.Ops);
        AssertFields(Apply(first, ("a", 1)), ("b", 1));
    }

    [Fact]
    public void requires_upgrade_refuses_a_read_only_load()
    {
        UpgradeRequiredException error = Assert.Throws<UpgradeRequiredException>(
            () => Apply(new Migration().RequiresUpgrade()));

        Assert.Contains("in-place", error.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void requires_upgrade_lets_the_rest_run_when_permitted()
    {
        IDictionary<string, object?> fields = new Dictionary<string, object?>(StringComparer.Ordinal);

        new Migration().RequiresUpgrade().Add("x", 1).Apply(fields, upgradeInPlace: true);

        AssertFields(fields, ("x", 1));
    }

    [Fact]
    public void the_ops_are_readable_data_rather_than_opaque_closures()
    {
        // What the phase-6 Python-to-C# converter emits from, and what makes a chain reviewable
        // without running it.
        Migration migration = new Migration().Rename("title", "name").Add("retries", 3).Drop("debug");

        Assert.Collection(
            migration.Ops,
            op => Assert.Equal(new RenameOp("title", "name"), op),
            op => Assert.Equal(new AddOp("retries", 3, null), op),
            op => Assert.Equal(new DropOp("debug"), op));
    }

    private static void AssertFields(
        IDictionary<string, object?> actual,
        params (string Key, object? Value)[] expected)
    {
        Assert.Equal(expected.Select(field => field.Key).OrderBy(key => key, StringComparer.Ordinal), actual.Keys.OrderBy(key => key, StringComparer.Ordinal));
        foreach ((string key, object? value) in expected)
        {
            Assert.Equal(value, actual[key]);
        }
    }

    private static IDictionary<string, object?> Apply(Migration migration, params (string Key, object? Value)[] fields)
    {
        Dictionary<string, object?> data = new(StringComparer.Ordinal);
        foreach ((string key, object? value) in fields)
        {
            data[key] = value;
        }

        return migration.Apply(data);
    }
}
