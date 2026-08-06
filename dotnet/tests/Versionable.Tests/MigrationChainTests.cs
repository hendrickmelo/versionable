using System.Globalization;
using Versionable.Errors;
using Versionable.Migrations;
using Xunit;

namespace Versionable.Tests;

/// <summary>
/// Composed chains: the two declaration forms in one chain, the version walk over them, and the
/// context an imperative migration is handed.
/// </summary>
/// <remarks>
/// Ported from <c>TestImperativeMigration</c>, <c>TestResolveMigrations</c>, and
/// <c>TestEndToEndMigration</c> in <c>tests/test_migration.py</c>. What Python resolves by walking
/// <c>dir(Migrate)</c> at load, the generator resolves at compile time, so the end-to-end tests
/// here go through a real <c>[Versionable]</c> type rather than through a hand-built chain.
/// </remarks>
[Collection(RegistryCollection.Name)]
public class MigrationChainTests
{
    public MigrationChainTests() => GoldenSchemas.EnsureRegistered();

    // ------------------------------------------------------------------
    // Composition
    // ------------------------------------------------------------------

    [Fact]
    public void a_chain_orders_its_steps_by_source_version()
    {
        MigrationChain chain = new(
            MigrationStep.Of(3, new Migration()),
            MigrationStep.Of(1, new Migration()),
            MigrationStep.Of(2, new Migration()));

        Assert.Equal([1, 2, 3], chain.FromVersions);
    }

    [Fact]
    public void a_chain_reaching_back_to_version_one_reports_no_floor()
    {
        Assert.Null(new MigrationChain(MigrationStep.Of(1, new Migration())).MinReversibleVersion);
    }

    [Fact]
    public void a_chain_starting_later_reports_the_oldest_file_it_can_read()
    {
        // Not an error: dropping the ancient migrations is how a schema says it no longer supports
        // files that old, and MinReversibleVersion is what the load path refuses them with.
        MigrationChain chain = new(MigrationStep.Of(4, new Migration()), MigrationStep.Of(5, new Migration()));

        Assert.Equal(4, chain.MinReversibleVersion);
    }

    [Fact]
    public void two_migrations_from_one_version_are_refused()
    {
        ArgumentException error = Assert.Throws<ArgumentException>(
            () => new MigrationChain(
                MigrationStep.Of(1, new Migration()),
                MigrationStep.Of(1, data => data.Drop("x"))));

        Assert.Contains("source version 1", error.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void applying_a_chain_runs_only_the_steps_in_the_range()
    {
        List<int> applied = [];
        MigrationChain chain = new(
            MigrationStep.Of(1, _ => applied.Add(1)),
            MigrationStep.Of(2, _ => applied.Add(2)),
            MigrationStep.Of(3, _ => applied.Add(3)));

        chain.Apply(new Dictionary<string, object?>(StringComparer.Ordinal), fromVersion: 2, toVersion: 4, upgradeInPlace: false);

        Assert.Equal([2, 3], applied);
    }

    [Fact]
    public void a_hole_in_the_range_is_reported_rather_than_skipped()
    {
        // Python's resolveMigrations raises MigrationError for the same case. Skipping the step
        // would hand the materializer a dictionary shaped for a version that never existed.
        MigrationChain chain = new(MigrationStep.Of(1, new Migration()), MigrationStep.Of(3, new Migration()));

        MigrationException error = Assert.Throws<MigrationException>(
            () => chain.Apply(
                new Dictionary<string, object?>(StringComparer.Ordinal),
                fromVersion: 1,
                toVersion: 4,
                upgradeInPlace: false));

        Assert.Contains("No migration from version 2 to 3", error.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void a_declarative_and_an_imperative_step_compose_into_one_chain()
    {
        MigrationChain chain = new(
            MigrationStep.Of(1, new Migration().Rename("title", "name")),
            MigrationStep.Of(2, data => data["retries"] = ((int)data.Pop("retry_count")!) + 1));

        IDictionary<string, object?> result = chain.Apply(
            new Dictionary<string, object?>(StringComparer.Ordinal) { ["title"] = "job", ["retry_count"] = 2 },
            fromVersion: 1,
            toVersion: 3,
            upgradeInPlace: false);

        Assert.Equal("job", result["name"]);
        Assert.Equal(3, result["retries"]);
    }

    [Fact]
    public void an_upgrade_only_migration_reaches_the_step_that_declares_it()
    {
        MigrationChain chain = new(MigrationStep.Of(1, new Migration().RequiresUpgrade().Rename("a", "b")));

        Assert.Throws<UpgradeRequiredException>(() => chain.Apply(
            new Dictionary<string, object?>(StringComparer.Ordinal) { ["a"] = 1 },
            fromVersion: 1,
            toVersion: 2,
            upgradeInPlace: false));

        IDictionary<string, object?> upgraded = chain.Apply(
            new Dictionary<string, object?>(StringComparer.Ordinal) { ["a"] = 1 },
            fromVersion: 1,
            toVersion: 2,
            upgradeInPlace: true);

        Assert.Equal(1, upgraded["b"]);
    }

    // ------------------------------------------------------------------
    // The imperative context
    // ------------------------------------------------------------------

    [Fact]
    public void an_imperative_migration_edits_the_dictionary_it_is_given()
    {
        Dictionary<string, object?> fields = new(StringComparer.Ordinal) { ["old_field"] = 5 };

        MigrationStep.Of(1, data => data["new_field"] = (int)data.Pop("old_field")! * 2).Apply(fields);

        Assert.Equal(new[] { "new_field" }, fields.Keys);
        Assert.Equal(10, fields["new_field"]);
    }

    [Fact]
    public void the_context_reads_conditionally_the_way_python_does()
    {
        // Ported from TestImperativeMigration.test_conditionalLogic: `if "mode" in ctx and ...`.
        Dictionary<string, object?> fields = new(StringComparer.Ordinal) { ["mode"] = "legacy", ["value"] = 5 };
        MigrationContext context = new(fields);

        if (context.Contains("mode") && Equals(context["mode"], "legacy"))
        {
            context["value"] = (int)context["value"]! * 1000;
        }

        context.Drop("mode");

        Assert.Equal(new[] { "value" }, fields.Keys);
        Assert.Equal(5000, fields["value"]);
    }

    [Fact]
    public void popping_an_absent_field_throws_unless_a_fallback_is_given()
    {
        MigrationContext context = new(new Dictionary<string, object?>(StringComparer.Ordinal));

        Assert.Throws<KeyNotFoundException>(() => context.Pop("missing"));
        Assert.Equal(7, context.Pop("missing", 7));
        context.Drop("missing");
    }

    [Fact]
    public void the_context_snapshot_is_a_copy()
    {
        Dictionary<string, object?> fields = new(StringComparer.Ordinal) { ["a"] = 1 };
        MigrationContext context = new(fields);

        Dictionary<string, object?> snapshot = context.ToDictionary();
        snapshot["a"] = 2;

        Assert.Equal(1, fields["a"]);
    }

    // ------------------------------------------------------------------
    // End to end, through the generated chain
    // ------------------------------------------------------------------

    [Fact]
    public void the_generator_composes_both_declaration_forms_into_the_metadata()
    {
        MigrationChain chain = Assert.IsType<MigrationChain>(MigratingConfig.VersionableMetadata.Migrations);

        Assert.Equal([1, 2], chain.FromVersions);
        Assert.Same(MigratingConfig.Migrate.V1, chain.Steps[0].Declarative);
        Assert.NotNull(chain.Steps[1].Imperative);
    }

    [Fact]
    public void a_version_one_file_walks_both_steps_on_load()
    {
        using Hdf5TempFile file = new(".json");
        File.WriteAllText(
            file.Path,
            """
            {
              "__versionable__": { "object": "MigratingConfig", "version": 1, "hash": "" },
              "title": "legacy-worker",
              "debug": true,
              "retries": 9,
              "timeout_s": 1.5
            }
            """);

        MigratingConfig loaded = VersionableFile.Load<MigratingConfig>(file.Path);

        Assert.Equal("legacy-worker", loaded.Name);
        Assert.Equal(9, loaded.Retries);
        Assert.Equal(1500, loaded.TimeoutMs);
    }

    [Fact]
    public void a_version_two_file_walks_only_the_imperative_step()
    {
        using Hdf5TempFile file = new(".json");
        File.WriteAllText(
            file.Path,
            """
            {
              "__versionable__": { "object": "MigratingConfig", "version": 2, "hash": "" },
              "name": "interim-worker",
              "retries": 2
            }
            """);

        MigratingConfig loaded = VersionableFile.Load<MigratingConfig>(file.Path);

        Assert.Equal("interim-worker", loaded.Name);
        Assert.Equal(2, loaded.Retries);

        // The migration decides what the field meant before it existed — 0, not the schema
        // default of 30000 that a plain load would have filled in.
        Assert.Equal(0, loaded.TimeoutMs);
    }

    [Fact]
    public void a_current_file_is_left_alone()
    {
        using Hdf5TempFile file = new(".json");
        File.WriteAllText(
            file.Path,
            """
            {
              "__versionable__": { "object": "MigratingConfig", "version": 3, "hash": "" },
              "name": "current",
              "retries": 1,
              "timeout_ms": 250
            }
            """);

        MigratingConfig loaded = VersionableFile.Load<MigratingConfig>(file.Path);

        Assert.Equal(250, loaded.TimeoutMs);
    }
}

/// <summary>
/// A type whose chain is half declarative and half imperative, which is the case the generator has
/// to compose rather than hand over.
/// </summary>
/// <remarks>
/// Python counterpart: the mixed <c>Migrate</c> classes of <c>tests/test_nested_migrations.py</c>,
/// where a <c>v1 = Migration()...</c> attribute and a <c>@migration(fromVersion=2)</c> function sit
/// in one class.
/// </remarks>
[Versionable(Version = 3, Hash = "aac8a2")]
internal sealed partial class MigratingConfig
{
    /// <summary>Worker name; <c>title</c> before v1.</summary>
    [VersionableField("name")]
    public required string Name { get; init; }

    /// <summary>Retry count.</summary>
    [VersionableField("retries")]
    public int Retries { get; init; } = 3;

    /// <summary>Timeout, in milliseconds; seconds before v2.</summary>
    [VersionableField("timeout_ms")]
    public int TimeoutMs { get; init; } = 30000;

    /// <summary>The chain, one member per form.</summary>
    public static class Migrate
    {
        /// <summary>Takes a version 1 file to version 2.</summary>
        public static readonly Migration V1 = new Migration().Rename("title", "name").Drop("debug");

        /// <summary>Takes a version 2 file to version 3, in seconds-to-milliseconds.</summary>
        /// <param name="data">The raw fields read from the file.</param>
        /// <remarks>
        /// Imperative because the value depends on what the file carries: a v2 file written before
        /// the timeout existed has neither key, and its timeout was 0, not the schema default.
        /// </remarks>
        [Migration(FromVersion = 2)]
        public static void ToV3(MigrationContext data)
        {
            ArgumentNullException.ThrowIfNull(data);

            data["timeout_ms"] = data.TryGetValue("timeout_s", out object? seconds)
                ? (int)(System.Convert.ToDouble(seconds, CultureInfo.InvariantCulture) * 1000)
                : 0;
            data.Drop("timeout_s");
        }
    }
}
