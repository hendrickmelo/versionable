namespace Versionable.Migrations;

/// <summary>
/// The migrations declared for one <c>[Versionable]</c> type, referenced from
/// <see cref="VersionableMetadata.Migrations"/>.
/// </summary>
/// <remarks>
/// Python counterpart: the migration lookup performed by <c>resolveMigrations()</c> and
/// <c>applyMigrationRange()</c> in <c>src/versionable/_migration.py</c>, which read
/// <c>Migrate.V1</c>-style class attributes and <c>@migration(fromVersion=n)</c> functions
/// off the class.
///
/// <para><b>Declaration convention.</b> A type declares its migrations in a nested class
/// named <c>Migrate</c>. The source generator looks for that exact name and collects two
/// kinds of member into one chain:</para>
/// <list type="bullet">
///   <item>
///     <description>
///     <b>Declarative</b> — a static field or property named <c>V</c> followed by a version
///     number (<c>V1</c>, <c>V2</c>, …) of type <c>Migration</c>, built with the op builder.
///     <c>V1</c> migrates data written at version 1 to version 2, so the name carries the
///     source version exactly as Python's <c>Migrate.V1</c> does.
///     </description>
///   </item>
///   <item>
///     <description>
///     <b>Imperative</b> — a static method carrying <see cref="MigrationAttribute"/> with its
///     <c>FromVersion</c> set.
///     </description>
///   </item>
/// </list>
///
/// <para>
/// The analyzer checks the chain for <em>contiguity</em>: no gaps above the oldest migration
/// present. Gaps below it are legitimate — they mean files that old are no longer supported,
/// which is what <see cref="MinReversibleVersion"/> reports.
/// </para>
///
/// <example>
/// <code>
/// [Versionable(Version = 3, Hash = "a1b2c3")]
/// public sealed partial class Config
/// {
///     public static class Migrate
///     {
///         public static readonly Migration V1 = new Migration().Rename("nm", "name");
///
///         [Migration(FromVersion = 2)]
///         public static void ToV3(MigrationContext data) =&gt; data["port"] = 8080;
///     }
/// }
/// </code>
/// From those members the generator emits
/// <c>Migrations = new MigrationChain(MigrationStep.Of(1, Migrate.V1), MigrationStep.Of(2, Migrate.ToV3))</c>.
/// </example>
///
/// <para><b>The hand-written form.</b> A nested <c>Migrate</c> that <em>is</em> an
/// <see cref="IMigrationChain"/> — a non-static, non-abstract, non-generic class with an accessible
/// parameterless constructor — is taken as the chain itself and emitted as
/// <c>Migrations = new Migrate()</c>. It is the escape hatch for a chain that is decided at run
/// time; its <see cref="FromVersions"/> is a run-time value, so the analyzer cannot check its
/// contiguity and does not try. When a type declares both shapes the chain type wins, since it is
/// the more explicit statement.
/// </para>
/// </remarks>
public interface IMigrationChain
{
    /// <summary>
    /// Source versions with a declared migration, ascending. A migration listed as <c>n</c>
    /// takes data written at version <c>n</c> to version <c>n + 1</c>.
    /// </summary>
    IReadOnlyList<int> FromVersions { get; }

    /// <summary>
    /// Oldest file version this chain can migrate forward, or <see langword="null"/> when the
    /// chain reaches back to version 1. Python counterpart:
    /// <c>VersionableMetadata.minReversibleVersion</c> in <c>src/versionable/_base.py</c>.
    /// </summary>
    int? MinReversibleVersion { get; }

    /// <summary>
    /// Runs every migration from <paramref name="fromVersion"/> up to
    /// <paramref name="toVersion"/> over <paramref name="fields"/>.
    /// </summary>
    /// <param name="fields">
    /// Raw field values keyed by the wire names found in the file, envelope keys already
    /// stripped. Implementations may mutate and return it.
    /// </param>
    /// <param name="fromVersion">Schema version the data is currently shaped for.</param>
    /// <param name="toVersion">Schema version to leave the data shaped for.</param>
    /// <param name="upgradeInPlace">
    /// Whether the caller has permitted migrations that need the file rewritten. Python
    /// counterpart: the <c>upgradeInPlace</c> argument; without it, an op that declares
    /// <c>requiresUpgrade()</c> raises <see cref="Errors.UpgradeRequiredException"/>.
    /// </param>
    /// <returns>The migrated field dictionary, shaped for <paramref name="toVersion"/>.</returns>
    /// <remarks>
    /// Python counterpart: <c>applyMigrationRange</c> in <c>src/versionable/_migration.py</c>. The
    /// same shape — raw field dictionary in, raw field dictionary out, before any value is
    /// materialized — because migrations rename, drop, and add wire keys, and doing that after
    /// materialization would need the old schema's CLR types to still exist.
    /// </remarks>
    /// <exception cref="Errors.MigrationException">A migration failed, or the chain has no step for a version in the range.</exception>
    /// <exception cref="Errors.UpgradeRequiredException">
    /// A migration needs the file rewritten and <paramref name="upgradeInPlace"/> is
    /// <see langword="false"/>.
    /// </exception>
    IDictionary<string, object?> Apply(
        IDictionary<string, object?> fields,
        int fromVersion,
        int toVersion,
        bool upgradeInPlace);
}
