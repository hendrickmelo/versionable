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
/// public sealed class Config
/// {
///     public static class Migrate
///     {
///         public static readonly Migration V1 = new Migration().Rename("nm", "name");
///
///         [Migration(FromVersion = 2)]
///         public static void V2(MigrationContext data) =&gt; data["port"] = 8080;
///     }
/// }
/// </code>
/// </example>
///
/// <para>
/// Contract only: the migration engine — builder ops, the context type, in-place upgrades —
/// lands with the migrations work in phase 4. Implementations may add members; those here
/// are what the load path needs to decide whether a file's version is reachable.
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
}
