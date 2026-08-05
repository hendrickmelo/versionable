namespace Versionable.Migrations;

/// <summary>
/// Marks a static method as the imperative migration from one schema version to the next.
/// </summary>
/// <remarks>
/// Python counterpart: the <c>@migration(fromVersion=n)</c> decorator in
/// <c>src/versionable/_migration.py</c>, which wraps a function taking a
/// <c>MigrationContext</c> over the raw field dict.
/// <para>
/// The method must be static, take the migration context, and return <see langword="void"/>;
/// it mutates the raw field dictionary read from the file, before any value is materialized
/// into the target type. It runs for data written at <see cref="FromVersion"/> and leaves it
/// shaped for version <c>FromVersion + 1</c>.
/// </para>
/// <para>
/// Declarative migrations — the <c>Migration</c> builder — are declared instead as fields of
/// the nested <c>Migrate</c> class described on <see cref="IMigrationChain"/>. Use whichever
/// suits; both land in the same chain.
/// </para>
/// </remarks>
[AttributeUsage(AttributeTargets.Method, Inherited = false)]
public sealed class MigrationAttribute : Attribute
{
    /// <summary>
    /// Schema version of the data this migration accepts. It produces data at
    /// <c>FromVersion + 1</c>.
    /// </summary>
    public int FromVersion { get; init; }
}
