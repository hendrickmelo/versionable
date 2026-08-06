using Versionable.Migrations;

namespace Versionable.Engine;

/// <summary>
/// The execution half of a migration chain: implemented alongside <see cref="IMigrationChain"/>
/// by whatever actually knows how to run the declared migrations.
/// </summary>
/// <remarks>
/// <see cref="IMigrationChain"/> describes a chain — which source versions it covers and how far
/// back it reaches — but has no member that runs one. That is deliberate on its part: its
/// documentation says its members "are what the load path needs to decide whether a file's
/// version is reachable", and that implementations may add members. This is that added member,
/// declared here rather than on the pinned contract so the load path can execute a chain without
/// waiting on the phase-4 migration builder.
/// <para>
/// Python counterpart: <c>applyMigrationRange</c> in <c>src/versionable/_migration.py</c>. The
/// same shape — raw field dictionary in, raw field dictionary out, before any value is
/// materialized — because migrations rename, drop, and add wire keys, and doing that after
/// materialization would need the old schema's CLR types to still exist.
/// </para>
/// <para>
/// A chain that implements <see cref="IMigrationChain"/> but not this interface is a declared
/// chain with no runner: <see cref="MigrationRunner"/> reports that as a
/// <see cref="Errors.MigrationException"/> rather than silently loading unmigrated data.
/// </para>
/// </remarks>
public interface IMigrationChainExecutor
{
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
    /// <exception cref="Errors.MigrationException">A migration failed.</exception>
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
