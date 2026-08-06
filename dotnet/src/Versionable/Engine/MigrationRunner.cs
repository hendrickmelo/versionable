using Versionable.Errors;
using Versionable.Migrations;

namespace Versionable.Engine;

/// <summary>
/// Decides what a file's recorded version means for the type loading it, and runs the migration
/// chain when one is needed.
/// </summary>
/// <remarks>
/// Python counterpart: the version branch of <c>load()</c> in <c>src/versionable/_api.py</c>
/// (lines 137-160), the matching branch of <c>_deserializeVersionable</c> in
/// <c>src/versionable/_types.py</c>, and <c>resolveMigrations</c> /
/// <c>applyMigrationRange</c> in <c>src/versionable/_migration.py</c>.
/// <para>
/// <b>The stored hash is never compared at load.</b> Python reads it into the metadata dict and
/// then ignores it, and so does this: the hash is a compile-time tripwire against undeclared
/// schema edits (ADR-0003), not a load-time gate. A file whose hash disagrees with the current
/// schema but whose version matches loads, because that is exactly the situation
/// <c>version</c> plus a migration is there to describe. Comparing hashes here would reject
/// every file written before a schema change that its migration chain already handles.
/// </para>
/// </remarks>
public static class MigrationRunner
{
    /// <summary>
    /// Brings <paramref name="fields"/> from <paramref name="fromVersion"/> up to
    /// <paramref name="metadata"/>'s version.
    /// </summary>
    /// <param name="fields">
    /// Raw field values keyed by wire name, envelope keys already stripped.
    /// </param>
    /// <param name="metadata">Metadata of the type being loaded into.</param>
    /// <param name="fromVersion">Schema version recorded in the file.</param>
    /// <param name="upgradeInPlace">Whether file-rewriting migrations are permitted.</param>
    /// <param name="description">
    /// How to name the object in error messages, e.g. <c>Config</c> or <c>nested Config</c>.
    /// </param>
    /// <returns>
    /// The field dictionary shaped for the current version — <paramref name="fields"/> itself
    /// when the versions already agree.
    /// </returns>
    /// <exception cref="VersionException">
    /// The file is newer than the code, or older with no chain reaching back that far.
    /// </exception>
    /// <exception cref="MigrationException">The chain has no runner, or a migration failed.</exception>
    public static IDictionary<string, object?> Run(
        IDictionary<string, object?> fields,
        VersionableMetadata metadata,
        int fromVersion,
        bool upgradeInPlace,
        string description)
    {
        ArgumentNullException.ThrowIfNull(fields);
        ArgumentNullException.ThrowIfNull(metadata);

        if (fromVersion == metadata.Version)
        {
            return fields;
        }

        if (fromVersion > metadata.Version)
        {
            throw new VersionException(
                $"{description}: file version ({fromVersion}) is newer than class version "
                    + $"({metadata.Version}). Cannot downgrade.");
        }

        IMigrationChain chain = metadata.Migrations
            ?? throw new VersionException(
                $"{description}: file version ({fromVersion}) is older than class version "
                    + $"({metadata.Version}) and the type declares no migrations. Add a nested "
                    + $"Migrate class with the migrations from version {fromVersion} onwards.");

        EnsureReachable(chain, metadata, fromVersion, description);

        if (chain is not IMigrationChainExecutor executor)
        {
            throw new MigrationException(
                $"{description}: the migration chain declares source versions "
                    + $"[{string.Join(", ", chain.FromVersions)}] but does not implement "
                    + $"{nameof(IMigrationChainExecutor)}, so it cannot be run.");
        }

        return executor.Apply(fields, fromVersion, metadata.Version, upgradeInPlace);
    }

    private static void EnsureReachable(
        IMigrationChain chain,
        VersionableMetadata metadata,
        int fromVersion,
        string description)
    {
        int oldestSupported = chain.MinReversibleVersion ?? 1;
        if (fromVersion < oldestSupported)
        {
            throw new VersionException(
                $"{description}: file version ({fromVersion}) predates the oldest migration in the "
                    + $"chain (version {oldestSupported}). Files this old are no longer supported.");
        }

        // Contiguity above the oldest migration is the analyzer's job at compile time, but a
        // chain assembled at runtime — or one whose type was recompiled against newer
        // migrations — can still have a hole. Report the hole rather than skipping a step and
        // handing the materializer a dictionary shaped for a version that never existed.
        for (int version = fromVersion; version < metadata.Version; version++)
        {
            if (!chain.FromVersions.Contains(version))
            {
                throw new VersionException(
                    $"{description}: no migration from version {version} to {version + 1}. "
                        + $"Migrating {fromVersion} -> {metadata.Version} needs every step in "
                        + $"between; declared steps: [{string.Join(", ", chain.FromVersions)}].");
            }
        }
    }
}
