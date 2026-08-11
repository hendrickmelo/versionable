namespace Versionable.Migrations;

/// <summary>
/// One version's worth of migration: everything that takes data written at
/// <see cref="FromVersion"/> to <c>FromVersion + 1</c>, in either of the two forms.
/// </summary>
/// <remarks>
/// Python counterpart: an entry of the <c>migrations</c> dict that <c>resolveMigrations</c> builds
/// in <c>src/versionable/_migration.py</c>, whose values are a <c>Migration</c> or an
/// <c>_ImperativeMigration</c>. Both forms are kept distinguishable rather than being erased into
/// one delegate, so a compiled chain can still be read back op by op.
/// </remarks>
public sealed class MigrationStep
{
    private MigrationStep(int fromVersion, Migration? declarative, Action<MigrationContext>? imperative)
    {
        FromVersion = fromVersion;
        Declarative = declarative;
        Imperative = imperative;
    }

    /// <summary>Schema version of the data this step accepts. It produces data at one version later.</summary>
    public int FromVersion { get; }

    /// <summary>The declarative migration, or <see langword="null"/> for an imperative step.</summary>
    public Migration? Declarative { get; }

    /// <summary>The imperative migration, or <see langword="null"/> for a declarative step.</summary>
    public Action<MigrationContext>? Imperative { get; }

    /// <summary>Creates a step from a declarative migration.</summary>
    /// <param name="fromVersion">Version the migration reads.</param>
    /// <param name="migration">The migration.</param>
    /// <returns>The step.</returns>
    public static MigrationStep Of(int fromVersion, Migration migration)
    {
        ArgumentNullException.ThrowIfNull(migration);
        return new MigrationStep(fromVersion, migration, null);
    }

    /// <summary>Creates a step from an imperative migration method.</summary>
    /// <param name="fromVersion">Version the migration reads, from <c>[Migration(FromVersion = n)]</c>.</param>
    /// <param name="migration">The method, which mutates the context it is given.</param>
    /// <returns>The step.</returns>
    public static MigrationStep Of(int fromVersion, Action<MigrationContext> migration)
    {
        ArgumentNullException.ThrowIfNull(migration);
        return new MigrationStep(fromVersion, null, migration);
    }

    /// <summary>Runs this step over <paramref name="fields"/>.</summary>
    /// <param name="fields">Raw field values keyed by wire name. Mutated in place and returned.</param>
    /// <param name="upgradeInPlace">Whether the caller has permitted file-rewriting migrations.</param>
    /// <returns><paramref name="fields"/>, reshaped for the next version.</returns>
    public IDictionary<string, object?> Apply(IDictionary<string, object?> fields, bool upgradeInPlace = false)
    {
        ArgumentNullException.ThrowIfNull(fields);

        if (Declarative is not null)
        {
            return Declarative.Apply(fields, upgradeInPlace);
        }

        // An imperative migration is arbitrary code, so upgradeInPlace has nothing to gate here:
        // Python's requiresUpgrade() is an op of the declarative builder and has no imperative
        // counterpart either.
        Imperative!(new MigrationContext(fields));
        return fields;
    }
}
