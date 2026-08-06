using Versionable.Errors;

namespace Versionable.Migrations;

/// <summary>
/// A migration chain composed of declared steps: what the source generator hands to
/// <see cref="VersionableMetadata.Migrations"/> for a type whose nested <c>Migrate</c> class holds
/// <c>V1</c>/<c>V2</c> members and <c>[Migration]</c> methods.
/// </summary>
/// <remarks>
/// Python counterpart: the dictionary <c>resolveMigrations</c> assembles from the <c>Migrate</c>
/// class plus the loop in <c>applyMigrations</c> that walks it one version at a time
/// (<c>src/versionable/_migration.py</c>). Python collects by reflection at load; here the
/// generator collects at compile time and emits the constructor call, which is the same chain
/// without the reflection (ADR-0003).
/// <para>
/// Usable by hand as well — a chain built at run time is just a
/// <see cref="MigrationChain"/> constructed with the steps it computed.
/// </para>
/// </remarks>
public sealed class MigrationChain : IMigrationChain
{
    private readonly Dictionary<int, MigrationStep> _byVersion;
    private readonly MigrationStep[] _steps;

    /// <summary>Initializes a new instance of the <see cref="MigrationChain"/> class.</summary>
    /// <param name="steps">The steps, in any order; duplicates are rejected.</param>
    /// <exception cref="ArgumentException">Two steps declare the same source version.</exception>
    public MigrationChain(params MigrationStep[] steps)
    {
        ArgumentNullException.ThrowIfNull(steps);

        _steps = [.. steps.OrderBy(step => step.FromVersion)];
        _byVersion = new Dictionary<int, MigrationStep>(_steps.Length);
        foreach (MigrationStep step in _steps)
        {
            if (!_byVersion.TryAdd(step.FromVersion, step))
            {
                throw new ArgumentException(
                    $"Two migrations declare source version {step.FromVersion}; a version can only be "
                        + "migrated one way.",
                    nameof(steps));
            }
        }

        Steps = Array.AsReadOnly(_steps);
        FromVersions = Array.AsReadOnly(_steps.Select(step => step.FromVersion).ToArray());
    }

    /// <summary>The steps, ascending by source version.</summary>
    /// <remarks>
    /// Read-only wrappers rather than the backing arrays, which a caller could otherwise cast
    /// back and rewrite — a chain is shared by every load of its type.
    /// </remarks>
    public IReadOnlyList<MigrationStep> Steps { get; }

    /// <inheritdoc/>
    public IReadOnlyList<int> FromVersions { get; }

    /// <inheritdoc/>
    /// <remarks>
    /// Derived from the steps: a chain starting at <c>v1</c> reaches every file, so it reports
    /// <see langword="null"/>; one starting later cannot read anything older than its first step.
    /// </remarks>
    public int? MinReversibleVersion => _steps.Length > 0 && _steps[0].FromVersion > 1
        ? _steps[0].FromVersion
        : null;

    /// <inheritdoc/>
    public IDictionary<string, object?> Apply(
        IDictionary<string, object?> fields,
        int fromVersion,
        int toVersion,
        bool upgradeInPlace)
    {
        ArgumentNullException.ThrowIfNull(fields);

        for (int version = fromVersion; version < toVersion; version++)
        {
            if (!_byVersion.TryGetValue(version, out MigrationStep? step))
            {
                throw new MigrationException(
                    MigrationMessages.MissingStep(version, fromVersion, toVersion, FromVersions));
            }

            fields = step.Apply(fields, upgradeInPlace);
        }

        return fields;
    }
}
