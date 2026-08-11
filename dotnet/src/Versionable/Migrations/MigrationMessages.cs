namespace Versionable.Migrations;

/// <summary>
/// Wording shared by the two places a missing migration step is reported.
/// </summary>
/// <remarks>
/// <c>MigrationRunner</c> checks the range against a chain's declared versions before running it,
/// so a load reports the hole with the object it was loading; <see cref="MigrationChain"/> checks
/// again as it walks, which is what a caller applying a chain directly hits. Same fault, so the
/// same sentence — the exception types differ because the load path has decided the file is
/// unreadable at all (<c>VersionException</c>) while the chain has been asked to do something it
/// cannot (<c>MigrationException</c>).
/// </remarks>
internal static class MigrationMessages
{
    /// <summary>Describes a version in the requested range that no step covers.</summary>
    /// <param name="version">The version with no migration.</param>
    /// <param name="fromVersion">Version the data is shaped for.</param>
    /// <param name="toVersion">Version it was to be brought to.</param>
    /// <param name="declared">Source versions the chain does cover.</param>
    /// <returns>The message.</returns>
    internal static string MissingStep(
        int version,
        int fromVersion,
        int toVersion,
        IEnumerable<int> declared) =>
        $"No migration from version {version} to {version + 1}. Migrating {fromVersion} -> "
            + $"{toVersion} needs every step in between; declared steps: "
            + $"[{string.Join(", ", declared)}].";
}
