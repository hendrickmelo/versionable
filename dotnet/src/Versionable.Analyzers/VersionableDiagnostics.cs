namespace Versionable.Analyzers;

/// <summary>
/// Diagnostic identifiers reported by the Versionable analyzer.
/// </summary>
/// <remarks>
/// The <c>VSN</c> prefix is the Versionable diagnostic namespace. Severities are
/// configurable per-id from <c>.editorconfig</c>; <see cref="HashMismatch"/> is the
/// analogue of Python's <c>versionable.ignoreHashErrors()</c> dev mode
/// (<c>src/versionable/_base.py</c>). Descriptors land with the analyzer in task 2b.
/// </remarks>
internal static class VersionableDiagnostics
{
    /// <summary>Category applied to every Versionable diagnostic.</summary>
    internal const string Category = "Versionable";

    /// <summary>Declared <c>Hash</c> disagrees with the hash computed from the symbol model.</summary>
    internal const string HashMismatch = "VSN0001";

    /// <summary>A declared type cannot be expressed in the canonical type grammar.</summary>
    internal const string UnsupportedType = "VSN0002";

    /// <summary>The migration chain has a gap above the oldest declared migration.</summary>
    internal const string MigrationChainGap = "VSN0003";
}
