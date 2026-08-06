namespace Versionable;

/// <summary>
/// Caller-facing options for <see cref="VersionableFile.Load{T}"/>.
/// </summary>
/// <remarks>
/// Python counterpart: the keyword arguments of <c>load()</c> in <c>src/versionable/_api.py</c> —
/// <c>preload</c>, <c>metadataOnly</c>, <c>upgradeInPlace</c>, <c>assumeVersion</c>.
/// <para>
/// Separate from <see cref="Backends.BackendLoadOptions"/> on purpose. That record is the
/// <em>backend</em> contract and its no-preference default is "materialize everything"; this one
/// is the <em>caller</em> contract and its no-preference default is Python's, which for a lazy
/// backend means materialize no arrays. <see cref="VersionableFile"/> translates between them —
/// see the table on <see cref="Preload"/>. Collapsing the two records would force one of those
/// two defaults to be wrong.
/// </para>
/// </remarks>
public sealed record VersionableLoadOptions
{
    /// <summary>
    /// Wire names to materialize eagerly, or <see langword="null"/> for the default.
    /// </summary>
    /// <remarks>
    /// How this reaches a backend, mirroring Python's <c>preload</c> argument:
    /// <list type="table">
    ///   <item>
    ///     <term>Unset (the default)</term>
    ///     <description>
    ///     An <em>empty</em> <see cref="Backends.BackendLoadOptions.Preload"/> — materialize
    ///     nothing eagerly. Backends with no lazy mode ignore it and load everything; a lazy
    ///     backend hands back sentinels, which is Python's observable default for HDF5
    ///     (<c>preload=None</c> routed through <c>loadLazy</c>).
    ///     </description>
    ///   </item>
    ///   <item>
    ///     <term>A set of names</term>
    ///     <description>Those names, and only those, are materialized eagerly.</description>
    ///   </item>
    ///   <item>
    ///     <term><see cref="PreloadAll"/></term>
    ///     <description>
    ///     A <see langword="null"/> <see cref="Backends.BackendLoadOptions.Preload"/> —
    ///     materialize everything. Python's <c>preload='*'</c>.
    ///     </description>
    ///   </item>
    /// </list>
    /// </remarks>
    public IReadOnlySet<string>? Preload { get; init; }

    /// <summary>
    /// Materialize every field eagerly, including arrays. Python counterpart: <c>preload='*'</c>.
    /// </summary>
    /// <remarks>Takes precedence over <see cref="Preload"/> when both are set.</remarks>
    public bool PreloadAll { get; init; }

    /// <summary>
    /// Read the envelope and the non-array fields only. Python counterpart: <c>metadataOnly</c>.
    /// </summary>
    /// <remarks>
    /// <b>What the skipped fields hold afterwards.</b> Python leaves a sentinel in each skipped
    /// field and raises <see cref="Errors.ArrayNotLoadedException"/> when something reads it, so the
    /// failure lands on the access. C# v1 has no lazy-instance proxy — that is Tier 3 and post-v1 —
    /// so a skipped field is resolved when the object is built instead:
    /// <list type="bullet">
    ///   <item>
    ///     <description>
    ///     A field with a <b>declared default</b> gets that default. A <c>MetadataOnly</c> load of
    ///     such a type succeeds and carries its array fields unpopulated, at whatever the schema
    ///     says an unset value is.
    ///     </description>
    ///   </item>
    ///   <item>
    ///     <description>
    ///     A field with <b>no default</b> raises <see cref="Errors.ArrayNotLoadedException"/>,
    ///     naming the field and this option. Substituting an empty array instead would be
    ///     indistinguishable from a genuinely empty one — a caller could not tell a skipped read
    ///     from real data, which is the one outcome worse than an error.
    ///     </description>
    ///   </item>
    /// </list>
    /// So <c>MetadataOnly</c> is for reading a file's envelope and scalars cheaply. To get an object
    /// with its arrays, use <see cref="PreloadAll"/> or name the fields in <see cref="Preload"/>.
    /// </remarks>
    public bool MetadataOnly { get; init; }

    /// <summary>
    /// Permit migrations that need the file rewritten. Python counterpart: <c>upgradeInPlace</c>.
    /// </summary>
    public bool UpgradeInPlace { get; init; }

    /// <summary>
    /// Version to assume when the file records none, instead of the type's current version.
    /// </summary>
    /// <remarks>
    /// Python counterpart: <c>assumeVersion</c>. Without it, a version-less file is treated as
    /// current and its migrations are skipped — which is right for a hand-written file and wrong
    /// for one written by older code, so the engine warns either way.
    /// </remarks>
    public int? AssumeVersion { get; init; }

    /// <summary>Options with no cross-backend meaning, forwarded verbatim.</summary>
    public IReadOnlyDictionary<string, object?> BackendOptions { get; init; } =
        new Dictionary<string, object?>(StringComparer.Ordinal);
}
