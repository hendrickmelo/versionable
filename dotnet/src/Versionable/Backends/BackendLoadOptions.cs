namespace Versionable.Backends;

/// <summary>
/// Options passed to <see cref="IVersionableBackend.Load"/>.
/// </summary>
/// <remarks>
/// Python counterpart: the arguments <c>load()</c> forwards to a backend in
/// <c>src/versionable/_api.py</c>. Python splits this across two entry points — the plain
/// <c>Backend.load(path)</c> and the optional <c>loadLazy(path, cls, preload, metadataOnly)</c>
/// that only the HDF5 backend defines and that <c>load()</c> probes for with
/// <c>getattr</c>. C# folds both into one method with an options record, so all four backends
/// implement one signature and the laziness contract is explicit rather than discovered by
/// duck typing.
///
/// <para>
/// <b>The default here is deliberately the opposite of Python's HDF5 default.</b>
/// <c>new BackendLoadOptions()</c> leaves <see cref="Preload"/> <see langword="null"/>, which
/// means <em>materialize everything</em>. Python's <c>load()</c> defaults <c>preload=None</c>
/// and routes HDF5 through <c>loadLazy</c>, so its observable default for HDF5 is
/// <em>materialize no arrays</em>. The inversion is intentional: a backend contract whose
/// zero-configuration behavior is "load the data" is the one that cannot silently hand back
/// half an object, and three of the four backends have no lazy mode at all.
/// </para>
/// <para>
/// <b>Consequence for the public API (task 2d).</b> The implementer of <c>Load&lt;T&gt;()</c>
/// owns the Python-parity default and must translate, not pass through: with no caller
/// preference and an HDF5 target, it must pass an <em>empty</em> <see cref="Preload"/> set to
/// reproduce Python's lazy-by-default behavior. Passing <c>new BackendLoadOptions()</c>
/// straight through would eagerly read every array and silently break parity on large files.
/// </para>
/// </remarks>
public sealed record BackendLoadOptions
{
    /// <summary>
    /// Metadata of the type the caller is loading into, when known.
    /// </summary>
    /// <remarks>
    /// The HDF5 backend needs it to map native datasets back onto declared field types.
    /// Python counterpart: the <c>cls</c> argument of <c>loadLazy</c>.
    /// <para>
    /// This describes the <em>target</em> type at its current version. The file may have been
    /// written at an older version, so its keys need not match
    /// <see cref="VersionableMetadata.Fields"/> — migrations run after the backend returns.
    /// Treat it as a hint for typing values, never as a schema to validate against.
    /// </para>
    /// </remarks>
    public VersionableMetadata? TargetMetadata { get; init; }

    /// <summary>
    /// Wire names to materialize eagerly, or <see langword="null"/> to materialize everything.
    /// </summary>
    /// <remarks>
    /// Python counterpart: the <c>preload</c> argument, where <c>'*'</c> means everything —
    /// <see langword="null"/> here. An empty set means "materialize nothing eagerly", which is
    /// what reproduces Python's HDF5 default; see the note on the type itself. Fields left
    /// unmaterialized come back in <see cref="BackendLoadResult.LazyFields"/>.
    /// </remarks>
    public IReadOnlySet<string>? Preload { get; init; }

    /// <summary>
    /// Read only the envelope and non-array fields, leaving every array field unmaterialized.
    /// </summary>
    /// <remarks>
    /// Python counterpart: <c>metadataOnly</c>. Reading such a field afterwards raises
    /// <see cref="Errors.ArrayNotLoadedException"/>.
    /// </remarks>
    public bool MetadataOnly { get; init; }

    /// <summary>Options with no cross-backend meaning, keyed by backend-defined names.</summary>
    public IReadOnlyDictionary<string, object?> BackendOptions { get; init; } =
        new Dictionary<string, object?>(StringComparer.Ordinal);
}
