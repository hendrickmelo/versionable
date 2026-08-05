namespace Versionable.Backends;

/// <summary>
/// Options passed through <c>Save</c> to a backend.
/// </summary>
/// <remarks>
/// Python counterpart: the keyword arguments <c>save()</c> forwards to
/// <c>Backend.save()</c> in <c>src/versionable/_api.py</c> — <c>commentDefaults</c> plus
/// <c>**kwargs</c> such as the HDF5 backend's <c>compression</c>. C# keeps the one
/// cross-backend option typed and puts the rest in
/// <see cref="BackendOptions"/>; backends ignore options they do not understand, as Python
/// does.
/// </remarks>
public sealed record BackendSaveOptions
{
    /// <summary>Options with no cross-backend meaning, keyed by backend-defined names.</summary>
    public IReadOnlyDictionary<string, object?> BackendOptions { get; init; } =
        new Dictionary<string, object?>(StringComparer.Ordinal);

    /// <summary>
    /// Write fields still at their default value as commented-out lines, where the format
    /// has comments (TOML, YAML). Ignored elsewhere. Python counterpart:
    /// <c>commentDefaults</c>.
    /// </summary>
    public bool CommentDefaults { get; init; }
}
