using System.Collections.Concurrent;
using Versionable.Errors;

namespace Versionable.Backends;

/// <summary>
/// Maps file extensions to backend factories and resolves the backend for a path.
/// </summary>
/// <remarks>
/// Python counterpart: <c>_BACKEND_REGISTRY</c>, <c>registerBackend()</c>, and
/// <c>getBackend()</c> in <c>src/versionable/_backend.py</c>. Unlike Python — where the
/// optional backends sit behind optional third-party imports — every C# backend ships in the
/// one package, so registration is unconditional from a <c>[ModuleInitializer]</c>.
/// <para>Thread-safe. Extensions are matched case-insensitively and include the leading dot.</para>
/// </remarks>
public static class BackendRegistry
{
    private static readonly ConcurrentDictionary<string, Func<IVersionableBackend>> _factories =
        new(StringComparer.OrdinalIgnoreCase);

    /// <summary>Registers <paramref name="factory"/> for each of <paramref name="extensions"/>.</summary>
    /// <param name="extensions">File extensions including the leading dot, e.g. <c>.json</c>.</param>
    /// <param name="factory">Creates a backend instance; called once per save or load.</param>
    public static void Register(IEnumerable<string> extensions, Func<IVersionableBackend> factory)
    {
        ArgumentNullException.ThrowIfNull(extensions);
        ArgumentNullException.ThrowIfNull(factory);

        foreach (string extension in extensions)
        {
            _factories[extension] = factory;
        }
    }

    /// <summary>Resolves the backend for <paramref name="path"/>.</summary>
    /// <param name="path">File path whose extension selects the backend.</param>
    /// <param name="explicitBackend">
    /// Backend to use instead of auto-detecting, or <see langword="null"/> to select by
    /// extension. Python counterpart: the <c>explicit</c> argument of <c>getBackend()</c>,
    /// which surfaces as <c>backend=</c> on <c>save()</c> and <c>load()</c>. Lets a caller
    /// write JSON to a path with any extension, or supply a backend that was never registered.
    /// </param>
    /// <returns>
    /// <paramref name="explicitBackend"/> when given, otherwise a new instance from the
    /// factory registered for the path's extension.
    /// </returns>
    /// <exception cref="BackendException">
    /// No backend is registered for the extension and none was supplied.
    /// </exception>
    public static IVersionableBackend Resolve(string path, IVersionableBackend? explicitBackend = null)
    {
        if (explicitBackend is not null)
        {
            return explicitBackend;
        }

        string extension = Path.GetExtension(path);
        if (!_factories.TryGetValue(extension, out Func<IVersionableBackend>? factory))
        {
            throw new BackendException(
                $"No backend registered for extension '{extension}'. "
                    + $"Known extensions: {string.Join(", ", RegisteredExtensions())}.");
        }

        return factory();
    }

    /// <summary>Extensions with a registered backend, sorted.</summary>
    /// <returns>A snapshot taken at call time.</returns>
    public static IReadOnlyList<string> RegisteredExtensions() => [.. _factories.Keys.Order(StringComparer.Ordinal)];

    /// <summary>Empties the registry. Test seam; there is no public unregister.</summary>
    internal static void Reset() => _factories.Clear();
}
