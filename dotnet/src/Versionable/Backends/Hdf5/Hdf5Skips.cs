namespace Versionable.Backends.Hdf5;

/// <summary>
/// Records which fields a read skipped, path-qualified, as it descends the object graph.
/// </summary>
/// <remarks>
/// The contract on <see cref="BackendLoadResult.LazyFields"/>: entries are <c>/</c>-separated
/// chains of field names relative to the load root, so a skip inside a nested object says which
/// field it belongs to. A bare name is the zero-depth case, and container indexes never appear —
/// elements descend with their container's prefix, because an index is not part of a skip's
/// identity and the engine could never match one back.
/// <para>
/// A struct carrying the prefix, rather than a prefix parameter beside the set, so the two cannot
/// drift apart as the reader recurses: there is one way to descend
/// (<see cref="Into"/>) and one way to record (<see cref="Skip"/>).
/// </para>
/// </remarks>
/// <param name="Names">The set being filled, or <see langword="null"/> to record nothing.</param>
/// <param name="Prefix">Path of the object currently being read, relative to the load root.</param>
internal readonly record struct Hdf5Skips(HashSet<string>? Names, string Prefix = "")
{
    /// <summary>Returns a recorder positioned inside the field named <paramref name="name"/>.</summary>
    /// <remarks>Only field descent calls this; container elements keep their container's prefix.</remarks>
    /// <param name="name">Field name.</param>
    /// <returns>The recorder for that field.</returns>
    internal Hdf5Skips Into(string name) =>
        Names is null ? this : this with { Prefix = Qualify(name) };

    /// <summary>Records that the child named <paramref name="name"/> was not read.</summary>
    /// <param name="name">Field name, relative to the object being read.</param>
    internal void Skip(string name) => Names?.Add(Qualify(name));

    private string Qualify(string name) => Prefix.Length == 0 ? name : $"{Prefix}/{name}";
}
