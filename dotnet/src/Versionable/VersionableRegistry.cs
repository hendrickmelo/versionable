using System.Collections.Concurrent;
using System.Diagnostics.CodeAnalysis;
using Versionable.Errors;

namespace Versionable;

/// <summary>
/// Process-wide index of <see cref="VersionableMetadata"/>. Generated
/// <c>[ModuleInitializer]</c> code populates it at assembly load.
/// </summary>
/// <remarks>
/// Python counterpart: the <c>_REGISTRY</c> dict, <c>registeredClasses()</c>, and the
/// registration branch of <c>Versionable.__init_subclass__</c> in
/// <c>src/versionable/_base.py</c>.
/// <para>
/// There are two indexes and they are populated independently, mirroring Python:
/// </para>
/// <list type="bullet">
///   <item>
///     <description>
///     <b>By Serialization Name</b> — the envelope-resolution index, used to turn an
///     <c>object</c> key read from a file into a type. Populated only for types declared
///     <c>Register = true</c>, matching Python's <c>_REGISTRY</c>, which <c>register=False</c>
///     skips.
///     </description>
///   </item>
///   <item>
///     <description>
///     <b>By CLR type</b> — populated for <em>every</em> <c>[Versionable]</c> type, including
///     <c>Register = false</c> ones, because saving a nested value needs the metadata of its
///     runtime type whether or not that type claims a global name. Python gets this for free
///     by keeping <c>_serializer_meta_</c> on the class itself, which the nested-class
///     resolution in <c>src/versionable/_types.py</c> falls back to.
///     </description>
///   </item>
/// </list>
/// <para>
/// Generated code calls <see cref="Register"/> for <c>Register = true</c> types and
/// <see cref="RegisterTypeOnly"/> for <c>Register = false</c> types. Statically known types
/// need neither: the generator also implements <see cref="IVersionableMetadataProvider"/> on
/// every type, which is the reflection-free, AOT-safe path.
/// </para>
/// <para>
/// Thread-safe. Registration takes a lock so name validation and the writes that follow are
/// atomic; lookups are lock-free.
/// </para>
/// </remarks>
public static class VersionableRegistry
{
    private static readonly ConcurrentDictionary<string, VersionableMetadata> _byName = new(StringComparer.Ordinal);
    private static readonly ConcurrentDictionary<Type, VersionableMetadata> _byType = new();
    private static readonly object _registrationLock = new();

    /// <summary>
    /// Registers <paramref name="metadata"/> under its Serialization Name, every entry of
    /// <see cref="VersionableMetadata.OldNames"/>, and its CLR type.
    /// </summary>
    /// <param name="metadata">Metadata to register.</param>
    /// <exception cref="VersionableException">
    /// A different type already claims one of the names. Nothing is registered when this
    /// throws — not the CLR type, and not the names validated before the collision.
    /// </exception>
    public static void Register(VersionableMetadata metadata)
    {
        ArgumentNullException.ThrowIfNull(metadata);

        lock (_registrationLock)
        {
            // Read-only pre-pass over every name this type claims, then write. A collision
            // on the second old name must not leave the primary name registered — Python
            // validates all names before touching _REGISTRY for the same reason
            // (src/versionable/_base.py).
            EnsureUnclaimed(metadata.Name, metadata);
            foreach (string oldName in metadata.OldNames ?? [])
            {
                EnsureUnclaimed(oldName, metadata);
            }

            _byName[metadata.Name] = metadata;
            foreach (string oldName in metadata.OldNames ?? [])
            {
                _byName[oldName] = metadata;
            }

            _byType[metadata.ClrType] = metadata;
        }
    }

    /// <summary>
    /// Registers <paramref name="metadata"/> in the CLR-type index only, claiming no
    /// Serialization Name.
    /// </summary>
    /// <remarks>
    /// The <c>Register = false</c> path. The type stays serializable as a nested value — its
    /// metadata is reachable by runtime type — but it never resolves from an envelope
    /// <c>object</c> key, so it cannot participate in polymorphic loads. Python counterpart:
    /// a class declared <c>register=False</c>, which keeps <c>_serializer_meta_</c> but stays
    /// out of <c>_REGISTRY</c>.
    /// </remarks>
    /// <param name="metadata">Metadata to register.</param>
    public static void RegisterTypeOnly(VersionableMetadata metadata)
    {
        ArgumentNullException.ThrowIfNull(metadata);

        lock (_registrationLock)
        {
            _byType[metadata.ClrType] = metadata;
        }
    }

    private static void EnsureUnclaimed(string name, VersionableMetadata metadata)
    {
        if (_byName.TryGetValue(name, out VersionableMetadata? registered)
            && registered.ClrType != metadata.ClrType)
        {
            throw new VersionableException(
                $"Serialization Name '{name}' is already registered to {registered.ClrType.FullName}. "
                    + $"Give one of the types an explicit, distinct name, e.g. "
                    + $"[SerializationName(\"{metadata.ClrType.Name}V2\")].");
        }
    }

    /// <summary>Looks up metadata by Serialization Name, including old names.</summary>
    /// <param name="name">Serialization Name as written in the envelope.</param>
    /// <param name="metadata">The registered metadata, when found.</param>
    /// <returns><see langword="true"/> when a registration exists.</returns>
    public static bool TryGetByName(string name, [NotNullWhen(true)] out VersionableMetadata? metadata) =>
        _byName.TryGetValue(name, out metadata);

    /// <summary>Looks up metadata by CLR type. Finds <c>Register = false</c> types too.</summary>
    /// <param name="type">A <c>[Versionable]</c> type.</param>
    /// <param name="metadata">The registered metadata, when found.</param>
    /// <returns><see langword="true"/> when a registration exists.</returns>
    public static bool TryGetByType(Type type, [NotNullWhen(true)] out VersionableMetadata? metadata) =>
        _byType.TryGetValue(type, out metadata);

    /// <summary>
    /// Snapshot of the Serialization Name index. Python counterpart:
    /// <c>registeredClasses()</c>.
    /// </summary>
    /// <returns>A copy taken at call time.</returns>
    public static IReadOnlyDictionary<string, VersionableMetadata> RegisteredTypes() =>
        new Dictionary<string, VersionableMetadata>(_byName, StringComparer.Ordinal);

    /// <summary>Empties both indexes. Test seam; there is no public unregister.</summary>
    internal static void Reset()
    {
        lock (_registrationLock)
        {
            _byName.Clear();
            _byType.Clear();
        }
    }
}
