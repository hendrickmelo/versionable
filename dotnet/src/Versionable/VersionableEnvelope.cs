using System.Collections.Frozen;

namespace Versionable;

/// <summary>
/// Wire-format key names for the metadata envelope that wraps every serialized
/// <c>[Versionable]</c> object.
/// </summary>
/// <remarks>
/// Python counterpart: <c>_serializeVersionable</c>, <c>_ENVELOPE_KEYS</c>,
/// <c>_readNestedEnvelope</c>, and <c>_stripEnvelope</c> in
/// <c>src/versionable/_types.py</c> (lines 648-740). Those are the source of truth for
/// these literals; changing one without the other breaks file interchange.
/// <para>
/// Two layouts exist:
/// </para>
/// <list type="number">
///   <item>
///     <description>
///     Wrapped (0.2.0+, written by every backend): the object dict carries
///     <see cref="WrappedKey"/> mapping to a nested dict of <see cref="ObjectKey"/>,
///     <see cref="VersionKey"/>, <see cref="HashKey"/>.
///     </description>
///   </item>
///   <item>
///     <description>
///     Flat dunder (0.1.x, read-only back-compat): the object dict carries top-level
///     <see cref="LegacyObjectKey"/>, <see cref="LegacyVersionKey"/>,
///     <see cref="LegacyHashKey"/>. Never written.
///     </description>
///   </item>
/// </list>
/// </remarks>
public static class VersionableEnvelope
{
    /// <summary>Key holding the nested envelope dict in the 0.2.0+ wrapped layout.</summary>
    public const string WrappedKey = "__versionable__";

    /// <summary>Serialization Name of the object, inside the wrapped envelope.</summary>
    public const string ObjectKey = "object";

    /// <summary>Schema version of the object, inside the wrapped envelope.</summary>
    public const string VersionKey = "version";

    /// <summary>Schema hash of the object, inside the wrapped envelope.</summary>
    public const string HashKey = "hash";

    /// <summary>0.1.x flat Serialization Name key. Read-only back-compat.</summary>
    public const string LegacyObjectKey = "__OBJECT__";

    /// <summary>0.1.x flat schema version key. Read-only back-compat.</summary>
    public const string LegacyVersionKey = "__VERSION__";

    /// <summary>0.1.x flat schema hash key. Read-only back-compat.</summary>
    public const string LegacyHashKey = "__HASH__";

    /// <summary>0.1.x flat array format key. Read-only back-compat.</summary>
    public const string LegacyFormatKey = "__FORMAT__";

    /// <summary>0.1.x flat big-endian array format key. Read-only back-compat.</summary>
    public const string LegacyFormatBigEndianKey = "__FORMAT_BE__";

    /// <summary>0.1.x flat shared-reference table key. Read-only back-compat.</summary>
    public const string LegacySharedRefsKey = "__SHARED_REFS__";

    /// <summary>
    /// Every key that is envelope metadata rather than a field, in both layouts. Strip these
    /// before migrations run over field-name keys. Python counterpart: <c>_ENVELOPE_KEYS</c>.
    /// </summary>
    public static readonly FrozenSet<string> ReservedKeys = new[]
    {
        WrappedKey,
        LegacyObjectKey,
        LegacyVersionKey,
        LegacyHashKey,
        LegacyFormatKey,
        LegacyFormatBigEndianKey,
        LegacySharedRefsKey,
    }.ToFrozenSet(StringComparer.Ordinal);
}
