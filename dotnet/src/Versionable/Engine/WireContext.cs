using System.Collections.Frozen;

namespace Versionable.Engine;

/// <summary>
/// Ambient state for one serialization walk: what the backend stores natively, which objects are
/// on the stack, and where in the object graph the walk currently is.
/// </summary>
/// <remarks>
/// Ambient rather than threaded through every call because the generated
/// <see cref="FieldDescriptor.WireWriter"/> delegates are <c>Func&lt;object?, object?&gt;</c> —
/// there is nowhere in that signature to pass a walk context. A generated container writer that
/// recurses through <see cref="WireValues.Write(object?)"/> would otherwise start a fresh walk
/// per element and lose cycle detection entirely.
/// </remarks>
internal sealed class WriteScope(IReadOnlySet<Type> nativeTypes)
{
    /// <summary>CLR types the backend writes natively, which must not go through a converter.</summary>
    public IReadOnlySet<Type> NativeTypes { get; } = nativeTypes;

    /// <summary>
    /// Objects currently on the serialization stack, by reference identity.
    /// </summary>
    /// <remarks>
    /// Entries are removed on the way back up, so a single instance reachable from two unrelated
    /// branches — a diamond — is duplicated on disk rather than reported as a cycle. Python does
    /// the same (<c>_serializeVersionable</c>), and 0.2.x has no shared-reference table to do
    /// better with.
    /// </remarks>
    public HashSet<object> Visited { get; } = new(ReferenceEqualityComparer.Instance);

    /// <summary>Field path of the value being written, for cycle-error messages.</summary>
    public string Path { get; set; } = string.Empty;
}

/// <summary>
/// Ambient state for one deserialization walk.
/// </summary>
/// <remarks>
/// Ambient for the same reason as <see cref="WriteScope"/>: generated
/// <see cref="FieldDescriptor.WireReader"/> delegates take a bare wire value and hand back a CLR
/// value, so anything a nested read needs has to reach it out of band.
/// </remarks>
internal sealed class ReadScope
{
    /// <summary>CLR types the backend handed back already materialized.</summary>
    public IReadOnlySet<Type> NativeTypes { get; init; } = FrozenSet<Type>.Empty;

    /// <summary>
    /// Whether migrations that rewrite the file are permitted. Load-global rather than
    /// class-scoped, so it crosses nested object boundaries unchanged — as it does in Python.
    /// </summary>
    public bool UpgradeInPlace { get; init; }
}
