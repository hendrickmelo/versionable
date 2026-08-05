namespace Versionable;

/// <summary>
/// Marks a type as a persistable root: it carries a schema version and hash, gets generated
/// metadata, and can be saved and loaded.
/// </summary>
/// <remarks>
/// Python counterpart: the <c>Versionable</c> base class and its <c>__init_subclass__</c>
/// parameters (<c>version</c>, <c>hash</c>, <c>old_names</c>, <c>register</c>,
/// <c>skip_defaults</c>, <c>unknown</c>, <c>validate_literals</c>) in
/// <c>src/versionable/_base.py</c>. C# uses an attribute rather than a base class so the
/// source generator can emit metadata without imposing an inheritance relationship.
/// <para>
/// Python's <c>name=</c> parameter is <em>not</em> here — the Serialization Name is declared
/// with <see cref="SerializationNameAttribute"/>, which also applies to enums and converter
/// types. One mechanism, one place to look.
/// </para>
/// <para>
/// <see cref="Hash"/> is validated at compile time by the analyzer, not at type-load time
/// (ADR-0003).
/// </para>
/// </remarks>
[AttributeUsage(AttributeTargets.Class | AttributeTargets.Struct, Inherited = false)]
public sealed class VersionableAttribute : Attribute
{
    /// <summary>Schema version of this type. Mirrors Python's <c>version=</c>.</summary>
    public int Version { get; init; }

    /// <summary>
    /// Declared schema hash: the first six hex characters of the SHA-256 digest of the
    /// canonical field payload (<c>conformance/GRAMMAR.md</c> §1). Mirrors Python's
    /// <c>hash=</c>.
    /// </summary>
    public string Hash { get; init; } = string.Empty;

    /// <summary>
    /// Serialization Names this type used to be known by, which still resolve to it on load.
    /// Mirrors Python's <c>old_names=</c>. Lives here rather than on
    /// <see cref="SerializationNameAttribute"/> because resolving an old name is an envelope
    /// concern, and only persistable roots have envelopes.
    /// </summary>
    public string[]? OldNames { get; init; }

    /// <summary>
    /// Whether this type claims its Serialization Name in the global index. Mirrors Python's
    /// <c>register=</c>.
    /// </summary>
    /// <remarks>
    /// <see langword="false"/> keeps the type serializable — its metadata is still generated
    /// and still reachable by CLR type — but it cannot be resolved from an envelope
    /// <c>object</c> key, so it cannot be the target of a polymorphic load. Use it to keep a
    /// type out of the global name space when its name would otherwise collide. See
    /// <see cref="VersionableRegistry"/>.
    /// </remarks>
    public bool Register { get; init; } = true;

    /// <summary>
    /// Omit fields still at their default value when saving. Mirrors Python's
    /// <c>skip_defaults=</c>.
    /// </summary>
    public bool SkipDefaults { get; init; }

    /// <summary>
    /// How to treat fields present in the file but not declared on this type. Mirrors
    /// Python's <c>unknown=</c>.
    /// </summary>
    public UnknownFieldPolicy Unknown { get; init; } = UnknownFieldPolicy.Ignore;

    /// <summary>
    /// Validate literal-typed fields against their declared options on load. Mirrors Python's
    /// <c>validate_literals=</c>.
    /// </summary>
    public bool ValidateLiterals { get; init; } = true;
}
